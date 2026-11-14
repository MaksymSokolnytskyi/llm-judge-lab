"""End-to-end cross-judge run on a scripted backend (no Ollama needed)."""

from __future__ import annotations

import json
from pathlib import Path

from judgelab.cross_judge import CrossJudgeExperiment, ExperimentConfig
from judgelab.cross_judge.analysis import join_rows, read_jsonl
from judgelab.cross_judge.prompts import ENGLISH
from judgelab.data import Record
from judgelab.llm import ChatRequest, LLMError, ScriptedBackend

RUN = Path(__file__).resolve().parents[1] / "results" / "cross_judge_2026-03-02"


def _responder(request: ChatRequest) -> str:
    """Meta-prompter, generator and judge, told apart by their system prompt."""
    if request.system == ENGLISH.meta_system:
        return f"PROMPT BY {request.model}"
    if request.system.startswith("PROMPT BY"):
        return f"rewritten by {request.model}"
    assert request.system == ENGLISH.judge_system
    if request.model == "refuser" and "PERSONA: toxic" in request.user:
        return '{"score": 0, "feedback": "I cannot evaluate offensive content."}'
    score = 8 if "rewritten by a" in request.user else 5
    return json.dumps({"score": score, "feedback": "ok", "accepted": False})


def _records(n: int) -> list[Record]:
    return [
        Record(
            id=f"r{i}",
            intent="x",
            category="C",
            instruction=f"q{i}",
            original_response=f"answer {i}",
        )
        for i in range(n)
    ]


def test_counts_and_statuses(tmp_path: Path) -> None:
    config = ExperimentConfig(models=("a", "refuser"), n_records=2, workers=4)
    result = CrossJudgeExperiment(ScriptedBackend(_responder), config).run(
        _records(2), out_dir=tmp_path
    )
    assert len(result.meta_prompts) == 2 * 2 * 3
    assert len(result.generations) == 2 * 2 * 2 * 3 * 2
    assert len(result.judgments) == len(result.generations) * 2

    refused = [j for j in result.judgments if j.status == "refused"]
    assert refused and all(j.judge_model == "refuser" for j in refused)
    assert all(j.score is None and j.passed is None for j in refused)
    # passed comes from the score even though the judge always says accepted=false
    assert any(j.passed for j in result.judgments if j.status == "scored")

    rows = join_rows(
        read_jsonl(tmp_path / "generations.jsonl"),
        read_jsonl(tmp_path / "judgments.jsonl"),
    )
    assert len(rows) == len(result.judgments)
    assert json.loads((tmp_path / "config.json").read_text())["models"] == [
        "a",
        "refuser",
    ]


def test_failed_generations_are_not_judged() -> None:
    def responder(request: ChatRequest) -> str:
        if request.system != ENGLISH.meta_system:
            raise LLMError("generator down")
        return "PROMPT"

    config = ExperimentConfig(models=("a",), n_records=1, workers=1)
    result = CrossJudgeExperiment(ScriptedBackend(responder), config).run(_records(1))
    assert all(g.failed for g in result.generations)
    assert result.judgments == []


def test_reproduces_original_generation_ids() -> None:
    """Same records + default config -> the exact 270 generation ids of the 2026 run."""
    original = read_jsonl(RUN / "generations.jsonl")
    seen: dict[str, Record] = {}
    for g in original:
        seen.setdefault(
            g["record_id"],
            Record(
                id=g["record_id"],
                intent=g["intent"],
                category=g["category"],
                instruction=g["instruction"],
                original_response=g["original_response"],
            ),
        )
    config = ExperimentConfig(language="uk", workers=8)
    backend = ScriptedBackend(lambda r: '{"score": 5, "feedback": "x"}')
    result = CrossJudgeExperiment(backend, config).run(list(seen.values()))
    assert {g.gen_id for g in result.generations} == {g["gen_id"] for g in original}
