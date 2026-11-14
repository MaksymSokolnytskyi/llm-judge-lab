"""Cross-judge experiment: every model plays every role.

For models M, personas P, prompting approaches A and records R:

1. **Meta-prompting** -- each model writes a system prompt for every (persona,
   approach): |M|·|P|·|A| prompts.
2. **Generation** -- each model rewrites each record under each meta-prompt:
   |R|·|P|·|M|·|A|·|M| texts (author of the prompt × generator).
3. **Judging** -- each model scores every generation: generations × |M| verdicts.

With the default 3 models, 2 personas, 3 approaches and 5 records this is 18 prompts,
270 generations and 810 verdicts. Because every model is also a judge of its own
output and of others', judge severity, self-preference and inter-judge agreement can
be measured on the same texts.

LLM calls are network I/O, so they run in a thread pool: the GIL is released while
a thread waits on the socket, and throughput is bounded by the Ollama server, not by
Python.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ConfigDict, Field

from judgelab.cross_judge.prompts import (
    PROMPT_SETS,
    Approach,
    Persona,
    PromptLanguage,
    PromptSet,
)
from judgelab.data import Record
from judgelab.llm import Backend, ChatRequest, LLMError
from judgelab.verdict import VerdictStatus, parse_verdict

log = logging.getLogger(__name__)

_J = TypeVar("_J")
_T = TypeVar("_T", bound=BaseModel)

DEFAULT_MODELS: tuple[str, ...] = ("llama3.1:8b", "qwen2.5:7b", "dolphin-mistral")


class ExperimentConfig(BaseModel):
    """Everything that defines a run. Saved next to the outputs."""

    model_config = ConfigDict(frozen=True)

    models: tuple[str, ...] = DEFAULT_MODELS
    personas: tuple[Persona, ...] = ("toxic", "cheerful")
    approaches: tuple[Approach, ...] = ("rules", "roleplay", "psychological")
    n_records: int = Field(default=5, ge=1)
    pass_score: int = Field(default=7, ge=0, le=10)
    language: PromptLanguage = "en"
    workers: int = Field(default=3, ge=1, description="Concurrent LLM calls.")
    meta_temperature: float = 0.35
    meta_max_tokens: int = 512
    gen_temperature: float = 0.85
    gen_max_tokens: int = 200
    judge_temperature: float = 0.1
    judge_max_tokens: int = Field(
        default=256,
        description="The original run used 128; 5 of its verdicts were prose with "
        "no JSON, so the cap is raised to leave room for a model that reasons first.",
    )

    @property
    def prompts(self) -> PromptSet:
        return PROMPT_SETS[self.language]


class MetaPrompt(BaseModel):
    author_model: str
    persona: Persona
    approach: Approach
    prompt_text: str
    used_fallback: bool
    latency_s: float


class Generation(BaseModel):
    gen_id: str
    record_id: str
    intent: str
    category: str
    instruction: str
    original_response: str
    persona: Persona
    meta_author: str
    approach: Approach
    generator_model: str
    system_prompt: str
    generated_text: str
    failed: bool
    gen_latency_s: float


class Judgment(BaseModel):
    gen_id: str
    judge_model: str
    status: VerdictStatus
    score: int | None
    passed: bool | None
    judge_accepted: bool | None
    feedback: str
    judge_latency_s: float


class RunResult(BaseModel):
    meta_prompts: list[MetaPrompt]
    generations: list[Generation]
    judgments: list[Judgment]


def generation_id(
    record_id: str, persona: str, meta_author: str, approach: str, generator: str
) -> str:
    """12-hex id; same recipe as the original run so ids stay comparable."""
    key = f"{record_id}{persona}{meta_author}{approach}{generator}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


class _JsonlSink:
    """Append-only, thread-safe JSONL writer (one file per record type)."""

    def __init__(self, out_dir: Path | None):
        self._out_dir = out_dir
        self._lock = threading.Lock()

    def write(self, name: str, item: BaseModel) -> None:
        if self._out_dir is None:
            return
        line = item.model_dump_json() + "\n"
        with (
            self._lock,
            open(self._out_dir / f"{name}.jsonl", "a", encoding="utf-8") as f,
        ):
            f.write(line)


class CrossJudgeExperiment:
    """Runs the three stages against one :class:`~judgelab.llm.Backend`."""

    def __init__(self, backend: Backend, config: ExperimentConfig | None = None):
        self.backend = backend
        self.config = config or ExperimentConfig()

    # -- stage 1 -------------------------------------------------------------------
    def write_meta_prompt(
        self, author: str, persona: Persona, approach: Approach
    ) -> MetaPrompt:
        cfg, prompts = self.config, self.config.prompts
        request = ChatRequest(
            model=author,
            system=prompts.meta_system,
            user=prompts.meta_user.format(persona=persona, approach=approach),
            temperature=cfg.meta_temperature,
            max_tokens=cfg.meta_max_tokens,
        )
        try:
            completion = self.backend.complete(request)
            text, latency = completion.text.strip(), completion.latency_s
        except LLMError as exc:
            log.warning(
                "meta-prompt failed (%s, %s, %s): %s", author, persona, approach, exc
            )
            text, latency = "", 0.0
        if text:
            return MetaPrompt(
                author_model=author,
                persona=persona,
                approach=approach,
                prompt_text=text,
                used_fallback=False,
                latency_s=latency,
            )
        return MetaPrompt(
            author_model=author,
            persona=persona,
            approach=approach,
            prompt_text=prompts.fallback_prompts[(persona, approach)],
            used_fallback=True,
            latency_s=latency,
        )

    # -- stage 2 -------------------------------------------------------------------
    def generate(self, record: Record, meta: MetaPrompt, generator: str) -> Generation:
        cfg = self.config
        request = ChatRequest(
            model=generator,
            system=meta.prompt_text,
            user=cfg.prompts.generator_user.format(
                original=record.original_response, instruction=record.instruction
            ),
            temperature=cfg.gen_temperature,
            max_tokens=cfg.gen_max_tokens,
        )
        try:
            completion = self.backend.complete(request)
            text, latency, failed = completion.text.strip(), completion.latency_s, False
        except LLMError as exc:
            log.error("generation failed (%s): %s", generator, exc)
            text, latency, failed = "", 0.0, True
        return Generation(
            gen_id=generation_id(
                record.id, meta.persona, meta.author_model, meta.approach, generator
            ),
            record_id=record.id,
            intent=record.intent,
            category=record.category,
            instruction=record.instruction,
            original_response=record.original_response,
            persona=meta.persona,
            meta_author=meta.author_model,
            approach=meta.approach,
            generator_model=generator,
            system_prompt=meta.prompt_text,
            generated_text=text,
            failed=failed or not text,
            gen_latency_s=latency,
        )

    # -- stage 3 -------------------------------------------------------------------
    def judge(self, generation: Generation, judge_model: str) -> Judgment:
        cfg = self.config
        request = ChatRequest(
            model=judge_model,
            system=cfg.prompts.judge_system,
            user=cfg.prompts.judge_user.format(
                persona=generation.persona,
                original=generation.original_response,
                rewritten=generation.generated_text,
            ),
            temperature=cfg.judge_temperature,
            max_tokens=cfg.judge_max_tokens,
        )
        try:
            completion = self.backend.complete(request)
        except LLMError as exc:
            log.error("judge call failed (%s): %s", judge_model, exc)
            return Judgment(
                gen_id=generation.gen_id,
                judge_model=judge_model,
                status="unparsed",
                score=None,
                passed=None,
                judge_accepted=None,
                feedback=f"call failed: {exc}",
                judge_latency_s=0.0,
            )
        verdict = parse_verdict(completion.text, pass_score=cfg.pass_score)
        return Judgment(
            gen_id=generation.gen_id,
            judge_model=judge_model,
            status=verdict.status,
            score=verdict.score,
            passed=verdict.passed,
            judge_accepted=verdict.judge_accepted,
            feedback=verdict.feedback,
            judge_latency_s=completion.latency_s,
        )

    # -- orchestration ---------------------------------------------------------------
    def run(self, records: Sequence[Record], out_dir: Path | None = None) -> RunResult:
        """Run all three stages. If ``out_dir`` is given, stream JSONL there."""
        cfg = self.config
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "config.json").write_text(
                json.dumps(cfg.model_dump(), indent=2), encoding="utf-8"
            )
        sink = _JsonlSink(out_dir)

        metas = [
            self.write_meta_prompt(author, persona, approach)
            for author, persona, approach in itertools.product(
                cfg.models, cfg.personas, cfg.approaches
            )
        ]
        for meta in metas:
            sink.write("meta_prompts", meta)
        log.info(
            "meta-prompts: %d (%d fallbacks)",
            len(metas),
            sum(m.used_fallback for m in metas),
        )

        gen_jobs = [
            (record, meta, generator)
            for record in records
            for meta in metas
            for generator in cfg.models
        ]
        generations = self._map(
            lambda job: self.generate(*job),
            gen_jobs,
            lambda g: sink.write("generations", g),
        )
        log.info("generations: %d", len(generations))

        judge_jobs = [
            (generation, judge_model)
            for generation in generations
            if not generation.failed
            for judge_model in cfg.models
        ]
        judgments = self._map(
            lambda job: self.judge(*job), judge_jobs, lambda j: sink.write("judgments", j)
        )
        log.info("judgments: %d", len(judgments))
        return RunResult(meta_prompts=metas, generations=generations, judgments=judgments)

    def _map(
        self, fn: Callable[[_J], _T], jobs: Sequence[_J], on_done: Callable[[_T], None]
    ) -> list[_T]:
        """Run ``fn`` over ``jobs`` concurrently; results keep the input order.

        Each result is handed to ``on_done`` as soon as it and all earlier ones are
        ready, so partial runs still leave usable JSONL behind.
        """
        results: list[_T] = []
        with ThreadPoolExecutor(max_workers=self.config.workers) as pool:
            for result in pool.map(fn, jobs):
                on_done(result)
                results.append(result)
        return results
