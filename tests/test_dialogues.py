"""Dialogue stages and the full pipeline on a scripted backend."""

from __future__ import annotations

import json
from collections import Counter

from judgelab.dialogues import run_pipeline
from judgelab.dialogues.dice import CASE_TYPE_WEIGHTS, roll
from judgelab.dialogues.drafting import DRAFTER_SYSTEM, clean_message, truncate_words
from judgelab.dialogues.macro_judge import MACRO_JUDGE_SYSTEM, PARSE_ERROR
from judgelab.dialogues.planning import BEAT_SHEET_SYSTEM, SCENARIO_SYSTEM
from judgelab.dialogues.routing import ArenaRegistry, DynamicRouter, MicroArena
from judgelab.dialogues.routing.micro_arena import JUDGE_SYSTEM
from judgelab.llm import ChatRequest, ScriptedBackend


def test_roll_is_deterministic_and_close_to_weights() -> None:
    assert roll(7) == roll(7)
    counts = Counter(roll(seed).case_type for seed in range(4000))
    for case, weight in CASE_TYPE_WEIGHTS.items():
        assert abs(counts[case] / 4000 - weight) < 0.03


# Outputs of the original arena/pipeline/stage0_dice.roll; the port was also checked
# against it on seeds 0..1999 with zero mismatches.
ORIGINAL_ROLLS = {
    0: {
        "seed": 0,
        "intent": "account_access",
        "case_type": "agent_mistake",
        "satisfaction": "unsatisfied",
        "hidden_dissatisfaction": True,
        "agent_mistakes": ("unnecessary_escalation", "no_resolution"),
        "client_style": "confused_elderly",
        "agent_style": "rushed_multitask",
        "n_turns": 7,
        "sector": "insurance",
    },
    42: {
        "seed": 42,
        "intent": "technical_error",
        "case_type": "problematic",
        "satisfaction": "satisfied",
        "hidden_dissatisfaction": False,
        "agent_mistakes": (),
        "client_style": "formal_corporate",
        "agent_style": "empathetic",
        "n_turns": 5,
        "sector": "healthcare",
    },
    1234: {
        "seed": 1234,
        "intent": "account_access",
        "case_type": "agent_mistake",
        "satisfaction": "neutral",
        "hidden_dissatisfaction": True,
        "agent_mistakes": (
            "ignored_question",
            "asked_for_info_already_provided",
            "wrong_department_redirect",
        ),
        "client_style": "passive_aggressive",
        "agent_style": "empathetic",
        "n_turns": 4,
        "sector": "telco",
    },
}


def test_roll_matches_original_generator() -> None:
    for seed, expected in ORIGINAL_ROLLS.items():
        assert roll(seed).model_dump() == expected


def test_clean_and_truncate() -> None:
    assert clean_message('[CLIENT]: "<|eot|>where is my refund"') == "where is my refund"
    long = " ".join(["word"] * 30)
    assert len(truncate_words(long).split()) <= 20
    assert truncate_words("short one") == "short one"


def _pipeline_backend(macro: str) -> ScriptedBackend:
    def responder(request: ChatRequest) -> str:
        if request.system == SCENARIO_SYSTEM:
            return json.dumps(
                {
                    "situation": "charged twice",
                    "client_goal": "refund",
                    "agent_action": "checks",
                    "resolution": "refund issued",
                }
            )
        if request.system == BEAT_SHEET_SYSTEM:
            return "not json at all"  # forces the flagged fallback plan
        if request.system == DRAFTER_SYSTEM:
            return "[AGENT]: i was charged twice for my plan this month"
        if request.system == JUDGE_SYSTEM:
            return '{"score": 7, "reason": "fine"}'
        if request.system == MACRO_JUDGE_SYSTEM:
            return macro
        if request.user.startswith("Write a system prompt"):
            return "be terse"
        return "stylized text"

    return ScriptedBackend(responder)


def _run(macro: str):
    registry = ArenaRegistry()
    backend = _pipeline_backend(macro)
    arena = MicroArena(backend, registry, prompters=("p1",), actors=("a1", "a2"))
    router = DynamicRouter(registry, epsilon=0.0)
    return run_pipeline(3, backend, router, arena, max_retries=1), backend


def test_pipeline_pass() -> None:
    dialogue, _ = _run('{"passed": true, "score": 8, "failures": [], "notes": "ok"}')
    params = roll(3)
    assert len(dialogue.messages) == params.n_turns
    assert [m.role for m in dialogue.messages][:2] == ["client", "agent"]
    assert dialogue.beats_used_fallback
    assert dialogue.verdict.passed and dialogue.attempts == 1
    assert {r.source for r in dialogue.routes} <= {"arena", "exact_cache"}
    assert all(m.content == "stylized text" for m in dialogue.messages)


def test_pipeline_unreadable_judge_fails_and_retries() -> None:
    dialogue, backend = _run("cannot decide")
    assert not dialogue.verdict.passed
    assert dialogue.verdict.failures == (PARSE_ERROR,)
    assert dialogue.attempts == 2
    beat_calls = [r for r in backend.requests if r.system == BEAT_SHEET_SYSTEM]
    assert len(beat_calls) == 2 and "Retry attempt 1" in beat_calls[1].user
