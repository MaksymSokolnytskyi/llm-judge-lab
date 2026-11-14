"""Orchestrates one dialogue: dice -> scenario -> (beats -> draft -> stylize -> judge)*.

The bracketed block is retried up to ``max_retries`` times when the macro judge fails
the dialogue; the scenario is kept, so retries vary the plan, not the story.
"""

from __future__ import annotations

import uuid

from pydantic import BaseModel, ConfigDict, Field

from judgelab.dialogues.dice import DiceRoll, roll
from judgelab.dialogues.drafting import Message, draft_dialogue
from judgelab.dialogues.macro_judge import MacroVerdict, judge_dialogue
from judgelab.dialogues.planning import (
    Beat,
    Scenario,
    generate_beat_sheet,
    generate_scenario,
)
from judgelab.dialogues.routing import DynamicRouter, MicroArena
from judgelab.dialogues.stylize import RouteLogEntry, stylize_dialogue
from judgelab.llm import Backend


class PipelineModels(BaseModel):
    """Which model runs each stage (defaults from the original runs)."""

    model_config = ConfigDict(frozen=True)

    scenario: str = "llama3.1:8b"
    beats: str = "qwen2.5:7b"
    drafter: str = "llama3.1:8b"
    judge: str = "qwen2.5:7b"


class Dialogue(BaseModel):
    """One generated dialogue with its ground-truth labels and provenance."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    params: DiceRoll
    scenario: Scenario
    beats: list[Beat]
    beats_used_fallback: bool
    messages: list[Message]
    verdict: MacroVerdict
    attempts: int
    routes: list[RouteLogEntry]


def run_pipeline(
    seed: int,
    backend: Backend,
    router: DynamicRouter,
    arena: MicroArena,
    models: PipelineModels | None = None,
    max_retries: int = 2,
) -> Dialogue:
    models = models or PipelineModels()
    params = roll(seed)
    scenario = generate_scenario(params, backend, models.scenario)

    attempt = 0
    while True:
        beats, beats_fallback = generate_beat_sheet(
            params, scenario, backend, models.beats, attempt=attempt
        )
        draft = draft_dialogue(params, beats, backend, models.drafter)
        stylized, routes = stylize_dialogue(params, draft, backend, router, arena)
        verdict = judge_dialogue(params, stylized, backend, models.judge)
        if verdict.passed or attempt == max_retries:
            return Dialogue(
                params=params,
                scenario=scenario,
                beats=beats,
                beats_used_fallback=beats_fallback,
                messages=stylized,
                verdict=verdict,
                attempts=attempt + 1,
                routes=routes,
            )
        attempt += 1
