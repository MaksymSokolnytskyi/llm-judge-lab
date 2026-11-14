"""Stages 1 and 1b: scenario brief, then a turn-by-turn beat sheet.

Separating *what happens* (scenario) from *when it is revealed* (beat sheet) is what
stops the drafting model from resolving the problem in turn 2: each drafted message
only sees its own beat, and ``may_reveal`` is false until the last two turns.

When a model's JSON cannot be read, a generic fallback is used and marked with
``used_fallback`` so it is visible in the output instead of silently passing as
model work.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, ValidationError

from judgelab.dialogues.dice import DiceRoll
from judgelab.llm import Backend, ChatRequest
from judgelab.parsing import extract_json_array, extract_json_object

SCENARIO_SYSTEM = """You are a scriptwriter for synthetic customer support
chat datasets. You write scenario briefs -- NOT dialogue.

Output ONLY a JSON object. No markdown. No explanation.

Rules:
- Be specific and concrete. No vague phrases like "issue occurred".
- hidden_signal must be subtle: client says "thanks anyway" or "ok fine"
  but their goal was NOT achieved. If no hidden dissatisfaction -- empty string.
- mistake_detail must sound like normal agent behavior, not obvious error.
  BAD: "agent gives completely wrong answer"
  GOOD: "agent follows script for old interface, doesn't notice client
        mentions they already tried that step"
- resolution must match satisfaction level exactly
- Keep each field to 1-2 sentences max"""

BEAT_SHEET_SYSTEM = """You are a screenwriting director planning
a customer support chat scene.

Output ONLY a JSON array of message plans. No markdown. No explanation.

Rules:
- Resolution may appear ONLY in the last 2 turns
- Turn 1: client states the problem they have RIGHT NOW
- Middle turns: back-and-forth, problem gets worse or stays stuck
- internal_goal must be specific and actionable
- If hidden_dissatisfaction=true: last client turn must say
  "thanks" or "ok fine" but NOT achieve their goal
- If agent has mistakes: describe exactly when and how the
  mistake appears (e.g. "turn 3: agent asks for account number
  even though client provided it in turn 1")
- may_reveal=false for all turns except last 2"""


class Scenario(BaseModel):
    model_config = ConfigDict(frozen=True)

    situation: str
    client_goal: str
    agent_action: str
    resolution: str
    hidden_signal: str = ""
    mistake_detail: str = ""
    used_fallback: bool = False


class Beat(BaseModel):
    model_config = ConfigDict(frozen=True)

    turn: int
    role: str
    internal_goal: str
    may_reveal: bool


def role_for_turn(turn: int) -> str:
    """Clients speak on odd turns, agents on even turns."""
    return "client" if turn % 2 == 1 else "agent"


def scenario_prompt(params: DiceRoll) -> str:
    mistakes = ""
    if params.agent_mistakes:
        mistakes = (
            "\nAgent mistakes to embed (must appear natural, not obvious):\n"
            f"{', '.join(params.agent_mistakes)}\n"
            "mistake_detail: describe exactly how these mistakes appear in conversation"
        )
    if params.hidden_dissatisfaction:
        hidden = (
            "\nhidden_signal: client formally closes conversation "
            "but goal NOT achieved.\n"
            "Give exact phrasing client will use to hide frustration."
        )
    else:
        hidden = 'hidden_signal: ""'
    return (
        "Write a scenario brief for this support chat:\n\n"
        f"Intent: {params.intent}\nCase type: {params.case_type}\n"
        f"Client satisfaction: {params.satisfaction}\n"
        f"Client style: {params.client_style}\n"
        f"Agent style: {params.agent_style}\nSector: {params.sector}\n"
        f"{mistakes}\n{hidden}\n\n"
        "Return JSON:\n"
        '{"situation": "...", "client_goal": "...", "agent_action": "...", '
        '"resolution": "...", "hidden_signal": "...", "mistake_detail": "..."}'
    )


def generate_scenario(params: DiceRoll, backend: Backend, model: str) -> Scenario:
    raw = backend.complete(
        ChatRequest(
            model=model,
            system=SCENARIO_SYSTEM,
            user=scenario_prompt(params),
            max_tokens=400,
        )
    ).text
    data = extract_json_object(raw)
    if data is not None:
        try:
            return Scenario.model_validate(data)
        except ValidationError:
            pass
    return Scenario(
        situation=f"Client contacts support about {params.intent}",
        client_goal="Resolve the issue",
        agent_action="Agent attempts to help",
        resolution="Conversation ends",
        used_fallback=True,
    )


def beat_sheet_prompt(params: DiceRoll, scenario: Scenario, attempt: int) -> str:
    lines = [f"Plan a {params.n_turns}-turn support chat."]
    if attempt > 0:
        lines.append(f"(Retry attempt {attempt}: produce a DIFFERENT plan than before.)")
    lines += [
        "Scenario:",
        f"  situation: {scenario.situation}",
        f"  client_goal: {scenario.client_goal}",
        f"  agent_action: {scenario.agent_action}",
        f"  mistake_detail: {scenario.mistake_detail}",
    ]
    if params.agent_mistakes:
        lines += [
            "Agent mistakes to embed at specific turns:",
            ", ".join(params.agent_mistakes),
            "Pick concrete turn numbers for each mistake.",
            "The mistake must look like normal agent behavior.",
        ]
    if params.hidden_dissatisfaction:
        lines += [
            'Last client turn: client closes with polite phrase ("thanks anyway", '
            '"ok I\'ll try that", "alright then")',
            "but their original goal was NOT achieved.",
        ]
    lines += ["", "Turn structure (STRICT):"]
    lines += [f"turn {t}: {role_for_turn(t)}" for t in range(1, params.n_turns + 1)]
    lines += [
        "",
        f"Resolution happens: turn {params.n_turns - 1} or {params.n_turns} ONLY.",
        "",
        "Return JSON array:",
        '[{"turn": 1, "role": "client", "internal_goal": "...", '
        '"may_reveal": false}, ...]',
    ]
    return "\n".join(lines)


def fallback_beats(params: DiceRoll) -> list[Beat]:
    beats = []
    for turn in range(1, params.n_turns + 1):
        role = role_for_turn(turn)
        may_reveal = turn >= params.n_turns - 1
        if role == "client" and turn == 1:
            goal = "State the problem clearly"
        elif may_reveal:
            goal = "Provide resolution"
        else:
            goal = "Continue troubleshooting"
        beats.append(
            Beat(turn=turn, role=role, internal_goal=goal, may_reveal=may_reveal)
        )
    return beats


def generate_beat_sheet(
    params: DiceRoll, scenario: Scenario, backend: Backend, model: str, attempt: int = 0
) -> tuple[list[Beat], bool]:
    """Return ``(beats, used_fallback)``.

    The model's plan is accepted only if it has exactly one beat per turn with the
    alternating roles the drafter expects; otherwise the fallback plan is used.
    """
    raw = backend.complete(
        ChatRequest(
            model=model,
            system=BEAT_SHEET_SYSTEM,
            user=beat_sheet_prompt(params, scenario, attempt),
            max_tokens=600,
        )
    ).text
    items = extract_json_array(raw)
    if items is not None:
        try:
            beats = [Beat.model_validate(item) for item in items]
        except ValidationError:
            beats = []
        expected = [(t, role_for_turn(t)) for t in range(1, params.n_turns + 1)]
        if [(b.turn, b.role) for b in beats] == expected:
            return beats, False
    return fallback_beats(params), True
