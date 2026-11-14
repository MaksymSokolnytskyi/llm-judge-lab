"""Stage 4: judge the whole dialogue as a QA gate (pass, or retry from the beat sheet)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from judgelab.dialogues.dice import DiceRoll
from judgelab.dialogues.drafting import Message
from judgelab.llm import Backend, ChatRequest
from judgelab.parsing import extract_json_object

MACRO_JUDGE_SYSTEM = """You are a QA Director evaluating a synthetic
customer support chat dataset entry.

Evaluate the ENTIRE chat log as a whole.
Reply ONLY with JSON:
{"passed": true/false, "score": N, "failures": [...], "notes": "..."}

Evaluation criteria (all must pass for passed=true):

1. SPOILER CHECK: Does the client know the resolution before it happens?
   - Client must NOT thank the agent before the agent has helped
   - Client must NOT mention the issue is resolved in turn 1 or 2
   - FAIL if: client says "thanks for fixing it" before agent did anything

2. FLOW CHECK: Does the conversation flow naturally?
   - Problem must escalate or develop across middle turns
   - Agent must respond to what client actually said
   - FAIL if: agent asks for info client already provided

3. FORMAT CHECK: Does it look like a real chat?
   - Messages should be under 30 words each
   - No email-style formatting
   - FAIL if: any message over 50 words

4. HIDDEN DISSATISFACTION CHECK (only if flagged):
   - Client must close with polite phrase but goal NOT achieved
   - Dissatisfaction must be subtle, not obvious
   - FAIL if: client explicitly states they are unhappy in final turn

5. AGENT MISTAKE CHECK (only if flagged):
   - Mistake must be present but look like normal behavior
   - FAIL if: no mistake found OR mistake is too obvious"""

PARSE_ERROR = "judge_parse_error"


class MacroVerdict(BaseModel):
    model_config = ConfigDict(frozen=True)

    passed: bool
    score: float = Field(ge=0, le=10)
    failures: tuple[str, ...] = ()
    notes: str = ""


def judge_prompt(params: DiceRoll, messages: list[Message]) -> str:
    chat_log = "\n".join(
        f"Turn {m.turn} [{m.role.upper()}]: {m.content}" for m in messages
    )
    checks = []
    if params.hidden_dissatisfaction:
        checks.append(
            "hidden_dissatisfaction=TRUE: verify client hides frustration in last turn"
        )
    if params.agent_mistakes:
        checks.append(
            f"agent_mistakes={list(params.agent_mistakes)}: verify they appear naturally"
        )
    return (
        "Evaluate this support chat:\n\n"
        f"Expected intent: {params.intent}\nExpected case type: {params.case_type}\n"
        f"Expected satisfaction: {params.satisfaction}\n" + "\n".join(checks) + "\n\n"
        f"CHAT LOG:\n{chat_log}\n\nEvaluate all 5 criteria. Return JSON."
    )


def judge_dialogue(
    params: DiceRoll, messages: list[Message], backend: Backend, model: str
) -> MacroVerdict:
    """An unreadable verdict fails the gate (and is labelled), never passes it."""
    raw = backend.complete(
        ChatRequest(
            model=model,
            system=MACRO_JUDGE_SYSTEM,
            user=judge_prompt(params, messages),
            max_tokens=300,
        )
    ).text
    data = extract_json_object(raw)
    if data is not None:
        try:
            return MacroVerdict.model_validate(data)
        except ValidationError:
            pass
    return MacroVerdict(passed=False, score=0.0, failures=(PARSE_ERROR,), notes=raw[:200])
