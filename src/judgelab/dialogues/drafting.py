"""Stage 2: draft the dialogue one message at a time from the beat sheet.

The drafter sees only the current beat and the last three messages, never the planned
resolution, so it cannot spoil the outcome early.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict

from judgelab.dialogues.dice import DiceRoll
from judgelab.dialogues.planning import Beat
from judgelab.llm import Backend, ChatRequest

MAX_WORDS = 20
HISTORY_WINDOW = 3
MISTAKE_KEYWORDS = ("mistake", "wrong", "incorrect", "ignore", "miss", "redirect")

DRAFTER_SYSTEM = f"""You are writing a LIVE MOBILE CHAT between a customer
and a support agent.

CRITICAL RULES:
- This is a TEXT CHAT -- short messages, real time, no email format
- Max {MAX_WORDS} words per message
- No greetings ("Hello!", "Hi there!")
- No sign-offs ("Best regards", "Thank you for contacting")
- No stage directions or actions in parentheses
- Write ONLY the message text -- nothing else
- Do NOT resolve the issue unless internal_goal explicitly says so
- Do NOT reference future turns"""

_TEMPLATE_TOKEN_RE = re.compile(r"<\|.*?\|>", re.DOTALL)
_ROLE_PREFIX_RE = re.compile(r"^\[?(CLIENT|AGENT|USER|ASSISTANT)\]?:\s*", re.IGNORECASE)
_QUOTE_WRAP_RE = re.compile(r'^"(.*)"$', re.DOTALL)


class Message(BaseModel):
    model_config = ConfigDict(frozen=True)

    turn: int
    role: str
    content: str
    flags: tuple[str, ...] = ()


def clean_message(text: str) -> str:
    """Strip chat-template tokens, role prefixes and wrapping quotes."""
    text = _TEMPLATE_TOKEN_RE.sub("", text).strip()
    text = _ROLE_PREFIX_RE.sub("", text).strip()
    return _QUOTE_WRAP_RE.sub(r"\1", text).strip()


def truncate_words(text: str, max_words: int = MAX_WORDS) -> str:
    """Cap at ``max_words``, preferring a cut at punctuation in the last 40%."""
    words = text.split()
    if len(words) <= max_words:
        return text
    truncated = " ".join(words[:max_words])
    for punct in ".!?,":
        idx = truncated.rfind(punct)
        if idx > len(truncated) * 0.6:
            return truncated[: idx + 1]
    return truncated + "..."


def beat_flags(params: DiceRoll, beat: Beat) -> tuple[str, ...]:
    """Ground-truth markers carried into the output and used by stylization."""
    flags = []
    if (
        params.hidden_dissatisfaction
        and beat.turn == params.n_turns
        and beat.role == "client"
    ):
        flags.append("hidden_signal")
    if beat.may_reveal:
        flags.append("resolution")
    if (
        beat.role == "agent"
        and params.agent_mistakes
        and not beat.may_reveal
        and any(k in beat.internal_goal.lower() for k in MISTAKE_KEYWORDS)
    ):
        flags.append("mistake")
    return tuple(flags)


def draft_dialogue(
    params: DiceRoll, beats: list[Beat], backend: Backend, model: str
) -> list[Message]:
    messages: list[Message] = []
    for beat in beats:
        history = "\n".join(
            f"[{m.role.upper()}]: {m.content}" for m in messages[-HISTORY_WINDOW:]
        )
        context = f"Chat so far:\n{history}\n\n" if history else ""
        warning = (
            ""
            if beat.may_reveal
            else "WARNING: DO NOT mention resolution or outcome yet."
        )
        user = (
            f"MAX {MAX_WORDS} WORDS. Count before writing.\n\n{context}"
            f"Your role: {beat.role.upper()}\n"
            f"Your task for this message: {beat.internal_goal}\n{warning}\n\n"
            f"Write the message now (max {MAX_WORDS} words):"
        )
        raw = backend.complete(
            ChatRequest(model=model, system=DRAFTER_SYSTEM, user=user, max_tokens=50)
        ).text
        messages.append(
            Message(
                turn=beat.turn,
                role=beat.role,
                content=truncate_words(clean_message(raw)),
                flags=beat_flags(params, beat),
            )
        )
    return messages
