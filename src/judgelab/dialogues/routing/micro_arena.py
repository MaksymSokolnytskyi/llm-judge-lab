"""Micro-arena: a small judged tournament that picks a stylization route.

Every prompter model writes a system prompt; every actor model rewrites a sample
message under every prompt; one judge scores all |prompters| x |actors| candidates.
The best-scoring pair is cached in the registry.

Candidates whose judge output cannot be read are dropped. The original code scored
them 5.0, which let a parse failure beat a genuinely judged 4.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from judgelab.dialogues.routing.models import (
    ArenaContext,
    Candidate,
    WinningRoute,
    confidence_for,
)
from judgelab.dialogues.routing.registry import ArenaRegistry
from judgelab.llm import Backend, ChatRequest
from judgelab.verdict import parse_verdict

log = logging.getLogger(__name__)

DEFAULT_PROMPTERS: tuple[str, ...] = (
    "llama3.1:8b",
    "qwen2.5:7b",
    "nous-hermes2:10.7b",
    "mistral:7b",
)
DEFAULT_ACTORS: tuple[str, ...] = (
    "llama3.1:8b",
    "dolphin-mistral",
    "mistral:7b",
    "nous-hermes2:10.7b",
)
DEFAULT_JUDGE = "qwen2.5:7b"

PROMPTER_SYSTEM = """You are writing a system prompt for an AI that will rewrite
chat messages. The prompt must be specific, short (max 100 words),
and enforce the exact persona and style needed.
Output ONLY the system prompt text -- nothing else."""

JUDGE_SYSTEM = """You are evaluating a single message from a synthetic
customer support chat transcript.
Score 1-10. Reply ONLY with a JSON object: {"score": N, "reason": "one sentence"}

BASE SCORING:
- 10: Perfectly natural chat message. Sounds like a real human typing.
- 7-9: Good. Minor issues but believable.
- 4-6: Acceptable but noticeably robotic or slightly off.
- 1-3: Clearly AI-generated, wrong persona, broken, or unreadable.

AUTOMATIC PENALTIES (apply before scoring, each penalty is cumulative):
-3 if message contains corporate filler phrases:
   'time is of essence', 'expedite', 'at this juncture',
   'going forward', 'utilize', 'as per', 'kindly',
   'please be advised', 'I apologize for any inconvenience',
   'we value your feedback', 'rest assured'
-2 if agent message opens with a formal greeting:
   'Hello!', 'Hi there!', 'Good day', 'Greetings'
-2 if agent message ends with a formal sign-off:
   'Best regards', 'Sincerely', 'Kind regards', 'Thank you for contacting'
-2 if agent message is longer than 35 words
-2 if client message repeats the exact same complaint phrase
   from a previous message (copy-paste, not paraphrase)
-3 if the message introduces a completely different topic
   than what was being discussed (topic switch bug)

BONUSES:
+1 if agent message is under 20 words AND resolves or advances the issue
+1 if client message shows clear emotional arc progression
   (escalating, deescalating, or resolving -- not flat)

PERSONA MATCH CHECK:
Before scoring, verify the message matches the expected persona.
If persona is elderly_confused: message should use simple words,
   possible repetition, mild confusion -- NOT technical jargon.
If persona is empathetic_resolver: agent should acknowledge emotion
   briefly then move to action -- NOT lecture the customer.
If persona is aggressive: short sentences, caps, demands --
   NOT polite requests.
Persona mismatch: -2 points.

FINAL RULE:
Minimum score is 1. Maximum after all bonuses is 10.
Always apply penalties first, then bonuses, then clamp to [1, 10]."""


class NoScoredCandidateError(RuntimeError):
    """Raised when the judge could not score any candidate."""


class MicroArena:
    def __init__(
        self,
        backend: Backend,
        registry: ArenaRegistry,
        prompters: Sequence[str] = DEFAULT_PROMPTERS,
        actors: Sequence[str] = DEFAULT_ACTORS,
        judge: str = DEFAULT_JUDGE,
    ):
        self.backend = backend
        self.registry = registry
        self.prompters = tuple(prompters)
        self.actors = tuple(actors)
        self.judge_model = judge

    def run(
        self,
        ctx: ArenaContext,
        sample_input: str,
        persona_hint: str,
        context_block: str = "",
    ) -> WinningRoute:
        """Run the tournament for ``ctx`` and cache the winner.

        ``context_block`` (recent conversation) is shown to actors only, so prompters
        write a general style prompt rather than one tied to a single exchange.
        """
        candidates: list[Candidate] = []
        for prompter in self.prompters:
            system_prompt = self._write_prompt(prompter, ctx, persona_hint)
            for actor in self.actors:
                candidates.append(
                    self._act(system_prompt, prompter, actor, sample_input, context_block)
                )
        role = "agent" if ctx.node_name == "stylization_agent" else "client"
        for candidate in candidates:
            candidate.judge_score = self._judge(
                candidate.generated_text, persona_hint, role
            )
        scored = [c for c in candidates if c.judge_score is not None]
        if not scored:
            raise NoScoredCandidateError(
                f"judge scored none of {len(candidates)} candidates"
            )
        winner = max(scored, key=lambda c: c.judge_score or 0.0)
        assert winner.judge_score is not None
        route = WinningRoute(
            prompter_model=winner.prompter_model,
            actor_model=winner.actor_model,
            system_prompt=winner.system_prompt,
            judge_score=winner.judge_score,
            confidence=confidence_for(winner.judge_score),
        )
        log.info(
            "arena %s/%s: %d/%d scored, winner %s+%s = %.1f",
            ctx.node_name,
            ctx.persona,
            len(scored),
            len(candidates),
            winner.prompter_model,
            winner.actor_model,
            winner.judge_score,
        )
        return self.registry.save(ctx, route).route

    def _write_prompt(self, prompter: str, ctx: ArenaContext, persona_hint: str) -> str:
        user = (
            f"Write a system prompt for rewriting a {ctx.node_name} message.\n"
            f"Persona: {persona_hint}\nStyle: {ctx.style}\nOutcome: {ctx.outcome}\n"
            f"Node: {ctx.node_name}\n\n"
            "The system prompt must enforce:\n- Correct persona voice\n"
            "- Correct message length for a chat (not email)\n- No AI cliches\n"
            "- Exact style markers"
        )
        request = ChatRequest(
            model=prompter, system=PROMPTER_SYSTEM, user=user, max_tokens=200
        )
        return self.backend.complete(request).text

    def _act(
        self,
        system_prompt: str,
        prompter: str,
        actor: str,
        sample_input: str,
        context_block: str,
    ) -> Candidate:
        if context_block:
            user = (
                f"{context_block}\n\nOriginal message to rewrite:\n{sample_input}\n\n"
                "Rewrite it in character. Stay on the same topic as the conversation "
                "above. Do NOT introduce new topics.\n"
                "Output ONLY the rewritten message text."
            )
        else:
            user = f"Original message:\n{sample_input}\n\nRewrite it:"
        request = ChatRequest(
            model=actor, system=system_prompt, user=user, max_tokens=150
        )
        return Candidate(
            prompter_model=prompter,
            actor_model=actor,
            system_prompt=system_prompt,
            generated_text=self.backend.complete(request).text,
        )

    def _judge(self, text: str, persona_hint: str, role: str) -> float | None:
        user = (
            f"Role: {role.upper()}\nPersona expected: {persona_hint}\n"
            f"Word count: {len(text.split())}\n\nGenerated message:\n{text}\n\n"
            "Apply the rubric. Score it."
        )
        request = ChatRequest(
            model=self.judge_model, system=JUDGE_SYSTEM, user=user, max_tokens=120
        )
        verdict = parse_verdict(self.backend.complete(request).text)
        if verdict.score is None:
            return None
        return float(min(10, max(1, verdict.score)))
