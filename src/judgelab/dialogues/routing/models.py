"""Domain models for route selection (which prompter + actor stylizes a context)."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Confidence = Literal["low", "medium", "high"]
NodeName = Literal["stylization_client", "stylization_agent"]


class ArenaContext(BaseModel):
    """The situation a route is chosen for; also the cache key."""

    model_config = ConfigDict(frozen=True)

    node_name: NodeName
    persona: str = Field(description="Client or agent style, e.g. 'aggressive'.")
    outcome: str = Field(description="Satisfaction level of the dialogue.")
    style: str = Field(description="Case type, e.g. 'conflict'.")
    agent_archetype: str | None = None

    @property
    def key(self) -> str:
        """Deterministic cache key (single source of truth for registry and router)."""
        parts = [self.node_name, self.persona, self.outcome, self.style]
        if self.agent_archetype:
            parts.append(self.agent_archetype)
        return hashlib.md5("|".join(parts).encode()).hexdigest()


def confidence_for(score: float) -> Confidence:
    """Bucket a 1-10 judge score: >= 8 high, >= 6 medium, otherwise low."""
    if score >= 8:
        return "high"
    if score >= 6:
        return "medium"
    return "low"


class WinningRoute(BaseModel):
    """The (prompter, actor, system prompt) that won a micro-arena."""

    model_config = ConfigDict(frozen=True)

    prompter_model: str
    actor_model: str
    system_prompt: str
    judge_score: float = Field(ge=1, le=10)
    confidence: Confidence


class RegistryEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    context: ArenaContext
    route: WinningRoute
    last_updated: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class Candidate(BaseModel):
    """One (prompter, actor) attempt inside a micro-arena."""

    prompter_model: str
    actor_model: str
    system_prompt: str
    generated_text: str
    judge_score: float | None = Field(
        default=None, description="None when the judge output could not be read."
    )
