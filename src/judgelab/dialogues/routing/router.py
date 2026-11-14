"""Choose a stylization route per context: cache first, explore occasionally.

This is an epsilon-greedy policy over cached routes:

* **exploit** -- an exact cache hit is used as is;
* **explore** -- with probability ``epsilon`` that hit is also queued for a fresh
  micro-arena *after* the current dialogue, so the incumbent can be replaced if a better
  pair turns up (the registry keeps the better of the two);
* **discover** -- a miss serves the fallback or baseline route and queues an arena run
  immediately, because the dialogue needs a route now.

In the original code ``epsilon`` only incremented a counter and never explored; here
exploration is real but deferred, so a dialogue never waits on an exploratory arena.
"""

from __future__ import annotations

import random
from typing import Literal

from judgelab.dialogues.routing.models import ArenaContext, NodeName, WinningRoute
from judgelab.dialogues.routing.registry import ArenaRegistry

RouteSource = Literal["exact_cache", "fallback_cache", "baseline", "arena"]

DEFAULT_EPSILON = 0.10

BASELINE_ROUTES: dict[NodeName, WinningRoute] = {
    "stylization_client": WinningRoute(
        prompter_model="llama3.1:8b",
        actor_model="qwen2.5:7b",
        system_prompt=(
            "Rewrite customer message. Match style. Max 30 words. "
            "Output ONLY the message."
        ),
        judge_score=5.0,
        confidence="low",
    ),
    "stylization_agent": WinningRoute(
        prompter_model="llama3.1:8b",
        actor_model="llama3.1:8b",
        system_prompt=(
            "Rewrite agent message. Direct, human, max 25 words. No corporate phrases."
        ),
        judge_score=5.0,
        confidence="low",
    ),
}


class DynamicRouter:
    def __init__(
        self,
        registry: ArenaRegistry,
        epsilon: float = DEFAULT_EPSILON,
        rng: random.Random | None = None,
    ):
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        self.registry = registry
        self.epsilon = epsilon
        self._rng = rng or random.Random(0)
        self._discover: list[ArenaContext] = []
        self._explore: list[ArenaContext] = []
        self._done_this_session: set[str] = set()

    def get_route(self, ctx: ArenaContext) -> tuple[WinningRoute, RouteSource]:
        entry = self.registry.get(ctx)
        if entry is not None:
            if (
                ctx.key not in self._done_this_session
                and self._rng.random() < self.epsilon
                and ctx not in self._explore
            ):
                self._explore.append(ctx)
            return entry.route, "exact_cache"

        self._queue(self._discover, ctx)
        fallback = self.registry.get_fallback(ctx)
        if fallback is not None:
            return fallback.route, "fallback_cache"
        return BASELINE_ROUTES[ctx.node_name], "baseline"

    def mark_done(self, ctx: ArenaContext) -> None:
        """Record that an arena ran for ``ctx``; it is not queued again this session."""
        self._done_this_session.add(ctx.key)

    def take_discoveries(self) -> list[ArenaContext]:
        """Contexts with no cached route -- run these before continuing."""
        queued, self._discover = self._discover, []
        return queued

    def take_explorations(self) -> list[ArenaContext]:
        """Cached contexts selected for re-exploration -- run between dialogues."""
        queued, self._explore = self._explore, []
        return queued

    def _queue(self, queue: list[ArenaContext], ctx: ArenaContext) -> None:
        if ctx.key not in self._done_this_session and ctx not in queue:
            queue.append(ctx)
