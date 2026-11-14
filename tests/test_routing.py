"""Registry, epsilon-greedy router and micro-arena on a scripted backend."""

from __future__ import annotations

import json
import random

import pytest

from judgelab.dialogues.routing import (
    ArenaContext,
    ArenaRegistry,
    DynamicRouter,
    MicroArena,
    NoScoredCandidateError,
    WinningRoute,
)
from judgelab.dialogues.routing.micro_arena import JUDGE_SYSTEM
from judgelab.llm import ChatRequest, ScriptedBackend

CTX = ArenaContext(
    node_name="stylization_client",
    persona="aggressive",
    outcome="neutral",
    style="conflict",
)


def _route(score: float, actor: str = "a") -> WinningRoute:
    return WinningRoute(
        prompter_model="p",
        actor_model=actor,
        system_prompt="sys",
        judge_score=score,
        confidence="low",
    )


def test_registry_exact_fallback_and_keep_best(tmp_path) -> None:
    registry = ArenaRegistry(tmp_path / "r.db")
    assert registry.get(CTX) is None and registry.get_fallback(CTX) is None

    registry.save(CTX, _route(7.0, actor="first"))
    registry.save(CTX, _route(4.0, actor="worse"))  # exploration found a worse pair
    assert registry.get(CTX).route.actor_model == "first"

    other_outcome = CTX.model_copy(update={"outcome": "satisfied"})
    assert registry.get(other_outcome) is None
    assert registry.get_fallback(other_outcome).route.actor_model == "first"

    registry.close()
    reopened = ArenaRegistry(tmp_path / "r.db")
    assert reopened.get(CTX).route.actor_model == "first"
    assert reopened.stats()["cached_routes"] == 1


def test_router_miss_queues_discovery_and_serves_baseline() -> None:
    router = DynamicRouter(ArenaRegistry(), epsilon=0.0)
    route, source = router.get_route(CTX)
    assert source == "baseline" and route.judge_score == 5.0
    assert router.take_discoveries() == [CTX]
    assert router.take_discoveries() == []


@pytest.mark.parametrize(("epsilon", "explored"), [(0.0, False), (1.0, True)])
def test_router_epsilon_controls_exploration(epsilon: float, explored: bool) -> None:
    registry = ArenaRegistry()
    registry.save(CTX, _route(6.0))
    router = DynamicRouter(registry, epsilon=epsilon, rng=random.Random(0))
    _, source = router.get_route(CTX)
    assert source == "exact_cache"
    assert (router.take_explorations() == [CTX]) is explored
    router.mark_done(CTX)
    router.get_route(CTX)
    assert router.take_explorations() == []  # never twice in one session


def _arena_backend(scores: dict[str, str]) -> ScriptedBackend:
    def responder(request: ChatRequest) -> str:
        if request.system == JUDGE_SYSTEM:
            actor = request.user.split("Generated message:\n")[1].split()[0]
            return scores[actor]
        if request.model.startswith("prompter"):
            return f"style prompt from {request.model}"
        return f"{request.model} says hi"

    return ScriptedBackend(responder)


def test_micro_arena_picks_best_and_drops_unparsed() -> None:
    scores = {
        "actor1": '{"score": 4, "reason": "meh"}',
        "actor2": "I think this is pretty good overall",  # unparsable
        "actor3": '{"score": 9, "reason": "great"}',
    }
    registry = ArenaRegistry()
    arena = MicroArena(
        _arena_backend(scores),
        registry,
        prompters=("prompter1", "prompter2"),
        actors=("actor1", "actor2", "actor3"),
    )
    route = arena.run(CTX, "my order is late", "aggressive client")
    assert route.actor_model == "actor3" and route.judge_score == 9.0
    assert route.confidence == "high"
    assert registry.get(CTX).route == route


def test_micro_arena_raises_when_nothing_scored() -> None:
    arena = MicroArena(
        _arena_backend({"actor1": "no json here"}),
        ArenaRegistry(),
        prompters=("prompter1",),
        actors=("actor1",),
    )
    with pytest.raises(NoScoredCandidateError):
        arena.run(CTX, "hi", "client")


def test_route_scores_are_validated() -> None:
    with pytest.raises(ValueError):
        WinningRoute.model_validate(
            {**json.loads(_route(5).model_dump_json()), "judge_score": 11}
        )
