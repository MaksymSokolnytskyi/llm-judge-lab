"""Stage 3: restyle each drafted message through the routed (prompter, actor) pair.

Context mapping: ``persona`` = client or agent style, ``outcome`` = satisfaction,
``style`` = case type. A message whose context has no cached route triggers a
micro-arena before it is stylized; contexts picked for exploration are returned to
the caller to run between dialogues.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from judgelab.dialogues.dice import DiceRoll
from judgelab.dialogues.drafting import Message
from judgelab.dialogues.routing import ArenaContext, DynamicRouter, MicroArena
from judgelab.dialogues.routing.models import NodeName
from judgelab.dialogues.routing.router import RouteSource
from judgelab.llm import Backend, ChatRequest

ARENA_SAMPLES = 3
HISTORY_WINDOW = 3


class RouteLogEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    turn: int
    node: NodeName
    source: RouteSource
    actor_model: str
    route_score: float


def node_for(role: str) -> NodeName:
    return "stylization_client" if role == "client" else "stylization_agent"


def persona_hint(role: str, params: DiceRoll) -> str:
    if role == "client":
        return (
            f"{params.client_style} client, {params.satisfaction} satisfaction, "
            f"{params.sector} sector, case: {params.case_type}"
        )
    return f"{params.agent_style} agent, {params.sector} sector, case: {params.case_type}"


def context_for(role: str, params: DiceRoll) -> ArenaContext:
    return ArenaContext(
        node_name=node_for(role),
        persona=params.client_style if role == "client" else params.agent_style,
        outcome=params.satisfaction,
        style=params.case_type,
    )


def arena_samples(messages: list[Message]) -> dict[NodeName, str]:
    """Representative raw messages per node, used as micro-arena input."""
    client = [m.content for m in messages if m.role == "client"][:ARENA_SAMPLES]
    agent = [m.content for m in messages if m.role == "agent"][:ARENA_SAMPLES]
    return {
        "stylization_client": " | ".join(client) or "I have a problem.",
        "stylization_agent": " | ".join(agent) or "I can help with that.",
    }


def _flag_hint(flags: tuple[str, ...]) -> str:
    if "hidden_signal" in flags:
        return (
            "\nIMPORTANT: client appears to accept the outcome, but subtly signals "
            "unresolved frustration -- do NOT be explicit."
        )
    if "mistake" in flags:
        return (
            "\nIMPORTANT: agent response should look completely normal but subtly fail "
            "to address the actual issue."
        )
    return ""


def stylize_dialogue(
    params: DiceRoll,
    messages: list[Message],
    backend: Backend,
    router: DynamicRouter,
    arena: MicroArena,
) -> tuple[list[Message], list[RouteLogEntry]]:
    samples = arena_samples(messages)
    stylized: list[Message] = []
    route_log: list[RouteLogEntry] = []

    for message in messages:
        ctx = context_for(message.role, params)
        if stylized:
            history = "\n".join(
                f"  [{m.role.upper()}]: {m.content}" for m in stylized[-HISTORY_WINDOW:]
            )
            context_block = f"Conversation so far:\n{history}"
        else:
            context_block = (
                "This is the opening message. "
                f"Topic: {params.intent}, sector: {params.sector}"
            )

        route, source = router.get_route(ctx)
        discovered = router.take_discoveries()
        for queued in discovered:
            role = "client" if queued.node_name == "stylization_client" else "agent"
            arena.run(
                queued,
                samples[queued.node_name],
                persona_hint(role, params),
                context_block=context_block,
            )
            router.mark_done(queued)
        if discovered and (fresh := router.registry.get(ctx)) is not None:
            route, source = fresh.route, "arena"

        user = (
            f"{context_block}\n\nOriginal message to rewrite:\n{message.content}\n\n"
            f"Rewrite in character. Stay strictly on topic. "
            f"Do NOT introduce new topics.{_flag_hint(message.flags)}\n"
            "Output ONLY the rewritten message text."
        )
        text = backend.complete(
            ChatRequest(
                model=route.actor_model,
                system=route.system_prompt,
                user=user,
                max_tokens=150,
            )
        ).text.strip()
        stylized.append(message.model_copy(update={"content": text or message.content}))
        route_log.append(
            RouteLogEntry(
                turn=message.turn,
                node=ctx.node_name,
                source=source,
                actor_model=route.actor_model,
                route_score=route.judge_score,
            )
        )
    return stylized, route_log
