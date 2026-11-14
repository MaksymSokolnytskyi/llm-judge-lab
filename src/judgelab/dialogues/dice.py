"""Stage 0: sample the hidden parameters of one dialogue from a seed.

Everything downstream is conditioned on these parameters, and they double as labels:
a generated dialogue comes with its ground-truth intent, case type, satisfaction,
agent mistakes and whether the client hides dissatisfaction. That makes the corpus
usable for evaluating a dialogue *analyzer* against known answers.
"""

from __future__ import annotations

import random

from pydantic import BaseModel, ConfigDict, Field

INTENTS = (
    "billing_issue",
    "technical_error",
    "account_access",
    "tariff_question",
    "refund_request",
)
SATISFACTIONS = ("satisfied", "neutral", "unsatisfied")
AGENT_MISTAKES_POOL = (
    "ignored_question",
    "incorrect_info",
    "rude_tone",
    "no_resolution",
    "unnecessary_escalation",
    "asked_for_info_already_provided",
    "wrong_department_redirect",
)
CONFLICT_MISTAKES = ("rude_tone", "no_resolution", "unnecessary_escalation")
CLIENT_STYLES = (
    "aggressive",
    "passive_aggressive",
    "polite_but_firm",
    "confused_elderly",
    "casual_young",
    "formal_corporate",
)
AGENT_STYLES = (
    "empathetic",
    "by_the_book",
    "rushed_multitask",
    "overly_apologetic",
    "curt_efficient",
)
SECTORS = ("telco", "banking", "retail", "saas", "insurance", "healthcare")

# Not every case is dramatic.
CASE_TYPE_WEIGHTS = {
    "successful": 0.35,
    "problematic": 0.30,
    "conflict": 0.15,
    "agent_mistake": 0.20,
}
# P(satisfied, neutral, unsatisfied | case type).
SATISFACTION_WEIGHTS = {
    "successful": (0.70, 0.25, 0.05),
    "problematic": (0.20, 0.50, 0.30),
    "conflict": (0.05, 0.20, 0.75),
    "agent_mistake": (0.10, 0.35, 0.55),
}
# P(client hides dissatisfaction | satisfaction).
HIDDEN_DISSATISFACTION_PROB = {"satisfied": 0.05, "neutral": 0.30, "unsatisfied": 0.60}
MIN_TURNS, MAX_TURNS = 4, 10


class DiceRoll(BaseModel):
    """Hidden parameters (and ground-truth labels) of one dialogue."""

    model_config = ConfigDict(frozen=True)

    seed: int
    intent: str
    case_type: str
    satisfaction: str
    hidden_dissatisfaction: bool
    agent_mistakes: tuple[str, ...]
    client_style: str
    agent_style: str
    n_turns: int = Field(ge=MIN_TURNS, le=MAX_TURNS)
    sector: str


def roll(seed: int) -> DiceRoll:
    """Deterministic for a given seed (same draw order as the original generator)."""
    rng = random.Random(seed)
    case_type = rng.choices(
        list(CASE_TYPE_WEIGHTS), weights=list(CASE_TYPE_WEIGHTS.values())
    )[0]
    satisfaction = rng.choices(SATISFACTIONS, weights=SATISFACTION_WEIGHTS[case_type])[0]
    hidden = rng.random() < HIDDEN_DISSATISFACTION_PROB[satisfaction]
    if case_type == "agent_mistake":
        mistakes = rng.sample(AGENT_MISTAKES_POOL, rng.randint(1, 3))
    elif case_type == "conflict":
        mistakes = rng.sample(CONFLICT_MISTAKES, k=rng.randint(0, 2))
    else:
        mistakes = []
    n_turns = rng.randint(MIN_TURNS, MAX_TURNS)
    return DiceRoll(
        seed=seed,
        intent=rng.choice(INTENTS),
        case_type=case_type,
        satisfaction=satisfaction,
        hidden_dissatisfaction=hidden,
        agent_mistakes=tuple(mistakes),
        client_style=rng.choice(CLIENT_STYLES),
        agent_style=rng.choice(AGENT_STYLES),
        n_turns=n_turns,
        sector=rng.choice(SECTORS),
    )
