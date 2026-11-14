"""Statistics primitives, plus a regression on the committed 2026-03-02 run.

The regression pins the numbers the README quotes; if the analysis changes, this
test says which claim moved.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from judgelab.cross_judge.analysis import (
    bootstrap_mean_ci,
    join_rows,
    judge_summaries,
    pairwise_spearman,
    ranks,
    read_jsonl,
    refusal_table,
    self_preference,
    spearman,
    unanimity,
)

RUN = Path(__file__).resolve().parents[1] / "results" / "cross_judge_2026-03-02"


def test_ranks_average_ties() -> None:
    assert ranks([10, 20, 20, 30]) == [1.0, 2.5, 2.5, 4.0]


def test_spearman_extremes() -> None:
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


def test_spearman_constant_is_nan() -> None:
    assert math.isnan(spearman([1, 2, 3], [5, 5, 5]))


def test_bootstrap_is_seeded_and_brackets_mean() -> None:
    values = [3.0, 5.0, 7.0, 9.0, 4.0, 6.0]
    assert bootstrap_mean_ci(values) == bootstrap_mean_ci(values)
    lo, hi = bootstrap_mean_ci(values)
    assert lo <= sum(values) / len(values) <= hi


@pytest.fixture(scope="module")
def rows():
    return join_rows(
        read_jsonl(RUN / "generations.jsonl"), read_jsonl(RUN / "judgments.jsonl")
    )


def test_run_shape(rows) -> None:
    assert len(rows) == 810
    assert len({r.gen_id for r in rows}) == 270


def test_refusals_explain_llama_mean(rows) -> None:
    by_judge = {s.judge: s for s in judge_summaries(rows)}
    llama = by_judge["llama3.1:8b"]
    assert (llama.scored, llama.refused, llama.unparsed) == (149, 116, 5)
    assert llama.mean_zeros_included == pytest.approx(2.51, abs=0.005)
    assert llama.mean_scored == pytest.approx(4.56, abs=0.005)
    assert refusal_table(rows)[("llama3.1:8b", "toxic")] == (114, 135)


def test_judge_spread_and_flag_contradictions(rows) -> None:
    by_judge = {s.judge: s for s in judge_summaries(rows)}
    common = {j: s.mean_common for j, s in by_judge.items()}
    assert common["dolphin-mistral"] == pytest.approx(7.45, abs=0.005)
    assert common["qwen2.5:7b"] == pytest.approx(6.43, abs=0.005)
    assert common["llama3.1:8b"] == pytest.approx(4.56, abs=0.005)
    assert by_judge["qwen2.5:7b"].flag_contradictions == 112
    assert by_judge["dolphin-mistral"].flag_contradictions == 67


def test_agreement_is_low(rows) -> None:
    assert unanimity(rows) == (27, 149)
    rhos = {(a, b): rho for a, b, _, rho in pairwise_spearman(rows)}
    assert all(rho < 0.5 for rho in rhos.values())


def test_qwen_self_preference_ci_excludes_zero(rows) -> None:
    qwen = next(sp for sp in self_preference(rows) if sp.judge == "qwen2.5:7b")
    assert qwen.bias == pytest.approx(0.84, abs=0.005)
    assert qwen.ci_low > 0
