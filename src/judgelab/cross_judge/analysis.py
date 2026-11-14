"""Statistics on a cross-judge run, computed from the raw JSONL.

Single source of truth for every number in the README tables: the report is
regenerated from ``generations.jsonl`` and ``judgments.jsonl`` and never edited by hand.

Reads both formats: the original 2026-03-02 run (``score`` always an int, refusals and
parse failures stored as 0) and the current one (explicit ``status``). For the old
format the status is recovered with :func:`judgelab.verdict.classify_legacy`.

Measures
--------
* **Severity**: mean score per judge over *scored* verdicts only, with a percentile
  bootstrap 95% CI. The legacy "zeros included" mean is shown next to it.
* **Refusals**: rate per judge × persona.
* **Self-preference**: for judge :math:`j` and generation :math:`g` scored by all
  judges, :math:`\\delta_j(g) = s_j(g) - \\frac{1}{|K|-1}\\sum_{k\\ne j} s_k(g)`.
  Self-preference is :math:`\\bar\\delta_j` over the judge's own generations minus
  :math:`\\bar\\delta_j` over everyone else's. Subtracting the other judges' view of the
  same text removes text quality; what remains is how much more generous the judge is
  to itself than it usually is.
* **Agreement**: pairwise Spearman rank correlation and the share of generations on
  which all judges reach the same pass/fail decision.
* **Flag consistency**: how often the judge's own ``accepted`` flag contradicts its
  score at the stated threshold.
"""

from __future__ import annotations

import itertools
import json
import random
import re
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from judgelab.verdict import VerdictStatus, classify_legacy

_RUSSIAN_ONLY = re.compile(r"[ыэъёЫЭЪЁ]")
_UKRAINIAN_ONLY = re.compile(r"[іїєґІЇЄҐ]")


class Row(BaseModel):
    """One verdict joined with the generation it judges."""

    model_config = ConfigDict(frozen=True)

    gen_id: str
    judge: str
    generator: str
    meta_author: str
    persona: str
    approach: str
    status: VerdictStatus
    score: int | None
    judge_accepted: bool | None
    feedback: str


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def join_rows(
    generations: Iterable[Mapping[str, Any]], judgments: Iterable[Mapping[str, Any]]
) -> list[Row]:
    """Join verdicts to generations, normalising legacy records."""
    by_id = {g["gen_id"]: g for g in generations}
    rows: list[Row] = []
    for j in judgments:
        g = by_id[j["gen_id"]]
        if "status" in j:
            status: VerdictStatus = j["status"]
            score = j["score"]
        else:
            status = classify_legacy(int(j["score"]), str(j.get("feedback", "")))
            score = int(j["score"]) if status == "scored" else None
        accepted = j.get("judge_accepted", j.get("accepted"))
        rows.append(
            Row(
                gen_id=j["gen_id"],
                judge=j["judge_model"],
                generator=g["generator_model"],
                meta_author=g["meta_author"],
                persona=g["persona"],
                approach=g["approach"],
                status=status,
                score=score,
                judge_accepted=accepted if isinstance(accepted, bool) else None,
                feedback=str(j.get("feedback", "")),
            )
        )
    return rows


# -- primitive statistics ---------------------------------------------------------------


def ranks(values: Sequence[float]) -> list[float]:
    """Average ranks (1-based), ties share the mean rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.0] * len(values)
    i = 0
    while i < len(order):
        k = i
        while k + 1 < len(order) and values[order[k + 1]] == values[order[i]]:
            k += 1
        for m in range(i, k + 1):
            result[order[m]] = (i + k) / 2 + 1
        i = k + 1
    return result


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("need two equal-length sequences of length >= 2")
    mx, my = statistics.fmean(x), statistics.fmean(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx == 0 or syy == 0:
        return float("nan")
    return sxy / (sxx * syy) ** 0.5


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman's rho as the Pearson correlation of average ranks."""
    return pearson(ranks(x), ranks(y))


def bootstrap_mean_ci(
    values: Sequence[float], n_resamples: int = 2000, seed: int = 0, level: float = 0.95
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean. Seeded, so reports are reproducible."""
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    means = sorted(statistics.fmean(rng.choices(values, k=n)) for _ in range(n_resamples))
    lo = means[int((1 - level) / 2 * n_resamples)]
    hi = means[int((1 + level) / 2 * n_resamples) - 1]
    return (lo, hi)


def bootstrap_diff_ci(
    a: Sequence[float],
    b: Sequence[float],
    n_resamples: int = 2000,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap CI for ``mean(a) - mean(b)``, resampling each group."""
    if not a or not b:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    diffs = sorted(
        statistics.fmean(rng.choices(a, k=len(a)))
        - statistics.fmean(rng.choices(b, k=len(b)))
        for _ in range(n_resamples)
    )
    return (
        diffs[int((1 - level) / 2 * n_resamples)],
        diffs[int((1 + level) / 2 * n_resamples) - 1],
    )


# -- summaries -----------------------------------------------------------------------


class JudgeSummary(BaseModel):
    judge: str
    n: int
    scored: int
    refused: int
    unparsed: int
    mean_zeros_included: float
    mean_scored: float
    ci_low: float
    ci_high: float
    mean_common: float
    pass_rate: float
    judge_accept_rate: float
    flag_contradictions: int
    flag_checked: int


def judge_summaries(rows: Sequence[Row], pass_score: int = 7) -> list[JudgeSummary]:
    common = _fully_scored(rows)
    out: list[JudgeSummary] = []
    for judge in sorted({r.judge for r in rows}):
        mine = [r for r in rows if r.judge == judge]
        scores = [r.score for r in mine if r.score is not None]
        flagged = [
            r for r in mine if r.score is not None and r.judge_accepted is not None
        ]
        lo, hi = bootstrap_mean_ci(scores)
        out.append(
            JudgeSummary(
                judge=judge,
                n=len(mine),
                scored=len(scores),
                refused=sum(r.status == "refused" for r in mine),
                unparsed=sum(r.status == "unparsed" for r in mine),
                mean_zeros_included=statistics.fmean(r.score or 0 for r in mine),
                mean_scored=statistics.fmean(scores) if scores else float("nan"),
                ci_low=lo,
                ci_high=hi,
                mean_common=statistics.fmean(c[judge] for c in common.values())
                if common
                else float("nan"),
                pass_rate=sum(s >= pass_score for s in scores) / len(mine),
                judge_accept_rate=sum(bool(r.judge_accepted) for r in mine) / len(mine),
                flag_contradictions=sum(
                    r.judge_accepted != (r.score >= pass_score)  # type: ignore[operator]
                    for r in flagged
                ),
                flag_checked=len(flagged),
            )
        )
    return out


def refusal_table(rows: Sequence[Row]) -> dict[tuple[str, str], tuple[int, int]]:
    """(judge, persona) -> (refused, total)."""
    table: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for r in rows:
        cell = table[(r.judge, r.persona)]
        cell[0] += r.status == "refused"
        cell[1] += 1
    return {k: (v[0], v[1]) for k, v in sorted(table.items())}


def _fully_scored(rows: Sequence[Row]) -> dict[str, dict[str, int]]:
    """gen_id -> {judge: score} for generations every judge scored."""
    judges = {r.judge for r in rows}
    by_gen: dict[str, dict[str, int]] = defaultdict(dict)
    for r in rows:
        if r.score is not None:
            by_gen[r.gen_id][r.judge] = r.score
    return {g: s for g, s in by_gen.items() if set(s) == judges}


class SelfPreference(BaseModel):
    judge: str
    delta_own: float
    delta_other: float
    n_own: int
    n_other: int
    ci_low: float
    ci_high: float

    @property
    def bias(self) -> float:
        return self.delta_own - self.delta_other


def self_preference(rows: Sequence[Row]) -> list[SelfPreference]:
    full = _fully_scored(rows)
    generator_of = {r.gen_id: r.generator for r in rows}
    judges = sorted({r.judge for r in rows})
    out: list[SelfPreference] = []
    for judge in judges:
        own: list[float] = []
        other: list[float] = []
        for gen_id, scores in full.items():
            peers = [s for k, s in scores.items() if k != judge]
            delta = scores[judge] - statistics.fmean(peers)
            (own if generator_of[gen_id] == judge else other).append(delta)
        lo, hi = bootstrap_diff_ci(own, other)
        out.append(
            SelfPreference(
                judge=judge,
                delta_own=statistics.fmean(own) if own else float("nan"),
                delta_other=statistics.fmean(other) if other else float("nan"),
                n_own=len(own),
                n_other=len(other),
                ci_low=lo,
                ci_high=hi,
            )
        )
    return out


def pairwise_spearman(rows: Sequence[Row]) -> list[tuple[str, str, int, float]]:
    """(judge_a, judge_b, n_pairs, rho) over generations both judges scored."""
    by_gen: dict[str, dict[str, int]] = defaultdict(dict)
    for r in rows:
        if r.score is not None:
            by_gen[r.gen_id][r.judge] = r.score
    judges = sorted({r.judge for r in rows})
    out = []
    for a, b in itertools.combinations(judges, 2):
        pairs = [(s[a], s[b]) for s in by_gen.values() if a in s and b in s]
        if len(pairs) >= 2:
            xs, ys = zip(*pairs, strict=True)
            out.append((a, b, len(pairs), spearman(xs, ys)))
    return out


def unanimity(rows: Sequence[Row], pass_score: int = 7) -> tuple[int, int]:
    """(generations where all judges agree on pass/fail, generations scored by all)."""
    full = _fully_scored(rows)
    agree = sum(
        len({s >= pass_score for s in scores.values()}) == 1 for scores in full.values()
    )
    return agree, len(full)


def feedback_languages(rows: Sequence[Row]) -> dict[str, dict[str, int]]:
    """Per judge: counts of feedback containing Russian-only / Ukrainian-only letters."""
    out: dict[str, dict[str, int]] = {}
    for judge in sorted({r.judge for r in rows}):
        mine = [r.feedback for r in rows if r.judge == judge]
        out[judge] = {
            "n": len(mine),
            "russian": sum(bool(_RUSSIAN_ONLY.search(t)) for t in mine),
            "ukrainian": sum(bool(_UKRAINIAN_ONLY.search(t)) for t in mine),
        }
    return out


def mean_by(rows: Sequence[Row], key: str) -> dict[str, tuple[float, int]]:
    """Mean scored verdict grouped by a Row attribute, e.g. ``approach``."""
    groups: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        if r.score is not None:
            groups[getattr(r, key)].append(r.score)
    return {k: (statistics.fmean(v), len(v)) for k, v in sorted(groups.items())}


# -- report --------------------------------------------------------------------------


def _short(model: str) -> str:
    return model.split(":")[0]


def render_report(rows: Sequence[Row], title: str, pass_score: int = 7) -> str:
    """Markdown report with every table the README quotes."""
    n_gens = len({r.gen_id for r in rows})
    lines = [
        f"# {title}",
        "",
        "Generated by `experiments/analyze_cross_judge.py` from the raw JSONL in this "
        "folder. Do not edit by hand.",
        "",
        f"{n_gens} generations, {len(rows)} verdicts, pass threshold {pass_score}.",
        "",
        "## Judge severity",
        "",
        "| judge | verdicts | scored | refused | unparsed | mean, zeros included "
        "| mean, scored only [95% CI] | mean, texts every judge scored "
        "| pass rate (score) | pass rate (judge's flag) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    summaries = judge_summaries(rows, pass_score)
    for s in summaries:
        lines.append(
            f"| {_short(s.judge)} | {s.n} | {s.scored} | {s.refused} | {s.unparsed} "
            f"| {s.mean_zeros_included:.2f} | {s.mean_scored:.2f} "
            f"[{s.ci_low:.2f}, {s.ci_high:.2f}] | {s.mean_common:.2f} "
            f"| {s.pass_rate:.0%} | {s.judge_accept_rate:.0%} |"
        )
    common_means = [s.mean_common for s in summaries]
    agree_n = unanimity(rows, pass_score)[1]
    lines += [
        "",
        "Means over each judge's own scored texts are not comparable when one judge "
        "refuses a whole persona, so the spread is measured on the "
        f"{agree_n} texts every judge scored: "
        f"{max(common_means) - min(common_means):.2f} points.",
        "",
        "## The judge's own pass/fail flag",
        "",
        "| judge | flag contradicts score >= threshold | checked |",
        "|---|---|---|",
    ]
    for s in summaries:
        lines.append(
            f"| {_short(s.judge)} | {s.flag_contradictions} | {s.flag_checked} |"
        )
    lines += [
        "",
        "## Refusals by persona",
        "",
        "| judge | persona | refused |",
        "|---|---|---|",
    ]
    for (judge, persona), (ref, tot) in refusal_table(rows).items():
        lines.append(f"| {_short(judge)} | {persona} | {ref}/{tot} |")
    lines += [
        "",
        "## Self-preference",
        "",
        "delta = judge's score minus the mean of the other judges on the same text "
        "(texts scored by every judge only).",
        "",
        "| judge | delta, own texts | delta, others' texts "
        "| self-preference [95% CI] | n own / other |",
        "|---|---|---|---|---|",
    ]
    for sp in self_preference(rows):
        lines.append(
            f"| {_short(sp.judge)} | {sp.delta_own:+.2f} | {sp.delta_other:+.2f} "
            f"| {sp.bias:+.2f} [{sp.ci_low:+.2f}, {sp.ci_high:+.2f}] "
            f"| {sp.n_own} / {sp.n_other} |"
        )
    agree, total = unanimity(rows, pass_score)
    lines += [
        "",
        "## Agreement",
        "",
        "| judges | pairs | Spearman rho |",
        "|---|---|---|",
    ]
    for a, b, n, rho in pairwise_spearman(rows):
        lines.append(f"| {_short(a)} vs {_short(b)} | {n} | {rho:.2f} |")
    lines += [
        "",
        f"All judges reach the same pass/fail decision on {agree} of {total} "
        "generations that every judge scored.",
        "",
        "## Feedback language",
        "",
        "| judge | feedback with Russian-only letters "
        "| with Ukrainian-only letters | n |",
        "|---|---|---|---|",
    ]
    for judge, c in feedback_languages(rows).items():
        lines.append(
            f"| {_short(judge)} | {c['russian']} | {c['ukrainian']} | {c['n']} |"
        )
    lines += ["", "## Mean score by condition (scored verdicts, all judges)", ""]
    for key in ("persona", "approach", "generator", "meta_author"):
        lines += [f"**{key}**", "", "| value | mean | n |", "|---|---|---|"]
        for value, (mean, n) in mean_by(rows, key).items():
            lines.append(f"| {_short(value)} | {mean:.2f} | {n} |")
        lines.append("")
    return "\n".join(lines)
