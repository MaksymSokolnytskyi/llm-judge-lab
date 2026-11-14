"""Regenerate REPORT.md for a cross-judge run from its raw JSONL.

    python experiments/analyze_cross_judge.py results/cross_judge_2026-03-02

Every number quoted in the README comes from this script's output.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from judgelab.cross_judge.analysis import (
    join_rows,
    judge_summaries,
    read_jsonl,
    render_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--pass-score", type=int, default=7)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    generations = read_jsonl(args.run_dir / "generations.jsonl")
    judgments = read_jsonl(args.run_dir / "judgments.jsonl")
    rows = join_rows(generations, judgments)

    # Sanity: every verdict joins to a generation and scores stay in range.
    assert len(rows) == len(judgments)
    assert all(r.score is None or 0 <= r.score <= 10 for r in rows)
    for s in judge_summaries(rows, args.pass_score):
        assert s.scored + s.refused + s.unparsed == s.n
        assert math.isnan(s.mean_scored) or s.ci_low <= s.mean_scored <= s.ci_high

    title = args.title or f"Cross-judge run: {args.run_dir.name}"
    report = render_report(rows, title, args.pass_score)
    (args.run_dir / "REPORT.md").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
