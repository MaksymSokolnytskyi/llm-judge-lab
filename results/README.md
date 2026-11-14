# Results

Raw outputs of past runs, kept so every number in the top-level README can be
recomputed. `REPORT.md` files are generated; do not edit them by hand.

- `cross_judge_2026-03-02/` -- the cross-judge run (original Ukrainian prompts):
  `meta_prompts.jsonl` (18), `generations.jsonl` (270), `judgments.jsonl` (810).
  Written by the pre-refactor pipeline: `score` is always an integer and refusals or
  parse failures appear as 0. `judgelab.verdict.classify_legacy` recovers the status.
- `earlier_runs/` -- single-judge runs from the same day that motivated the
  cross-judge design.

Records quote requests and reference answers from the Bitext customer-support dataset
(CDLA-Sharing-1.0, (c) Bitext Innovations 2024); those parts are under that licence.
Judge feedback in these files is mostly Ukrainian, as the rubric was.
