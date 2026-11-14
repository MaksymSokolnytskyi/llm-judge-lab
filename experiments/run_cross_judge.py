"""Run the cross-judge experiment against a local Ollama server.

Question: how much of an LLM judge's score is the judge rather than the text?

    ollama pull llama3.1:8b && ollama pull qwen2.5:7b && ollama pull dolphin-mistral
    pip install -e ".[data]"
    python experiments/run_cross_judge.py --records 5 --language en

Outputs go to ``runs/cross_judge_<timestamp>/`` (git-ignored); analyse them with
``experiments/analyze_cross_judge.py``.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

from judgelab.cross_judge import CrossJudgeExperiment, ExperimentConfig
from judgelab.data import load_bitext
from judgelab.llm import DEFAULT_OLLAMA_URL, OllamaBackend


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--records", type=int, default=5)
    parser.add_argument("--language", choices=["en", "uk"], default="en")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    config = ExperimentConfig(
        n_records=args.records, language=args.language, workers=args.workers
    )
    if args.models:
        config = config.model_copy(update={"models": tuple(args.models)})
    out = args.out or Path("runs") / f"cross_judge_{datetime.now():%Y%m%d_%H%M%S}"

    records = load_bitext(config.n_records)
    experiment = CrossJudgeExperiment(OllamaBackend(args.ollama_url), config)
    result = experiment.run(records, out_dir=out)

    expected = len(config.models) ** 2 * len(config.personas) * len(config.approaches)
    expected *= len(records)
    assert len(result.generations) == expected, (len(result.generations), expected)
    n_gen, n_jdg = len(result.generations), len(result.judgments)
    print(f"{n_gen} generations, {n_jdg} verdicts -> {out}")


if __name__ == "__main__":
    main()
