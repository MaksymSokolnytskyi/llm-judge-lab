"""Generate labelled synthetic support dialogues with a local Ollama server.

Question: can small local models produce chat dialogues realistic enough to test a
dialogue analyzer, with known intent, satisfaction, agent mistakes and hidden
dissatisfaction?

    python experiments/generate_dialogues.py --n 10 --seed 42
    python experiments/generate_dialogues.py --stats

The route cache lives in ``runs/arena_registry.db`` and persists across runs, so the
first run is slow (micro-arenas for every new context) and later runs mostly reuse
cached routes. Contexts drawn for exploration are re-run between dialogues.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from datetime import datetime
from pathlib import Path

from judgelab.dialogues import run_pipeline
from judgelab.dialogues.routing import ArenaRegistry, DynamicRouter, MicroArena
from judgelab.dialogues.stylize import arena_samples, persona_hint
from judgelab.llm import DEFAULT_OLLAMA_URL, OllamaBackend


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=0.10)
    parser.add_argument("--db", type=Path, default=Path("runs/arena_registry.db"))
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args.db.parent.mkdir(parents=True, exist_ok=True)
    registry = ArenaRegistry(args.db)
    if args.stats:
        print(registry.stats())
        return

    backend = OllamaBackend(args.ollama_url)
    router = DynamicRouter(registry, epsilon=args.epsilon, rng=random.Random(args.seed))
    arena = MicroArena(backend, registry)

    dialogues, errors = [], []
    for seed in range(args.seed, args.seed + args.n):
        try:
            dialogue = run_pipeline(seed, backend, router, arena)
        except Exception as exc:  # keep the batch going; the error is recorded
            logging.exception("seed %d failed", seed)
            errors.append({"seed": seed, "error": repr(exc)})
            continue
        dialogues.append(dialogue)
        samples = arena_samples(dialogue.messages)
        for ctx in router.take_explorations():
            role = "client" if ctx.node_name == "stylization_client" else "agent"
            arena.run(ctx, samples[ctx.node_name], persona_hint(role, dialogue.params))
            router.mark_done(ctx)

    out = Path("runs") / f"dialogues_{datetime.now():%Y%m%d_%H%M%S}.json"
    payload = {
        "meta": {"n_requested": args.n, "n_ok": len(dialogues), "base_seed": args.seed},
        "dialogues": [d.model_dump() for d in dialogues],
        "errors": errors,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    passed = sum(d.verdict.passed for d in dialogues)
    print(
        f"{len(dialogues)}/{args.n} generated, {passed} passed the macro judge -> {out}"
    )
    print(registry.stats())


if __name__ == "__main__":
    main()
