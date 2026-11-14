"""Seed records from the Bitext customer-support dataset.

The dataset (``bitext/Bitext-customer-support-llm-chatbot-training-dataset``, licence
CDLA-Sharing-1.0, (c) Bitext Innovations 2024) is downloaded on demand through the
optional ``datasets`` dependency and is not stored in this repository.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping

from pydantic import BaseModel, ConfigDict

BITEXT_DATASET = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"


class Record(BaseModel):
    """One customer request with the dataset's reference answer."""

    model_config = ConfigDict(frozen=True)

    id: str
    intent: str
    category: str
    instruction: str
    original_response: str


def record_id(instruction: str) -> str:
    """Stable 8-hex id derived from the request text (matches the original runs)."""
    return hashlib.md5(instruction.encode()).hexdigest()[:8]


def select_records(rows: Iterable[Mapping[str, str]], n: int) -> list[Record]:
    """Pick ``n`` records round-robin across categories (sorted by name).

    Within a category the first row whose instruction prefix (40 chars) has not been
    seen is taken, so near-duplicate phrasings are skipped. Deterministic for a fixed
    dataset order.
    """
    by_category: dict[str, list[Mapping[str, str]]] = {}
    for row in rows:
        by_category.setdefault(row.get("category", "OTHER"), []).append(row)
    categories = sorted(by_category)
    if not categories:
        return []

    records: list[Record] = []
    seen: set[str] = set()
    cursors = dict.fromkeys(categories, 0)
    exhausted: set[str] = set()
    turn = 0
    while len(records) < n and len(exhausted) < len(categories):
        category = categories[turn % len(categories)]
        turn += 1
        rows_in_category = by_category[category]
        while cursors[category] < len(rows_in_category):
            row = rows_in_category[cursors[category]]
            cursors[category] += 1
            key = row["instruction"][:40]
            if key in seen:
                continue
            seen.add(key)
            records.append(
                Record(
                    id=record_id(row["instruction"]),
                    intent=row.get("intent", ""),
                    category=row.get("category", ""),
                    instruction=row["instruction"],
                    original_response=row["response"],
                )
            )
            break
        else:
            exhausted.add(category)
    return records


def load_bitext(n: int) -> list[Record]:
    """Download the Bitext train split and select ``n`` records."""
    try:
        from datasets import load_dataset  # pyright: ignore[reportMissingImports]
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError("Install the data extra: pip install -e '.[data]'") from exc
    dataset = load_dataset(BITEXT_DATASET, split="train")
    return select_records(dataset, n)
