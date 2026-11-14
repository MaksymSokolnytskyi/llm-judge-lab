"""Extract JSON from free-form model output (fences, prose around the object)."""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(r"```(?:json)?", re.IGNORECASE)


def _first_json(raw: str, opener: str) -> Any | None:
    cleaned = _FENCE_RE.sub("", raw).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    start = cleaned.find(opener)
    while start != -1:
        try:
            value, _ = decoder.raw_decode(cleaned, start)
            return value
        except json.JSONDecodeError:
            start = cleaned.find(opener, start + 1)
    return None


def extract_json_object(raw: str) -> dict[str, Any] | None:
    """First JSON object in ``raw``, or None."""
    value = _first_json(raw, "{")
    return value if isinstance(value, dict) else None


def extract_json_array(raw: str) -> list[Any] | None:
    """First JSON array in ``raw``, or None."""
    value = _first_json(raw, "[")
    return value if isinstance(value, list) else None
