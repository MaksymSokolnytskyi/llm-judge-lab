"""Seed-record selection from Bitext-shaped rows."""

from __future__ import annotations

from judgelab.data import record_id, select_records


def _row(category: str, instruction: str) -> dict[str, str]:
    return {
        "category": category,
        "intent": "i",
        "instruction": instruction,
        "response": "r",
    }


def test_round_robin_over_sorted_categories_skipping_near_duplicates() -> None:
    rows = [
        _row("REFUND", "refund please " + "x" * 40),
        _row("ACCOUNT", "open account"),
        _row("REFUND", "refund please " + "x" * 40 + " again"),  # same 40-char prefix
        _row("REFUND", "where is my money"),
        _row("ACCOUNT", "close account"),
    ]
    picked = select_records(rows, 4)
    assert [r.instruction for r in picked] == [
        "open account",
        rows[0]["instruction"],
        "close account",
        "where is my money",
    ]
    assert picked[0].id == record_id("open account")


def test_stops_when_categories_run_out() -> None:
    assert len(select_records([_row("A", "one"), _row("B", "two")], 10)) == 2
    assert select_records([], 3) == []
