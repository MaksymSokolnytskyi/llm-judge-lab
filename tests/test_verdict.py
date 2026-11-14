"""Verdict parsing: refusals and parse failures must never become a score of 0."""

from __future__ import annotations

import pytest

from judgelab.verdict import classify_legacy, is_refusal, parse_verdict


def test_plain_json_is_scored() -> None:
    v = parse_verdict('{"score": 8, "feedback": "good", "accepted": true}')
    assert (v.status, v.score, v.passed, v.judge_accepted) == ("scored", 8, True, True)


def test_fenced_json_with_prose_is_scored() -> None:
    raw = 'Here you go:\n```json\n{"score": 4, "feedback": "list used"}\n```\nThanks'
    v = parse_verdict(raw)
    assert (v.status, v.score, v.passed) == ("scored", 4, False)


def test_genuine_zero_stays_a_score() -> None:
    v = parse_verdict('{"score": 0, "feedback": "Auto-fail: contains Happy to help"}')
    assert (v.status, v.score) == ("scored", 0)


@pytest.mark.parametrize(
    "raw",
    [
        '{"score": 0, "feedback": "Я не можу оцінити цей діалог, бо він образливий"}',
        "I can't evaluate this text because it is offensive.",
        "Я відмовлюся оцінювати цей текст.",
        "Я не могу оценить этот текст.",
    ],
)
def test_refusals_are_not_scores(raw: str) -> None:
    v = parse_verdict(raw)
    assert v.status == "refused"
    assert v.score is None and v.passed is None


def test_truncated_output_is_unparsed() -> None:
    v = parse_verdict("Я оцінюю ПЕРЕПИСАНИЙ ТЕКСТ за шкалою 0–10:\n\n* R1 Не AI-болванка")
    assert (v.status, v.score) == ("unparsed", None)


def test_passed_follows_score_not_the_judge_flag() -> None:
    v = parse_verdict('{"score": 8, "feedback": "ok", "accepted": false}', pass_score=7)
    assert v.passed is True
    assert v.judge_accepted is False


def test_out_of_range_score_is_not_trusted() -> None:
    assert parse_verdict('{"score": 42, "feedback": "x"}').status == "unparsed"


def test_saying_you_can_evaluate_is_not_a_refusal() -> None:
    assert not is_refusal("Я можу оцінити цей текст за шкалою 0–10")


@pytest.mark.parametrize(
    ("score", "feedback", "expected"),
    [
        (7, "fine", "scored"),
        (0, "Автопровал: містить кліше", "scored"),
        (0, "Я не можу оцінити цей текст", "refused"),
        (0, "", "unparsed"),
        (0, "Я можу оцінити цей текст за шкалою 0-10.\n\nР1: Н", "unparsed"),
    ],
)
def test_classify_legacy(score: int, feedback: str, expected: str) -> None:
    assert classify_legacy(score, feedback) == expected
