"""Parsing judge output into an explicit verdict.

Single source of truth for turning raw judge text into a score.

The original pipeline mapped every output it could not read to ``score = 0``. That
silently merged three different events into one number:

* the judge **graded** the text and gave it 0 (a real verdict),
* the judge **refused** to grade (safety behaviour, e.g. on a deliberately toxic
  persona),
* the output was **unparsable** (truncated by ``max_tokens``, prose instead of JSON).

Refusals and parse failures are not quality signals, so averaging them as zeros biases
the judge's mean downward. :func:`parse_verdict` keeps them apart; analyses then
decide explicitly how to treat each status.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field

from judgelab.parsing import extract_json_object

VerdictStatus = Literal["scored", "refused", "unparsed"]

SCORE_MIN, SCORE_MAX = 0, 10

# Phrases a judge uses when it declines to grade. Covers the three languages that
# appeared in the runs (the judge prompt was Ukrainian; one model drifted into Russian).
_REFUSAL_RE = re.compile(
    r"\b(?:I\s+(?:can(?:no|')t|cannot|won't|will\s+not|refuse|am\s+unable|'m\s+unable)"
    r"|не\s+(?:можу|зможу|могу|буду)"
    r"|відмовл\w*"
    r"|отказ\w*"
    r"|заборонено\s+оцінювати)",
    re.IGNORECASE,
)


class Verdict(BaseModel):
    """A judge's decision about one text."""

    model_config = ConfigDict(frozen=True)

    status: VerdictStatus
    score: int | None = Field(
        default=None, ge=SCORE_MIN, le=SCORE_MAX, description="None unless scored."
    )
    feedback: str = ""
    judge_accepted: bool | None = Field(
        default=None,
        description="The judge's own pass/fail flag, kept only to measure its "
        "consistency with the score; never used as the decision.",
    )
    pass_score: int = Field(default=7, ge=SCORE_MIN, le=SCORE_MAX)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def passed(self) -> bool | None:
        """Decision derived from the score, so it cannot contradict it."""
        return None if self.score is None else self.score >= self.pass_score


def is_refusal(text: str) -> bool:
    """True if ``text`` reads as the judge declining to grade."""
    return bool(_REFUSAL_RE.search(text))


def _as_score(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float) and SCORE_MIN <= value <= SCORE_MAX:
        return round(value)
    if isinstance(value, str) and value.strip().isdigit():
        return _as_score(int(value.strip()))
    return None


def parse_verdict(raw: str, pass_score: int = 7) -> Verdict:
    """Classify raw judge output as ``scored``, ``refused`` or ``unparsed``.

    A JSON object with a valid score counts as scored, unless the score is 0 and the
    feedback is a refusal (models often wrap a refusal in the requested JSON).
    """
    obj = extract_json_object(raw)
    if obj is not None:
        feedback = str(obj.get("feedback", obj.get("reason", "")))
        accepted = obj.get("accepted")
        judge_accepted = accepted if isinstance(accepted, bool) else None
        score = _as_score(obj.get("score"))
        if score is not None and not (score == 0 and is_refusal(feedback)):
            return Verdict(
                status="scored",
                score=score,
                feedback=feedback,
                judge_accepted=judge_accepted,
                pass_score=pass_score,
            )
        if is_refusal(feedback):
            return Verdict(status="refused", feedback=feedback, pass_score=pass_score)
    if is_refusal(raw):
        return Verdict(
            status="refused", feedback=raw.strip()[:300], pass_score=pass_score
        )
    return Verdict(status="unparsed", feedback=raw.strip()[:300], pass_score=pass_score)


def classify_legacy(score: int, feedback: str) -> VerdictStatus:
    """Recover the status of a record written by the original pipeline.

    The old parser stored ``score=0`` for refusals and failures, and on failure put
    the first 80 raw characters into ``feedback``. A refusal is recognised by its
    wording; a failure by raw-output traces (empty text or a line break, which never
    appears in a parsed one-sentence feedback).
    """
    if score != 0:
        return "scored"
    if is_refusal(feedback):
        return "refused"
    if not feedback.strip() or "\n" in feedback:
        return "unparsed"
    return "scored"
