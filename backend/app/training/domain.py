"""Deterministic business rules for GDPR training questions."""
from __future__ import annotations

from typing import Literal, Sequence


LearnerResult = Literal[
    "correct",
    "incorrect",
    "partially_correct",
    "not_answered",
]


def determine_learner_result(
    correct_option_ids: Sequence[str],
    selected_option_ids: Sequence[str],
) -> LearnerResult:
    if not correct_option_ids:
        raise ValueError("correct_option_ids must not be empty")

    if not selected_option_ids:
        return "not_answered"

    correct_ids = set(correct_option_ids)
    selected_ids = set(selected_option_ids)

    if selected_ids == correct_ids:
        return "correct"

    if selected_ids.isdisjoint(correct_ids):
        return "incorrect"

    return "partially_correct"