"""Lifecycle rules for reviewed training explanations."""
from __future__ import annotations

from enum import StrEnum


class ExplanationStatus(StrEnum):
    """Lifecycle states of one immutable explanation version."""

    REVIEW_REQUIRED = "review_required"
    APPROVED = "approved"
    PUBLISHED = "published"
    REJECTED = "rejected"
    RETIRED = "retired"


class InvalidExplanationStatusTransition(
    ValueError
):
    """The requested lifecycle transition is not allowed."""


_ALLOWED_TRANSITIONS: dict[
    ExplanationStatus,
    frozenset[ExplanationStatus],
] = {
    ExplanationStatus.REVIEW_REQUIRED: frozenset(
        {
            ExplanationStatus.APPROVED,
            ExplanationStatus.REJECTED,
        }
    ),
    ExplanationStatus.APPROVED: frozenset(
        {
            ExplanationStatus.PUBLISHED,
            ExplanationStatus.REVIEW_REQUIRED,
        }
    ),
    ExplanationStatus.PUBLISHED: frozenset(
        {
            ExplanationStatus.RETIRED,
        }
    ),
    ExplanationStatus.REJECTED: frozenset(),
    ExplanationStatus.RETIRED: frozenset(),
}


def allowed_next_statuses(
    current: ExplanationStatus,
) -> frozenset[ExplanationStatus]:
    """Return valid next states for the current state."""

    return _ALLOWED_TRANSITIONS[current]


def validate_status_transition(
    *,
    current: ExplanationStatus,
    target: ExplanationStatus,
) -> None:
    """Reject lifecycle changes that bypass governance."""

    if target in allowed_next_statuses(current):
        return

    raise InvalidExplanationStatusTransition(
        "Cannot transition explanation status "
        f"from {current.value!r} to {target.value!r}."
    )