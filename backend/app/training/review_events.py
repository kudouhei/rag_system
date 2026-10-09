"""Append-only audit events for explanation governance."""
from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal, Self
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from app.training.lifecycle import (
    ExplanationStatus,
    validate_status_transition,
)


class ReviewAction(StrEnum):
    """Human actions that change artifact lifecycle state."""

    APPROVE = "approve"
    REJECT = "reject"
    RETURN_TO_REVIEW = "return_to_review"
    PUBLISH = "publish"
    RETIRE = "retire"


_ACTION_TRANSITIONS: dict[
    ReviewAction,
    tuple[
        ExplanationStatus,
        ExplanationStatus,
    ],
] = {
    ReviewAction.APPROVE: (
        ExplanationStatus.REVIEW_REQUIRED,
        ExplanationStatus.APPROVED,
    ),
    ReviewAction.REJECT: (
        ExplanationStatus.REVIEW_REQUIRED,
        ExplanationStatus.REJECTED,
    ),
    ReviewAction.RETURN_TO_REVIEW: (
        ExplanationStatus.APPROVED,
        ExplanationStatus.REVIEW_REQUIRED,
    ),
    ReviewAction.PUBLISH: (
        ExplanationStatus.APPROVED,
        ExplanationStatus.PUBLISHED,
    ),
    ReviewAction.RETIRE: (
        ExplanationStatus.PUBLISHED,
        ExplanationStatus.RETIRED,
    ),
}


class ExplanationReviewEvent(BaseModel):
    """One immutable human governance decision."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    event_id: UUID
    artifact_id: UUID
    tenant_id: str = Field(
        min_length=1,
        max_length=100,
    )

    action: ReviewAction
    from_status: ExplanationStatus
    to_status: ExplanationStatus

    actor_id: str = Field(
        min_length=1,
        max_length=200,
    )
    actor_role: Literal[
        "reviewer",
        "publisher",
        "administrator",
    ]

    comment: str | None = Field(
        default=None,
        max_length=4000,
    )
    created_at: datetime

    @model_validator(mode="after")
    def validate_event(
        self,
    ) -> Self:
        validate_status_transition(
            current=self.from_status,
            target=self.to_status,
        )

        expected_transition = (
            _ACTION_TRANSITIONS[self.action]
        )

        actual_transition = (
            self.from_status,
            self.to_status,
        )

        if actual_transition != expected_transition:
            raise ValueError(
                f"action {self.action.value!r} requires "
                f"{expected_transition[0].value!r} -> "
                f"{expected_transition[1].value!r}"
            )

        comment_required_actions = {
            ReviewAction.REJECT,
            ReviewAction.RETURN_TO_REVIEW,
            ReviewAction.RETIRE,
        }

        if (
            self.action in comment_required_actions
            and not (
                self.comment
                and self.comment.strip()
            )
        ):
            raise ValueError(
                f"action {self.action.value!r} "
                "requires a comment"
            )

        if (
            self.created_at.tzinfo is None
            or self.created_at.utcoffset() is None
        ):
            raise ValueError(
                "created_at must be timezone-aware"
            )

        return self