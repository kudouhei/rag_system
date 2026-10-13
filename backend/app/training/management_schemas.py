"""HTTP request contracts for explanation governance."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.training.review_events import ReviewAction


class TrainingReviewActionRequest(BaseModel):
    """Client-supplied action, revision, and review comment."""

    model_config = ConfigDict(
        extra="forbid",
    )

    action: ReviewAction

    expected_revision: int = Field(
        ge=0,
        strict=True,
    )

    comment: str | None = Field(
        default=None,
        max_length=4000,
    )