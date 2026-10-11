"""Application service for explanation review and publication."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import UUID, uuid4

from app.training.artifacts import (
    TrainingExplanationArtifact,
)
from app.training.repository import (
    ExplanationArtifactNotFound,
    ExplanationArtifactRepository,
    ExplanationArtifactStatusConflict,
)
from app.training.review_events import (
    ExplanationReviewEvent,
    ReviewAction,
    target_status_for_action,
)


class ExplanationReviewPermissionDenied(PermissionError):
    """The actor's role does not allow the requested action."""


_ALLOWED_ACTIONS: dict[
    str,
    frozenset[ReviewAction],
] = {
    "reviewer": frozenset(
        {
            ReviewAction.APPROVE,
            ReviewAction.REJECT,
            ReviewAction.RETURN_TO_REVIEW,
        }
    ),
    "publisher": frozenset(
        {
            ReviewAction.PUBLISH,
            ReviewAction.RETIRE,
        }
    ),
    "administrator": frozenset(ReviewAction),
}


class ExplanationReviewService:
    """Apply governance actions using trusted actor identity."""

    def __init__(
        self,
        repository: ExplanationArtifactRepository,
    ) -> None:
        self._repository = repository

    async def execute_action(
        self,
        *,
        tenant_id: str,
        artifact_id: UUID,
        expected_revision: int,
        action: ReviewAction,
        actor_id: str,
        actor_role: Literal[
            "reviewer",
            "publisher",
            "administrator",
        ],
        comment: str | None = None,
    ) -> TrainingExplanationArtifact:
        action = ReviewAction(action)

        allowed_actions = _ALLOWED_ACTIONS.get(
            actor_role,
            frozenset(),
        )

        if action not in allowed_actions:
            raise ExplanationReviewPermissionDenied(
                f"Role {actor_role!r} cannot perform "
                f"action {action.value!r}."
            )

        artifact = await self._repository.get_by_id(
            tenant_id=tenant_id,
            artifact_id=artifact_id,
        )

        if artifact is None:
            raise ExplanationArtifactNotFound(
                "Artifact was not found for this tenant."
            )

        if artifact.revision != expected_revision:
            raise ExplanationArtifactStatusConflict(
                "Artifact revision has changed. "
                "Reload the artifact before reviewing."
            )

        event = ExplanationReviewEvent(
            event_id=uuid4(),
            artifact_id=artifact.artifact_id,
            tenant_id=tenant_id,
            action=action,
            from_status=artifact.status,
            to_status=target_status_for_action(action),
            actor_id=actor_id,
            actor_role=actor_role,
            comment=comment,
            created_at=datetime.now(timezone.utc),
        )

        return await self._repository.apply_review_event(
            tenant_id=tenant_id,
            expected_revision=expected_revision,
            event=event,
        )