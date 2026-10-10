"""Persistence port for training explanation artifacts."""
from __future__ import annotations

from typing import Protocol
from uuid import UUID

from app.training.artifacts import (
    TrainingExplanationArtifact,
)
from app.training.lifecycle import (
    ExplanationStatus,
)
from app.training.review_events import (
    ExplanationReviewEvent,
)


class ExplanationArtifactAlreadyExists(
    ValueError
):
    """An artifact with the same content identity exists."""


class ExplanationArtifactNotFound(
    LookupError
):
    """The requested artifact does not exist for the tenant."""


class ExplanationArtifactStatusConflict(
    RuntimeError
):
    """The persisted status or revision differs from expectation."""


class ExplanationReviewEventAlreadyExists(
    ValueError
):
    """A review event with this event_id has already been stored."""   

class ExplanationPublicationConflict(
    RuntimeError
):
    """Another explanation is published for this question version."""
    
class ExplanationArtifactRepository(Protocol):
    """Storage operations required by the training domain."""

    async def add(
        self,
        artifact: TrainingExplanationArtifact,
    ) -> None:
        """Persist one new immutable explanation artifact."""
        ...

    async def get_by_id(
        self,
        *,
        tenant_id: str,
        artifact_id: UUID,
    ) -> TrainingExplanationArtifact | None:
        """Return one artifact within the tenant boundary."""
        ...

    async def find_by_version(
        self,
        *,
        tenant_id: str,
        course_id: str,
        question_id: str,
        explanation_version: str,
    ) -> TrainingExplanationArtifact | None:
        """Find an existing content version for idempotency."""
        ...

    async def list_by_status(
        self,
        *,
        tenant_id: str,
        status: ExplanationStatus,
        course_id: str | None = None,
        limit: int = 100,
    ) -> list[TrainingExplanationArtifact]:
        """List artifacts visible in a review queue."""
        ...

    async def get_published(
        self,
        *,
        tenant_id: str,
        course_id: str,
        question_id: str,
        question_fingerprint: str,
    ) -> TrainingExplanationArtifact | None:
        """Return the published version for the exact question."""
        ...

    async def apply_review_event(
        self,
        *,
        tenant_id: str,
        expected_revision: int,
        event: ExplanationReviewEvent,
    ) -> TrainingExplanationArtifact:
        """Atomically update lifecycle state and append an event.

        Require event.tenant_id to match tenant_id.
        Find the artifact within that tenant.
        Match its status against event.from_status and its
        revision against expected_revision.
        Apply event.to_status, increment revision, and append
        the event in one atomic operation.

        Leave both artifact and event history unchanged if
        any check or write fails.
        """
        ...

    async def list_review_events(
        self,
        *,
        tenant_id: str,
        artifact_id: UUID,
        limit: int = 100,
    ) -> list[ExplanationReviewEvent]:
        """Return tenant-scoped audit events, newest first."""
        ...