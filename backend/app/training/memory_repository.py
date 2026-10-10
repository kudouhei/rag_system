"""In-memory storage for local training-artifact workflows."""
from __future__ import annotations

from asyncio import Lock
from uuid import UUID

from app.training.artifacts import (
    TrainingExplanationArtifact,
)
from app.training.lifecycle import (
    ExplanationStatus,
)
from app.training.repository import (
    ExplanationArtifactAlreadyExists,
    ExplanationArtifactNotFound,
    ExplanationArtifactStatusConflict,
    ExplanationPublicationConflict,
    ExplanationReviewEventAlreadyExists,
)
from app.training.review_events import (
    ExplanationReviewEvent,
)

class InMemoryExplanationArtifactRepository:
    """Store isolated artifact snapshots in one process."""

    def __init__(self) -> None:
        self._artifacts: dict[
            UUID,
            TrainingExplanationArtifact,
        ] = {}

        self._version_index: dict[
            tuple[str, str, str, str],
            UUID,
        ] = {}
        self._review_events: dict[UUID, list[ExplanationReviewEvent]] = {}

        self._lock = Lock()

    @staticmethod
    def _version_key(
        artifact: TrainingExplanationArtifact,
    ) -> tuple[str, str, str, str]:
        return (
            artifact.tenant_id,
            artifact.course_id,
            artifact.question_id,
            artifact.explanation_version,
        )

    async def add(
        self,
        artifact: TrainingExplanationArtifact,
    ) -> None:
        if (
            artifact.status
            != ExplanationStatus.REVIEW_REQUIRED
            or artifact.revision != 0
        ):
            raise ValueError(
                "New artifacts must start in "
                "'review_required' with revision 0."
            )

        snapshot = artifact.model_copy(deep=True)
        version_key = self._version_key(snapshot)

        async with self._lock:
            if snapshot.artifact_id in self._artifacts:
                raise ExplanationArtifactAlreadyExists(
                    "Artifact ID already exists."
                )

            if version_key in self._version_index:
                raise ExplanationArtifactAlreadyExists(
                    "Explanation version already exists "
                    "for this tenant, course, and question."
                )

            self._artifacts[
                snapshot.artifact_id
            ] = snapshot

            self._version_index[
                version_key
            ] = snapshot.artifact_id

    async def get_by_id(
        self,
        *,
        tenant_id: str,
        artifact_id: UUID,
    ) -> TrainingExplanationArtifact | None:
        async with self._lock:
            artifact = self._artifacts.get(
                artifact_id
            )

            if (
                artifact is None
                or artifact.tenant_id != tenant_id
            ):
                return None

            return artifact.model_copy(deep=True)

    async def find_by_version(
        self,
        *,
        tenant_id: str,
        course_id: str,
        question_id: str,
        explanation_version: str,
    ) -> TrainingExplanationArtifact | None:
        version_key = (
            tenant_id,
            course_id,
            question_id,
            explanation_version,
        )

        async with self._lock:
            artifact_id = self._version_index.get(
                version_key
            )

            if artifact_id is None:
                return None

            artifact = self._artifacts[artifact_id]

            return artifact.model_copy(deep=True)

    async def apply_review_event(
        self,
        *,
        tenant_id: str,
        expected_revision: int,
        event: ExplanationReviewEvent,
    ) -> TrainingExplanationArtifact:
        # Revalidate at the storage boundary.
        event_snapshot = (
            ExplanationReviewEvent.model_validate(
                event.model_dump()
            )
        )

        if event_snapshot.tenant_id != tenant_id:
            raise ValueError(
                "Review event tenant does not match "
                "the operation tenant."
            )

        async with self._lock:
            artifact = self._artifacts.get(
                event_snapshot.artifact_id
            )

            if (
                artifact is None
                or artifact.tenant_id != tenant_id
            ):
                raise ExplanationArtifactNotFound(
                    "Artifact was not found in this tenant."
                )

            if (
                event_snapshot.event_id
                in self._review_events
            ):
                raise ExplanationReviewEventAlreadyExists(
                    "Review event ID already exists."
                )

            if (
                artifact.status
                != event_snapshot.from_status
                or artifact.revision != expected_revision
            ):
                raise ExplanationArtifactStatusConflict(
                    "Artifact status or revision has changed. "
                    "Reload the artifact before reviewing."
                )

            if (
                event_snapshot.to_status
                == ExplanationStatus.PUBLISHED
            ):
                another_published = any(
                    existing.status
                    == ExplanationStatus.PUBLISHED
                    and existing.tenant_id
                    == artifact.tenant_id
                    and existing.course_id
                    == artifact.course_id
                    and existing.question_id
                    == artifact.question_id
                    and existing.question_fingerprint
                    == artifact.question_fingerprint
                    for existing in self._artifacts.values()
                )

                if another_published:
                    raise ExplanationPublicationConflict(
                        "An explanation is already published "
                        "for this question version."
                    )

            updated_payload = artifact.model_dump()
            updated_payload["status"] = (
                event_snapshot.to_status
            )
            updated_payload["revision"] = (
                artifact.revision + 1
            )

            updated_artifact = (
                TrainingExplanationArtifact.model_validate(
                    updated_payload
                )
            )

            result = updated_artifact.model_copy(
                deep=True
            )

            # Prepare both changes before replacing stored state.
            next_artifacts = self._artifacts.copy()
            next_events = self._review_events.copy()

            next_artifacts[
                updated_artifact.artifact_id
            ] = updated_artifact

            next_events[
                event_snapshot.event_id
            ] = event_snapshot

            self._artifacts = next_artifacts
            self._review_events = next_events

            return result

    async def list_review_events(
        self,
        *,
        tenant_id: str,
        artifact_id: UUID,
        limit: int = 100,
    ) -> list[ExplanationReviewEvent]:
        if not 1 <= limit <= 100:
            raise ValueError(
                "limit must be between 1 and 100."
            )

        async with self._lock:
            artifact = self._artifacts.get(
                artifact_id
            )

            if (
                artifact is None
                or artifact.tenant_id != tenant_id
            ):
                return []

            events = [
                event
                for event in self._review_events.values()
                if event.tenant_id == tenant_id
                and event.artifact_id == artifact_id
            ]

            events.sort(
                key=lambda event: (
                    event.created_at,
                    str(event.event_id),
                ),
                reverse=True,
            )

            return [
                event.model_copy(deep=True)
                for event in events[:limit]
            ]