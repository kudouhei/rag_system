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