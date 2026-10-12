"""Application service for saving explanation candidates."""
from __future__ import annotations

from app.training.artifacts import (
    TrainingExplanationArtifact,
)
from app.training.repository import (
    ExplanationArtifactAlreadyExists,
    ExplanationArtifactRepository,
)


class ExplanationCandidateService:
    """Save new content or reuse its existing artifact."""

    def __init__(
        self,
        repository: ExplanationArtifactRepository,
    ) -> None:
        self._repository = repository

    async def save_candidate(
        self,
        artifact: TrainingExplanationArtifact,
    ) -> TrainingExplanationArtifact:
        snapshot = (
            TrainingExplanationArtifact.model_validate(
                artifact.model_dump()
            )
        )

        try:
            await self._repository.add(snapshot)

        except ExplanationArtifactAlreadyExists:
            existing = (
                await self._repository.find_by_version(
                    tenant_id=snapshot.tenant_id,
                    course_id=snapshot.course_id,
                    question_id=snapshot.question_id,
                    explanation_version=(
                        snapshot.explanation_version
                    ),
                )
            )

            if existing is None:
                # An ID collision is not a content-version match.
                raise

            return existing

        return snapshot.model_copy(deep=True)