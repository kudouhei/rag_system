"""Generate and save shared explanation candidates."""
from __future__ import annotations

from typing import TYPE_CHECKING

from app.training.artifact_factory import (
    build_review_artifact,
)
from app.training.artifacts import (
    TrainingExplanationArtifact,
)
from app.training.candidate_service import (
    ExplanationCandidateService,
)
from app.training.repository import (
    ExplanationArtifactRepository,
)
from app.training.schemas import (
    TrainingExplanationRequest,
)
from app.training.service import (
    build_training_explanation_result,
)

if TYPE_CHECKING:
    from app.training.trace import (
        TrainingTraceCollector,
    )


class InsufficientExplanationEvidence(ValueError):
    """The generated result lacks evidence for a review artifact."""


class ExplanationAuthoringService:
    """Run the pipeline and persist learner-independent content."""

    def __init__(
        self,
        repository: ExplanationArtifactRepository,
    ) -> None:
        self._candidate_service = (
            ExplanationCandidateService(repository)
        )

    async def create_candidate(
        self,
        *,
        request: TrainingExplanationRequest,
        created_by: str,
        trace: TrainingTraceCollector | None = None,
    ) -> TrainingExplanationArtifact:
        authoring_request = request.model_copy(
            update={"selected_option_ids": []},
            deep=True,
        )

        result = await build_training_explanation_result(
            authoring_request,
            trace=trace,
        )

        if result.response.status == "insufficient_evidence":
            raise InsufficientExplanationEvidence(
                "Cannot create a review artifact because "
                "some answer options lack regulatory evidence."
            )

        artifact = build_review_artifact(
            request=authoring_request,
            response=result.response,
            grounding_model=result.grounding_model,
            created_by=created_by,
        )

        return await self._candidate_service.save_candidate(
            artifact
        )