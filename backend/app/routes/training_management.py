"""Management endpoints for training explanation content."""
from __future__ import annotations

import logging
from typing import Annotated
from uuid import UUID, uuid4
from fastapi import APIRouter, Depends, HTTPException, Query

from app.training.artifacts import (
    TrainingExplanationArtifact,
)
from app.training.authoring_service import (
    ExplanationAuthoringService,
    InsufficientExplanationEvidence,
)
from app.training.dependencies import (
    get_authoring_service,
    get_training_repository,
)
from app.training.identity import (
    TrainingActor,
    get_training_actor,
)
from app.training.schemas import (
    TrainingExplanationRequest,
)
from app.training.lifecycle import (
    ExplanationStatus,
)
from app.training.repository import (
    ExplanationArtifactRepository,
)


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/training/management",
    tags=["training-management"],
)


@router.get(
    "/me",
    response_model=TrainingActor,
)
async def get_management_identity(
    actor: Annotated[
        TrainingActor,
        Depends(get_training_actor),
    ],
) -> TrainingActor:
    return actor


@router.post(
    "/candidates",
    response_model=TrainingExplanationArtifact,
    status_code=200,
)
async def create_explanation_candidate(
    request: TrainingExplanationRequest,
    actor: Annotated[
        TrainingActor,
        Depends(get_training_actor),
    ],
    service: Annotated[
        ExplanationAuthoringService,
        Depends(get_authoring_service),
    ],
) -> TrainingExplanationArtifact:
    if actor.actor_role not in {
        "reviewer",
        "administrator",
    }:
        raise HTTPException(
            status_code=403,
            detail="This role cannot create explanation candidates.",
        )

    if request.tenant_id != actor.tenant_id:
        raise HTTPException(
            status_code=403,
            detail="Request tenant does not match authenticated tenant.",
        )

    try:
        return await service.create_candidate(
            request=request,
            created_by=actor.actor_id,
        )

    except InsufficientExplanationEvidence as error:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "insufficient_explanation_evidence",
                "message": str(error),
            },
        ) from None

    except Exception:
        error_id = str(uuid4())

        logger.exception(
            "Candidate creation failed: "
            "error_id=%s tenant_id=%s question_id=%s",
            error_id,
            actor.tenant_id,
            request.question_id,
        )

        raise HTTPException(
            status_code=500,
            detail={
                "code": "candidate_creation_failed",
                "message": "The explanation candidate could not be created.",
                "error_id": error_id,
            },
        ) from None

@router.get(
    "/candidates",
    response_model=list[TrainingExplanationArtifact],
)
async def list_explanation_candidates(
    actor: Annotated[
        TrainingActor,
        Depends(get_training_actor),
    ],
    repository: Annotated[
        ExplanationArtifactRepository,
        Depends(get_training_repository),
    ],
    candidate_status: Annotated[
        ExplanationStatus,
        Query(alias="status"),
    ] = ExplanationStatus.REVIEW_REQUIRED,
    course_id: Annotated[
        str | None,
        Query(min_length=1, max_length=100),
    ] = None,
    limit: Annotated[
        int,
        Query(ge=1, le=100),
    ] = 100,
) -> list[TrainingExplanationArtifact]:
    return await repository.list_by_status(
        tenant_id=actor.tenant_id,
        status=candidate_status,
        course_id=course_id,
        limit=limit,
    )


@router.get(
    "/candidates/{artifact_id}",
    response_model=TrainingExplanationArtifact,
)
async def get_explanation_candidate(
    artifact_id: UUID,
    actor: Annotated[
        TrainingActor,
        Depends(get_training_actor),
    ],
    repository: Annotated[
        ExplanationArtifactRepository,
        Depends(get_training_repository),
    ],
) -> TrainingExplanationArtifact:
    artifact = await repository.get_by_id(
        tenant_id=actor.tenant_id,
        artifact_id=artifact_id,
    )

    if artifact is None:
        raise HTTPException(
            status_code=404,
            detail="Explanation artifact was not found.",
        )

    return artifact