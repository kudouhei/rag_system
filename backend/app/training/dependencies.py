"""FastAPI dependencies for training application services."""
from __future__ import annotations

from typing import Annotated, cast

from fastapi import Depends, Request

from app.training.authoring_service import (
    ExplanationAuthoringService,
)
from app.training.repository import (
    ExplanationArtifactRepository,
)
from app.training.review_service import (
    ExplanationReviewService,
)


def get_training_repository(
    request: Request,
) -> ExplanationArtifactRepository:
    repository = getattr(
        request.app.state,
        "training_repository",
        None,
    )

    if repository is None:
        raise RuntimeError(
            "Training repository was not initialized "
            "during application startup."
        )

    return cast(
        ExplanationArtifactRepository,
        repository,
    )


def get_authoring_service(
    repository: Annotated[
        ExplanationArtifactRepository,
        Depends(get_training_repository),
    ],
) -> ExplanationAuthoringService:
    return ExplanationAuthoringService(repository)


def get_review_service(
    repository: Annotated[
        ExplanationArtifactRepository,
        Depends(get_training_repository),
    ],
) -> ExplanationReviewService:
    return ExplanationReviewService(repository)