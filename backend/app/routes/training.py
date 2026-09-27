"""REST endpoints for GDPR training explanations."""
from __future__ import annotations

import logging
from uuid import uuid4

from fastapi import APIRouter, HTTPException, status

from app.training.schemas import (
    TrainingExplanationRequest,
    TrainingExplanationResponse,
)
from app.training.service import build_training_explanation


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/training",
    tags=["training"],
)


@router.post(
    "/explanations",
    response_model=TrainingExplanationResponse,
    status_code=status.HTTP_200_OK,
)
async def create_training_explanation(
    request: TrainingExplanationRequest,
) -> TrainingExplanationResponse:
    try:
        return await build_training_explanation(request)

    except Exception:
        error_id = str(uuid4())

        logger.exception(
            "Training explanation failed: "
            "error_id=%s tenant_id=%s course_id=%s question_id=%s",
            error_id,
            request.tenant_id,
            request.course_id,
            request.question_id,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "training_explanation_failed",
                "message": (
                    "The training explanation could not be generated."
                ),
                "error_id": error_id,
            },
        ) from None