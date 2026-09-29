"""REST endpoints for GDPR training explanations."""
from __future__ import annotations

import logging
from typing import NoReturn
from uuid import uuid4

from fastapi import (
    APIRouter,
    HTTPException,
    status,
)

from app.core.config import (
    TRAINING_DEBUG_ENABLED,
)
from app.training.schemas import (
    TrainingExplanationRequest,
    TrainingExplanationResponse,
)
from app.training.service import (
    build_training_explanation,
)
from app.training.trace import (
    TrainingPipelineDebugResponse,
    TrainingTraceCollector,
)


logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/api/v1/training",
    tags=["training"],
)


def _raise_training_error(
    request: TrainingExplanationRequest,
) -> NoReturn:
    error_id = str(uuid4())

    logger.exception(
        "Training explanation failed: "
        "error_id=%s tenant_id=%s "
        "course_id=%s question_id=%s",
        error_id,
        request.tenant_id,
        request.course_id,
        request.question_id,
    )

    raise HTTPException(
        status_code=(
            status.HTTP_500_INTERNAL_SERVER_ERROR
        ),
        detail={
            "code": (
                "training_explanation_failed"
            ),
            "message": (
                "The training explanation "
                "could not be generated."
            ),
            "error_id": error_id,
        },
    ) from None


async def _execute_training_request(
    request: TrainingExplanationRequest,
    trace: TrainingTraceCollector | None = None,
) -> TrainingExplanationResponse:
    try:
        return await build_training_explanation(
            request,
            trace=trace,
        )

    except Exception:
        _raise_training_error(request)


@router.post(
    "/explanations",
    response_model=TrainingExplanationResponse,
    status_code=status.HTTP_200_OK,
)
async def create_training_explanation(
    request: TrainingExplanationRequest,
) -> TrainingExplanationResponse:
    return await _execute_training_request(
        request
    )


@router.post(
    "/explanations/debug",
    response_model=TrainingPipelineDebugResponse,
    status_code=status.HTTP_200_OK,
    include_in_schema=False,
)
async def debug_training_explanation(
    request: TrainingExplanationRequest,
) -> TrainingPipelineDebugResponse:
    if not TRAINING_DEBUG_ENABLED:
        # Return 404 instead of advertising that an internal endpoint exists.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Not Found",
        )

    trace = TrainingTraceCollector()

    result = await _execute_training_request(
        request,
        trace=trace,
    )

    return TrainingPipelineDebugResponse(
        result=result,
        trace=trace.snapshot(),
    )