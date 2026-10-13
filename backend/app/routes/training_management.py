"""Management endpoints for training explanation content."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from app.training.identity import (
    TrainingActor,
    get_training_actor,
)


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