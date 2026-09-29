"""Request-scoped trace models for the training pipeline."""
from __future__ import annotations

from time import perf_counter
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.training.schemas import TrainingExplanationResponse


TrainingStageName = Literal[
    "request",
    "access_control",
    "query_planning",
    "retrieval",
    "evidence_selection",
    "generation",
    "grounding",
    "final_response",
]

TrainingStageStatus = Literal[
    "complete",
    "skipped",
    "not_evaluated",
    "failed",
]


class TrainingPipelineStage(BaseModel):
    stage: TrainingStageName
    status: TrainingStageStatus

    duration_ms: float = Field(
        ge=0.0,
    )

    summary: str = Field(
        min_length=1,
        max_length=1000,
    )

    details: dict[str, Any] = Field(
        default_factory=dict,
    )


class TrainingPipelineDebugResponse(BaseModel):
    result: TrainingExplanationResponse

    trace: list[TrainingPipelineStage] = Field(
        min_length=1,
    )


class TrainingTraceCollector:
    """Collect trace stages for one request only."""

    def __init__(self) -> None:
        self._stages: list[TrainingPipelineStage] = []

    def record(
        self,
        *,
        stage: TrainingStageName,
        status: TrainingStageStatus,
        started_at: float,
        summary: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        duration_ms = round(
            (perf_counter() - started_at) * 1000,
            2,
        )

        self._stages.append(
            TrainingPipelineStage(
                stage=stage,
                status=status,
                duration_ms=duration_ms,
                summary=summary,
                details=details or {},
            )
        )

    def snapshot(self) -> list[TrainingPipelineStage]:
        """Return a copy so callers cannot mutate internal state."""

        return list(self._stages)