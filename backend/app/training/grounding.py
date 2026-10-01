"""Claim-level grounding contracts and deterministic scoring."""
from __future__ import annotations

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from app.training.generation import (
    GeneratedTrainingContent,
)


class InvalidGroundingOutput(ValueError):
    """The grounding judge output violated its contract."""


class ClaimGroundingVerdict(BaseModel):
    """One semantic-support decision for one generated claim."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    option_id: str = Field(
        min_length=1,
        max_length=100,
    )

    claim_index: int = Field(
        ge=0,
        le=7,
    )

    supported: bool

    reason: str = Field(
        min_length=1,
        max_length=1000,
    )


class GroundingJudgeOutput(BaseModel):
    """Structured output expected from the grounding judge."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    verdicts: list[
        ClaimGroundingVerdict
    ] = Field(
        min_length=1,
        max_length=80,
    )


def expected_claim_keys(
    generated: GeneratedTrainingContent,
) -> set[tuple[str, int]]:
    """Return the exact claim identities that must be judged."""

    return {
        (
            option.option_id,
            claim_index,
        )
        for option in generated.option_explanations
        for claim_index, _claim in enumerate(
            option.claims
        )
    }


def validate_grounding_output(
    generated: GeneratedTrainingContent,
    judged: GroundingJudgeOutput,
) -> None:
    """Require exactly one verdict for every generated claim."""

    actual_keys = [
        (
            verdict.option_id,
            verdict.claim_index,
        )
        for verdict in judged.verdicts
    ]

    if len(actual_keys) != len(set(actual_keys)):
        raise InvalidGroundingOutput(
            "Grounding output contains duplicate claim verdicts"
        )

    expected_keys = expected_claim_keys(
        generated
    )

    actual_key_set = set(actual_keys)

    if actual_key_set != expected_keys:
        missing_keys = sorted(
            expected_keys - actual_key_set
        )

        unexpected_keys = sorted(
            actual_key_set - expected_keys
        )

        raise InvalidGroundingOutput(
            "Grounding verdicts do not match generated claims: "
            f"missing={missing_keys}, "
            f"unexpected={unexpected_keys}"
        )


def calculate_grounding_score(
    generated: GeneratedTrainingContent,
    judged: GroundingJudgeOutput,
) -> float:
    """Return the proportion of claims judged as supported."""

    validate_grounding_output(
        generated=generated,
        judged=judged,
    )

    supported_count = sum(
        verdict.supported
        for verdict in judged.verdicts
    )

    return round(
        supported_count / len(judged.verdicts),
        4,
    )