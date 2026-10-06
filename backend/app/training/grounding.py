"""Claim-level grounding contracts and deterministic scoring."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
)

from app.llm.client import ( get_active_llm_model, llm_structured_call)

from app.training.generation import (
    GeneratedTrainingContent,
)

GROUNDING_PROMPT_VERSION = (
    "grounding-entailment-v1"
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

GroundingEvaluationStatus = Literal[
    "evaluated",
    "disabled",
    "empty_response",
    "invalid_output",
]

@dataclass(frozen=True, slots=True)
class GroundingEvaluationAttempt:
    """Result of attempting claim-level grounding evaluation."""

    status: GroundingEvaluationStatus
    score: float | None
    output: GroundingJudgeOutput | None
    model: str | None
    reason: str


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

def build_grounding_messages(
    generated: GeneratedTrainingContent,
) -> list[dict[str, str]]:
    """Build a strict claim-versus-quote evaluation request."""

    claim_payload = [
        {
            "option_id": option.option_id,
            "claim_index": claim_index,
            "claim": claim.claim,
            "evidence_id": claim.evidence_id,
            "supporting_quote": claim.supporting_quote,
        }
        for option in generated.option_explanations
        for claim_index, claim in enumerate(
            option.claims
        )
    ]

    system_prompt = (
        "You are a strict claim-evidence evaluator for GDPR "
        "training content. Evaluate semantic entailment, not "
        "verbatim word overlap. For each supplied item, decide "
        "whether the supporting_quote supports the claim. "
        "Mark supported=true when the claim is: "
        "(1) explicitly stated by the quote; "
        "(2) a faithful paraphrase of the quote; or "
        "(3) a straightforward application or contradiction "
        "derived from the quote using ordinary language and logic. "
        "For example, a quote saying that access is limited to "
        "authorised staff supports the claim that allowing every "
        "employee access conflicts with that rule. "
        "Do not require the quote to contain the claim's exact words; "
        "quote provenance has already been verified separately. "
        "Mark supported=false when the claim adds a legal duty, "
        "exception, actor, threshold, deadline, sanction, causal "
        "relationship, or other fact that is not present in the "
        "quote and requires external knowledge. Shared terminology "
        "alone is not sufficient. Do not rewrite or correct claims. "
        "Return exactly one verdict for every supplied option_id "
        "and claim_index. Keep each reason concise."
    )

    user_prompt = json.dumps(
        {
            "claims": claim_payload,
        },
        ensure_ascii=False,
        indent=2,
    )

    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

async def evaluate_training_grounding(
    generated: GeneratedTrainingContent,
) -> GroundingEvaluationAttempt:
    """Evaluate every generated claim against its quoted support."""

    active_model = get_active_llm_model()

    if active_model is None:
        return GroundingEvaluationAttempt(
            status="disabled",
            score=None,
            output=None,
            model=None,
            reason="No active model configured",
        )

    messages = build_grounding_messages(generated)

    raw_content = await llm_structured_call(
        messages=messages,
        response_model=GroundingJudgeOutput,
        max_tokens=2500,
        temperature=0.0,
        reasoning_effort="low",
    )

    if not raw_content:
        return GroundingEvaluationAttempt(
            status="empty_response",
            score=None,
            output=None,
            model=active_model,
            reason=(
                "The grounding judge returned "
                "no usable content."
            ),
        )
    
    try: 
        judged = (GroundingJudgeOutput.model_validate_json(raw_content))

    except ValidationError as e:
        return GroundingEvaluationAttempt(
            status="invalid_output",
            score=None,
            output=None,
            model=active_model,
            reason=(
                "The grounding judge response does not "
                "match the required JSON contract."
            ),
        )

    try:
        score = calculate_grounding_score(
            generated=generated,
            judged=judged,
        )
    except InvalidGroundingOutput as error:
        return GroundingEvaluationAttempt(
            status="invalid_output",
            score=None,
            output=judged,
            model=active_model,
            reason=str(error),
        )

    supported_count = sum(
        verdict.supported
        for verdict in judged.verdicts
    )

    return GroundingEvaluationAttempt(
        status="evaluated",
        score=score,
        output=judged,
        model=active_model,
        reason=(
            f"{supported_count} of "
            f"{len(judged.verdicts)} claims "
            "were judged as supported."
        ),
    )
        
    