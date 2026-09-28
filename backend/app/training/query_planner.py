"""Build retrieval queries for a training question and its options."""
from __future__ import annotations

from dataclasses import dataclass

from app.training.schemas import TrainingExplanationRequest

@dataclass(frozen=True, slots=True)
class TrainingRetrievalQuery:
    query_id: str
    option_id: str | None
    text: str

def build_retrieval_queries(
    request: TrainingExplanationRequest,
) -> list[TrainingRetrievalQuery]:
    correct_ids = set(request.correct_option_ids)
    correct_option_texts = [
        option.text
        for option in request.options
        if option.option_id in correct_ids
    ]

    question_parts = [
        f"Jurisdiction: {request.jurisdiction}",
        f"GDPR training question: {request.question}",
        "Correct answer concepts: " + " | ".join(correct_option_texts),
    ]

    if request.source_references:
        question_parts.append(
            "Authoritative source hints: "
            + " | ".join(request.source_references)
        )

    if request.reference_analysis:
        question_parts.append(
            f"Instructor analysis: {request.reference_analysis}"
        )

    queries = [
        TrainingRetrievalQuery(
            query_id="question",
            option_id=None,
            text="\n".join(question_parts),
        )
    ]

    for option in request.options:
        option_parts = [
            f"Jurisdiction: {request.jurisdiction}",
            f"GDPR training question: {request.question}",
            f"Candidate answer {option.option_id}: {option.text}",
            (
                "Find the authoritative GDPR rule needed to assess "
                "this candidate answer."
            ),
        ]

        if request.source_references:
            option_parts.append(
                "Authoritative source hints: "
                + " | ".join(request.source_references)
            )

        queries.append(
            TrainingRetrievalQuery(
                query_id=f"option:{option.option_id}",
                option_id=option.option_id,
                text="\n".join(option_parts),
            )
        )

    return queries