"""Coverage-aware evidence selection for training explanations."""
from __future__ import annotations

from dataclasses import dataclass

from app.training.retrieval import TrainingRetrievalResult
from app.training.schemas import (
    TrainingEvidence,
    TrainingExplanationRequest,
)


@dataclass(slots=True)
class TrainingEvidenceSelection:
    docs: list[dict]
    doc_ids_by_option: dict[str, list[str]]
    option_ids_without_candidates: list[str]


def select_training_evidence(
    request: TrainingExplanationRequest,
    retrieval: TrainingRetrievalResult,
    max_evidence: int = 8,
    per_option_limit: int = 1,
) -> TrainingEvidenceSelection:
    if max_evidence < 1:
        raise ValueError("max_evidence must be at least 1")

    if per_option_limit < 1:
        raise ValueError("per_option_limit must be at least 1")

    option_ids = [
        option.option_id
        for option in request.options
    ]

    # If a question has more options than max_evidence, allow at least one
    # distinct evidence document per option.
    effective_limit = max(max_evidence, len(option_ids))

    doc_ids_by_option = {
        option_id: []
        for option_id in option_ids
    }

    # merged_docs contains the canonical, RRF-enriched document copies.
    canonical_docs = {
        doc["id"]: doc
        for doc in retrieval.merged_docs
    }

    # Regular dict preserves insertion order in Python 3.12.
    selected_by_id: dict[str, dict] = {}

    # Pass 1: coverage first — select the best candidate for each option.
    for option_id in option_ids:
        query_id = f"option:{option_id}"
        candidates = retrieval.results_by_query.get(query_id, [])

        if not candidates:
            continue

        best_candidate = candidates[0]
        doc_id = best_candidate.get("id")

        if not doc_id:
            continue

        selected_by_id.setdefault(
            doc_id,
            canonical_docs.get(doc_id, best_candidate),
        )
        doc_ids_by_option[option_id].append(doc_id)

    # Pass 2: use remaining context budget for globally strong evidence.
    for doc in retrieval.merged_docs:
        if len(selected_by_id) >= effective_limit:
            break

        doc_id = doc.get("id")
        if not doc_id:
            continue

        selected_by_id.setdefault(doc_id, doc)

    # Pass 3: link selected global evidence back to the options it matched.
    for doc_id, doc in selected_by_id.items():
        for option_id in doc.get("matched_option_ids", []):
            if option_id not in doc_ids_by_option:
                continue

            linked_ids = doc_ids_by_option[option_id]

            if (
                doc_id not in linked_ids
                and len(linked_ids) < per_option_limit
            ):
                linked_ids.append(doc_id)

    option_ids_without_candidates = [
        option_id
        for option_id, doc_ids in doc_ids_by_option.items()
        if not doc_ids
    ]

    return TrainingEvidenceSelection(
        docs=list(selected_by_id.values()),
        doc_ids_by_option=doc_ids_by_option,
        option_ids_without_candidates=option_ids_without_candidates,
    )

@dataclass(slots=True)
class MaterializedTrainingEvidence:
    evidence: list[TrainingEvidence]
    evidence_ids_by_option: dict[str, list[str]]


def _required_doc_text(doc: dict, field: str) -> str:
    value = str(doc.get(field, "")).strip()

    if not value:
        raise ValueError(
            f"retrieved document is missing required field: {field}"
        )

    return value


def _optional_doc_text(doc: dict, field: str) -> str | None:
    value = doc.get(field)

    if value is None:
        return None

    text = str(value).strip()
    return text or None

def _build_section_label(doc: dict) -> str | None:
    explicit_section = _optional_doc_text(doc, "section")

    if explicit_section:
        return explicit_section

    article = _optional_doc_text(doc, "article")

    if article:
        return f"Article {article}"

    return None


def materialize_training_evidence(
    selection: TrainingEvidenceSelection,
) -> MaterializedTrainingEvidence:
    evidence: list[TrainingEvidence] = []
    evidence_id_by_doc_id: dict[str, str] = {}

    for doc in selection.docs:
        doc_id = _required_doc_text(doc, "id")

        stable_chunk_id = str(
            doc.get("stable_id") or doc_id
        ).strip()
        source = _required_doc_text(doc, "source")
        content = _required_doc_text(doc, "content")

        evidence_id = f"E{len(evidence) + 1}"
        evidence_id_by_doc_id[doc_id] = evidence_id

        raw_score = float(
            doc.get(
                "training_score",
                doc.get(
                    "ce_score",
                    doc.get("final_score", 0.0),
                ),
            )
        )
        relevance_score = min(1.0, max(0.0, raw_score))

        evidence.append(
            TrainingEvidence(
                evidence_id=evidence_id,
                chunk_id=stable_chunk_id,
                source=source,
                title=_optional_doc_text(doc, "title"),
                section=_build_section_label(doc),
                excerpt=content[:4000],
                relevance_score=relevance_score,
                regulation_number=_optional_doc_text(
                    doc,
                    "regulation_number",
                ),
                issuing_authority=_optional_doc_text(
                    doc,
                    "issuing_authority",
                ),
                effective_date=_optional_doc_text(
                    doc,
                    "effective_date",
                ),
            )
        )

    evidence_ids_by_option = {
        option_id: [
            evidence_id_by_doc_id[doc_id]
            for doc_id in doc_ids
            if doc_id in evidence_id_by_doc_id
        ]
        for option_id, doc_ids
        in selection.doc_ids_by_option.items()
    }

    return MaterializedTrainingEvidence(
        evidence=evidence,
        evidence_ids_by_option=evidence_ids_by_option,
    )