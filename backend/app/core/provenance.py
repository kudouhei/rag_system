"""Stable provenance identifiers for the indexed knowledge corpus."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

_CORPUS_VERSION_FIELDS = (
    "source",
    "title",
    "content",
    "chunk_index",
    "total_chunks",
    "regulation_number",
    "issuing_authority",
    "jurisdiction",
    "document_type",
    "effective_date",
    "status",
    "version",
    "access_scope",
    "tenant_id",
    "course_ids",
)

def compute_corpus_version(docs: Sequence[dict]) -> str:
    digest = hashlib.sha256()

    sorted_docs = sorted(
        docs,
        key=lambda doc: (
            str(doc.get("source", "")),
            int(doc.get("chunk_index", 0)),
            str(doc.get("id", "")),
        ),
    )
    for doc in sorted_docs:
        semantic_payload = {
            field: doc.get(field)
            for field in _CORPUS_VERSION_FIELDS
        }

        encoded_payload = json.dumps(
            semantic_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")

        digest.update(encoded_payload)
        digest.update(b"\n")

    return f"sha256:{digest.hexdigest()}"