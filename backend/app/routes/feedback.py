"""User feedback endpoint."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.core.audit import FEEDBACK_FILE, _append_jsonl, _utc_now_iso, redact_text
from app.core.schemas import FeedbackRequest

router = APIRouter()


@router.post("/feedback")
async def feedback(req: FeedbackRequest):
    if req.rating not in (-1, 1):
        raise HTTPException(400, "rating must be 1 or -1")

    _append_jsonl(FEEDBACK_FILE, {
        "ts": _utc_now_iso(),
        "tenant_id": req.tenant_id,
        "user_id": req.user_id,
        "user_role": req.user_role,
        "language": req.language,
        "rating": req.rating,
        "doc_ids": req.doc_ids or [],
        "comment": redact_text(req.comment),
        # Store redacted content only (avoid accidental PII persistence)
        "query": redact_text(req.query),
        "answer": redact_text(req.answer),
    })
    return {"status": "ok"}
