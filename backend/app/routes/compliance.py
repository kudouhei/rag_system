"""Compliance check endpoint — scenario-based regulatory compliance assessment."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.core.schemas import ComplianceCheckRequest, ComplianceCheckResponse
from app.pipeline.compliance import run_compliance_check

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/compliance_check", response_model=ComplianceCheckResponse)
async def compliance_check(req: ComplianceCheckRequest):
    """
    Assess a business/product scenario against the indexed regulatory corpus.

    Retrieves the most relevant regulatory clauses (optionally scoped by
    jurisdiction / product_type / regulation_number) and returns a structured
    per-requirement compliance verdict with citations, plus an overall status.
    """
    if not req.scenario or not req.scenario.strip():
        raise HTTPException(400, "scenario is required")
    try:
        return await run_compliance_check(req)
    except Exception as e:
        logger.error("Compliance check failed: %s", e)
        raise HTTPException(500, f"Compliance check failed: {e}")
