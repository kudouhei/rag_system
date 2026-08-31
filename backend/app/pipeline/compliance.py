"""
Compliance Check Pipeline
==========================
Given a free-text business/product scenario (e.g. "We plan to launch a money
market fund offering daily redemption with no swing pricing mechanism"), this
pipeline:

  1. Retrieves the most relevant regulatory clauses (hybrid dense+sparse,
     optionally scoped by jurisdiction/product_type/regulation_number filters);
  2. Asks the LLM to assess the scenario against each retrieved requirement,
     returning a structured compliant/non-compliant/uncertain verdict per
     requirement with citations, plus an overall status;
  3. Falls back to a citation-only summary (no verdicts) if no LLM key is
     configured, so the endpoint still returns useful grounded evidence.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import List, Optional

import numpy as np

from app.core import state
from app.core.messages import _t
from app.core.schemas import ComplianceCheckRequest, ComplianceCheckResponse, ComplianceFinding
from app.llm.client import llm_call
from app.pipeline.utils import format_doc_context
from app.retrieval.bm25_index import compute_bm25_scores
from app.retrieval.embeddings import compute_embedding_scores
from app.retrieval.fusion import fuse_scores
from app.retrieval.reranker import rerank_docs

logger = logging.getLogger(__name__)


def _apply_filters(docs: List[dict], req: ComplianceCheckRequest) -> List[bool]:
    """Boolean mask: True if a chunk satisfies the requested metadata filters."""
    mask = []
    for d in docs:
        ok = True
        if req.jurisdiction and str(d.get("jurisdiction", "")).lower() != req.jurisdiction.lower():
            ok = False
        if req.product_type:
            pts = d.get("product_type") or []
            if isinstance(pts, str):
                pts = [pts]
            if req.product_type.lower() not in [str(p).lower() for p in pts]:
                ok = False
        if req.regulation_number and req.regulation_number.lower() not in str(d.get("regulation_number", "")).lower():
            ok = False
        mask.append(ok)
    return mask


async def run_compliance_check(req: ComplianceCheckRequest) -> ComplianceCheckResponse:
    t0 = time.time()
    lang = req.language or "en"

    emb_scores = compute_embedding_scores(req.scenario)
    bm25_arr   = compute_bm25_scores(req.scenario)
    graph_arr  = np.zeros(len(state.KNOWLEDGE_BASE), dtype=np.float32)
    final_arr  = fuse_scores(emb_scores, bm25_arr, graph_arr, "hybrid", False)

    mask = _apply_filters(state.KNOWLEDGE_BASE, req)
    docs_scored = []
    for idx, doc in enumerate(state.KNOWLEDGE_BASE):
        if not mask[idx]:
            continue
        d = doc.copy()
        d.update(
            embedding_score=float(emb_scores[idx]),
            bm25_score=float(bm25_arr[idx]),
            final_score=float(final_arr[idx]),
        )
        docs_scored.append(d)

    # If filters excluded everything, fall back to unfiltered retrieval.
    if not docs_scored:
        for idx, doc in enumerate(state.KNOWLEDGE_BASE):
            d = doc.copy()
            d.update(
                embedding_score=float(emb_scores[idx]),
                bm25_score=float(bm25_arr[idx]),
                final_score=float(final_arr[idx]),
            )
            docs_scored.append(d)

    results = sorted(docs_scored, key=lambda x: x["final_score"], reverse=True)[: max(req.top_k, 1)]

    if state.cross_encoder is not None and results:
        results = rerank_docs(req.scenario, results)

    if not results:
        return ComplianceCheckResponse(
            scenario=req.scenario,
            overall_status="needs_review",
            summary=("No relevant regulatory documents were found for this scenario." if lang == "en"
                      else "未找到与该场景相关的监管文档。"),
            findings=[],
            elapsed_seconds=round(time.time() - t0, 2),
        )

    context = format_doc_context(results[:req.top_k], lang)

    if not state.llm_client:
        # Graceful degrade: return retrieved clauses without an LLM verdict.
        findings = [
            ComplianceFinding(
                requirement=d.get("title", d.get("source", "")),
                citation=d.get("regulation_number", d.get("source", "")),
                source=d.get("source", ""),
                assessment="uncertain",
                rationale=("LLM not configured — showing the most relevant regulatory excerpt only."
                           if lang == "en" else
                           "未配置 LLM — 仅展示最相关的监管条款原文。"),
            )
            for d in results[:req.top_k]
        ]
        return ComplianceCheckResponse(
            scenario=req.scenario,
            overall_status="needs_review",
            summary=("DEEPSEEK_API_KEY not configured — returning retrieved clauses without an automated "
                      "compliance verdict. Review the citations manually." if lang == "en" else
                      "未配置 DEEPSEEK_API_KEY — 仅返回检索到的相关条款，未生成自动合规判断，请人工复核。"),
            findings=findings,
            elapsed_seconds=round(time.time() - t0, 2),
        )

    raw = await llm_call(
        messages=[
            {"role": "system", "content": _t("sys_compliance", lang)},
            {"role": "user",   "content": _t("usr_compliance", lang, scenario=req.scenario, context=context)},
        ],
        max_tokens=1800,
        temperature=0.2,
    )

    parsed = _parse_compliance_json(raw)
    if parsed is None:
        return ComplianceCheckResponse(
            scenario=req.scenario,
            overall_status="needs_review",
            summary=raw[:800] if raw else ("Compliance assessment failed to parse." if lang == "en" else "合规评估解析失败。"),
            findings=[],
            elapsed_seconds=round(time.time() - t0, 2),
        )

    # Attach source filenames by matching citation text back to retrieved docs where possible.
    findings = []
    for f in parsed.get("findings", []):
        citation = str(f.get("citation", ""))
        matched_src = ""
        for d in results:
            reg_no = str(d.get("regulation_number", ""))
            if reg_no and reg_no in citation:
                matched_src = d.get("source", "")
                break
        findings.append(ComplianceFinding(
            requirement=str(f.get("requirement", "")),
            citation=citation,
            source=matched_src,
            assessment=str(f.get("assessment", "uncertain")),
            rationale=str(f.get("rationale", "")),
        ))

    return ComplianceCheckResponse(
        scenario=req.scenario,
        overall_status=str(parsed.get("overall_status", "needs_review")),
        summary=str(parsed.get("summary", "")),
        findings=findings,
        elapsed_seconds=round(time.time() - t0, 2),
    )


def _parse_compliance_json(raw: Optional[str]) -> Optional[dict]:
    if not raw:
        return None
    try:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return None
        return json.loads(m.group())
    except Exception as e:
        logger.warning("Compliance JSON parse error: %s — raw=%r", e, raw[:300])
        return None
