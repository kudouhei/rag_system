from __future__ import annotations

from typing import List

from app.core.messages import _t

# ── Helpers ───────────────────────────────────────────────────────────────────

def format_doc_context(docs: List[dict], lang: str = "en") -> str:
    """
    Assemble the retrieved-document context block passed to the LLM.
    Regulatory metadata (regulation number, issuing authority, article/section)
    is surfaced in the citation label so the model can cite it directly,
    e.g. "[Doc 1] REG-FM-105 Article 3 — Meridian Financial Conduct Authority (MFCA)".
    """
    doc_label = "文档" if lang == "zh" else "Doc"
    blocks = []
    for i, d in enumerate(docs):
        reg_no   = d.get("regulation_number")
        issuer   = d.get("issuing_authority")
        cite_bits = [b for b in [reg_no, d.get("title")] if b]
        cite = " — ".join(cite_bits) if cite_bits else d.get("title", "")
        if issuer:
            cite += f" ({issuer})"
        blocks.append(f"[{doc_label} {i + 1}] {cite}\nSource: {d.get('source', '')}\n{d['content']}")
    return "\n\n".join(blocks)


def _diagnose_failure(top_score: float, threshold: float, lang: str = "zh") -> str:
    if top_score < 0.35:
        return _t("failure_extreme", lang)
    if top_score < threshold:
        return _t("failure_moderate", lang)
    return _t("failure_low", lang)


def _iter_summary(iteration, query, strategy, top_score, reflected, results):
    return {
        "iteration": iteration, "query": query, "strategy": strategy,
        "top_score": round(top_score, 3), "reflected": reflected,
        "results": [{"id": r["id"], "title": r["title"], "score": round(r["final_score"], 3)} for r in results[:3]],
    }
