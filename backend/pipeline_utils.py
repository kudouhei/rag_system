from __future__ import annotations

from messages import _t

# ── Helpers ───────────────────────────────────────────────────────────────────

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
