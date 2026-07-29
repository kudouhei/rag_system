"""Health, knowledge-base stats, and asset-inventory endpoints."""
from __future__ import annotations

import json
import time

from fastapi import APIRouter

from app.core import state
from app.core.audit import AUDIT_FILE, FEEDBACK_FILE
from app.core.config import (
    CACHE_DIR,
    CONTEXTUAL_CHUNKING,
    DEEPSEEK_MODEL,
    EMBED_MODEL_NAME,
    ENABLE_GRAPH,
    RERANKER_MODEL,
)

router = APIRouter()


@router.get("/health")
async def health():
    return {
        "status":               "ok",
        "docs_count":           len(state.KNOWLEDGE_BASE),
        "embed_model":          EMBED_MODEL_NAME,
        "cross_encoder":        RERANKER_MODEL or "disabled (cosine fallback)",
        "llm_enabled":          state.llm_client is not None,
        "llm_model":            DEEPSEEK_MODEL if state.llm_client else None,
        "contextual_chunking":  CONTEXTUAL_CHUNKING,
        "graph_enabled":        ENABLE_GRAPH,
        "graph_nodes":          len(state.KNOWLEDGE_GRAPH["nodes"]),
        "graph_edges":          len(state.KNOWLEDGE_GRAPH["edges"]),
        "embedding_cache_dir":  str(CACHE_DIR),
    }


@router.get("/stats")
async def get_stats():
    """Comprehensive knowledge base analytics for operations teams."""
    # Per-source breakdown
    sources: dict = {}
    total_words = 0
    for d in state.KNOWLEDGE_BASE:
        src = d.get("source") or "unknown"
        if src not in sources:
            sources[src] = {
                "source":     src,
                "chunks":     0,
                "words":      0,
                "file_size_kb": d.get("file_size_kb", 0),
                "last_modified": d.get("file_mtime", ""),
                "tags":       d.get("tags", []),
            }
        sources[src]["chunks"] += 1
        sources[src]["words"]  += d.get("word_count", 0)
        total_words            += d.get("word_count", 0)

    # Freshness: flag sources not updated in > 90 days
    now_ts = time.time()
    stale_sources = []
    for d in state.KNOWLEDGE_BASE:
        mtime = d.get("mtime", now_ts)
        if now_ts - mtime > 90 * 86400:
            src = d.get("source", "")
            if src and src not in stale_sources:
                stale_sources.append(src)

    # Feedback summary (if enabled)
    fb_total = 0
    fb_pos = 0
    if FEEDBACK_FILE.exists():
        try:
            with FEEDBACK_FILE.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        fb_total += 1
                        if int(obj.get("rating", 0)) > 0:
                            fb_pos += 1
                    except Exception:
                        continue
        except Exception:
            pass

    return {
        "total_chunks":          len(state.KNOWLEDGE_BASE),
        "total_sources":         len(sources),
        "total_words":           total_words,
        "contextual_chunking":   CONTEXTUAL_CHUNKING,
        "sources": sorted(sources.values(), key=lambda x: x["source"]),
        "stale_sources":         stale_sources,      # not updated in 90+ days
        "feedback": {
            "total": fb_total,
            "positive": fb_pos,
            "satisfaction_rate": (fb_pos / fb_total) if fb_total else 0.0,
        },
    }


@router.get("/inventory")
async def knowledge_asset_inventory():
    """
    Knowledge Asset Inventory:
    - per-source size metrics (chunks/words/mtime)
    - freshness (stale > 90 days)
    - usage frequency (from audit.jsonl retrieval_complete events)
    - feedback attribution (positive/negative per source via feedback doc_ids)
    """
    from app.core.audit import _utc_now_iso

    # Base: per-source stats from indexed chunks
    per_src: dict = {}
    doc_id_to_src = {}
    now_ts = time.time()

    for d in state.KNOWLEDGE_BASE:
        src = d.get("source") or "unknown"
        doc_id_to_src[d.get("id")] = src
        if src not in per_src:
            per_src[src] = {
                "source": src,
                "chunks": 0,
                "words": 0,
                "file_size_kb": d.get("file_size_kb", 0),
                "last_modified": d.get("file_mtime", ""),
                "mtime": d.get("mtime", now_ts),
                "tags": d.get("tags", []),
                # analytics
                "usage_hits": 0,          # how often this source appeared in retrieved docs
                "usage_queries": 0,        # how many retrieval events included this source (unique per event)
                "feedback_total": 0,
                "feedback_positive": 0,
            }
        per_src[src]["chunks"] += 1
        per_src[src]["words"] += int(d.get("word_count", 0))

    # Usage: parse audit retrieval_complete events
    if AUDIT_FILE.exists():
        try:
            with AUDIT_FILE.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if obj.get("type") != "retrieval_complete":
                        continue
                    retrieved = obj.get("retrieved") or []
                    seen_sources = set()
                    for r in retrieved:
                        src = r.get("source")
                        if not src or src not in per_src:
                            continue
                        per_src[src]["usage_hits"] += 1
                        seen_sources.add(src)
                    for src in seen_sources:
                        per_src[src]["usage_queries"] += 1
        except Exception:
            pass

    # Feedback attribution: join feedback doc_ids → source
    if FEEDBACK_FILE.exists():
        try:
            with FEEDBACK_FILE.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    rating = int(obj.get("rating", 0))
                    doc_ids = obj.get("doc_ids") or []
                    # Count each source at most once per feedback event (avoid overcounting multi-chunk from same file)
                    seen_sources = set()
                    for did in doc_ids:
                        src = doc_id_to_src.get(did)
                        if not src or src not in per_src:
                            continue
                        seen_sources.add(src)
                    for src in seen_sources:
                        per_src[src]["feedback_total"] += 1
                        if rating > 0:
                            per_src[src]["feedback_positive"] += 1
        except Exception:
            pass

    assets = []
    for src, s in per_src.items():
        stale = (now_ts - float(s.get("mtime", now_ts))) > 90 * 86400
        fb_total = s["feedback_total"]
        fb_pos = s["feedback_positive"]
        assets.append({
            "source": src,
            "chunks": s["chunks"],
            "words": s["words"],
            "file_size_kb": s.get("file_size_kb", 0),
            "last_modified": s.get("last_modified", ""),
            "tags": s.get("tags", []),
            "stale": stale,
            "usage_hits": s["usage_hits"],
            "usage_queries": s["usage_queries"],
            "feedback_total": fb_total,
            "feedback_positive": fb_pos,
            "feedback_satisfaction_rate": (fb_pos / fb_total) if fb_total else None,
        })

    # Sort by usage, then recency
    assets.sort(key=lambda a: (a["usage_queries"], a["usage_hits"]), reverse=True)

    return {
        "generated_at": _utc_now_iso(),
        "total_sources": len(assets),
        "total_chunks": len(state.KNOWLEDGE_BASE),
        "assets": assets,
    }
