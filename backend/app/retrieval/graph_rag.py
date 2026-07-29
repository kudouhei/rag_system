"""
Knowledge Graph — GraphRAG (Graph-enhanced Retrieval)
========================================================
Builds an entity/relation graph from the indexed chunks (LLM extraction when
available, jieba keyword co-occurrence as a fallback), caches it to disk, and
exposes a query-time scoring function that expands matched entities one hop
through the graph.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from typing import List, Optional

import numpy as np

from app.core import state
from app.core.config import CACHE_DIR, ENABLE_GRAPH
from app.retrieval.embeddings import doc_fingerprint
from app.llm.client import llm_call

logger = logging.getLogger(__name__)

_ZH_STOPWORDS = {
    "的","是","在","了","和","与","或","但","如","从","到","为","以","及",
    "其","这","那","有","无","一","不","也","都","将","中","上","下","我",
    "你","他","她","它","们","对","所","把","被","让","使","由","按","等",
}


def _extract_keywords_as_entities(doc: dict) -> dict:
    """Keyword-based entity extraction (fast, no LLM). Uses jieba if available."""
    if state.tokenize_fn is None:
        return {"entities": [], "relations": []}
    from collections import Counter
    tokens = [
        t for t in state.tokenize_fn(doc["title"] + " " + doc["content"])
        if len(t) > 1 and t not in _ZH_STOPWORDS and not t.isdigit()
    ]
    top_kw = [w for w, _ in Counter(tokens).most_common(7)]
    return {"entities": [{"name": kw, "type": "keyword"} for kw in top_kw], "relations": []}


async def _extract_entities_llm(doc: dict) -> dict:
    """LLM-based entity + relation extraction for a single chunk."""
    prompt = (
        "从以下文本中提取关键实体和它们的关系。返回严格JSON（无多余文字）：\n"
        '{"entities":[{"name":"实体名","type":"概念|技术|方法|系统|其他"}],'
        '"relations":[{"source":"...","target":"...","relation":"..."}]}\n'
        "要求：最多6个实体（名称≤8字），最多4条关系。\n\n文本：\n"
    )
    raw = await llm_call(
        messages=[{"role": "user", "content": prompt + doc["content"][:600]}],
        max_tokens=300, temperature=0.1,
    )
    if not raw:
        return _extract_keywords_as_entities(doc)
    try:
        import re as _re
        m = _re.search(r'\{.*\}', raw, _re.DOTALL)
        if m:
            d = json.loads(m.group())
            return {"entities": d.get("entities", []), "relations": d.get("relations", [])}
    except Exception:
        pass
    return _extract_keywords_as_entities(doc)


def _graph_fingerprint(docs: List[dict]) -> str:
    fp     = doc_fingerprint(docs)
    flavor = "llm" if state.llm_client else "kw"
    return hashlib.sha256(f"{fp}|graph|{flavor}".encode()).hexdigest()[:20]


def _load_graph_cache(docs: List[dict]) -> Optional[dict]:
    cache_file = CACHE_DIR / f"graph_{_graph_fingerprint(docs)}.json"
    if cache_file.exists():
        logger.info("Graph cache hit — loading from %s", cache_file.name)
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _save_graph_cache(graph: dict, docs: List[dict]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fp         = _graph_fingerprint(docs)
    cache_file = CACHE_DIR / f"graph_{fp}.json"
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)
    for old in CACHE_DIR.glob("graph_*.json"):
        if old != cache_file:
            old.unlink(missing_ok=True)
    logger.info("Graph cached: %d nodes, %d edges (%s)", len(graph["nodes"]), len(graph["edges"]), fp)


async def _build_knowledge_graph(docs: List[dict]) -> dict:
    """
    Build knowledge graph from all document chunks.
    • LLM available  → semantic entity + relation extraction
    • LLM unavailable → jieba keyword co-occurrence graph
    Edges are added for:
      1. Explicit LLM-extracted relations
      2. Co-occurrence within the same chunk (both modes)
    """
    graph: dict = {"nodes": {}, "edges": {}}
    use_llm     = bool(state.llm_client)
    logger.info("Building knowledge graph (%s mode, %d chunks)…", "LLM" if use_llm else "keyword", len(docs))

    for i, doc in enumerate(docs):
        chunk_id   = doc["id"]
        extraction = await _extract_entities_llm(doc) if use_llm else _extract_keywords_as_entities(doc)

        # Register nodes
        chunk_ents: List[str] = []
        for ent in extraction.get("entities", []):
            name = ent.get("name", "").strip()
            if not name or len(name) < 2:
                continue
            if name not in graph["nodes"]:
                graph["nodes"][name] = {"type": ent.get("type", "other"), "chunk_ids": [], "freq": 0}
            if chunk_id not in graph["nodes"][name]["chunk_ids"]:
                graph["nodes"][name]["chunk_ids"].append(chunk_id)
            graph["nodes"][name]["freq"] += 1
            chunk_ents.append(name)

        # Explicit LLM relations
        for rel in extraction.get("relations", []):
            src, tgt = rel.get("source", "").strip(), rel.get("target", "").strip()
            if src in graph["nodes"] and tgt in graph["nodes"]:
                key = f"{src}||{tgt}"
                if key not in graph["edges"]:
                    graph["edges"][key] = {"source": src, "target": tgt,
                                           "relation": rel.get("relation", "related_to"), "weight": 1}
                else:
                    graph["edges"][key]["weight"] += 1

        # Co-occurrence edges within the same chunk
        for j in range(len(chunk_ents)):
            for k in range(j + 1, len(chunk_ents)):
                a, b = chunk_ents[j], chunk_ents[k]
                key  = f"{a}||{b}"
                if key not in graph["edges"]:
                    graph["edges"][key] = {"source": a, "target": b, "relation": "co-occurs", "weight": 1}
                else:
                    graph["edges"][key]["weight"] += 1

        if (i + 1) % 5 == 0 or (i + 1) == len(docs):
            logger.info("  graph: %d/%d chunks, %d nodes, %d edges",
                        i + 1, len(docs), len(graph["nodes"]), len(graph["edges"]))
        await asyncio.sleep(0.01)

    return graph


async def init_graph(docs: List[dict]) -> None:
    if not ENABLE_GRAPH:
        logger.info("GraphRAG disabled (ENABLE_GRAPH=false)")
        return
    cached = _load_graph_cache(docs)
    if cached is not None:
        state.KNOWLEDGE_GRAPH = cached
        return
    state.KNOWLEDGE_GRAPH = await _build_knowledge_graph(docs)
    _save_graph_cache(state.KNOWLEDGE_GRAPH, docs)
    logger.info("✓ Knowledge graph ready: %d nodes, %d edges",
                len(state.KNOWLEDGE_GRAPH["nodes"]), len(state.KNOWLEDGE_GRAPH["edges"]))


def compute_graph_scores(query: str) -> np.ndarray:
    """
    Graph-based retrieval scores.
    1. Match query tokens → graph nodes (direct match, weight 1.5)
    2. Expand to 1-hop neighbours (weight 1.0)
    3. Aggregate chunk scores, normalise to [0, 1]
    Returns zeros if graph is empty or no nodes match.
    """
    scores = np.zeros(len(state.KNOWLEDGE_BASE), dtype=np.float32)
    if not state.KNOWLEDGE_GRAPH["nodes"] or not state.KNOWLEDGE_BASE:
        return scores

    q_tokens = {t for t in (state.tokenize_fn(query) if state.tokenize_fn else query.split()) if len(t) > 1}

    # Direct node matches
    matched: set = set()
    for node_name in state.KNOWLEDGE_GRAPH["nodes"]:
        if node_name in query or any(t in node_name for t in q_tokens):
            matched.add(node_name)

    if not matched:
        return scores

    # 1-hop expansion
    neighbours: set = set()
    for edge in state.KNOWLEDGE_GRAPH["edges"].values():
        if edge["source"] in matched:
            neighbours.add(edge["target"])
        if edge["target"] in matched:
            neighbours.add(edge["source"])
    neighbours -= matched

    # Aggregate chunk scores
    chunk_scores: dict = {}
    for node_name, w in [(n, 1.5) for n in matched] + [(n, 1.0) for n in neighbours]:
        for cid in state.KNOWLEDGE_GRAPH["nodes"].get(node_name, {}).get("chunk_ids", []):
            chunk_scores[cid] = chunk_scores.get(cid, 0.0) + w

    if not chunk_scores:
        return scores

    mx        = max(chunk_scores.values())
    id_to_idx = {doc["id"]: i for i, doc in enumerate(state.KNOWLEDGE_BASE)}
    for cid, s in chunk_scores.items():
        if cid in id_to_idx:
            scores[id_to_idx[cid]] = float(s) / mx

    return scores
