"""
Adaptive RAG — MCP Server
=========================
Exposes the enterprise knowledge base as MCP tools so Claude, Cursor,
and any MCP-compatible AI client can search internal documents directly.

Tools
─────
  • search_knowledge_base   — Full RAG pipeline: retrieve + rerank + LLM answer
  • retrieve_documents      — Retrieval only (no LLM), returns ranked doc chunks
  • list_documents          — List every indexed document chunk in the KB
  • get_kb_stats            — System health, doc counts, and quality metrics

Usage
─────
  python mcp_server.py          # stdio transport (default, for Cursor / Claude Desktop)
  python mcp_server.py --http   # HTTP/SSE transport (for remote clients)

Configuration
─────────────
  Reads the same backend/.env as main.py (DEEPSEEK_API_KEY, EMBED_MODEL, etc.)
"""

import asyncio
import json
import os
import sys
import logging
from contextlib import asynccontextmanager

# ── Resolve backend directory so imports work regardless of cwd ───────────────
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from fastmcp import FastMCP, Context

# Import shared RAG state and functions from main.py.
# We import lazily inside the lifespan so that heavy models (sentence-transformers,
# BM25) are only loaded after the MCP handshake completes.
import main as rag

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# Lifespan — initialise the RAG system once at server start
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def rag_lifespan(server):
    """Load embedding model, BM25 index, and LLM client on startup."""
    logger.info("RAG MCP Server: initialising knowledge base…")
    await rag._startup()
    logger.info(
        "RAG MCP Server: ready — %d chunks indexed, LLM=%s",
        len(rag.KNOWLEDGE_BASE),
        rag.DEEPSEEK_MODEL if rag.llm_client else "disabled",
    )
    yield
    logger.info("RAG MCP Server: shutting down")


mcp = FastMCP(
    name="企业知识库 RAG",
    instructions=(
        "This server provides access to an enterprise knowledge base using "
        "Adaptive RAG (HyDE · Hybrid Dense-Sparse Retrieval · Cross-Encoder Reranking). "
        "Use `search_knowledge_base` to ask questions and get grounded answers with citations. "
        "Use `retrieve_documents` when you only need raw document chunks. "
        "Use `list_documents` to explore what topics are covered."
    ),
    lifespan=rag_lifespan,
)

# ══════════════════════════════════════════════════════════════════════════════
# Tool 1 — Full RAG Q&A
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def search_knowledge_base(
    query: str,
    top_k: int = 5,
    strategy: str = "adaptive",
    enable_hyde: bool = False,
    language: str = "zh",
) -> str:
    """
    Search the enterprise knowledge base and return a grounded answer.

    Runs the full Adaptive RAG pipeline:
      1. Optional HyDE (hypothetical-document embeddings) for better recall
      2. Hybrid dense+sparse retrieval with iterative reflection
      3. Cross-encoder reranking
      4. LLM answer generation with source citations

    Args:
        query:       The question to answer (Chinese or English).
        top_k:       Number of document chunks to retrieve (default 5).
        strategy:    Retrieval strategy — "adaptive" | "hybrid" | "vector" | "bm25".
                     "adaptive" automatically cycles strategies on low confidence.
        enable_hyde: If True, generate a hypothetical document to improve recall
                     on vague or abstract queries. Requires DEEPSEEK_API_KEY.
        language:    Response language — "zh" (default) or "en".

    Returns:
        Formatted string with the LLM answer followed by source citations.
    """
    result = await rag.query_rag(
        query=query,
        strategy=strategy,
        enable_hyde=enable_hyde,
        enable_iterative=True,
        top_k=top_k,
        language=language,
    )

    lines = [result["answer"], ""]

    # Source citations
    cite_header = "**来源文档**" if language == "zh" else "**Sources**"
    lines.append(cite_header)
    for i, doc in enumerate(result["docs"], 1):
        score_pct = int(doc["score"] * 100)
        lines.append(f"{i}. [{doc['title']}]  `{doc['source']}`  (相关度 {score_pct}%)")

    # Quality metrics (compact)
    m = result.get("metrics", {})
    if m:
        cr  = m.get("context_relevance", 0)
        af  = m.get("answer_faithfulness", 0)
        elapsed = result.get("elapsed", 0)
        metrics_line = (
            f"\n*检索相关度 {cr:.0%} · 答案忠实度 {af:.0%} · "
            f"耗时 {elapsed}s · {result['iterations']} 轮迭代*"
            if language == "zh" else
            f"\n*Context relevance {cr:.0%} · Faithfulness {af:.0%} · "
            f"{elapsed}s · {result['iterations']} iteration(s)*"
        )
        lines.append(metrics_line)

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Tool 2 — Retrieval only (no LLM answer)
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def retrieve_documents(
    query: str,
    top_k: int = 5,
    strategy: str = "hybrid",
) -> str:
    """
    Retrieve the most relevant document chunks for a query without generating an answer.

    Useful when you want to read the raw source material yourself, check coverage,
    or pass the chunks to your own prompt.

    Args:
        query:    The search query.
        top_k:    Number of chunks to return (default 5, max 20).
        strategy: "hybrid" (default) | "vector" | "bm25".

    Returns:
        JSON array of document chunks sorted by relevance score.
    """
    top_k = min(top_k, 20)
    result = await rag.query_rag(
        query=query,
        strategy=strategy,
        enable_hyde=False,
        enable_iterative=False,
        top_k=top_k,
        language="zh",
    )

    docs_out = [
        {
            "rank":    i + 1,
            "id":      d["id"],
            "title":   d["title"],
            "source":  d["source"],
            "score":   d["score"],
            "content": d["content"],
        }
        for i, d in enumerate(result["docs"])
    ]
    return json.dumps(docs_out, ensure_ascii=False, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# Tool 3 — List all indexed documents
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def list_documents() -> str:
    """
    List every document chunk currently indexed in the knowledge base.

    Returns a JSON array with id, title, source file, tags, and character count
    for each chunk. Use this to understand what topics the KB covers before
    deciding how to query it.
    """
    docs = [
        {
            "id":         d["id"],
            "title":      d["title"],
            "source":     d.get("source", ""),
            "tags":       d.get("tags", []),
            "char_count": d.get("char_count", len(d["content"])),
        }
        for d in rag.KNOWLEDGE_BASE
    ]
    return json.dumps(docs, ensure_ascii=False, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# Tool 4 — System health & stats
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def get_kb_stats() -> str:
    """
    Return knowledge base statistics and system health information.

    Includes: total chunks, source files, embedding model, LLM status,
    cross-encoder status, and user feedback satisfaction rate (if available).
    """
    sources: dict = {}
    for d in rag.KNOWLEDGE_BASE:
        src = d.get("source") or "unknown"
        if src not in sources:
            sources[src] = {"chunks": 0, "words": 0, "tags": d.get("tags", [])}
        sources[src]["chunks"] += 1
        sources[src]["words"]  += d.get("word_count", 0)

    stats = {
        "status":              "ok",
        "total_chunks":        len(rag.KNOWLEDGE_BASE),
        "total_sources":       len(sources),
        "sources":             list(sources.keys()),
        "embed_model":         rag.EMBED_MODEL_NAME,
        "llm_enabled":         rag.llm_client is not None,
        "llm_model":           rag.DEEPSEEK_MODEL if rag.llm_client else None,
        "cross_encoder":       rag.RERANKER_MODEL or "cosine fallback",
        "contextual_chunking": rag.CONTEXTUAL_CHUNKING,
        "hyde_available":      rag.llm_client is not None,
    }
    return json.dumps(stats, ensure_ascii=False, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Adaptive RAG MCP Server")
    parser.add_argument(
        "--http", action="store_true",
        help="Run as HTTP/SSE server on port 8001 instead of stdio",
    )
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.http:
        mcp.run(transport="streamable-http", host="0.0.0.0", port=args.port)
    else:
        mcp.run(transport="stdio")
