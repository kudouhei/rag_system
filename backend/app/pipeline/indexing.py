"""
Indexing — build and rebuild all retrieval indexes
=====================================================
Orchestrates document loading, contextual chunking, embedding (with cache),
BM25, cross-encoder, and knowledge-graph construction. Called once at startup
and again on `/reload`, `/upload`, and document deletion.
"""
from __future__ import annotations

import asyncio
import logging

from app.core import state
from app.core.config import CONTEXTUAL_CHUNKING, DOCS_DIR
from app.ingestion.contextual_chunking import contextualize_chunks
from app.ingestion.documents import load_documents_from_folder
from app.llm.client import init_llm_client
from app.retrieval.bm25_index import init_bm25
from app.retrieval.embeddings import compute_doc_embeddings, init_embed_model, load_emb_cache, save_emb_cache
from app.retrieval.graph_rag import init_graph
from app.retrieval.reranker import init_cross_encoder

logger = logging.getLogger(__name__)


async def startup() -> None:
    raw_docs = load_documents_from_folder(DOCS_DIR)
    if not raw_docs:
        logger.warning("No documents found in %s", DOCS_DIR)
        raw_docs = [{
            "id": "placeholder", "title": "No documents indexed",
            "content": f"Add regulatory .txt/.md/.pdf documents to {DOCS_DIR} and restart the service.",
            "source": "", "tags": [], "embedding_score": 0.0, "bm25_score": 0.0,
            "word_count": 0, "char_count": 0, "chunk_index": 0, "total_chunks": 1,
        }]

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, init_embed_model)

    # Contextual Chunking (optional, requires LLM key)
    init_llm_client()
    docs = await contextualize_chunks(raw_docs)

    # Embedding Cache — avoid re-encoding unchanged docs
    cached = load_emb_cache(docs)
    if cached is not None:
        state.doc_embeddings = cached
    else:
        state.doc_embeddings = await loop.run_in_executor(None, compute_doc_embeddings, docs)
        save_emb_cache(docs, state.doc_embeddings)

    await loop.run_in_executor(None, init_bm25, docs)
    await loop.run_in_executor(None, init_cross_encoder)

    state.KNOWLEDGE_BASE = docs
    # Knowledge Graph — build in background so the API starts fast
    asyncio.create_task(init_graph(docs))

    logger.info("✓ RAG system ready — %d chunks indexed (contextual=%s, cached=%s, graph_nodes=%d)",
                len(state.KNOWLEDGE_BASE), CONTEXTUAL_CHUNKING, cached is not None,
                len(state.KNOWLEDGE_GRAPH["nodes"]))


async def rebuild_index(force_reembed: bool = False) -> None:
    """
    Trigger a full index rebuild.
    - force_reembed=False (default): use embedding cache if docs unchanged
    - force_reembed=True: always re-embed (needed after changing EMBED_MODEL or CONTEXTUAL_CHUNKING)
    """
    logger.info("Rebuilding index (force_reembed=%s)…", force_reembed)
    raw_docs = load_documents_from_folder(DOCS_DIR)
    if not raw_docs:
        logger.warning("No documents found during rebuild")
        return

    docs = await contextualize_chunks(raw_docs)
    loop = asyncio.get_event_loop()

    cached = None if force_reembed else load_emb_cache(docs)
    if cached is not None:
        embs = cached
    else:
        embs = await loop.run_in_executor(None, compute_doc_embeddings, docs)
        save_emb_cache(docs, embs)

    await loop.run_in_executor(None, init_bm25, docs)
    state.KNOWLEDGE_BASE = docs
    state.doc_embeddings = embs
    # Rebuild graph in background so /reload returns quickly
    asyncio.create_task(init_graph(docs))
    logger.info("Index rebuild complete — %d chunks (cached=%s, graph_nodes=%d)",
                len(state.KNOWLEDGE_BASE), cached is not None, len(state.KNOWLEDGE_GRAPH["nodes"]))
