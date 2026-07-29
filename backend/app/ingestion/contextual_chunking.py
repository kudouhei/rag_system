"""
Contextual Chunking  (Anthropic, 2024)
=========================================
"""
from __future__ import annotations

import asyncio
import logging
from typing import List

from app.core import state
from app.core.config import CONTEXTUAL_CHUNKING, DOCS_DIR
from app.llm.client import llm_call

logger = logging.getLogger(__name__)


async def contextualize_chunks(docs: List[dict], lang: str = "zh") -> List[dict]:
    """
    Anthropic Contextual Retrieval (2024):
    For each chunk, use LLM to generate a 1-2 sentence document-level context
    and store it as `embedding_content`.  The original `content` (shown to users)
    is unchanged; only the embedded text is enriched.

    This significantly reduces the "out-of-context chunk" problem, e.g.:
      Raw chunk    : "Sales increased by 23% compared to last quarter."
      With context : "This passage is from Q3 2024 Financial Report, discussing
                      revenue performance. Sales increased by 23%..."
    """
    if not state.llm_client or not CONTEXTUAL_CHUNKING:
        return docs

    # Preload full text per source file (up to 2000 chars for context window)
    source_cache: dict = {}
    for doc in docs:
        src = doc.get("source", "")
        if src and src not in source_cache:
            try:
                source_cache[src] = (DOCS_DIR / src).read_text(encoding="utf-8")[:2000]
            except Exception:
                source_cache[src] = ""

    logger.info("Contextual chunking: enriching %d chunks with LLM context…", len(docs))

    sys_prompt = (
        "You are a document analyst. Given a document excerpt and a chunk from it, "
        "write 1-2 sentences of context explaining where this chunk sits in the document "
        "and what it's about. Output ONLY the context sentences."
    ) if lang == "en" else (
        "你是文档分析专家。给定文档摘录和其中的一个段落，"
        "用1-2句话说明该段落在文档中的位置和主题。只输出上下文说明。"
    )

    enriched, ok_count = [], 0
    for doc in docs:
        full_text = source_cache.get(doc.get("source", ""), "")
        ctx = ""
        if full_text:
            ctx = await llm_call(
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user",   "content":
                        f"Document (excerpt):\n{full_text[:1200]}\n\nChunk:\n{doc['content'][:400]}"},
                ],
                max_tokens=80,
                temperature=0.1,
            )
            await asyncio.sleep(0.05)   # gentle rate-limiting

        d = doc.copy()
        if ctx:
            d["embedding_content"] = f"{ctx}\n\n{doc['content']}"
            d["context_added"] = True
            ok_count += 1
        enriched.append(d)

    logger.info("Contextual chunking: %d/%d chunks enriched", ok_count, len(docs))
    return enriched
