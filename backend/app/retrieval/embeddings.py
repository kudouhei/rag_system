"""
Dense Retrieval — sentence-transformers (BAAI/bge)
====================================================
Embedding model lifecycle, disk-backed embedding cache (SHA-256 fingerprinted
so unchanged document sets skip re-encoding), and cosine-similarity scoring.
"""
from __future__ import annotations

import hashlib
import logging
from typing import List, Optional

import numpy as np

from app.core import state
from app.core.config import CACHE_DIR, CONTEXTUAL_CHUNKING, EMBED_MODEL_NAME

logger = logging.getLogger(__name__)


def init_embed_model() -> None:
    from sentence_transformers import SentenceTransformer
    logger.info("Loading embedding model: %s", EMBED_MODEL_NAME)
    state.embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    logger.info("Embedding model ready")


def compute_doc_embeddings(docs: List[dict]) -> np.ndarray:
    # Use `embedding_content` if set by contextual chunking, otherwise title+content
    texts = [
        d.get("embedding_content") or (d["title"] + " " + d["content"])
        for d in docs
    ]
    logger.info("Encoding %d documents…", len(texts))
    embs = state.embed_model.encode(
        texts, normalize_embeddings=True, show_progress_bar=True, batch_size=32,
    )
    return embs.astype(np.float32)

# ══════════════════════════════════════════════════════════════════════════════
# Embedding Cache — skip re-encoding unchanged document sets
# ══════════════════════════════════════════════════════════════════════════════

def doc_fingerprint(docs: List[dict]) -> str:
    """SHA-256 fingerprint of document IDs + content (first 100 chars each)."""
    raw = "|".join(
        f"{d['id']}:{d['content'][:100]}"
        for d in sorted(docs, key=lambda x: x["id"])
    )
    prefix = f"ctx={CONTEXTUAL_CHUNKING}|model={EMBED_MODEL_NAME}|"
    return hashlib.sha256((prefix + raw).encode()).hexdigest()[:20]


def load_emb_cache(docs: List[dict]) -> Optional[np.ndarray]:
    fp = doc_fingerprint(docs)
    cache_file = CACHE_DIR / f"emb_{fp}.npy"
    if cache_file.exists():
        logger.info("Embedding cache hit (%s) — skipping re-encoding", fp)
        return np.load(str(cache_file)).astype(np.float32)
    return None


def save_emb_cache(docs: List[dict], embs: np.ndarray) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fp = doc_fingerprint(docs)
    cache_file = CACHE_DIR / f"emb_{fp}.npy"
    np.save(str(cache_file), embs)
    # Remove stale caches (keep only the latest)
    for old in CACHE_DIR.glob("emb_*.npy"):
        if old != cache_file:
            old.unlink(missing_ok=True)
    logger.info("Embedding cache saved (%s, %d docs)", fp, len(docs))


def encode_text(text: str) -> np.ndarray:
    """Encode arbitrary text (with BGE retrieval prefix if applicable)."""
    if "bge" in EMBED_MODEL_NAME.lower():
        text = "为这个句子生成表示以用于检索相关文章：" + text
    return state.embed_model.encode([text], normalize_embeddings=True)[0].astype(np.float32)


def compute_embedding_scores(query_or_text: str) -> np.ndarray:
    """Cosine similarity between query (or hypothetical doc) and all indexed docs → [0, 1]."""
    q_emb = encode_text(query_or_text)
    raw = state.doc_embeddings @ q_emb          # already L2-normalised
    return ((raw + 1.0) / 2.0).astype(np.float32)
