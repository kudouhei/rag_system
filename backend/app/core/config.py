from __future__ import annotations

import os
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
# backend/app/core/config.py -> backend/ is 3 levels up.
_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent

DOCS_DIR             = Path(os.getenv("DOCS_DIR", _BACKEND_DIR / "docs"))
# Default corpus is English-language regulatory/fund documents — use an English
# BGE model. Set EMBED_MODEL=BAAI/bge-small-zh-v1.5 (or similar) for CJK corpora.
EMBED_MODEL_NAME     = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
RERANKER_MODEL       = os.getenv("RERANKER_MODEL", "")
DEEPSEEK_API_KEY     = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL       = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_BASE_URL    = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

# Generic LLM configuration.
#
# The legacy DEEPSEEK_* variables remain as fallbacks so existing local
# environments continue working while the application moves toward a
# provider-neutral interface.
LLM_PROVIDER = os.getenv(
    "LLM_PROVIDER",
    "deepseek",
).strip().lower()

LLM_API_KEY = os.getenv(
    "LLM_API_KEY",
    DEEPSEEK_API_KEY,
)

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    DEEPSEEK_MODEL,
)

LLM_BASE_URL = os.getenv(
    "LLM_BASE_URL",
    DEEPSEEK_BASE_URL,
).rstrip("/")
# Regulatory articles/clauses run longer than IT runbook paragraphs.
MAX_CHUNK_CHARS      = int(os.getenv("MAX_CHUNK_CHARS", "900"))
# ⑦ Contextual Chunking: prepend LLM-generated context to each chunk before embedding
CONTEXTUAL_CHUNKING  = os.getenv("CONTEXTUAL_CHUNKING", "false").lower() == "true"
# ⑧ Embedding cache directory
CACHE_DIR            = Path(os.getenv("CACHE_DIR", _BACKEND_DIR / "cache"))
# ⑩ Knowledge Graph (GraphRAG)
ENABLE_GRAPH         = os.getenv("ENABLE_GRAPH", "true").lower() == "true"

TRAINING_DEBUG_ENABLED = os.getenv("TRAINING_DEBUG_ENABLED", "false").lower() == "true"