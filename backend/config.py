from __future__ import annotations

import os
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

DOCS_DIR             = Path(os.getenv("DOCS_DIR", Path(__file__).parent / "docs"))
EMBED_MODEL_NAME     = os.getenv("EMBED_MODEL", "BAAI/bge-small-zh-v1.5")
RERANKER_MODEL       = os.getenv("RERANKER_MODEL", "")
DEEPSEEK_API_KEY     = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL       = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_BASE_URL    = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
MAX_CHUNK_CHARS      = int(os.getenv("MAX_CHUNK_CHARS", "600"))
# ⑦ Contextual Chunking: prepend LLM-generated context to each chunk before embedding
CONTEXTUAL_CHUNKING  = os.getenv("CONTEXTUAL_CHUNKING", "false").lower() == "true"
# ⑧ Embedding cache directory
CACHE_DIR            = Path(os.getenv("CACHE_DIR", Path(__file__).parent / "cache"))
# ⑩ Knowledge Graph (GraphRAG)
ENABLE_GRAPH         = os.getenv("ENABLE_GRAPH", "true").lower() == "true"
