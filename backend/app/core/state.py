"""
Shared Mutable Runtime State
============================
All modules that need access to the "live" RAG system state (indexed docs,
embeddings, loaded models, knowledge graph, …) import this module and read/write
its attributes directly, e.g. `state.KNOWLEDGE_BASE`.

This avoids circular imports between the retrieval/LLM/graph modules (none of
them need to import each other just to share state) and keeps `main.py` free
of global variable plumbing.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

KNOWLEDGE_BASE: List[dict]           = []
doc_embeddings: Optional[np.ndarray] = None
embed_model                          = None
cross_encoder                        = None     # BAAI/bge-reranker (optional)
bm25_index                           = None
tokenize_fn                          = None
llm_client                           = None
KNOWLEDGE_GRAPH: dict                = {"nodes": {}, "edges": {}}
