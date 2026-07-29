from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel

# ══════════════════════════════════════════════════════════════════════════════
# Request / Response Models
# ══════════════════════════════════════════════════════════════════════════════

class ConversationTurn(BaseModel):
    role: str       # "user" | "assistant"
    content: str


class QueryRequest(BaseModel):
    query: str
    # Enterprise context (optional)
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    user_role: Optional[str] = None
    ticket_id: Optional[str] = None
    product: Optional[str] = None
    version: Optional[str] = None
    environment: Optional[str] = None
    strategy: str = "adaptive"          # vector | bm25 | hybrid | adaptive
    enable_iterative: bool = True
    enable_graph: bool = False          # ⑩ GraphRAG knowledge-graph lane
    confidence_threshold: float = 0.55
    top_k: int = 5
    language: str = "zh"               # "zh" | "en"
    history: List[ConversationTurn] = []


class FeedbackRequest(BaseModel):
    query: str
    answer: str
    rating: int                      # 1 | -1
    comment: Optional[str] = None
    doc_ids: List[str] = []
    language: Optional[str] = "zh"
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    user_role: Optional[str] = None
