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
    # Regulatory analyst context (optional) — used for audit trail + light
    # query augmentation, and as candidate filters for compliance workflows.
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    user_role: Optional[str] = None
    jurisdiction: Optional[str] = None       # e.g. "Republic of Meridia"
    product_type: Optional[str] = None       # e.g. "money_market_fund"
    regulation_number: Optional[str] = None  # e.g. "REG-FM-105"
    document_type: Optional[str] = None      # regulation | circular | enforcement_notice | amendment
    strategy: str = "adaptive"          # vector | bm25 | hybrid | adaptive
    enable_iterative: bool = True
    enable_graph: bool = False          # ⑩ GraphRAG knowledge-graph lane
    confidence_threshold: float = 0.55
    top_k: int = 5
    language: str = "en"               # "en" | "zh"
    history: List[ConversationTurn] = []


class FeedbackRequest(BaseModel):
    query: str
    answer: str
    rating: int                      # 1 | -1
    comment: Optional[str] = None
    doc_ids: List[str] = []
    language: Optional[str] = "en"
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    user_role: Optional[str] = None


class ComplianceCheckRequest(BaseModel):
    """A compliance analyst describes a product/business scenario; the system
    retrieves relevant regulatory clauses and produces a structured compliance
    assessment with citations."""
    scenario: str
    jurisdiction: Optional[str] = None
    product_type: Optional[str] = None
    regulation_number: Optional[str] = None   # focus the check on one specific regulation
    top_k: int = 8
    language: str = "en"


class ComplianceFinding(BaseModel):
    requirement: str          # short label for the specific obligation assessed
    citation: str              # e.g. "REG-FM-105 Article 3"
    source: str                 # source filename
    assessment: str             # compliant | non_compliant | uncertain | not_applicable
    rationale: str


class ComplianceCheckResponse(BaseModel):
    scenario: str
    overall_status: str          # compliant | non_compliant | needs_review
    summary: str
    findings: List[ComplianceFinding] = []
    elapsed_seconds: float = 0.0
