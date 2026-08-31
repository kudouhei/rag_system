# ⚖ Regulatory Document Intelligence RAG

> A fund / asset-management regulatory compliance RAG prototype — retrieve
> and analyse regulatory documents (regulations, circulars, enforcement
> notices) with citation-grade answers and scenario-based compliance checks.

Hybrid Retrieval (BGE + BM25) · Iterative Reflection · GraphRAG · Cross-Encoder
Rerank · Agentic Routing · **Compliance Check** · RAGAS

---

## What this is

This system retrieves and analyses **fund/asset-management regulatory
documents** — the kind of corpus a compliance, legal, or product team at an
asset manager would work with: regulations, guidance circulars, and
enforcement notices.

The sample knowledge base (`backend/docs/`) contains **19 synthetic but
realistic regulatory documents** (regulations, an amendment, a guidance
circular, and an enforcement notice) issued by a fictional regulator — the
**Meridian Financial Conduct Authority (MFCA)** of the **Republic of
Meridia** — covering: prospectus disclosure, NAV calculation, AML/KYC,
custody, liquidity risk management, redemption gates/side pockets, marketing
rules, ESG/sustainability disclosure, cross-border passporting, fund
governance, conflicts of interest, outsourcing, operational resilience,
complaints handling, best execution, and periodic regulatory filing.

> These documents are **synthetic demo content**, not real regulation — the
> fictional regulator/jurisdiction avoids misrepresenting any real legal
> text while still exercising realistic regulatory structure (articles,
> regulation numbers, effective dates, amendments, enforcement cases).

Each document carries structured regulatory metadata (YAML frontmatter):
`regulation_number`, `issuing_authority`, `jurisdiction`, `document_type`
(regulation | circular | enforcement_notice | amendment), `product_type`,
`risk_level`, `effective_date`, `status`, `version`. This metadata flows
through retrieval, citations, filtering, and the compliance-check feature.

Bring your own documents by dropping `.md` files (with or without the
frontmatter block) — or `.txt` / `.pdf` — into `backend/docs/`.

---

## Architecture

```
Frontend :3000  ──WebSocket/REST──►  Backend :8000
                                       │
                    ws/query      ──► RAG Pipeline (retrieve → reflect → rerank → generate)
                    ws/agent      ──► Router → direct | rag | tools | multi-step
                    /compliance_check ──► Retrieve clauses → LLM compliance verdict + citations
                                       │
                    Index: frontmatter → chunk → embed (local BGE-en) → BM25 → [graph]
                    Docs: backend/docs/  (.md with regulatory frontmatter, .txt, .pdf)
```

**RAG flow:** multi-strategy retrieval (up to 3 rounds, query rewritten on low
confidence) → rerank (enable via `RERANKER_MODEL`) → LLM answer generation
with regulation-number/article citations → RAGAS-style online proxy metrics.

**Compliance Check flow:** describe a business/product scenario → hybrid
retrieval of relevant clauses (optionally filtered by jurisdiction / product
type / regulation number) → LLM assesses the scenario against each retrieved
requirement → structured verdict (`compliant` / `non_compliant` /
`needs_review`) per requirement, with citations.

---

## Quick Start

```bash
cp backend/.env.example backend/.env
# Set DEEPSEEK_API_KEY (retrieval works without it; answer generation,
# query rewriting, and compliance verdicts require an LLM key)

chmod +x start.sh && ./start.sh
```

- Frontend: http://localhost:3000
- API docs: http://localhost:8000/docs

Documents in `backend/docs/` are indexed automatically on startup. Hot
reload after adding/editing files:

```bash
curl -X POST http://localhost:8000/reload
```

**Requirements:** Python 3.9+ · Node 18+ · RAM ≥ 4 GB

---

## Configuration

See `backend/.env.example`. Key options:

| Variable | Description |
|----------|--------------|
| `DEEPSEEK_API_KEY` | LLM answer generation, query rewriting, compliance verdicts |
| `EMBED_MODEL` | Default `BAAI/bge-small-en-v1.5` (local embedding, English corpus) |
| `RERANKER_MODEL` | Optional, e.g. `BAAI/bge-reranker-base` |
| `MAX_CHUNK_CHARS` | Chunk size, default 900 (regulatory articles run longer than IT runbooks) |
| `CONTEXTUAL_CHUNKING` | LLM adds per-chunk context at index time |
| `DOCS_DIR` | Document directory, default `backend/docs/` |

Embeddings are computed locally. Audit/feedback logs redact query/answer
text.

### Adding regulatory documents

Markdown files may start with a YAML frontmatter block:

```markdown
---
regulation_number: REG-FM-105
title: Liquidity Risk Management for Open-Ended Funds
issuing_authority: Meridian Financial Conduct Authority (MFCA)
jurisdiction: Republic of Meridia
document_type: regulation      # regulation | circular | enforcement_notice | amendment
product_type: [mutual_fund, money_market_fund]
risk_level: high                # low | medium | high
effective_date: 2021-01-01
status: in_force                # in_force | superseded | proposed
version: "1.0"
---

# Article 1 — ...
```

This metadata is surfaced in citations, `/stats`, `/inventory`, and the
Compliance Check filters. Files without frontmatter (or `.txt`/`.pdf`) are
indexed normally, just without the structured fields.

---

## API

### WebSocket

| Path | Purpose |
|------|---------|
| `/ws/query` | Standard streaming RAG pipeline |
| `/ws/agent` | Smart routing (retrieval / direct answer / tools / multi-step) |

Request fields: `query`, `strategy` (`adaptive` \| `hybrid` \| `vector` \|
`bm25`), `enable_iterative`, `enable_graph`, `confidence_threshold`, `top_k`,
`language`, `history`, plus optional regulatory filters: `jurisdiction`,
`product_type`, `regulation_number`, `document_type`.

Key events: `pipeline_complete` (answer, documents, RAGAS metrics),
`answer_token`, `doc_scored`, `query_rewrite`.

### REST

| Method | Path | Description |
|--------|------|--------------|
| `GET` | `/health` | Service and model status |
| `GET` | `/stats` | Knowledge base statistics (incl. regulatory metadata) |
| `GET` | `/inventory` | Per-document usage/feedback analytics |
| `GET` | `/docs_list` | List indexed document chunks |
| `POST` | `/upload` | Upload documents |
| `DELETE` | `/docs/{filename}` | Delete a document |
| `POST` | `/reload` | Rebuild index (`?force=true` ignores the embedding cache) |
| `POST` | `/feedback` | Submit user feedback (±1) |
| `POST` | `/compliance_check` | Scenario-based compliance assessment with citations |

`POST /compliance_check` request body:

```json
{
  "scenario": "We plan to launch a money market fund offering daily redemption with no swing pricing mechanism.",
  "jurisdiction": null,
  "product_type": "money_market_fund",
  "regulation_number": null,
  "top_k": 8,
  "language": "en"
}
```

Response: `overall_status`, `summary`, and a list of `findings`, each with
`requirement`, `citation` (e.g. `"REG-FM-105A Article 2"`), `source`,
`assessment`, and `rationale`.

---

## MCP

Expose the knowledge base to Claude / Cursor or any MCP client:

```bash
python backend/mcp_server.py              # stdio
python backend/mcp_server.py --http --port 8001
```

Tools: `search_knowledge_base` · `retrieve_documents` · `list_documents` ·
`get_kb_stats`.

Client configuration example: `mcp_config_example.json`.

---

## Project Structure

```
rag_system/
├── backend/main.py               # API + RAG / Agent / Compliance pipelines
├── backend/mcp_server.py
├── backend/docs/                 # Synthetic fund regulatory knowledge base (19 docs)
├── backend/docs_legacy_it_kb/    # Previous IT/security sample docs (kept for reference)
├── backend/app/pipeline/compliance.py   # Compliance Check pipeline
├── backend/app/routes/compliance.py     # POST /compliance_check
├── frontend/src/App.jsx
├── frontend/src/components/tabs/ComplianceTab.jsx
├── start.sh
└── mcp_config_example.json
```

**Tech stack:** FastAPI · sentence-transformers (BGE) · rank-bm25 · DeepSeek ·
React · Vite
