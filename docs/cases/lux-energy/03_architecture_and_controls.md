# Luxembourg Energy — Architecture & Controls (how you’d build it)

## Architecture (high level)
- **UI**: operator/support UI with ticket context + citations + exportable response
- **API**: FastAPI WebSocket streaming (`/ws/query`, `/ws/agent`)
- **Retrieval**: hybrid dense+BM25, optional rerank, optional GraphRAG for asset/entity linkage
- **Generation**: grounded answer prompt with strict “no-evidence → ask/decline” policy

## Key enterprise controls (what makes it hire-worthy)
### 1) Role-based access & multi-tenancy
- Require `tenant_id`, `user_id`, `user_role` in requests
- Filter documents by `access_level` and business unit

### 2) Audit logging (evidence)
- Log: request metadata + redacted query + config toggles + outcome
- Purpose: regulator evidence, incident reviews, forensic analysis

### 3) PII & sensitive data redaction
- Redact email/tokens/IBAN/long numbers before writing logs/feedback
- Keep raw content only in volatile memory when needed

### 4) Change management
- Index rebuild tied to document version updates
- “Approved procedures” have priority in ranking

### 5) Safety gates
When query indicates switching or physical operations:
- require runbook citation with procedure ID + version
- otherwise respond with escalation guidance

## Demo hooks in this repo
In this repository you can demonstrate:
- request context fields (tenant/user/role/ticket/version/env)
- audit trail (`backend/audit.jsonl`)
- feedback loop (`POST /feedback`, summarized in `GET /stats`)

