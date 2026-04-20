# Bank — Controls, Model Risk, and Evaluation

## Controls (what makes it “bank-grade”)
### 1) Model Risk Management (MRM) alignment
- Define intended use, limitations, and fallback behavior
- Maintain versioned prompt + retrieval configs
- Evidence: offline eval sets + change approvals

### 2) Role-based access & least privilege
- Roles: `contact_center`, `branch_advisor`, `ops_payments`, `compliance`
- Retrieval is filtered by access level + product line
- Some documents: “read-only for compliance” or “internal only”

### 3) PII handling (GDPR)
Principle: **do not store PII in logs**.
- Redact patterns before persistence (emails, IBAN, long account numbers)
- Store only redacted query/answer in feedback & audit trails

### 4) Prompt injection & unsafe tool use
- Treat user text as untrusted; prefer retrieval citations
- If asked for prohibited actions (e.g., “bypass KYC”), refuse and cite policy

## Evaluation plan (practical)
### Offline
- Dataset: 200–2,000 historical tickets + policy Q&A pairs
- Metrics:
  - retrieval hit@k for correct policy section
  - groundedness / citation correctness
  - refusal correctness (when policy requires escalation)

### Online
- A/B rollout by team
- Metrics:
  - AHT, escalation, QA findings, customer satisfaction
  - “time-to-correct-answer” and “reopen rate”

## What you can demonstrate in this repo
- Feedback loop: `POST /feedback` writes JSONL; `/stats` exposes satisfaction rate
- Audit trail: websocket queries recorded in `backend/audit.jsonl`
- Enterprise context: tenant/user/role/ticket/version/env fields from UI to backend

