# Case — Bank (Retail + Corporate) AI Assistant with RAG Governance

## One-liner (for interview)
Build a **front-to-back Banking Knowledge Assistant** that helps contact center + branch staff + ops teams answer questions with **policy-grounded citations**, reduces compliance risk, and improves resolution time across products (cards, loans, payments, AML/KYC).

## Why banks hire for this
- **Compliance-first** (GDPR, AML, internal policy, model risk management)
- Must be **auditable**, role-based, and safe against prompt injection
- Multisource knowledge: policy PDFs, procedures, product specs, regulator notes, internal Q&A

## Target users
- Contact center (L1/L2)
- Branch advisors
- Back-office ops (payments, chargebacks, disputes)
- Compliance / risk (policy interpretation with citations)

## Core scenarios (high-value)
### 1) Customer inquiry → “answer + next best action”
Input: customer question + product + jurisdiction + channel  
Output:
- answer with **citations to policy/product docs**
- steps checklist (what to ask, what to verify)
- “cannot answer” gates if it requires personal data access

### 2) Payments & disputes triage (high volume)
Input: transaction type + error code + network (SEPA/SWIFT/cards)  
Output: root-cause candidates + operational steps + references

### 3) AML/KYC support (strict guardrails)
Input: scenario summary (no PII) + product + risk category  
Output: which policy applies + required evidence list + escalation rules + citations

## Success metrics
- **AHT** (average handling time) ↓ 15–35%
- **Compliance QA findings** ↓ 30–60%
- **Correct routing** to L2/ops ↑ 10–20%
- **Grounded answer rate** ≥ 97%
- **Unsafe answer rate** (no-citation, policy conflict) ≈ 0%

## Data sources (typical)
- Product & fees documentation (versioned)
- Operating procedures (chargebacks, disputes, payment returns)
- Internal policy (AML/KYC, data handling, complaints)
- Regulatory guidance summaries (curated)
- Historical resolved cases (sanitized)

