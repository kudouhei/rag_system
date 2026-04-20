# AI Research Scientist — Talk Track (Energy & Banking RAG)

This document gives you a **3-minute** and **10-minute** interview narrative tailored to an **AI Research Scientist** role.

---

## 3-minute version (high impact)

### Problem
In regulated industries like **power utilities** and **banks**, “chatbot-style” LLMs are not acceptable because:
- hallucinations have **safety/compliance** consequences
- answers must be **auditable** and **version-correct**
- user requests carry **sensitive data** constraints (GDPR, critical infrastructure)

### My approach
I built a **governed RAG system** that treats the LLM as a **reasoning layer** on top of:
- **hybrid retrieval** (dense + BM25) for robustness to vocabulary gaps
- **iterative retrieval** (ReAct-style) to recover when initial recall is low
- **optional reranking** to improve precision at top-k
- **multi-turn memory** and enterprise context (tenant/user/role/ticket/version/env)

### Research contribution (what I investigated)
Instead of “does RAG work?”, I focused on *scientifically testable questions*:
- **When does iterative retrieval help**, and when does it overfit / drift?
- How do **HyDE** and **reranking** interact under multilingual + domain shift?
- Can we predict **unsafe generation** and enforce **abstention** using retrieval signals?
- How do we measure **citation correctness** and **policy-consistency** beyond BLEU-style metrics?

### Governance & safety
I added enterprise-grade controls:
- **audit logging** of tenant/user/context + redaction before persistence
- **feedback loop** stored as JSONL and summarized for operations
- clear low-confidence behavior: ask clarifying questions or refuse/escalate

### Results (what to say)
I demonstrate a measurable improvement in:
- grounded answer rate, citation validity, and reduced unsafe responses
- operational KPIs like handling time / escalation rate (in realistic workflows)

---

## 10-minute version (deep + research-oriented)

### 1) Domain framing (1 min)
I focused on **Luxembourg utility** and **banking** because they stress-test RAG:
- multilingual, versioned procedures, safety-critical decisions, auditability

### 2) System + method (2–3 min)
Pipeline design:
- retrieval: dense + BM25 fusion (+ GraphRAG for entity linkage when needed)
- reflection loop: detect low-confidence top score → rewrite query → re-retrieve
- generation: strict instruction “no evidence → say unknown / ask / escalate”

Signals I treat as research variables:
- retrieval confidence distribution (top-1, margin, entropy)
- iteration count and rewrite distance
- citation overlap and contradiction checks

### 3) Evaluation protocol (3–4 min)
I propose a bank/utility-appropriate eval beyond generic QA:

**Datasets**
- historical tickets/incidents (sanitized)
- policy/runbook Q&A with known ground truth citations
- multilingual paraphrases (EN/FR/DE) for the same intents

**Metrics**
- retrieval: hit@k for correct clause/procedure section
- generation: groundedness, citation correctness, refusal correctness
- safety: leakage rate of sensitive patterns, prompt injection success rate
- ops: time-to-resolution proxy, escalation recommendation correctness

**Ablations**
- dense only vs BM25 only vs hybrid
- single-pass vs iterative retrieval (vary max_iters)
- HyDE on/off
- rerank on/off
- citation-required decoding (reject if no citation)

### 4) Key findings & hypotheses (2 min)
What I expect and how I test it:
- Iterative retrieval improves recall on “vocabulary mismatch” queries but may drift on ambiguous intents → add stop criteria (confidence gain threshold)
- HyDE helps with sparse KB but can introduce stylistic bias → constrain with domain templates + shorter HyDE outputs
- Reranking improves top-k precision but can hurt latency → dynamic rerank only when fusion margin is small

### 5) What I’d research next (1 min)
- Train a lightweight **risk classifier** predicting unsafe/low-grounding responses from retrieval signals
- Explore **contrastive evaluation** for citation correctness (pairwise: correct vs near-miss clause)
- Study multilingual embedding alignment + domain adaptation for EN/FR/DE in Luxembourg context

---

## Q&A ammo (short answers)

### “What’s novel here?”
Not the existence of RAG, but a **testable, governed RAG** where:
- retrieval signals become **risk predictors**
- the system is evaluated on **citation correctness** and **refusal behavior**

### “How do you prevent hallucinations?”
I treat hallucination as a *system design + measurement* problem:
- enforce evidence-first prompts
- abstain/escalate when retrieval confidence is weak
- evaluate citation validity directly, not just answer similarity

### “How would you publish this?”
Paper angle: **Risk-aware iterative RAG** for regulated domains with a benchmark for
citations + refusal correctness + multilingual domain shift.

