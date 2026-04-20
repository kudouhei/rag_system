# Case — Luxembourg Power System Company (Utility / TSO / DSO)

## One-liner (for interview)
Build an **Operations + Customer Support Knowledge Copilot** for a Luxembourg power system company to reduce outage handling time, standardize answers, and enforce **EU/GDPR + critical infrastructure** governance with auditability.

## Why this is compelling (what the hiring team cares about)
- **High stakes**: grid reliability, safety, regulatory reporting
- **Low tolerance for hallucination**: answers must be grounded with citations + escalation rules
- **Heterogeneous data**: runbooks, incident timelines, SCADA/EMS notes, vendor manuals, change tickets
- **Multilingual**: typical mix of **EN/FR/DE** (and sometimes LU); your system’s language switch matters

## Target users
- **Grid operations (NOC / control room)**: alarms, switching, restoration
- **Field engineers**: procedures, safety steps, equipment manuals
- **Customer support**: outage FAQs, planned maintenance communications
- **Compliance**: audit, reporting, evidence

## Core scenarios
### 1) Outage / alarm triage (minutes matter)
**Input**: alarm summary + substation/feeder + timestamp + last change ticket + short log snippet  
**Output**:
- probable causes (ranked) + “what evidence supports this”
- step-by-step runbook (role-based: operator vs field)
- **safety gates** (do-not-suggest actions) + escalation to on-call
- references: runbook sections, last incident postmortem, vendor manual excerpts

### 2) Planned maintenance & switching
**Input**: planned window + topology area + device model + procedure ID  
**Output**: switching order checklist, rollback, communications template, references

### 3) Customer outage communication (consistency)
**Input**: outage region + ETA + known cause + customer segment  
**Output**: approved wording, multilingual templates, references to policy/SLAs

## Success metrics (measurable)
- **MTTA** (mean time to acknowledge) ↓ 20–40%
- **MTTR** ↓ 10–25%
- **First-contact resolution** in support ↑ 10–20%
- **Escalation rate** ↓ 15–30%
- **Grounded answer rate** (has valid citations) ≥ 95%
- **Audit completeness** (tenant/user/request logged) ≥ 99%

## Data sources (typical)
- Runbooks (procedures, safety) — Confluence/SharePoint/PDF
- Incident postmortems (timeline, root cause, action items)
- Change management tickets (ServiceNow/Jira)
- Vendor manuals (ABB/Siemens/Schneider) PDFs
- Network topology exports (non-public) — controlled access

## Risk & controls (critical infrastructure)
- **Hard rule**: no operational switching commands without explicit approved procedure reference
- **Low-confidence path**: ask clarifying questions or escalate (do not guess)
- **Data minimization**: redact PII in logs; store only necessary context

