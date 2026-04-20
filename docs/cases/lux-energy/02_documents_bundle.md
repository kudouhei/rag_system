# Luxembourg Energy — Example Document Bundle (what you ingest)

This is the “what documents exist” pack you can show in an interview to prove realism.

## A. Operations & safety
- **Switching runbook** (approved procedures; versioned)
- **Safety policy** (LOTO, PPE, access control)
- **Alarm taxonomy** (SCADA/EMS alarm codes → meaning → actions)
- **Restoration playbook** (storm / equipment failure / planned cut)

## B. Incidents & learning
- **Major incident postmortems** (timeline, root cause, mitigations)
- **Near-miss reports** (safety/operational)
- **Known issues list** (recurring causes, vendor advisory mapping)

## C. Change & asset management
- **Change tickets** (planned maintenance, firmware upgrades)
- **Asset inventory** (device model, location, firmware, vendor)
- **Network topology snapshots** (access-restricted)

## D. Customer comms (approved language)
- **Outage communication templates** (EN/FR/DE; internal + external)
- **Regulator reporting templates** (incident classification, SLA impacts)

## Metadata you should capture per chunk
- `source_system` (Confluence / PDF / Ticket)
- `doc_type` (runbook / policy / postmortem / vendor_manual)
- `valid_from`, `valid_to`, `version`
- `asset_model`, `site`, `region`
- `access_level` (public/internal/critical)
- `language`

## Why it matters for RAG
Energy companies live on **versioned procedures**. Retrieval must prioritize:
- latest approved version
- procedures over informal notes
- access-controlled chunks based on role

