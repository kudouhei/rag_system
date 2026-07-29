from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── Enterprise-style audit & feedback (JSONL) ──────────────────────────────────
# backend/app/core/audit.py -> backend/ is 3 levels up; keep the log files at
# the backend root regardless of how the app/ package is organised internally.
BASE_DIR = Path(__file__).resolve().parent.parent.parent
AUDIT_FILE = BASE_DIR / "audit.jsonl"
FEEDBACK_FILE = BASE_DIR / "feedback.jsonl"
_jsonl_lock = threading.Lock()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, ensure_ascii=False)
    with _jsonl_lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


_REDACT_PATTERNS = [
    # Email
    (re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I), "[REDACTED_EMAIL]"),
    # Bearer tokens / API keys (very rough)
    (re.compile(r"\bBearer\s+[A-Za-z0-9._-]{16,}\b"), "Bearer [REDACTED_TOKEN]"),
    (re.compile(r"\bsk-[A-Za-z0-9]{16,}\b"), "[REDACTED_API_KEY]"),
    # IBAN (Luxembourg starts with LU; keep generic)
    (re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"), "[REDACTED_IBAN]"),
    # Long digit sequences (cards / account ids)
    (re.compile(r"\b\d{12,19}\b"), "[REDACTED_NUMBER]"),
]


def redact_text(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    out = s
    for rx, repl in _REDACT_PATTERNS:
        out = rx.sub(repl, out)
    return out
