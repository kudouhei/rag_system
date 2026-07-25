from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from config import MAX_CHUNK_CHARS

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# Document Management
# ══════════════════════════════════════════════════════════════════════════════

def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> List[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) <= 1:
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_chars:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(current)
            if len(para) > max_chars:
                for i in range(0, len(para), max_chars):
                    chunks.append(para[i: i + max_chars])
                current = ""
            else:
                current = para
    if current:
        chunks.append(current)
    return chunks or [text[:max_chars]]


def load_documents_from_folder(docs_dir: Path) -> List[dict]:
    if not docs_dir.exists():
        docs_dir.mkdir(parents=True, exist_ok=True)
        logger.warning("Created docs dir: %s", docs_dir)
        return []

    docs, doc_id = [], 0
    for path in sorted(docs_dir.glob("**/*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        text = ""
        try:
            if suffix in (".txt", ".md"):
                text = path.read_text(encoding="utf-8")
            elif suffix == ".pdf":
                try:
                    import pypdf
                    reader = pypdf.PdfReader(str(path))
                    text = "\n\n".join(p.extract_text() or "" for p in reader.pages)
                except ImportError:
                    logger.warning("pypdf not installed — skipping %s", path.name)
                    continue
            else:
                continue
        except Exception as e:
            logger.warning("Cannot read %s: %s", path.name, e)
            continue

        if not text.strip():
            continue

        chunks = chunk_text(text)
        tags = [suffix.lstrip(".")]
        if path.parent != docs_dir:
            tags.append(path.parent.name)

        stat = path.stat()
        for i, chunk in enumerate(chunks):
            doc_id += 1
            title = path.stem + (f" (§{i + 1})" if len(chunks) > 1 else "")
            docs.append({
                "id":           f"doc_{doc_id:04d}",
                "title":        title,
                "content":      chunk,
                "source":       str(path.relative_to(docs_dir)),
                "tags":         tags,
                "embedding_score": 0.0,
                "bm25_score":   0.0,
                # ── Enhanced metadata for knowledge base management ──
                "word_count":   len(chunk.split()),
                "char_count":   len(chunk),
                "chunk_index":  i,
                "total_chunks": len(chunks),
                "file_size_kb": round(stat.st_size / 1024, 1),
                "mtime":        stat.st_mtime,
                "file_mtime":   datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                "indexed_at":   datetime.now(tz=timezone.utc).isoformat(),
            })

    logger.info("Loaded %d chunks from %d files in %s", len(docs), len({d["source"] for d in docs}), docs_dir)
    return docs
