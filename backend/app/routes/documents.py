"""Document management endpoints — list, preview, upload, delete, reload."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core import state
from app.core.config import DOCS_DIR
from app.pipeline.indexing import rebuild_index

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/docs_list")
async def docs_list():
    return [
        {"id": d["id"], "title": d["title"], "source": d.get("source", ""), "tags": d["tags"]}
        for d in state.KNOWLEDGE_BASE
    ]


@router.get("/docs/preview")
async def preview_doc(source: str, max_chars: int = 6000):
    """
    Preview a knowledge base file by its `source` (relative filename under DOCS_DIR).
    Returns chunk metadata and a text preview assembled from the indexed chunks.
    """
    if not source:
        raise HTTPException(400, "source is required")
    # Basic traversal guard (source is always relative paths in our index)
    if ".." in source or source.startswith(("/", "\\")):
        raise HTTPException(400, "Invalid source")

    chunks = [d for d in state.KNOWLEDGE_BASE if d.get("source") == source]
    if not chunks:
        raise HTTPException(404, f"Not found: {source}")

    chunks_sorted = sorted(chunks, key=lambda d: int(d.get("chunk_index", 0)))
    total_chunks = int(chunks_sorted[0].get("total_chunks", len(chunks_sorted)))
    file_mtime = chunks_sorted[0].get("file_mtime", "")
    file_size_kb = chunks_sorted[0].get("file_size_kb", 0)
    tags = chunks_sorted[0].get("tags", [])

    # Assemble a preview from chunk contents
    preview_parts = []
    used = 0
    for c in chunks_sorted:
        txt = c.get("content", "")
        if not txt:
            continue
        remain = max_chars - used
        if remain <= 0:
            break
        piece = txt[:remain]
        preview_parts.append(piece)
        used += len(piece)

    preview_text = ("\n\n---\n\n".join(preview_parts)).strip()
    truncated = used >= max_chars

    return {
        "source": source,
        "title": Path(source).stem,
        "tags": tags,
        "file_mtime": file_mtime,
        "file_size_kb": file_size_kb,
        "total_chunks": total_chunks,
        "chunks": [
            {
                "id": c.get("id"),
                "title": c.get("title"),
                "chunk_index": c.get("chunk_index"),
                "char_count": c.get("char_count"),
                "word_count": c.get("word_count"),
            }
            for c in chunks_sorted
        ],
        "preview_text": preview_text,
        "truncated": truncated,
        "max_chars": max_chars,
    }


@router.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload one or more documents (.txt / .md / .pdf) to the knowledge base.
    All files are saved first, then a single index rebuild is triggered.
    """
    allowed = {".txt", ".md", ".pdf"}
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for file in files:
        suffix = Path(file.filename).suffix.lower()
        if suffix not in allowed:
            results.append({
                "filename": file.filename,
                "status":   "error",
                "error":    f"Unsupported type '{suffix}'. Allowed: {sorted(allowed)}",
            })
            continue

        try:
            content = await file.read()
            dest    = DOCS_DIR / Path(file.filename).name   # strip any path traversal
            dest.write_bytes(content)
            logger.info("Uploaded: %s (%d bytes)", dest.name, len(content))
            results.append({
                "filename":   dest.name,
                "status":     "ok",
                "size_bytes": len(content),
                "size_kb":    round(len(content) / 1024, 1),
            })
        except Exception as e:
            logger.error("Upload failed for %s: %s", file.filename, e)
            results.append({"filename": file.filename, "status": "error", "error": str(e)})

    ok_count = sum(1 for r in results if r["status"] == "ok")
    if ok_count > 0:
        asyncio.create_task(rebuild_index())   # single rebuild after all files saved

    return {
        "uploaded": ok_count,
        "errors":   len(results) - ok_count,
        "files":    results,
    }


@router.delete("/docs/{filename}")
async def delete_document(filename: str):
    """
    Delete a document from the knowledge base and rebuild the index.
    Only files inside DOCS_DIR can be deleted (no path traversal).
    """
    # Resolve and validate path — prevent directory traversal
    target = (DOCS_DIR / filename).resolve()
    if not str(target).startswith(str(DOCS_DIR.resolve())):
        raise HTTPException(400, "Invalid filename")
    if not target.exists():
        raise HTTPException(404, f"File not found: {filename}")

    target.unlink()
    logger.info("Deleted document: %s", filename)
    asyncio.create_task(rebuild_index())
    return {"status": "ok", "deleted": filename}


@router.post("/reload")
async def reload_index(force: bool = False):
    """
    Trigger index rebuild.
    - force=false (default): use embedding cache if docs unchanged
    - force=true: always re-embed (needed after changing EMBED_MODEL or CONTEXTUAL_CHUNKING)
    """
    asyncio.create_task(rebuild_index(force_reembed=force))
    return {"status": "rebuilding", "force_reembed": force,
            "message": "Index rebuild started in background"}
