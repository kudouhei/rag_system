"""WebSocket endpoints — streaming RAG pipeline."""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.audit import AUDIT_FILE, _append_jsonl, _utc_now_iso, redact_text
from app.core.schemas import QueryRequest
from app.pipeline.agent_pipeline import run_agentic_pipeline
from app.pipeline.rag_pipeline import run_rag_pipeline

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/query")
async def websocket_query(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            request = QueryRequest(**json.loads(data))
            _append_jsonl(AUDIT_FILE, {
                "ts": _utc_now_iso(),
                "type": "ws_query",
                "client": getattr(websocket, "client", None).host if getattr(websocket, "client", None) else None,
                "tenant_id": request.tenant_id,
                "user_id": request.user_id,
                "user_role": request.user_role,
                "jurisdiction": request.jurisdiction,
                "product_type": request.product_type,
                "regulation_number": request.regulation_number,
                "document_type": request.document_type,
                "query": redact_text(request.query),
                "strategy": request.strategy,
                "enable_iterative": request.enable_iterative,
                "enable_graph": request.enable_graph,
            })
            await run_rag_pipeline(websocket, request)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error("WebSocket error: %s", e)
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
        except Exception:
            pass


@router.websocket("/ws/agent")
async def websocket_agent(websocket: WebSocket) -> None:
    """Agentic RAG pipeline with intelligent routing (direct / rag / realtime / complex)."""
    await websocket.accept()
    try:
        while True:
            data    = await websocket.receive_text()
            request = QueryRequest(**json.loads(data))
            _append_jsonl(AUDIT_FILE, {
                "ts": _utc_now_iso(),
                "type": "ws_agent",
                "client": getattr(websocket, "client", None).host if getattr(websocket, "client", None) else None,
                "tenant_id": request.tenant_id,
                "user_id": request.user_id,
                "user_role": request.user_role,
                "jurisdiction": request.jurisdiction,
                "product_type": request.product_type,
                "regulation_number": request.regulation_number,
                "document_type": request.document_type,
                "query": redact_text(request.query),
            })
            await run_agentic_pipeline(websocket, request)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error("Agent WebSocket error: %s", e)
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
        except Exception:
            pass
