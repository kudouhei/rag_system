from __future__ import annotations

from fastapi import APIRouter

from . import compliance as compliance_routes
from . import documents as documents_routes
from . import feedback as feedback_routes
from . import graph as graph_routes
from . import health as health_routes
from . import websocket as websocket_routes

router = APIRouter()
router.include_router(websocket_routes.router)
router.include_router(health_routes.router)
router.include_router(documents_routes.router)
router.include_router(feedback_routes.router)
router.include_router(graph_routes.router)
router.include_router(compliance_routes.router)
