"""Knowledge graph visualisation data endpoint."""
from __future__ import annotations

from fastapi import APIRouter

from app.core import state
from app.core.config import ENABLE_GRAPH

router = APIRouter()


@router.get("/graph")
async def get_graph_info():
    """Knowledge graph stats and top nodes/edges for visualisation."""
    nodes = state.KNOWLEDGE_GRAPH.get("nodes", {})
    edges = state.KNOWLEDGE_GRAPH.get("edges", {})

    top_nodes = sorted(nodes.items(), key=lambda x: x[1].get("freq", 0), reverse=True)[:20]
    top_edges = sorted(edges.values(), key=lambda x: x.get("weight", 0), reverse=True)[:20]

    return {
        "enabled":     ENABLE_GRAPH,
        "node_count":  len(nodes),
        "edge_count":  len(edges),
        "top_nodes": [
            {"name": n, "type": d.get("type"), "freq": d.get("freq", 0),
             "chunk_count": len(d.get("chunk_ids", []))}
            for n, d in top_nodes
        ],
        "top_edges": [
            {"source": e["source"], "target": e["target"],
             "relation": e.get("relation"), "weight": e.get("weight", 1)}
            for e in top_edges
        ],
    }
