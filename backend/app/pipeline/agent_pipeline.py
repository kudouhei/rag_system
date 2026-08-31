"""
Agentic RAG — Router Agent + Tools + Orchestrator
====================================================
Routes each query to the most efficient execution path:
  direct   → Direct LLM answer (no retrieval overhead)
  rag      → Full Adaptive RAG pipeline (delegates to rag_pipeline.run_rag_pipeline)
  realtime → Tool calls: datetime / calculator / web_search
  complex  → Multi-step decomposition → per-subtask RAG → synthesis
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import List

from fastapi import WebSocket

from app.core import state
from app.core.audit import AUDIT_FILE, _append_jsonl, _utc_now_iso, redact_text
from app.core.config import DEEPSEEK_MODEL
from app.core.schemas import QueryRequest
from app.llm.client import llm_call, llm_stream_answer
from app.pipeline.rag_pipeline import query_rag, run_rag_pipeline
from app.pipeline.tools import tool_calculator, tool_datetime, tool_web_search

logger = logging.getLogger(__name__)

# ── Router system prompts ──────────────────────────────────────────────────────

_ROUTER_SYS: dict = {
    "zh": (
        "你是智能问题路由器。将用户查询分配到最合适的路由（返回严格JSON，无多余文字）。\n\n"
        "路由类型：\n"
        '• "direct"   — 通用知识、简单定义、创意写作、编程问题（LLM 可直接回答）\n'
        '• "rag"      — 需查询基金监管法规/合规文档知识库才能回答\n'
        '• "realtime" — 需实时数据：当前时间/日期、数学计算、网络搜索\n'
        '• "complex"  — 需拆分为多个子问题才能完整回答的复杂合规分析题\n\n'
        "返回格式：\n"
        '{"route":"direct|rag|realtime|complex","reason":"路由原因（10字以内）",'
        '"sub_queries":["子问题1","子问题2"],"tools":["datetime","calculator","web_search"]}\n\n'
        "注：sub_queries 仅 complex 时填写；tools 仅 realtime 时填写；其余为空数组"
    ),
    "en": (
        "You are an intelligent query router. Classify the query into one route "
        "(return strict JSON, no extra text).\n\n"
        "Routes:\n"
        '• "direct"   — General knowledge, definitions, creative/coding tasks (LLM answers directly)\n'
        '• "rag"      — Questions about fund/asset-management regulatory documents or compliance rules\n'
        '• "realtime" — Needs real-time data: current time/date, math calculation, web search\n'
        '• "complex"  — Needs decomposition into multiple sub-questions for a complete compliance analysis\n\n'
        "Return format:\n"
        '{"route":"direct|rag|realtime|complex","reason":"brief reason (≤8 words)",'
        '"sub_queries":["sub-q 1","sub-q 2"],"tools":["datetime","calculator","web_search"]}\n\n'
        "Note: sub_queries only for complex; tools only for realtime; others empty arrays"
    ),
}


async def route_query(query: str, language: str = "en") -> dict:
    """Query router with fast heuristics + LLM fallback (with timeout)."""
    import re as _re
    default = {
        "route": "rag",
        "reason": "默认路由" if language == "zh" else "default",
        "sub_queries": [], "tools": [],
    }

    q = (query or "").strip()
    ql = q.lower()

    # ── Fast heuristic routing (ms-level) ────────────────────────────────────
    # Realtime: calculator / datetime / explicit "today" / math expressions
    if _re.search(r"[\d]+\s*[\+\-\*\/\^]\s*[\d]+", q) or any(k in ql for k in ["计算", "calculator", "what time", "time now", "现在几点", "当前时间", "today", "日期"]):
        return {"route": "realtime", "reason": "公式/时间工具" if language == "zh" else "math/time tool",
                "sub_queries": [], "tools": ["calculator" if _re.search(r"[\d]+\s*[\+\-\*\/\^]\s*[\d]+", q) else "datetime"]}

    # Fund/regulatory KB intents (prospectus, NAV, AML/KYC, custody, liquidity, etc.)
    kb_keywords = [
        "prospectus", "nav", "valuation", "aml", "kyc", "custody", "custodian",
        "depositary", "liquidity", "redemption", "gate", "side pocket",
        "marketing", "advertising", "esg", "sustainability", "sfdr",
        "passporting", "cross-border", "governance", "conflict of interest",
        "outsourcing", "business continuity", "complaint", "best execution",
        "reporting", "filing", "regulation", "regulatory", "compliance",
        "fund manager", "money market fund", "mfca", "reg-fm", "circular",
        "enforcement", "招募说明书", "净值", "估值", "反洗钱", "托管",
        "流动性", "赎回", "合规", "监管", "基金",
    ]
    if any(k in ql for k in kb_keywords):
        return {"route": "rag", "reason": "命中监管知识库关键词" if language == "zh" else "regulatory KB keywords",
                "sub_queries": [], "tools": []}

    # If LLM is unavailable, stop here.
    if not state.llm_client:
        return default

    # ── LLM router fallback (timeout + safe degrade) ─────────────────────────
    try:
        raw = await asyncio.wait_for(
            llm_call(
                messages=[
                    {"role": "system", "content": _ROUTER_SYS.get(language, _ROUTER_SYS["zh"])},
                    {"role": "user",   "content": query},
                ],
                max_tokens=200,
                temperature=0.1,
            ),
            timeout=1.6,
        )
    except Exception:
        return default
    if not raw:
        return default

    try:
        m = _re.search(r'\{.*?\}', raw, _re.DOTALL)
        if m:
            p = json.loads(m.group())
            return {
                "route":       p.get("route", "rag"),
                "reason":      p.get("reason", ""),
                "sub_queries": p.get("sub_queries", []) or [],
                "tools":       p.get("tools", []) or [],
            }
    except Exception as e:
        logger.warning("Router parse error: %s — raw=%r", e, raw[:200])
    return default


# ── Agentic Pipeline Orchestrator ─────────────────────────────────────────────

async def run_agentic_pipeline(ws: WebSocket, req: QueryRequest) -> None:
    """
    Route query to the optimal execution path:
      direct   → Direct LLM answer (no retrieval overhead)
      rag      → Existing Adaptive RAG pipeline (full feature set)
      realtime → Tool calls: datetime / calculator / web_search
      complex  → Multi-step decomposition → per-subtask RAG → synthesis
    """
    t0   = time.time()
    lang = req.language or "en"

    # ── Step 1: Route ──────────────────────────────────────────────────────
    await ws.send_text(json.dumps({
        "type":    "agent_routing",
        "message": "分析问题类型…" if lang == "zh" else "Analyzing query intent…",
    }))
    route_info = await route_query(req.query, lang)
    route      = route_info["route"]
    await ws.send_text(json.dumps({
        "type":        "agent_route",
        "route":       route,
        "reason":      route_info["reason"],
        "sub_queries": route_info.get("sub_queries", []),
        "tools":       route_info.get("tools", []),
    }))

    # ── 2a. RAG → delegate to existing pipeline ────────────────────────────
    if route == "rag":
        await run_rag_pipeline(ws, req)
        return

    # ── 2b. Direct LLM ────────────────────────────────────────────────────
    if route == "direct":
        await ws.send_text(json.dumps({
            "type": "phase_start", "phase": "generation",
            "message": "直接生成答案（无需检索）…" if lang == "zh" else "Generating answer directly (no retrieval)…",
        }))
        history_dicts = [h.model_dump() for h in req.history]
        full_answer   = await llm_stream_answer(ws, req.query, [], history_dicts, lang)
        await ws.send_text(json.dumps({
            "type": "agent_complete", "route": "direct",
            "final_answer": full_answer, "elapsed_seconds": round(time.time() - t0, 2),
            "retrieved_docs": [], "metrics": {},
        }))
        return

    # ── 2c. Realtime: tool calls ───────────────────────────────────────────
    if route == "realtime":
        import re as _re
        tools        = route_info.get("tools", [])
        tool_results = {}

        for tool_name in tools:
            await ws.send_text(json.dumps({
                "type":    "agent_tool_call",
                "tool":    tool_name,
                "message": f"调用工具：{tool_name}" if lang == "zh" else f"Calling tool: {tool_name}",
            }))
            if tool_name == "datetime":
                result_str = tool_datetime()
            elif tool_name == "calculator":
                expr_m     = _re.search(r'[\d\s\+\-\*\/\^\(\)\.]+', req.query)
                result_str = tool_calculator(expr_m.group().strip() if expr_m else req.query)
            elif tool_name == "web_search":
                result_str = await tool_web_search(req.query)
            else:
                result_str = json.dumps({"error": f"Unknown tool: {tool_name}"})

            tool_results[tool_name] = result_str
            await ws.send_text(json.dumps({
                "type": "agent_tool_result", "tool": tool_name,
                "result": result_str[:600],
            }))

        await ws.send_text(json.dumps({
            "type": "phase_start", "phase": "generation",
            "message": "基于工具结果生成答案…" if lang == "zh" else "Synthesising from tool results…",
        }))
        tools_ctx  = "\n".join(f"[{t}]: {r}" for t, r in tool_results.items())
        sys_prompt = (
            "你是智能助手，根据工具调用结果准确回答用户问题，语言自然。"
            if lang == "zh" else
            "You are an assistant. Answer accurately based on tool results. Be direct and natural."
        )
        full_answer = ""
        if state.llm_client:
            try:
                stream = await state.llm_client.chat.completions.create(
                    model=DEEPSEEK_MODEL,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user",   "content": f"工具结果：\n{tools_ctx}\n\n问题：{req.query}"},
                    ],
                    stream=True, max_tokens=800, temperature=0.3,
                )
                async for chunk in stream:
                    token = chunk.choices[0].delta.content or ""
                    if token:
                        full_answer += token
                        await ws.send_text(json.dumps({
                            "type": "answer_token", "token": token,
                            "full_answer_so_far": full_answer,
                        }))
            except Exception as e:
                full_answer = tools_ctx
                logger.error("Realtime synthesis error: %s", e)
        else:
            full_answer = tools_ctx
            await ws.send_text(json.dumps({
                "type": "answer_token", "token": full_answer, "full_answer_so_far": full_answer,
            }))

        await ws.send_text(json.dumps({
            "type": "agent_complete", "route": "realtime",
            "final_answer": full_answer, "elapsed_seconds": round(time.time() - t0, 2),
            "retrieved_docs": [], "metrics": {}, "tool_calls": list(tool_results.keys()),
        }))
        return

    # ── 2d. Complex: decompose → per-subtask RAG → synthesise ─────────────
    if route == "complex":
        sub_queries = route_info.get("sub_queries", [])
        if not sub_queries:
            await run_rag_pipeline(ws, req)
            return

        await ws.send_text(json.dumps({
            "type":        "agent_decompose",
            "sub_queries": sub_queries,
            "message":     f"拆解为 {len(sub_queries)} 个子任务…" if lang == "zh" else f"Decomposed into {len(sub_queries)} sub-tasks…",
        }))

        sub_results: List[dict] = []
        for i, sub_q in enumerate(sub_queries):
            await ws.send_text(json.dumps({
                "type": "agent_subquery", "index": i + 1,
                "total": len(sub_queries), "query": sub_q,
                "message": (f"子任务 {i+1}/{len(sub_queries)}：{sub_q}"
                            if lang == "zh" else f"Sub-task {i+1}/{len(sub_queries)}: {sub_q}"),
            }))
            sub = await query_rag(sub_q, strategy="adaptive", top_k=3, language=lang)
            preview = sub["answer"][:200] + ("…" if len(sub["answer"]) > 200 else "")
            sub_results.append({"query": sub_q, "answer": sub["answer"], "docs": sub["docs"][:2]})
            await ws.send_text(json.dumps({
                "type": "agent_subresult", "index": i + 1,
                "query": sub_q, "answer_preview": preview,
            }))

        await ws.send_text(json.dumps({
            "type": "phase_start", "phase": "generation",
            "message": "整合子任务，生成综合答案…" if lang == "zh" else "Synthesising sub-task results…",
        }))
        sub_ctx = "\n\n".join(
            f"【子问题{i+1}】{r['query']}\n【答案{i+1}】{r['answer']}"
            for i, r in enumerate(sub_results)
        )
        sys_syn = (
            "你是综合分析专家。根据多个子问题的答案，给出最终全面回答，结构清晰，语言流畅。"
            if lang == "zh" else
            "You are an expert synthesiser. Combine the sub-question answers into a comprehensive, well-structured final answer."
        )
        full_answer = ""
        if state.llm_client:
            try:
                stream = await state.llm_client.chat.completions.create(
                    model=DEEPSEEK_MODEL,
                    messages=[
                        {"role": "system", "content": sys_syn},
                        {"role": "user",   "content": f"原始问题：{req.query}\n\n子问题答案：\n{sub_ctx}"},
                    ],
                    stream=True, max_tokens=2000, temperature=0.5,
                )
                async for chunk in stream:
                    token = chunk.choices[0].delta.content or ""
                    if token:
                        full_answer += token
                        await ws.send_text(json.dumps({
                            "type": "answer_token", "token": token,
                            "full_answer_so_far": full_answer,
                        }))
            except Exception as e:
                full_answer = sub_ctx
                logger.error("Complex synthesis error: %s", e)
        else:
            full_answer = sub_ctx
            await ws.send_text(json.dumps({
                "type": "answer_token", "token": full_answer, "full_answer_so_far": full_answer,
            }))

        all_docs, seen = [], set()
        for r in sub_results:
            for d in r["docs"]:
                if d["id"] not in seen:
                    all_docs.append(d)
                    seen.add(d["id"])

        await ws.send_text(json.dumps({
            "type": "agent_complete", "route": "complex",
            "final_answer": full_answer, "elapsed_seconds": round(time.time() - t0, 2),
            "retrieved_docs": all_docs,
            "sub_results": [{"query": r["query"], "answer_preview": r["answer"][:200]} for r in sub_results],
            "metrics": {},
        }))

        # Audit: record sources used by agent complex route (inventory analytics)
        try:
            _append_jsonl(AUDIT_FILE, {
                "ts": _utc_now_iso(),
                "type": "retrieval_complete",
                "tenant_id": req.tenant_id,
                "user_id": req.user_id,
                "user_role": req.user_role,
                "query": redact_text(req.query),
                "strategy": "agent_complex",
                "top_k": 0,
                "final_confidence": None,
                "retrieved": [
                    {"id": d.get("id"), "source": d.get("source"), "score": d.get("score")}
                    for d in all_docs
                ],
            })
        except Exception:
            pass
