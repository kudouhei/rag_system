from __future__ import annotations

import asyncio
import json
from datetime import datetime

# ── Tool: current datetime ─────────────────────────────────────────────────────

def tool_datetime() -> str:
    weekdays_zh = ["一", "二", "三", "四", "五", "六", "日"]
    weekdays_en = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    now = datetime.now()
    return json.dumps({
        "datetime":   now.strftime("%Y-%m-%d %H:%M:%S"),
        "date_zh":    now.strftime("%Y年%m月%d日"),
        "weekday_zh": f"星期{weekdays_zh[now.weekday()]}",
        "weekday_en": weekdays_en[now.weekday()],
        "timestamp":  int(now.timestamp()),
    }, ensure_ascii=False)


# ── Tool: safe calculator ──────────────────────────────────────────────────────

def tool_calculator(expression: str) -> str:
    """Evaluate a math expression using Python's ast module (no eval() risk)."""
    import ast as _ast
    import operator as _op
    _OPS = {
        _ast.Add: _op.add, _ast.Sub: _op.sub,
        _ast.Mult: _op.mul, _ast.Div: _op.truediv,
        _ast.Pow: _op.pow, _ast.Mod: _op.mod, _ast.FloorDiv: _op.floordiv,
    }

    def _eval(node):
        if isinstance(node, _ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, _ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, _ast.UnaryOp) and isinstance(node.op, _ast.USub):
            return -_eval(node.operand)
        raise ValueError(f"Unsupported expression node: {type(node).__name__}")

    try:
        result = _eval(_ast.parse(expression.strip(), mode="eval").body)
        return json.dumps({"expression": expression, "result": result})
    except Exception as e:
        return json.dumps({"expression": expression, "error": str(e)})


# ── Tool: web search (DuckDuckGo, no API key needed) ──────────────────────────

async def tool_web_search(query: str, max_results: int = 4) -> str:
    try:
        from duckduckgo_search import DDGS
        loop = asyncio.get_event_loop()

        def _search():
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=max_results))

        results = await loop.run_in_executor(None, _search)
        formatted = [
            {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")[:300]}
            for r in results
        ]
        return json.dumps({"query": query, "results": formatted}, ensure_ascii=False, indent=2)
    except ImportError:
        return json.dumps({"error": "duckduckgo_search not installed — run: pip install duckduckgo_search"})
    except Exception as e:
        return json.dumps({"error": f"Web search unavailable: {e}"})
