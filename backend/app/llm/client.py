"""
LLM Client — DeepSeek API (OpenAI-compatible)
==============================================
Owns the LLM client lifecycle and every LLM-backed capability used by the
pipeline: plain completions, query rewriting, and streaming answer generation.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import List

from fastapi import WebSocket

from app.core import state
from app.core.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from app.core.messages import _t

logger = logging.getLogger(__name__)


def init_llm_client() -> None:
    if not DEEPSEEK_API_KEY:
        logger.warning("DEEPSEEK_API_KEY not set — LLM features disabled")
        return
    from openai import AsyncOpenAI
    state.llm_client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    logger.info("LLM client ready (model=%s)", DEEPSEEK_MODEL)


async def llm_call(messages: list, max_tokens: int = 100, temperature: float = 0.3) -> str:
    """Non-streaming LLM helper with graceful fallback."""
    if not state.llm_client:
        return ""
    try:
        resp = await state.llm_client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        return ""


async def llm_rewrite_query(original: str, failure_reason: str, lang: str = "zh") -> str:
    result = await llm_call(
        messages=[
            {"role": "system", "content": _t("sys_rewrite", lang)},
            {"role": "user",   "content": _t("usr_rewrite", lang, original=original, reason=failure_reason)},
        ],
        max_tokens=60,
    )
    fallback = (original + " 详细介绍") if lang == "zh" else (original + " detailed explanation")
    return result if result else fallback


async def llm_stream_answer(
    ws: WebSocket,
    query: str,
    docs: List[dict],
    history: List[dict],
    lang: str = "zh",
) -> str:
    """Stream LLM answer token-by-token; supports multi-turn conversation history."""
    doc_label = "文档" if lang == "zh" else "Document"
    context = "\n\n".join(
        f"【{doc_label}{i + 1}】{d['title']}\n{d['content']}"
        for i, d in enumerate(docs[:4])
    )
    system_prompt = _t("sys_answer", lang)
    user_prompt   = _t("usr_answer", lang, context=context, query=query)

    if not state.llm_client:
        top = docs[0] if docs else {}
        no_content = "暂无相关内容" if lang == "zh" else "No relevant content found."
        fallback = (
            _t("fallback_prefix", lang, title=top.get("title", ""), query=query)
            + top.get("content", no_content)
            + _t("fallback_suffix", lang)
        )
        full = ""
        for part in fallback.split("。" if lang == "zh" else ". "):
            if not part.strip():
                continue
            sep = "。" if lang == "zh" else ". "
            full += part + sep
            await ws.send_text(json.dumps({
                "type": "answer_token", "token": part + sep, "full_answer_so_far": full,
            }))
            await asyncio.sleep(0.08)
        return full

    # Build messages with conversation history (last 6 turns max)
    messages = [{"role": "system", "content": system_prompt}]
    for turn in history[-6:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_prompt})

    full_answer = ""
    try:
        stream = await state.llm_client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=messages,
            stream=True,
            max_tokens=1500,
            temperature=0.7,
        )
        async for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            if token:
                full_answer += token
                await ws.send_text(json.dumps({
                    "type": "answer_token",
                    "token": token,
                    "full_answer_so_far": full_answer,
                }))
    except Exception as e:
        logger.error("LLM stream error: %s", e)
        err = f"\n\n[答案生成出错：{e}]"
        full_answer += err
        await ws.send_text(json.dumps({
            "type": "answer_token", "token": err, "full_answer_so_far": full_answer,
        }))

    return full_answer
