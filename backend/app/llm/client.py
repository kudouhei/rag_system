"""
Provider-neutral LLM client
===========================
Owns the LLM client lifecycle and the provider-specific API adaptation used by
plain completions, query rewriting, and streaming answer generation.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import List

from fastapi import WebSocket
from pydantic import BaseModel

from app.core import state
from app.core.config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_PROVIDER,
)
from app.core.messages import _t

logger = logging.getLogger(__name__)


SUPPORTED_LLM_PROVIDERS = {
    "deepseek",
    "azure_openai",
}


def init_llm_client() -> None:
    """Initialise the configured OpenAI-compatible LLM client."""

    state.llm_client = None

    if LLM_PROVIDER not in SUPPORTED_LLM_PROVIDERS:
        logger.error(
            "Unsupported LLM provider '%s' — LLM features disabled",
            LLM_PROVIDER,
        )
        return

    if not LLM_API_KEY:
        logger.warning(
            "LLM_API_KEY not set for provider '%s' — "
            "LLM features disabled",
            LLM_PROVIDER,
        )
        return

    from openai import AsyncOpenAI

    state.llm_client = AsyncOpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
    )

    logger.info(
        "LLM client ready (provider=%s, model=%s)",
        LLM_PROVIDER,
        LLM_MODEL,
    )

def get_active_llm_model() -> str | None:
    """Return the configured model when an LLM client is active."""

    if state.llm_client is None:
        return None

    return LLM_MODEL


async def llm_call(
    messages: list[dict[str, str]],
    max_tokens: int = 100,
    temperature: float = 0.3,
) -> str:
    """Run a non-streaming request through the configured provider."""

    if not state.llm_client:
        return ""

    try:
        if LLM_PROVIDER == "azure_openai":
            response = await state.llm_client.responses.create(
                model=LLM_MODEL,
                input=messages,
                max_output_tokens=max(
                    max_tokens,
                    128,
                ),
            )

            return (
                response.output_text or ""
            ).strip()

        response = await (
            state.llm_client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        )

        content = response.choices[0].message.content

        return (
            content or ""
        ).strip()

    except Exception:
        logger.exception(
            "LLM call failed "
            "(provider=%s, model=%s)",
            LLM_PROVIDER,
            LLM_MODEL,
        )
        return ""

async def llm_structured_call(
    messages: list[dict[str, str]],
    response_model: type[BaseModel],
    max_tokens: int = 1000,
    temperature: float = 0.1,
) -> str:
    """Generate JSON using provider-supported structured output."""

    if not state.llm_client:
        return ""

    if LLM_PROVIDER != "azure_openai":
        return await llm_call(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    try:
        response = await state.llm_client.responses.parse(
            model=LLM_MODEL,
            input=messages,
            text_format=response_model,
            max_output_tokens=max(
                max_tokens,
                128,
            ),
        )

        parsed = response.output_parsed

        if parsed is None:
            logger.warning(
                "Structured LLM response had no parsed output "
                "(provider=%s, model=%s, response_id=%s)",
                LLM_PROVIDER,
                LLM_MODEL,
                getattr(response, "id", None),
            )
            return ""

        return parsed.model_dump_json()

    except Exception:
        logger.exception(
            "Structured LLM call failed "
            "(provider=%s, model=%s)",
            LLM_PROVIDER,
            LLM_MODEL,
        )
        return ""

async def llm_rewrite_query(original: str, failure_reason: str, lang: str = "en") -> str:
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
    lang: str = "en",
) -> str:
    """Stream LLM answer token-by-token; supports multi-turn conversation history."""
    from app.pipeline.utils import format_doc_context
    context = format_doc_context(docs[:4], lang)
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
            model=LLM_MODEL,
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
