# ── Bilingual message table ────────────────────────────────────────────────────
# All user-facing strings are defined here so the pipeline is language-agnostic.

_MSG: dict = {
    # Pipeline event messages
    "phase_retrieval": {
        "zh": "第 {iteration} 轮检索（{strategy}）：「{query}」",
        "en": "Round {iteration} retrieval ({strategy}): \"{query}\"",
    },
    "phase_reranking": {
        "zh": "{ce}精排 {n} 个候选文档…",
        "en": "{ce}Reranking {n} candidate documents…",
    },
    "phase_generation": {
        "zh": "DeepSeek 流式生成答案…",
        "en": "Generating answer with DeepSeek (streaming)…",
    },
    "phase_ragas": {
        "zh": "RAGAS 评估：计算检索与生成质量指标…",
        "en": "RAGAS evaluation: computing retrieval & generation quality metrics…",
    },
    # Failure diagnostics
    "failure_extreme": {
        "zh": "检索分数极低，查询词与知识库词汇差异较大，需重写为更通用的术语",
        "en": "Retrieval score very low — query vocabulary differs greatly from the knowledge base; rewrite with more general terms",
    },
    "failure_moderate": {
        "zh": "召回文档相关性不足，查询语义与文档内容存在偏差，尝试更换检索角度",
        "en": "Retrieved documents lack relevance — semantic mismatch between query and docs; try rephrasing from a different angle",
    },
    "failure_low": {
        "zh": "召回文档相关性低于置信阈值",
        "en": "Retrieved document relevance below confidence threshold",
    },
    # LLM system prompts
    "sys_answer": {
        "zh": (
            "你是监管文档智能分析助手，专注于基金/资产管理监管合规领域。"
            "请根据以下参考监管文档准确、详细地回答用户问题，并在陈述具体要求时"
            "明确引用法规编号和条款（如「REG-FM-105 第3条」）。"
            "如文档中信息不足，请如实说明，不要编造监管要求。回答使用中文，语言专业、自然流畅。"
        ),
        "en": (
            "You are a Regulatory Document Intelligence assistant specialising in fund/asset-management "
            "compliance. Answer the user's question accurately and in detail based on the provided reference "
            "regulatory documents. When you state a specific requirement, cite the regulation number and "
            "article (e.g. \"REG-FM-105, Article 3\"). "
            "If the documents lack sufficient information, say so honestly — do not fabricate regulatory "
            "requirements. Reply in English with precise, professional language."
        ),
    },
    "usr_answer": {
        "zh": "参考监管文档：\n{context}\n\n用户问题：{query}",
        "en": "Reference regulatory documents:\n{context}\n\nUser question: {query}",
    },
    "sys_compliance": {
        "zh": (
            "你是资深监管合规分析师。给定一个业务/产品场景描述和相关监管文档条款，"
            "请逐条评估该场景是否符合每条引用的监管要求。"
            "对每条要求，给出：assessment（compliant | non_compliant | uncertain | not_applicable）"
            "以及简明理由，并引用具体法规编号和条款。"
            "最后给出总体结论（compliant | non_compliant | needs_review）和一段摘要。"
            "严格以JSON格式输出，不要输出多余文字。"
        ),
        "en": (
            "You are a senior regulatory compliance analyst. Given a business/product scenario description "
            "and relevant regulatory document excerpts, assess whether the scenario complies with each cited "
            "regulatory requirement. For each requirement, provide: assessment "
            "(compliant | non_compliant | uncertain | not_applicable) and a concise rationale citing the "
            "specific regulation number and article. Conclude with an overall_status "
            "(compliant | non_compliant | needs_review) and a short summary. "
            "Output strict JSON only, no extra text, in this shape:\n"
            '{"overall_status":"compliant|non_compliant|needs_review","summary":"...",'
            '"findings":[{"requirement":"...","citation":"REG-FM-XXX Article N","assessment":"...","rationale":"..."}]}'
        ),
    },
    "usr_compliance": {
        "zh": "业务场景：\n{scenario}\n\n相关监管条款：\n{context}",
        "en": "Business/product scenario:\n{scenario}\n\nRelevant regulatory excerpts:\n{context}",
    },
    "sys_rewrite": {
        "zh": "你是检索优化专家。根据失败原因重写查询，使其更易命中知识库。只输出重写后的查询，不超过30字。",
        "en": "You are a retrieval optimisation expert. Rewrite the query based on the failure reason to better match the knowledge base. Output only the rewritten query (≤15 words).",
    },
    "usr_rewrite": {
        "zh": "原始查询：{original}\n失败原因：{reason}",
        "en": "Original query: {original}\nFailure reason: {reason}",
    },
    # Fallback answer (no LLM key)
    "fallback_prefix": {
        "zh": "根据知识库文档「{title}」，针对问题「{query}」：\n\n",
        "en": "Based on the knowledge base document \"{title}\", regarding the question \"{query}\":\n\n",
    },
    "fallback_suffix": {
        "zh": "\n\n（提示：未配置 DEEPSEEK_API_KEY，以上为文档直接摘录。）",
        "en": "\n\n(Note: DEEPSEEK_API_KEY not configured — the above is a direct document excerpt.)",
    },
    # Cross-encoder label
    "ce_label": {
        "zh": "Cross-Encoder ",
        "en": "Cross-Encoder ",
    },
}


def _t(key: str, lang: str = "zh", **kwargs) -> str:
    """Resolve a bilingual message key, interpolating kwargs."""
    entry = _MSG.get(key, {})
    text  = entry.get(lang) or entry.get("zh") or key
    return text.format(**kwargs) if kwargs else text
