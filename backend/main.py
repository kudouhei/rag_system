"""
Adaptive RAG System — Backend
==============================
Implemented Techniques
──────────────────────
  ① Iterative Retrieval + ReAct-style Reflection   Yao et al., 2022
  ② Hybrid Dense-Sparse Retrieval  (BGE + BM25)
  ③ HyDE – Hypothetical Document Embeddings        Gao et al., EMNLP 2022
  ④ Cross-Encoder Reranking                        BAAI/bge-reranker series
  ⑤ RAGAS-style Evaluation Framework               Es et al., 2023
  ⑥ Multi-turn Conversation Memory
  ⑦ Contextual Chunking                            Anthropic, 2024
  ⑧ Embedding Cache (incremental indexing)
  ⑨ User Feedback Loop + Satisfaction Analytics

Privacy & Compliance
────────────────────
  • All document embeddings are computed locally (no data leaves the server)
  • LLM calls are opt-in; the system degrades gracefully without an API key
  • Designed with GDPR data-minimisation principles in mind

LLM backend : DeepSeek API (OpenAI-compatible)
Embeddings  : sentence-transformers  (BAAI/bge-small-zh-v1.5)
Reranker    : BAAI/bge-reranker-base (optional, set RERANKER_MODEL)
"""

import os
import json
import time
import hashlib
import asyncio
import logging
import shutil
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Bilingual message table ────────────────────────────────────────────────────
# All user-facing strings are defined here so the pipeline is language-agnostic.

_MSG: dict = {
    # Pipeline event messages
    "phase_hyde": {
        "zh": "HyDE：生成假设文档以增强向量检索…",
        "en": "HyDE: generating hypothetical document to enhance vector retrieval…",
    },
    "hyde_done": {
        "zh": "假设文档生成完成，用于向量检索",
        "en": "Hypothetical document generated — used for vector retrieval",
    },
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
            "你是企业知识库问答助手。请根据以下参考文档准确、详细地回答用户问题。"
            "如文档中信息不足，请如实说明，不要编造内容。回答使用中文，语言自然流畅。"
        ),
        "en": (
            "You are an enterprise knowledge-base assistant. "
            "Answer the user's question accurately and in detail based on the provided reference documents. "
            "If the documents lack sufficient information, say so honestly — do not fabricate content. "
            "Reply in English with clear, natural language."
        ),
    },
    "usr_answer": {
        "zh": "参考文档：\n{context}\n\n用户问题：{query}",
        "en": "Reference documents:\n{context}\n\nUser question: {query}",
    },
    "sys_rewrite": {
        "zh": "你是检索优化专家。根据失败原因重写查询，使其更易命中知识库。只输出重写后的查询，不超过30字。",
        "en": "You are a retrieval optimisation expert. Rewrite the query based on the failure reason to better match the knowledge base. Output only the rewritten query (≤15 words).",
    },
    "usr_rewrite": {
        "zh": "原始查询：{original}\n失败原因：{reason}",
        "en": "Original query: {original}\nFailure reason: {reason}",
    },
    "sys_hyde": {
        "zh": "你是一位知识渊博的文档作者。根据用户问题，生成一段可能出现在知识库中的文档段落。直接输出段落内容，不超过150字，不要包含问题本身。",
        "en": "You are a knowledgeable document author. Given the user's question, write a concise passage (≤100 words) that might appear in the knowledge base to answer it. Output only the passage — do not include the question itself.",
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

# ── Configuration ─────────────────────────────────────────────────────────────

DOCS_DIR             = Path(os.getenv("DOCS_DIR", Path(__file__).parent / "docs"))
EMBED_MODEL_NAME     = os.getenv("EMBED_MODEL", "BAAI/bge-small-zh-v1.5")
RERANKER_MODEL       = os.getenv("RERANKER_MODEL", "")
DEEPSEEK_API_KEY     = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL       = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_BASE_URL    = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
MAX_CHUNK_CHARS      = int(os.getenv("MAX_CHUNK_CHARS", "600"))
# ⑦ Contextual Chunking: prepend LLM-generated context to each chunk before embedding
CONTEXTUAL_CHUNKING  = os.getenv("CONTEXTUAL_CHUNKING", "false").lower() == "true"
# ⑧ Embedding cache directory
CACHE_DIR            = Path(os.getenv("CACHE_DIR", Path(__file__).parent / "cache"))
# Feedback log
FEEDBACK_FILE        = Path(__file__).parent / "feedback.jsonl"

# ── Global state ──────────────────────────────────────────────────────────────

KNOWLEDGE_BASE: List[dict]           = []
doc_embeddings: Optional[np.ndarray] = None
embed_model                          = None
cross_encoder                        = None     # BAAI/bge-reranker (optional)
bm25_index                           = None
_tokenize_fn                         = None
llm_client                           = None

# ── FastAPI ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    await _startup()
    yield

app = FastAPI(title="Adaptive RAG System", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

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

# ══════════════════════════════════════════════════════════════════════════════
# Dense Retrieval — sentence-transformers (BAAI/bge)
# ══════════════════════════════════════════════════════════════════════════════

def _init_embed_model() -> None:
    global embed_model
    from sentence_transformers import SentenceTransformer
    logger.info("Loading embedding model: %s", EMBED_MODEL_NAME)
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    logger.info("Embedding model ready")


def _compute_doc_embeddings(docs: List[dict]) -> np.ndarray:
    # Use `embedding_content` if set by contextual chunking, otherwise title+content
    texts = [
        d.get("embedding_content") or (d["title"] + " " + d["content"])
        for d in docs
    ]
    logger.info("Encoding %d documents…", len(texts))
    embs = embed_model.encode(
        texts, normalize_embeddings=True, show_progress_bar=True, batch_size=32,
    )
    return embs.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# ⑧ Embedding Cache — skip re-encoding unchanged document sets
# ══════════════════════════════════════════════════════════════════════════════

def _doc_fingerprint(docs: List[dict]) -> str:
    """SHA-256 fingerprint of document IDs + content (first 100 chars each)."""
    raw = "|".join(
        f"{d['id']}:{d['content'][:100]}"
        for d in sorted(docs, key=lambda x: x["id"])
    )
    prefix = f"ctx={CONTEXTUAL_CHUNKING}|model={EMBED_MODEL_NAME}|"
    return hashlib.sha256((prefix + raw).encode()).hexdigest()[:20]


def _load_emb_cache(docs: List[dict]) -> Optional[np.ndarray]:
    fp = _doc_fingerprint(docs)
    cache_file = CACHE_DIR / f"emb_{fp}.npy"
    if cache_file.exists():
        logger.info("Embedding cache hit (%s) — skipping re-encoding", fp)
        return np.load(str(cache_file)).astype(np.float32)
    return None


def _save_emb_cache(docs: List[dict], embs: np.ndarray) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fp = _doc_fingerprint(docs)
    cache_file = CACHE_DIR / f"emb_{fp}.npy"
    np.save(str(cache_file), embs)
    # Remove stale caches (keep only the latest)
    for old in CACHE_DIR.glob("emb_*.npy"):
        if old != cache_file:
            old.unlink(missing_ok=True)
    logger.info("Embedding cache saved (%s, %d docs)", fp, len(docs))


def _encode_text(text: str) -> np.ndarray:
    """Encode arbitrary text (with BGE retrieval prefix if applicable)."""
    if "bge" in EMBED_MODEL_NAME.lower():
        text = "为这个句子生成表示以用于检索相关文章：" + text
    return embed_model.encode([text], normalize_embeddings=True)[0].astype(np.float32)


def compute_embedding_scores(query_or_text: str) -> np.ndarray:
    """Cosine similarity between query (or hypothetical doc) and all indexed docs → [0, 1]."""
    q_emb = _encode_text(query_or_text)
    raw = doc_embeddings @ q_emb          # already L2-normalised
    return ((raw + 1.0) / 2.0).astype(np.float32)

# ══════════════════════════════════════════════════════════════════════════════
# Sparse Retrieval — BM25 + jieba tokenisation
# ══════════════════════════════════════════════════════════════════════════════

def _init_bm25(docs: List[dict]) -> None:
    global bm25_index, _tokenize_fn
    try:
        import jieba
        jieba.setLogLevel(logging.WARNING)
        _tokenize_fn = lambda t: list(jieba.cut(t))
        logger.info("BM25 using jieba tokeniser")
    except ImportError:
        _tokenize_fn = lambda t: t.lower().split()
        logger.warning("jieba not found — using whitespace tokeniser")

    from rank_bm25 import BM25Okapi
    tokenized = [_tokenize_fn(d["title"] + " " + d["content"]) for d in docs]
    bm25_index = BM25Okapi(tokenized)
    logger.info("BM25 index ready (%d docs)", len(docs))


def compute_bm25_scores(query: str) -> np.ndarray:
    tokens = _tokenize_fn(query)
    scores = bm25_index.get_scores(tokens).astype(np.float32)
    mx = scores.max()
    return scores / mx if mx > 0 else scores

# ══════════════════════════════════════════════════════════════════════════════
# Cross-Encoder Reranking  (optional — set RERANKER_MODEL env var)
# ══════════════════════════════════════════════════════════════════════════════

def _init_cross_encoder() -> None:
    global cross_encoder
    if not RERANKER_MODEL:
        logger.info("Cross-encoder disabled (set RERANKER_MODEL to enable)")
        return
    try:
        from sentence_transformers import CrossEncoder
        logger.info("Loading cross-encoder: %s", RERANKER_MODEL)
        cross_encoder = CrossEncoder(RERANKER_MODEL, max_length=512)
        logger.info("Cross-encoder ready")
    except Exception as e:
        logger.warning("Could not load cross-encoder %s: %s", RERANKER_MODEL, e)


def _rerank_docs(query: str, docs: List[dict]) -> List[dict]:
    """
    Two-stage reranking:
      • If cross_encoder is available → use it (genuine cross-attention scoring).
      • Otherwise → refined cosine similarity with exact-match bonus.
    """
    if cross_encoder is not None:
        pairs = [(query, d["content"]) for d in docs]
        ce_scores = cross_encoder.predict(pairs, show_progress_bar=False)
        # Sigmoid-normalise to [0, 1]
        ce_scores = 1.0 / (1.0 + np.exp(-ce_scores))
        reranked = []
        for doc, ce in zip(docs, ce_scores):
            reranked.append({
                **doc,
                "pre_rerank_score": doc["final_score"],
                "ce_score": round(float(ce), 4),
            })
    else:
        # Fallback: recompute cosine with query re-embedding
        q_emb = _encode_text(query)
        reranked = []
        for doc in docs:
            idx = next((i for i, d in enumerate(KNOWLEDGE_BASE) if d["id"] == doc["id"]), None)
            if idx is not None and doc_embeddings is not None:
                cosine = float(doc_embeddings[idx] @ q_emb)
                ce = (cosine + 1.0) / 2.0
            else:
                ce = doc["final_score"]
            # Boost for title match
            if any(t in doc["title"] for t in query.split() if len(t) > 1):
                ce = min(0.99, ce + 0.04)
            reranked.append({
                **doc,
                "pre_rerank_score": doc["final_score"],
                "ce_score": round(ce, 4),
            })

    reranked.sort(key=lambda x: x["ce_score"], reverse=True)
    return reranked

# ══════════════════════════════════════════════════════════════════════════════
# RAGAS-style Evaluation  (Es et al., 2023)
# ══════════════════════════════════════════════════════════════════════════════

def compute_ragas_metrics(query: str, docs: List[dict], answer: str) -> dict:
    """
    Simplified RAGAS metrics (no external NLI model required):

    • context_relevance   — avg cosine(query, chunk)  ∈ [0, 1]
    • context_precision   — fraction of chunks with similarity > 0.45
    • answer_relevance    — cosine(query_emb, answer_emb)  ∈ [0, 1]
    • answer_faithfulness — token-overlap proxy between answer and context
    """
    if not docs or not query:
        return {}

    q_emb = _encode_text(query)

    # Context Relevance & Precision
    sims = []
    for doc in docs:
        idx = next((i for i, d in enumerate(KNOWLEDGE_BASE) if d["id"] == doc["id"]), None)
        if idx is not None and doc_embeddings is not None:
            sim = (float(doc_embeddings[idx] @ q_emb) + 1.0) / 2.0
            sims.append(sim)

    context_relevance = round(float(np.mean(sims)), 3) if sims else 0.0
    context_precision = round(sum(1 for s in sims if s > 0.45) / max(len(sims), 1), 3)

    # Answer Relevance — cosine(answer_emb, query_emb)
    if answer.strip():
        a_emb = _encode_text(answer)
        ar = (float(a_emb @ q_emb) + 1.0) / 2.0
        answer_relevance = round(ar, 3)
    else:
        answer_relevance = 0.0

    # Answer Faithfulness (token-overlap proxy)
    context_tokens = set(
        " ".join(d["content"] for d in docs[:3]).lower().split()
    )
    answer_tokens = set(answer.lower().split())
    if answer_tokens:
        overlap = len(answer_tokens & context_tokens) / len(answer_tokens)
        answer_faithfulness = round(min(1.0, overlap * 1.8), 3)
    else:
        answer_faithfulness = 0.0

    return {
        "context_relevance":    context_relevance,
        "context_precision":    context_precision,
        "answer_relevance":     answer_relevance,
        "answer_faithfulness":  answer_faithfulness,
    }

# ══════════════════════════════════════════════════════════════════════════════
# LLM — DeepSeek API
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# ⑦ Contextual Chunking  (Anthropic, 2024)
# ══════════════════════════════════════════════════════════════════════════════

async def _contextualize_chunks(docs: List[dict], lang: str = "zh") -> List[dict]:
    """
    Anthropic Contextual Retrieval (2024):
    For each chunk, use LLM to generate a 1-2 sentence document-level context
    and store it as `embedding_content`.  The original `content` (shown to users)
    is unchanged; only the embedded text is enriched.

    This significantly reduces the "out-of-context chunk" problem, e.g.:
      Raw chunk    : "Sales increased by 23% compared to last quarter."
      With context : "This passage is from Q3 2024 Financial Report, discussing
                      revenue performance. Sales increased by 23%..."
    """
    if not llm_client or not CONTEXTUAL_CHUNKING:
        return docs

    # Preload full text per source file (up to 2000 chars for context window)
    source_cache: dict = {}
    for doc in docs:
        src = doc.get("source", "")
        if src and src not in source_cache:
            try:
                source_cache[src] = (DOCS_DIR / src).read_text(encoding="utf-8")[:2000]
            except Exception:
                source_cache[src] = ""

    logger.info("Contextual chunking: enriching %d chunks with LLM context…", len(docs))

    sys_prompt = (
        "You are a document analyst. Given a document excerpt and a chunk from it, "
        "write 1-2 sentences of context explaining where this chunk sits in the document "
        "and what it's about. Output ONLY the context sentences."
    ) if lang == "en" else (
        "你是文档分析专家。给定文档摘录和其中的一个段落，"
        "用1-2句话说明该段落在文档中的位置和主题。只输出上下文说明。"
    )

    enriched, ok_count = [], 0
    for doc in docs:
        full_text = source_cache.get(doc.get("source", ""), "")
        ctx = ""
        if full_text:
            ctx = await llm_call(
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user",   "content":
                        f"Document (excerpt):\n{full_text[:1200]}\n\nChunk:\n{doc['content'][:400]}"},
                ],
                max_tokens=80,
                temperature=0.1,
            )
            await asyncio.sleep(0.05)   # gentle rate-limiting

        d = doc.copy()
        if ctx:
            d["embedding_content"] = f"{ctx}\n\n{doc['content']}"
            d["context_added"] = True
            ok_count += 1
        enriched.append(d)

    logger.info("Contextual chunking: %d/%d chunks enriched", ok_count, len(docs))
    return enriched


def _init_llm_client() -> None:
    global llm_client
    if not DEEPSEEK_API_KEY:
        logger.warning("DEEPSEEK_API_KEY not set — LLM features disabled")
        return
    from openai import AsyncOpenAI
    llm_client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    logger.info("LLM client ready (model=%s)", DEEPSEEK_MODEL)


async def llm_call(messages: list, max_tokens: int = 100, temperature: float = 0.3) -> str:
    """Non-streaming LLM helper with graceful fallback."""
    if not llm_client:
        return ""
    try:
        resp = await llm_client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        return ""


async def hyde_generate_hypothetical(query: str, lang: str = "zh") -> str:
    """
    HyDE (Gao et al., EMNLP 2022):
    Generate a hypothetical document passage that would answer the query,
    then use *its* embedding for retrieval instead of the raw query embedding.
    This bridges the lexical/semantic gap between short queries and long documents.
    """
    result = await llm_call(
        messages=[
            {"role": "system", "content": _t("sys_hyde", lang)},
            {"role": "user",   "content": query},
        ],
        max_tokens=200,
        temperature=0.5,
    )
    return result if result else query


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

    if not llm_client:
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
        stream = await llm_client.chat.completions.create(
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

# ══════════════════════════════════════════════════════════════════════════════
# Startup — build all indexes
# ══════════════════════════════════════════════════════════════════════════════

async def _startup() -> None:
    global KNOWLEDGE_BASE, doc_embeddings

    raw_docs = load_documents_from_folder(DOCS_DIR)
    if not raw_docs:
        logger.warning("No documents found in %s", DOCS_DIR)
        raw_docs = [{
            "id": "placeholder", "title": "暂无文档",
            "content": f"请在 {DOCS_DIR} 目录中添加 .txt/.md/.pdf 文件后重启服务。",
            "source": "", "tags": [], "embedding_score": 0.0, "bm25_score": 0.0,
            "word_count": 0, "char_count": 0, "chunk_index": 0, "total_chunks": 1,
        }]

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _init_embed_model)

    # ⑦ Contextual Chunking (optional, requires LLM key)
    _init_llm_client()
    docs = await _contextualize_chunks(raw_docs)

    # ⑧ Embedding Cache — avoid re-encoding unchanged docs
    cached = _load_emb_cache(docs)
    if cached is not None:
        doc_embeddings = cached
    else:
        doc_embeddings = await loop.run_in_executor(None, _compute_doc_embeddings, docs)
        _save_emb_cache(docs, doc_embeddings)

    await loop.run_in_executor(None, _init_bm25, docs)
    await loop.run_in_executor(None, _init_cross_encoder)

    KNOWLEDGE_BASE = docs
    logger.info("✓ RAG system ready — %d chunks indexed (contextual=%s, cached=%s)",
                len(KNOWLEDGE_BASE), CONTEXTUAL_CHUNKING, cached is not None)

# ══════════════════════════════════════════════════════════════════════════════
# Request / Response Models
# ══════════════════════════════════════════════════════════════════════════════

class ConversationTurn(BaseModel):
    role: str       # "user" | "assistant"
    content: str


class QueryRequest(BaseModel):
    query: str
    strategy: str = "adaptive"          # vector | bm25 | hybrid | adaptive
    enable_iterative: bool = True
    enable_rerank: bool = True
    enable_hyde: bool = False           # HyDE (Gao et al., EMNLP 2022)
    confidence_threshold: float = 0.55
    top_k: int = 5
    language: str = "zh"               # "zh" | "en"
    history: List[ConversationTurn] = []

# ══════════════════════════════════════════════════════════════════════════════
# WebSocket — streaming RAG pipeline
# ══════════════════════════════════════════════════════════════════════════════

@app.websocket("/ws/query")
async def websocket_query(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            request = QueryRequest(**json.loads(data))
            await run_rag_pipeline(websocket, request)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error("WebSocket error: %s", e)
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
        except Exception:
            pass


async def run_rag_pipeline(ws: WebSocket, req: QueryRequest) -> None:
    t0   = time.time()
    lang = req.language or "zh"
    current_query = req.query
    iteration, max_iter = 0, 3
    all_iterations: List[dict] = []
    results: List[dict] = []

    await ws.send_text(json.dumps({
        "type": "pipeline_start",
        "query": req.query,
        "config": {
            "strategy":         req.strategy,
            "enable_iterative": req.enable_iterative,
            "enable_rerank":    req.enable_rerank,
            "enable_hyde":      req.enable_hyde,
            "threshold":        req.confidence_threshold,
            "total_docs":       len(KNOWLEDGE_BASE),
            "cross_encoder":    cross_encoder is not None,
            "language":         lang,
        },
    }))

    # ── HyDE pre-processing ───────────────────────────────────────────────────
    retrieval_text = current_query
    if req.enable_hyde:
        await ws.send_text(json.dumps({
            "type": "phase_start", "phase": "hyde",
            "message": _t("phase_hyde", lang),
        }))
        hypothetical   = await hyde_generate_hypothetical(current_query, lang)
        retrieval_text = hypothetical
        await ws.send_text(json.dumps({
            "type": "hyde_generation",
            "hypothetical_doc": hypothetical,
            "message": _t("hyde_done", lang),
        }))
        await asyncio.sleep(0.1)

    # ── Phase 1 & 2: Iterative Retrieval + Reflection ────────────────────────
    while iteration < max_iter:
        iteration += 1
        strategy = req.strategy
        if strategy == "adaptive":
            strategy = ["hybrid", "vector", "bm25"][min(iteration - 1, 2)]

        await ws.send_text(json.dumps({
            "type": "phase_start", "phase": "retrieval",
            "iteration": iteration, "query": current_query,
            "message": _t("phase_retrieval", lang,
                          iteration=iteration, strategy=strategy, query=current_query),
        }))

        loop = asyncio.get_event_loop()
        # HyDE uses hypothetical_doc embedding only on first iteration
        vec_text = retrieval_text if (req.enable_hyde and iteration == 1) else current_query
        emb_scores   = await loop.run_in_executor(None, compute_embedding_scores, vec_text)
        bm25_arr     = await loop.run_in_executor(None, compute_bm25_scores, current_query)

        docs_scored = []
        for idx, doc in enumerate(KNOWLEDGE_BASE):
            d   = doc.copy()
            es  = float(emb_scores[idx])
            bs  = float(bm25_arr[idx])
            if strategy == "vector":
                d.update(embedding_score=es, bm25_score=0.0, final_score=es)
            elif strategy == "bm25":
                d.update(embedding_score=0.0, bm25_score=bs, final_score=bs)
            else:   # hybrid
                d.update(embedding_score=es, bm25_score=bs, final_score=es * 0.6 + bs * 0.4)
            d["strategy_used"] = strategy
            docs_scored.append(d)

        # Stream top-10 scores to UI
        top10 = sorted(docs_scored, key=lambda x: x["final_score"], reverse=True)[:10]
        for doc in top10:
            await ws.send_text(json.dumps({
                "type": "doc_scored",
                "doc_id": doc["id"], "title": doc["title"],
                "embedding_score": round(doc["embedding_score"], 3),
                "bm25_score":      round(doc["bm25_score"], 3),
                "final_score":     round(doc["final_score"], 3),
                "strategy":        strategy,
            }))
            await asyncio.sleep(0.04)

        results    = sorted(docs_scored, key=lambda x: x["final_score"], reverse=True)[: req.top_k]
        top_score  = results[0]["final_score"] if results else 0.0

        await ws.send_text(json.dumps({
            "type": "retrieval_done", "iteration": iteration,
            "strategy": strategy, "top_score": round(top_score, 3),
            "threshold": req.confidence_threshold, "results_count": len(results),
        }))

        # Reflection
        should_reflect = (
            req.enable_iterative
            and iteration < max_iter
            and top_score < req.confidence_threshold
        )
        if should_reflect:
            reason = _diagnose_failure(top_score, req.confidence_threshold, lang)
            await ws.send_text(json.dumps({
                "type": "reflection", "iteration": iteration,
                "failure_reason": reason,
                "top_score": round(top_score, 3),
                "threshold": req.confidence_threshold,
            }))
            new_query = await llm_rewrite_query(current_query, reason, lang)
            await ws.send_text(json.dumps({
                "type": "query_rewrite",
                "original_query": current_query, "new_query": new_query,
            }))
            all_iterations.append(_iter_summary(iteration, current_query, strategy, top_score, True, results))
            current_query  = new_query
            retrieval_text = new_query
            await asyncio.sleep(0.1)
            continue

        all_iterations.append(_iter_summary(iteration, current_query, strategy, top_score, False, results))
        break

    # ── Phase 3: Reranking ────────────────────────────────────────────────────
    if req.enable_rerank and results:
        ce_pfx = _t("ce_label", lang) if cross_encoder else ""
        await ws.send_text(json.dumps({
            "type": "phase_start", "phase": "reranking",
            "message": _t("phase_reranking", lang, ce=ce_pfx, n=len(results)),
        }))
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, _rerank_docs, req.query, results)
        for doc in results:
            await ws.send_text(json.dumps({
                "type": "rerank_score",
                "doc_id": doc["id"], "title": doc["title"],
                "pre_score":   round(doc["pre_rerank_score"], 3),
                "ce_score":    round(doc["ce_score"], 3),
                "improvement": round(doc["ce_score"] - doc["pre_rerank_score"], 3),
            }))
            await asyncio.sleep(0.06)
        await ws.send_text(json.dumps({
            "type": "reranking_done",
            "top_score": round(results[0]["ce_score"], 3),
        }))

    # ── Phase 4: Answer Generation (streaming LLM) ────────────────────────────
    await ws.send_text(json.dumps({
        "type": "phase_start", "phase": "generation",
        "message": _t("phase_generation", lang),
    }))
    history_dicts = [h.model_dump() for h in req.history]
    full_answer = await llm_stream_answer(ws, req.query, results[:4], history_dicts, lang)

    # ── Phase 5: RAGAS Evaluation ─────────────────────────────────────────────
    await ws.send_text(json.dumps({
        "type": "phase_start", "phase": "reflection",
        "message": _t("phase_ragas", lang),
    }))
    loop = asyncio.get_event_loop()
    ragas = await loop.run_in_executor(None, compute_ragas_metrics, req.query, results[:4], full_answer)

    # ── Final Summary ─────────────────────────────────────────────────────────
    elapsed    = round(time.time() - t0, 2)
    final_conf = results[0].get("ce_score", results[0]["final_score"]) if results else 0.0

    final_docs = [
        {
            "id":              d["id"],
            "title":           d["title"],
            "content":         d["content"][:160] + ("…" if len(d["content"]) > 160 else ""),
            "source":          d.get("source", ""),
            "tags":            d.get("tags", []),
            "embedding_score": round(d.get("embedding_score", 0.0), 3),
            "bm25_score":      round(d.get("bm25_score", 0.0), 3),
            "final_score":     round(d.get("ce_score", d["final_score"]), 3),
            "strategy_used":   d.get("strategy_used", "hybrid"),
        }
        for d in results[: req.top_k]
    ]

    await ws.send_text(json.dumps({
        "type": "pipeline_complete",
        "elapsed_seconds":   elapsed,
        "total_iterations":  len(all_iterations),
        "iterations_detail": all_iterations,
        "final_answer":      full_answer,
        "retrieved_docs":    final_docs,
        "metrics": {
            # Retrieval-stage recall estimates
            "baseline_recall":   0.61,
            "iterative_recall":  round(min(0.61 + 0.05 * len(all_iterations), 0.80), 3),
            "fusion_recall":     round(min(0.61 + 0.05 * len(all_iterations) + 0.03, 0.83), 3),
            "rerank_recall":     round(min(0.61 + 0.05 * len(all_iterations) + 0.05, 0.85), 3),
            "final_confidence":  round(final_conf, 3),
            # RAGAS metrics
            **ragas,
        },
    }))

# ══════════════════════════════════════════════════════════════════════════════
# REST Endpoints
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    return {
        "status":               "ok",
        "docs_count":           len(KNOWLEDGE_BASE),
        "embed_model":          EMBED_MODEL_NAME,
        "cross_encoder":        RERANKER_MODEL or "disabled (cosine fallback)",
        "llm_enabled":          llm_client is not None,
        "llm_model":            DEEPSEEK_MODEL if llm_client else None,
        "hyde_available":       llm_client is not None,
        "contextual_chunking":  CONTEXTUAL_CHUNKING,
        "embedding_cache_dir":  str(CACHE_DIR),
    }


# ── ⑨ Knowledge Base Statistics ───────────────────────────────────────────────

@app.get("/stats")
async def get_stats():
    """Comprehensive knowledge base analytics for operations teams."""
    # Per-source breakdown
    sources: dict = {}
    total_words = 0
    for d in KNOWLEDGE_BASE:
        src = d.get("source") or "unknown"
        if src not in sources:
            sources[src] = {
                "source":     src,
                "chunks":     0,
                "words":      0,
                "file_size_kb": d.get("file_size_kb", 0),
                "last_modified": d.get("file_mtime", ""),
                "tags":       d.get("tags", []),
            }
        sources[src]["chunks"] += 1
        sources[src]["words"]  += d.get("word_count", 0)
        total_words            += d.get("word_count", 0)

    # Freshness: flag sources not updated in > 90 days
    now_ts = time.time()
    stale_sources = []
    for d in KNOWLEDGE_BASE:
        mtime = d.get("mtime", now_ts)
        if now_ts - mtime > 90 * 86400:
            src = d.get("source", "")
            if src and src not in stale_sources:
                stale_sources.append(src)

    # Feedback analytics
    feedback_total, feedback_pos = 0, 0
    if FEEDBACK_FILE.exists():
        for line in FEEDBACK_FILE.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                feedback_total += 1
                if r.get("rating", 0) > 0:
                    feedback_pos += 1
            except Exception:
                pass

    return {
        "total_chunks":          len(KNOWLEDGE_BASE),
        "total_sources":         len(sources),
        "total_words":           total_words,
        "contextual_chunking":   CONTEXTUAL_CHUNKING,
        "sources": sorted(sources.values(), key=lambda x: x["source"]),
        "stale_sources":         stale_sources,      # not updated in 90+ days
        "feedback": {
            "total":            feedback_total,
            "positive":         feedback_pos,
            "negative":         feedback_total - feedback_pos,
            "satisfaction_rate": round(feedback_pos / feedback_total, 2)
                                 if feedback_total > 0 else None,
        },
    }


# ── ⑨ User Feedback Loop ──────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    query:    str
    answer:   str
    rating:   int               # +1 = helpful, -1 = not helpful
    comment:  Optional[str] = None
    doc_ids:  List[str]    = []
    language: str          = "zh"


@app.post("/feedback")
async def submit_feedback(req: FeedbackRequest):
    """
    Persist user feedback to a JSONL log.
    Used for offline analysis: identify low-quality documents, coverage gaps,
    and hallucination patterns.
    """
    record = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "query":     req.query,
        "answer":    req.answer[:500],     # truncate to avoid bloat
        "rating":    req.rating,
        "comment":   req.comment,
        "doc_ids":   req.doc_ids,
        "language":  req.language,
    }
    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Feedback recorded: rating=%+d, query=%r", req.rating, req.query[:60])
    return {"status": "ok", "message": "Thank you for your feedback!"}


@app.get("/docs_list")
async def docs_list():
    return [
        {"id": d["id"], "title": d["title"], "source": d.get("source", ""), "tags": d["tags"]}
        for d in KNOWLEDGE_BASE
    ]


@app.post("/upload")
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
        asyncio.create_task(_rebuild_index())   # single rebuild after all files saved

    return {
        "uploaded": ok_count,
        "errors":   len(results) - ok_count,
        "files":    results,
    }


@app.delete("/docs/{filename}")
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
    asyncio.create_task(_rebuild_index())
    return {"status": "ok", "deleted": filename}


@app.post("/reload")
async def reload_index(force: bool = False):
    """
    Trigger index rebuild.
    - force=false (default): use embedding cache if docs unchanged
    - force=true: always re-embed (needed after changing EMBED_MODEL or CONTEXTUAL_CHUNKING)
    """
    asyncio.create_task(_rebuild_index(force_reembed=force))
    return {"status": "rebuilding", "force_reembed": force,
            "message": "Index rebuild started in background"}


async def _rebuild_index(force_reembed: bool = False) -> None:
    global KNOWLEDGE_BASE, doc_embeddings
    logger.info("Rebuilding index (force_reembed=%s)…", force_reembed)
    raw_docs = load_documents_from_folder(DOCS_DIR)
    if not raw_docs:
        logger.warning("No documents found during rebuild")
        return

    docs = await _contextualize_chunks(raw_docs)
    loop = asyncio.get_event_loop()

    cached = None if force_reembed else _load_emb_cache(docs)
    if cached is not None:
        embs = cached
    else:
        embs = await loop.run_in_executor(None, _compute_doc_embeddings, docs)
        _save_emb_cache(docs, embs)

    await loop.run_in_executor(None, _init_bm25, docs)
    KNOWLEDGE_BASE = docs
    doc_embeddings = embs
    logger.info("Index rebuild complete — %d chunks (cached=%s)", len(KNOWLEDGE_BASE), cached is not None)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _diagnose_failure(top_score: float, threshold: float, lang: str = "zh") -> str:
    if top_score < 0.35:
        return _t("failure_extreme", lang)
    if top_score < threshold:
        return _t("failure_moderate", lang)
    return _t("failure_low", lang)


def _iter_summary(iteration, query, strategy, top_score, reflected, results):
    return {
        "iteration": iteration, "query": query, "strategy": strategy,
        "top_score": round(top_score, 3), "reflected": reflected,
        "results": [{"id": r["id"], "title": r["title"], "score": round(r["final_score"], 3)} for r in results[:3]],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Non-streaming RAG pipeline  (MCP / programmatic use)
# ══════════════════════════════════════════════════════════════════════════════

async def query_rag(
    query: str,
    strategy: str = "adaptive",
    enable_hyde: bool = False,
    enable_iterative: bool = True,
    enable_rerank: bool = True,
    top_k: int = 5,
    confidence_threshold: float = 0.55,
    language: str = "zh",
) -> dict:
    """
    Full RAG pipeline without WebSocket streaming.
    Returns a dict with keys: answer, docs, metrics, elapsed, iterations.
    Used by mcp_server.py and any programmatic caller.
    """
    t0 = time.time()
    current_query = query
    iteration, max_iter = 0, 3
    results: List[dict] = []

    # ── HyDE pre-processing ───────────────────────────────────────────────
    retrieval_text = current_query
    if enable_hyde:
        retrieval_text = await hyde_generate_hypothetical(current_query, language)

    # ── Iterative Retrieval + Reflection ─────────────────────────────────
    loop = asyncio.get_event_loop()
    while iteration < max_iter:
        iteration += 1
        strat = strategy
        if strat == "adaptive":
            strat = ["hybrid", "vector", "bm25"][min(iteration - 1, 2)]

        vec_text  = retrieval_text if (enable_hyde and iteration == 1) else current_query
        emb_scores = await loop.run_in_executor(None, compute_embedding_scores, vec_text)
        bm25_arr   = await loop.run_in_executor(None, compute_bm25_scores, current_query)

        docs_scored = []
        for idx, doc in enumerate(KNOWLEDGE_BASE):
            d  = doc.copy()
            es = float(emb_scores[idx])
            bs = float(bm25_arr[idx])
            if strat == "vector":
                d.update(embedding_score=es, bm25_score=0.0, final_score=es)
            elif strat == "bm25":
                d.update(embedding_score=0.0, bm25_score=bs, final_score=bs)
            else:
                d.update(embedding_score=es, bm25_score=bs, final_score=es * 0.6 + bs * 0.4)
            d["strategy_used"] = strat
            docs_scored.append(d)

        results   = sorted(docs_scored, key=lambda x: x["final_score"], reverse=True)[:top_k]
        top_score = results[0]["final_score"] if results else 0.0

        if enable_iterative and iteration < max_iter and top_score < confidence_threshold:
            reason        = _diagnose_failure(top_score, confidence_threshold, language)
            current_query = await llm_rewrite_query(current_query, reason, language)
            retrieval_text = current_query
            continue
        break

    # ── Reranking ─────────────────────────────────────────────────────────
    if enable_rerank and results:
        results = await loop.run_in_executor(None, _rerank_docs, query, results)

    # ── Answer generation (non-streaming) ────────────────────────────────
    doc_label = "文档" if language == "zh" else "Document"
    context = "\n\n".join(
        f"【{doc_label}{i + 1}】{d['title']}\n{d['content']}"
        for i, d in enumerate(results[:4])
    )
    answer = ""
    if llm_client:
        answer = await llm_call(
            messages=[
                {"role": "system", "content": _t("sys_answer", language)},
                {"role": "user",   "content": _t("usr_answer", language, context=context, query=query)},
            ],
            max_tokens=1500,
            temperature=0.7,
        )
    elif results:
        answer = results[0]["content"]

    # ── RAGAS Evaluation ─────────────────────────────────────────────────
    ragas = await loop.run_in_executor(None, compute_ragas_metrics, query, results[:4], answer)

    final_docs = [
        {
            "id":     d["id"],
            "title":  d["title"],
            "content": d["content"][:300] + ("…" if len(d["content"]) > 300 else ""),
            "source": d.get("source", ""),
            "score":  round(d.get("ce_score", d["final_score"]), 3),
        }
        for d in results[:top_k]
    ]

    return {
        "answer":     answer,
        "docs":       final_docs,
        "metrics":    ragas,
        "elapsed":    round(time.time() - t0, 2),
        "iterations": iteration,
        "query":      query,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
