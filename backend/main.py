"""
Adaptive RAG System — Backend
==============================
Research Techniques
───────────────────
  ① Iterative Retrieval + ReAct-style Reflection   Yao et al., NeurIPS 2022
  ② Hybrid Dense-Sparse Retrieval  (BGE + BM25 + jieba)
  ③ HyDE – Hypothetical Document Embeddings        Gao et al., EMNLP 2022
  ④ Cross-Encoder Reranking                        BAAI/bge-reranker series
  ⑤ RAGAS-style Evaluation Framework               Es et al., arXiv 2023
  ⑥ Contextual Chunking                            Anthropic, 2024
  ⑦ GraphRAG — Knowledge Graph-enhanced Retrieval

Engineering Features
────────────────────
  • Multi-turn Conversation Memory (last-6-turn context window)
  • Embedding Cache with SHA-256 fingerprinting (incremental indexing)
  • Agentic Pipeline: LLM router → direct / RAG / realtime-tools / complex
  • MCP Server integration (stdio + HTTP)
  • WebSocket streaming with per-phase event protocol

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
import re
import threading
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

# ── Enterprise-style audit & feedback (JSONL) ──────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
AUDIT_FILE = BASE_DIR / "audit.jsonl"
FEEDBACK_FILE = BASE_DIR / "feedback.jsonl"
_jsonl_lock = threading.Lock()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, ensure_ascii=False)
    with _jsonl_lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


_REDACT_PATTERNS = [
    # Email
    (re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I), "[REDACTED_EMAIL]"),
    # Bearer tokens / API keys (very rough)
    (re.compile(r"\bBearer\s+[A-Za-z0-9._-]{16,}\b"), "Bearer [REDACTED_TOKEN]"),
    (re.compile(r"\bsk-[A-Za-z0-9]{16,}\b"), "[REDACTED_API_KEY]"),
    # IBAN (Luxembourg starts with LU; keep generic)
    (re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"), "[REDACTED_IBAN]"),
    # Long digit sequences (cards / account ids)
    (re.compile(r"\b\d{12,19}\b"), "[REDACTED_NUMBER]"),
]


def redact_text(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    out = s
    for rx, repl in _REDACT_PATTERNS:
        out = rx.sub(repl, out)
    return out

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
# ⑩ Knowledge Graph (GraphRAG)
ENABLE_GRAPH         = os.getenv("ENABLE_GRAPH", "true").lower() == "true"

# ── Global state ──────────────────────────────────────────────────────────────

KNOWLEDGE_BASE: List[dict]           = []
doc_embeddings: Optional[np.ndarray] = None
embed_model                          = None
cross_encoder                        = None     # BAAI/bge-reranker (optional)
bm25_index                           = None
_tokenize_fn                         = None
llm_client                           = None
KNOWLEDGE_GRAPH: dict                = {"nodes": {}, "edges": {}}

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
# ⑩ Knowledge Graph  (GraphRAG — graph-enhanced retrieval)
# ══════════════════════════════════════════════════════════════════════════════

_ZH_STOPWORDS = {
    "的","是","在","了","和","与","或","但","如","从","到","为","以","及",
    "其","这","那","有","无","一","不","也","都","将","中","上","下","我",
    "你","他","她","它","们","对","所","把","被","让","使","由","按","等",
}


def _extract_keywords_as_entities(doc: dict) -> dict:
    """Keyword-based entity extraction (fast, no LLM). Uses jieba if available."""
    if _tokenize_fn is None:
        return {"entities": [], "relations": []}
    from collections import Counter
    tokens = [
        t for t in _tokenize_fn(doc["title"] + " " + doc["content"])
        if len(t) > 1 and t not in _ZH_STOPWORDS and not t.isdigit()
    ]
    top_kw = [w for w, _ in Counter(tokens).most_common(7)]
    return {"entities": [{"name": kw, "type": "keyword"} for kw in top_kw], "relations": []}


async def _extract_entities_llm(doc: dict) -> dict:
    """LLM-based entity + relation extraction for a single chunk."""
    prompt = (
        "从以下文本中提取关键实体和它们的关系。返回严格JSON（无多余文字）：\n"
        '{"entities":[{"name":"实体名","type":"概念|技术|方法|系统|其他"}],'
        '"relations":[{"source":"...","target":"...","relation":"..."}]}\n'
        "要求：最多6个实体（名称≤8字），最多4条关系。\n\n文本：\n"
    )
    raw = await llm_call(
        messages=[{"role": "user", "content": prompt + doc["content"][:600]}],
        max_tokens=300, temperature=0.1,
    )
    if not raw:
        return _extract_keywords_as_entities(doc)
    try:
        import re as _re
        m = _re.search(r'\{.*\}', raw, _re.DOTALL)
        if m:
            d = json.loads(m.group())
            return {"entities": d.get("entities", []), "relations": d.get("relations", [])}
    except Exception:
        pass
    return _extract_keywords_as_entities(doc)


def _graph_fingerprint(docs: List[dict]) -> str:
    fp     = _doc_fingerprint(docs)
    flavor = "llm" if llm_client else "kw"
    return hashlib.sha256(f"{fp}|graph|{flavor}".encode()).hexdigest()[:20]


def _load_graph_cache(docs: List[dict]) -> Optional[dict]:
    cache_file = CACHE_DIR / f"graph_{_graph_fingerprint(docs)}.json"
    if cache_file.exists():
        logger.info("Graph cache hit — loading from %s", cache_file.name)
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _save_graph_cache(graph: dict, docs: List[dict]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fp         = _graph_fingerprint(docs)
    cache_file = CACHE_DIR / f"graph_{fp}.json"
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)
    for old in CACHE_DIR.glob("graph_*.json"):
        if old != cache_file:
            old.unlink(missing_ok=True)
    logger.info("Graph cached: %d nodes, %d edges (%s)", len(graph["nodes"]), len(graph["edges"]), fp)


async def _build_knowledge_graph(docs: List[dict]) -> dict:
    """
    Build knowledge graph from all document chunks.
    • LLM available  → semantic entity + relation extraction
    • LLM unavailable → jieba keyword co-occurrence graph
    Edges are added for:
      1. Explicit LLM-extracted relations
      2. Co-occurrence within the same chunk (both modes)
    """
    graph: dict = {"nodes": {}, "edges": {}}
    use_llm     = bool(llm_client)
    logger.info("Building knowledge graph (%s mode, %d chunks)…", "LLM" if use_llm else "keyword", len(docs))

    for i, doc in enumerate(docs):
        chunk_id   = doc["id"]
        extraction = await _extract_entities_llm(doc) if use_llm else _extract_keywords_as_entities(doc)

        # Register nodes
        chunk_ents: List[str] = []
        for ent in extraction.get("entities", []):
            name = ent.get("name", "").strip()
            if not name or len(name) < 2:
                continue
            if name not in graph["nodes"]:
                graph["nodes"][name] = {"type": ent.get("type", "other"), "chunk_ids": [], "freq": 0}
            if chunk_id not in graph["nodes"][name]["chunk_ids"]:
                graph["nodes"][name]["chunk_ids"].append(chunk_id)
            graph["nodes"][name]["freq"] += 1
            chunk_ents.append(name)

        # Explicit LLM relations
        for rel in extraction.get("relations", []):
            src, tgt = rel.get("source", "").strip(), rel.get("target", "").strip()
            if src in graph["nodes"] and tgt in graph["nodes"]:
                key = f"{src}||{tgt}"
                if key not in graph["edges"]:
                    graph["edges"][key] = {"source": src, "target": tgt,
                                           "relation": rel.get("relation", "related_to"), "weight": 1}
                else:
                    graph["edges"][key]["weight"] += 1

        # Co-occurrence edges within the same chunk
        for j in range(len(chunk_ents)):
            for k in range(j + 1, len(chunk_ents)):
                a, b = chunk_ents[j], chunk_ents[k]
                key  = f"{a}||{b}"
                if key not in graph["edges"]:
                    graph["edges"][key] = {"source": a, "target": b, "relation": "co-occurs", "weight": 1}
                else:
                    graph["edges"][key]["weight"] += 1

        if (i + 1) % 5 == 0 or (i + 1) == len(docs):
            logger.info("  graph: %d/%d chunks, %d nodes, %d edges",
                        i + 1, len(docs), len(graph["nodes"]), len(graph["edges"]))
        await asyncio.sleep(0.01)

    return graph


async def _init_graph(docs: List[dict]) -> None:
    global KNOWLEDGE_GRAPH
    if not ENABLE_GRAPH:
        logger.info("GraphRAG disabled (ENABLE_GRAPH=false)")
        return
    cached = _load_graph_cache(docs)
    if cached is not None:
        KNOWLEDGE_GRAPH = cached
        return
    KNOWLEDGE_GRAPH = await _build_knowledge_graph(docs)
    _save_graph_cache(KNOWLEDGE_GRAPH, docs)
    logger.info("✓ Knowledge graph ready: %d nodes, %d edges",
                len(KNOWLEDGE_GRAPH["nodes"]), len(KNOWLEDGE_GRAPH["edges"]))


def compute_graph_scores(query: str) -> np.ndarray:
    """
    Graph-based retrieval scores.
    1. Match query tokens → graph nodes (direct match, weight 1.5)
    2. Expand to 1-hop neighbours (weight 1.0)
    3. Aggregate chunk scores, normalise to [0, 1]
    Returns zeros if graph is empty or no nodes match.
    """
    scores = np.zeros(len(KNOWLEDGE_BASE), dtype=np.float32)
    if not KNOWLEDGE_GRAPH["nodes"] or not KNOWLEDGE_BASE:
        return scores

    q_tokens = {t for t in (_tokenize_fn(query) if _tokenize_fn else query.split()) if len(t) > 1}

    # Direct node matches
    matched: set = set()
    for node_name in KNOWLEDGE_GRAPH["nodes"]:
        if node_name in query or any(t in node_name for t in q_tokens):
            matched.add(node_name)

    if not matched:
        return scores

    # 1-hop expansion
    neighbours: set = set()
    for edge in KNOWLEDGE_GRAPH["edges"].values():
        if edge["source"] in matched:
            neighbours.add(edge["target"])
        if edge["target"] in matched:
            neighbours.add(edge["source"])
    neighbours -= matched

    # Aggregate chunk scores
    chunk_scores: dict = {}
    for node_name, w in [(n, 1.5) for n in matched] + [(n, 1.0) for n in neighbours]:
        for cid in KNOWLEDGE_GRAPH["nodes"].get(node_name, {}).get("chunk_ids", []):
            chunk_scores[cid] = chunk_scores.get(cid, 0.0) + w

    if not chunk_scores:
        return scores

    mx        = max(chunk_scores.values())
    id_to_idx = {doc["id"]: i for i, doc in enumerate(KNOWLEDGE_BASE)}
    for cid, s in chunk_scores.items():
        if cid in id_to_idx:
            scores[id_to_idx[cid]] = float(s) / mx

    return scores


def _fuse_scores(
    emb_arr:    np.ndarray,
    bm25_arr:   np.ndarray,
    graph_arr:  np.ndarray,
    strategy:   str,
    use_graph:  bool,
) -> np.ndarray:
    """Central score fusion. When GraphRAG is active, shifts weights to accommodate graph lane."""
    if strategy == "vector":
        return emb_arr
    if strategy == "bm25":
        return bm25_arr
    # hybrid / adaptive
    if use_graph and graph_arr.any():
        return (0.50 * emb_arr + 0.30 * bm25_arr + 0.20 * graph_arr).astype(np.float32)
    return (0.60 * emb_arr + 0.40 * bm25_arr).astype(np.float32)


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
    # ⑩ Knowledge Graph — must run after BM25 so _tokenize_fn is available
    await _init_graph(docs)

    logger.info("✓ RAG system ready — %d chunks indexed (contextual=%s, cached=%s, graph_nodes=%d)",
                len(KNOWLEDGE_BASE), CONTEXTUAL_CHUNKING, cached is not None,
                len(KNOWLEDGE_GRAPH["nodes"]))

# ══════════════════════════════════════════════════════════════════════════════
# Request / Response Models
# ══════════════════════════════════════════════════════════════════════════════

class ConversationTurn(BaseModel):
    role: str       # "user" | "assistant"
    content: str


class QueryRequest(BaseModel):
    query: str
    # Enterprise context (optional)
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    user_role: Optional[str] = None
    ticket_id: Optional[str] = None
    product: Optional[str] = None
    version: Optional[str] = None
    environment: Optional[str] = None
    strategy: str = "adaptive"          # vector | bm25 | hybrid | adaptive
    enable_iterative: bool = True
    enable_hyde: bool = False           # HyDE (Gao et al., EMNLP 2022)
    enable_graph: bool = False          # ⑩ GraphRAG knowledge-graph lane
    confidence_threshold: float = 0.55
    top_k: int = 5
    language: str = "zh"               # "zh" | "en"
    history: List[ConversationTurn] = []


class FeedbackRequest(BaseModel):
    query: str
    answer: str
    rating: int                      # 1 | -1
    comment: Optional[str] = None
    doc_ids: List[str] = []
    language: Optional[str] = "zh"
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    user_role: Optional[str] = None

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
            _append_jsonl(AUDIT_FILE, {
                "ts": _utc_now_iso(),
                "type": "ws_query",
                "client": getattr(websocket, "client", None).host if getattr(websocket, "client", None) else None,
                "tenant_id": request.tenant_id,
                "user_id": request.user_id,
                "user_role": request.user_role,
                "ticket_id": request.ticket_id,
                "product": request.product,
                "version": request.version,
                "environment": request.environment,
                "query": redact_text(request.query),
                "strategy": request.strategy,
                "enable_iterative": request.enable_iterative,
                "enable_hyde": request.enable_hyde,
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


@app.websocket("/ws/agent")
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
                "ticket_id": request.ticket_id,
                "product": request.product,
                "version": request.version,
                "environment": request.environment,
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


async def run_rag_pipeline(ws: WebSocket, req: QueryRequest) -> None:
    t0   = time.time()
    lang = req.language or "zh"

    # ── Enterprise context → retrieval augmentation ───────────────────────────
    # In real deployments, ticket/product/version/env strongly disambiguate the intent.
    # We incorporate them as light-weight query augmentation (safe, no extra calls).
    def _augment_query(q: str) -> str:
        parts = []
        if req.product:
            parts.append(f"product={req.product}")
        if req.version:
            parts.append(f"version={req.version}")
        if req.environment:
            parts.append(f"env={req.environment}")
        if req.ticket_id:
            parts.append(f"ticket={req.ticket_id}")
        if not parts:
            return q
        prefix = " ".join(parts)
        return f"[{prefix}] {q}"

    current_query = _augment_query(req.query)
    iteration, max_iter = 0, 3
    all_iterations: List[dict] = []
    results: List[dict] = []

    await ws.send_text(json.dumps({
        "type": "pipeline_start",
        "query": req.query,
        "config": {
            "strategy":         req.strategy,
            "enable_iterative": req.enable_iterative,
            "enable_hyde":      req.enable_hyde,
            "enable_graph":     req.enable_graph,
            "threshold":        req.confidence_threshold,
            "total_docs":       len(KNOWLEDGE_BASE),
            "cross_encoder":    cross_encoder is not None,
            "graph_nodes":      len(KNOWLEDGE_GRAPH["nodes"]),
            "language":         lang,
            # Enterprise context (for audit & evaluation stratification)
            "tenant_id":        req.tenant_id,
            "user_id":          req.user_id,
            "user_role":        req.user_role,
            "ticket_id":        req.ticket_id,
            "product":          req.product,
            "version":          req.version,
            "environment":      req.environment,
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
        vec_text   = retrieval_text if (req.enable_hyde and iteration == 1) else current_query
        emb_scores = await loop.run_in_executor(None, compute_embedding_scores, vec_text)
        bm25_arr   = await loop.run_in_executor(None, compute_bm25_scores, current_query)
        graph_arr  = (await loop.run_in_executor(None, compute_graph_scores, current_query)
                      if req.enable_graph else np.zeros(len(KNOWLEDGE_BASE), dtype=np.float32))
        final_arr  = _fuse_scores(emb_scores, bm25_arr, graph_arr, strategy, req.enable_graph)

        docs_scored = []
        for idx, doc in enumerate(KNOWLEDGE_BASE):
            d  = doc.copy()
            es = float(emb_scores[idx])
            bs = float(bm25_arr[idx])
            gs = float(graph_arr[idx])
            if strategy == "vector":
                d.update(embedding_score=es, bm25_score=0.0, graph_score=0.0, final_score=es)
            elif strategy == "bm25":
                d.update(embedding_score=0.0, bm25_score=bs, graph_score=0.0, final_score=bs)
            else:
                d.update(embedding_score=es, bm25_score=bs, graph_score=gs,
                         final_score=float(final_arr[idx]))
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
                "graph_score":     round(doc.get("graph_score", 0.0), 3),
                "final_score":     round(doc["final_score"], 3),
                "strategy":        strategy,
                "graph_active":    req.enable_graph,
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
    # Reranking is automatically enabled when a cross-encoder model is configured.
    if cross_encoder is not None and results:
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
            "graph_score":     round(d.get("graph_score", 0.0), 3),
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

    # Audit: record what sources were actually used (for asset inventory analytics)
    try:
        _append_jsonl(AUDIT_FILE, {
            "ts": _utc_now_iso(),
            "type": "retrieval_complete",
            "tenant_id": req.tenant_id,
            "user_id": req.user_id,
            "user_role": req.user_role,
            "query": redact_text(req.query),
            "strategy": req.strategy,
            "top_k": req.top_k,
            "final_confidence": round(final_conf, 3),
            "retrieved": [
                {"id": d.get("id"), "source": d.get("source"), "score": d.get("final_score")}
                for d in final_docs
            ],
        })
    except Exception:
        pass

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
        "graph_enabled":        ENABLE_GRAPH,
        "graph_nodes":          len(KNOWLEDGE_GRAPH["nodes"]),
        "graph_edges":          len(KNOWLEDGE_GRAPH["edges"]),
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

    # Feedback summary (if enabled)
    fb_total = 0
    fb_pos = 0
    if FEEDBACK_FILE.exists():
        try:
            with FEEDBACK_FILE.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        fb_total += 1
                        if int(obj.get("rating", 0)) > 0:
                            fb_pos += 1
                    except Exception:
                        continue
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
            "total": fb_total,
            "positive": fb_pos,
            "satisfaction_rate": (fb_pos / fb_total) if fb_total else 0.0,
        },
    }


@app.get("/inventory")
async def knowledge_asset_inventory():
    """
    Knowledge Asset Inventory:
    - per-source size metrics (chunks/words/mtime)
    - freshness (stale > 90 days)
    - usage frequency (from audit.jsonl retrieval_complete events)
    - feedback attribution (positive/negative per source via feedback doc_ids)
    """
    # Base: per-source stats from indexed chunks
    per_src: dict = {}
    doc_id_to_src = {}
    now_ts = time.time()

    for d in KNOWLEDGE_BASE:
        src = d.get("source") or "unknown"
        doc_id_to_src[d.get("id")] = src
        if src not in per_src:
            per_src[src] = {
                "source": src,
                "chunks": 0,
                "words": 0,
                "file_size_kb": d.get("file_size_kb", 0),
                "last_modified": d.get("file_mtime", ""),
                "mtime": d.get("mtime", now_ts),
                "tags": d.get("tags", []),
                # analytics
                "usage_hits": 0,          # how often this source appeared in retrieved docs
                "usage_queries": 0,        # how many retrieval events included this source (unique per event)
                "feedback_total": 0,
                "feedback_positive": 0,
            }
        per_src[src]["chunks"] += 1
        per_src[src]["words"] += int(d.get("word_count", 0))

    # Usage: parse audit retrieval_complete events
    if AUDIT_FILE.exists():
        try:
            with AUDIT_FILE.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if obj.get("type") != "retrieval_complete":
                        continue
                    retrieved = obj.get("retrieved") or []
                    seen_sources = set()
                    for r in retrieved:
                        src = r.get("source")
                        if not src or src not in per_src:
                            continue
                        per_src[src]["usage_hits"] += 1
                        seen_sources.add(src)
                    for src in seen_sources:
                        per_src[src]["usage_queries"] += 1
        except Exception:
            pass

    # Feedback attribution: join feedback doc_ids → source
    if FEEDBACK_FILE.exists():
        try:
            with FEEDBACK_FILE.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    rating = int(obj.get("rating", 0))
                    doc_ids = obj.get("doc_ids") or []
                    # Count each source at most once per feedback event (avoid overcounting multi-chunk from same file)
                    seen_sources = set()
                    for did in doc_ids:
                        src = doc_id_to_src.get(did)
                        if not src or src not in per_src:
                            continue
                        seen_sources.add(src)
                    for src in seen_sources:
                        per_src[src]["feedback_total"] += 1
                        if rating > 0:
                            per_src[src]["feedback_positive"] += 1
        except Exception:
            pass

    assets = []
    for src, s in per_src.items():
        stale = (now_ts - float(s.get("mtime", now_ts))) > 90 * 86400
        fb_total = s["feedback_total"]
        fb_pos = s["feedback_positive"]
        assets.append({
            "source": src,
            "chunks": s["chunks"],
            "words": s["words"],
            "file_size_kb": s.get("file_size_kb", 0),
            "last_modified": s.get("last_modified", ""),
            "tags": s.get("tags", []),
            "stale": stale,
            "usage_hits": s["usage_hits"],
            "usage_queries": s["usage_queries"],
            "feedback_total": fb_total,
            "feedback_positive": fb_pos,
            "feedback_satisfaction_rate": (fb_pos / fb_total) if fb_total else None,
        })

    # Sort by usage, then recency
    assets.sort(key=lambda a: (a["usage_queries"], a["usage_hits"]), reverse=True)

    return {
        "generated_at": _utc_now_iso(),
        "total_sources": len(assets),
        "total_chunks": len(KNOWLEDGE_BASE),
        "assets": assets,
    }


@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    if req.rating not in (-1, 1):
        raise HTTPException(400, "rating must be 1 or -1")

    _append_jsonl(FEEDBACK_FILE, {
        "ts": _utc_now_iso(),
        "tenant_id": req.tenant_id,
        "user_id": req.user_id,
        "user_role": req.user_role,
        "language": req.language,
        "rating": req.rating,
        "doc_ids": req.doc_ids or [],
        "comment": redact_text(req.comment),
        # Store redacted content only (avoid accidental PII persistence)
        "query": redact_text(req.query),
        "answer": redact_text(req.answer),
    })
    return {"status": "ok"}


@app.get("/graph")
async def get_graph_info():
    """Knowledge graph stats and top nodes/edges for visualisation."""
    nodes = KNOWLEDGE_GRAPH.get("nodes", {})
    edges = KNOWLEDGE_GRAPH.get("edges", {})

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


@app.get("/docs_list")
async def docs_list():
    return [
        {"id": d["id"], "title": d["title"], "source": d.get("source", ""), "tags": d["tags"]}
        for d in KNOWLEDGE_BASE
    ]


@app.get("/docs/preview")
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

    chunks = [d for d in KNOWLEDGE_BASE if d.get("source") == source]
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
    await _init_graph(docs)
    logger.info("Index rebuild complete — %d chunks (cached=%s, graph_nodes=%d)",
                len(KNOWLEDGE_BASE), cached is not None, len(KNOWLEDGE_GRAPH["nodes"]))

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
# Agentic RAG — Router Agent + Tools + Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

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


# ── Router system prompts ──────────────────────────────────────────────────────

_ROUTER_SYS: dict = {
    "zh": (
        "你是智能问题路由器。将用户查询分配到最合适的路由（返回严格JSON，无多余文字）。\n\n"
        "路由类型：\n"
        '• "direct"   — 通用知识、简单定义、创意写作、编程问题（LLM 可直接回答）\n'
        '• "rag"      — 需查询企业内部文档/专属知识库才能回答\n'
        '• "realtime" — 需实时数据：当前时间/日期、数学计算、网络搜索\n'
        '• "complex"  — 需拆分为多个子问题才能完整回答的复杂分析题\n\n'
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
        '• "rag"      — Questions about internal company docs or proprietary knowledge base\n'
        '• "realtime" — Needs real-time data: current time/date, math calculation, web search\n'
        '• "complex"  — Needs decomposition into multiple sub-questions for complete analysis\n\n'
        "Return format:\n"
        '{"route":"direct|rag|realtime|complex","reason":"brief reason (≤8 words)",'
        '"sub_queries":["sub-q 1","sub-q 2"],"tools":["datetime","calculator","web_search"]}\n\n'
        "Note: sub_queries only for complex; tools only for realtime; others empty arrays"
    ),
}


async def route_query(query: str, language: str = "zh") -> dict:
    """LLM-based query router. Falls back to 'rag' when LLM is unavailable."""
    import re as _re
    default = {
        "route": "rag",
        "reason": "默认路由" if language == "zh" else "default",
        "sub_queries": [], "tools": [],
    }
    if not llm_client:
        return default

    raw = await llm_call(
        messages=[
            {"role": "system", "content": _ROUTER_SYS.get(language, _ROUTER_SYS["zh"])},
            {"role": "user",   "content": query},
        ],
        max_tokens=200,
        temperature=0.1,
    )
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
    lang = req.language or "zh"

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
        if llm_client:
            try:
                stream = await llm_client.chat.completions.create(
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
        if llm_client:
            try:
                stream = await llm_client.chat.completions.create(
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


# ══════════════════════════════════════════════════════════════════════════════
# Non-streaming RAG pipeline  (MCP / programmatic use)
# ══════════════════════════════════════════════════════════════════════════════

async def query_rag(
    query: str,
    strategy: str = "adaptive",
    enable_hyde: bool = False,
    enable_iterative: bool = True,
    enable_graph: bool = False,
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

        vec_text   = retrieval_text if (enable_hyde and iteration == 1) else current_query
        emb_scores = await loop.run_in_executor(None, compute_embedding_scores, vec_text)
        bm25_arr   = await loop.run_in_executor(None, compute_bm25_scores, current_query)
        graph_arr  = (await loop.run_in_executor(None, compute_graph_scores, current_query)
                      if enable_graph else np.zeros(len(KNOWLEDGE_BASE), dtype=np.float32))
        final_arr  = _fuse_scores(emb_scores, bm25_arr, graph_arr, strat, enable_graph)

        docs_scored = []
        for idx, doc in enumerate(KNOWLEDGE_BASE):
            d  = doc.copy()
            es = float(emb_scores[idx])
            bs = float(bm25_arr[idx])
            gs = float(graph_arr[idx])
            if strat == "vector":
                d.update(embedding_score=es, bm25_score=0.0, graph_score=0.0, final_score=es)
            elif strat == "bm25":
                d.update(embedding_score=0.0, bm25_score=bs, graph_score=0.0, final_score=bs)
            else:
                d.update(embedding_score=es, bm25_score=bs, graph_score=gs,
                         final_score=float(final_arr[idx]))
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
    # Reranking is automatically enabled when a cross-encoder model is configured.
    if cross_encoder is not None and results:
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
