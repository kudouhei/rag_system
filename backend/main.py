"""
Adaptive RAG System — Backend
==============================
Research Techniques
───────────────────
  ① Iterative Retrieval + ReAct-style Reflection   Yao et al., NeurIPS 2022
  ② Hybrid Dense-Sparse Retrieval  (BGE + BM25 + jieba)
  ③ Cross-Encoder Reranking                        BAAI/bge-reranker series
  ④ RAGAS-style Evaluation Framework               Es et al., arXiv 2023
  ⑤ Contextual Chunking                            Anthropic, 2024
  ⑥ GraphRAG — Knowledge Graph-enhanced Retrieval

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

Package Layout  (backend/app/)
──────────────────────────────
  core/         config, shared runtime state, Pydantic schemas, i18n messages,
                audit/feedback JSONL logging + PII redaction
  ingestion/    document loading & chunking, Anthropic contextual retrieval
  retrieval/    dense (embeddings) + sparse (BM25) + graph retrieval, score
                fusion, cross-encoder reranking
  evaluation/   RAGAS-style evaluation metrics
  llm/          DeepSeek client, query rewrite, streaming answers
  pipeline/     index build/rebuild orchestration, the adaptive RAG pipeline,
                the agentic router + orchestrator, agent tool implementations
  routes/       FastAPI routers (REST + WebSocket)

main.py         app wiring — creates the FastAPI app and mounts routes (this file)
mcp_server.py   MCP server exposing the same RAG pipeline as MCP tools
"""
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.pipeline.indexing import startup as _startup
from app.routes import router as api_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await _startup()
    yield


app = FastAPI(title="Adaptive RAG System", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)
app.include_router(api_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
