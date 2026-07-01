# ⚡ Adaptive RAG System

> Enterprise Knowledge Base with Adaptive Retrieval-Augmented Generation  
> 企业知识库自适应检索增强生成系统

HyDE · Hybrid Retrieval (BGE + BM25) · Iterative Reflection · GraphRAG · Cross-Encoder Rerank · Agentic Routing · RAGAS

---

## Architecture

```
Frontend :3000  ──WebSocket──►  Backend :8000
                                  │
                    ws/query ──► RAG Pipeline
                    ws/agent ──► Router → direct | rag | tools | multi-step
                                  │
                    Index: chunk → embed (local BGE) → BM25 → [graph]
                    Docs: backend/docs/  (.txt / .md / .pdf)
```

**RAG 流程：** HyDE（可选）→ 多策略检索（最多 3 轮，低置信度则改写 query）→ 重排（需配置 `RERANKER_MODEL`）→ LLM 生成 → RAGAS 评估

---

## Quick Start

```bash
cp backend/.env.example backend/.env
# 填写 DEEPSEEK_API_KEY（无 key 时可检索，但无法生成答案 / HyDE / 改写）

chmod +x start.sh && ./start.sh
```

- 前端：http://localhost:3000  
- API：http://localhost:8000/docs  

文档放入 `backend/docs/` 后自动索引；热重载：

```bash
curl -X POST http://localhost:8000/reload
```

**环境要求：** Python 3.9+ · Node 18+ · RAM ≥ 4 GB

---

## Configuration

详见 `backend/.env.example`。常用项：

| 变量 | 说明 |
|------|------|
| `DEEPSEEK_API_KEY` | LLM 答案、HyDE、query 改写 |
| `EMBED_MODEL` | 默认 `BAAI/bge-small-zh-v1.5`（本地 embedding） |
| `RERANKER_MODEL` | 可选，如 `BAAI/bge-reranker-base` |
| `MAX_CHUNK_CHARS` | 分块大小，默认 600 |
| `CONTEXTUAL_CHUNKING` | 索引时用 LLM 为 chunk 添加上下文 |
| `DOCS_DIR` | 文档目录，默认 `backend/docs/` |

Embedding 在本地计算；审计与反馈日志会对 query/answer 做脱敏。

---

## API

### WebSocket

| 路径 | 用途 |
|------|------|
| `/ws/query` | 标准 RAG 流式管道 |
| `/ws/agent` | 智能路由（检索 / 直答 / 工具 / 多步） |

请求字段：`query`、`strategy`（`adaptive` \| `hybrid` \| `vector` \| `bm25`）、`enable_iterative`、`enable_hyde`、`enable_graph`、`confidence_threshold`、`top_k`、`language`、`history`。

主要事件：`pipeline_complete`（含答案、文档、RAGAS 指标）、`answer_token`、`doc_scored`、`query_rewrite`。

### REST

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 服务与模型状态 |
| `GET` | `/stats` | 知识库统计 |
| `GET` | `/docs_list` | 已索引文档列表 |
| `POST` | `/upload` | 上传文档 |
| `DELETE` | `/docs/{filename}` | 删除文档 |
| `POST` | `/reload` | 重建索引（`?force=true` 忽略 embedding 缓存） |
| `POST` | `/feedback` | 用户反馈（±1） |

---

## MCP

将知识库暴露给 Claude / Cursor 等 MCP 客户端：

```bash
python backend/mcp_server.py              # stdio
python backend/mcp_server.py --http --port 8001
```

工具：`search_knowledge_base` · `retrieve_documents` · `list_documents` · `get_kb_stats`

客户端配置见 `mcp_config_example.json`。

---

## Project Structure

```
rag_system/
├── backend/main.py          # API + RAG / Agent 管道
├── backend/mcp_server.py
├── backend/docs/            # 默认知识库
├── frontend/src/App.jsx
├── start.sh
└── mcp_config_example.json
```

**Tech stack：** FastAPI · sentence-transformers · rank-bm25 · jieba · DeepSeek · React · Vite
