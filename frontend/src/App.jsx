import { useState, useRef, useEffect, useCallback } from "react";

// ── Backend API ───────────────────────────────────────────────────────────────
// In dev, Vite proxies /stats, /upload, /ws, … to 127.0.0.1:8000 (see vite.config.js).
// In production builds, call the API on localhost:8000 unless VITE_API_BASE is set.
const API_BASE =
  (typeof import.meta.env.VITE_API_BASE === "string" && import.meta.env.VITE_API_BASE.trim()) ||
  (import.meta.env.DEV ? "" : "http://localhost:8000");

function backendWsUrl(path) {
  const custom = typeof import.meta.env.VITE_API_BASE === "string" && import.meta.env.VITE_API_BASE.trim();
  if (custom) {
    const u = custom.replace(/\/$/, "");
    const wsOrigin = u.startsWith("https") ? u.replace(/^https/, "wss") : u.replace(/^http/, "ws");
    return `${wsOrigin}${path}`;
  }
  if (import.meta.env.DEV) {
    const scheme = window.location.protocol === "https:" ? "wss" : "ws";
    return `${scheme}://${window.location.host}${path}`;
  }
  return `ws://localhost:8000${path}`;
}

// ── Color palette ──────────────────────────────────────────────────────────────
const C = {
  bg:           "#ffffff",
  surface:      "#f8fafc",
  surfaceHover: "#f1f5f9",
  border:       "#e2e8f0",
  borderBright: "#cbd5e1",
  accent:       "#0284c7",
  accentDim:    "#0369a1",
  green:        "#059669",
  orange:       "#ea580c",
  red:          "#dc2626",
  purple:       "#7c3aed",
  teal:         "#0891b2",
  text:         "#0f172a",
  textMid:      "#64748b",
  textDim:      "#94a3b8",
};

// ── Internationalisation ──────────────────────────────────────────────────────
const I18N = {
  zh: {
    appSubtitle:        "HyDE · 迭代检索 · 交叉编码器 · RAGAS 评估",
    badge_conv:         "多轮对话",
    queryLabel:         "QUERY INPUT",
    queryPlaceholder:   "输入问题… (Enter 发送)",
    configLabel:        "CONFIGURATION",
    strategyLabel:      "检索策略",
    strategies:         { adaptive:"自适应", hybrid:"混合", vector:"向量", bm25:"BM25" },
    toggle_iterative:   "迭代检索",
    toggle_rerank:      "精排 Rerank",
    toggle_hyde:        "HyDE 增强",
    toggle_conv:        "对话模式",
    tip_iterative:      "ReAct 风格的反思式迭代检索 (Yao et al., 2022)",
    tip_rerank:         "Cross-Encoder 精排（BAAI/bge-reranker）",
    tip_hyde:           "Hypothetical Document Embeddings (Gao et al., EMNLP 2022)",
    tip_conv:           "多轮对话：保留历史上下文",
    thresholdLabel:     "置信度阈值",
    convCtx:            (n) => `💬 携带 ${n} 轮对话上下文`,
    clearBtn:           "清空",
    runBtn:             "⚡ 执行检索",
    runningBtn:         "检索中…",
    backendOffline:     "后端未就绪：请先启动后端并等待其完成初始化（Embedding/索引加载）。",
    backendOnline:      "后端已就绪",
    stat_elapsed:       "耗时",
    stat_iters:         "迭代轮次",
    stat_docs:          "召回文档",
    tab_process:        "执行过程",
    tab_results:        "检索结果",
    tab_metrics:        "效果分析",
    tab_conv:           "对话历史",
    logTitle:           "EXECUTION LOG",
    logEmpty:           "等待执行…",
    hydeTitle:          "HyDE HYPOTHETICAL DOCUMENT",
    answerTitle:        "GENERATED ANSWER",
    iterTitle:          "ITERATION TRACE",
    iterEmpty:          "暂无迭代数据",
    resultsTitle:       (n) => `TOP-${n} 检索结果（已精排）`,
    resEmpty_idle:      "请先执行检索",
    resEmpty_run:       "检索中…",
    resEmpty_done:      "无结果",
    score_vec:          "向量",
    score_bm25:         "BM25",
    ragasTitle:         "RAGAS EVALUATION",
    ragasOverall:       (p) => `综合 ${p}%`,
    cr_label:           "Context Relevance",
    cp_label:           "Context Precision",
    ar_label:           "Answer Relevance",
    af_label:           "Answer Faithfulness",
    cr_desc:            "检索文档与查询的平均语义相似度 (Es et al., 2023)",
    cp_desc:            "检索结果中真正相关文档的比例",
    ar_desc:            "生成答案与查询问题的语义匹配程度",
    af_desc:            "答案内容与检索文档的一致性（幻觉检测代理指标）",
    recallTitle:        "RECALL@10 提升对比",
    recallSub:          "(Natural Questions)",
    r_base:             "基线",
    r_iter:             "迭代检索",
    r_fus:              "多策略",
    r_re:               "重排序",
    qMetricsTitle:      "本次查询指标",
    m_conf:             "最终置信度",
    m_iter:             "迭代后召回率",
    m_fus:              "融合后召回率",
    m_re:               "精排后召回率",
    modTitle:           "模块贡献分析",
    mod_iter:           "迭代式检索 (ReAct)",
    mod_fus:            "多策略融合检索",
    mod_ce:             "Cross-Encoder 精排",
    note_dataset:       "评测基准：Natural Questions",
    note_iter:          (n) => `迭代轮次：${n} 轮`,
    note_strat:         (s) => `检索策略：${s==="adaptive"?"自适应（动态切换）":s}`,
    note_hyde:          (on) => `HyDE 增强：${on?"已启用":"未启用"}`,
    note_hist:          (n) => `对话历史：${n>0?`${n} 轮`:"无"}`,
    metricsEmpty:       "请先执行检索以查看效果分析",
    convHistTitle:      (n) => `CONVERSATION HISTORY (${n} turns)`,
    convEmpty:          "暂无对话记录。开启「对话模式」后，每次问答将自动保存在此。",
    role_user:          "YOU",
    role_asst:          "ASSISTANT",
    // Knowledge Base stats
    kbTitle:            "知识库状态",
    kbChunks:           "个分块",
    kbDocs:             "个文档",
    kbWords:            "词",
    kbContextual:       "上下文增强",
    kbStale:            "条文档可能过期（>90天）",
    kbRebuilding:       "正在重建索引…",
    kbRebuildBtn:       "重建索引",
    kbForceBtn:         "强制重新嵌入",
    kbSatRate:          (p) => `好评率 ${p}%`,
    kbNoFeedback:       "暂无反馈",
    kbLoading:          "加载知识库信息…",
    kbFeedbackTotal:    (n) => `${n} 条反馈`,
    // Answer feedback
    feedbackTitle:      "这个回答有帮助吗？",
    feedback_yes:       "有帮助",
    feedback_no:        "没有帮助",
    feedback_thanks:    "感谢你的反馈！",
    feedback_comment:   "添加备注（可选）",
    // Docs tab
    tab_docs:           "文档",
    docsUploadTitle:    "上传文档",
    docsDropHint:       "将文件拖放到此处，或点击选择文件",
    docsDropActive:     "松开以添加文件",
    docsAllowedTypes:   "支持：.txt  .md  .pdf  （可多选）",
    docsUploadBtn:      (n) => `上传 ${n} 个文件`,
    docsUploading:      "上传中…",
    docsUploadDone:     (ok, err) => `完成：${ok} 成功${err>0?`，${err} 失败`:""}`,
    docsCurrentTitle:   "当前文档",
    docsEmpty:          "知识库中暂无文档",
    docsChunks:         (n) => `${n} 块`,
    docsWords:          (n) => `~${n} 词`,
    docsDeleteBtn:      "删除",
    docsDeleteConfirm:  (f) => `确认删除「${f}」？删除后将重建索引。`,
    docsDeleting:       "删除中…",
    docsRebuildNotice:  "上传/删除后将自动重建索引，请稍等",
    sampleQueries: [
      "企业知识库如何实现高效检索？",
      "HyDE 假设文档嵌入的原理是什么？",
      "如何评估 RAG 系统的召回率？",
      "交叉编码器重排序的优势在哪里？",
    ],
    // Agent mode
    toggle_graph:       "Graph RAG",
    tip_graph:          "知识图谱检索：在向量+BM25的基础上叠加实体关系图，增强跨文档推理",
    badge_graph:        "Graph RAG",
    score_graph:        "Graph",
    toggle_agent:       "Agent 模式",
    tip_agent:          "智能路由：自动判断直接回答/知识库检索/实时工具/多步推理",
    badge_agent:        "Agentic RAG",
    agent_route_label:  "路由决策",
    agent_routes: {
      direct:   { label:"直接回答",   color:"#059669", icon:"💡" },
      rag:      { label:"知识库检索", color:"#0284c7", icon:"📚" },
      realtime: { label:"实时工具",   color:"#ea580c", icon:"⚡" },
      complex:  { label:"多步推理",   color:"#7c3aed", icon:"🧩" },
    },
    log_agent_routing:  "分析问题类型…",
    log_agent_route:    (e) => {
      const r = { direct:"直接回答", rag:"知识库检索", realtime:"实时工具", complex:"多步推理" };
      return <><span style={{fontWeight:700}}>路由→ {r[e.route]||e.route}</span>&nbsp;<span style={{color:C.textMid}}>({e.reason})</span></>;
    },
    log_agent_tool:     (e) => <><span style={{color:C.orange,fontWeight:700}}>🔧 {e.tool}</span>&nbsp;<span style={{color:C.textMid}}>调用中…</span></>,
    log_agent_result:   (e) => <><span style={{color:C.green,fontWeight:700}}>✓ {e.tool}</span>&nbsp;<span style={{color:C.textMid,fontSize:11,fontFamily:"monospace"}}>{e.result?.slice(0,80)}…</span></>,
    log_agent_decomp:   (e) => <><span style={{color:C.purple,fontWeight:700}}>🧩 拆解：</span>&nbsp;{e.sub_queries?.join(" / ")}</>,
    log_agent_subq:     (e) => <><span style={{color:C.textMid}}>子任务 {e.index}/{e.total}：</span><span style={{color:C.accent}}>「{e.query}」</span></>,
    log_agent_subr:     (e) => <><span style={{color:C.green}}>✓ 子任务 {e.index} 完成</span>&nbsp;<span style={{color:C.textMid,fontSize:11}}>{e.answer_preview?.slice(0,60)}…</span></>,
    log_agent_done:     (e) => <span style={{color:C.green,fontWeight:700}}>✓ Agent 完成（{e.route}），{e.elapsed_seconds}s</span>,
    agent_sub_title:    "SUB-TASK RESULTS",
    // LogEntry dynamic strings
    log_start:          (q) => <>开始处理：<em style={{color:C.accent}}>「{q}」</em></>,
    log_hyde:           (doc) => <><span style={{color:C.teal,fontWeight:700}}>HyDE 假设文档：</span>&nbsp;<span style={{color:C.textMid,fontStyle:"italic"}}>「{doc.slice(0,80)}{doc.length>80?"…":""}」</span></>,
    log_docScored:      (e) => <><span style={{color:C.textMid}}>{e.title.slice(0,22)}…</span>&nbsp;→&nbsp;<span style={{color:C.accent}}>向量 {(e.embedding_score*100).toFixed(0)}%</span>&nbsp;<span style={{color:C.green}}>BM25 {(e.bm25_score*100).toFixed(0)}%</span>{e.graph_score>0&&<>&nbsp;<span style={{color:"#059669"}}>Graph {(e.graph_score*100).toFixed(0)}%</span></>}&nbsp;<span style={{color:C.text,fontWeight:700}}>综合 {(e.final_score*100).toFixed(0)}%</span></>,
    log_retDone:        (e) => <><span>第 {e.iteration} 轮检索，最高分：</span><span style={{color:e.top_score>=e.threshold?C.green:C.orange,fontWeight:700}}>{(e.top_score*100).toFixed(1)}%</span><span> (阈值 {(e.threshold*100).toFixed(0)}%)</span></>,
    log_reflect:        (e) => <><span style={{color:C.orange}}>反思：</span>&nbsp;{e.failure_reason}</>,
    log_rewrite:        (e) => <><span>查询重写：</span><span style={{color:C.textMid,textDecoration:"line-through"}}>「{e.original_query}」</span>&nbsp;→&nbsp;<span style={{color:C.accent}}>「{e.new_query}」</span></>,
    log_rerank:         (e) => <><span style={{color:C.textMid}}>{e.title.slice(0,20)}…</span>&nbsp;精排：<span style={{color:C.purple,fontWeight:700}}>{(e.ce_score*100).toFixed(1)}%</span>&nbsp;<span style={{color:e.improvement>0?C.green:C.orange,fontSize:11}}>({e.improvement>0?"+":""}{(e.improvement*100).toFixed(1)}%)</span></>,
    log_rerankDone:     (e) => <><span>重排完成，最高：</span><span style={{color:C.purple,fontWeight:700}}>{(e.top_score*100).toFixed(1)}%</span></>,
    log_done:           (e) => <span style={{color:C.green,fontWeight:700}}>✓ 完成，{e.elapsed_seconds}s，{e.total_iterations} 轮迭代</span>,
  },
  en: {
    appSubtitle:        "HyDE · Iterative Retrieval · Cross-Encoder · RAGAS Evaluation",
    badge_conv:         "Multi-turn Conv.",
    queryLabel:         "QUERY INPUT",
    queryPlaceholder:   "Ask a question… (Enter to send)",
    configLabel:        "CONFIGURATION",
    strategyLabel:      "Retrieval Strategy",
    strategies:         { adaptive:"Adaptive", hybrid:"Hybrid", vector:"Vector", bm25:"BM25" },
    toggle_iterative:   "Iterative",
    toggle_rerank:      "Reranking",
    toggle_hyde:        "HyDE",
    toggle_conv:        "Chat Mode",
    tip_iterative:      "ReAct-style reflective iterative retrieval (Yao et al., 2022)",
    tip_rerank:         "Cross-Encoder reranking (BAAI/bge-reranker)",
    tip_hyde:           "Hypothetical Document Embeddings (Gao et al., EMNLP 2022)",
    tip_conv:           "Multi-turn conversation: retain context across queries",
    thresholdLabel:     "Confidence Threshold",
    convCtx:            (n) => `💬 ${n} turn(s) of history`,
    clearBtn:           "Clear",
    runBtn:             "⚡ Run Retrieval",
    runningBtn:         "Retrieving…",
    backendOffline:     "Backend not ready. Start the API and wait for initialization (embeddings/index).",
    backendOnline:      "Backend ready",
    stat_elapsed:       "Elapsed",
    stat_iters:         "Iterations",
    stat_docs:          "Docs",
    tab_process:        "Process",
    tab_results:        "Results",
    tab_metrics:        "Analytics",
    tab_conv:           "History",
    logTitle:           "EXECUTION LOG",
    logEmpty:           "Waiting for query…",
    hydeTitle:          "HyDE HYPOTHETICAL DOCUMENT",
    answerTitle:        "GENERATED ANSWER",
    iterTitle:          "ITERATION TRACE",
    iterEmpty:          "No iterations yet",
    resultsTitle:       (n) => `TOP-${n} RETRIEVED DOCS (reranked)`,
    resEmpty_idle:      "Run a query to see results",
    resEmpty_run:       "Retrieving…",
    resEmpty_done:      "No results",
    score_vec:          "Vector",
    score_bm25:         "BM25",
    ragasTitle:         "RAGAS EVALUATION",
    ragasOverall:       (p) => `Overall ${p}%`,
    cr_label:           "Context Relevance",
    cp_label:           "Context Precision",
    ar_label:           "Answer Relevance",
    af_label:           "Answer Faithfulness",
    cr_desc:            "Avg. semantic similarity between retrieved docs and the query (Es et al., 2023)",
    cp_desc:            "Fraction of retrieved docs that are genuinely relevant",
    ar_desc:            "Semantic alignment between the generated answer and the query",
    af_desc:            "How grounded the answer is in retrieved docs (hallucination proxy)",
    recallTitle:        "RECALL@10 IMPROVEMENT",
    recallSub:          "(Natural Questions)",
    r_base:             "Baseline",
    r_iter:             "Iterative",
    r_fus:              "Fusion",
    r_re:               "Reranked",
    qMetricsTitle:      "Query Metrics",
    m_conf:             "Final Confidence",
    m_iter:             "Recall (iterative)",
    m_fus:              "Recall (fusion)",
    m_re:               "Recall (reranked)",
    modTitle:           "Module Contributions",
    mod_iter:           "Iterative Retrieval (ReAct)",
    mod_fus:            "Multi-strategy Fusion",
    mod_ce:             "Cross-Encoder Reranking",
    note_dataset:       "Benchmark: Natural Questions",
    note_iter:          (n) => `Iterations: ${n}`,
    note_strat:         (s) => `Strategy: ${s==="adaptive"?"Adaptive (auto-switch)":s}`,
    note_hyde:          (on) => `HyDE: ${on?"Enabled":"Disabled"}`,
    note_hist:          (n) => `History: ${n>0?`${n} turn(s)`:"None"}`,
    metricsEmpty:       "Run a query to view analytics",
    convHistTitle:      (n) => `CONVERSATION HISTORY (${n} turns)`,
    convEmpty:          "No conversation yet. Enable Chat Mode to save Q&A pairs here.",
    role_user:          "YOU",
    role_asst:          "ASSISTANT",
    // Knowledge Base stats
    kbTitle:            "Knowledge Base",
    kbChunks:           "chunks",
    kbDocs:             "sources",
    kbWords:            "words",
    kbContextual:       "Contextual",
    kbStale:            "source(s) may be stale (>90 days)",
    kbRebuilding:       "Rebuilding index…",
    kbRebuildBtn:       "Rebuild Index",
    kbForceBtn:         "Force Re-embed",
    kbSatRate:          (p) => `${p}% satisfied`,
    kbNoFeedback:       "No feedback yet",
    kbLoading:          "Loading knowledge base…",
    kbFeedbackTotal:    (n) => `${n} feedback(s)`,
    // Answer feedback
    feedbackTitle:      "Was this answer helpful?",
    feedback_yes:       "Helpful",
    feedback_no:        "Not helpful",
    feedback_thanks:    "Thanks for your feedback!",
    feedback_comment:   "Add a comment (optional)",
    // Docs tab
    tab_docs:           "Docs",
    docsUploadTitle:    "Upload Documents",
    docsDropHint:       "Drop files here, or click to browse",
    docsDropActive:     "Release to add files",
    docsAllowedTypes:   "Supported: .txt  .md  .pdf  (multi-select allowed)",
    docsUploadBtn:      (n) => `Upload ${n} file${n!==1?"s":""}`,
    docsUploading:      "Uploading…",
    docsUploadDone:     (ok, err) => `Done: ${ok} uploaded${err>0?`, ${err} failed`:""}`,
    docsCurrentTitle:   "Knowledge Base Documents",
    docsEmpty:          "No documents in the knowledge base yet",
    docsChunks:         (n) => `${n} chunks`,
    docsWords:          (n) => `~${n} words`,
    docsDeleteBtn:      "Delete",
    docsDeleteConfirm:  (f) => `Delete "${f}"? The index will be rebuilt.`,
    docsDeleting:       "Deleting…",
    docsRebuildNotice:  "Index rebuilds automatically after upload or delete",
    sampleQueries: [
      "How to implement efficient enterprise knowledge retrieval?",
      "What is the HyDE hypothetical document embedding technique?",
      "How do you evaluate the recall rate of a RAG system?",
      "What are the advantages of cross-encoder reranking?",
    ],
    // Agent mode
    toggle_graph:       "Graph RAG",
    tip_graph:          "Knowledge graph retrieval: adds entity-relation graph lane on top of dense+sparse",
    badge_graph:        "Graph RAG",
    score_graph:        "Graph",
    toggle_agent:       "Agent Mode",
    tip_agent:          "Smart routing: auto-selects direct answer / RAG / realtime tools / multi-step reasoning",
    badge_agent:        "Agentic RAG",
    agent_route_label:  "Route Decision",
    agent_routes: {
      direct:   { label:"Direct Answer", color:"#059669", icon:"💡" },
      rag:      { label:"RAG Retrieval", color:"#0284c7", icon:"📚" },
      realtime: { label:"Realtime Tools", color:"#ea580c", icon:"⚡" },
      complex:  { label:"Multi-step",    color:"#7c3aed", icon:"🧩" },
    },
    log_agent_routing:  "Analyzing query intent…",
    log_agent_route:    (e) => {
      const r = { direct:"Direct Answer", rag:"RAG Retrieval", realtime:"Realtime Tools", complex:"Multi-step" };
      return <><span style={{fontWeight:700}}>Route → {r[e.route]||e.route}</span>&nbsp;<span style={{color:C.textMid}}>({e.reason})</span></>;
    },
    log_agent_tool:     (e) => <><span style={{color:C.orange,fontWeight:700}}>🔧 {e.tool}</span>&nbsp;<span style={{color:C.textMid}}>calling…</span></>,
    log_agent_result:   (e) => <><span style={{color:C.green,fontWeight:700}}>✓ {e.tool}</span>&nbsp;<span style={{color:C.textMid,fontSize:11,fontFamily:"monospace"}}>{e.result?.slice(0,80)}…</span></>,
    log_agent_decomp:   (e) => <><span style={{color:C.purple,fontWeight:700}}>🧩 Decomposed: </span>{e.sub_queries?.join(" / ")}</>,
    log_agent_subq:     (e) => <><span style={{color:C.textMid}}>Sub-task {e.index}/{e.total}: </span><span style={{color:C.accent}}>"{e.query}"</span></>,
    log_agent_subr:     (e) => <><span style={{color:C.green}}>✓ Sub-task {e.index} done</span>&nbsp;<span style={{color:C.textMid,fontSize:11}}>{e.answer_preview?.slice(0,60)}…</span></>,
    log_agent_done:     (e) => <span style={{color:C.green,fontWeight:700}}>✓ Agent done ({e.route}) in {e.elapsed_seconds}s</span>,
    agent_sub_title:    "SUB-TASK RESULTS",
    log_start:          (q) => <>Processing: <em style={{color:C.accent}}>"{q}"</em></>,
    log_hyde:           (doc) => <><span style={{color:C.teal,fontWeight:700}}>HyDE doc: </span><span style={{color:C.textMid,fontStyle:"italic"}}>"{doc.slice(0,80)}{doc.length>80?"…":""}"</span></>,
    log_docScored:      (e) => <><span style={{color:C.textMid}}>{e.title.slice(0,22)}…</span>&nbsp;→&nbsp;<span style={{color:C.accent}}>Vec {(e.embedding_score*100).toFixed(0)}%</span>&nbsp;<span style={{color:C.green}}>BM25 {(e.bm25_score*100).toFixed(0)}%</span>{e.graph_score>0&&<>&nbsp;<span style={{color:"#059669"}}>Graph {(e.graph_score*100).toFixed(0)}%</span></>}&nbsp;<span style={{color:C.text,fontWeight:700}}>Score {(e.final_score*100).toFixed(0)}%</span></>,
    log_retDone:        (e) => <>Round {e.iteration} done, top score: <span style={{color:e.top_score>=e.threshold?C.green:C.orange,fontWeight:700}}>{(e.top_score*100).toFixed(1)}%</span> (threshold {(e.threshold*100).toFixed(0)}%)</>,
    log_reflect:        (e) => <><span style={{color:C.orange}}>Reflection: </span>{e.failure_reason}</>,
    log_rewrite:        (e) => <>Query rewrite: <span style={{color:C.textMid,textDecoration:"line-through"}}>"{e.original_query}"</span>&nbsp;→&nbsp;<span style={{color:C.accent}}>"{e.new_query}"</span></>,
    log_rerank:         (e) => <><span style={{color:C.textMid}}>{e.title.slice(0,20)}…</span>&nbsp;CE score: <span style={{color:C.purple,fontWeight:700}}>{(e.ce_score*100).toFixed(1)}%</span>&nbsp;<span style={{color:e.improvement>0?C.green:C.orange,fontSize:11}}>({e.improvement>0?"+":""}{(e.improvement*100).toFixed(1)}%)</span></>,
    log_rerankDone:     (e) => <>Reranking done, top: <span style={{color:C.purple,fontWeight:700}}>{(e.top_score*100).toFixed(1)}%</span></>,
    log_done:           (e) => <span style={{color:C.green,fontWeight:700}}>✓ Done in {e.elapsed_seconds}s · {e.total_iterations} iteration(s)</span>,
  },
};

/** Module-level translation helper — pass lang explicitly */
const tL = (lang, key, ...args) => {
  const v = I18N[lang]?.[key] ?? I18N.zh[key] ?? key;
  return typeof v === "function" ? v(...args) : v;
};

// ── Utility Components ────────────────────────────────────────────────────────

const Tag = ({ label, color = C.accent }) => (
  <span style={{
    padding:"2px 8px", borderRadius:4, fontSize:11, fontWeight:600,
    background:`${color}18`, color, border:`1px solid ${color}40`,
    letterSpacing:"0.04em", whiteSpace:"nowrap",
  }}>{label}</span>
);

const ScoreBar = ({ value, color = C.accent, label, showPercent = true }) => (
  <div style={{ display:"flex", alignItems:"center", gap:8 }}>
    {label && <span style={{ color:C.textMid, fontSize:11, minWidth:56 }}>{label}</span>}
    <div style={{ flex:1, height:6, background:C.border, borderRadius:3, overflow:"hidden" }}>
      <div style={{
        width:`${Math.min(value*100,100)}%`, height:"100%", borderRadius:3,
        background:`linear-gradient(90deg,${color}88,${color})`,
        transition:"width 0.6s cubic-bezier(0.4,0,0.2,1)",
      }}/>
    </div>
    {showPercent && (
      <span style={{ color, fontSize:11, fontWeight:700, minWidth:38, textAlign:"right" }}>
        {(value*100).toFixed(1)}%
      </span>
    )}
  </div>
);

const Pill = ({ children, active, onClick, color = C.accent }) => (
  <button onClick={onClick} style={{
    padding:"6px 14px", borderRadius:6, fontSize:12, fontWeight:600,
    cursor:"pointer", transition:"all 0.2s",
    background: active ? `${color}18` : "transparent",
    color:       active ? color : C.textMid,
    border:`1px solid ${active ? color+"55" : C.border}`,
    outline:"none",
  }}>{children}</button>
);

const Spinner = ({ size = 16, color = C.accent }) => (
  <div style={{
    width:size, height:size, borderRadius:"50%",
    border:`2px solid ${color}33`, borderTopColor:color,
    animation:"spin 0.7s linear infinite", display:"inline-block",
  }}/>
);

// ── Phase Badge ───────────────────────────────────────────────────────────────
const PHASES = {
  retrieval:  { label:"RETRIEVAL",  color:C.accent  },
  reranking:  { label:"RERANKING",  color:C.purple  },
  generation: { label:"GENERATION", color:C.green   },
  reflection: { label:"EVALUATION", color:C.orange  },
  hyde:       { label:"HyDE",       color:C.teal    },
};
const PhaseBadge = ({ phase }) => {
  const p = PHASES[phase] || { label:phase.toUpperCase(), color:C.textMid };
  return (
    <span style={{
      padding:"2px 10px", borderRadius:4, fontSize:10, fontWeight:800,
      background:`${p.color}18`, color:p.color, border:`1px solid ${p.color}44`,
      letterSpacing:"0.08em",
    }}>{p.label}</span>
  );
};

// ── Log Entry ─────────────────────────────────────────────────────────────────
const LogEntry = ({ entry, lang }) => {
  const icons = {
    pipeline_start:"⚡", phase_start:"▶", doc_scored:"📄",
    retrieval_done:"✅", reflection:"🤔", query_rewrite:"✏️",
    rerank_score:"🔢", reranking_done:"🎯", answer_token:"💬",
    pipeline_complete:"🏁", error:"❌", hyde_generation:"🔮",
    // Agent events
    agent_routing:"🔍", agent_route:"🚦", agent_tool_call:"🔧",
    agent_tool_result:"✅", agent_decompose:"🧩", agent_subquery:"▷",
    agent_subresult:"◈", agent_complete:"🏁",
  };

  const getContent = () => {
    switch (entry.type) {
      case "pipeline_start":     return tL(lang, "log_start", entry.query);
      case "phase_start":        return <span><PhaseBadge phase={entry.phase}/>&nbsp;&nbsp;{entry.message}</span>;
      case "hyde_generation":    return tL(lang, "log_hyde", entry.hypothetical_doc || "");
      case "doc_scored":         return <span style={{fontSize:12}}>{tL(lang,"log_docScored",entry)}</span>;
      case "retrieval_done":     return tL(lang, "log_retDone", entry);
      case "reflection":         return tL(lang, "log_reflect", entry);
      case "query_rewrite":      return tL(lang, "log_rewrite", entry);
      case "rerank_score":       return <span style={{fontSize:12}}>{tL(lang,"log_rerank",entry)}</span>;
      case "reranking_done":     return tL(lang, "log_rerankDone", entry);
      case "pipeline_complete":  return tL(lang, "log_done", entry);
      // Agent events
      case "agent_routing":      return <span style={{color:C.textMid}}>{tL(lang,"log_agent_routing")}</span>;
      case "agent_route":        return tL(lang, "log_agent_route", entry);
      case "agent_tool_call":    return tL(lang, "log_agent_tool", entry);
      case "agent_tool_result":  return tL(lang, "log_agent_result", entry);
      case "agent_decompose":    return tL(lang, "log_agent_decomp", entry);
      case "agent_subquery":     return tL(lang, "log_agent_subq", entry);
      case "agent_subresult":    return tL(lang, "log_agent_subr", entry);
      case "agent_complete":     return tL(lang, "log_agent_done", entry);
      default: return <span>{entry.message || JSON.stringify(entry).slice(0,80)}</span>;
    }
  };

  return (
    <div style={{
      padding:"5px 0", borderBottom:`1px solid ${C.border}`,
      fontSize:12.5, color:C.text, display:"flex", gap:8, alignItems:"flex-start",
    }}>
      <span style={{opacity:0.55, flexShrink:0, marginTop:1}}>{icons[entry.type]||"•"}</span>
      <span style={{lineHeight:1.5}}>{getContent()}</span>
    </div>
  );
};

// ── Doc Card ──────────────────────────────────────────────────────────────────
const DocCard = ({ doc, rank, lang }) => (
  <div style={{
    background:C.surface, border:`1px solid ${C.borderBright}`,
    borderRadius:8, padding:"12px 14px", marginBottom:8,
  }}>
    <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:8 }}>
      <div style={{ display:"flex", alignItems:"center", gap:8 }}>
        <span style={{
          width:22, height:22, borderRadius:4, background:`${C.accent}18`,
          color:C.accent, fontSize:11, fontWeight:800, display:"flex",
          alignItems:"center", justifyContent:"center", flexShrink:0,
        }}>#{rank}</span>
        <span style={{fontWeight:600, fontSize:13, color:C.text}}>{doc.title}</span>
      </div>
      <span style={{
        fontSize:14, fontWeight:800,
        color: doc.final_score>0.75?C.green:doc.final_score>0.55?C.accent:C.orange,
      }}>{(doc.final_score*100).toFixed(1)}%</span>
    </div>
    <p style={{fontSize:12, color:C.textMid, margin:"0 0 8px", lineHeight:1.6}}>{doc.content}</p>
    {doc.source && (
      <div style={{fontSize:10, color:C.textDim, marginBottom:6, display:"flex", alignItems:"center", gap:6, flexWrap:"wrap"}}>
        <span>📁</span><span style={{fontFamily:"monospace"}}>{doc.source}</span>
        {doc.file_mtime && (
          <span style={{color:C.textDim}}>
            · {new Date(doc.file_mtime).toLocaleDateString()}
          </span>
        )}
        {doc.context_added && (
          <span style={{
            fontSize:9, padding:"1px 5px", borderRadius:3,
            background:`${C.teal}15`, color:C.teal, border:`1px solid ${C.teal}33`,
          }}>✦ contextual</span>
        )}
      </div>
    )}
    <div style={{display:"flex", gap:6, flexWrap:"wrap", marginBottom:8}}>
      {doc.tags?.map(t => <Tag key={t} label={t}/>)}
    </div>
    <div style={{display:"grid", gridTemplateColumns:"1fr 1fr", gap:6}}>
      {doc.embedding_score>0 && <ScoreBar value={doc.embedding_score} color={C.accent}  label={tL(lang,"score_vec")}/>}
      {doc.bm25_score>0     && <ScoreBar value={doc.bm25_score}      color={C.green}   label={tL(lang,"score_bm25")}/>}
      {doc.graph_score>0    && <ScoreBar value={doc.graph_score}     color="#059669"   label={tL(lang,"score_graph")}/>}
    </div>
  </div>
);

// ── RAGAS Panel ───────────────────────────────────────────────────────────────
const RagasPanel = ({ metrics, lang }) => {
  if (!metrics?.context_relevance) return null;
  const rows = [
    { vk:"context_relevance",  lk:"cr_label", dk:"cr_desc", color:C.accent  },
    { vk:"context_precision",  lk:"cp_label", dk:"cp_desc", color:C.green   },
    { vk:"answer_relevance",   lk:"ar_label", dk:"ar_desc", color:C.purple  },
    { vk:"answer_faithfulness",lk:"af_label", dk:"af_desc", color:C.orange  },
  ];
  const overall = rows.reduce((s,r)=>s+(metrics[r.vk]||0),0)/rows.length;
  return (
    <div style={{background:C.surface, border:`1px solid ${C.borderBright}`, borderRadius:8, padding:16}}>
      <div style={{display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:16}}>
        <span style={{fontSize:11, color:C.textMid, fontWeight:700, letterSpacing:"0.08em"}}>
          {tL(lang,"ragasTitle")}&nbsp;
        </span>
        <span style={{fontSize:13, fontWeight:800, color:overall>0.75?C.green:overall>0.55?C.accent:C.orange}}>
          {tL(lang,"ragasOverall",(overall*100).toFixed(1))}
        </span>
      </div>
      {rows.map(r=>(
        <div key={r.vk} style={{marginBottom:14}}>
          <span title={tL(lang,r.dk)} style={{fontSize:12, color:C.text, borderBottom:`1px dashed ${C.borderBright}`, cursor:"help", display:"inline-block", marginBottom:4}}>
            {tL(lang,r.lk)}
          </span>
          <ScoreBar value={metrics[r.vk]||0} color={r.color}/>
        </div>
      ))}
    </div>
  );
};

// ── Recall Chart ──────────────────────────────────────────────────────────────
const RecallChart = ({ metrics, lang }) => {
  if (!metrics) return null;
  const stages = [
    { lk:"r_base", value:metrics.baseline_recall,   color:C.textDim                },
    { lk:"r_iter", value:metrics.iterative_recall,  color:C.accent,  delta:"+15%"  },
    { lk:"r_fus",  value:metrics.fusion_recall,     color:C.green,   delta:"+3%"   },
    { lk:"r_re",   value:metrics.rerank_recall,     color:C.purple,  delta:"+2%"   },
  ];
  const maxVal = Math.max(...stages.map(s=>s.value));
  const chartH = 80;
  return (
    <div style={{background:C.surface, border:`1px solid ${C.borderBright}`, borderRadius:8, padding:16, marginTop:12}}>
      <div style={{fontSize:11, color:C.textMid, fontWeight:700, letterSpacing:"0.08em", marginBottom:16}}>
        {tL(lang,"recallTitle")}&nbsp;<span style={{fontSize:10,color:C.textDim}}>{tL(lang,"recallSub")}</span>
      </div>
      <div style={{display:"flex", alignItems:"flex-end", gap:12, height:chartH+36}}>
        {stages.map((s,i)=>{
          const barH=(s.value/maxVal)*chartH;
          return (
            <div key={i} style={{flex:1, display:"flex", flexDirection:"column", alignItems:"center", gap:4}}>
              <span style={{fontSize:11,color:s.color,fontWeight:700}}>{(s.value*100).toFixed(1)}%</span>
              {s.delta ? <span style={{fontSize:10,color:C.green,fontWeight:600}}>{s.delta}</span>
                       : <span style={{fontSize:10}}>&nbsp;</span>}
              <div style={{
                width:"100%", height:barH,
                background:`linear-gradient(180deg,${s.color}cc,${s.color}44)`,
                borderRadius:"4px 4px 0 0", border:`1px solid ${s.color}55`,
                transition:"height 0.8s cubic-bezier(0.4,0,0.2,1)",
              }}/>
              <span style={{fontSize:10,color:C.textMid,textAlign:"center",lineHeight:1.3}}>
                {tL(lang,s.lk)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
};

// ── Iteration Timeline ─────────────────────────────────────────────────────────
const IterationTimeline = ({ iterations }) => {
  if (!iterations?.length) return null;
  return (
    <div>
      {iterations.map((it,i)=>(
        <div key={i} style={{display:"flex", gap:10, marginBottom:8, alignItems:"flex-start"}}>
          <div style={{display:"flex", flexDirection:"column", alignItems:"center"}}>
            <div style={{
              width:24, height:24, borderRadius:"50%", flexShrink:0,
              background: it.reflected?`${C.orange}18`:`${C.green}18`,
              border:`2px solid ${it.reflected?C.orange:C.green}`,
              display:"flex", alignItems:"center", justifyContent:"center",
              fontSize:10, fontWeight:800, color:it.reflected?C.orange:C.green,
            }}>#{it.iteration}</div>
            {i<iterations.length-1 && <div style={{width:2,height:16,background:C.border,margin:"2px 0"}}/>}
          </div>
          <div style={{flex:1, background:C.surface, border:`1px solid ${C.border}`, borderRadius:6, padding:"8px 12px", fontSize:12}}>
            <div style={{display:"flex", justifyContent:"space-between", marginBottom:4}}>
              <span style={{color:C.textMid}}>Strategy: <span style={{color:C.accent}}>{it.strategy}</span></span>
              <span style={{color:it.top_score>=0.55?C.green:C.orange, fontWeight:700}}>{(it.top_score*100).toFixed(1)}%</span>
            </div>
            <div style={{color:C.text, marginBottom:4}}>
              Query: <span style={{color:C.textMid}}>「{it.query}」</span>
            </div>
            {it.reflected && <Tag label="Reflection triggered" color={C.orange}/>}
          </div>
        </div>
      ))}
    </div>
  );
};

// ── Conversation Panel ─────────────────────────────────────────────────────────
const ConversationPanel = ({ history, onClear, lang }) => {
  if (!history.length) {
    return (
      <div style={{textAlign:"center", padding:"60px 0", color:C.textDim, fontSize:13}}>
        {tL(lang,"convEmpty")}
      </div>
    );
  }
  return (
    <div>
      <div style={{display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:12}}>
        <span style={{fontSize:11, color:C.textMid, fontWeight:700, letterSpacing:"0.08em"}}>
          {tL(lang,"convHistTitle", Math.floor(history.length/2))}
        </span>
        <button onClick={onClear} style={{
          fontSize:11, color:C.red, background:"transparent",
          border:`1px solid ${C.red}44`, borderRadius:4, padding:"2px 8px", cursor:"pointer",
        }}>{tL(lang,"clearBtn")}</button>
      </div>
      <div style={{maxHeight:560, overflowY:"auto", paddingRight:4}}>
        {history.map((turn,i)=>(
          <div key={i} style={{
            display:"flex",
            justifyContent: turn.role==="user"?"flex-end":"flex-start",
            marginBottom:10,
          }}>
            <div style={{
              maxWidth:"80%", padding:"10px 14px", borderRadius:10,
              fontSize:13, lineHeight:1.7,
              background: turn.role==="user"?`${C.accent}12`:C.surface,
              border:`1px solid ${turn.role==="user"?C.accent+"33":C.borderBright}`,
              color:C.text,
              borderTopRightRadius: turn.role==="user"?2:10,
              borderTopLeftRadius:  turn.role==="user"?10:2,
            }}>
              <div style={{fontSize:10, color:C.textDim, marginBottom:4, fontWeight:600}}>
                {turn.role==="user"?tL(lang,"role_user"):tL(lang,"role_asst")}
              </div>
              <div style={{whiteSpace:"pre-wrap"}}>{turn.content}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

// ══════════════════════════════════════════════════════════════════════════════
// Docs Tab  —  Upload + Knowledge Base Document Management
// ══════════════════════════════════════════════════════════════════════════════
const ALLOWED_EXTS = [".txt", ".md", ".pdf"];

const DocsTab = ({ lang, kbStats, onRefresh }) => {
  const [pendingFiles, setPendingFiles] = useState([]);   // File objects waiting to upload
  const [fileStatuses, setFileStatuses] = useState({});   // filename -> "ok"|"error"|"uploading"
  const [uploading,    setUploading]    = useState(false);
  const [uploadMsg,    setUploadMsg]    = useState("");    // summary after upload
  const [isDragging,   setIsDragging]   = useState(false);
  const [deletingFile, setDeletingFile] = useState(null); // filename being deleted
  const [previewOpen,  setPreviewOpen]  = useState(false);
  const [previewSrc,   setPreviewSrc]   = useState(null);
  const [previewData,  setPreviewData]  = useState(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewErr,   setPreviewErr]   = useState("");
  const inputRef = useRef(null);

  const addFiles = (rawFiles) => {
    const valid = Array.from(rawFiles).filter(f =>
      ALLOWED_EXTS.some(ext => f.name.toLowerCase().endsWith(ext))
    );
    setPendingFiles(prev => {
      const existing = new Set(prev.map(f => f.name));
      return [...prev, ...valid.filter(f => !existing.has(f.name))];
    });
  };

  const removeFile = (name) =>
    setPendingFiles(prev => prev.filter(f => f.name !== name));

  const onDragOver = (e) => { e.preventDefault(); setIsDragging(true); };
  const onDragLeave = ()  => setIsDragging(false);
  const onDrop = (e)      => { e.preventDefault(); setIsDragging(false); addFiles(e.dataTransfer.files); };

  const uploadAll = async () => {
    if (!pendingFiles.length || uploading) return;
    setUploading(true);
    setUploadMsg("");
    // Mark all as uploading
    const st = {};
    pendingFiles.forEach(f => { st[f.name] = "uploading"; });
    setFileStatuses(st);

    try {
      const formData = new FormData();
      pendingFiles.forEach(f => formData.append("files", f));
      const res  = await fetch(`${API_BASE}/upload`, { method: "POST", body: formData });
      const data = await res.json();
      const newSt = {};
      data.files.forEach(r => { newSt[r.filename] = r.status; });
      setFileStatuses(newSt);
      setUploadMsg(tL(lang, "docsUploadDone", data.uploaded, data.errors));
      if (data.uploaded > 0) {
        // Wait 3s for backend rebuild, then refresh stats
        setTimeout(() => { onRefresh(); setPendingFiles([]); setFileStatuses({}); setUploadMsg(""); }, 3500);
      }
    } catch {
      const errSt = {};
      pendingFiles.forEach(f => { errSt[f.name] = "error"; });
      setFileStatuses(errSt);
      setUploadMsg("Upload failed — is the backend running?");
    } finally {
      setUploading(false);
    }
  };

  const deleteDoc = async (filename) => {
    if (!window.confirm(tL(lang, "docsDeleteConfirm", filename))) return;
    setDeletingFile(filename);
    try {
      await fetch(`${API_BASE}/docs/${encodeURIComponent(filename)}`, { method: "DELETE" });
      setTimeout(() => { onRefresh(); setDeletingFile(null); }, 3500);
    } catch {
      setDeletingFile(null);
    }
  };

  const openPreview = async (source) => {
    setPreviewOpen(true);
    setPreviewSrc(source);
    setPreviewData(null);
    setPreviewErr("");
    setPreviewLoading(true);
    try {
      const res = await fetch(`${API_BASE}/docs/preview?source=${encodeURIComponent(source)}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setPreviewData(await res.json());
    } catch (e) {
      setPreviewErr(lang === "en" ? `Preview failed: ${String(e)}` : `预览失败：${String(e)}`);
    } finally {
      setPreviewLoading(false);
    }
  };

  const closePreview = () => {
    setPreviewOpen(false);
    setPreviewSrc(null);
    setPreviewData(null);
    setPreviewErr("");
    setPreviewLoading(false);
  };

  const fmtSize = (bytes) =>
    bytes >= 1024*1024 ? `${(bytes/1024/1024).toFixed(1)} MB`
    : bytes >= 1024    ? `${(bytes/1024).toFixed(1)} KB`
    : `${bytes} B`;

  const statusIcon = (name) => {
    const s = fileStatuses[name];
    if (s === "ok")        return <span style={{color:C.green,  fontSize:13}}>✓</span>;
    if (s === "error")     return <span style={{color:C.red,    fontSize:13}}>✗</span>;
    if (s === "uploading") return <Spinner size={12} color={C.accent}/>;
    return null;
  };

  return (
    <div style={{display:"flex", flexDirection:"column", gap:16}}>

      {/* ── Preview Modal ── */}
      {previewOpen && (
        <div onClick={closePreview} style={{
          position:"fixed", inset:0, background:"rgba(15,23,42,0.55)",
          display:"flex", alignItems:"center", justifyContent:"center",
          padding:20, zIndex: 9999,
        }}>
          <div onClick={(e)=>e.stopPropagation()} style={{
            width:"min(920px, 96vw)", maxHeight:"86vh",
            background:C.bg, borderRadius:12,
            border:`1px solid ${C.borderBright}`,
            boxShadow:"0 20px 60px rgba(0,0,0,0.25)",
            display:"flex", flexDirection:"column", overflow:"hidden",
          }}>
            <div style={{
              padding:"12px 14px", borderBottom:`1px solid ${C.border}`,
              display:"flex", alignItems:"center", justifyContent:"space-between", gap:10,
            }}>
              <div style={{minWidth:0}}>
                <div style={{fontSize:12, fontWeight:800, color:C.text, overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap"}}>
                  {previewData?.source || previewSrc || (lang==="en" ? "Preview" : "预览")}
                </div>
                {previewData && (
                  <div style={{fontSize:11, color:C.textDim, marginTop:2}}>
                    {previewData.file_size_kb ? `${previewData.file_size_kb} KB · ` : ""}{previewData.total_chunks} chunk(s)
                    {previewData.file_mtime ? ` · ${new Date(previewData.file_mtime).toLocaleString()}` : ""}
                    {previewData.truncated ? (lang==="en" ? " · truncated" : " · 已截断") : ""}
                  </div>
                )}
              </div>
              <button onClick={closePreview} style={{
                border:`1px solid ${C.border}`, background:"transparent",
                borderRadius:6, padding:"6px 10px", cursor:"pointer",
                color:C.textMid, fontSize:12, fontWeight:700, fontFamily:"inherit",
              }}>{lang==="en" ? "Close" : "关闭"}</button>
            </div>

            <div style={{padding:14, overflow:"auto"}}>
              {previewLoading && (
                <div style={{display:"flex", alignItems:"center", gap:8, color:C.textMid, fontSize:12}}>
                  <Spinner size={14} color={C.accent}/> {lang==="en" ? "Loading preview…" : "加载预览中…"}
                </div>
              )}
              {previewErr && (
                <div style={{
                  background:`${C.red}10`, border:`1px solid ${C.red}33`, color:C.red,
                  borderRadius:8, padding:"10px 12px", fontSize:12, marginBottom:10,
                }}>{previewErr}</div>
              )}
              {previewData?.preview_text && (
                <pre style={{
                  whiteSpace:"pre-wrap", wordBreak:"break-word",
                  margin:0, fontSize:12.5, lineHeight:1.7, color:C.text,
                  background:C.surface, border:`1px solid ${C.borderBright}`,
                  borderRadius:10, padding:12,
                }}>{previewData.preview_text}</pre>
              )}
              {previewData && !previewData.preview_text && !previewLoading && !previewErr && (
                <div style={{color:C.textDim, fontSize:12}}>
                  {lang==="en" ? "No preview text available." : "暂无可预览内容。"}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ── Upload Zone ── */}
      <div style={{background:C.surface, border:`1px solid ${C.borderBright}`, borderRadius:10, padding:16}}>
        <div style={{fontSize:11, color:C.textMid, fontWeight:700, letterSpacing:"0.08em", marginBottom:12}}>
          {tL(lang,"docsUploadTitle")}
        </div>

        {/* Drop area */}
        <div
          onDragOver={onDragOver} onDragLeave={onDragLeave} onDrop={onDrop}
          onClick={()=>inputRef.current?.click()}
          style={{
            border:`2px dashed ${isDragging ? C.accent : C.borderBright}`,
            borderRadius:8, padding:"28px 16px", textAlign:"center", cursor:"pointer",
            background: isDragging ? `${C.accent}08` : C.bg,
            transition:"all 0.2s", marginBottom:12,
          }}
        >
          <div style={{fontSize:28, marginBottom:8}}>📂</div>
          <div style={{fontSize:13, color:isDragging?C.accent:C.textMid, fontWeight:600}}>
            {isDragging ? tL(lang,"docsDropActive") : tL(lang,"docsDropHint")}
          </div>
          <div style={{fontSize:11, color:C.textDim, marginTop:4}}>
            {tL(lang,"docsAllowedTypes")}
          </div>
          <input
            ref={inputRef} type="file" multiple accept=".txt,.md,.pdf"
            style={{display:"none"}}
            onChange={e => { addFiles(e.target.files); e.target.value=""; }}
          />
        </div>

        {/* Pending file list */}
        {pendingFiles.length > 0 && (
          <div style={{marginBottom:12}}>
            {pendingFiles.map(f => (
              <div key={f.name} style={{
                display:"flex", alignItems:"center", gap:8,
                padding:"5px 8px", borderRadius:5, marginBottom:4,
                background:C.surface, border:`1px solid ${C.border}`, fontSize:12,
              }}>
                <span style={{flex:1, overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap", color:C.text}}>
                  📄 {f.name}
                </span>
                <span style={{color:C.textDim, flexShrink:0, fontSize:11}}>{fmtSize(f.size)}</span>
                {statusIcon(f.name)}
                {!uploading && !fileStatuses[f.name] && (
                  <button onClick={(e)=>{e.stopPropagation();removeFile(f.name);}} style={{
                    background:"transparent", border:"none", color:C.red,
                    cursor:"pointer", fontSize:14, padding:"0 2px", lineHeight:1,
                  }}>×</button>
                )}
              </div>
            ))}
          </div>
        )}

        {/* Upload result message */}
        {uploadMsg && (
          <div style={{
            fontSize:12, color:C.green, background:`${C.green}10`,
            border:`1px solid ${C.green}33`, borderRadius:5, padding:"6px 10px", marginBottom:10,
          }}>
            ✓ {uploadMsg} — <span style={{color:C.textDim}}>{tL(lang,"docsRebuildNotice")}</span>
          </div>
        )}

        {/* Upload button */}
        <button
          onClick={uploadAll}
          disabled={!pendingFiles.length || uploading}
          style={{
            width:"100%", padding:"10px", borderRadius:7, fontSize:13, fontWeight:700,
            cursor: (!pendingFiles.length || uploading) ? "not-allowed" : "pointer",
            background: (!pendingFiles.length || uploading)
              ? `${C.accent}18`
              : `linear-gradient(135deg,${C.accentDim},${C.accent})`,
            color: (!pendingFiles.length || uploading) ? C.accentDim : "#fff",
            border:"none", fontFamily:"inherit",
            display:"flex", alignItems:"center", justifyContent:"center", gap:8,
          }}
        >
          {uploading
            ? <><Spinner size={13} color={C.accent}/>{tL(lang,"docsUploading")}</>
            : tL(lang, "docsUploadBtn", pendingFiles.length || 0)
          }
        </button>
      </div>

      {/* ── Document List ── */}
      <div style={{background:C.surface, border:`1px solid ${C.borderBright}`, borderRadius:10, padding:16}}>
        <div style={{fontSize:11, color:C.textMid, fontWeight:700, letterSpacing:"0.08em", marginBottom:12}}>
          {tL(lang,"docsCurrentTitle")}
          {kbStats && (
            <span style={{fontWeight:400, marginLeft:8, color:C.textDim}}>
              ({kbStats.total_chunks} {tL(lang,"kbChunks")} · {kbStats.total_sources} {tL(lang,"kbDocs")})
            </span>
          )}
        </div>

        {!kbStats || kbStats.sources?.length === 0 ? (
          <div style={{textAlign:"center", padding:"24px 0", color:C.textDim, fontSize:13}}>
            {tL(lang,"docsEmpty")}
          </div>
        ) : (
          <div>
            {kbStats.sources.map((src, i) => {
              const isStale   = kbStats.stale_sources?.includes(src.source);
              const isDeleting = deletingFile === src.source;
              return (
                <div key={i} style={{
                  display:"flex", alignItems:"center", gap:8,
                  padding:"9px 10px", borderRadius:7, marginBottom:6,
                  background: isStale ? `${C.orange}08` : C.bg,
                  border:`1px solid ${isStale ? C.orange+"44" : C.border}`,
                }}>
                  {/* Icon */}
                  <span style={{fontSize:16, flexShrink:0}}>
                    {src.source.endsWith(".pdf") ? "📕"
                     : src.source.endsWith(".md") ? "📝" : "📄"}
                  </span>

                  {/* Info */}
                  <div style={{flex:1, minWidth:0}}>
                    <div style={{
                      fontSize:12, fontWeight:600, color:C.text,
                      overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap",
                    }} title={src.source}>{src.source}</div>
                    <div style={{fontSize:11, color:C.textDim, marginTop:2, display:"flex", gap:8}}>
                      <span>{tL(lang,"docsChunks", src.chunks)}</span>
                      <span>{tL(lang,"docsWords",  src.words)}</span>
                      {src.file_size_kb && <span>{src.file_size_kb} KB</span>}
                      {isStale && (
                        <span style={{color:C.orange}}>⚠ stale</span>
                      )}
                    </div>
                  </div>

                  {/* Delete button */}
                  <div style={{display:"flex", gap:6, alignItems:"center", flexShrink:0}}>
                    <button
                      onClick={() => openPreview(src.source)}
                      style={{
                        padding:"3px 9px", borderRadius:4, fontSize:11,
                        cursor:"pointer", background:"transparent", color:C.accent,
                        border:`1px solid ${C.accent}33`, fontFamily:"inherit",
                      }}
                    >
                      {lang==="en" ? "Preview" : "预览"}
                    </button>
                    <button
                      onClick={() => deleteDoc(src.source)}
                      disabled={isDeleting}
                      style={{
                        padding:"3px 9px", borderRadius:4,
                        fontSize:11, cursor: isDeleting ? "not-allowed" : "pointer",
                        background:"transparent", color: isDeleting ? C.textDim : C.red,
                        border:`1px solid ${isDeleting ? C.border : C.red+"44"}`,
                        fontFamily:"inherit",
                      }}
                    >
                      {isDeleting ? tL(lang,"docsDeleting") : tL(lang,"docsDeleteBtn")}
                    </button>
                  </div>
                </div>
              );
            })}

            {/* Rebuild notice */}
            <div style={{fontSize:11, color:C.textDim, marginTop:8, textAlign:"center"}}>
              ℹ {tL(lang,"docsRebuildNotice")}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

// ── localStorage helpers ──────────────────────────────────────────────────────
const LS_KEY = "rag_ui_state";

const loadLS = () => {
  try { return JSON.parse(localStorage.getItem(LS_KEY)) || {}; } catch { return {}; }
};

const usePersistedState = (key, defaultValue) => {
  const [value, setValue] = useState(() => {
    const saved = loadLS()[key];
    return saved !== undefined ? saved : defaultValue;
  });
  const setPersisted = useCallback((v) => {
    setValue(prev => {
      const next = typeof v === "function" ? v(prev) : v;
      try {
        const current = loadLS();
        localStorage.setItem(LS_KEY, JSON.stringify({ ...current, [key]: next }));
      } catch { /* quota exceeded — silent */ }
      return next;
    });
  }, [key]);
  return [value, setPersisted];
};

// ══════════════════════════════════════════════════════════════════════════════
// Main App
// ══════════════════════════════════════════════════════════════════════════════
export default function RAGDashboard() {
  const [lang, setLang]                   = usePersistedState("lang", "zh");
  const t = useCallback((key,...args)=>tL(lang,key,...args),[lang]);

  const [query, setQuery]                 = useState("企业知识库如何实现高效检索？");
  const [strategy, setStrategy]           = usePersistedState("strategy", "adaptive");
  const [enableIterative, setEnableIterative] = usePersistedState("enableIterative", true);
  const [enableRerank, setEnableRerank]   = usePersistedState("enableRerank", true);
  const [enableHyde, setEnableHyde]       = usePersistedState("enableHyde", false);
  const [enableConversation, setEnableConversation] = usePersistedState("enableConversation", false);
  const [enableGraph, setEnableGraph]     = usePersistedState("enableGraph", false);
  const [agentMode, setAgentMode]         = usePersistedState("agentMode", false);
  const [agentRoute, setAgentRoute]       = useState(null);   // {route, reason, ...}
  const [agentSubResults, setAgentSubResults] = useState([]);
  const [threshold, setThreshold]         = usePersistedState("threshold", 0.55);

  const [status, setStatus]               = useState("idle");
  const [logs, setLogs]                   = useState([]);
  const [docs, setDocs]                   = useState([]);
  const [answer, setAnswer]               = useState("");
  const [metrics, setMetrics]             = useState(null);
  const [iterations, setIterations]       = useState([]);
  const [elapsed, setElapsed]             = useState(null);
  const [hydeDoc, setHydeDoc]             = useState("");
  const [conversationHistory, setConversationHistory] = usePersistedState("conversationHistory", []);
  const [activeTab, setActiveTab]         = useState("process");

  // Knowledge Base stats
  const [kbStats, setKbStats]             = useState(null);
  const [kbLoading, setKbLoading]         = useState(false);
  const [kbRebuilding, setKbRebuilding]   = useState(false);

  // Backend readiness (prevents WS attempts while the API is still starting)
  const [backendReady, setBackendReady]   = useState(false);

  // Answer feedback
  const [feedbackGiven, setFeedbackGiven] = useState(null);   // "pos" | "neg" | null
  const [feedbackLoading, setFeedbackLoading] = useState(false);
  const [feedbackComment, setFeedbackComment] = useState("");

  const wsRef             = useRef(null);
  const logsEndRef        = useRef(null);
  const submittedQueryRef = useRef("");

  // Fetch KB stats on mount and after index rebuild
  const fetchKbStats = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/stats`);
      if (res.ok) setKbStats(await res.json());
    } catch { /* backend not running yet */ }
  }, []);

  useEffect(() => { fetchKbStats(); }, [fetchKbStats]);

  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const res = await fetch(`${API_BASE}/health`, { method: "GET" });
        if (!alive) return;
        setBackendReady(res.ok);
      } catch {
        if (!alive) return;
        setBackendReady(false);
      }
    };
    tick();
    const id = setInterval(tick, 2000);
    return () => { alive = false; clearInterval(id); };
  }, []);

  const triggerRebuild = useCallback(async (force = false) => {
    setKbRebuilding(true);
    try {
      await fetch(`${API_BASE}/reload?force=${force}`, { method:"POST" });
      // Poll until rebuild finishes (simple approach: wait 3s then refetch)
      setTimeout(() => { setKbRebuilding(false); fetchKbStats(); }, 4000);
    } catch { setKbRebuilding(false); }
  }, [fetchKbStats]);

  // Submit answer feedback
  const submitFeedback = useCallback(async (rating) => {
    if (feedbackGiven || feedbackLoading) return;
    setFeedbackLoading(true);
    try {
      await fetch(`${API_BASE}/feedback`, {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({
          query:    submittedQueryRef.current,
          answer,
          rating,
          comment:  feedbackComment || null,
          doc_ids:  docs.map(d=>d.id),
          language: lang,
        }),
      });
      setFeedbackGiven(rating > 0 ? "pos" : "neg");
    } finally { setFeedbackLoading(false); }
  }, [feedbackGiven, feedbackLoading, answer, docs, lang, feedbackComment]);

  useEffect(()=>{
    if (logsEndRef.current && activeTab==="process")
      logsEndRef.current.scrollIntoView({behavior:"smooth"});
  },[logs, activeTab]);

  useEffect(()=>{
    if (status==="done" && enableConversation && answer && submittedQueryRef.current) {
      setConversationHistory(prev=>{
        const last=prev[prev.length-1];
        if (last?.role==="assistant"&&last?.content===answer) return prev;
        return [...prev,{role:"user",content:submittedQueryRef.current},{role:"assistant",content:answer}];
      });
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  },[status]);

  const handleMessage = useCallback((evt)=>{
    const msg=JSON.parse(evt.data);
    if (msg.type==="hyde_generation") { setHydeDoc(msg.hypothetical_doc||""); setLogs(p=>[...p,msg]); return; }
    if (msg.type==="answer_token")    { setAnswer(msg.full_answer_so_far||""); return; }
    if (msg.type==="pipeline_complete") {
      setDocs(msg.retrieved_docs||[]);
      setMetrics(msg.metrics||null);
      setIterations(msg.iterations_detail||[]);
      setElapsed(msg.elapsed_seconds);
      setStatus("done");
    }
    // Agent events
    if (msg.type==="agent_route") { setAgentRoute(msg); }
    if (msg.type==="agent_complete") {
      setDocs(msg.retrieved_docs||[]);
      setMetrics(msg.metrics||null);
      setIterations([]);
      setElapsed(msg.elapsed_seconds);
      if (msg.sub_results) setAgentSubResults(msg.sub_results);
      setStatus("done");
    }
    if (msg.type==="error") setStatus("error");
    setLogs(p=>[...p,msg]);
  },[]);

  const runQuery = useCallback(()=>{
    if (!backendReady) {
      setStatus("error");
      setLogs(p=>[...p,{type:"error",message:lang==="en"
        ? "Backend not ready yet. Start the API on port 8000 and wait for it to finish initialization."
        : "后端尚未就绪。请先启动后端（8000 端口）并等待初始化完成后再执行。"}]);
      return;
    }
    if (!query.trim()||status==="running") return;
    submittedQueryRef.current=query;
    setStatus("running"); setLogs([]); setDocs([]); setAnswer("");
    setMetrics(null); setIterations([]); setElapsed(null);
    setHydeDoc(""); setActiveTab("process");
    setFeedbackGiven(null); setFeedbackComment("");
    setAgentRoute(null); setAgentSubResults([]);

    const ws = new WebSocket(backendWsUrl(agentMode ? "/ws/agent" : "/ws/query"));
    wsRef.current=ws;
    ws.onopen=()=>ws.send(JSON.stringify({
      query, strategy,
      enable_iterative:     enableIterative,
      enable_rerank:        enableRerank,
      enable_hyde:          enableHyde,
      enable_graph:         enableGraph,
      confidence_threshold: threshold,
      top_k: 5,
      language: lang,
      history: enableConversation?conversationHistory:[],
    }));
    ws.onmessage=handleMessage;
    ws.onerror=()=>{
      setStatus("error");
      setLogs(p=>[...p,{type:"error",message:lang==="en"
        ? "WebSocket failed: start the API on port 8000 (e.g. ./start.sh or: cd backend && python3 main.py)."
        : "WebSocket 无法连接：请先启动后端（监听 8000 端口）。可在项目根目录执行 ./start.sh，或：cd backend && python3 main.py"}]);
    };
    ws.onclose=()=>{ if(status==="running") setStatus("done"); };
  },[backendReady,query,strategy,enableIterative,enableRerank,enableHyde,enableGraph,enableConversation,agentMode,threshold,status,lang,conversationHistory,handleMessage]);

  const handleKeyDown=(e)=>{ if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();runQuery();} };

  const STRATEGIES=[
    {key:"adaptive"},{key:"hybrid"},{key:"vector"},{key:"bm25"},
  ];
  const TABS=[
    {key:"process",      lk:"tab_process", icon:"⚙"},
    {key:"results",      lk:"tab_results", icon:"📋"},
    {key:"metrics",      lk:"tab_metrics", icon:"📊"},
    {key:"conversation", lk:"tab_conv",    icon:"💬"},
    {key:"docs",         lk:"tab_docs",    icon:"📁"},
  ];

  const ToggleBtn=({labelKey, val, onToggle, color, tipKey})=>(
    <button onClick={onToggle} title={t(tipKey)} style={{
      padding:"8px 10px", borderRadius:6, cursor:"pointer",
      background: val?`${color}12`:"transparent",
      border:`1px solid ${val?color+"44":C.border}`,
      color: val?color:C.textMid,
      fontSize:11, fontWeight:600, fontFamily:"inherit",
      display:"flex", alignItems:"center", gap:6, transition:"all 0.2s",
    }}>
      <span style={{width:8,height:8,borderRadius:"50%",background:val?color:C.textDim,animation:val?"pulse 2s infinite":"none",flexShrink:0}}/>
      {t(labelKey)}
    </button>
  );

  return (
    <div style={{
      minHeight:"100vh", background:C.bg, color:C.text,
      fontFamily:"'Inter','SF Pro Display',-apple-system,sans-serif",
      padding:"24px",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        @keyframes spin  { to { transform: rotate(360deg); } }
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
        * { box-sizing: border-box; }
        ::-webkit-scrollbar{width:4px;}
        ::-webkit-scrollbar-track{background:${C.surface};}
        ::-webkit-scrollbar-thumb{background:${C.borderBright};border-radius:2px;}
        textarea:focus,input:focus{outline:none;}
        button:hover{opacity:0.8;}
      `}</style>

      {/* ── Header ── */}
      <div style={{display:"flex", alignItems:"center", justifyContent:"space-between", marginBottom:24}}>
        <div style={{display:"flex", alignItems:"center", gap:12}}>
          <div style={{
            width:38, height:38, borderRadius:10,
            background:`linear-gradient(135deg,${C.accent}28,${C.purple}28)`,
            border:`1px solid ${C.accent}44`,
            display:"flex", alignItems:"center", justifyContent:"center", fontSize:20,
          }}>⚡</div>
          <div>
            <h1 style={{margin:0, fontSize:18, fontWeight:800, color:C.text}}>Adaptive RAG System</h1>
            <p style={{margin:0, fontSize:11, color:C.textMid}}>{t("appSubtitle")}</p>
          </div>
        </div>

        <div style={{display:"flex", alignItems:"center", gap:10}}>
          {/* Language Toggle */}
          <div style={{
            display:"flex", borderRadius:8, overflow:"hidden",
            border:`1px solid ${C.borderBright}`, fontSize:12, fontWeight:700,
          }}>
            {["zh","en"].map(l=>(
              <button key={l} onClick={()=>{
                setLang(l);
                setQuery(I18N[l].sampleQueries[0]);
              }} style={{
                padding:"5px 14px", cursor:"pointer", fontFamily:"inherit",
                fontWeight:700, fontSize:12, letterSpacing:"0.04em",
                background: lang===l?C.accent:"transparent",
                color:       lang===l?"#fff":C.textMid,
                border:"none",
              }}>{l==="zh"?"中文":"EN"}</button>
            ))}
          </div>

          <div style={{display:"flex", gap:6, flexWrap:"wrap"}}>
            {[
              {label:"HyDE (EMNLP'22)", color:C.teal  },
              {label:"RAGAS (2023)",    color:C.purple },
              {label:"Cross-Encoder",  color:C.accent },
              {label:t("badge_conv"),  color:C.green  },
            ].map(b=><Tag key={b.label} label={b.label} color={b.color}/>)}
            {enableGraph && <Tag label={t("badge_graph")} color="#059669"/>}
            {agentMode && <Tag label={t("badge_agent")} color={C.orange}/>}
          </div>
        </div>
      </div>

      <div style={{display:"grid", gridTemplateColumns:"380px 1fr", gap:16, maxWidth:1440}}>

        {/* ══ LEFT PANEL ══ */}
        <div style={{display:"flex", flexDirection:"column", gap:12}}>

          {/* Backend status */}
          <div style={{
            background: backendReady ? `${C.green}08` : `${C.orange}08`,
            border: `1px solid ${backendReady ? C.green+"33" : C.orange+"33"}`,
            borderRadius: 10,
            padding: "10px 12px",
            fontSize: 12,
            color: backendReady ? C.green : C.orange,
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: 10,
          }}>
            <span style={{fontWeight:700}}>
              {backendReady ? `✓ ${t("backendOnline")}` : `⚠ ${t("backendOffline")}`}
            </span>
            <span style={{fontFamily:"monospace", fontSize:11, color:C.textDim}}>
              {API_BASE || window.location.origin}
            </span>
          </div>

          {/* Query Input */}
          <div style={{background:C.surface, border:`1px solid ${C.borderBright}`, borderRadius:10, padding:16}}>
            <div style={{fontSize:10, color:C.textMid, fontWeight:700, letterSpacing:"0.1em", marginBottom:10}}>
              {t("queryLabel")}
            </div>
            <textarea
              value={query} onChange={e=>setQuery(e.target.value)} onKeyDown={handleKeyDown}
              placeholder={t("queryPlaceholder")} rows={3}
              style={{
                width:"100%", background:"#fff", border:`1px solid ${C.borderBright}`,
                borderRadius:6, padding:"10px 12px", fontSize:13,
                color:C.text, resize:"vertical", lineHeight:1.6, fontFamily:"inherit",
              }}
            />
            <div style={{marginTop:8, display:"flex", flexWrap:"wrap", gap:4}}>
              {t("sampleQueries").map(q=>(
                <button key={q} onClick={()=>setQuery(q)} style={{
                  background:"transparent", border:`1px solid ${C.border}`,
                  borderRadius:4, padding:"3px 8px", fontSize:10,
                  color:C.textMid, cursor:"pointer", fontFamily:"inherit",
                  whiteSpace:"nowrap", overflow:"hidden", maxWidth:170, textOverflow:"ellipsis",
                }}>{q.slice(0,22)}…</button>
              ))}
            </div>
          </div>

          {/* Config */}
          <div style={{background:C.surface, border:`1px solid ${C.borderBright}`, borderRadius:10, padding:16}}>
            <div style={{fontSize:10, color:C.textMid, fontWeight:700, letterSpacing:"0.1em", marginBottom:14}}>
              {t("configLabel")}
            </div>

            {/* ── 运行模式 ── */}
            <div style={{marginBottom:12}}>
              <div style={{
                fontSize:9, fontWeight:700, letterSpacing:"0.12em",
                color:C.textDim, marginBottom:6, textTransform:"uppercase",
              }}>{lang==="zh" ? "运行模式" : "Mode"}</div>
              <div style={{display:"flex", gap:6}}>
                {/* Agent Mode — full-width prominent toggle */}
                <button onClick={()=>setAgentMode(!agentMode)} title={t("tip_agent")} style={{
                  flex:1, padding:"8px 10px", borderRadius:6, cursor:"pointer",
                  background: agentMode ? `${C.red}12` : "transparent",
                  border: `1px solid ${agentMode ? C.red+"55" : C.border}`,
                  color: agentMode ? C.red : C.textMid,
                  fontSize:11, fontWeight:700, fontFamily:"inherit",
                  display:"flex", alignItems:"center", gap:6, transition:"all 0.2s",
                }}>
                  <span style={{width:7,height:7,borderRadius:"50%",background:agentMode?C.red:C.textDim,flexShrink:0,animation:agentMode?"pulse 2s infinite":"none"}}/>
                  {t("toggle_agent")}
                  {agentMode && <span style={{fontSize:9,opacity:0.7,marginLeft:"auto"}}>🚦 auto-route</span>}
                </button>
                <ToggleBtn labelKey="toggle_conv" val={enableConversation} onToggle={()=>setEnableConversation(!enableConversation)} color={C.orange} tipKey="tip_conv"/>
              </div>
              {enableConversation && conversationHistory.length>0 && (
                <div style={{
                  marginTop:6, padding:"5px 10px", borderRadius:5,
                  background:`${C.orange}10`, border:`1px solid ${C.orange}33`,
                  fontSize:11, color:C.orange, display:"flex", alignItems:"center", justifyContent:"space-between",
                }}>
                  <span>{t("convCtx", Math.floor(conversationHistory.length/2))}</span>
                  <button onClick={()=>setConversationHistory([])} style={{
                    fontSize:10, color:C.red, background:"none", border:"none", cursor:"pointer", padding:0,
                  }}>{t("clearBtn")}</button>
                </div>
              )}
            </div>

            {/* ── 检索策略 ── */}
            <div style={{marginBottom:12}}>
              <div style={{
                fontSize:9, fontWeight:700, letterSpacing:"0.12em",
                color:C.textDim, marginBottom:6, textTransform:"uppercase",
              }}>{lang==="zh" ? "检索策略" : "Strategy"}</div>
              <div style={{display:"flex", gap:4, flexWrap:"wrap"}}>
                {STRATEGIES.map(s=>(
                  <Pill key={s.key} active={strategy===s.key} onClick={()=>setStrategy(s.key)}>
                    {t("strategies")[s.key]}
                  </Pill>
                ))}
              </div>
            </div>

            {/* ── 检索增强 ── */}
            <div style={{marginBottom:12}}>
              <div style={{
                fontSize:9, fontWeight:700, letterSpacing:"0.12em",
                color:C.textDim, marginBottom:6, textTransform:"uppercase",
              }}>{lang==="zh" ? "检索增强" : "Enhancements"}</div>
              <div style={{display:"grid", gridTemplateColumns:"1fr 1fr", gap:6}}>
                <ToggleBtn labelKey="toggle_iterative" val={enableIterative} onToggle={()=>setEnableIterative(!enableIterative)} color={C.accent}  tipKey="tip_iterative"/>
                <ToggleBtn labelKey="toggle_rerank"    val={enableRerank}    onToggle={()=>setEnableRerank(!enableRerank)}       color={C.purple} tipKey="tip_rerank"/>
                <ToggleBtn labelKey="toggle_hyde"      val={enableHyde}      onToggle={()=>setEnableHyde(!enableHyde)}           color={C.teal}   tipKey="tip_hyde"/>
                <ToggleBtn labelKey="toggle_graph"     val={enableGraph}     onToggle={()=>setEnableGraph(!enableGraph)}         color="#059669"  tipKey="tip_graph"/>
              </div>
            </div>

            {/* ── 置信度阈值 ── */}
            <div style={{
              paddingTop:10, borderTop:`1px solid ${C.border}`,
            }}>
              <div style={{display:"flex", justifyContent:"space-between", marginBottom:6}}>
                <span style={{fontSize:11, color:C.textMid}}>{t("thresholdLabel")}</span>
                <span style={{fontSize:11, color:C.accent, fontWeight:700}}>{(threshold*100).toFixed(0)}%</span>
              </div>
              <input type="range" min={0.3} max={0.85} step={0.05} value={threshold}
                onChange={e=>setThreshold(Number(e.target.value))}
                style={{width:"100%", accentColor:C.accent}}/>
            </div>
          </div>

          {/* Run Button */}
          <button onClick={runQuery} disabled={status==="running" || !backendReady} style={{
            padding:"12px 20px", borderRadius:8, fontSize:13, fontWeight:800,
            cursor: (status==="running" || !backendReady) ? "not-allowed" : "pointer",
            background: status==="running"?`${C.accent}18`:`linear-gradient(135deg,${C.accentDim},${C.accent})`,
            color: status==="running"?C.accentDim:"#fff",
            border:"none", letterSpacing:"0.04em", fontFamily:"inherit",
            display:"flex", alignItems:"center", justifyContent:"center", gap:8, transition:"all 0.2s",
          }}>
            {status==="running"?<><Spinner size={14} color={C.accent}/>{t("runningBtn")}</>:t("runBtn")}
          </button>

          {/* Query Stats */}
          {status==="done" && elapsed && (
            <div style={{background:C.surface, border:`1px solid ${C.green}44`, borderRadius:8, padding:12}}>
              <div style={{display:"grid", gridTemplateColumns:"1fr 1fr 1fr", gap:8, textAlign:"center"}}>
                {[
                  {lk:"stat_elapsed", value:`${elapsed}s`,       color:C.green  },
                  {lk:"stat_iters",   value:iterations.length,   color:C.accent },
                  {lk:"stat_docs",    value:docs.length,          color:C.purple },
                ].map(s=>(
                  <div key={s.lk}>
                    <div style={{fontSize:18, fontWeight:800, color:s.color}}>{s.value}</div>
                    <div style={{fontSize:10, color:C.textMid}}>{t(s.lk)}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Knowledge Base Stats Panel */}
          <div style={{background:C.surface, border:`1px solid ${C.borderBright}`, borderRadius:10, padding:14}}>
            <div style={{display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:10}}>
              <span style={{fontSize:10, color:C.textMid, fontWeight:700, letterSpacing:"0.1em"}}>
                {t("kbTitle")}
              </span>
              <button onClick={()=>fetchKbStats()} style={{
                fontSize:10, color:C.accent, background:"transparent",
                border:`1px solid ${C.accent}33`, borderRadius:4, padding:"2px 7px", cursor:"pointer",
              }}>↻</button>
            </div>

            {!kbStats ? (
              <div style={{fontSize:11, color:C.textDim, textAlign:"center", padding:"8px 0"}}>
                {t("kbLoading")}
              </div>
            ) : (
              <>
                {/* Main numbers */}
                <div style={{display:"grid", gridTemplateColumns:"1fr 1fr 1fr", gap:6, textAlign:"center", marginBottom:10}}>
                  {[
                    {value: kbStats.total_chunks, label: t("kbChunks"), color: C.accent},
                    {value: kbStats.total_sources, label: t("kbDocs"), color: C.green},
                    {value: kbStats.total_words >= 1000
                        ? `${(kbStats.total_words/1000).toFixed(1)}k`
                        : kbStats.total_words,
                      label: t("kbWords"), color: C.purple},
                  ].map((s,i)=>(
                    <div key={i} style={{background:C.bg, borderRadius:6, padding:"6px 4px", border:`1px solid ${C.border}`}}>
                      <div style={{fontSize:15, fontWeight:800, color:s.color}}>{s.value}</div>
                      <div style={{fontSize:9, color:C.textDim}}>{s.label}</div>
                    </div>
                  ))}
                </div>

                {/* Per-source list */}
                {kbStats.sources?.length > 0 && (
                  <div style={{maxHeight:120, overflowY:"auto", marginBottom:8}}>
                    {kbStats.sources.map((src,i)=>(
                      <div key={i} style={{
                        display:"flex", justifyContent:"space-between", alignItems:"center",
                        padding:"3px 0", borderBottom:`1px solid ${C.border}`, fontSize:11,
                      }}>
                        <span style={{
                          color: kbStats.stale_sources?.includes(src.source) ? C.orange : C.textMid,
                          fontFamily:"monospace", fontSize:10,
                          overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap", maxWidth:180,
                        }} title={src.source}>
                          {kbStats.stale_sources?.includes(src.source) ? "⚠ " : "📄 "}{src.source}
                        </span>
                        <span style={{color:C.textDim, fontSize:10, flexShrink:0}}>{src.chunks}×</span>
                      </div>
                    ))}
                  </div>
                )}

                {/* Stale warning */}
                {kbStats.stale_sources?.length > 0 && (
                  <div style={{
                    fontSize:10, color:C.orange, background:`${C.orange}10`,
                    border:`1px solid ${C.orange}33`, borderRadius:4, padding:"4px 8px", marginBottom:8,
                  }}>
                    ⚠ {kbStats.stale_sources.length} {t("kbStale")}
                  </div>
                )}

                {/* Contextual chunking badge */}
                {kbStats.contextual_chunking && (
                  <div style={{
                    display:"inline-flex", alignItems:"center", gap:4,
                    fontSize:10, color:C.teal, background:`${C.teal}10`,
                    border:`1px solid ${C.teal}33`, borderRadius:4, padding:"2px 7px", marginBottom:8,
                  }}>
                    ✦ {t("kbContextual")}
                  </div>
                )}

                {/* Feedback summary */}
                {kbStats.feedback?.total > 0 ? (
                  <div style={{fontSize:11, color:C.textMid, display:"flex", alignItems:"center", gap:6}}>
                    <span>{t("kbFeedbackTotal", kbStats.feedback.total)}</span>
                    {kbStats.feedback.satisfaction_rate != null && (
                      <Tag
                        label={t("kbSatRate", (kbStats.feedback.satisfaction_rate*100).toFixed(0))}
                        color={kbStats.feedback.satisfaction_rate>=0.7?C.green:C.orange}
                      />
                    )}
                  </div>
                ) : (
                  <div style={{fontSize:11, color:C.textDim}}>{t("kbNoFeedback")}</div>
                )}

                {/* Rebuild buttons */}
                <div style={{display:"flex", gap:6, marginTop:10}}>
                  <button onClick={()=>triggerRebuild(false)} disabled={kbRebuilding} style={{
                    flex:1, fontSize:10, padding:"4px 0", borderRadius:4, cursor:"pointer",
                    background:`${C.accent}12`, color:C.accent,
                    border:`1px solid ${C.accent}33`, fontFamily:"inherit",
                  }}>
                    {kbRebuilding ? t("kbRebuilding") : t("kbRebuildBtn")}
                  </button>
                  <button onClick={()=>triggerRebuild(true)} disabled={kbRebuilding} title={lang==="en"?"Re-compute all embeddings ignoring cache":"强制重新计算所有嵌入，忽略缓存"} style={{
                    fontSize:10, padding:"4px 8px", borderRadius:4, cursor:"pointer",
                    background:"transparent", color:C.textDim,
                    border:`1px solid ${C.border}`, fontFamily:"inherit",
                  }}>
                    {t("kbForceBtn")}
                  </button>
                </div>
              </>
            )}
          </div>
        </div>

        {/* ══ RIGHT PANEL ══ */}
        <div style={{display:"flex", flexDirection:"column", gap:12}}>

          {/* Tabs */}
          <div style={{display:"flex", gap:4, borderBottom:`1px solid ${C.border}`, paddingBottom:8}}>
            {TABS.map(tab=>(
              <button key={tab.key} onClick={()=>setActiveTab(tab.key)} style={{
                padding:"6px 16px", borderRadius:6, fontSize:12, fontWeight:600,
                cursor:"pointer", fontFamily:"inherit",
                background: activeTab===tab.key?`${C.accent}12`:"transparent",
                color:       activeTab===tab.key?C.accent:C.textMid,
                border:`1px solid ${activeTab===tab.key?C.accent+"55":"transparent"}`,
              }}>
                {tab.icon} {t(tab.lk)}
                {tab.key==="conversation" && conversationHistory.length>0 && (
                  <span style={{
                    marginLeft:6, background:C.orange, color:"#fff",
                    borderRadius:"50%", width:16, height:16, fontSize:9,
                    display:"inline-flex", alignItems:"center", justifyContent:"center", fontWeight:800,
                  }}>{Math.floor(conversationHistory.length/2)}</span>
                )}
              </button>
            ))}
          </div>

          {/* ── Process Tab ── */}
          {activeTab==="process" && (
            <div style={{display:"grid", gridTemplateColumns:"1fr 340px", gap:12}}>
              <div>
                {hydeDoc && (
                  <div style={{
                    background:`${C.teal}08`, border:`1px solid ${C.teal}33`,
                    borderRadius:8, padding:"10px 14px", marginBottom:12,
                  }}>
                    <div style={{fontSize:10, color:C.teal, fontWeight:700, letterSpacing:"0.1em", marginBottom:6}}>
                      {t("hydeTitle")}
                    </div>
                    <p style={{margin:0, fontSize:12, color:C.textMid, lineHeight:1.6, fontStyle:"italic"}}>{hydeDoc}</p>
                  </div>
                )}
                <div style={{
                  background:C.surface, border:`1px solid ${C.borderBright}`,
                  borderRadius:8, padding:12, height:380, overflowY:"auto",
                }}>
                  <div style={{fontSize:10, color:C.textMid, fontWeight:700, letterSpacing:"0.1em", marginBottom:8}}>
                    {t("logTitle")} {status==="running"&&<Spinner size={10}/>}
                  </div>
                  {logs.length===0 && (
                    <div style={{color:C.textDim, fontSize:12, textAlign:"center", marginTop:40}}>{t("logEmpty")}</div>
                  )}
                  {logs.map((entry,i)=><LogEntry key={i} entry={entry} lang={lang}/>)}
                  <div ref={logsEndRef}/>
                </div>
                {/* Agent Route Decision card */}
                {agentMode && agentRoute && (()=>{
                  const routes = t("agent_routes");
                  const ri = routes[agentRoute.route] || { label:agentRoute.route, color:C.textMid, icon:"•" };
                  return (
                    <div style={{
                      background:`${ri.color}08`, border:`1px solid ${ri.color}33`,
                      borderRadius:8, padding:"10px 14px", marginTop:12,
                      display:"flex", alignItems:"center", gap:10,
                    }}>
                      <span style={{fontSize:18}}>{ri.icon}</span>
                      <div>
                        <div style={{fontSize:10, color:ri.color, fontWeight:700, letterSpacing:"0.08em"}}>
                          {t("agent_route_label")}
                        </div>
                        <div style={{fontSize:13, fontWeight:700, color:ri.color}}>
                          {ri.label}
                          {agentRoute.reason && <span style={{fontSize:11, fontWeight:400, color:C.textMid, marginLeft:8}}>— {agentRoute.reason}</span>}
                        </div>
                        {agentRoute.sub_queries?.length > 0 && (
                          <div style={{fontSize:11, color:C.textMid, marginTop:4}}>
                            {agentRoute.sub_queries.map((q,i)=>(
                              <span key={i} style={{
                                display:"inline-block", margin:"2px 4px 2px 0",
                                padding:"1px 7px", borderRadius:3,
                                background:`${C.purple}12`, color:C.purple, border:`1px solid ${C.purple}30`,
                              }}>{q}</span>
                            ))}
                          </div>
                        )}
                        {agentRoute.tools?.length > 0 && (
                          <div style={{fontSize:11, color:C.textMid, marginTop:4}}>
                            {agentRoute.tools.map((t,i)=>(
                              <span key={i} style={{
                                display:"inline-block", margin:"2px 4px 2px 0",
                                padding:"1px 7px", borderRadius:3,
                                background:`${C.orange}12`, color:C.orange, border:`1px solid ${C.orange}30`,
                              }}>🔧 {t}</span>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })()}

                {/* Sub-task results for complex route */}
                {agentMode && agentSubResults.length > 0 && (
                  <div style={{background:C.surface, border:`1px solid ${C.purple}33`, borderRadius:8, padding:14, marginTop:12}}>
                    <div style={{fontSize:10, color:C.purple, fontWeight:700, letterSpacing:"0.1em", marginBottom:8}}>
                      {t("agent_sub_title")}
                    </div>
                    {agentSubResults.map((r,i)=>(
                      <div key={i} style={{
                        borderLeft:`3px solid ${C.purple}44`, paddingLeft:10,
                        marginBottom:10, paddingBottom:6,
                        borderBottom: i < agentSubResults.length-1 ? `1px solid ${C.border}` : "none",
                      }}>
                        <div style={{fontSize:11, color:C.purple, fontWeight:700, marginBottom:2}}>
                          {i+1}. {r.query}
                        </div>
                        <div style={{fontSize:12, color:C.textMid, lineHeight:1.5}}>
                          {r.answer_preview}
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                {answer && (
                  <div style={{background:C.surface, border:`1px solid ${C.green}44`, borderRadius:8, padding:14, marginTop:12}}>
                    <div style={{fontSize:10, color:C.green, fontWeight:700, letterSpacing:"0.1em", marginBottom:8}}>
                      {t("answerTitle")}
                    </div>
                    <p style={{margin:0, fontSize:13, lineHeight:1.8, color:C.text, whiteSpace:"pre-line"}}>
                      {answer}
                      {status==="running"&&<span style={{animation:"pulse 0.8s infinite",display:"inline-block"}}>▌</span>}
                    </p>

                    {/* ⑨ Answer Feedback */}
                    {status==="done" && (
                      <div style={{
                        marginTop:14, paddingTop:12,
                        borderTop:`1px solid ${C.border}`,
                        display:"flex", alignItems:"center", gap:10, flexWrap:"wrap",
                      }}>
                        {feedbackGiven ? (
                          <span style={{fontSize:12, color:C.green, fontWeight:600}}>
                            {feedbackGiven==="pos" ? "👍 " : "👎 "}{t("feedback_thanks")}
                          </span>
                        ) : (
                          <>
                            <span style={{fontSize:11, color:C.textMid}}>{t("feedbackTitle")}</span>
                            <button onClick={()=>submitFeedback(1)} disabled={feedbackLoading} style={{
                              padding:"4px 12px", borderRadius:20, fontSize:11, cursor:"pointer",
                              background:`${C.green}12`, color:C.green,
                              border:`1px solid ${C.green}44`, fontFamily:"inherit",
                            }}>👍 {t("feedback_yes")}</button>
                            <button onClick={()=>submitFeedback(-1)} disabled={feedbackLoading} style={{
                              padding:"4px 12px", borderRadius:20, fontSize:11, cursor:"pointer",
                              background:`${C.red}10`, color:C.red,
                              border:`1px solid ${C.red}33`, fontFamily:"inherit",
                            }}>👎 {t("feedback_no")}</button>
                            <input
                              value={feedbackComment}
                              onChange={e=>setFeedbackComment(e.target.value)}
                              placeholder={t("feedback_comment")}
                              style={{
                                flex:1, minWidth:120, fontSize:11, padding:"4px 8px",
                                border:`1px solid ${C.border}`, borderRadius:4,
                                background:C.bg, color:C.text, fontFamily:"inherit",
                              }}
                            />
                          </>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </div>
              <div style={{background:C.surface, border:`1px solid ${C.borderBright}`, borderRadius:8, padding:14, height:"fit-content"}}>
                <div style={{fontSize:10, color:C.textMid, fontWeight:700, letterSpacing:"0.1em", marginBottom:10}}>
                  {t("iterTitle")}
                </div>
                <IterationTimeline iterations={iterations}/>
                {iterations.length===0 && (
                  <div style={{color:C.textDim, fontSize:12, textAlign:"center", marginTop:20}}>{t("iterEmpty")}</div>
                )}
              </div>
            </div>
          )}

          {/* ── Results Tab ── */}
          {activeTab==="results" && (
            <div>
              {docs.length===0 ? (
                <div style={{textAlign:"center", padding:"60px 0", color:C.textDim, fontSize:13}}>
                  {status==="idle"?t("resEmpty_idle"):status==="running"?t("resEmpty_run"):t("resEmpty_done")}
                </div>
              ):(
                <div style={{maxHeight:640, overflowY:"auto", paddingRight:4}}>
                  <div style={{fontSize:10, color:C.textMid, fontWeight:700, letterSpacing:"0.1em", marginBottom:10}}>
                    {t("resultsTitle",docs.length)}
                  </div>
                  {docs.map((doc,i)=><DocCard key={doc.id} doc={doc} rank={i+1} lang={lang}/>)}
                </div>
              )}
            </div>
          )}

          {/* ── Metrics Tab ── */}
          {activeTab==="metrics" && (
            <div style={{display:"grid", gridTemplateColumns:"1fr 1fr", gap:12}}>
              <div>
                <RagasPanel metrics={metrics} lang={lang}/>
                <RecallChart metrics={metrics} lang={lang}/>
              </div>
              <div>
                {metrics ? (
                  <>
                    <div style={{background:C.surface, border:`1px solid ${C.borderBright}`, borderRadius:8, padding:14}}>
                      <div style={{fontSize:10, color:C.textMid, fontWeight:700, letterSpacing:"0.1em", marginBottom:12}}>
                        {t("qMetricsTitle")}
                      </div>
                      {[
                        {lk:"m_conf", value:metrics.final_confidence, color:C.green  },
                        {lk:"m_iter", value:metrics.iterative_recall, color:C.accent },
                        {lk:"m_fus",  value:metrics.fusion_recall,    color:C.green  },
                        {lk:"m_re",   value:metrics.rerank_recall,    color:C.purple },
                      ].map(m=>(
                        <div key={m.lk} style={{marginBottom:12}}>
                          <div style={{marginBottom:4}}>
                            <span style={{fontSize:12, color:C.textMid}}>{t(m.lk)}</span>
                          </div>
                          <ScoreBar value={m.value} color={m.color}/>
                        </div>
                      ))}
                    </div>

                    <div style={{background:C.surface, border:`1px solid ${C.borderBright}`, borderRadius:8, padding:14, marginTop:12}}>
                      <div style={{fontSize:10, color:C.textMid, fontWeight:700, letterSpacing:"0.1em", marginBottom:12}}>
                        {t("modTitle")}
                      </div>
                      {[
                        {lk:"mod_iter", value:0.75, color:C.accent, tag:"+15%"},
                        {lk:"mod_fus",  value:0.15, color:C.green,  tag:"+3%" },
                        {lk:"mod_ce",   value:0.10, color:C.purple, tag:"+2%" },
                      ].map(m=>(
                        <div key={m.lk} style={{marginBottom:10}}>
                          <div style={{display:"flex", justifyContent:"space-between", marginBottom:4}}>
                            <span style={{fontSize:12, color:C.text}}>{t(m.lk)}</span>
                            <Tag label={m.tag} color={m.color}/>
                          </div>
                          <ScoreBar value={m.value} color={m.color} showPercent={false}/>
                        </div>
                      ))}
                    </div>

                    <div style={{
                      marginTop:12, padding:"10px 12px", background:C.surfaceHover,
                      borderRadius:6, border:`1px solid ${C.border}`, fontSize:11, color:C.textMid, lineHeight:1.9,
                    }}>
                      📌 {t("note_dataset")}<br/>
                      📌 {t("note_iter",iterations.length)}<br/>
                      📌 {t("note_strat",strategy)}<br/>
                      📌 {t("note_hyde",enableHyde)}<br/>
                      📌 {t("note_hist",Math.floor(conversationHistory.length/2))}
                    </div>
                  </>
                ):(
                  <div style={{textAlign:"center", padding:"60px 0", color:C.textDim, fontSize:13}}>
                    {t("metricsEmpty")}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* ── Conversation Tab ── */}
          {activeTab==="conversation" && (
            <ConversationPanel history={conversationHistory} onClear={()=>setConversationHistory([])} lang={lang}/>
          )}

          {/* ── Docs Tab ── */}
          {activeTab==="docs" && (
            <DocsTab lang={lang} kbStats={kbStats} onRefresh={fetchKbStats}/>
          )}
        </div>
      </div>
    </div>
  );
}
