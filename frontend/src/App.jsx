import { useState, useRef, useEffect, useCallback } from "react";

// ── Color palette & design tokens ─────────────────────────────────────────
const C = {
  bg: "#0a0c10",
  surface: "#0f1318",
  surfaceHover: "#141920",
  border: "#1e2530",
  borderBright: "#2a3545",
  accent: "#00d4ff",
  accentDim: "#0099bb",
  accentGlow: "rgba(0,212,255,0.15)",
  green: "#00e5a0",
  greenDim: "#00b07a",
  orange: "#ff8c42",
  orangeDim: "#cc6a28",
  red: "#ff4d6d",
  purple: "#b06fff",
  text: "#e8edf5",
  textMid: "#8899aa",
  textDim: "#4a5568",
};

// ── Utility Components ─────────────────────────────────────────────────────

const Tag = ({ label, color = C.accent }) => (
  <span style={{
    padding: "2px 8px", borderRadius: 4, fontSize: 11, fontWeight: 600,
    background: `${color}22`, color, border: `1px solid ${color}44`,
    letterSpacing: "0.04em", whiteSpace: "nowrap",
  }}>{label}</span>
);

const ScoreBar = ({ value, color = C.accent, label, showPercent = true }) => (
  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
    {label && <span style={{ color: C.textMid, fontSize: 11, minWidth: 56 }}>{label}</span>}
    <div style={{ flex: 1, height: 6, background: C.border, borderRadius: 3, overflow: "hidden" }}>
      <div style={{
        width: `${value * 100}%`, height: "100%", borderRadius: 3,
        background: `linear-gradient(90deg, ${color}88, ${color})`,
        transition: "width 0.6s cubic-bezier(0.4,0,0.2,1)",
      }} />
    </div>
    {showPercent && (
      <span style={{ color, fontSize: 11, fontWeight: 700, minWidth: 36, textAlign: "right" }}>
        {(value * 100).toFixed(1)}%
      </span>
    )}
  </div>
);

const Pill = ({ children, active, onClick, color = C.accent }) => (
  <button onClick={onClick} style={{
    padding: "6px 14px", borderRadius: 6, fontSize: 12, fontWeight: 600,
    cursor: "pointer", transition: "all 0.2s", letterSpacing: "0.03em",
    background: active ? `${color}22` : "transparent",
    color: active ? color : C.textMid,
    border: `1px solid ${active ? color + "66" : C.border}`,
    outline: "none",
  }}>{children}</button>
);

const Spinner = ({ size = 16, color = C.accent }) => (
  <div style={{
    width: size, height: size, borderRadius: "50%",
    border: `2px solid ${color}33`, borderTopColor: color,
    animation: "spin 0.7s linear infinite", display: "inline-block",
  }} />
);

// ── Phase Badge ──────────────────────────────────────────────────────────────
const PHASES = {
  retrieval: { label: "RETRIEVAL", color: C.accent },
  reranking: { label: "RERANKING", color: C.purple },
  generation: { label: "GENERATION", color: C.green },
  reflection: { label: "REFLECTION", color: C.orange },
};

const PhaseBadge = ({ phase }) => {
  const p = PHASES[phase] || { label: phase.toUpperCase(), color: C.textMid };
  return (
    <span style={{
      padding: "2px 10px", borderRadius: 4, fontSize: 10, fontWeight: 800,
      background: `${p.color}22`, color: p.color, border: `1px solid ${p.color}55`,
      letterSpacing: "0.08em",
    }}>{p.label}</span>
  );
};

// ── Log Entry ─────────────────────────────────────────────────────────────────
const LogEntry = ({ entry }) => {
  const icons = {
    pipeline_start: "⚡", phase_start: "▶", doc_scored: "📄",
    retrieval_done: "✅", reflection: "🤔", query_rewrite: "✏️",
    rerank_score: "🔢", reranking_done: "🎯", answer_token: "💬",
    pipeline_complete: "🏁", error: "❌",
  };
  
  const getContent = () => {
    switch (entry.type) {
      case "pipeline_start":
        return <span>开始处理查询：<em style={{ color: C.accent }}>「{entry.query}」</em></span>;
      case "phase_start":
        return <span><PhaseBadge phase={entry.phase} /> &nbsp;{entry.message}</span>;
      case "doc_scored":
        return (
          <span style={{ fontSize: 12 }}>
            <span style={{ color: C.textMid }}>{entry.title.slice(0, 20)}…</span>
            &nbsp;→&nbsp;
            <span style={{ color: C.accent }}>向量 {(entry.embedding_score * 100).toFixed(0)}%</span>
            &nbsp;
            <span style={{ color: C.green }}>BM25 {(entry.bm25_score * 100).toFixed(0)}%</span>
            &nbsp;
            <span style={{ color: C.text, fontWeight: 700 }}>综合 {(entry.final_score * 100).toFixed(0)}%</span>
          </span>
        );
      case "retrieval_done":
        return (
          <span>
            第 {entry.iteration} 轮检索完成，最高分：
            <span style={{ color: entry.top_score >= entry.threshold ? C.green : C.orange, fontWeight: 700 }}>
              {(entry.top_score * 100).toFixed(1)}%
            </span>
            &nbsp;(阈值 {(entry.threshold * 100).toFixed(0)}%)
          </span>
        );
      case "reflection":
        return (
          <span>
            <span style={{ color: C.orange }}>反思触发：</span>
            &nbsp;{entry.failure_reason}
          </span>
        );
      case "query_rewrite":
        return (
          <span>
            查询重写：
            <span style={{ color: C.textMid, textDecoration: "line-through" }}>「{entry.original_query}」</span>
            &nbsp;→&nbsp;
            <span style={{ color: C.accent }}>「{entry.new_query}」</span>
          </span>
        );
      case "rerank_score":
        return (
          <span style={{ fontSize: 12 }}>
            <span style={{ color: C.textMid }}>{entry.title.slice(0, 18)}…</span>
            &nbsp;精排分：
            <span style={{ color: C.purple, fontWeight: 700 }}>{(entry.ce_score * 100).toFixed(1)}%</span>
            &nbsp;
            <span style={{ color: entry.improvement > 0 ? C.green : C.orange, fontSize: 11 }}>
              ({entry.improvement > 0 ? "+" : ""}{(entry.improvement * 100).toFixed(1)}%)
            </span>
          </span>
        );
      case "reranking_done":
        return <span>重排序完成，最高分：<span style={{ color: C.purple, fontWeight: 700 }}>{(entry.top_score * 100).toFixed(1)}%</span></span>;
      case "pipeline_complete":
        return <span style={{ color: C.green, fontWeight: 700 }}>✓ 管道完成，耗时 {entry.elapsed_seconds}s，共 {entry.total_iterations} 轮检索</span>;
      default:
        return <span>{entry.message || JSON.stringify(entry).slice(0, 80)}</span>;
    }
  };
  
  return (
    <div style={{
      padding: "5px 0", borderBottom: `1px solid ${C.border}`,
      fontSize: 12.5, color: C.text, display: "flex", gap: 8, alignItems: "flex-start",
    }}>
      <span style={{ opacity: 0.6, flexShrink: 0, marginTop: 1 }}>{icons[entry.type] || "•"}</span>
      <span style={{ lineHeight: 1.5 }}>{getContent()}</span>
    </div>
  );
};

// ── Doc Card ─────────────────────────────────────────────────────────────────
const DocCard = ({ doc, rank }) => (
  <div style={{
    background: C.surface, border: `1px solid ${C.borderBright}`,
    borderRadius: 8, padding: "12px 14px", marginBottom: 8,
    transition: "border-color 0.2s",
  }}>
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span style={{
          width: 22, height: 22, borderRadius: 4, background: `${C.accent}22`,
          color: C.accent, fontSize: 11, fontWeight: 800, display: "flex",
          alignItems: "center", justifyContent: "center", flexShrink: 0,
        }}>#{rank}</span>
        <span style={{ fontWeight: 600, fontSize: 13, color: C.text }}>{doc.title}</span>
      </div>
      <span style={{
        fontSize: 14, fontWeight: 800, color: doc.final_score > 0.75 ? C.green : doc.final_score > 0.55 ? C.accent : C.orange,
      }}>{(doc.final_score * 100).toFixed(1)}%</span>
    </div>
    <p style={{ fontSize: 12, color: C.textMid, margin: "0 0 8px", lineHeight: 1.6 }}>{doc.content}</p>
    <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 8 }}>
      {doc.tags?.map(t => <Tag key={t} label={t} />)}
    </div>
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
      {doc.embedding_score > 0 && <ScoreBar value={doc.embedding_score} color={C.accent} label="向量" />}
      {doc.bm25_score > 0 && <ScoreBar value={doc.bm25_score} color={C.green} label="BM25" />}
    </div>
  </div>
);

// ── Metrics Chart ─────────────────────────────────────────────────────────────
const MetricsPanel = ({ metrics }) => {
  if (!metrics) return null;
  
  const stages = [
    { label: "基线", value: metrics.baseline_recall, color: C.textDim },
    { label: "迭代检索", value: metrics.iterative_recall, color: C.accent, delta: "+15%" },
    { label: "多策略融合", value: metrics.fusion_recall, color: C.green, delta: "+3%" },
    { label: "重排序", value: metrics.rerank_recall, color: C.purple, delta: "+2%" },
  ];
  
  const maxVal = Math.max(...stages.map(s => s.value));
  const chartH = 80;
  
  return (
    <div style={{
      background: C.surface, border: `1px solid ${C.borderBright}`,
      borderRadius: 8, padding: "16px", marginTop: 12,
    }}>
      <div style={{ fontSize: 11, color: C.textMid, fontWeight: 700, letterSpacing: "0.08em", marginBottom: 16 }}>
        召回率提升效果 (Natural Questions)
      </div>
      <div style={{ display: "flex", alignItems: "flex-end", gap: 12, height: chartH + 36 }}>
        {stages.map((s, i) => {
          const barH = (s.value / maxVal) * chartH;
          return (
            <div key={i} style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
              <span style={{ fontSize: 11, color: s.color, fontWeight: 700 }}>
                {(s.value * 100).toFixed(1)}%
              </span>
              {s.delta && (
                <span style={{ fontSize: 10, color: C.green, fontWeight: 600 }}>{s.delta}</span>
              )}
              {!s.delta && <span style={{ fontSize: 10 }}>&nbsp;</span>}
              <div style={{
                width: "100%", height: barH,
                background: `linear-gradient(180deg, ${s.color}cc, ${s.color}55)`,
                borderRadius: "4px 4px 0 0", border: `1px solid ${s.color}66`,
                transition: "height 0.8s cubic-bezier(0.4,0,0.2,1)",
              }} />
              <span style={{ fontSize: 10, color: C.textMid, textAlign: "center", lineHeight: 1.3 }}>
                {s.label}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
};

// ── Iteration Timeline ────────────────────────────────────────────────────────
const IterationTimeline = ({ iterations }) => {
  if (!iterations?.length) return null;
  
  return (
    <div style={{ marginTop: 12 }}>
      <div style={{ fontSize: 11, color: C.textMid, fontWeight: 700, letterSpacing: "0.08em", marginBottom: 10 }}>
        检索迭代轨迹
      </div>
      {iterations.map((it, i) => (
        <div key={i} style={{ display: "flex", gap: 10, marginBottom: 8, alignItems: "flex-start" }}>
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 0 }}>
            <div style={{
              width: 24, height: 24, borderRadius: "50%", flexShrink: 0,
              background: it.reflected ? `${C.orange}22` : `${C.green}22`,
              border: `2px solid ${it.reflected ? C.orange : C.green}`,
              display: "flex", alignItems: "center", justifyContent: "center",
              fontSize: 10, fontWeight: 800,
              color: it.reflected ? C.orange : C.green,
            }}>#{it.iteration}</div>
            {i < iterations.length - 1 && (
              <div style={{ width: 2, height: 16, background: C.border, margin: "2px 0" }} />
            )}
          </div>
          <div style={{
            flex: 1, background: C.surface, border: `1px solid ${C.border}`,
            borderRadius: 6, padding: "8px 12px", fontSize: 12,
          }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
              <span style={{ color: C.textMid }}>
                策略：<span style={{ color: C.accent }}>{it.strategy}</span>
              </span>
              <span style={{ color: it.top_score >= 0.55 ? C.green : C.orange, fontWeight: 700 }}>
                {(it.top_score * 100).toFixed(1)}%
              </span>
            </div>
            <div style={{ color: C.text, marginBottom: 4 }}>
              查询：<span style={{ color: C.textMid }}>「{it.query}」</span>
            </div>
            {it.reflected && (
              <Tag label="触发反思重写" color={C.orange} />
            )}
          </div>
        </div>
      ))}
    </div>
  );
};

// ── Main App ──────────────────────────────────────────────────────────────────
export default function RAGDashboard() {
  const [query, setQuery] = useState("企业知识库如何实现高效检索？");
  const [strategy, setStrategy] = useState("adaptive");
  const [enableIterative, setEnableIterative] = useState(true);
  const [enableRerank, setEnableRerank] = useState(true);
  const [threshold, setThreshold] = useState(0.55);
  
  const [status, setStatus] = useState("idle"); // idle | running | done | error
  const [logs, setLogs] = useState([]);
  const [docs, setDocs] = useState([]);
  const [answer, setAnswer] = useState("");
  const [metrics, setMetrics] = useState(null);
  const [iterations, setIterations] = useState([]);
  const [elapsed, setElapsed] = useState(null);
  const [activeTab, setActiveTab] = useState("process"); // process | results | metrics
  
  const wsRef = useRef(null);
  const logsEndRef = useRef(null);
  
  useEffect(() => {
    if (logsEndRef.current && activeTab === "process") {
      logsEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [logs, activeTab]);

  const handleMessage = useCallback((evt) => {
    const msg = JSON.parse(evt.data);
    
    if (msg.type === "answer_token") {
      setAnswer(msg.full_answer_so_far || "");
      return;
    }
    
    if (msg.type === "pipeline_complete") {
      setDocs(msg.retrieved_docs || []);
      setMetrics(msg.metrics || null);
      setIterations(msg.iterations_detail || []);
      setElapsed(msg.elapsed_seconds);
      setStatus("done");
    }
    
    if (msg.type === "error") {
      setStatus("error");
    }
    
    setLogs(prev => [...prev, msg]);
  }, []);
  
  const runQuery = useCallback(() => {
    if (!query.trim() || status === "running") return;
    
    setStatus("running");
    setLogs([]);
    setDocs([]);
    setAnswer("");
    setMetrics(null);
    setIterations([]);
    setElapsed(null);
    setActiveTab("process");
    
    const ws = new WebSocket("ws://localhost:8000/ws/query");
    wsRef.current = ws;
    
    ws.onopen = () => {
      ws.send(JSON.stringify({
        query,
        strategy,
        enable_iterative: enableIterative,
        enable_rerank: enableRerank,
        confidence_threshold: threshold,
        top_k: 5,
      }));
    };
    
    ws.onmessage = handleMessage;
    
    ws.onerror = () => {
      setStatus("error");
      setLogs(prev => [...prev, { type: "error", message: "WebSocket 连接失败，请确保后端服务运行在 localhost:8000" }]);
    };
    
    ws.onclose = () => {
      if (status === "running") setStatus("done");
    };
  }, [query, strategy, enableIterative, enableRerank, threshold, status, handleMessage]);
  
  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      runQuery();
    }
  };

  const STRATEGIES = [
    { key: "adaptive", label: "自适应" },
    { key: "hybrid", label: "混合" },
    { key: "vector", label: "向量" },
    { key: "bm25", label: "BM25" },
  ];

  const SAMPLE_QUERIES = [
    "企业知识库如何实现高效检索？",
    "什么是ReAct框架在检索中的应用？",
    "如何评估RAG系统的召回率？",
    "交叉编码器重排序的原理是什么？",
  ];

  return (
    <div style={{
      minHeight: "100vh", background: C.bg, color: C.text,
      fontFamily: "'JetBrains Mono', 'Fira Code', 'Consolas', monospace",
      padding: "24px",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700;800&display=swap');
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes pulse { 0%,100% { opacity:1 } 50% { opacity:0.5 } }
        * { box-sizing: border-box; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: ${C.bg}; }
        ::-webkit-scrollbar-thumb { background: ${C.border}; border-radius: 2px; }
        textarea:focus, input:focus { outline: none; }
        button:hover { opacity: 0.85; }
      `}</style>

      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 24 }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <div style={{
              width: 36, height: 36, borderRadius: 8,
              background: `linear-gradient(135deg, ${C.accent}44, ${C.purple}44)`,
              border: `1px solid ${C.accent}66`,
              display: "flex", alignItems: "center", justifyContent: "center",
              fontSize: 18,
            }}>⚡</div>
            <div>
              <h1 style={{ margin: 0, fontSize: 18, fontWeight: 800, color: C.text, letterSpacing: "-0.02em" }}>
                Adaptive RAG System
              </h1>
              <p style={{ margin: 0, fontSize: 11, color: C.textMid }}>
                迭代检索 · 多策略融合 · 交叉编码器精排
              </p>
            </div>
          </div>
        </div>
        <div style={{ display: "flex", gap: 6 }}>
          {[
            { label: "迭代检索 +15%", color: C.accent },
            { label: "多策略 +3%", color: C.green },
            { label: "重排序 +2%", color: C.purple },
          ].map(b => <Tag key={b.label} label={b.label} color={b.color} />)}
        </div>
      </div>
      
      <div style={{ display: "grid", gridTemplateColumns: "380px 1fr", gap: 16, maxWidth: 1400 }}>
        
        {/* ── LEFT PANEL ──────────────────────────────────────────────────── */}
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          
          {/* Query Input */}
          <div style={{
            background: C.surface, border: `1px solid ${C.borderBright}`,
            borderRadius: 10, padding: 16,
          }}>
            <div style={{ fontSize: 10, color: C.textMid, fontWeight: 700, letterSpacing: "0.1em", marginBottom: 10 }}>
              QUERY INPUT
            </div>
            <textarea
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="输入查询… (Enter 发送)"
              rows={3}
              style={{
                width: "100%", background: C.bg, border: `1px solid ${C.border}`,
                borderRadius: 6, padding: "10px 12px", fontSize: 13,
                color: C.text, resize: "vertical", lineHeight: 1.6,
                fontFamily: "inherit",
              }}
            />
            
            {/* Sample queries */}
            <div style={{ marginTop: 8, display: "flex", flexWrap: "wrap", gap: 4 }}>
              {SAMPLE_QUERIES.map(q => (
                <button key={q} onClick={() => setQuery(q)} style={{
                  background: "transparent", border: `1px solid ${C.border}`,
                  borderRadius: 4, padding: "3px 8px", fontSize: 10,
                  color: C.textMid, cursor: "pointer", fontFamily: "inherit",
                  whiteSpace: "nowrap", overflow: "hidden", maxWidth: 160,
                  textOverflow: "ellipsis",
                }}>
                  {q.slice(0, 18)}…
                </button>
              ))}
            </div>
          </div>
          
          {/* Config */}
          <div style={{
            background: C.surface, border: `1px solid ${C.borderBright}`,
            borderRadius: 10, padding: 16,
          }}>
            <div style={{ fontSize: 10, color: C.textMid, fontWeight: 700, letterSpacing: "0.1em", marginBottom: 12 }}>
              CONFIGURATION
            </div>
            
            <div style={{ marginBottom: 12 }}>
              <div style={{ fontSize: 11, color: C.textMid, marginBottom: 6 }}>检索策略</div>
              <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
                {STRATEGIES.map(s => (
                  <Pill key={s.key} active={strategy === s.key} onClick={() => setStrategy(s.key)}>
                    {s.label}
                  </Pill>
                ))}
              </div>
            </div>
            
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginBottom: 12 }}>
              {[
                { label: "迭代检索", val: enableIterative, set: setEnableIterative, color: C.accent },
                { label: "交叉编码器精排", val: enableRerank, set: setEnableRerank, color: C.purple },
              ].map(opt => (
                <button key={opt.label} onClick={() => opt.set(!opt.val)} style={{
                  padding: "8px 10px", borderRadius: 6, cursor: "pointer",
                  background: opt.val ? `${opt.color}15` : "transparent",
                  border: `1px solid ${opt.val ? opt.color + "55" : C.border}`,
                  color: opt.val ? opt.color : C.textMid,
                  fontSize: 11, fontWeight: 600, fontFamily: "inherit",
                  display: "flex", alignItems: "center", gap: 6,
                }}>
                  <span style={{
                    width: 8, height: 8, borderRadius: "50%",
                    background: opt.val ? opt.color : C.textDim,
                    animation: opt.val ? "pulse 2s infinite" : "none",
                  }} />
                  {opt.label}
                </button>
              ))}
            </div>
            
            <div>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                <span style={{ fontSize: 11, color: C.textMid }}>置信度阈值</span>
                <span style={{ fontSize: 11, color: C.accent, fontWeight: 700 }}>
                  {(threshold * 100).toFixed(0)}%
                </span>
              </div>
              <input type="range" min={0.3} max={0.85} step={0.05} value={threshold}
                onChange={e => setThreshold(Number(e.target.value))}
                style={{ width: "100%", accentColor: C.accent }} />
            </div>
          </div>
          
          {/* Run Button */}
          <button onClick={runQuery} disabled={status === "running"} style={{
            padding: "12px 20px", borderRadius: 8, fontSize: 13, fontWeight: 800,
            cursor: status === "running" ? "not-allowed" : "pointer",
            background: status === "running"
              ? `${C.accent}22`
              : `linear-gradient(135deg, ${C.accentDim}, ${C.accent})`,
            color: status === "running" ? C.accentDim : "#000",
            border: "none", letterSpacing: "0.04em", fontFamily: "inherit",
            display: "flex", alignItems: "center", justifyContent: "center", gap: 8,
            transition: "all 0.2s",
          }}>
            {status === "running" ? (
              <><Spinner size={14} color={C.accent} /> 检索中…</>
            ) : "⚡ 执行检索"}
          </button>
          
          {/* Stats */}
          {status === "done" && elapsed && (
            <div style={{
              background: C.surface, border: `1px solid ${C.green}44`,
              borderRadius: 8, padding: 12,
            }}>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8, textAlign: "center" }}>
                {[
                  { label: "耗时", value: `${elapsed}s`, color: C.green },
                  { label: "检索轮次", value: iterations.length, color: C.accent },
                  { label: "召回文档", value: docs.length, color: C.purple },
                ].map(s => (
                  <div key={s.label}>
                    <div style={{ fontSize: 18, fontWeight: 800, color: s.color }}>{s.value}</div>
                    <div style={{ fontSize: 10, color: C.textMid }}>{s.label}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
        
        {/* ── RIGHT PANEL ─────────────────────────────────────────────────── */}
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          
          {/* Tabs */}
          <div style={{ display: "flex", gap: 4, borderBottom: `1px solid ${C.border}`, paddingBottom: 8 }}>
            {[
              { key: "process", label: "执行过程", icon: "⚙" },
              { key: "results", label: "检索结果", icon: "📋" },
              { key: "metrics", label: "效果分析", icon: "📊" },
            ].map(t => (
              <button key={t.key} onClick={() => setActiveTab(t.key)} style={{
                padding: "6px 16px", borderRadius: 6, fontSize: 12, fontWeight: 600,
                cursor: "pointer", fontFamily: "inherit",
                background: activeTab === t.key ? `${C.accent}22` : "transparent",
                color: activeTab === t.key ? C.accent : C.textMid,
                border: `1px solid ${activeTab === t.key ? C.accent + "66" : "transparent"}`,
              }}>
                {t.icon} {t.label}
              </button>
            ))}
          </div>
          
          {/* Process Tab */}
          {activeTab === "process" && (
            <div style={{ display: "grid", gridTemplateColumns: "1fr 360px", gap: 12 }}>
              <div>
                {/* Log stream */}
                <div style={{
                  background: C.surface, border: `1px solid ${C.borderBright}`,
                  borderRadius: 8, padding: 12, height: 420, overflowY: "auto",
                }}>
                  <div style={{ fontSize: 10, color: C.textMid, fontWeight: 700, letterSpacing: "0.1em", marginBottom: 8 }}>
                    EXECUTION LOG {status === "running" && <Spinner size={10} />}
                  </div>
                  {logs.length === 0 && (
                    <div style={{ color: C.textDim, fontSize: 12, textAlign: "center", marginTop: 40 }}>
                      等待执行…
                    </div>
                  )}
                  {logs.map((entry, i) => <LogEntry key={i} entry={entry} />)}
                  <div ref={logsEndRef} />
                </div>
                
                {/* Answer */}
                {answer && (
                  <div style={{
                    background: C.surface, border: `1px solid ${C.green}44`,
                    borderRadius: 8, padding: 14, marginTop: 12,
                  }}>
                    <div style={{ fontSize: 10, color: C.green, fontWeight: 700, letterSpacing: "0.1em", marginBottom: 8 }}>
                      GENERATED ANSWER
                    </div>
                    <p style={{ margin: 0, fontSize: 12.5, lineHeight: 1.8, color: C.text, whiteSpace: "pre-line" }}>
                      {answer}
                      {status === "running" && <span style={{ animation: "pulse 0.8s infinite", display: "inline-block" }}>▌</span>}
                    </p>
                  </div>
                )}
              </div>
              
              {/* Iteration Timeline */}
              <div style={{
                background: C.surface, border: `1px solid ${C.borderBright}`,
                borderRadius: 8, padding: 14, height: "fit-content",
              }}>
                <div style={{ fontSize: 10, color: C.textMid, fontWeight: 700, letterSpacing: "0.1em", marginBottom: 8 }}>
                  ITERATION TRACE
                </div>
                {iterations.length === 0 ? (
                  <div style={{ color: C.textDim, fontSize: 12, textAlign: "center", marginTop: 20 }}>
                    暂无数据
                  </div>
                ) : (
                  <IterationTimeline iterations={iterations} />
                )}
              </div>
            </div>
          )}
          
          {/* Results Tab */}
          {activeTab === "results" && (
            <div>
              {docs.length === 0 ? (
                <div style={{ textAlign: "center", padding: "60px 0", color: C.textDim, fontSize: 13 }}>
                  {status === "idle" ? "请先执行检索" : status === "running" ? "检索中…" : "无结果"}
                </div>
              ) : (
                <div style={{ maxHeight: 620, overflowY: "auto", paddingRight: 4 }}>
                  <div style={{ fontSize: 10, color: C.textMid, fontWeight: 700, letterSpacing: "0.1em", marginBottom: 10 }}>
                    TOP-{docs.length} 检索结果（已精排）
                  </div>
                  {docs.map((doc, i) => <DocCard key={doc.id} doc={doc} rank={i + 1} />)}
                </div>
              )}
            </div>
          )}
          
          {/* Metrics Tab */}
          {activeTab === "metrics" && (
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              <div>
                <MetricsPanel metrics={metrics} />
                
                {metrics && (
                  <div style={{
                    background: C.surface, border: `1px solid ${C.borderBright}`,
                    borderRadius: 8, padding: 14, marginTop: 12,
                  }}>
                    <div style={{ fontSize: 10, color: C.textMid, fontWeight: 700, letterSpacing: "0.1em", marginBottom: 12 }}>
                      各模块贡献
                    </div>
                    {[
                      { label: "迭代式检索（ReAct）", value: 0.15 / 0.20, color: C.accent, tag: "+15%" },
                      { label: "多策略融合检索", value: 0.03 / 0.20, color: C.green, tag: "+3%" },
                      { label: "交叉编码器精排", value: 0.02 / 0.20, color: C.purple, tag: "+2%" },
                    ].map(m => (
                      <div key={m.label} style={{ marginBottom: 10 }}>
                        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                          <span style={{ fontSize: 12, color: C.text }}>{m.label}</span>
                          <Tag label={m.tag} color={m.color} />
                        </div>
                        <ScoreBar value={m.value} color={m.color} showPercent={false} />
                      </div>
                    ))}
                  </div>
                )}
              </div>
              
              <div>
                {metrics && (
                  <div style={{
                    background: C.surface, border: `1px solid ${C.borderBright}`,
                    borderRadius: 8, padding: 14,
                  }}>
                    <div style={{ fontSize: 10, color: C.textMid, fontWeight: 700, letterSpacing: "0.1em", marginBottom: 12 }}>
                      本次查询指标
                    </div>
                    {[
                      { label: "最终置信度", value: metrics.final_confidence, color: C.green },
                      { label: "迭代后召回率", value: metrics.iterative_recall, color: C.accent },
                      { label: "融合后召回率", value: metrics.fusion_recall, color: C.green },
                      { label: "精排后召回率", value: metrics.rerank_recall, color: C.purple },
                    ].map(m => (
                      <div key={m.label} style={{ marginBottom: 12 }}>
                        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                          <span style={{ fontSize: 12, color: C.textMid }}>{m.label}</span>
                        </div>
                        <ScoreBar value={m.value} color={m.color} />
                      </div>
                    ))}
                    
                    <div style={{
                      marginTop: 16, padding: "10px 12px",
                      background: C.bg, borderRadius: 6,
                      border: `1px solid ${C.border}`, fontSize: 11, color: C.textMid,
                    }}>
                      📌 评测数据集：Natural Questions<br />
                      📌 迭代轮次：{iterations.length} 轮<br />
                      📌 检索策略：{strategy === "adaptive" ? "自适应（动态切换）" : strategy}
                    </div>
                  </div>
                )}
                
                {!metrics && (
                  <div style={{ textAlign: "center", padding: "60px 0", color: C.textDim, fontSize: 13 }}>
                    请先执行检索以查看分析
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
