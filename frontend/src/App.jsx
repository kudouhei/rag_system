import { useState, useRef, useEffect, useCallback } from "react";
import { API_BASE, backendWsUrl } from "./config/api";
import { C } from "./config/theme";
import { I18N, tL } from "./i18n/index.jsx";
import { usePersistedState } from "./hooks/usePersistedState";
import { Pill, ScoreBar, Spinner, Tag, TopProgress } from "./components/ui.jsx";
import { LogEntry } from "./components/logging.jsx";
import { ConversationPanel, DocCard, IterationTimeline, RagasPanel, RecallChart } from "./components/results.jsx";
import { DocsTab } from "./components/DocsTab.jsx";

// ══════════════════════════════════════════════════════════════════════════════
// Main App
// ══════════════════════════════════════════════════════════════════════════════
export default function RAGDashboard() {
  const [lang, setLang]                   = usePersistedState("lang", "zh");
  const t = useCallback((key,...args)=>tL(lang,key,...args),[lang]);

  const [query, setQuery]                 = useState("忘记 SSO 密码或 MFA 丢失怎么办？需要提交什么工单？");
  const [strategy, setStrategy]           = usePersistedState("strategy", "adaptive");
  const [enableIterative, setEnableIterative] = usePersistedState("enableIterative", true);
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
  const [kbRebuilding, setKbRebuilding]   = useState(false);

  // Backend readiness (prevents WS attempts while the API is still starting)
  const [backendReady, setBackendReady]   = useState(false);
  const [isNarrow, setIsNarrow]           = useState(false);

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

  // Responsive layout helper (Process tab)
  useEffect(() => {
    const onResize = () => setIsNarrow(window.innerWidth < 1100);
    onResize();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

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
  },[backendReady,query,strategy,enableIterative,enableHyde,enableGraph,enableConversation,agentMode,threshold,status,lang,conversationHistory,handleMessage]);

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
        @keyframes progress { 0%{transform:translateX(-60%)} 100%{transform:translateX(260%)} }
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
            {enableGraph && <Tag label={t("badge_graph")} color="#059669"/>}
            {agentMode && <Tag label={t("badge_agent")} color={C.orange}/>}
          </div>
        </div>
      </div>

      <div style={{
        display:"grid",
        gridTemplateColumns: isNarrow ? "minmax(0, 1fr)" : "420px minmax(0, 1fr)",
        gap:16,
        width:"100%",
      }}>

        {/* ══ LEFT PANEL ══ */}
        <div style={{display:"flex", flexDirection:"column", gap:12, minWidth:0, width:"100%"}}>

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
            {status==="running"?<><Spinner size={18} color={C.accent}/>{t("runningBtn")}</>:t("runBtn")}
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
        <div style={{display:"flex", flexDirection:"column", gap:12, minWidth:0, width:"100%"}}>
          <TopProgress active={status==="running"} color={C.accent}/>

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
            <div style={{
              display:"grid",
              // Two-column responsive layout (no middle column)
              gridTemplateColumns: isNarrow ? "minmax(0, 1fr)" : "minmax(0, 1fr) minmax(0, 1fr)",
              gap:12,
              minWidth:0,
            }}>
              {/* Left column: HyDE + Execution log + Iteration trace */}
              <div style={{minWidth:0}}>
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
                  borderRadius:8, padding:12,
                  height:"min(420px, 45vh)",
                  overflowY:"auto",
                  minWidth:0, overflowX:"hidden",
                }}>
                  <div style={{fontSize:10, color:C.textMid, fontWeight:700, letterSpacing:"0.1em", marginBottom:8}}>
                    {t("logTitle")} {status==="running"&&<span style={{display:"inline-flex",alignItems:"center",gap:6,marginLeft:6}}>
                      <Spinner size={14} color={C.accent}/>
                      <span style={{color:C.accent, fontWeight:800}}>{lang==="zh" ? "执行中" : "Running"}</span>
                    </span>}
                  </div>
                  {logs.length===0 && (
                    <div style={{color:C.textDim, fontSize:12, textAlign:"center", marginTop:40}}>{t("logEmpty")}</div>
                  )}
                  {logs.map((entry,i)=><LogEntry key={i} entry={entry} lang={lang}/>)}
                  <div ref={logsEndRef}/>
                </div>

                {/* Agent Route Decision card (below execution log) */}
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

                {/* Sub-task results (moved here after removing middle column) */}
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

                {/* Iteration trace (below execution log) */}
                <div style={{background:C.surface, border:`1px solid ${C.borderBright}`, borderRadius:8, padding:14, marginTop:12}}>
                  <div style={{fontSize:10, color:C.textMid, fontWeight:700, letterSpacing:"0.1em", marginBottom:10}}>
                    {t("iterTitle")}
                  </div>
                  <IterationTimeline iterations={iterations}/>
                  {iterations.length===0 && (
                    <div style={{color:C.textDim, fontSize:12, textAlign:"center", marginTop:20}}>{t("iterEmpty")}</div>
                  )}
                </div>
              </div>

              {/* Right column: Generated answer */}
              {!isNarrow && <div style={{minWidth:0}}>
                {answer ? (
                  <div style={{
                    background:C.surface, border:`1px solid ${C.green}44`, borderRadius:8, padding:14,
                    maxHeight:"calc(100vh - 260px)", overflowY:"auto",
                  }}>
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
                ) : (
                  <div style={{background:C.surface, border:`1px solid ${C.borderBright}`, borderRadius:8, padding:14, color:C.textDim, fontSize:12}}>
                    {status==="idle"
                      ? (lang==="en" ? "Run a query to generate an answer." : "执行检索后将在此处显示生成答案。")
                      : (lang==="en" ? "Generating…" : "生成中…")}
                  </div>
                )}
              </div>}

              {/* Narrow layout: answer goes below */}
              {isNarrow && (
                <div style={{minWidth:0}}>
                  {answer ? (
                    <div style={{
                      background:C.surface, border:`1px solid ${C.green}44`, borderRadius:8, padding:14,
                      maxHeight:"min(520px, 55vh)", overflowY:"auto",
                    }}>
                      <div style={{fontSize:10, color:C.green, fontWeight:700, letterSpacing:"0.1em", marginBottom:8}}>
                        {t("answerTitle")}
                      </div>
                      <p style={{margin:0, fontSize:13, lineHeight:1.8, color:C.text, whiteSpace:"pre-line"}}>
                        {answer}
                        {status==="running"&&<span style={{animation:"pulse 0.8s infinite",display:"inline-block"}}>▌</span>}
                      </p>
                    </div>
                  ) : (
                    <div style={{background:C.surface, border:`1px solid ${C.borderBright}`, borderRadius:8, padding:14, color:C.textDim, fontSize:12}}>
                      {status==="idle"
                        ? (lang==="en" ? "Run a query to generate an answer." : "执行检索后将在此处显示生成答案。")
                        : (lang==="en" ? "Generating…" : "生成中…")}
                    </div>
                  )}
                </div>
              )}
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
