import { useState, useRef, useEffect, useCallback } from "react";
import { API_BASE, backendWsUrl } from "./config/api";
import { C } from "./config/theme";
import { tL } from "./i18n/index.jsx";
import { usePersistedState } from "./hooks/usePersistedState";
import { TopProgress } from "./components/ui.jsx";
import { ConversationPanel } from "./components/results.jsx";
import { DocsTab } from "./components/DocsTab.jsx";
import { Header } from "./components/layout/Header.jsx";
import { LeftPanel } from "./components/layout/LeftPanel.jsx";
import { TabBar } from "./components/layout/TabBar.jsx";
import { ProcessTab } from "./components/process/ProcessTab.jsx";
import { ResultsTab } from "./components/tabs/ResultsTab.jsx";
import { MetricsTab } from "./components/tabs/MetricsTab.jsx";
import { GraphTab } from "./components/tabs/GraphTab.jsx";

const TABS = [
  {key:"process",      lk:"tab_process", icon:"⚙"},
  {key:"results",      lk:"tab_results", icon:"📋"},
  {key:"metrics",      lk:"tab_metrics", icon:"📊"},
  {key:"graph",        lk:"tab_graph",   icon:"🕸"},
  {key:"conversation", lk:"tab_conv",    icon:"💬"},
  {key:"docs",         lk:"tab_docs",    icon:"📁"},
];

export default function RAGDashboard() {
  const [lang, setLang]                   = usePersistedState("lang", "zh");
  const t = useCallback((key,...args)=>tL(lang,key,...args),[lang]);

  const [query, setQuery]                 = useState("忘记 SSO 密码或 MFA 丢失怎么办？需要提交什么工单？");
  const [strategy, setStrategy]           = usePersistedState("strategy", "adaptive");
  const [enableIterative, setEnableIterative] = usePersistedState("enableIterative", true);
  const [enableConversation, setEnableConversation] = usePersistedState("enableConversation", false);
  const [enableGraph, setEnableGraph]     = usePersistedState("enableGraph", false);
  const [agentMode, setAgentMode]         = usePersistedState("agentMode", false);
  const [agentRoute, setAgentRoute]       = useState(null);
  const [agentSubResults, setAgentSubResults] = useState([]);
  const [threshold, setThreshold]         = usePersistedState("threshold", 0.55);

  const [status, setStatus]               = useState("idle");
  const [logs, setLogs]                   = useState([]);
  const [docs, setDocs]                   = useState([]);
  const [answer, setAnswer]               = useState("");
  const [metrics, setMetrics]             = useState(null);
  const [iterations, setIterations]       = useState([]);
  const [elapsed, setElapsed]             = useState(null);
  const [conversationHistory, setConversationHistory] = usePersistedState("conversationHistory", []);
  const [activeTab, setActiveTab]         = useState("process");

  const [kbStats, setKbStats]             = useState(null);
  const [kbRebuilding, setKbRebuilding]   = useState(false);
  const [backendReady, setBackendReady]   = useState(false);
  const [isNarrow, setIsNarrow]           = useState(false);

  const [feedbackGiven, setFeedbackGiven] = useState(null);
  const [feedbackLoading, setFeedbackLoading] = useState(false);
  const [feedbackComment, setFeedbackComment] = useState("");

  const wsRef             = useRef(null);
  const logsEndRef        = useRef(null);
  const submittedQueryRef = useRef("");

  const fetchKbStats = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/stats`);
      if (res.ok) setKbStats(await res.json());
    } catch { /* backend not running yet */ }
  }, []);

  useEffect(() => { fetchKbStats(); }, [fetchKbStats]);

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
      setTimeout(() => { setKbRebuilding(false); fetchKbStats(); }, 4000);
    } catch { setKbRebuilding(false); }
  }, [fetchKbStats]);

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
    if (msg.type==="answer_token")    { setAnswer(msg.full_answer_so_far||""); return; }
    if (msg.type==="pipeline_complete") {
      setDocs(msg.retrieved_docs||[]);
      setMetrics(msg.metrics||null);
      setIterations(msg.iterations_detail||[]);
      setElapsed(msg.elapsed_seconds);
      setStatus("done");
    }
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
    setActiveTab("process");
    setFeedbackGiven(null); setFeedbackComment("");
    setAgentRoute(null); setAgentSubResults([]);

    const ws = new WebSocket(backendWsUrl(agentMode ? "/ws/agent" : "/ws/query"));
    wsRef.current=ws;
    ws.onopen=()=>ws.send(JSON.stringify({
      query, strategy,
      enable_iterative:     enableIterative,
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
  },[backendReady,query,strategy,enableIterative,enableGraph,enableConversation,agentMode,threshold,status,lang,conversationHistory,handleMessage]);

  const handleKeyDown=(e)=>{ if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();runQuery();} };

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

      <Header
        lang={lang}
        setLang={setLang}
        setQuery={setQuery}
        agentMode={agentMode}
        t={t}
      />

      <div style={{
        display:"grid",
        gridTemplateColumns: isNarrow ? "minmax(0, 1fr)" : "420px minmax(0, 1fr)",
        gap:16,
        width:"100%",
      }}>
        <LeftPanel
          lang={lang}
          t={t}
          query={query}
          setQuery={setQuery}
          handleKeyDown={handleKeyDown}
          backendReady={backendReady}
          agentMode={agentMode}
          setAgentMode={setAgentMode}
          enableConversation={enableConversation}
          setEnableConversation={setEnableConversation}
          conversationHistory={conversationHistory}
          setConversationHistory={setConversationHistory}
          strategy={strategy}
          setStrategy={setStrategy}
          enableIterative={enableIterative}
          setEnableIterative={setEnableIterative}
          enableGraph={enableGraph}
          setEnableGraph={setEnableGraph}
          threshold={threshold}
          setThreshold={setThreshold}
          status={status}
          runQuery={runQuery}
          elapsed={elapsed}
          iterations={iterations}
          docs={docs}
          kbStats={kbStats}
          kbRebuilding={kbRebuilding}
          fetchKbStats={fetchKbStats}
          triggerRebuild={triggerRebuild}
        />

        <div style={{display:"flex", flexDirection:"column", gap:12, minWidth:0, width:"100%"}}>
          <TopProgress active={status==="running"} color={C.accent}/>

          <TabBar
            tabs={TABS}
            activeTab={activeTab}
            setActiveTab={setActiveTab}
            conversationHistory={conversationHistory}
            t={t}
          />

          {activeTab==="process" && (
            <ProcessTab
              lang={lang}
              t={t}
              status={status}
              isNarrow={isNarrow}
              logs={logs}
              logsEndRef={logsEndRef}
              agentMode={agentMode}
              agentRoute={agentRoute}
              agentSubResults={agentSubResults}
              iterations={iterations}
              answer={answer}
              feedbackGiven={feedbackGiven}
              feedbackLoading={feedbackLoading}
              feedbackComment={feedbackComment}
              setFeedbackComment={setFeedbackComment}
              submitFeedback={submitFeedback}
            />
          )}

          {activeTab==="results" && (
            <ResultsTab docs={docs} status={status} lang={lang} t={t}/>
          )}

          {activeTab==="metrics" && (
            <MetricsTab
              metrics={metrics}
              iterations={iterations}
              strategy={strategy}
              conversationHistory={conversationHistory}
              lang={lang}
              t={t}
            />
          )}

          {activeTab==="graph" && (
            <GraphTab lang={lang} t={t} active={activeTab==="graph"}/>
          )}

          {activeTab==="conversation" && (
            <ConversationPanel history={conversationHistory} onClear={()=>setConversationHistory([])} lang={lang}/>
          )}

          {activeTab==="docs" && (
            <DocsTab lang={lang} kbStats={kbStats} onRefresh={fetchKbStats}/>
          )}
        </div>
      </div>
    </div>
  );
}
