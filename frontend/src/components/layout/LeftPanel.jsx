import { API_BASE } from "../../config/api";
import { C } from "../../config/theme";
import { Pill, Spinner } from "../ui.jsx";
import { KbStatsPanel } from "../kb/KbStatsPanel.jsx";

const STRATEGIES = [
  {key:"adaptive"},{key:"hybrid"},{key:"vector"},{key:"bm25"},
];

const ToggleBtn = ({ labelKey, val, onToggle, color, tipKey, t }) => (
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

export const LeftPanel = ({
  lang,
  t,
  query,
  setQuery,
  handleKeyDown,
  backendReady,
  agentMode,
  setAgentMode,
  enableConversation,
  setEnableConversation,
  conversationHistory,
  setConversationHistory,
  strategy,
  setStrategy,
  enableIterative,
  setEnableIterative,
  enableHyde,
  setEnableHyde,
  enableGraph,
  setEnableGraph,
  threshold,
  setThreshold,
  status,
  runQuery,
  elapsed,
  iterations,
  docs,
  kbStats,
  kbRebuilding,
  fetchKbStats,
  triggerRebuild,
}) => (
  <div style={{display:"flex", flexDirection:"column", gap:12, minWidth:0, width:"100%"}}>
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

    <div style={{background:C.surface, border:`1px solid ${C.borderBright}`, borderRadius:10, padding:16}}>
      <div style={{fontSize:10, color:C.textMid, fontWeight:700, letterSpacing:"0.1em", marginBottom:14}}>
        {t("configLabel")}
      </div>

      <div style={{marginBottom:12}}>
        <div style={{
          fontSize:9, fontWeight:700, letterSpacing:"0.12em",
          color:C.textDim, marginBottom:6, textTransform:"uppercase",
        }}>{lang==="zh" ? "运行模式" : "Mode"}</div>
        <div style={{display:"flex", gap:6}}>
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
          <ToggleBtn labelKey="toggle_conv" val={enableConversation} onToggle={()=>setEnableConversation(!enableConversation)} color={C.orange} tipKey="tip_conv" t={t}/>
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

      <div style={{marginBottom:12}}>
        <div style={{
          fontSize:9, fontWeight:700, letterSpacing:"0.12em",
          color:C.textDim, marginBottom:6, textTransform:"uppercase",
        }}>{lang==="zh" ? "检索增强" : "Enhancements"}</div>
        <div style={{display:"grid", gridTemplateColumns:"1fr 1fr", gap:6}}>
          <ToggleBtn labelKey="toggle_iterative" val={enableIterative} onToggle={()=>setEnableIterative(!enableIterative)} color={C.accent}  tipKey="tip_iterative" t={t}/>
          <ToggleBtn labelKey="toggle_hyde"      val={enableHyde}      onToggle={()=>setEnableHyde(!enableHyde)}           color={C.teal}   tipKey="tip_hyde" t={t}/>
          <ToggleBtn labelKey="toggle_graph"     val={enableGraph}     onToggle={()=>setEnableGraph(!enableGraph)}         color="#059669"  tipKey="tip_graph" t={t}/>
        </div>
      </div>

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

    <KbStatsPanel
      kbStats={kbStats}
      kbRebuilding={kbRebuilding}
      fetchKbStats={fetchKbStats}
      triggerRebuild={triggerRebuild}
      lang={lang}
      t={t}
    />
  </div>
);
