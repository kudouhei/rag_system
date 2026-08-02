import { C } from "../../config/theme";
import { LogEntry } from "../logging.jsx";
import { IterationTimeline } from "../results.jsx";
import { Spinner } from "../ui.jsx";
import { AnswerPanel } from "../feedback/AnswerPanel.jsx";

const EvidenceSummary = ({ docs, status, lang, t, onOpenEvidence }) => (
  <div style={{
    background:C.surface, border:`1px solid ${C.borderBright}`,
    borderRadius:10, padding:14, minWidth:0,
  }}>
    <div style={{display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:10}}>
      <div style={{fontSize:10, color:C.textMid, fontWeight:700, letterSpacing:"0.1em"}}>
        {t("evidenceSummary")}
      </div>
      {docs.length>0 && (
        <button onClick={onOpenEvidence} style={{
          border:"none", background:"transparent", color:C.accent,
          cursor:"pointer", fontSize:10.5, fontWeight:700, fontFamily:"inherit",
        }}>{t("viewAllEvidence")} →</button>
      )}
    </div>

    {docs.length===0 ? (
      <div style={{color:C.textDim, fontSize:12, padding:"26px 4px", textAlign:"center"}}>
        {status==="running"?t("resEmpty_run"):t("evidenceEmpty")}
      </div>
    ) : (
      <>
        {docs.slice(0,3).map((doc,i)=>(
          <button key={doc.id} onClick={onOpenEvidence} style={{
            width:"100%", display:"block", textAlign:"left", cursor:"pointer",
            background:C.bg, border:`1px solid ${C.border}`, borderRadius:8,
            padding:"9px 10px", marginBottom:7, fontFamily:"inherit",
          }}>
            <div style={{display:"flex", alignItems:"center", gap:7, minWidth:0}}>
              <span style={{
                width:20, height:20, borderRadius:5, flexShrink:0,
                display:"inline-flex", alignItems:"center", justifyContent:"center",
                color:C.accent, background:`${C.accent}12`, fontSize:10, fontWeight:800,
              }}>[{i+1}]</span>
              <span style={{
                flex:1, minWidth:0, overflow:"hidden", textOverflow:"ellipsis",
                whiteSpace:"nowrap", color:C.text, fontSize:11.5, fontWeight:650,
              }}>{doc.title}</span>
              <span title={t("rankingScoreTip")} style={{
                color:C.textMid, fontSize:10.5, fontFamily:"monospace",
              }}>{Number(doc.final_score||0).toFixed(3)}</span>
            </div>
            <div style={{
              marginTop:5, color:C.textDim, fontSize:10, overflow:"hidden",
              textOverflow:"ellipsis", whiteSpace:"nowrap",
            }}>{doc.source || (lang==="en"?"Source unavailable":"暂无来源信息")}</div>
          </button>
        ))}
        <div style={{fontSize:10, color:C.textDim, lineHeight:1.5, marginTop:8}}>
          ⓘ {t("rankingScoreTip")}
        </div>
      </>
    )}
  </div>
);

export const ProcessTab = ({
  lang,
  t,
  status,
  isNarrow,
  logs,
  logsEndRef,
  agentMode,
  agentRoute,
  agentSubResults,
  iterations,
  answer,
  docs,
  onOpenEvidence,
  feedbackGiven,
  feedbackLoading,
  feedbackComment,
  setFeedbackComment,
  submitFeedback,
}) => (
  <div style={{display:"flex", flexDirection:"column", gap:12, minWidth:0}}>
    <div style={{
      display:"grid",
      gridTemplateColumns:isNarrow?"minmax(0,1fr)":"minmax(0,1.7fr) minmax(280px,.8fr)",
      gap:12, minWidth:0, alignItems:"start",
    }}>
      <AnswerPanel
        answer={answer}
        status={status}
        lang={lang}
        t={t}
        docs={docs}
        onOpenEvidence={onOpenEvidence}
        maxHeight={isNarrow?"none":"calc(100vh - 230px)"}
        showFeedback
        feedbackGiven={feedbackGiven}
        feedbackLoading={feedbackLoading}
        feedbackComment={feedbackComment}
        setFeedbackComment={setFeedbackComment}
        submitFeedback={submitFeedback}
      />
      <EvidenceSummary
        docs={docs}
        status={status}
        lang={lang}
        t={t}
        onOpenEvidence={onOpenEvidence}
      />
    </div>

    {agentMode && agentRoute && (()=>{
        const routes = t("agent_routes");
        const ri = routes[agentRoute.route] || { label:agentRoute.route, color:C.textMid, icon:"•" };
        return (
          <div style={{
            background:`${ri.color}08`, border:`1px solid ${ri.color}33`,
            borderRadius:8, padding:"10px 14px",
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
                  {agentRoute.tools.map((tool,i)=>(
                    <span key={i} style={{
                      display:"inline-block", margin:"2px 4px 2px 0",
                      padding:"1px 7px", borderRadius:3,
                      background:`${C.orange}12`, color:C.orange, border:`1px solid ${C.orange}30`,
                    }}>🔧 {tool}</span>
                  ))}
                </div>
              )}
            </div>
          </div>
        );
    })()}

    {agentMode && agentSubResults.length > 0 && (
        <div style={{background:C.surface, border:`1px solid ${C.purple}33`, borderRadius:8, padding:14}}>
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

    <details style={{
      background:C.surface, border:`1px solid ${C.borderBright}`,
      borderRadius:10, padding:"11px 14px",
    }}>
      <summary style={{
        cursor:"pointer", color:C.textMid, fontSize:11, fontWeight:750,
        letterSpacing:"0.07em", userSelect:"none",
      }}>
        {t("technicalDetails")}
        <span style={{marginLeft:8, fontWeight:500, color:C.textDim, letterSpacing:0}}>
          {status==="running"
            ? <><Spinner size={11} color={C.accent}/> {t("runningStatus")}</>
            : t("technicalSummary", logs.length, iterations.length)}
        </span>
      </summary>
      <div style={{
        display:"grid", gridTemplateColumns:isNarrow?"1fr":"1.5fr 1fr",
        gap:12, marginTop:12,
      }}>
        <div style={{
          background:C.bg, border:`1px solid ${C.border}`,
          borderRadius:8, padding:12, maxHeight:360, overflowY:"auto",
          minWidth:0, overflowX:"hidden",
        }}>
          <div style={{fontSize:10, color:C.textMid, fontWeight:700, letterSpacing:"0.1em", marginBottom:8}}>
            {t("logTitle")}
          </div>
          {logs.length===0 && (
            <div style={{color:C.textDim, fontSize:12, textAlign:"center", marginTop:30}}>{t("logEmpty")}</div>
          )}
          {logs.map((entry,i)=><LogEntry key={i} entry={entry} lang={lang}/>)}
          <div ref={logsEndRef}/>
        </div>
        <div style={{background:C.bg, border:`1px solid ${C.border}`, borderRadius:8, padding:12}}>
          <div style={{fontSize:10, color:C.textMid, fontWeight:700, letterSpacing:"0.1em", marginBottom:10}}>
            {t("iterTitle")}
          </div>
          <IterationTimeline iterations={iterations}/>
          {iterations.length===0 && (
            <div style={{color:C.textDim, fontSize:12, textAlign:"center", marginTop:20}}>{t("iterEmpty")}</div>
          )}
        </div>
      </div>
    </details>
  </div>
);
