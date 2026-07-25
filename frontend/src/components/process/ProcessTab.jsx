import { C } from "../../config/theme";
import { LogEntry } from "../logging.jsx";
import { IterationTimeline } from "../results.jsx";
import { Spinner } from "../ui.jsx";
import { AnswerPanel } from "../feedback/AnswerPanel.jsx";

export const ProcessTab = ({
  lang,
  t,
  status,
  isNarrow,
  hydeDoc,
  logs,
  logsEndRef,
  agentMode,
  agentRoute,
  agentSubResults,
  iterations,
  answer,
  feedbackGiven,
  feedbackLoading,
  feedbackComment,
  setFeedbackComment,
  submitFeedback,
}) => (
  <div style={{
    display:"grid",
    gridTemplateColumns: isNarrow ? "minmax(0, 1fr)" : "minmax(0, 1fr) minmax(0, 1fr)",
    gap:12,
    minWidth:0,
  }}>
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
        {logs.map((entry,i)=><LogEntry key={i} entry={entry} lang={lang}/>) }
        <div ref={logsEndRef}/>
      </div>

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

    {!isNarrow && (
      <AnswerPanel
        answer={answer}
        status={status}
        lang={lang}
        t={t}
        maxHeight="calc(100vh - 260px)"
        showFeedback
        feedbackGiven={feedbackGiven}
        feedbackLoading={feedbackLoading}
        feedbackComment={feedbackComment}
        setFeedbackComment={setFeedbackComment}
        submitFeedback={submitFeedback}
      />
    )}

    {isNarrow && (
      <AnswerPanel
        answer={answer}
        status={status}
        lang={lang}
        t={t}
        maxHeight="min(520px, 55vh)"
      />
    )}
  </div>
);
