import { C } from "../../config/theme";

export const AnswerPanel = ({
  answer,
  status,
  lang,
  t,
  maxHeight,
  showFeedback = false,
  feedbackGiven,
  feedbackLoading,
  feedbackComment,
  setFeedbackComment,
  submitFeedback,
}) => (
  <div style={{minWidth:0}}>
    {answer ? (
      <div style={{
        background:C.surface, border:`1px solid ${C.green}44`, borderRadius:8, padding:14,
        maxHeight, overflowY:"auto",
      }}>
        <div style={{fontSize:10, color:C.green, fontWeight:700, letterSpacing:"0.1em", marginBottom:8}}>
          {t("answerTitle")}
        </div>
        <p style={{margin:0, fontSize:13, lineHeight:1.8, color:C.text, whiteSpace:"pre-line"}}>
          {answer}
          {status==="running"&&<span style={{animation:"pulse 0.8s infinite",display:"inline-block"}}>▌</span>}
        </p>

        {showFeedback && status==="done" && (
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
  </div>
);
