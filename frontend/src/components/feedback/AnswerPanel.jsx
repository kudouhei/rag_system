import { C } from "../../config/theme";

const renderInline = (text) =>
  text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).filter(Boolean).map((part,i) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={i}>{part.slice(2,-2)}</strong>;
    }
    if (part.startsWith("`") && part.endsWith("`")) {
      return <code key={i} style={{
        padding:"1px 5px", borderRadius:4, background:C.surfaceHover,
        border:`1px solid ${C.border}`, color:C.purple, fontSize:"0.92em",
      }}>{part.slice(1,-1)}</code>;
    }
    return part;
  });

const renderLine = (line, i) => {
  const trimmed = line.trim();
  if (!trimmed) return <div key={i} style={{height:8}}/>;

  const heading = trimmed.match(/^#{1,3}\s+(.+)$/);
  if (heading) {
    return (
      <div key={i} style={{fontWeight:800, fontSize:13.5, marginTop:i?10:0, marginBottom:2}}>
        {renderInline(heading[1])}
      </div>
    );
  }

  const bullet = trimmed.match(/^[-*]\s+(.+)$/);
  if (bullet) {
    return (
      <div key={i} style={{display:"flex", gap:8, paddingLeft:4}}>
        <span style={{color:C.accent, fontWeight:800}}>•</span>
        <span>{renderInline(bullet[1])}</span>
      </div>
    );
  }

  return <div key={i}>{renderInline(line)}</div>;
};

export const AnswerPanel = ({
  answer,
  status,
  lang,
  t,
  maxHeight,
  docs = [],
  onOpenEvidence,
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
        <div style={{display:"flex", alignItems:"center", justifyContent:"space-between", gap:8, marginBottom:10}}>
          <div style={{fontSize:10, color:C.green, fontWeight:700, letterSpacing:"0.1em"}}>
            {t("answerTitle")}
          </div>
          <span style={{
            fontSize:10, color:C.green, background:`${C.green}10`,
            border:`1px solid ${C.green}33`, padding:"2px 7px", borderRadius:999,
          }}>
            {t("groundedBadge", docs.length)}
          </span>
        </div>
        <div style={{fontSize:13, lineHeight:1.8, color:C.text}}>
          {answer.split("\n").map(renderLine)}
          {status==="running"&&<span style={{animation:"pulse 0.8s infinite",display:"inline-block"}}>▌</span>}
        </div>

        {docs.length > 0 && (
          <div style={{marginTop:14, paddingTop:12, borderTop:`1px solid ${C.border}`}}>
            <div style={{fontSize:10, color:C.textMid, fontWeight:700, letterSpacing:"0.08em", marginBottom:7}}>
              {t("supportingEvidence")}
            </div>
            <div style={{display:"flex", gap:6, flexWrap:"wrap"}}>
              {docs.slice(0,4).map((doc,i)=>(
                <button key={doc.id} onClick={onOpenEvidence} title={doc.source || doc.title} style={{
                  padding:"4px 9px", borderRadius:6, cursor:"pointer",
                  color:C.accent, background:`${C.accent}0d`,
                  border:`1px solid ${C.accent}33`, fontSize:10.5, fontFamily:"inherit",
                  maxWidth:190, overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap",
                }}>
                  [{i+1}] {doc.title}
                </button>
              ))}
            </div>
            <div style={{fontSize:10, color:C.textDim, marginTop:7}}>
              {t("citationHint")}
            </div>
          </div>
        )}

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
