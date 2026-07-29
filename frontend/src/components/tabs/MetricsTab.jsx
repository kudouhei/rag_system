import { C } from "../../config/theme";
import { RagasPanel, RecallChart } from "../results.jsx";
import { ScoreBar, Tag } from "../ui.jsx";

export const MetricsTab = ({ metrics, iterations, strategy, conversationHistory, lang, t }) => (
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
);
