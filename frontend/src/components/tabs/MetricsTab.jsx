import { C } from "../../config/theme";
import { RagasPanel } from "../results.jsx";
import { Tag } from "../ui.jsx";

const FactRow = ({ label, value, hint }) => (
  <div style={{
    display:"flex", justifyContent:"space-between", alignItems:"baseline",
    gap:14, padding:"9px 0", borderBottom:`1px solid ${C.border}`,
  }}>
    <span style={{fontSize:12, color:C.textMid}}>{label}</span>
    <span title={hint} style={{fontSize:12, color:C.text, fontWeight:700, textAlign:"right"}}>{value}</span>
  </div>
);

export const MetricsTab = ({
  metrics, iterations, strategy, conversationHistory, docs, elapsed,
  pipelineConfig, isNarrow, lang, t,
}) => (
  <div style={{display:"grid", gridTemplateColumns:isNarrow?"1fr":"1fr 1fr", gap:12}}>
    <div>
      <RagasPanel metrics={metrics} lang={lang}/>
      {metrics && (
        <div style={{
          marginTop:12, padding:"11px 12px", borderRadius:8,
          background:`${C.orange}0b`, border:`1px solid ${C.orange}30`,
          color:C.textMid, fontSize:11, lineHeight:1.7,
        }}>
          <strong style={{color:C.orange}}>{t("evaluationScopeTitle")}</strong><br/>
          {t("evaluationScopeBody")}
        </div>
      )}
    </div>
    <div>
      {metrics ? (
        <>
          <div style={{background:C.surface, border:`1px solid ${C.borderBright}`, borderRadius:8, padding:14}}>
            <div style={{fontSize:10, color:C.textMid, fontWeight:700, letterSpacing:"0.1em", marginBottom:12}}>
              {t("qMetricsTitle")}
            </div>
            <FactRow
              label={t("topRankingScore")}
              value={Number(metrics.final_confidence||0).toFixed(3)}
              hint={t("rankingScoreTip")}
            />
            <FactRow label={t("evidenceReturned")} value={docs.length}/>
            <FactRow label={t("stat_iters")} value={iterations.length}/>
            <FactRow label={t("stat_elapsed")} value={elapsed!=null?`${elapsed}s`:"—"}/>
            <FactRow label={t("strategyLabel")} value={strategy}/>
          </div>

          <div style={{background:C.surface, border:`1px solid ${C.borderBright}`, borderRadius:8, padding:14, marginTop:12}}>
            <div style={{fontSize:10, color:C.textMid, fontWeight:700, letterSpacing:"0.1em", marginBottom:12}}>
              {t("runtimeConfigTitle")}
            </div>
            <FactRow
              label={t("mod_ce")}
              value={<Tag
                label={pipelineConfig?.cross_encoder?t("enabled"):t("disabled")}
                color={pipelineConfig?.cross_encoder?C.green:C.textDim}
              />}
            />
            <FactRow
              label={t("mod_iter")}
              value={<Tag
                label={pipelineConfig?.enable_iterative?t("enabled"):t("disabled")}
                color={pipelineConfig?.enable_iterative?C.green:C.textDim}
              />}
            />
            <FactRow
              label={t("toggle_graph")}
              value={<Tag
                label={pipelineConfig?.enable_graph?t("enabled"):t("disabled")}
                color={pipelineConfig?.enable_graph?C.green:C.textDim}
              />}
            />
          </div>

          <div style={{
            marginTop:12, padding:"10px 12px", background:C.surfaceHover,
            borderRadius:6, border:`1px solid ${C.border}`, fontSize:11, color:C.textMid, lineHeight:1.9,
          }}>
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
