import { useState } from "react";
import { C } from "../../config/theme";
import { Tag } from "../ui.jsx";

export const KbStatsPanel = ({ kbStats, kbRebuilding, fetchKbStats, triggerRebuild, lang, t }) => {
  const [expanded, setExpanded] = useState(false);
  return (
  <div style={{background:C.surface, border:`1px solid ${C.borderBright}`, borderRadius:10, padding:14}}>
    <div style={{display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:10}}>
      <span style={{fontSize:10, color:C.textMid, fontWeight:700, letterSpacing:"0.1em"}}>
        {t("kbTitle")}
      </span>
      <div style={{display:"flex", gap:6}}>
      <button onClick={()=>setExpanded(v=>!v)} title={expanded?t("collapse"):t("expand")} style={{
        fontSize:10, color:C.textMid, background:"transparent",
        border:`1px solid ${C.border}`, borderRadius:4, padding:"2px 7px", cursor:"pointer",
      }}>{expanded?"−":"+"}</button>
      <button onClick={()=>fetchKbStats()} style={{
        fontSize:10, color:C.accent, background:"transparent",
        border:`1px solid ${C.accent}33`, borderRadius:4, padding:"2px 7px", cursor:"pointer",
      }}>↻</button>
      </div>
    </div>

    {!kbStats ? (
      <div style={{fontSize:11, color:C.textDim, textAlign:"center", padding:"8px 0"}}>
        {t("kbLoading")}
      </div>
    ) : (
      <>
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

        {expanded && kbStats.sources?.length > 0 && (
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

        {expanded && kbStats.stale_sources?.length > 0 && (
          <div style={{
            fontSize:10, color:C.orange, background:`${C.orange}10`,
            border:`1px solid ${C.orange}33`, borderRadius:4, padding:"4px 8px", marginBottom:8,
          }}>
            ⚠ {kbStats.stale_sources.length} {t("kbStale")}
          </div>
        )}

        {expanded && kbStats.contextual_chunking && (
          <div style={{
            display:"inline-flex", alignItems:"center", gap:4,
            fontSize:10, color:C.teal, background:`${C.teal}10`,
            border:`1px solid ${C.teal}33`, borderRadius:4, padding:"2px 7px", marginBottom:8,
          }}>
            ✦ {t("kbContextual")}
          </div>
        )}

        {expanded && (kbStats.feedback?.total > 0 ? (
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
        ))}

        {expanded && <div style={{display:"flex", gap:6, marginTop:10}}>
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
        </div>}
      </>
    )}
  </div>
  );
};
