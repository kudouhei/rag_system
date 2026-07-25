import { C } from "../../config/theme";
import { I18N } from "../../i18n/index.jsx";
import { Tag } from "../ui.jsx";

export const Header = ({ lang, setLang, setQuery, enableGraph, agentMode, t }) => (
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
);
