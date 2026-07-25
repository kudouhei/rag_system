import { C } from "../../config/theme";
import { DocCard } from "../results.jsx";

export const ResultsTab = ({ docs, status, lang, t }) => (
  <div>
    {docs.length===0 ? (
      <div style={{textAlign:"center", padding:"60px 0", color:C.textDim, fontSize:13}}>
        {status==="idle"?t("resEmpty_idle"):status==="running"?t("resEmpty_run"):t("resEmpty_done")}
      </div>
    ):(
      <div style={{maxHeight:640, overflowY:"auto", paddingRight:4}}>
        <div style={{fontSize:10, color:C.textMid, fontWeight:700, letterSpacing:"0.1em", marginBottom:10}}>
          {t("resultsTitle",docs.length)}
        </div>
        {docs.map((doc,i)=><DocCard key={doc.id} doc={doc} rank={i+1} lang={lang}/>) }
      </div>
    )}
  </div>
);
