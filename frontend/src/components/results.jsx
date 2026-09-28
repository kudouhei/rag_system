import { C } from "../config/theme";
import { tL } from "../i18n/index.jsx";
import { ScoreBar, Tag } from "./ui.jsx";

// ── Doc Card ──────────────────────────────────────────────────────────────────
export const DocCard = ({ doc, rank, lang, rerankerActive = false }) => (
  <div style={{
    background:C.surface, border:`1px solid ${C.borderBright}`,
    borderRadius:8, padding:"12px 14px", marginBottom:8,
  }}>
    <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:8 }}>
      <div style={{ display:"flex", alignItems:"center", gap:8 }}>
        <span style={{
          width:22, height:22, borderRadius:4, background:`${C.accent}18`,
          color:C.accent, fontSize:11, fontWeight:800, display:"flex",
          alignItems:"center", justifyContent:"center", flexShrink:0,
        }}>#{rank}</span>
        <span style={{fontWeight:600, fontSize:13, color:C.text}}>{doc.title}</span>
      </div>
      <div style={{textAlign:"right"}}>
        <div style={{
          fontSize:13, fontWeight:800, fontFamily:"monospace",
          color: doc.final_score>0.75?C.green:doc.final_score>0.55?C.accent:C.orange,
        }}>{Number(doc.final_score||0).toFixed(3)}</div>
        <div style={{fontSize:9, color:C.textDim}}>
          {rerankerActive
            ? (lang==="en"?"reranker score":"精排分数")
            : (lang==="en"?"ranking score":"排序分数")}
        </div>
      </div>
    </div>
    <p style={{fontSize:12, color:C.textMid, margin:"0 0 8px", lineHeight:1.6}}>{doc.content}</p>
    {doc.source && (
      <div style={{fontSize:10, color:C.textDim, marginBottom:6, display:"flex", alignItems:"center", gap:6, flexWrap:"wrap"}}>
        <span>📁</span><span style={{fontFamily:"monospace"}}>{doc.source}</span>
        {doc.file_mtime && (
          <span style={{color:C.textDim}}>
            · {new Date(doc.file_mtime).toLocaleDateString()}
          </span>
        )}
        {doc.context_added && (
          <span style={{
            fontSize:9, padding:"1px 5px", borderRadius:3,
            background:`${C.teal}15`, color:C.teal, border:`1px solid ${C.teal}33`,
          }}>✦ contextual</span>
        )}
      </div>
    )}
    <div style={{display:"flex", gap:6, flexWrap:"wrap", marginBottom:8}}>
      {doc.tags?.map(t => <Tag key={t} label={t}/>)}
    </div>
    <div style={{display:"grid", gridTemplateColumns:"1fr 1fr", gap:6}}>
      {doc.embedding_score>0 && <ScoreBar value={doc.embedding_score} color={C.accent} label={tL(lang,"score_vec")} valueFormat="decimal"/>}
      {doc.bm25_score>0     && <ScoreBar value={doc.bm25_score} color={C.green} label={tL(lang,"score_bm25")} valueFormat="decimal"/>}
      {doc.graph_score>0    && <ScoreBar value={doc.graph_score} color="#059669" label={tL(lang,"score_graph")} valueFormat="decimal"/>}
    </div>
  </div>
);

// ── RAGAS Panel ───────────────────────────────────────────────────────────────
export const RagasPanel = ({ metrics, lang }) => {
  if (!metrics?.context_relevance) return null;
  const rows = [
    { vk:"context_relevance",  lk:"cr_label", dk:"cr_desc", color:C.accent  },
    { vk:"context_precision",  lk:"cp_label", dk:"cp_desc", color:C.green   },
    { vk:"answer_relevance",   lk:"ar_label", dk:"ar_desc", color:C.purple  },
    { vk:"answer_faithfulness",lk:"af_label", dk:"af_desc", color:C.orange  },
  ];
  return (
    <div style={{background:C.surface, border:`1px solid ${C.borderBright}`, borderRadius:8, padding:16}}>
      <div style={{display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:16}}>
        <span style={{fontSize:11, color:C.textMid, fontWeight:700, letterSpacing:"0.08em"}}>
          {tL(lang,"ragasTitle")}&nbsp;
        </span>
        <span title={tL(lang,"proxyMetricTip")} style={{fontSize:11, fontWeight:700, color:C.orange}}>
          {tL(lang,"proxyBadge")}
        </span>
      </div>
      {rows.map(r=>(
        <div key={r.vk} style={{marginBottom:14}}>
          <span title={tL(lang,r.dk)} style={{fontSize:12, color:C.text, borderBottom:`1px dashed ${C.borderBright}`, cursor:"help", display:"inline-block", marginBottom:4}}>
            {tL(lang,r.lk)}
          </span>
          <ScoreBar value={metrics[r.vk]||0} color={r.color} valueFormat="decimal"/>
        </div>
      ))}
    </div>
  );
};

// ── Iteration Timeline ─────────────────────────────────────────────────────────
export const IterationTimeline = ({ iterations }) => {
  if (!iterations?.length) return null;
  return (
    <div>
      {iterations.map((it,i)=>(
        <div key={i} style={{display:"flex", gap:10, marginBottom:8, alignItems:"flex-start"}}>
          <div style={{display:"flex", flexDirection:"column", alignItems:"center"}}>
            <div style={{
              width:24, height:24, borderRadius:"50%", flexShrink:0,
              background: it.reflected?`${C.orange}18`:`${C.green}18`,
              border:`2px solid ${it.reflected?C.orange:C.green}`,
              display:"flex", alignItems:"center", justifyContent:"center",
              fontSize:10, fontWeight:800, color:it.reflected?C.orange:C.green,
            }}>#{it.iteration}</div>
            {i<iterations.length-1 && <div style={{width:2,height:16,background:C.border,margin:"2px 0"}}/>}
          </div>
          <div style={{flex:1, background:C.surface, border:`1px solid ${C.border}`, borderRadius:6, padding:"8px 12px", fontSize:12}}>
            <div style={{display:"flex", justifyContent:"space-between", marginBottom:4}}>
              <span style={{color:C.textMid}}>Strategy: <span style={{color:C.accent}}>{it.strategy}</span></span>
              <span style={{color:it.top_score>=0.55?C.green:C.orange, fontWeight:700}}>{(it.top_score*100).toFixed(1)}%</span>
            </div>
            <div style={{color:C.text, marginBottom:4}}>
              Query: <span style={{color:C.textMid}}>「{it.query}」</span>
            </div>
            {it.reflected && <Tag label="Reflection triggered" color={C.orange}/>}
          </div>
        </div>
      ))}
    </div>
  );
};

// ── Conversation Panel ─────────────────────────────────────────────────────────
export const ConversationPanel = ({ history, onClear, lang }) => {
  if (!history.length) {
    return (
      <div style={{textAlign:"center", padding:"60px 0", color:C.textDim, fontSize:13}}>
        {tL(lang,"convEmpty")}
      </div>
    );
  }
  return (
    <div>
      <div style={{display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:12}}>
        <span style={{fontSize:11, color:C.textMid, fontWeight:700, letterSpacing:"0.08em"}}>
          {tL(lang,"convHistTitle", Math.floor(history.length/2))}
        </span>
        <button onClick={onClear} style={{
          fontSize:11, color:C.red, background:"transparent",
          border:`1px solid ${C.red}44`, borderRadius:4, padding:"2px 8px", cursor:"pointer",
        }}>{tL(lang,"clearBtn")}</button>
      </div>
      <div style={{maxHeight:560, overflowY:"auto", paddingRight:4}}>
        {history.map((turn,i)=>(
          <div key={i} style={{
            display:"flex",
            justifyContent: turn.role==="user"?"flex-end":"flex-start",
            marginBottom:10,
          }}>
            <div style={{
              maxWidth:"80%", padding:"10px 14px", borderRadius:10,
              fontSize:13, lineHeight:1.7,
              background: turn.role==="user"?`${C.accent}12`:C.surface,
              border:`1px solid ${turn.role==="user"?C.accent+"33":C.borderBright}`,
              color:C.text,
              borderTopRightRadius: turn.role==="user"?2:10,
              borderTopLeftRadius:  turn.role==="user"?10:2,
            }}>
              <div style={{fontSize:10, color:C.textDim, marginBottom:4, fontWeight:600}}>
                {turn.role==="user"?tL(lang,"role_user"):tL(lang,"role_asst")}
              </div>
              <div style={{whiteSpace:"pre-wrap"}}>{turn.content}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
