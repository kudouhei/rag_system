import { C } from "../config/theme";
import { tL } from "../i18n/index.jsx";

// ── Phase Badge ───────────────────────────────────────────────────────────────
export const PHASES = {
  retrieval:  { label:"RETRIEVAL",  color:C.accent  },
  reranking:  { label:"RERANKING",  color:C.purple  },
  generation: { label:"GENERATION", color:C.green   },
  reflection: { label:"EVALUATION", color:C.orange  },
  hyde:       { label:"HyDE",       color:C.teal    },
};
export const PhaseBadge = ({ phase }) => {
  const p = PHASES[phase] || { label:phase.toUpperCase(), color:C.textMid };
  return (
    <span style={{
      padding:"2px 10px", borderRadius:4, fontSize:10, fontWeight:800,
      background:`${p.color}18`, color:p.color, border:`1px solid ${p.color}44`,
      letterSpacing:"0.08em",
    }}>{p.label}</span>
  );
};

// ── Log Entry ─────────────────────────────────────────────────────────────────
export const LogEntry = ({ entry, lang }) => {
  const icons = {
    pipeline_start:"⚡", phase_start:"▶", doc_scored:"📄",
    retrieval_done:"✅", reflection:"🤔", query_rewrite:"✏️",
    rerank_score:"🔢", reranking_done:"🎯", answer_token:"💬",
    pipeline_complete:"🏁", error:"❌", hyde_generation:"🔮",
    // Agent events
    agent_routing:"🔍", agent_route:"🚦", agent_tool_call:"🔧",
    agent_tool_result:"✅", agent_decompose:"🧩", agent_subquery:"▷",
    agent_subresult:"◈", agent_complete:"🏁",
  };

  const getContent = () => {
    switch (entry.type) {
      case "pipeline_start":     return tL(lang, "log_start", entry.query);
      case "phase_start":        return <span><PhaseBadge phase={entry.phase}/>&nbsp;&nbsp;{entry.message}</span>;
      case "hyde_generation":    return tL(lang, "log_hyde", entry.hypothetical_doc || "");
      case "doc_scored":         return <span style={{fontSize:12}}>{tL(lang,"log_docScored",entry)}</span>;
      case "retrieval_done":     return tL(lang, "log_retDone", entry);
      case "reflection":         return tL(lang, "log_reflect", entry);
      case "query_rewrite":      return tL(lang, "log_rewrite", entry);
      case "rerank_score":       return <span style={{fontSize:12}}>{tL(lang,"log_rerank",entry)}</span>;
      case "reranking_done":     return tL(lang, "log_rerankDone", entry);
      case "pipeline_complete":  return tL(lang, "log_done", entry);
      // Agent events
      case "agent_routing":      return <span style={{color:C.textMid}}>{tL(lang,"log_agent_routing")}</span>;
      case "agent_route":        return tL(lang, "log_agent_route", entry);
      case "agent_tool_call":    return tL(lang, "log_agent_tool", entry);
      case "agent_tool_result":  return tL(lang, "log_agent_result", entry);
      case "agent_decompose":    return tL(lang, "log_agent_decomp", entry);
      case "agent_subquery":     return tL(lang, "log_agent_subq", entry);
      case "agent_subresult":    return tL(lang, "log_agent_subr", entry);
      case "agent_complete":     return tL(lang, "log_agent_done", entry);
      default: return <span>{entry.message || JSON.stringify(entry).slice(0,80)}</span>;
    }
  };

  return (
    <div style={{
      padding:"5px 0", borderBottom:`1px solid ${C.border}`,
      fontSize:12.5, color:C.text, display:"flex", gap:8, alignItems:"flex-start",
    }}>
      <span style={{opacity:0.55, flexShrink:0, marginTop:1}}>{icons[entry.type]||"•"}</span>
      <span style={{
        lineHeight:1.5,
        minWidth:0,
        overflowWrap:"anywhere",
        wordBreak:"break-word",
      }}>{getContent()}</span>
    </div>
  );
};
