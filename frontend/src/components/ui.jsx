import { C } from "../config/theme";

// ── Utility Components ────────────────────────────────────────────────────────

export const Tag = ({ label, color = C.accent }) => (
  <span style={{
    padding:"2px 8px", borderRadius:4, fontSize:11, fontWeight:600,
    background:`${color}18`, color, border:`1px solid ${color}40`,
    letterSpacing:"0.04em", whiteSpace:"nowrap",
  }}>{label}</span>
);

export const ScoreBar = ({ value, color = C.accent, label, showPercent = true }) => (
  <div style={{ display:"flex", alignItems:"center", gap:8 }}>
    {label && <span style={{ color:C.textMid, fontSize:11, minWidth:56 }}>{label}</span>}
    <div style={{ flex:1, height:6, background:C.border, borderRadius:3, overflow:"hidden" }}>
      <div style={{
        width:`${Math.min(value*100,100)}%`, height:"100%", borderRadius:3,
        background:`linear-gradient(90deg,${color}88,${color})`,
        transition:"width 0.6s cubic-bezier(0.4,0,0.2,1)",
      }}/>
    </div>
    {showPercent && (
      <span style={{ color, fontSize:11, fontWeight:700, minWidth:38, textAlign:"right" }}>
        {(value*100).toFixed(1)}%
      </span>
    )}
  </div>
);

export const Pill = ({ children, active, onClick, color = C.accent }) => (
  <button onClick={onClick} style={{
    padding:"6px 14px", borderRadius:6, fontSize:12, fontWeight:600,
    cursor:"pointer", transition:"all 0.2s",
    background: active ? `${color}18` : "transparent",
    color:       active ? color : C.textMid,
    border:`1px solid ${active ? color+"55" : C.border}`,
    outline:"none",
  }}>{children}</button>
);

export const Spinner = ({ size = 16, color = C.accent }) => (
  <div style={{
    width:size, height:size, borderRadius:"50%",
    border:`2px solid ${color}33`, borderTopColor:color,
    animation:"spin 0.7s linear infinite", display:"inline-block",
  }}/>
);

export const TopProgress = ({ active, color = C.accent }) => {
  if (!active) return null;
  return (
    <div style={{
      position:"sticky", top:0, zIndex:50,
      height:3, background:`${color}22`, borderRadius:999, overflow:"hidden",
    }}>
      <div style={{
        height:"100%", width:"40%",
        background:`linear-gradient(90deg,transparent,${color},transparent)`,
        animation:"progress 1.1s ease-in-out infinite",
      }}/>
    </div>
  );
};
