import { C } from "../../config/theme";

export const TabBar = ({ tabs, activeTab, setActiveTab, conversationHistory, t }) => (
  <div style={{display:"flex", gap:4, borderBottom:`1px solid ${C.border}`, paddingBottom:8}}>
    {tabs.map(tab=>(
      <button key={tab.key} onClick={()=>setActiveTab(tab.key)} style={{
        padding:"6px 16px", borderRadius:6, fontSize:12, fontWeight:600,
        cursor:"pointer", fontFamily:"inherit",
        background: activeTab===tab.key?`${C.accent}12`:"transparent",
        color:       activeTab===tab.key?C.accent:C.textMid,
        border:`1px solid ${activeTab===tab.key?C.accent+"55":"transparent"}`,
      }}>
        {tab.icon} {t(tab.lk)}
        {tab.key==="conversation" && conversationHistory.length>0 && (
          <span style={{
            marginLeft:6, background:C.orange, color:"#fff",
            borderRadius:"50%", width:16, height:16, fontSize:9,
            display:"inline-flex", alignItems:"center", justifyContent:"center", fontWeight:800,
          }}>{Math.floor(conversationHistory.length/2)}</span>
        )}
      </button>
    ))}
  </div>
);
