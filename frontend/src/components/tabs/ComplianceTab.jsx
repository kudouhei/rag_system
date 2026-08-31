import { useState } from "react";
import { API_BASE } from "../../config/api";
import { C } from "../../config/theme";
import { Spinner, Tag } from "../ui.jsx";

// ══════════════════════════════════════════════════════════════════════════════
// Compliance Check Tab — scenario-based regulatory compliance assessment
// ══════════════════════════════════════════════════════════════════════════════

const STATUS_COLOR = {
  compliant:      C.green,
  non_compliant:  C.red,
  needs_review:   C.orange,
  uncertain:      C.orange,
  not_applicable: C.textDim,
};

const STATUS_LABEL = {
  en: {
    compliant: "Compliant", non_compliant: "Non-Compliant",
    needs_review: "Needs Review", uncertain: "Uncertain", not_applicable: "N/A",
  },
  zh: {
    compliant: "合规", non_compliant: "不合规",
    needs_review: "需人工复核", uncertain: "不确定", not_applicable: "不适用",
  },
};

const StatusBadge = ({ status, lang }) => (
  <Tag
    label={(STATUS_LABEL[lang]||STATUS_LABEL.en)[status] || status}
    color={STATUS_COLOR[status] || C.textMid}
  />
);

export const ComplianceTab = ({ lang }) => {
  const [scenario, setScenario]   = useState(
    lang === "zh"
      ? "我们计划推出一只货币市场基金，提供每日赎回，但未设置摆动定价（swing pricing）机制。"
      : "We plan to launch a money market fund offering daily redemption with no swing pricing or anti-dilution levy mechanism in place."
  );
  const [jurisdiction, setJurisdiction] = useState("");
  const [productType, setProductType]   = useState("");
  const [regulationNumber, setRegulationNumber] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState("");
  const [result, setResult]   = useState(null);

  const isEn = lang !== "zh";
  const T = {
    title:        isEn ? "COMPLIANCE CHECK" : "合规检查",
    subtitle:     isEn
      ? "Describe a business or product scenario. The system retrieves the most relevant regulatory clauses and assesses compliance with citations."
      : "描述一个业务/产品场景，系统将检索最相关的监管条款并给出带引用的合规评估。",
    scenarioLabel: isEn ? "SCENARIO" : "业务场景",
    placeholder:  isEn ? "Describe the product, process, or business scenario to assess…" : "请描述需要评估的产品、流程或业务场景…",
    jurisdiction: isEn ? "Jurisdiction (optional)" : "司法辖区（可选）",
    productType:  isEn ? "Product type (optional)" : "产品类型（可选）",
    regNumber:    isEn ? "Focus regulation # (optional)" : "指定法规编号（可选）",
    runBtn:       isEn ? "⚖ Run Compliance Check" : "⚖ 执行合规检查",
    runningBtn:   isEn ? "Assessing…" : "评估中…",
    overall:      isEn ? "Overall Status" : "总体结论",
    summary:      isEn ? "Summary" : "摘要",
    findings:     isEn ? "FINDINGS" : "评估明细",
    noFindings:   isEn ? "No specific findings returned." : "未返回具体评估条目。",
    idle:         isEn
      ? "Describe a scenario above and run a compliance check to see grounded findings here."
      : "在上方描述场景并执行合规检查，评估结果将显示在这里。",
    errorPfx:     isEn ? "Compliance check failed: " : "合规检查失败：",
    citation:     isEn ? "Citation" : "引用",
    source:       isEn ? "Source" : "来源文档",
  };

  const runCheck = async () => {
    if (!scenario.trim() || loading) return;
    setLoading(true); setError(""); setResult(null);
    try {
      const res = await fetch(`${API_BASE}/compliance_check`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          scenario,
          jurisdiction: jurisdiction || null,
          product_type: productType || null,
          regulation_number: regulationNumber || null,
          top_k: 8,
          language: lang,
        }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setResult(await res.json());
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

  const inputStyle = {
    width: "100%", background: "#fff", border: `1px solid ${C.borderBright}`,
    borderRadius: 6, padding: "8px 10px", fontSize: 12.5,
    color: C.text, fontFamily: "inherit",
  };

  return (
    <div style={{display:"flex", flexDirection:"column", gap:14}}>
      <div style={{background:C.surface, border:`1px solid ${C.borderBright}`, borderRadius:10, padding:16}}>
        <div style={{fontSize:11, color:C.textMid, fontWeight:700, letterSpacing:"0.08em", marginBottom:6}}>
          {T.title}
        </div>
        <div style={{fontSize:12, color:C.textDim, marginBottom:14, lineHeight:1.6}}>
          {T.subtitle}
        </div>

        <div style={{fontSize:10, color:C.textMid, fontWeight:700, letterSpacing:"0.08em", marginBottom:6}}>
          {T.scenarioLabel}
        </div>
        <textarea
          value={scenario} onChange={e=>setScenario(e.target.value)}
          placeholder={T.placeholder} rows={4}
          style={{...inputStyle, resize:"vertical", lineHeight:1.6, marginBottom:12}}
        />

        <div style={{display:"grid", gridTemplateColumns:"1fr 1fr 1fr", gap:8, marginBottom:14}}>
          <input value={jurisdiction} onChange={e=>setJurisdiction(e.target.value)}
            placeholder={T.jurisdiction} style={inputStyle}/>
          <input value={productType} onChange={e=>setProductType(e.target.value)}
            placeholder={T.productType} style={inputStyle}/>
          <input value={regulationNumber} onChange={e=>setRegulationNumber(e.target.value)}
            placeholder={T.regNumber} style={inputStyle}/>
        </div>

        <button onClick={runCheck} disabled={!scenario.trim() || loading} style={{
          padding:"11px 20px", borderRadius:8, fontSize:13, fontWeight:800,
          cursor: (!scenario.trim() || loading) ? "not-allowed" : "pointer",
          background: loading ? `${C.purple}18` : `linear-gradient(135deg,${C.purple}cc,${C.purple})`,
          color: loading ? C.purple : "#fff",
          border:"none", letterSpacing:"0.04em", fontFamily:"inherit", width:"100%",
          display:"flex", alignItems:"center", justifyContent:"center", gap:8,
        }}>
          {loading ? <><Spinner size={16} color={C.purple}/>{T.runningBtn}</> : T.runBtn}
        </button>
      </div>

      {error && (
        <div style={{
          background:`${C.red}10`, border:`1px solid ${C.red}33`, color:C.red,
          borderRadius:8, padding:"10px 12px", fontSize:12.5,
        }}>{T.errorPfx}{error}</div>
      )}

      {!result && !error && !loading && (
        <div style={{
          background:C.surface, border:`1px dashed ${C.border}`, borderRadius:10,
          padding:"28px 16px", textAlign:"center", color:C.textDim, fontSize:13,
        }}>{T.idle}</div>
      )}

      {result && (
        <div style={{background:C.surface, border:`1px solid ${C.borderBright}`, borderRadius:10, padding:16}}>
          <div style={{display:"flex", alignItems:"center", gap:10, marginBottom:10}}>
            <span style={{fontSize:11, color:C.textMid, fontWeight:700, letterSpacing:"0.08em"}}>{T.overall}</span>
            <StatusBadge status={result.overall_status} lang={lang}/>
            <span style={{fontSize:11, color:C.textDim, marginLeft:"auto"}}>{result.elapsed_seconds}s</span>
          </div>
          {result.summary && (
            <div style={{fontSize:13, color:C.text, lineHeight:1.7, marginBottom:14, padding:"10px 12px", background:C.bg, border:`1px solid ${C.border}`, borderRadius:8}}>
              {result.summary}
            </div>
          )}

          <div style={{fontSize:11, color:C.textMid, fontWeight:700, letterSpacing:"0.08em", marginBottom:8}}>
            {T.findings} {result.findings?.length ? `(${result.findings.length})` : ""}
          </div>

          {(!result.findings || result.findings.length === 0) ? (
            <div style={{fontSize:12, color:C.textDim}}>{T.noFindings}</div>
          ) : (
            <div style={{display:"flex", flexDirection:"column", gap:8}}>
              {result.findings.map((f, i) => (
                <div key={i} style={{
                  border:`1px solid ${C.border}`, borderRadius:8, padding:"10px 12px",
                  background: C.bg,
                }}>
                  <div style={{display:"flex", alignItems:"flex-start", justifyContent:"space-between", gap:10, marginBottom:6}}>
                    <div style={{fontSize:12.5, fontWeight:700, color:C.text}}>{f.requirement}</div>
                    <StatusBadge status={f.assessment} lang={lang}/>
                  </div>
                  {f.rationale && (
                    <div style={{fontSize:12, color:C.textMid, lineHeight:1.6, marginBottom:6}}>{f.rationale}</div>
                  )}
                  <div style={{fontSize:11, color:C.textDim, display:"flex", gap:12, flexWrap:"wrap"}}>
                    {f.citation && <span>{T.citation}: <span style={{color:C.accent, fontWeight:600}}>{f.citation}</span></span>}
                    {f.source && <span>{T.source}: <span style={{fontFamily:"monospace"}}>{f.source}</span></span>}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
