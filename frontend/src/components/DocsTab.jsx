import { useEffect, useRef, useState } from "react";
import { API_BASE } from "../config/api";
import { C } from "../config/theme";
import { tL } from "../i18n/index.jsx";
import { Spinner } from "./ui.jsx";

// ══════════════════════════════════════════════════════════════════════════════
// Docs Tab  —  Upload + Knowledge Base Document Management
// ══════════════════════════════════════════════════════════════════════════════
const ALLOWED_EXTS = [".txt", ".md", ".pdf"];

export const DocsTab = ({ lang, kbStats, onRefresh }) => {
  const [pendingFiles, setPendingFiles] = useState([]);   // File objects waiting to upload
  const [fileStatuses, setFileStatuses] = useState({});   // filename -> "ok"|"error"|"uploading"
  const [uploading,    setUploading]    = useState(false);
  const [uploadMsg,    setUploadMsg]    = useState("");    // summary after upload
  const [isDragging,   setIsDragging]   = useState(false);
  const [deletingFile, setDeletingFile] = useState(null); // filename being deleted
  const [previewOpen,  setPreviewOpen]  = useState(false);
  const [previewSrc,   setPreviewSrc]   = useState(null);
  const [previewData,  setPreviewData]  = useState(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewErr,   setPreviewErr]   = useState("");
  const [invLoading,   setInvLoading]   = useState(false);
  const [inventory,    setInventory]    = useState(null);
  const [docsExpanded, setDocsExpanded] = useState(true);
  const inputRef = useRef(null);

  const addFiles = (rawFiles) => {
    const valid = Array.from(rawFiles).filter(f =>
      ALLOWED_EXTS.some(ext => f.name.toLowerCase().endsWith(ext))
    );
    setPendingFiles(prev => {
      const existing = new Set(prev.map(f => f.name));
      return [...prev, ...valid.filter(f => !existing.has(f.name))];
    });
  };

  const removeFile = (name) =>
    setPendingFiles(prev => prev.filter(f => f.name !== name));

  const onDragOver = (e) => { e.preventDefault(); setIsDragging(true); };
  const onDragLeave = ()  => setIsDragging(false);
  const onDrop = (e)      => { e.preventDefault(); setIsDragging(false); addFiles(e.dataTransfer.files); };

  const uploadAll = async () => {
    if (!pendingFiles.length || uploading) return;
    setUploading(true);
    setUploadMsg("");
    // Mark all as uploading
    const st = {};
    pendingFiles.forEach(f => { st[f.name] = "uploading"; });
    setFileStatuses(st);

    try {
      const formData = new FormData();
      pendingFiles.forEach(f => formData.append("files", f));
      const res  = await fetch(`${API_BASE}/upload`, { method: "POST", body: formData });
      const data = await res.json();
      const newSt = {};
      data.files.forEach(r => { newSt[r.filename] = r.status; });
      setFileStatuses(newSt);
      setUploadMsg(tL(lang, "docsUploadDone", data.uploaded, data.errors));
      if (data.uploaded > 0) {
        // Wait 3s for backend rebuild, then refresh stats
        setTimeout(() => { onRefresh(); setPendingFiles([]); setFileStatuses({}); setUploadMsg(""); }, 3500);
      }
    } catch {
      const errSt = {};
      pendingFiles.forEach(f => { errSt[f.name] = "error"; });
      setFileStatuses(errSt);
      setUploadMsg("Upload failed — is the backend running?");
    } finally {
      setUploading(false);
    }
  };

  const deleteDoc = async (filename) => {
    if (!window.confirm(tL(lang, "docsDeleteConfirm", filename))) return;
    setDeletingFile(filename);
    try {
      await fetch(`${API_BASE}/docs/${encodeURIComponent(filename)}`, { method: "DELETE" });
      setTimeout(() => { onRefresh(); setDeletingFile(null); }, 3500);
    } catch {
      setDeletingFile(null);
    }
  };

  const openPreview = async (source) => {
    setPreviewOpen(true);
    setPreviewSrc(source);
    setPreviewData(null);
    setPreviewErr("");
    setPreviewLoading(true);
    try {
      const res = await fetch(`${API_BASE}/docs/preview?source=${encodeURIComponent(source)}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setPreviewData(await res.json());
    } catch (e) {
      setPreviewErr(lang === "en" ? `Preview failed: ${String(e)}` : `预览失败：${String(e)}`);
    } finally {
      setPreviewLoading(false);
    }
  };

  const closePreview = () => {
    setPreviewOpen(false);
    setPreviewSrc(null);
    setPreviewData(null);
    setPreviewErr("");
    setPreviewLoading(false);
  };

  const fetchInventory = async () => {
    setInvLoading(true);
    try {
      const res = await fetch(`${API_BASE}/inventory`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setInventory(await res.json());
    } catch {
      setInventory(null);
    } finally {
      setInvLoading(false);
    }
  };

  useEffect(() => { fetchInventory(); }, [kbStats?.total_chunks]); // refresh after rebuild/upload

  const fmtSize = (bytes) =>
    bytes >= 1024*1024 ? `${(bytes/1024/1024).toFixed(1)} MB`
    : bytes >= 1024    ? `${(bytes/1024).toFixed(1)} KB`
    : `${bytes} B`;

  const statusIcon = (name) => {
    const s = fileStatuses[name];
    if (s === "ok")        return <span style={{color:C.green,  fontSize:13}}>✓</span>;
    if (s === "error")     return <span style={{color:C.red,    fontSize:13}}>✗</span>;
    if (s === "uploading") return <Spinner size={12} color={C.accent}/>;
    return null;
  };

  return (
    <div style={{display:"flex", flexDirection:"column", gap:16}}>

      {/* ── Preview Modal ── */}
      {previewOpen && (
        <div onClick={closePreview} style={{
          position:"fixed", inset:0, background:"rgba(15,23,42,0.55)",
          display:"flex", alignItems:"center", justifyContent:"center",
          padding:20, zIndex: 9999,
        }}>
          <div onClick={(e)=>e.stopPropagation()} style={{
            width:"min(920px, 96vw)", maxHeight:"86vh",
            background:C.bg, borderRadius:12,
            border:`1px solid ${C.borderBright}`,
            boxShadow:"0 20px 60px rgba(0,0,0,0.25)",
            display:"flex", flexDirection:"column", overflow:"hidden",
          }}>
            <div style={{
              padding:"12px 14px", borderBottom:`1px solid ${C.border}`,
              display:"flex", alignItems:"center", justifyContent:"space-between", gap:10,
            }}>
              <div style={{minWidth:0}}>
                <div style={{fontSize:12, fontWeight:800, color:C.text, overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap"}}>
                  {previewData?.source || previewSrc || (lang==="en" ? "Preview" : "预览")}
                </div>
                {previewData && (
                  <div style={{fontSize:11, color:C.textDim, marginTop:2}}>
                    {previewData.file_size_kb ? `${previewData.file_size_kb} KB · ` : ""}{previewData.total_chunks} chunk(s)
                    {previewData.file_mtime ? ` · ${new Date(previewData.file_mtime).toLocaleString()}` : ""}
                    {previewData.truncated ? (lang==="en" ? " · truncated" : " · 已截断") : ""}
                  </div>
                )}
              </div>
              <button onClick={closePreview} style={{
                border:`1px solid ${C.border}`, background:"transparent",
                borderRadius:6, padding:"6px 10px", cursor:"pointer",
                color:C.textMid, fontSize:12, fontWeight:700, fontFamily:"inherit",
              }}>{lang==="en" ? "Close" : "关闭"}</button>
            </div>

            <div style={{padding:14, overflow:"auto"}}>
              {previewLoading && (
                <div style={{display:"flex", alignItems:"center", gap:8, color:C.textMid, fontSize:12}}>
                  <Spinner size={14} color={C.accent}/> {lang==="en" ? "Loading preview…" : "加载预览中…"}
                </div>
              )}
              {previewErr && (
                <div style={{
                  background:`${C.red}10`, border:`1px solid ${C.red}33`, color:C.red,
                  borderRadius:8, padding:"10px 12px", fontSize:12, marginBottom:10,
                }}>{previewErr}</div>
              )}
              {previewData?.preview_text && (
                <pre style={{
                  whiteSpace:"pre-wrap", wordBreak:"break-word",
                  margin:0, fontSize:12.5, lineHeight:1.7, color:C.text,
                  background:C.surface, border:`1px solid ${C.borderBright}`,
                  borderRadius:10, padding:12,
                }}>{previewData.preview_text}</pre>
              )}
              {previewData && !previewData.preview_text && !previewLoading && !previewErr && (
                <div style={{color:C.textDim, fontSize:12}}>
                  {lang==="en" ? "No preview text available." : "暂无可预览内容。"}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ── Upload Zone ── */}
      <div style={{background:C.surface, border:`1px solid ${C.borderBright}`, borderRadius:10, padding:16}}>
        <div style={{fontSize:11, color:C.textMid, fontWeight:700, letterSpacing:"0.08em", marginBottom:12}}>
          {tL(lang,"docsUploadTitle")}
        </div>

        {/* Drop area */}
        <div
          onDragOver={onDragOver} onDragLeave={onDragLeave} onDrop={onDrop}
          onClick={()=>inputRef.current?.click()}
          style={{
            border:`2px dashed ${isDragging ? C.accent : C.borderBright}`,
            borderRadius:8, padding:"28px 16px", textAlign:"center", cursor:"pointer",
            background: isDragging ? `${C.accent}08` : C.bg,
            transition:"all 0.2s", marginBottom:12,
          }}
        >
          <div style={{fontSize:28, marginBottom:8}}>📂</div>
          <div style={{fontSize:13, color:isDragging?C.accent:C.textMid, fontWeight:600}}>
            {isDragging ? tL(lang,"docsDropActive") : tL(lang,"docsDropHint")}
          </div>
          <div style={{fontSize:11, color:C.textDim, marginTop:4}}>
            {tL(lang,"docsAllowedTypes")}
          </div>
          <input
            ref={inputRef} type="file" multiple accept=".txt,.md,.pdf"
            style={{display:"none"}}
            onChange={e => { addFiles(e.target.files); e.target.value=""; }}
          />
        </div>

        {/* Pending file list */}
        {pendingFiles.length > 0 && (
          <div style={{marginBottom:12}}>
            {pendingFiles.map(f => (
              <div key={f.name} style={{
                display:"flex", alignItems:"center", gap:8,
                padding:"5px 8px", borderRadius:5, marginBottom:4,
                background:C.surface, border:`1px solid ${C.border}`, fontSize:12,
              }}>
                <span style={{flex:1, overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap", color:C.text}}>
                  📄 {f.name}
                </span>
                <span style={{color:C.textDim, flexShrink:0, fontSize:11}}>{fmtSize(f.size)}</span>
                {statusIcon(f.name)}
                {!uploading && !fileStatuses[f.name] && (
                  <button onClick={(e)=>{e.stopPropagation();removeFile(f.name);}} style={{
                    background:"transparent", border:"none", color:C.red,
                    cursor:"pointer", fontSize:14, padding:"0 2px", lineHeight:1,
                  }}>×</button>
                )}
              </div>
            ))}
          </div>
        )}

        {/* Upload result message */}
        {uploadMsg && (
          <div style={{
            fontSize:12, color:C.green, background:`${C.green}10`,
            border:`1px solid ${C.green}33`, borderRadius:5, padding:"6px 10px", marginBottom:10,
          }}>
            ✓ {uploadMsg} — <span style={{color:C.textDim}}>{tL(lang,"docsRebuildNotice")}</span>
          </div>
        )}

        {/* Upload button */}
        <button
          onClick={uploadAll}
          disabled={!pendingFiles.length || uploading}
          style={{
            width:"100%", padding:"10px", borderRadius:7, fontSize:13, fontWeight:700,
            cursor: (!pendingFiles.length || uploading) ? "not-allowed" : "pointer",
            background: (!pendingFiles.length || uploading)
              ? `${C.accent}18`
              : `linear-gradient(135deg,${C.accentDim},${C.accent})`,
            color: (!pendingFiles.length || uploading) ? C.accentDim : "#fff",
            border:"none", fontFamily:"inherit",
            display:"flex", alignItems:"center", justifyContent:"center", gap:8,
          }}
        >
          {uploading
            ? <><Spinner size={13} color={C.accent}/>{tL(lang,"docsUploading")}</>
            : tL(lang, "docsUploadBtn", pendingFiles.length || 0)
          }
        </button>
      </div>

      {/* ── Document List ── */}
      <div style={{background:C.surface, border:`1px solid ${C.borderBright}`, borderRadius:10, padding:16}}>
        <button
          onClick={() => setDocsExpanded(v => !v)}
          style={{
            width:"100%",
            display:"flex",
            alignItems:"center",
            justifyContent:"space-between",
            gap:10,
            background:"transparent",
            border:"none",
            padding:0,
            cursor:"pointer",
            fontFamily:"inherit",
          }}
        >
          <div style={{fontSize:11, color:C.textMid, fontWeight:700, letterSpacing:"0.08em"}}>
            {tL(lang,"docsCurrentTitle")}
            {kbStats && (
              <span style={{fontWeight:400, marginLeft:8, color:C.textDim}}>
                ({kbStats.total_chunks} {tL(lang,"kbChunks")} · {kbStats.total_sources} {tL(lang,"kbDocs")})
              </span>
            )}
          </div>
          <span style={{color:C.textDim, fontSize:12, fontWeight:800}}>
            {docsExpanded ? "▾" : "▸"}
          </span>
        </button>

        {docsExpanded && (!kbStats || kbStats.sources?.length === 0) ? (
          <div style={{textAlign:"center", padding:"24px 0", color:C.textDim, fontSize:13}}>
            {tL(lang,"docsEmpty")}
          </div>
        ) : docsExpanded ? (
          <div>
            {kbStats.sources.map((src, i) => {
              const isStale   = kbStats.stale_sources?.includes(src.source);
              const isDeleting = deletingFile === src.source;
              return (
                <div key={i} style={{
                  display:"flex", alignItems:"center", gap:8,
                  padding:"9px 10px", borderRadius:7, marginBottom:6,
                  background: isStale ? `${C.orange}08` : C.bg,
                  border:`1px solid ${isStale ? C.orange+"44" : C.border}`,
                }}>
                  {/* Icon */}
                  <span style={{fontSize:16, flexShrink:0}}>
                    {src.source.endsWith(".pdf") ? "📕"
                     : src.source.endsWith(".md") ? "📝" : "📄"}
                  </span>

                  {/* Info */}
                  <div style={{flex:1, minWidth:0}}>
                    <div style={{
                      fontSize:12, fontWeight:600, color:C.text,
                      overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap",
                    }} title={src.source}>{src.source}</div>
                    <div style={{fontSize:11, color:C.textDim, marginTop:2, display:"flex", gap:8}}>
                      <span>{tL(lang,"docsChunks", src.chunks)}</span>
                      <span>{tL(lang,"docsWords",  src.words)}</span>
                      {src.file_size_kb && <span>{src.file_size_kb} KB</span>}
                      {isStale && (
                        <span style={{color:C.orange}}>⚠ stale</span>
                      )}
                    </div>
                  </div>

                  {/* Delete button */}
                  <div style={{display:"flex", gap:6, alignItems:"center", flexShrink:0}}>
                    <button
                      onClick={() => openPreview(src.source)}
                      style={{
                        padding:"3px 9px", borderRadius:4, fontSize:11,
                        cursor:"pointer", background:"transparent", color:C.accent,
                        border:`1px solid ${C.accent}33`, fontFamily:"inherit",
                      }}
                    >
                      {lang==="en" ? "Preview" : "预览"}
                    </button>
                    <button
                      onClick={() => deleteDoc(src.source)}
                      disabled={isDeleting}
                      style={{
                        padding:"3px 9px", borderRadius:4,
                        fontSize:11, cursor: isDeleting ? "not-allowed" : "pointer",
                        background:"transparent", color: isDeleting ? C.textDim : C.red,
                        border:`1px solid ${isDeleting ? C.border : C.red+"44"}`,
                        fontFamily:"inherit",
                      }}
                    >
                      {isDeleting ? tL(lang,"docsDeleting") : tL(lang,"docsDeleteBtn")}
                    </button>
                  </div>
                </div>
              );
            })}

            {/* Rebuild notice */}
            <div style={{fontSize:11, color:C.textDim, marginTop:8, textAlign:"center"}}>
              ℹ {tL(lang,"docsRebuildNotice")}
            </div>
          </div>
        ) : null}
      </div>

      {/* ── Knowledge Asset Inventory ── */}
      <div style={{background:C.surface, border:`1px solid ${C.borderBright}`, borderRadius:10, padding:16}}>
        <div style={{display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:12}}>
          <div style={{fontSize:11, color:C.textMid, fontWeight:700, letterSpacing:"0.08em"}}>
            {lang==="en" ? "KNOWLEDGE ASSET INVENTORY" : "数据资产盘点（Knowledge Asset Inventory）"}
          </div>
          <button onClick={fetchInventory} style={{
            fontSize:10, color:C.accent, background:"transparent",
            border:`1px solid ${C.accent}33`, borderRadius:4, padding:"2px 7px", cursor:"pointer",
          }}>{invLoading ? (lang==="en" ? "…" : "…") : "↻"}</button>
        </div>

        {!inventory ? (
          <div style={{fontSize:12, color:C.textDim}}>
            {invLoading ? (lang==="en" ? "Loading…" : "加载中…") : (lang==="en" ? "No inventory data." : "暂无盘点数据")}
          </div>
        ) : (
          <div style={{maxHeight:220, overflowY:"auto", paddingRight:4}}>
            {inventory.assets.map((a,i)=>(
              <div key={i} style={{
                display:"grid",
                gridTemplateColumns:"1.4fr 0.6fr 0.6fr 0.7fr 0.7fr",
                gap:8,
                padding:"8px 10px",
                border:`1px solid ${C.border}`,
                borderRadius:8,
                background: a.stale ? `${C.orange}08` : C.bg,
                marginBottom:6,
                alignItems:"center",
                minWidth:0,
              }}>
                <div style={{minWidth:0}}>
                  <div style={{fontSize:12, fontWeight:700, color:C.text, overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap"}} title={a.source}>
                    {a.source}
                  </div>
                  <div style={{fontSize:10, color:C.textDim, marginTop:2}}>
                    {a.last_modified ? new Date(a.last_modified).toLocaleDateString() : ""}{a.stale ? (lang==="en" ? " · stale" : " · ⚠ 过期") : ""}
                  </div>
                </div>
                <div style={{fontSize:11, color:C.textMid}}>
                  <span style={{fontWeight:700, color:C.accent}}>{a.chunks}</span> {lang==="en" ? "chunks" : "块"}
                </div>
                <div style={{fontSize:11, color:C.textMid}}>
                  <span style={{fontWeight:700, color:C.green}}>{a.words}</span> {lang==="en" ? "words" : "词"}
                </div>
                <div style={{fontSize:11, color:C.textMid}}>
                  <span style={{fontWeight:700}}>{a.usage_queries}</span> {lang==="en" ? "uses" : "次命中"}
                </div>
                <div style={{fontSize:11, color:C.textMid, textAlign:"right"}}>
                  {a.feedback_total > 0
                    ? <span style={{fontWeight:800, color:(a.feedback_satisfaction_rate>=0.7?C.green:C.orange)}}>
                        {(a.feedback_satisfaction_rate*100).toFixed(0)}%
                      </span>
                    : <span style={{color:C.textDim}}>{lang==="en" ? "—" : "暂无"}</span>
                  }
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
