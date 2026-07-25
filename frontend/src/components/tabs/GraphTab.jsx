import { useCallback, useEffect, useState } from "react";
import { API_BASE } from "../../config/api";
import { C } from "../../config/theme";
import { Spinner } from "../ui.jsx";
import { GraphCanvas } from "../graph/GraphCanvas.jsx";

export const GraphTab = ({ lang, t, active }) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  const fetchGraph = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/graph`);
      if (res.ok) setData(await res.json());
    } catch { /* backend not ready */ }
    finally { setLoading(false); }
  }, []);

  useEffect(() => { if (active) fetchGraph(); }, [active, fetchGraph]);

  if (loading && !data) {
    return (
      <div style={{ textAlign: "center", padding: "60px 0", color: C.textDim, fontSize: 13 }}>
        <Spinner /> &nbsp;{t("graphLoading")}
      </div>
    );
  }

  if (!data || !data.enabled) {
    return (
      <div style={{ textAlign: "center", padding: "60px 0", color: C.textDim, fontSize: 13 }}>
        {t("graphDisabled")}
      </div>
    );
  }

  if (!data.top_nodes || data.top_nodes.length === 0) {
    return (
      <div style={{ textAlign: "center", padding: "60px 0", color: C.textDim, fontSize: 13 }}>
        {t("graphEmpty")}
      </div>
    );
  }

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
        <div style={{ fontSize: 10, color: C.textMid, fontWeight: 700, letterSpacing: "0.1em" }}>
          {t("graphTitle")}
          <span style={{ marginLeft: 10, fontWeight: 500, letterSpacing: 0 }}>
            {t("graphNodeCount", data.node_count)} · {t("graphEdgeCount", data.edge_count)}
          </span>
        </div>
        <button
          onClick={fetchGraph}
          style={{
            padding: "4px 10px", borderRadius: 6, fontSize: 11, fontWeight: 600,
            background: "transparent", color: C.accent, border: `1px solid ${C.accent}55`, cursor: "pointer",
          }}
        >
          {loading ? "…" : `↻ ${t("graphRefreshBtn")}`}
        </button>
      </div>

      <GraphCanvas nodes={data.top_nodes} edges={data.top_edges} t={t} />

      <div style={{ marginTop: 10, fontSize: 11, color: C.textDim }}>{t("graphHint")}</div>
    </div>
  );
};
