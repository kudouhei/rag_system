import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { C } from "../../config/theme";
import { computeForceLayout } from "./forceLayout";

const TYPE_COLORS = {
  keyword: C.accent,
  概念: C.purple, concept: C.purple,
  技术: C.teal, technology: C.teal,
  方法: C.green, method: C.green,
  系统: C.orange, system: C.orange,
  其他: C.textMid, other: C.textMid,
};

function colorForType(type) {
  if (TYPE_COLORS[type]) return TYPE_COLORS[type];
  const palette = [C.accent, C.purple, C.teal, C.green, C.orange, C.red];
  let hash = 0;
  const s = type || "";
  for (let i = 0; i < s.length; i++) hash = (hash * 31 + s.charCodeAt(i)) | 0;
  return palette[Math.abs(hash) % palette.length];
}

const WIDTH = 760;
const HEIGHT = 460;

export const GraphCanvas = ({ nodes, edges, t }) => {
  const svgRef = useRef(null);
  const [positions, setPositions] = useState({});
  const [dragging, setDragging] = useState(null);
  const [hovered, setHovered] = useState(null);

  useEffect(() => {
    setPositions(computeForceLayout(nodes, edges, WIDTH, HEIGHT));
  }, [nodes, edges]);

  const neighbours = useMemo(() => {
    if (!hovered) return null;
    const set = new Set([hovered]);
    edges.forEach((e) => {
      if (e.source === hovered) set.add(e.target);
      if (e.target === hovered) set.add(e.source);
    });
    return set;
  }, [hovered, edges]);

  const maxFreq = useMemo(() => Math.max(1, ...nodes.map((n) => n.freq || 1)), [nodes]);

  const clientToSvg = useCallback((clientX, clientY) => {
    const rect = svgRef.current.getBoundingClientRect();
    const scaleX = WIDTH / rect.width;
    const scaleY = HEIGHT / rect.height;
    return { x: (clientX - rect.left) * scaleX, y: (clientY - rect.top) * scaleY };
  }, []);

  useEffect(() => {
    if (!dragging) return;
    const onMove = (e) => {
      const { x, y } = clientToSvg(e.clientX, e.clientY);
      setPositions((prev) => ({
        ...prev,
        [dragging]: { x: Math.max(20, Math.min(WIDTH - 20, x)), y: Math.max(20, Math.min(HEIGHT - 20, y)) },
      }));
    };
    const onUp = () => setDragging(null);
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, [dragging, clientToSvg]);

  const typesPresent = useMemo(() => {
    const s = new Set(nodes.map((n) => n.type || "other"));
    return Array.from(s);
  }, [nodes]);

  if (nodes.length === 0) return null;

  return (
    <div>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        style={{ width: "100%", height: HEIGHT, background: C.surface, borderRadius: 8, border: `1px solid ${C.borderBright}`, cursor: dragging ? "grabbing" : "default" }}
      >
        <g opacity={0.65}>
          {edges.map((edge, i) => {
            const a = positions[edge.source], b = positions[edge.target];
            if (!a || !b) return null;
            const dim = neighbours && !(neighbours.has(edge.source) && neighbours.has(edge.target));
            return (
              <line
                key={i}
                x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                stroke={C.borderBright}
                strokeWidth={Math.min(1 + (edge.weight || 1) * 0.6, 5)}
                opacity={dim ? 0.15 : 0.7}
              />
            );
          })}
        </g>

        {nodes.map((node) => {
          const p = positions[node.name];
          if (!p) return null;
          const r = 6 + (Math.sqrt((node.freq || 1) / maxFreq) * 16);
          const dim = neighbours && !neighbours.has(node.name);
          const color = colorForType(node.type);
          return (
            <g
              key={node.name}
              transform={`translate(${p.x},${p.y})`}
              style={{ cursor: "grab" }}
              opacity={dim ? 0.25 : 1}
              onMouseDown={(e) => { e.preventDefault(); setDragging(node.name); }}
              onMouseEnter={() => setHovered(node.name)}
              onMouseLeave={() => setHovered(null)}
            >
              <circle r={r} fill={`${color}26`} stroke={color} strokeWidth={2} />
              <text
                y={r + 13}
                textAnchor="middle"
                fontSize={11}
                fontWeight={hovered === node.name ? 700 : 500}
                fill={C.text}
                style={{ userSelect: "none", pointerEvents: "none" }}
              >
                {node.name.length > 12 ? node.name.slice(0, 12) + "…" : node.name}
              </text>
            </g>
          );
        })}
      </svg>

      <div style={{ display: "flex", flexWrap: "wrap", gap: 12, marginTop: 10, alignItems: "center" }}>
        <span style={{ fontSize: 10, color: C.textMid, fontWeight: 700, letterSpacing: "0.06em" }}>
          {t("graphLegendTitle")}
        </span>
        {typesPresent.map((type) => (
          <span key={type} style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 11, color: C.textMid }}>
            <span style={{ width: 9, height: 9, borderRadius: "50%", background: colorForType(type), display: "inline-block" }} />
            {type}
          </span>
        ))}
      </div>

      {hovered && (
        <div style={{
          marginTop: 8, padding: "8px 12px", background: C.surfaceHover, border: `1px solid ${C.border}`,
          borderRadius: 6, fontSize: 12, color: C.text,
        }}>
          <strong>{hovered}</strong>
          {(() => {
            const n = nodes.find((x) => x.name === hovered);
            if (!n) return null;
            return (
              <span style={{ color: C.textMid }}>
                &nbsp;· {n.type} · freq {n.freq} · {n.chunk_count} chunk(s)
              </span>
            );
          })()}
        </div>
      )}
    </div>
  );
};
