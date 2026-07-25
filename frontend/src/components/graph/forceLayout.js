// Minimal Fruchterman–Reingold force-directed layout (no external deps).
// Runs a fixed number of iterations synchronously and returns final {x,y} per node.
export function computeForceLayout(nodes, edges, width, height) {
  const positions = {};
  const n = nodes.length;
  if (n === 0) return positions;

  const cx = width / 2, cy = height / 2;
  const radius = Math.min(width, height) * 0.35;
  nodes.forEach((node, i) => {
    const angle = (2 * Math.PI * i) / n;
    positions[node.name] = {
      x: cx + Math.cos(angle) * radius,
      y: cy + Math.sin(angle) * radius,
    };
  });

  const k = Math.sqrt((width * height) / Math.max(n, 1)) * 0.85;
  const iterations = 200;

  for (let iter = 0; iter < iterations; iter++) {
    const disp = {};
    nodes.forEach((node) => (disp[node.name] = { x: 0, y: 0 }));

    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const a = nodes[i].name, b = nodes[j].name;
        let dx = positions[a].x - positions[b].x;
        let dy = positions[a].y - positions[b].y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 0.01;
        const force = (k * k) / dist;
        dx = (dx / dist) * force;
        dy = (dy / dist) * force;
        disp[a].x += dx; disp[a].y += dy;
        disp[b].x -= dx; disp[b].y -= dy;
      }
    }

    edges.forEach((edge) => {
      const a = edge.source, b = edge.target;
      if (!positions[a] || !positions[b]) return;
      let dx = positions[a].x - positions[b].x;
      let dy = positions[a].y - positions[b].y;
      const dist = Math.sqrt(dx * dx + dy * dy) || 0.01;
      const pull = (dist * dist) / k * (0.5 + Math.min(edge.weight || 1, 5) * 0.15);
      dx = (dx / dist) * pull;
      dy = (dy / dist) * pull;
      disp[a].x -= dx; disp[a].y -= dy;
      disp[b].x += dx; disp[b].y += dy;
    });

    const temp = Math.max(width, height) * (1 - iter / iterations) * 0.05;
    nodes.forEach((node) => {
      const p = positions[node.name];
      const gx = (cx - p.x) * 0.01;
      const gy = (cy - p.y) * 0.01;
      const dx = disp[node.name].x + gx;
      const dy = disp[node.name].y + gy;
      const dlen = Math.sqrt(dx * dx + dy * dy) || 0.01;
      const capped = Math.min(dlen, temp + 1);
      p.x += (dx / dlen) * capped;
      p.y += (dy / dlen) * capped;
      p.x = Math.max(36, Math.min(width - 36, p.x));
      p.y = Math.max(36, Math.min(height - 36, p.y));
    });
  }

  return positions;
}
