// ── Backend API ───────────────────────────────────────────────────────────────
// In dev, Vite proxies /stats, /upload, /ws, … to 127.0.0.1:8000 (see vite.config.js).
// In production builds, call the API on localhost:8000 unless VITE_API_BASE is set.
export const API_BASE =
  (typeof import.meta.env.VITE_API_BASE === "string" && import.meta.env.VITE_API_BASE.trim()) ||
  (import.meta.env.DEV ? "" : "http://localhost:8000");

export function backendWsUrl(path) {
  const custom = typeof import.meta.env.VITE_API_BASE === "string" && import.meta.env.VITE_API_BASE.trim();
  if (custom) {
    const u = custom.replace(/\/$/, "");
    const wsOrigin = u.startsWith("https") ? u.replace(/^https/, "wss") : u.replace(/^http/, "ws");
    return `${wsOrigin}${path}`;
  }
  if (import.meta.env.DEV) {
    const scheme = window.location.protocol === "https:" ? "wss" : "ws";
    return `${scheme}://${window.location.host}${path}`;
  }
  return `ws://localhost:8000${path}`;
}
