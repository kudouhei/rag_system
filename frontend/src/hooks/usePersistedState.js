import { useCallback, useState } from "react";

// ── localStorage helpers (internal — not used outside this module) ────────────
const LS_KEY = "rag_ui_state";

const loadLS = () => {
  try { return JSON.parse(localStorage.getItem(LS_KEY)) || {}; } catch { return {}; }
};

export const usePersistedState = (key, defaultValue) => {
  const [value, setValue] = useState(() => {
    const saved = loadLS()[key];
    return saved !== undefined ? saved : defaultValue;
  });
  const setPersisted = useCallback((v) => {
    setValue(prev => {
      const next = typeof v === "function" ? v(prev) : v;
      try {
        const current = loadLS();
        localStorage.setItem(LS_KEY, JSON.stringify({ ...current, [key]: next }));
      } catch { /* quota exceeded — silent */ }
      return next;
    });
  }, [key]);
  return [value, setPersisted];
};
