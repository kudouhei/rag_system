import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";


export default defineConfig(({ mode }) => {
  const env = loadEnv(
    mode,
    process.cwd(),
    "",
  );

  const backend =
    env.VITE_DEV_API_TARGET
    || "http://127.0.0.1:8000";

  return {
    plugins: [
      react(),
    ],

    server: {
      port: 3000,

      proxy: {
        "/api": {
          target: backend,
          changeOrigin: true,
        },

        "/ws": {
          target: backend,
          ws: true,
          changeOrigin: true,
        },

        "/stats": backend,
        "/inventory": backend,
        "/upload": backend,
        "/reload": backend,
        "/feedback": backend,
        "/health": backend,
        "/graph": backend,
        "/docs_list": backend,
        "/docs": backend,
        "/compliance_check": backend,
      },
    },
  };
});