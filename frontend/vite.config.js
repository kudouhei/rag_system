import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

/** Dev-only: forward API + WebSocket to FastAPI so the UI can use same-origin URLs. */
const backend = 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/ws': { target: backend, ws: true, changeOrigin: true },
      '/stats': backend,
      '/upload': backend,
      '/reload': backend,
      '/feedback': backend,
      '/health': backend,
      '/graph': backend,
      '/docs_list': backend,
      '/docs': backend,
    },
  },
})
