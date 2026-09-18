import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig(({ mode }) => {
  const env = { ...process.env, ...loadEnv(mode, process.cwd(), '') };
  const backendPort = env.BACKEND_PORT || env.PORT || '8000';
  const backendUrl = env.BACKEND_URL || `http://localhost:${backendPort}`;
  const frontendPort = parseInt(env.FRONTEND_PORT || env.VITE_PORT || '5173', 10);

  return {
    plugins: [react()],
    root: __dirname,
    build: {
      outDir: path.resolve(__dirname, 'dist'),
      emptyOutDir: true,
    },
    server: {
      port: frontendPort,
      proxy: {
        '/api': {
          target: backendUrl,
          changeOrigin: true,
        },
      },
    },
  };
});

