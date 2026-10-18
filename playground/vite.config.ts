import { fileURLToPath, URL } from 'node:url';

import vue from '@vitejs/plugin-vue';
import { defineConfig } from 'vite';

const isolationHeaders = {
  'Cross-Origin-Embedder-Policy': 'require-corp',
  'Cross-Origin-Opener-Policy': 'same-origin',
};

export default defineConfig({
  base: process.env.JET_PLAYGROUND_BASE || '/',
  build: {
    outDir: process.env.JET_PLAYGROUND_OUT_DIR || 'dist',
    emptyOutDir: true,
  },
  plugins: [
    vue({
      features: {
        vapor: true,
      },
    }),
  ],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  server: {
    fs: {
      allow: [fileURLToPath(new URL('..', import.meta.url))],
    },
    headers: isolationHeaders,
  },
  preview: {
    headers: isolationHeaders,
  },
});
