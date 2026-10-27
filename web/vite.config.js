import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { VitePWA } from 'vite-plugin-pwa'

export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      registerType: 'autoUpdate',
      includeAssets: ['favicon.svg', 'apple-touch-icon.png'],
      manifest: {
        name: 'Stetho', short_name: 'Stetho',
        description: 'A stethoscope for machines: on-device acoustic fault detection.',
        start_url: '/', scope: '/', display: 'standalone',
        theme_color: '#070b10', background_color: '#070b10',
        icons: [
          { src: 'pwa-192.png', sizes: '192x192', type: 'image/png' },
          { src: 'pwa-512.png', sizes: '512x512', type: 'image/png' },
          { src: 'pwa-maskable-512.png', sizes: '512x512', type: 'image/png', purpose: 'maskable' },
        ],
      },
      workbox: {
        // the model (1 MB) and the WASM runtime (~12 MB) must be precached to work offline
        globPatterns: ['**/*.{js,mjs,css,html,svg,png,wasm,onnx,wav}'],
        maximumFileSizeToCacheInBytes: 30 * 1024 * 1024,
        navigateFallbackDenylist: [/^\/api\//],
        cleanupOutdatedCaches: true,
      },
    }),
  ],
  optimizeDeps: { exclude: ['onnxruntime-web'] },
})
