// Autoencoder inference with ONNX Runtime Web (WASM, single thread: the model
// is ~266k weights, so threads would cost more than they save).
import * as ort from 'onnxruntime-web/wasm'

// Vite fingerprints and serves the WASM binary itself (self-hosted, offline-safe).
const BASE = import.meta.env.BASE_URL
ort.env.wasm.numThreads = 1

let session = null
export async function loadModel() {
  session ??= await ort.InferenceSession.create(`${BASE}models/stetho_ae.onnx`, {
    executionProviders: ['wasm'], graphOptimizationLevel: 'all',
  })
  return session
}

/** (n x 640) stacked log-mel windows -> Float32Array(n) reconstruction errors */
export async function reconstructionErrors({ data, n }) {
  const s = await loadModel()
  const out = await s.run({ x: new ort.Tensor('float32', data, [n, 640]) })
  return out.err.data
}
