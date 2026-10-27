// Log-mel front end matching librosa.feature.melspectrogram defaults used in
// ml/stetho/features.py: sr 16 kHz, n_fft 1024, hop 512, 128 Slaney mels,
// periodic Hann, center=True with zero padding, then 10*log10(mel + eps).
import { hann, powerSpectrum } from './fft.js'

export const SR = 16000, N_FFT = 1024, HOP = 512, N_MELS = 128, CONTEXT = 5
const EPS = 2.220446049250313e-16

const F_SP = 200 / 3, MIN_LOG_HZ = 1000, MIN_LOG_MEL = MIN_LOG_HZ / F_SP, LOGSTEP = Math.log(6.4) / 27
const hzToMel = (f) => (f >= MIN_LOG_HZ ? MIN_LOG_MEL + Math.log(f / MIN_LOG_HZ) / LOGSTEP : f / F_SP)
const melToHz = (m) => (m >= MIN_LOG_MEL ? MIN_LOG_HZ * Math.exp(LOGSTEP * (m - MIN_LOG_MEL)) : F_SP * m)

let filterCache = null
export function melFilterbank() {
  if (filterCache) return filterCache
  const nb = N_FFT / 2 + 1
  const lo = hzToMel(0), hi = hzToMel(SR / 2)
  const mf = Array.from({ length: N_MELS + 2 }, (_, i) => melToHz(lo + ((hi - lo) * i) / (N_MELS + 1)))
  const rows = []
  for (let i = 0; i < N_MELS; i++) {
    const w = new Float64Array(nb)
    const enorm = 2 / (mf[i + 2] - mf[i])
    let first = -1, last = -1
    for (let k = 0; k < nb; k++) {
      const f = (k * SR) / N_FFT
      const lower = (f - mf[i]) / (mf[i + 1] - mf[i])
      const upper = (mf[i + 2] - f) / (mf[i + 2] - mf[i + 1])
      const v = Math.max(0, Math.min(lower, upper)) * enorm
      w[k] = v
      if (v > 0) { if (first < 0) first = k; last = k }
    }
    rows.push({ w, first: Math.max(first, 0), last })
  }
  filterCache = rows
  return rows
}

/** y: Float32Array at 16 kHz -> Array<Float32Array(128)> frames (dB) */
export function logMel(y) {
  const pad = N_FFT / 2
  const x = new Float64Array(y.length + 2 * pad)
  x.set(y, pad)
  const win = hann(N_FFT)
  const nF = 1 + Math.floor(y.length / HOP)
  const P = new Float64Array(N_FFT / 2 + 1)
  const fb = melFilterbank()
  const out = []
  for (let t = 0; t < nF; t++) {
    powerSpectrum(x, t * HOP, win, N_FFT, P)
    const m = new Float32Array(N_MELS)
    for (let i = 0; i < N_MELS; i++) {
      const { w, first, last } = fb[i]
      let s = 0
      for (let k = first; k <= last; k++) s += w[k] * P[k]
      m[i] = 10 * Math.log10(s + EPS)
    }
    out.push(m)
  }
  return out
}

/** sliding 5-frame context -> Float32Array(N * 640) */
export function stackFrames(frames, context = CONTEXT) {
  const n = frames.length - context + 1
  const out = new Float32Array(n * N_MELS * context)
  for (let i = 0; i < n; i++)
    for (let c = 0; c < context; c++) out.set(frames[i + c], (i * context + c) * N_MELS)
  return { data: out, n }
}
