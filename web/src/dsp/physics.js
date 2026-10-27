// Port of ml/stetho/physics.py. Keep the two in lockstep: tests/parity.test.js
// compares both on the same WAV files.
import { fft, hann, nextPow2 } from './fft.js'

const FS = 16000
const WELCH_N = 32768
const ENV_DEC = 32
const ENV_NFFT = 16384
const MID_BAND = [300, 3000]
const HF_BAND = [2500, 7500]
const F1_RANGE = [8, 50]
const BLADES = [3, 4, 5, 7, 9]

// numpy.searchsorted(side='left') on an ascending array
function searchsorted(a, v) {
  let lo = 0, hi = a.length
  while (lo < hi) { const mid = (lo + hi) >> 1; if (a[mid] < v) lo = mid + 1; else hi = mid }
  return lo
}

function median(a, i0, i1) {
  const s = Array.prototype.slice.call(a, i0, i1).sort((x, y) => x - y)
  const n = s.length, h = n >> 1
  return n % 2 ? s[h] : (s[h - 1] + s[h]) / 2
}

function maxRange(a, i0, i1) {
  let m = -Infinity
  for (let i = i0; i < i1; i++) if (a[i] > m) m = a[i]
  return m
}

const pyRound = (x) => { const r = Math.round(x); return Math.abs(x % 1) === 0.5 && r % 2 ? r - 1 : r }

function rfftfreq(n, d) {
  const val = 1 / (n * d), f = new Float64Array(n / 2 + 1)
  for (let i = 0; i < f.length; i++) f[i] = i * val
  return f
}

export function welchDb(x, n = WELCH_N) {
  const w = hann(n), hop = n / 2
  const acc = new Float64Array(n / 2 + 1)
  let cnt = 0, w2 = 0
  for (let i = 0; i < n; i++) w2 += w[i] * w[i]
  const re = new Float64Array(n), im = new Float64Array(n)
  for (let s = 0; s + n <= x.length; s += hop) {
    for (let i = 0; i < n; i++) { re[i] = x[s + i] * w[i]; im[i] = 0 }
    fft(re, im)
    for (let k = 0; k <= n / 2; k++) acc[k] += re[k] * re[k] + im[k] * im[k]
    cnt++
  }
  const db = new Float64Array(n / 2 + 1)
  for (let k = 0; k < db.length; k++) db[k] = 10 * Math.log10(acc[k] / Math.max(cnt, 1) / w2 + 1e-12)
  return { f: rfftfreq(n, 1 / FS), db }
}

// analytic signal of x restricted to [lo, hi] Hz (FFT brick-wall)
export function bandAnalytic(x, lo, hi) {
  const n = nextPow2(x.length)
  const re = new Float64Array(n), im = new Float64Array(n)
  re.set(x)
  fft(re, im)
  const df = FS / n
  for (let k = 0; k < n; k++) {
    const f = k < n / 2 ? k * df : (k - n) * df   // numpy.fft.fftfreq
    const h = f >= lo && f <= hi ? 2 : 0
    re[k] *= h; im[k] *= h
  }
  fft(re, im, true)
  return { re: re.subarray(0, x.length), im: im.subarray(0, x.length) }
}

export function envelopeSpectrum(z) {
  const m = Math.floor(z.re.length / ENV_DEC)
  const env = new Float64Array(m)
  for (let i = 0; i < m; i++) {
    let s = 0
    for (let j = i * ENV_DEC; j < (i + 1) * ENV_DEC; j++) s += Math.hypot(z.re[j], z.im[j])
    env[i] = s / ENV_DEC
  }
  let dc = 0
  for (let i = 0; i < m; i++) dc += env[i]
  dc = dc / m + 1e-12
  const w = hann(m)
  let wsum = 0
  for (let i = 0; i < m; i++) wsum += w[i]
  const re = new Float64Array(ENV_NFFT), im = new Float64Array(ENV_NFFT)
  for (let i = 0; i < m; i++) re[i] = (env[i] - dc) * w[i]
  fft(re, im)
  const spec = new Float64Array(ENV_NFFT / 2 + 1)
  for (let k = 0; k < spec.length; k++) spec[k] = (Math.hypot(re[k], im[k]) * 2) / wsum / dc
  return { f: rfftfreq(ENV_NFFT, ENV_DEC / FS), s: spec }
}

function peak(f, s, f0, tol = 0.015) {
  const half = Math.max(f0 * tol, 0.25)
  const i0 = searchsorted(f, f0 - half), i1 = searchsorted(f, f0 + half)
  const wdt = Math.max(4.0, 0.15 * f0)
  const j0 = searchsorted(f, f0 - wdt), j1 = searchsorted(f, f0 + wdt)
  if (i1 <= i0 || j1 <= j0) return [0, 0]
  const pk = maxRange(s, i0, i1)
  const floor = median(s, j0, j1) + 1e-12
  return [pk, 20 * Math.log10(pk / floor + 1e-12)]
}

function peakDb(f, sdb, f0, tol = 0.015) {
  const half = Math.max(f0 * tol, 0.5)
  const i0 = searchsorted(f, f0 - half), i1 = searchsorted(f, f0 + half)
  const wdt = Math.max(8.0, 0.15 * f0)
  const j0 = searchsorted(f, f0 - wdt), j1 = searchsorted(f, f0 + wdt)
  const pk = maxRange(sdb, i0, i1)
  return [pk, pk - median(sdb, j0, j1)]
}

export function estimateF1(env, psd, hint = null, blades = null) {
  const [lo, hi] = hint ? [hint * 0.92, hint * 1.08] : F1_RANGE
  const n = Math.ceil((hi - lo) / 0.05)
  let best = [-1e9, lo, BLADES[0]]
  const bl = blades ? [blades] : BLADES
  for (let c = 0; c < n; c++) {
    const f1 = lo + c * 0.05
    const envPnr = [1, 2, 3].map((h) => Math.max(peak(env.f, env.s, h * f1)[1], 0))
    for (const B of bl) {
      const bpf = B * f1
      if (bpf * 4 > FS / 2 || bpf < 30) continue
      let ps = 0
      for (const k of [1, 2, 3, 4]) ps += Math.max(peakDb(psd.f, psd.db, k * bpf)[1], 0)
      let sb = 0
      for (const d of [-1, 1]) sb += Math.max(peakDb(psd.f, psd.db, bpf + d * f1, 0.01)[1], 0)
      sb *= 0.5
      let e = 0
      for (let h = 1; h <= 3; h++) if (h % B !== 0) e += envPnr[h - 1]
      e *= 2.0
      let sub = 0
      for (const k of [1, 3]) sub += Math.max(peakDb(psd.f, psd.db, (k * bpf) / 2)[1], 0)
      const sc = ps + sb + e - 1.5 * sub
      if (sc > best[0]) best = [sc, f1, B]
    }
  }
  return { f1: best[1], blades: best[2], score: best[0] }
}

function kurtosis(v) {
  let mu = 0
  for (let i = 0; i < v.length; i++) mu += v[i]
  mu /= v.length
  let m2 = 0, m4 = 0
  for (let i = 0; i < v.length; i++) { const d = v[i] - mu, d2 = d * d; m2 += d2; m4 += d2 * d2 }
  m2 /= v.length; m4 /= v.length
  return m4 / (m2 * m2 + 1e-20)
}

/** Measure physical quantities of one 16 kHz recording. Also returns spectra for plotting. */
export function analyze(y32, hint = null) {
  const N = y32.length
  let mean = 0
  for (let i = 0; i < N; i++) mean += y32[i]
  mean /= N
  const y = new Float64Array(N)
  let pow = 0
  for (let i = 0; i < N; i++) { y[i] = y32[i] - mean; pow += y[i] * y[i] }
  const totalDb = 10 * Math.log10(pow / N + 1e-12)

  const psd = welchDb(y)
  const zm = bandAnalytic(y, ...MID_BAND)
  const env = envelopeSpectrum(zm)
  const zh = bandAnalytic(y, ...HF_BAND)
  const envH = envelopeSpectrum(zh)

  const { f1, blades: B } = estimateF1(env, psd, hint?.f1 ?? null, hint?.blades ?? null)

  const ORD = [0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10]
  const orders = {}
  for (const h of ORD) orders[h] = peak(env.f, env.s, h * f1)
  let nHarm = 0
  for (let h = 1; h <= 10; h++) if (orders[h][1] > 6.0 && h % B !== 0) nHarm++
  const half = Math.max(orders[0.5][1], orders[1.5][1], orders[2.5][1])
  let hm = 0
  for (let h = 4; h <= 10; h++) if (h % B !== 0) hm += orders[h][0] ** 2
  const harmMod = Math.sqrt(hm)
  const halfMod = Math.sqrt(orders[0.5][0] ** 2 + orders[1.5][0] ** 2 + orders[2.5][0] ** 2)

  let nsBest = 0, nsFreq = 0
  const loI = searchsorted(envH.f, 1.8 * f1), hiI = searchsorted(envH.f, 12 * f1)
  const sh = envH.s
  for (let i = loI + 1; i < hiI - 1; i++) {
    if (!(sh[i] >= sh[i - 1] && sh[i] >= sh[i + 1])) continue
    const order = envH.f[i] / f1
    if (Math.abs(order - pyRound(order)) < 0.06 || Math.abs(order - pyRound(order * 2) / 2) < 0.03) continue
    const pnr = peak(envH.f, sh, envH.f[i], 0.004)[1]
    if (pnr > nsBest) { nsBest = pnr; nsFreq = envH.f[i] }
  }

  let hp = 0
  const hfBand = new Float64Array(N)
  for (let i = 0; i < N; i++) { hfBand[i] = zh.re[i] / 2; hp += hfBand[i] * hfBand[i] }
  const hfDb = 10 * Math.log10(hp / N + 1e-12) - totalDb

  const bpf = B * f1
  const bpfPnr = peakDb(psd.f, psd.db, bpf)[1]
  const floorDb = median(psd.db, searchsorted(psd.f, 200), searchsorted(psd.f, 2000)) - totalDb

  return {
    measures: {
      f1_hz: f1, rpm: f1 * 60, blades: B, bpf_hz: bpf,
      mod_1x: orders[1][0], mod_2x: orders[2][0], mod_3x: orders[3][0],
      pnr_1x: orders[1][1], pnr_2x: orders[2][1], pnr_3x: orders[3][1],
      n_harmonics: nHarm, half_order_pnr: half, harm_mod: harmMod, half_mod: halfMod,
      hf_db: hfDb, hf_kurtosis: kurtosis(hfBand),
      nonsync_pnr: nsBest, nonsync_order: nsFreq ? nsFreq / f1 : 0,
      bpf_tonality_db: bpfPnr, floor_db: floorDb,
    },
    spectra: { psd, env },
  }
}

const db = (a, b) => 20 * Math.log10((a + 1e-6) / (b + 1e-6))
const ramp = (v, lo, hi) => Math.min(1, Math.max(0, (v - lo) / (hi - lo)))

export const FAULTS = {
  imbalance: 'Imbalance',
  misalignment: 'Misalignment',
  looseness: 'Mechanical looseness',
  bearing: 'Bearing wear',
  airflow: 'Airflow blockage',
}

/** Compare measurement m against healthy baseline b -> rules sorted by severity. */
export function diagnose(m, b) {
  const d1 = db(m.mod_1x, b.mod_1x), d2 = db(m.mod_2x, b.mod_2x), d3 = db(m.mod_3x, b.mod_3x)
  const r21 = m.mod_2x / (m.mod_1x + 1e-9), r21b = b.mod_2x / (b.mod_1x + 1e-9)
  const dhf = m.hf_db - b.hf_db
  const dk = m.hf_kurtosis - b.hf_kurtosis
  const dfloor = m.floor_db - b.floor_db
  const dton = m.bpf_tonality_db - b.bpf_tonality_db
  const drpm = (100 * (m.rpm - b.rpm)) / b.rpm
  const dharm = db(m.harm_mod, b.harm_mod), dhalf = db(m.half_mod, b.half_mod)

  const out = []
  const loose = ramp(dharm, 2, 7) * (0.5 + 0.5 * ramp(dhalf, 2, 7))
  out.push({ id: 'looseness', severity: loose, evidence: [
    ['4x-10x harmonic modulation change', dharm, 0, 'dB'],
    ['half-order (0.5x, 1.5x, 2.5x) change', dhalf, 0, 'dB'],
    ['shaft harmonics present', m.n_harmonics, b.n_harmonics, 'count']] })

  const imb = ramp(d1, 4, 12) * (1 - 0.6 * ramp(r21, 0.8, 1.5)) * (1 - Math.max(ramp(dharm, 3, 8), ramp(dhalf, 3, 8)))
  out.push({ id: 'imbalance', severity: imb, evidence: [
    ['1x shaft modulation change', d1, 0, 'dB'],
    ['2x/1x ratio', r21, r21b, 'ratio']] })

  const mis = ramp(d2, 4, 12) * ramp(r21, 0.5, 1.0) * (1 - ramp(dharm, 3, 8))
  out.push({ id: 'misalignment', severity: mis, evidence: [
    ['2x shaft modulation change', d2, 0, 'dB'],
    ['3x shaft modulation change', d3, 0, 'dB'],
    ['2x/1x ratio', r21, r21b, 'ratio']] })

  const brg = Math.max(ramp(dhf, 3, 10), ramp(dk, 1.0, 6.0), ramp(m.nonsync_pnr - b.nonsync_pnr, 4, 12))
  out.push({ id: 'bearing', severity: brg, evidence: [
    ['2.5-7.5 kHz energy change', dhf, 0, 'dB'],
    ['high-band kurtosis', m.hf_kurtosis, b.hf_kurtosis, ''],
    ['non-synchronous tone prominence', m.nonsync_pnr, b.nonsync_pnr, 'dB']] })

  const air = ramp(-dton, 4, 10) * (0.6 + 0.4 * Math.max(ramp(dfloor, 1, 4), ramp(Math.abs(drpm), 2.5, 5)))
  out.push({ id: 'airflow', severity: air, evidence: [
    ['broadband noise floor change', dfloor, 0, 'dB'],
    ['blade-pass tonality change', dton, 0, 'dB'],
    ['speed change', drpm, 0, '%']] })

  for (const r of out) r.name = FAULTS[r.id]
  return out.sort((a, b2) => b2.severity - a.severity)
}
