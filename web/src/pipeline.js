// recording (16 kHz) -> log-mel -> AE errors + physics measures -> fused verdict
import { logMel, stackFrames } from './dsp/mel.js'
import { analyze, diagnose } from './dsp/physics.js'
import { baselineStats, fuse } from './dsp/health.js'
import { reconstructionErrors } from './ml/ae.js'
import { buildEvidence, templateReport } from './dsp/report.js'

const tick = () => new Promise((r) => setTimeout(r, 0))

function quality(y) {
  let s = 0, clip = 0
  for (const v of y) { s += v * v; if (Math.abs(v) > 0.99) clip++ }
  const rms = Math.sqrt(s / y.length)
  return { rms, rmsDb: 20 * Math.log10(rms + 1e-9), clipped: clip / y.length,
    ok: rms > 0.002 && clip / y.length < 0.001 }
}

async function measure(y, hint) {
  const t0 = performance.now()
  const errors = await reconstructionErrors(stackFrames(logMel(y)))
  const t1 = performance.now()
  await tick()
  const { measures, spectra } = analyze(y, hint)
  const t2 = performance.now()
  return { errors, measures, spectra, timing: { aeMs: t1 - t0, physicsMs: t2 - t1 } }
}

/** Learn what this machine sounds like when healthy. */
export async function makeBaseline(y) {
  const q = quality(y)
  const { errors, measures, spectra, timing } = await measure(y, null)
  return {
    ts: Date.now(), quality: q, timing,
    ae: baselineStats(errors),
    measures,
    env: { f: Array.from(spectra.env.f.subarray(0, 8000)), s: Array.from(spectra.env.s.subarray(0, 8000)) },
  }
}

/** Compare a new recording against the machine's baseline. */
export async function runCheck(y, machine) {
  const t0 = performance.now()
  const b = machine.baseline
  const q = quality(y)
  const { errors, measures, spectra, timing } = await measure(y, { f1: b.measures.f1_hz, blades: b.measures.blades })
  const rules = diagnose(measures, b.measures)
  const fused = fuse(errors, b.ae, rules)
  if (!q.ok) fused.confidence *= 0.5
  const res = { ts: Date.now(), quality: q, measures, spectra, rules, fused }
  res.evidence = buildEvidence(res, machine.name)
  res.report = templateReport(res.evidence)
  res.timing = { ...timing, totalMs: performance.now() - t0 }
  return res
}
