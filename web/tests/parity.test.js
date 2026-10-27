// Python <-> browser parity: the JS DSP must reproduce ml/stetho exactly.
// Regenerate the reference with: cd ml && python export_parity.py
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { decodeWav } from '../src/audio/wav.js'
import { logMel } from '../src/dsp/mel.js'
import { analyze } from '../src/dsp/physics.js'

const ref = JSON.parse(readFileSync(new URL('./fixtures/parity_ref.json', import.meta.url)))
const load = (n) => decodeWav(readFileSync(new URL(`../public/demo/${n}.wav`, import.meta.url))).samples
let hint = null

for (const name of Object.keys(ref)) {
  test(`log-mel parity: ${name}`, () => {
    const lm = logMel(load(name))
    assert.equal(lm.length, 313)
    let worst = 0
    ref[name].logmel_head.forEach((row, t) => row.forEach((v, i) => { worst = Math.max(worst, Math.abs(v - lm[t][i])) }))
    const mean = new Float64Array(128)
    for (const f of lm) f.forEach((v, i) => { mean[i] += v / lm.length })
    ref[name].logmel_mean.forEach((v, i) => { worst = Math.max(worst, Math.abs(v - mean[i])) })
    assert.ok(worst < 0.01, `max |dB diff| ${worst}`)
  })

  test(`physics parity: ${name}`, () => {
    const t0 = performance.now()
    const { measures } = analyze(load(name), name === 'baseline' ? null : hint)
    if (name === 'baseline') hint = { f1: measures.f1_hz, blades: measures.blades }
    for (const [k, v] of Object.entries(ref[name].physics)) {
      const tol = Math.max(1e-3, Math.abs(v) * 1e-3)
      assert.ok(Math.abs(measures[k] - v) <= tol, `${k}: js ${measures[k]} vs py ${v}`)
    }
    console.log(`  ${name}: analyze ${(performance.now() - t0).toFixed(0)} ms`)
  })
}
