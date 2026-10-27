// The verifier is what makes the LLM report "grounded": it must accept the
// template's own output and reject invented numbers or faults.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { templateReport, verify } from '../src/dsp/report.js'

const ev = {
  machine: 'Table fan', health: 30, band: 'red', label: 'imbalance', confidence_pct: 80, ae_z: 2.3, rpm: 1328,
  fired: ['imbalance'],
  top_rule: { id: 'imbalance', severity_pct: 100, evidence: [
    { name: '1x shaft modulation change', now: 16.3, baseline: 0, unit: 'dB' },
    { name: '2x/1x ratio', now: 0.1, baseline: 0.5, unit: 'ratio' }] },
}

test('template output passes its own verifier', () => {
  for (const label of ['imbalance', 'healthy', 'anomaly']) {
    const e = { ...ev, label, fired: label === 'imbalance' ? ['imbalance'] : [] }
    assert.deepEqual(verify(templateReport(e).text, e), [])
  }
})

test('rejects a number that is not in the evidence', () => {
  const p = verify('Health 30/100. Vibration at 1x rose by 9.7 dB, indicating imbalance.', ev)
  assert.ok(p.some((s) => s.includes('9.7')))
})

test('rejects a fault that did not fire', () => {
  const p = verify('Health 30/100: imbalance, and the bearing is also worn.', ev)
  assert.ok(p.some((s) => s.includes('bearing')))
})

test('accepts a faithful paraphrase', () => {
  assert.deepEqual(verify('Health is 30/100. The 1x modulation rose 16.3 dB, which points to imbalance. Clean the blades.', ev), [])
})
