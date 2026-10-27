// Grounded report. The writer (template or LLM) only ever sees the evidence
// object; a verifier rejects any text containing a number or fault name that
// is not in that evidence. The template is the offline default and fallback.

export const ADVICE = {
  imbalance: 'Check the blades for dust build-up, a chip or a bent blade, and make sure the blade hub is seated straight.',
  misalignment: 'Check that the motor shaft and blade hub sit in line and that the guard or housing is not twisted.',
  looseness: 'Tighten the mounting screws, the guard clips and the blade-retaining nut; look for a cracked base.',
  bearing: 'Listen at the motor housing for grinding; clean and lubricate the bearing or plan a bearing/motor replacement.',
  airflow: 'Remove anything blocking the inlet or outlet, clean the grille, and check the filter if there is one.',
  anomaly: 'The sound changed but matches no known fault pattern. Re-record in a quiet room; if it persists, inspect the machine.',
  healthy: 'No action needed. Keep checking regularly to track the trend.',
}

const LABELS = {
  imbalance: 'imbalance', misalignment: 'misalignment', looseness: 'mechanical looseness',
  bearing: 'bearing wear', airflow: 'airflow blockage', anomaly: 'an unclassified anomaly', healthy: 'healthy',
}
const LEXICON = {
  imbalance: ['imbalance', 'unbalance', 'unbalanced'],
  misalignment: ['misalign'],
  looseness: ['loose'],
  bearing: ['bearing'],
  airflow: ['blockage', 'blocked', 'airflow', 'clog'],
}

const fmt = (v, d = 1) => Number(v).toFixed(d)

/** Compact, number-rounded evidence: the ONLY input a report writer gets. */
export function buildEvidence(res, machineName) {
  const top = res.rules[0]
  return {
    machine: machineName,
    health: Math.round(res.fused.health),
    band: res.fused.band,
    label: res.fused.label,
    confidence_pct: Math.round(res.fused.confidence * 100),
    ae_z: Number(fmt(res.fused.z)),
    rpm: Math.round(res.measures.rpm),
    fired: res.rules.filter((r) => r.severity >= 0.5).map((r) => r.id),
    top_rule: {
      id: top.id, severity_pct: Math.round(top.severity * 100),
      evidence: top.evidence.map(([name, now, base, unit]) => ({
        name, now: Number(fmt(now)), baseline: Number(fmt(base)), unit })),
    },
  }
}

export function templateReport(ev) {
  const t = ev.top_rule
  const lead = ev.label === 'healthy'
    ? `${ev.machine} sounds healthy: health ${ev.health}/100, running at about ${ev.rpm} RPM.`
    : ev.label === 'anomaly'
      ? `${ev.machine} sounds different from its healthy baseline (health ${ev.health}/100), but no specific fault pattern matched.`
      : `Health ${ev.health}/100. Most likely cause: ${LABELS[ev.label]} (confidence ${ev.confidence_pct}%).`
  let why = ''
  if (ev.label !== 'healthy' && ev.label !== 'anomaly') {
    const e = t.evidence[0]
    const change = e.unit === 'dB' && e.baseline === 0
      ? `${e.name} is ${e.now > 0 ? '+' : ''}${e.now} dB versus baseline`
      : `${e.name} is ${e.now}${e.unit && e.unit !== 'count' ? ' ' + e.unit : ''} (baseline ${e.baseline})`
    why = ` Evidence: ${change}.`
  } else if (ev.label === 'anomaly') {
    why = ` The autoencoder's error is ${ev.ae_z} standard deviations above this machine's baseline.`
  }
  return { text: `${lead}${why} ${ADVICE[ev.label]}`, source: 'on-device template' }
}

/** Reject text that cites numbers or faults not present in the evidence. */
export function verify(text, ev) {
  const allowed = new Set()
  const walk = (o) => {
    if (typeof o === 'number') { allowed.add(Math.abs(o)); allowed.add(Math.abs(Math.round(o))) }
    else if (o && typeof o === 'object') Object.values(o).forEach(walk)
  }
  walk(ev)
  ;[0, 1, 2, 3, 4, 5, 10, 100, 50, 80].forEach((v) => allowed.add(v))
  const problems = []
  for (const m of text.matchAll(/\d+(?:\.\d+)?/g)) {
    const v = parseFloat(m[0])
    if (![...allowed].some((a) => Math.abs(a - v) < 0.051)) problems.push(`number ${m[0]} not in evidence`)
  }
  const lower = text.toLowerCase()
  const permitted = new Set([...ev.fired, ev.label])
  for (const [id, words] of Object.entries(LEXICON)) {
    if (permitted.has(id)) continue
    if (words.some((w) => lower.includes(w)) && !/\bno\b|\bnot\b/.test(lower)) problems.push(`mentions ${id}, which did not fire`)
  }
  return problems
}

const SYSTEM = `You write short machine-health reports for non-experts.
Use ONLY the numbers in EVIDENCE and copy them exactly. Name only faults listed in "fired" or "label".
Never guess causes, costs or timelines. Three sentences max: the verdict, the key evidence, what to check.`

export async function llmReport(ev, signal) {
  const r = await fetch('/api/report', {
    method: 'POST', signal,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages: [
      { role: 'system', content: SYSTEM },
      { role: 'user', content: `EVIDENCE=${JSON.stringify(ev)}\nSuggested check: ${ADVICE[ev.label]}` },
    ] }),
  })
  if (!r.ok) throw new Error(`report api ${r.status}`)
  const j = await r.json()
  const text = j.choices?.[0]?.message?.content?.trim()
  if (!text) throw new Error('empty report')
  const problems = verify(text, ev)
  if (problems.length) throw new Error('verifier rejected: ' + problems.join('; '))
  return { text, source: 'LLM (verified against evidence)' }
}
