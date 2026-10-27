// Port of ml/stetho/health.py: AE evidence + physics rules -> 0-100 health.
export const Z0 = 6.0, K = 1.6, WIN = 31, FAULT_SEV = 0.5, UNCLASSIFIED_H = 65

function windowMeans(e, win = WIN) {
  const n = Math.floor(e.length / win), out = []
  for (let i = 0; i < n; i++) {
    let s = 0
    for (let j = i * win; j < (i + 1) * win; j++) s += e[j]
    out.push(s / win)
  }
  return out
}

function median(a) {
  const s = [...a].sort((x, y) => x - y), h = s.length >> 1
  return s.length % 2 ? s[h] : (s[h - 1] + s[h]) / 2
}

export function baselineStats(errors) {
  const w = windowMeans(errors)
  const m = median(w)
  const mad = median(w.map((v) => Math.abs(v - m)))
  return { m, sd: Math.max(1.4826 * mad, 0.08 * m, 1e-6) }
}

export function aeHealth(errors, base) {
  let s = 0
  for (const v of errors) s += v
  s /= errors.length
  const z = (s - base.m) / base.sd
  return { h: 100 / (1 + Math.exp((z - Z0) / K)), z, score: s }
}

export function fuse(errors, baseAe, rules) {
  const { h: hAe, z, score } = aeHealth(errors, baseAe)
  const top = rules[0], sec = rules[1]?.severity ?? 0
  const h = Math.min(hAe, 100 - 70 * top.severity)
  let label, conf
  if (top.severity >= FAULT_SEV) {
    label = top.id
    conf = top.severity * (1 - (0.5 * sec) / (top.severity + 1e-9))
    conf = hAe < 80 ? 0.5 * conf + 0.5 * (1 - hAe / 100) : 0.8 * conf
  } else if (h < UNCLASSIFIED_H) {
    label = 'anomaly'
    conf = 1 - h / 100
  } else {
    label = 'healthy'
    conf = (h / 100) * (1 - top.severity)
  }
  const band = h >= 80 ? 'green' : h >= 50 ? 'yellow' : 'red'
  return { health: Math.round(h * 10) / 10, healthAe: hAe, z, aeScore: score, label, band,
    confidence: Math.min(1, Math.max(0, conf)) }
}
