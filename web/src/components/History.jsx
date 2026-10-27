import { bandColor } from './HealthDial.jsx'

const NAMES = { healthy: 'Healthy', imbalance: 'Imbalance', misalignment: 'Misalignment', looseness: 'Looseness',
  bearing: 'Bearing wear', airflow: 'Airflow blocked', anomaly: 'Anomaly' }

/** Health timeline for one machine. */
export default function History({ checks }) {
  if (!checks.length) return <p className="muted">No checks yet. Run one and it will show up here.</p>
  const W = 340, H = 150, P = { l: 28, r: 10, t: 10, b: 22 }
  const t0 = checks[0].ts, t1 = checks.at(-1).ts
  const x = (t, i) => P.l + (W - P.l - P.r) * (checks.length === 1 ? 0.5 : t1 === t0 ? i / (checks.length - 1) : (t - t0) / (t1 - t0))
  const y = (h) => P.t + (H - P.t - P.b) * (1 - h / 100)
  const pts = checks.map((c, i) => [x(c.ts, i), y(c.health)])
  const first = checks[0], last = checks.at(-1)
  const days = Math.max(1, Math.round((t1 - t0) / 864e5))
  return (
    <div>
      {checks.length > 1 && (
        <p className="trend">
          Health {last.health < first.health ? 'dropped' : 'went'} from <b>{Math.round(first.health)}</b> to{' '}
          <b style={{ color: bandColor(last.health) }}>{Math.round(last.health)}</b>{' '}
          over {t1 - t0 < 864e5 ? `${checks.length} checks` : `${days} day${days > 1 ? 's' : ''}`}.
        </p>
      )}
      <svg viewBox={`0 0 ${W} ${H}`} className="history">
        <rect x={P.l} y={y(100)} width={W - P.l - P.r} height={y(80) - y(100)} fill="rgba(25,227,196,.06)" />
        <rect x={P.l} y={y(80)} width={W - P.l - P.r} height={y(50) - y(80)} fill="rgba(255,176,32,.05)" />
        <rect x={P.l} y={y(50)} width={W - P.l - P.r} height={y(0) - y(50)} fill="rgba(255,77,95,.05)" />
        {[0, 50, 80, 100].map((v) => (
          <text key={v} x={P.l - 6} y={y(v) + 3} className="axis" textAnchor="end">{v}</text>
        ))}
        <polyline points={pts.map((p) => p.join(',')).join(' ')} className="history-line" />
        {checks.map((c, i) => (
          <circle key={c.id ?? i} cx={pts[i][0]} cy={pts[i][1]} r="4.5" fill={bandColor(c.health)} />
        ))}
      </svg>
      <ul className="checks">
        {[...checks].reverse().slice(0, 8).map((c) => (
          <li key={c.id ?? c.ts}>
            <span className="dot" style={{ background: bandColor(c.health) }} />
            <span>{NAMES[c.label] ?? c.label}</span>
            <span className="mono">{Math.round(c.health)}</span>
            <span className="muted">{new Date(c.ts).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}
