import { useEffect, useState } from 'react'

const START = 135, SWEEP = 270, R = 92, C = 110

const polar = (deg, r = R) => {
  const a = (deg * Math.PI) / 180
  return [C + r * Math.cos(a), C + r * Math.sin(a)]
}
const arc = (a0, a1, r = R) => {
  const [x0, y0] = polar(a0, r), [x1, y1] = polar(a1, r)
  return `M ${x0} ${y0} A ${r} ${r} 0 ${a1 - a0 > 180 ? 1 : 0} 1 ${x1} ${y1}`
}
export const bandColor = (h) => (h >= 80 ? 'var(--ok)' : h >= 50 ? 'var(--warn)' : 'var(--bad)')

/** 0-100 health gauge: red < 50 <= yellow < 80 <= green. */
export default function HealthDial({ value, label, sub }) {
  const [shown, setShown] = useState(0)
  useEffect(() => {
    let raf, t0
    const step = (t) => {
      t0 ??= t
      const k = Math.min(1, (t - t0) / 1100)
      setShown(value * (1 - Math.pow(1 - k, 3)))
      if (k < 1) raf = requestAnimationFrame(step)
    }
    raf = requestAnimationFrame(step)
    return () => cancelAnimationFrame(raf)
  }, [value])

  const end = START + (SWEEP * shown) / 100
  const [nx, ny] = polar(end, R)
  return (
    <svg viewBox="0 0 220 200" className="dial" role="img" aria-label={`Health ${Math.round(value)} of 100`}>
      <path d={arc(START, START + SWEEP)} className="dial-track" />
      <path d={arc(START, START + SWEEP * 0.5)} className="dial-zone" stroke="var(--bad)" />
      <path d={arc(START + SWEEP * 0.5, START + SWEEP * 0.8)} className="dial-zone" stroke="var(--warn)" />
      <path d={arc(START + SWEEP * 0.8, START + SWEEP)} className="dial-zone" stroke="var(--ok)" />
      {shown > 0.5 && <path d={arc(START, end)} className="dial-value" stroke={bandColor(shown)} />}
      <circle cx={nx} cy={ny} r="7" fill={bandColor(shown)} className="dial-knob" />
      <text x={C} y={C + 6} className="dial-num" fill={bandColor(shown)}>{Math.round(shown)}</text>
      <text x={C} y={C + 32} className="dial-label">{label}</text>
      {sub && <text x={C} y={C + 52} className="dial-sub">{sub}</text>}
      <text x={polar(START, R + 2)[0] - 2} y={polar(START)[1] + 22} className="dial-tick">0</text>
      <text x={polar(START + SWEEP, R + 2)[0] + 2} y={polar(START)[1] + 22} className="dial-tick">100</text>
    </svg>
  )
}
