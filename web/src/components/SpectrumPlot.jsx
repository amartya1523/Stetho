import { useEffect, useRef } from 'react'

/** Envelope (modulation) spectrum with shaft-order markers, current vs baseline. */
export default function SpectrumPlot({ env, baseEnv, f1, height = 150 }) {
  const ref = useRef(null)
  useEffect(() => {
    const cv = ref.current
    const dpr = Math.min(window.devicePixelRatio, 2)
    const W = cv.clientWidth, H = height
    cv.width = W * dpr; cv.height = H * dpr
    const g = cv.getContext('2d')
    g.scale(dpr, dpr)
    const fMax = Math.min(11.5 * f1, 240)
    const pad = { l: 8, r: 8, t: 14, b: 22 }
    const x = (f) => pad.l + ((W - pad.l - pad.r) * f) / fMax
    const toDb = (v) => 20 * Math.log10(v + 1e-5)
    const lo = -80, hi = -10
    const y = (v) => pad.t + (H - pad.t - pad.b) * (1 - (Math.min(hi, Math.max(lo, toDb(v))) - lo) / (hi - lo))

    g.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace'
    for (let h = 1; h <= 10; h++) {
      const xx = x(h * f1)
      g.strokeStyle = h <= 3 ? 'rgba(255,176,32,.35)' : 'rgba(255,255,255,.08)'
      g.setLineDash([3, 4]); g.beginPath(); g.moveTo(xx, pad.t); g.lineTo(xx, H - pad.b); g.stroke()
      g.setLineDash([])
      if (h <= 3 || h % 2 === 0) { g.fillStyle = h <= 3 ? '#ffb020' : '#5c6b7a'; g.fillText(`${h}×`, xx - 6, H - 7) }
    }
    const line = (spec, color, width, fill) => {
      // max per pixel column: peaks survive, the 0.03 Hz-resolution noise does not
      const cols = Math.floor(W - pad.l - pad.r), mx = new Float64Array(cols)
      for (let i = 0; i < spec.f.length && spec.f[i] <= fMax; i++) {
        const c = Math.min(cols - 1, Math.floor(((x(spec.f[i]) - pad.l) / (W - pad.l - pad.r)) * cols))
        if (spec.s[i] > mx[c]) mx[c] = spec.s[i]
      }
      g.beginPath()
      for (let c = 0; c < cols; c++) {
        const yy = y(mx[c])
        c ? g.lineTo(pad.l + c, yy) : g.moveTo(pad.l, yy)
      }
      g.strokeStyle = color; g.lineWidth = width; g.stroke()
      if (fill) {
        g.lineTo(x(fMax), H - pad.b); g.lineTo(x(0), H - pad.b); g.closePath()
        const gr = g.createLinearGradient(0, pad.t, 0, H)
        gr.addColorStop(0, 'rgba(25,227,196,.28)'); gr.addColorStop(1, 'rgba(25,227,196,0)')
        g.fillStyle = gr; g.fill()
      }
    }
    if (baseEnv) line(baseEnv, 'rgba(160,175,190,.45)', 1, false)
    line(env, '#19e3c4', 1.4, true)
  }, [env, baseEnv, f1, height])
  return <canvas ref={ref} className="spectrum" style={{ height }} aria-label="Envelope spectrum" />
}
