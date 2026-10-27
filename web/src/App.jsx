import { useCallback, useEffect, useRef, useState } from 'react'
import Waterfall from './components/Waterfall.jsx'
import HealthDial, { bandColor } from './components/HealthDial.jsx'
import SpectrumPlot from './components/SpectrumPlot.jsx'
import History from './components/History.jsx'
import { listenFile, listenMic } from './audio/capture.js'
import { makeBaseline, runCheck } from './pipeline.js'
import { addCheck, deleteMachine, listMachines, machineHistory, saveMachine } from './store/db.js'
import { llmReport } from './dsp/report.js'
import { loadModel } from './ml/ae.js'

const SECONDS = 10
const DEMO = [
  { id: 'healthy', label: 'Normal', hint: 'fan running fine' },
  { id: 'imbalance', label: 'Coin on a blade', hint: 'imbalance' },
  { id: 'looseness', label: 'Loose screw', hint: 'looseness' },
  { id: 'blocked', label: 'Airflow blocked', hint: 'blockage' },
  { id: 'bearing', label: 'Worn bearing', hint: 'bearing' },
  { id: 'misalignment', label: 'Bent shaft', hint: 'misalignment' },
]
const VERDICT = { healthy: 'Healthy', imbalance: 'Imbalance', misalignment: 'Misalignment', looseness: 'Looseness',
  bearing: 'Bearing wear', airflow: 'Airflow blocked', anomaly: 'Abnormal sound' }
const fmt = (v, d = 1) => (Math.abs(v) >= 100 ? Math.round(v) : Number(v).toFixed(d))

function ListenButton({ mode, progress, onClick, disabled }) {
  const r = 86, c = 2 * Math.PI * r
  const listening = mode === 'listening'
  return (
    <button id="listen" className={`listen ${listening ? 'on' : ''}`} onClick={onClick} disabled={disabled}
      aria-label={listening ? 'Listening' : 'Listen'}>
      <span className="pulse" /><span className="pulse p2" />
      <svg viewBox="0 0 200 200" className="ring">
        <circle cx="100" cy="100" r={r} className="ring-track" />
        {listening && <circle cx="100" cy="100" r={r} className="ring-fill" strokeDasharray={c}
          strokeDashoffset={c * (1 - progress)} />}
      </svg>
      <span className="listen-inner">
        <svg viewBox="0 0 24 24" className="steth" aria-hidden="true">
          <path d="M6 3v6a5 5 0 0 0 10 0V3M6 3H4.5M16 3h1.5M11 14v2a5 5 0 0 0 10 0v-2.2" />
          <circle cx="21" cy="11.5" r="2.3" />
        </svg>
        <b>{listening ? `${Math.ceil(SECONDS * (1 - progress))}s` : mode === 'baseline' ? 'Baseline' : 'Listen'}</b>
      </span>
    </button>
  )
}

function Evidence({ res, baseline }) {
  const top = res.rules[0]
  return (
    <section className="card">
      <h3>Evidence</h3>
      <div className="rules">
        {res.rules.map((r) => (
          <div key={r.id} className="rule">
            <span>{r.name}</span>
            <div className="bar"><i style={{ width: `${Math.max(2, r.severity * 100)}%`,
              background: r.severity >= 0.5 ? 'var(--bad)' : r.severity > 0.2 ? 'var(--warn)' : 'var(--ok)' }} /></div>
            <span className="mono">{Math.round(r.severity * 100)}%</span>
          </div>
        ))}
      </div>
      <table className="ev">
        <thead><tr><th>{top.name}: measured</th><th>now</th><th>baseline</th></tr></thead>
        <tbody>
          {top.evidence.map(([name, now, base, unit]) => (
            <tr key={name}>
              <td>{name}</td>
              <td className="mono">{unit === 'dB' && base === 0 && now > 0 ? '+' : ''}{fmt(now)}{unit && unit !== 'count' && unit !== 'ratio' ? ` ${unit}` : ''}</td>
              <td className="mono muted">{base === 0 && (unit === 'dB' || unit === '%') ? '0' : fmt(base)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <h4>Envelope spectrum <span className="muted">· shaft orders of {fmt(res.measures.f1_hz)} Hz ({Math.round(res.measures.rpm)} RPM)</span></h4>
      <SpectrumPlot env={res.spectra.env} baseEnv={baseline.env} f1={res.measures.f1_hz} />
      <p className="legend"><i className="sw now" /> now <i className="sw base" /> baseline</p>
    </section>
  )
}

export default function App() {
  const [machines, setMachines] = useState([])
  const [mid, setMid] = useState(null)
  const [tab, setTab] = useState('listen')
  const [state, setState] = useState('idle') // idle | listening | analyzing | result | baseline-done
  const [progress, setProgress] = useState(0)
  const [analyser, setAnalyser] = useState(null)
  const [res, setRes] = useState(null)
  const [demoCond, setDemoCond] = useState('healthy')
  const [checks, setChecks] = useState([])
  const [err, setErr] = useState(null)
  const [newName, setNewName] = useState('')
  const [llm, setLlm] = useState(null)
  const session = useRef(null)

  const machine = machines.find((m) => m.id === mid)
  const refresh = useCallback(async () => {
    const ms = (await listMachines()).sort((a, b) => a.created - b.created)
    setMachines(ms)
    setMid((cur) => cur ?? ms[0]?.id ?? null)
  }, [])
  useEffect(() => { refresh(); loadModel().catch(() => {}) }, [refresh])
  useEffect(() => { if (mid) machineHistory(mid).then(setChecks) }, [mid, res])

  const addMachine = async (name, demo = false) => {
    const m = { id: crypto.randomUUID(), name: name || 'My machine', demo, created: Date.now(), baseline: null }
    await saveMachine(m)
    await refresh()
    setMid(m.id); setState('idle'); setRes(null)
  }

  const listen = async () => {
    if (!machine || state === 'listening') return
    setErr(null); setRes(null); setLlm(null); setProgress(0)
    const isBaseline = !machine.baseline
    try {
      const src = machine.demo
        ? await listenFile(`${import.meta.env.BASE_URL}demo/${isBaseline ? 'baseline' : demoCond}.wav`, SECONDS, setProgress)
        : await listenMic(SECONDS, setProgress)
      session.current = src
      setAnalyser(src.analyser)
      setState('listening')
      const y = await src.done
      setState('analyzing')
      await new Promise((r) => setTimeout(r, 30))
      if (isBaseline) {
        const baseline = await makeBaseline(y)
        const m = { ...machine, baseline }
        await saveMachine(m)
        await refresh()
        setState('baseline-done')
      } else {
        const r = await runCheck(y, machine)
        await addCheck({ machineId: machine.id, ts: r.ts, health: r.fused.health, label: r.fused.label,
          confidence: r.fused.confidence, demo: machine.demo ? demoCond : null })
        setRes(r)
        setState('result')
      }
    } catch (e) {
      console.error(e)
      setErr(e.name === 'NotAllowedError' ? 'Microphone permission was denied.' : String(e.message || e))
      setState('idle')
    }
  }

  const rewrite = async () => {
    setLlm({ busy: true })
    try { setLlm(await llmReport(res.evidence)) } catch (e) { setLlm({ error: String(e.message || e) }) }
  }

  const mode = state === 'listening' ? 'listening' : machine && !machine.baseline ? 'baseline' : 'listen'
  return (
    <div className="app">
      <header className="top">
        <div className="brand"><span className="logo" />Stetho</div>
        {machines.length > 0 && (
          <select id="machine" value={mid ?? ''} onChange={(e) => { setMid(e.target.value); setState('idle'); setRes(null) }}>
            {machines.map((m) => <option key={m.id} value={m.id}>{m.name}{m.demo ? ' (demo rig)' : ''}</option>)}
          </select>
        )}
      </header>

      {tab === 'listen' && (
        <main className="screen">
          {!machine && (
            <section className="card onboard">
              <h2>A stethoscope for machines</h2>
              <p>Hold your phone near a fan, motor or pump. Stetho learns how it sounds when healthy, then tells you what changed and what to check.</p>
              <div className="row">
                <input id="name" placeholder="Machine name, e.g. Bedroom fan" value={newName} onChange={(e) => setNewName(e.target.value)} />
                <button className="btn" onClick={() => addMachine(newName)}>Add</button>
              </div>
              <button id="demo" className="btn ghost" onClick={() => addMachine('Table fan', true)}>Try the demo rig</button>
              <p className="muted small">Audio is processed on this device and never uploaded.</p>
            </section>
          )}

          {machine && (state === 'listening' || state === 'analyzing' || state === 'idle' || state === 'baseline-done') && (
            <>
              <div className="stage">
                {state === 'listening' || state === 'analyzing'
                  ? <Waterfall analyser={analyser} frozen={state === 'analyzing'} height={250} />
                  : <div className="stage-idle"><div className="grid-floor" /></div>}
              </div>
              <ListenButton mode={mode} progress={progress} onClick={listen} disabled={state === 'analyzing'} />
              <p className="hint">
                {state === 'analyzing' ? 'Analyzing on-device…'
                  : state === 'listening' ? 'Listening. Keep the phone still, 20–30 cm from the machine.'
                  : state === 'baseline-done' ? <>Baseline saved: <b>{Math.round(machine.baseline.measures.rpm)} RPM</b>, {machine.baseline.measures.blades} blades detected. Now tap Listen any time to check it.</>
                  : !machine.baseline ? 'First, record the machine while it runs normally. Stetho learns its healthy sound.'
                  : 'Tap to record 10 seconds and check this machine.'}
              </p>
              {machine.demo && machine.baseline && state !== 'listening' && state !== 'analyzing' && (
                <div className="chips" role="radiogroup" aria-label="Demo condition">
                  {DEMO.map((d) => (
                    <button key={d.id} id={`cond-${d.id}`} className={`chip ${demoCond === d.id ? 'sel' : ''}`}
                      onClick={() => setDemoCond(d.id)} role="radio" aria-checked={demoCond === d.id}>{d.label}</button>
                  ))}
                </div>
              )}
              {err && <p className="error">{err}</p>}
            </>
          )}

          {machine && state === 'result' && res && (
            <div className="result">
              <section className="card center">
                <HealthDial value={res.fused.health} label={VERDICT[res.fused.label]}
                  sub={`confidence ${Math.round(res.fused.confidence * 100)}%`} />
                <p className="report" id="report">{llm?.text ?? res.report.text}</p>
                <p className="muted small">
                  {llm?.source ?? res.report.source} · analysed in {Math.round(res.timing.totalMs)} ms on this device
                  {llm?.error && <><br />LLM unavailable ({llm.error}). Showing the on-device report.</>}
                </p>
                <div className="row">
                  <button id="again" className="btn" onClick={() => { setState('idle'); setRes(null) }}>Check again</button>
                  <button className="btn ghost" onClick={rewrite} disabled={llm?.busy}>{llm?.busy ? 'Writing…' : 'LLM report'}</button>
                </div>
              </section>
              <Evidence res={res} baseline={machine.baseline} />
            </div>
          )}
        </main>
      )}

      {tab === 'history' && (
        <main className="screen">
          <section className="card">
            <h3>{machine?.name ?? 'History'}</h3>
            <History checks={checks} />
          </section>
          <section className="card">
            <h3>Machines</h3>
            {machines.map((m) => (
              <div key={m.id} className="mrow">
                <span>{m.name}{m.demo && <span className="tag">demo</span>}</span>
                <span className="muted small">{m.baseline ? `${Math.round(m.baseline.measures.rpm)} RPM baseline` : 'no baseline'}</span>
                <button className="link" onClick={async () => { await deleteMachine(m.id); setMid(null); refresh() }}>remove</button>
              </div>
            ))}
            <div className="row">
              <input placeholder="New machine" value={newName} onChange={(e) => setNewName(e.target.value)} />
              <button className="btn" onClick={() => { addMachine(newName); setTab('listen') }}>Add</button>
            </div>
            {!machines.some((m) => m.demo) && <button className="btn ghost" onClick={() => { addMachine('Table fan', true); setTab('listen') }}>Add demo rig</button>}
          </section>
        </main>
      )}

      <nav className="tabs">
        <button id="tab-listen" className={tab === 'listen' ? 'on' : ''} onClick={() => setTab('listen')}>Listen</button>
        <button id="tab-history" className={tab === 'history' ? 'on' : ''} onClick={() => setTab('history')}>
          History {checks.length > 0 && <span className="badge" style={{ background: bandColor(checks.at(-1).health) }} />}
        </button>
      </nav>
    </div>
  )
}
