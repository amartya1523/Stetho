// Audio sources. Both expose { analyser, done: Promise<Float32Array @16 kHz>, stop() }.
import { decodeWav } from './wav.js'

const TARGET_SR = 16000

const WORKLET = `
class Tap extends AudioWorkletProcessor {
  process(inputs) { const ch = inputs[0][0]; if (ch) this.port.postMessage(ch.slice()); return true }
}
registerProcessor('stetho-tap', Tap)`

async function resample(x, srIn) {
  if (srIn === TARGET_SR) return x
  const oc = new OfflineAudioContext(1, Math.ceil((x.length * TARGET_SR) / srIn), TARGET_SR)
  const b = oc.createBuffer(1, x.length, srIn)
  b.copyToChannel(x, 0)
  const s = oc.createBufferSource()
  s.buffer = b
  s.connect(oc.destination)
  s.start()
  return (await oc.startRendering()).getChannelData(0)
}

function makeAnalyser(ctx) {
  const a = ctx.createAnalyser()
  a.fftSize = 4096
  a.smoothingTimeConstant = 0.55
  a.minDecibels = -95
  a.maxDecibels = -12
  return a
}

/** Raw microphone capture: all voice processing disabled so machine tones survive. */
export async function listenMic(seconds = 10, onProgress) {
  const ctx = new AudioContext()
  await ctx.resume()
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 },
  })
  const src = ctx.createMediaStreamSource(stream)
  const analyser = makeAnalyser(ctx)
  src.connect(analyser)
  const need = Math.round(seconds * ctx.sampleRate)
  const chunks = []
  let got = 0, node

  const finish = () => {
    stream.getTracks().forEach((t) => t.stop())
    node?.disconnect()
  }
  const done = new Promise((resolve, reject) => {
    const push = (ch) => {
      if (got >= need) return
      chunks.push(ch); got += ch.length
      onProgress?.(Math.min(1, got / need))
      if (got >= need) {
        finish()
        const x = new Float32Array(need)
        let o = 0
        for (const c of chunks) { x.set(c.subarray(0, Math.min(c.length, need - o)), o); o += c.length; if (o >= need) break }
        resample(x, ctx.sampleRate).then(resolve, reject).finally(() => ctx.close())
      }
    }
    const silent = ctx.createGain()
    silent.gain.value = 0
    silent.connect(ctx.destination)
    if (ctx.audioWorklet) {
      const url = URL.createObjectURL(new Blob([WORKLET], { type: 'application/javascript' }))
      ctx.audioWorklet.addModule(url).then(() => {
        node = new AudioWorkletNode(ctx, 'stetho-tap')
        node.port.onmessage = (e) => push(e.data)
        src.connect(node).connect(silent)
      }, reject)
    } else {
      node = ctx.createScriptProcessor(4096, 1, 1)
      node.onaudioprocess = (e) => push(e.inputBuffer.getChannelData(0).slice())
      src.connect(node).connect(silent)
    }
  })
  const settings = stream.getAudioTracks()[0]?.getSettings?.() ?? {}
  return { analyser, done, stop: finish, settings, sampleRate: ctx.sampleRate }
}

/** Demo rig: plays a recorded clip (audible) and analyses its exact samples. */
export async function listenFile(url, seconds = 10, onProgress) {
  const ctx = new AudioContext()
  await ctx.resume()
  const { samples, sampleRate } = decodeWav(await (await fetch(url)).arrayBuffer())
  const n = Math.min(samples.length, Math.round(seconds * sampleRate))
  const buf = ctx.createBuffer(1, n, sampleRate)
  buf.copyToChannel(samples.subarray(0, n), 0)
  const src = ctx.createBufferSource()
  src.buffer = buf
  const analyser = makeAnalyser(ctx)
  const vol = ctx.createGain()
  vol.gain.value = 0.35
  src.connect(analyser)
  analyser.connect(vol).connect(ctx.destination)
  const t0 = ctx.currentTime
  src.start()
  let timer
  const done = new Promise((resolve) => {
    timer = setInterval(() => {
      const k = (ctx.currentTime - t0) / (n / sampleRate)
      onProgress?.(Math.min(1, k))
      if (k >= 1) { clearInterval(timer); ctx.close(); resolve(samples.subarray(0, n)) }
    }, 50)
  })
  return { analyser, done, stop: () => { clearInterval(timer); src.stop(); ctx.close() }, settings: {}, sampleRate }
}
