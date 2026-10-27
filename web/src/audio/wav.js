// Minimal RIFF/WAVE reader (PCM16 / float32, first channel) and PCM16 writer.
export function decodeWav(buf) {
  const v = new DataView(buf instanceof ArrayBuffer ? buf : buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength))
  let p = 12, fmt = null
  while (p + 8 <= v.byteLength) {
    const id = String.fromCharCode(v.getUint8(p), v.getUint8(p + 1), v.getUint8(p + 2), v.getUint8(p + 3))
    const size = v.getUint32(p + 4, true)
    if (id === 'fmt ') {
      fmt = { tag: v.getUint16(p + 8, true), ch: v.getUint16(p + 10, true),
        sr: v.getUint32(p + 12, true), bits: v.getUint16(p + 22, true) }
    } else if (id === 'data' && fmt) {
      const bytes = fmt.bits / 8, n = Math.floor(size / (bytes * fmt.ch))
      const out = new Float32Array(n)
      for (let i = 0; i < n; i++) {
        const o = p + 8 + i * bytes * fmt.ch
        out[i] = fmt.bits === 16 ? v.getInt16(o, true) / 32768 : v.getFloat32(o, true)
      }
      return { sampleRate: fmt.sr, samples: out }
    }
    p += 8 + size + (size & 1)
  }
  throw new Error('unsupported wav')
}

export function encodeWav(samples, sr) {
  const b = new ArrayBuffer(44 + samples.length * 2), v = new DataView(b)
  const w = (o, s) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)))
  w(0, 'RIFF'); v.setUint32(4, 36 + samples.length * 2, true); w(8, 'WAVE'); w(12, 'fmt ')
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true)
  v.setUint32(24, sr, true); v.setUint32(28, sr * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true)
  w(36, 'data'); v.setUint32(40, samples.length * 2, true)
  for (let i = 0; i < samples.length; i++) v.setInt16(44 + i * 2, Math.max(-1, Math.min(1, samples[i])) * 32767, true)
  return b
}
