// In-place iterative radix-2 complex FFT. `re`/`im` are Float64Arrays of length 2^k.
const twiddleCache = new Map()

function twiddles(n) {
  let t = twiddleCache.get(n)
  if (!t) {
    const c = new Float64Array(n / 2), s = new Float64Array(n / 2)
    for (let k = 0; k < n / 2; k++) {
      c[k] = Math.cos((-2 * Math.PI * k) / n)
      s[k] = Math.sin((-2 * Math.PI * k) / n)
    }
    t = { c, s }
    twiddleCache.set(n, t)
  }
  return t
}

export function fft(re, im, inverse = false) {
  const n = re.length
  for (let i = 1, j = 0; i < n; i++) {
    let b = n >> 1
    for (; j & b; b >>= 1) j ^= b
    j ^= b
    if (i < j) {
      let t = re[i]; re[i] = re[j]; re[j] = t
      t = im[i]; im[i] = im[j]; im[j] = t
    }
  }
  const { c, s } = twiddles(n)
  const sign = inverse ? -1 : 1
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1, step = n / len
    for (let i = 0; i < n; i += len) {
      for (let k = 0; k < half; k++) {
        const wr = c[k * step], wi = sign * s[k * step]
        const u = i + k, v = u + half
        const tr = re[v] * wr - im[v] * wi
        const ti = re[v] * wi + im[v] * wr
        re[v] = re[u] - tr; im[v] = im[u] - ti
        re[u] += tr; im[u] += ti
      }
    }
  }
  if (inverse) for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n }
}

export const nextPow2 = (n) => 1 << Math.ceil(Math.log2(n))

// periodic Hann, same as scipy.signal.get_window('hann', n, fftbins=True)
export function hann(n) {
  const w = new Float64Array(n)
  for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n)
  return w
}

// |rfft(x * w, n)|^2 for the first n/2+1 bins
export function powerSpectrum(x, offset, w, n, out) {
  const re = new Float64Array(n), im = new Float64Array(n)
  for (let i = 0; i < w.length; i++) re[i] = x[offset + i] * w[i]
  fft(re, im)
  for (let k = 0; k <= n / 2; k++) out[k] = re[k] * re[k] + im[k] * im[k]
  return out
}
