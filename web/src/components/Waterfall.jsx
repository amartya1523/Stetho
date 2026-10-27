import { useEffect, useRef } from 'react'
import * as THREE from 'three'

const BINS = 128, ROWS = 96

const vertexShader = /* glsl */ `
  uniform sampler2D uTex; uniform float uHead, uRows, uAmp;
  varying float vH; varying float vFog; varying vec2 vUv;
  void main() {
    vUv = uv;
    // uv.y = 0 is the front edge (newest row), 1 the back (oldest)
    float r = mod(uHead - 1.0 - uv.y * (uRows - 1.0), uRows);
    float h = texture2D(uTex, vec2(uv.x, (r + 0.5) / uRows)).r;
    h *= smoothstep(0.0, 0.05, uv.x) * smoothstep(1.0, 0.95, uv.x);
    vH = h;
    vec3 p = position; p.z = h * uAmp;
    vec4 mv = modelViewMatrix * vec4(p, 1.0);
    vFog = smoothstep(1.6, 4.2, -mv.z);
    gl_Position = projectionMatrix * mv;
  }`

const fragmentShader = /* glsl */ `
  precision mediump float;
  uniform vec3 uFog; varying float vH, vFog; varying vec2 vUv;
  vec3 heat(float t) {
    vec3 a = vec3(0.02, 0.07, 0.10), b = vec3(0.05, 0.85, 0.78), c = vec3(1.0, 0.70, 0.12), d = vec3(1.0, 0.22, 0.25);
    return t < .4 ? mix(a, b, t / .4) : t < .75 ? mix(b, c, (t - .4) / .35) : mix(c, d, (t - .75) / .25);
  }
  void main() {
    vec3 col = heat(vH);
    col += col * pow(vH, 3.0) * 0.9;
    vec2 g = abs(fract(vUv * vec2(127., 95.) - 0.5) - 0.5);
    col *= 0.7 + 0.3 * smoothstep(0.0, 0.1, min(g.x, g.y));
    gl_FragColor = vec4(mix(col, uFog, vFog), 1.0);
  }`

/** Live 3D spectrogram waterfall fed by a Web Audio AnalyserNode. */
export default function Waterfall({ analyser, frozen = false, height = 260 }) {
  const host = useRef(null)
  const frozenRef = useRef(frozen)
  frozenRef.current = frozen

  useEffect(() => {
    const el = host.current
    const renderer = new THREE.WebGLRenderer({ antialias: false, alpha: true, powerPreference: 'high-performance', preserveDrawingBuffer: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    el.appendChild(renderer.domElement)
    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 20)
    camera.position.set(0, 1.25, 2.35)
    camera.lookAt(0, 0.05, -0.25)

    const data = new Uint8Array(BINS * ROWS)
    const tex = new THREE.DataTexture(data, BINS, ROWS, THREE.RedFormat, THREE.UnsignedByteType)
    tex.minFilter = tex.magFilter = THREE.LinearFilter
    tex.needsUpdate = true
    const mat = new THREE.ShaderMaterial({
      uniforms: { uTex: { value: tex }, uHead: { value: 0 }, uRows: { value: ROWS }, uAmp: { value: 0.55 },
        uFog: { value: new THREE.Color(0x070b10) } },
      vertexShader, fragmentShader,
    })
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(2.2, 2.4, BINS - 1, ROWS - 1), mat)
    mesh.rotation.x = -Math.PI / 2
    scene.add(mesh)

    let fft = null, map = null
    if (analyser) {
      fft = new Uint8Array(analyser.frequencyBinCount)
      const nyq = analyser.context.sampleRate / 2, fMin = 40, fMax = 8000
      map = Uint16Array.from({ length: BINS }, (_, i) =>
        Math.min(fft.length - 1, Math.round(((fMin * Math.pow(fMax / fMin, i / (BINS - 1))) / nyq) * fft.length)))
    }
    let head = 0, last = 0
    const resize = () => {
      const w = el.clientWidth, h = el.clientHeight
      renderer.setSize(w, h, false)
      renderer.domElement.style.width = '100%'
      renderer.domElement.style.height = '100%'
      camera.aspect = w / h
      camera.updateProjectionMatrix()
    }
    const ro = new ResizeObserver(resize)
    ro.observe(el)
    resize()

    renderer.setAnimationLoop((t) => {
      if (analyser && !frozenRef.current && t - last > 28) {
        last = t
        analyser.getByteFrequencyData(fft)
        const row = head * BINS
        for (let i = 0; i < BINS; i++) {
          const a = map[i], b = i < BINS - 1 ? Math.max(a, map[i + 1] - 1) : a
          let m = fft[a]
          for (let k = a + 1; k <= b; k++) if (fft[k] > m) m = fft[k]
          data[row + i] = m
        }
        tex.needsUpdate = true
        head = (head + 1) % ROWS
        mat.uniforms.uHead.value = head
      }
      mesh.rotation.z = Math.sin(t / 4000) * 0.06
      renderer.render(scene, camera)
    })
    return () => {
      renderer.setAnimationLoop(null)
      ro.disconnect()
      mesh.geometry.dispose(); mat.dispose(); tex.dispose(); renderer.dispose()
      el.removeChild(renderer.domElement)
    }
  }, [analyser])

  return <div ref={host} className="waterfall" style={{ height }} aria-label="Live 3D spectrogram" />
}
