"""Physics layer: shaft-speed estimation and order-based fault rules.

Everything here uses plain FFTs (no IIR filters) so web/src/dsp/physics.js can
reproduce it exactly.

Why envelope analysis? Phone microphones barely capture the 13-40 Hz shaft
frequency directly. But every rotating defect *modulates* the much louder
aerodynamic/blade-pass sound at the shaft rate. Demodulating a band
(|analytic signal|) and taking its spectrum, the "envelope spectrum", exposes
1x, 2x and bearing defect frequencies even when the raw spectrum shows nothing
below 100 Hz.
"""
from __future__ import annotations

import numpy as np

FS = 16000
WELCH_N = 32768          # 0.49 Hz resolution
ENV_DEC = 32             # envelope sample rate 500 Hz
ENV_NFFT = 16384         # 0.0305 Hz envelope-spectrum resolution
MID_BAND = (300.0, 3000.0)
HF_BAND = (2500.0, 7500.0)
F1_RANGE = (8.0, 50.0)
BLADES = (3, 4, 5, 7, 9)


def _next_pow2(n: int) -> int:
    return 1 << (int(n) - 1).bit_length()


def hann(n: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)


def welch_db(x: np.ndarray, n: int = WELCH_N):
    w = hann(n)
    hop = n // 2
    acc = np.zeros(n // 2 + 1)
    cnt = 0
    for s in range(0, len(x) - n + 1, hop):
        acc += np.abs(np.fft.rfft(x[s:s + n] * w)) ** 2
        cnt += 1
    psd = acc / max(cnt, 1) / np.sum(w ** 2)
    return np.fft.rfftfreq(n, 1 / FS), 10 * np.log10(psd + 1e-12)


def band_analytic(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    n = _next_pow2(len(x))
    X = np.fft.fft(x, n)
    f = np.fft.fftfreq(n, 1 / FS)
    H = np.where((f >= lo) & (f <= hi), 2.0, 0.0)
    return np.fft.ifft(X * H)[: len(x)]


def envelope_spectrum(z: np.ndarray):
    """Modulation-depth spectrum of |z|: amplitude of each envelope tone / DC."""
    env = np.abs(z)
    m = len(env) // ENV_DEC
    env = env[: m * ENV_DEC].reshape(m, ENV_DEC).mean(1)
    dc = env.mean() + 1e-12
    w = hann(m)
    spec = np.abs(np.fft.rfft((env - dc) * w, ENV_NFFT)) * 2 / w.sum() / dc
    return np.fft.rfftfreq(ENV_NFFT, ENV_DEC / FS), spec


def _peak(f, s, f0, tol=0.015):
    """Peak value near f0 and its prominence over the local median (dB)."""
    half = max(f0 * tol, 0.25)
    i0, i1 = np.searchsorted(f, [f0 - half, f0 + half])
    j0, j1 = np.searchsorted(f, [f0 - max(4.0, 0.15 * f0), f0 + max(4.0, 0.15 * f0)])
    if i1 <= i0 or j1 <= j0:
        return 0.0, 0.0
    pk = float(np.max(s[i0:i1]))
    floor = float(np.median(s[j0:j1])) + 1e-12
    return pk, 20 * np.log10(pk / floor + 1e-12)


def _peak_db(f, sdb, f0, tol=0.015):
    half = max(f0 * tol, 0.5)
    i0, i1 = np.searchsorted(f, [f0 - half, f0 + half])
    j0, j1 = np.searchsorted(f, [f0 - max(8.0, 0.15 * f0), f0 + max(8.0, 0.15 * f0)])
    pk = float(np.max(sdb[i0:i1]))
    return pk, pk - float(np.median(sdb[j0:j1]))


def estimate_f1(fe, se, fp, pdb, hint: float | None = None, blades: int | None = None):
    """Harmonic-sum search for shaft frequency and blade count.

    score(f1, B) = sum_k PNR_psd(k*B*f1) + sum_h PNR_env(h*f1)
    With a baseline hint we only search +-8 % around it (avoids octave errors
    when a 2x fault becomes dominant)."""
    lo, hi = (hint * 0.92, hint * 1.08) if hint else F1_RANGE
    cands = np.arange(lo, hi, 0.05)
    best = (-1e9, cands[0], BLADES[0])
    for f1 in cands:
        env_pnr = {h: max(_peak(fe, se, h * f1)[1], 0) for h in (1, 2, 3)}
        for B in ((blades,) if blades else BLADES):
            bpf = B * f1
            if bpf * 4 > FS / 2 or bpf < 30:
                continue
            ps = sum(max(_peak_db(fp, pdb, k * bpf)[1], 0) for k in (1, 2, 3, 4))
            # sidebands BPF +- f1 confirm the shaft rate
            sb = 0.5 * sum(max(_peak_db(fp, pdb, bpf + d * f1, 0.01)[1], 0) for d in (-1, 1))
            # envelope orders that coincide with blade-pass multiples prove nothing
            env = 2.0 * sum(v for h, v in env_pnr.items() if h % B != 0)
            # octave guard: a real blade-pass family has nothing at BPF/2, 3BPF/2
            sub = sum(max(_peak_db(fp, pdb, k * bpf / 2)[1], 0) for k in (1, 3))
            sc = ps + sb + env - 1.5 * sub
            if sc > best[0]:
                best = (sc, f1, B)
    return float(best[1]), int(best[2]), float(best[0])


def kurtosis(v: np.ndarray) -> float:
    v = v - v.mean()
    return float(np.mean(v ** 4) / (np.mean(v ** 2) ** 2 + 1e-20))


def analyze(y: np.ndarray, hint: dict | None = None) -> dict:
    """Measure physical quantities of one recording (dict of floats)."""
    y = y.astype(np.float64)
    y = y - y.mean()
    fp, pdb = welch_db(y)
    total_db = 10 * np.log10(np.mean(y ** 2) + 1e-12)

    zm = band_analytic(y, *MID_BAND)
    fe, se = envelope_spectrum(zm)
    zh = band_analytic(y, *HF_BAND)
    fh, sh = envelope_spectrum(zh)

    hint = hint or {}
    f1, B, _ = estimate_f1(fe, se, fp, pdb, hint.get("f1"), hint.get("blades"))

    orders = {}
    for h in (0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10):
        amp, pnr = _peak(fe, se, h * f1)
        orders[h] = (amp, pnr)
    n_harm = sum(1 for h in range(1, 11) if orders[h][1] > 6.0 and h % B != 0)
    half = max(orders[h][1] for h in (0.5, 1.5, 2.5))
    # modulation energy in the higher shaft orders (blade-pass multiples excluded)
    harm_mod = float(np.sqrt(sum(orders[h][0] ** 2 for h in range(4, 11) if h % B != 0)))
    half_mod = float(np.sqrt(sum(orders[h][0] ** 2 for h in (0.5, 1.5, 2.5))))

    # bearing: non-synchronous peaks in the HF envelope spectrum
    ns_best = 0.0
    ns_freq = 0.0
    lo_i, hi_i = np.searchsorted(fh, [1.8 * f1, 12 * f1])
    for i in range(lo_i + 1, hi_i - 1):
        if not (sh[i] >= sh[i - 1] and sh[i] >= sh[i + 1]):
            continue
        order = fh[i] / f1
        if abs(order - round(order)) < 0.06 or abs(order - round(order * 2) / 2) < 0.03:
            continue
        _, pnr = _peak(fh, sh, fh[i], 0.004)
        if pnr > ns_best:
            ns_best, ns_freq = pnr, fh[i]

    hf_band = np.real(zh) / 2
    hf_db = 10 * np.log10(np.mean(hf_band ** 2) + 1e-12) - total_db

    bpf = B * f1
    bpf_db, bpf_pnr = _peak_db(fp, pdb, bpf)
    m0, m1 = np.searchsorted(fp, [200, 2000])
    floor_db = float(np.median(pdb[m0:m1])) - total_db

    return {
        "f1_hz": f1, "rpm": f1 * 60, "blades": B, "bpf_hz": bpf,
        "mod_1x": orders[1][0], "mod_2x": orders[2][0], "mod_3x": orders[3][0],
        "pnr_1x": orders[1][1], "pnr_2x": orders[2][1], "pnr_3x": orders[3][1],
        "n_harmonics": int(n_harm), "half_order_pnr": float(half),
        "harm_mod": harm_mod, "half_mod": half_mod,
        "hf_db": float(hf_db), "hf_kurtosis": kurtosis(hf_band),
        "nonsync_pnr": float(ns_best), "nonsync_order": float(ns_freq / f1 if ns_freq else 0),
        "bpf_tonality_db": float(bpf_pnr), "floor_db": floor_db,
    }


def _db(a, b):
    return 20 * np.log10((a + 1e-6) / (b + 1e-6))


def _ramp(v, lo, hi):
    return float(np.clip((v - lo) / (hi - lo), 0, 1))


FAULTS = {
    "imbalance": "Imbalance",
    "misalignment": "Misalignment",
    "looseness": "Mechanical looseness",
    "bearing": "Bearing wear",
    "airflow": "Airflow blockage",
}


def diagnose(m: dict, b: dict) -> list[dict]:
    """Compare measurement m with healthy baseline b. Returns rule results,
    each with severity in [0, 1] and the numbers that triggered it."""
    d1 = _db(m["mod_1x"], b["mod_1x"])
    d2 = _db(m["mod_2x"], b["mod_2x"])
    d3 = _db(m["mod_3x"], b["mod_3x"])
    r21 = m["mod_2x"] / (m["mod_1x"] + 1e-9)
    dhf = m["hf_db"] - b["hf_db"]
    dk = m["hf_kurtosis"] - b["hf_kurtosis"]
    dfloor = m["floor_db"] - b["floor_db"]
    dton = m["bpf_tonality_db"] - b["bpf_tonality_db"]
    drpm = 100 * (m["rpm"] - b["rpm"]) / b["rpm"]

    out = []
    dharm = _db(m["harm_mod"], b["harm_mod"])
    dhalf = _db(m["half_mod"], b["half_mod"])
    loose = _ramp(dharm, 2, 7) * (0.5 + 0.5 * _ramp(dhalf, 2, 7))
    out.append({"id": "looseness", "severity": loose, "evidence": [
        ("4x-10x harmonic modulation change", dharm, 0.0, "dB"),
        ("half-order (0.5x, 1.5x, 2.5x) change", dhalf, 0.0, "dB"),
        ("shaft harmonics present", m["n_harmonics"], b["n_harmonics"], "count")]})

    # imbalance is a *pure* 1x effect: penalise when the harmonic comb also grew
    imb = _ramp(d1, 4, 12) * (1 - 0.6 * _ramp(r21, 0.8, 1.5)) * (1 - max(_ramp(dharm, 3, 8), _ramp(dhalf, 3, 8)))
    out.append({"id": "imbalance", "severity": imb, "evidence": [
        ("1x shaft modulation change", d1, 0.0, "dB"),
        ("2x/1x ratio", r21, b["mod_2x"] / (b["mod_1x"] + 1e-9), "ratio")]})

    mis = _ramp(d2, 4, 12) * _ramp(r21, 0.5, 1.0) * (1 - _ramp(dharm, 3, 8))
    out.append({"id": "misalignment", "severity": mis, "evidence": [
        ("2x shaft modulation change", d2, 0.0, "dB"),
        ("3x shaft modulation change", d3, 0.0, "dB"),
        ("2x/1x ratio", r21, b["mod_2x"] / (b["mod_1x"] + 1e-9), "ratio")]})

    brg = max(_ramp(dhf, 3, 10), _ramp(dk, 1.0, 6.0),
              _ramp(m["nonsync_pnr"] - b["nonsync_pnr"], 4, 12))
    out.append({"id": "bearing", "severity": brg, "evidence": [
        ("2.5-7.5 kHz energy change", dhf, 0.0, "dB"),
        ("high-band kurtosis", m["hf_kurtosis"], b["hf_kurtosis"], ""),
        ("non-synchronous tone prominence", m["nonsync_pnr"], b["nonsync_pnr"], "dB")]})

    # stalled / blocked flow: blade-pass tone drowns in turbulence, speed shifts
    air = _ramp(-dton, 4, 10) * (0.6 + 0.4 * max(_ramp(dfloor, 1, 4), _ramp(abs(drpm), 2.5, 5)))
    out.append({"id": "airflow", "severity": air, "evidence": [
        ("broadband noise floor change", dfloor, 0.0, "dB"),
        ("blade-pass tonality change", dton, 0.0, "dB"),
        ("speed change", drpm, 0.0, "%")]})

    for r in out:
        r["name"] = FAULTS[r["id"]]
    return sorted(out, key=lambda r: -r["severity"])
