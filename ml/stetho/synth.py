"""Parametric fan acoustics simulator.

Generates physically-motivated audio of an axial fan in six conditions so the
whole pipeline (autoencoder + physics rules + UI) can be tested end-to-end and
so the demo is reproducible without a microphone.

Signal model (per clip, sample rate fs):

    x(t) = [1 + m1 cos(2*pi*fr*t + p1) + m2 cos(2*pi*2fr*t + p2)]
           * ( sum_k a_k sin(2*pi*k*B*fr*t + phi_k)        # blade-pass tones
               + g * n_aero(t) )                            # aerodynamic noise
         + sum_h s_h sin(2*pi*h*fr*t)                       # shaft orders
         + w sin(2*pi*f_motor*t)                            # motor whine
         + fault-specific terms                             # see FAULTS

fr = RPM / 60 is the shaft (1x) frequency and B the blade count, so
BPF = B * fr. Imbalance shows up as 1x amplitude modulation (sidebands at
k*BPF +- fr), misalignment as 2x modulation, looseness as a comb of 1x
harmonics plus half orders and rattle impacts, bearing wear as a structural
resonance (3-6 kHz) excited at the non-synchronous outer-race frequency
BPFO ~ 3.57x, and blocked airflow as broadband stall noise with a small speed
change.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal as sps

FS = 16000
CONDITIONS = ["healthy", "imbalance", "misalignment", "looseness", "bearing", "blocked"]


@dataclass
class FanParams:
    rpm: float = 1250.0
    blades: int = 5
    motor_hz: float = 2400.0
    noise_gain: float = 0.05
    seed: int = 0            # per recording: noise, small jitter
    machine_seed: int = 0    # per machine: tone amplitudes, phases-independent character
    severity: float = 1.0
    extra: dict = field(default_factory=dict)


def _pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    f = np.arange(spec.size)
    f[0] = 1
    spec /= np.sqrt(f)
    out = np.fft.irfft(spec, n)
    return out / (np.std(out) + 1e-12)


def _band(x: np.ndarray, lo: float, hi: float, fs: int = FS, order: int = 4) -> np.ndarray:
    sos = sps.butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    return sps.sosfilt(sos, x)


def synth_fan(condition: str = "healthy", seconds: float = 10.0, p: FanParams | None = None,
              fs: int = FS) -> np.ndarray:
    p = p or FanParams()
    rng = np.random.default_rng(p.seed)
    mrng = np.random.default_rng(p.machine_seed + 7)
    jit = lambda: rng.uniform(0.95, 1.05)  # recording-to-recording variation (mic placement)
    n = int(seconds * fs)
    t = np.arange(n) / fs
    sev = p.severity

    rpm = p.rpm * (1 + rng.normal(0, 0.01))
    if condition == "blocked":
        rpm *= 1 + 0.05 * sev           # unloaded axial fan speeds up slightly
    # slow speed wander (+-0.3%) so tones are not perfectly stationary
    wander = 1 + 0.003 * np.sin(2 * np.pi * rng.uniform(0.05, 0.2) * t + rng.uniform(0, 6.28))
    fr_inst = rpm / 60.0 * wander
    phase_r = 2 * np.pi * np.cumsum(fr_inst) / fs
    fr = rpm / 60.0

    # --- modulation by shaft orders (residual imbalance always exists) ---
    m1 = 0.04 * mrng.uniform(0.7, 1.3) * jit()
    m2 = 0.015 * mrng.uniform(0.7, 1.3) * jit()
    if condition == "imbalance":
        m1 = 0.30 * sev
    if condition == "misalignment":
        m2 = 0.22 * sev
        m1 *= 1.4
    p1, p2 = rng.uniform(0, 2 * np.pi, 2)
    mod = 1 + m1 * np.cos(phase_r + p1) + m2 * np.cos(2 * phase_r + p2)
    if condition == "misalignment":
        mod += 0.08 * sev * np.cos(3 * phase_r)

    # --- blade-pass tones ---
    tones = np.zeros(n)
    for k in range(1, 9):
        a = 0.5 / k ** 1.1 * mrng.uniform(0.6, 1.4) * jit()
        tones += a * np.sin(p.blades * k * phase_r + rng.uniform(0, 2 * np.pi))

    # --- aerodynamic broadband noise (pink, band-limited) ---
    aero = _band(_pink_noise(n, rng), 60, 7000)
    aero /= np.std(aero) + 1e-12
    g = p.noise_gain * 4 * jit()
    if condition == "blocked":
        # rotating stall: louder, low-mid heavy, randomly modulated turbulence
        stall_env = 1 + 0.6 * sev * np.abs(_band(rng.standard_normal(n), 2, 12))
        stall_env /= np.mean(stall_env)
        rumble = _band(_pink_noise(n, rng), 80, 900)
        aero = aero * (1 + 0.8 * sev) * stall_env + 1.5 * sev * rumble / (np.std(rumble) + 1e-12)
        tones *= 1 - 0.25 * sev

    x = mod * (tones + g * aero)

    # --- direct shaft orders (weak in audio; microphones roll off < 100 Hz) ---
    x += 0.03 * np.sin(phase_r) + 0.01 * np.sin(2 * phase_r)
    if condition == "imbalance":
        x += 0.15 * sev * np.sin(phase_r)
    if condition == "misalignment":
        x += 0.12 * sev * np.sin(2 * phase_r) + 0.05 * sev * np.sin(3 * phase_r)

    # --- motor electromagnetic whine ---
    x += 0.02 * mrng.uniform(0.7, 1.3) * np.sin(2 * np.pi * p.motor_hz * t)

    # --- looseness: harmonic comb + half orders + rattle impacts once per rev ---
    if condition == "looseness":
        for h in range(1, 13):
            x += 0.05 * sev / h ** 0.5 * np.sin(h * phase_r + rng.uniform(0, 6.28))
        for h in (0.5, 1.5, 2.5, 3.5):
            x += 0.035 * sev * np.sin(h * phase_r + rng.uniform(0, 6.28))
        rev_idx = np.where(np.diff(np.floor(phase_r / (2 * np.pi))) > 0)[0]
        imp = np.zeros(n)
        keep = rng.random(rev_idx.size) < 0.85
        # the loose part lifts and lands on alternate revolutions -> 0.5x orders
        alt = np.where(np.arange(rev_idx.size) % 2 == 0, 1.0, 0.35)
        imp[rev_idx[keep]] = (alt * rng.uniform(0.7, 1.0, rev_idx.size))[keep]
        ring = np.exp(-np.arange(int(0.012 * fs)) / (0.003 * fs)) * np.sin(
            2 * np.pi * 1300 * np.arange(int(0.012 * fs)) / fs)
        x += 1.4 * sev * np.convolve(imp, ring)[:n]
        # rattle also modulates the whole sound at 1x (the mount moves)
        x *= 1 + 0.06 * sev * np.cos(phase_r) + 0.08 * sev * np.cos(0.5 * phase_r)

    # --- bearing wear: resonance excited at BPFO (non-synchronous ~3.57x) ---
    if condition == "bearing":
        bpfo = 3.57 * fr
        period = fs / bpfo
        idx = (np.arange(0, n / period) * period
               + rng.normal(0, 0.01 * period, int(np.ceil(n / period)))).astype(int)
        idx = idx[(idx >= 0) & (idx < n)]
        imp = np.zeros(n)
        imp[idx] = 1.0
        res = 4200 * rng.uniform(0.9, 1.1)
        L = int(0.004 * fs)
        tt = np.arange(L) / fs
        ring = np.exp(-tt / 0.0006) * np.sin(2 * np.pi * res * tt)
        # load-zone modulation at 1x (outer race defect passes under load)
        burst = np.convolve(imp, ring)[:n] * (1 + 0.5 * np.cos(phase_r))
        x += 0.35 * sev * burst
        x += 0.004 * sev * _band(rng.standard_normal(n), 3000, 7500)   # rough-surface hiss

    # room noise floor + mild mic coloration (roll-off below 80 Hz)
    x += 0.004 * rng.standard_normal(n)
    sos = sps.butter(2, 60, btype="high", fs=fs, output="sos")
    x = sps.sosfilt(sos, x)
    x = x / (np.max(np.abs(x)) + 1e-9) * 0.6 * rng.uniform(0.8, 1.0)
    return x.astype(np.float32)


def random_params(seed: int, machine: str = "table_fan") -> FanParams:
    rng = np.random.default_rng(seed + 10_000)
    if machine == "pc_fan":
        return FanParams(rpm=rng.uniform(1100, 2200), blades=int(rng.choice([7, 9])),
                         motor_hz=rng.uniform(2800, 3600), noise_gain=rng.uniform(0.03, 0.07),
                         seed=seed, machine_seed=seed)
    return FanParams(rpm=rng.uniform(950, 1400), blades=int(rng.choice([3, 4, 5])),
                     motor_hz=rng.uniform(1800, 2800), noise_gain=rng.uniform(0.04, 0.08),
                     seed=seed, machine_seed=seed)
