"""Fusion of autoencoder evidence and physics rules into a 0-100 health score.

    z       = (s - m_b) / sigma_b                 (AE error vs. this machine's baseline)
    H_ae    = 100 / (1 + exp((z - Z0) / K))
    H       = min(H_ae, 100 - 70 * max_severity)

Mirrored 1:1 in web/src/dsp/health.js.
"""
from __future__ import annotations

import numpy as np

Z0 = 6.0          # z at which AE alone says 50/100
K = 1.6
WIN = 31          # vectors per ~1 s window
FAULT_SEV = 0.5   # rule severity that names a fault
UNCLASSIFIED_H = 65


def window_means(errors: np.ndarray, win: int = WIN) -> np.ndarray:
    n = len(errors) // win
    return errors[: n * win].reshape(n, win).mean(1)


def baseline_stats(errors: np.ndarray) -> dict:
    w = window_means(errors)
    m = float(np.median(w))
    sd = float(max(1.4826 * np.median(np.abs(w - m)), 0.08 * m, 1e-6))
    return {"m": m, "sd": sd}


def ae_health(errors: np.ndarray, base: dict) -> tuple[float, float]:
    s = float(np.mean(errors))
    z = (s - base["m"]) / base["sd"]
    return float(100 / (1 + np.exp((z - Z0) / K))), float(z)


def fuse(errors: np.ndarray, base_ae: dict, rules: list[dict]) -> dict:
    h_ae, z = ae_health(errors, base_ae)
    top = rules[0]
    sec = rules[1]["severity"] if len(rules) > 1 else 0.0
    h = min(h_ae, 100 - 70 * top["severity"])
    if top["severity"] >= FAULT_SEV:
        label = top["id"]
        conf = top["severity"] * (1 - 0.5 * sec / (top["severity"] + 1e-9))
        conf = 0.5 * conf + 0.5 * (1 - h_ae / 100) if h_ae < 80 else 0.8 * conf
    elif h < UNCLASSIFIED_H:
        label = "anomaly"
        conf = 1 - h / 100
    else:
        label = "healthy"
        conf = h / 100 * (1 - top["severity"])
    band = "green" if h >= 80 else "yellow" if h >= 50 else "red"
    return {"health": round(h, 1), "health_ae": round(h_ae, 1), "z": round(z, 2),
            "label": label, "band": band, "confidence": round(float(np.clip(conf, 0, 1)), 2)}
