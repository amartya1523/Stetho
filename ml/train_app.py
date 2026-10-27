"""Train the on-device model, evaluate the full pipeline, export ONNX + demo audio.

1. Train one AE on healthy fans: simulated table/PC fans + real MIMII fan
   normals (if fetched). It's a *generic* "healthy rotating machine" model;
   each user machine is calibrated by its own baseline recording.
2. Evaluate the complete app pipeline (baseline -> check -> AE + physics ->
   fused health + label) on UNSEEN simulated machines: confusion matrix,
   false-alarm rate, severity sweep, noise robustness, AE/physics ablation.
3. Export web/public/models/stetho_ae.onnx and the demo WAVs.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from sklearn.metrics import roc_auc_score

from stetho.features import log_mel, stack_frames
from stetho.health import baseline_stats, fuse
from stetho.model import ErrorHead, train_ae
from stetho.physics import analyze, diagnose
from stetho.synth import CONDITIONS, FS, random_params, synth_fan

ROOT = Path(__file__).parent.parent
RES = ROOT / "results"
WEB = ROOT / "web" / "public"
LABEL = {"healthy": "healthy", "imbalance": "imbalance", "misalignment": "misalignment",
         "looseness": "looseness", "bearing": "bearing", "blocked": "airflow"}
CLASSES = ["healthy", "imbalance", "misalignment", "looseness", "bearing", "airflow", "anomaly"]


def errors(ae, y):
    with torch.no_grad():
        return ae.errors(torch.from_numpy(stack_frames(log_mel(y)))).numpy()


def add_noise(y, snr_db, rng):
    n = len(y)
    pink = np.fft.irfft(np.fft.rfft(rng.standard_normal(n)) / np.sqrt(np.arange(n // 2 + 1) + 1), n)
    babble = sum(np.sin(2 * np.pi * rng.uniform(150, 900) * np.arange(n) / FS)
                 * (0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(2, 6) * np.arange(n) / FS)) for _ in range(6))
    noise = pink / pink.std() + 0.5 * babble / babble.std()
    noise *= np.sqrt(np.mean(y ** 2) / np.mean(noise ** 2)) * 10 ** (-snr_db / 20)
    return (y + noise).astype(np.float32)


class Machine:
    """One monitored machine: a baseline recording, then checks against it."""

    def __init__(self, ae, seed, kind, snr=None):
        self.p = random_params(seed, kind)
        self.ae, self.snr = ae, snr
        self.p.seed = seed * 100
        y = synth_fan("healthy", p=self.p)
        if snr is not None:     # same room noise during baseline and checks
            y = add_noise(y, snr, np.random.default_rng(seed))
        self.base_phys = analyze(y)
        self.hint = {"f1": self.base_phys["f1_hz"], "blades": self.base_phys["blades"]}
        self.base_ae = baseline_stats(errors(ae, y))

    def check(self, cond, sev=1.0, clip_seed=0):
        self.p.seed, self.p.severity = clip_seed, sev
        y = synth_fan(cond, p=self.p)
        if self.snr is not None:
            y = add_noise(y, self.snr, np.random.default_rng(clip_seed))
        e = errors(self.ae, y)
        rules = diagnose(analyze(y, self.hint), self.base_phys)
        return fuse(e, self.base_ae, rules), rules, e


def main():
    RES.mkdir(exist_ok=True)
    t0 = time.time()
    # ---------------- 1. training data ----------------
    train = []
    for i in range(36):
        p = random_params(i, "pc_fan" if i % 3 == 0 else "table_fan")
        for k in range(6):
            p.seed = i * 1000 + k
            train.append(log_mel(synth_fan("healthy", p=p)))
    n_sim = len(train)
    real = sorted((ROOT / "ml" / "data" / "dcase").glob("fan_id*.npz"))
    for f in real:
        train.extend(np.load(f)["train"].astype(np.float32))
    print(f"training clips: {n_sim} simulated + {len(train) - n_sim} real MIMII fan  ({time.time()-t0:.0f}s)")
    ae = train_ae(np.stack(train), epochs=40, normalize=True, verbose=True)

    # ---------------- 2. pipeline evaluation ----------------
    machines = [Machine(ae, 500 + j, "pc_fan" if j % 3 == 0 else "table_fan") for j in range(12)]
    conf = np.zeros((6, len(CLASSES)), int)
    records = []
    for mi, m in enumerate(machines):
        for ci, c in enumerate(CONDITIONS):
            for r in range(4):
                sev = [0.6, 0.8, 1.0, 1.0][r]
                out, rules, e = m.check(c, sev, clip_seed=mi * 7919 + ci * 101 + r)
                conf[ci, CLASSES.index(out["label"])] += 1
                records.append({"machine": mi, "cond": c, "sev": sev, **out,
                                "phys_max": rules[0]["severity"]})
    acc = np.trace(conf[:, :6]) / conf.sum()
    healthy = [r for r in records if r["cond"] == "healthy"]
    faulty = [r for r in records if r["cond"] != "healthy"]
    far = np.mean([r["label"] != "healthy" for r in healthy])
    det = np.mean([r["label"] != "healthy" for r in faulty])
    y = np.r_[np.zeros(len(healthy)), np.ones(len(faulty))]
    abl = {
        "ae_only_auc": roc_auc_score(y, [r["z"] for r in healthy + faulty]),
        "physics_only_auc": roc_auc_score(y, [r["phys_max"] for r in healthy + faulty]),
        "fused_auc": roc_auc_score(y, [100 - r["health"] for r in healthy + faulty]),
    }
    for r in healthy:
        if r["label"] != "healthy":
            print("  false alarm:", {k: r[k] for k in ("machine", "health", "health_ae", "z", "label", "phys_max")})
    print(f"\npipeline accuracy (6 classes) {acc:.3f}  detection {det:.3f}  false alarms {far:.3f}")
    print("ablation", {k: round(v, 3) for k, v in abl.items()})
    print("confusion (rows=truth, cols=pred):", CLASSES)
    for c, row in zip(CONDITIONS, conf):
        print(f"  {c:12s}", row)

    # severity sweep: detection rate as the fault grows
    sweep = {}
    for c in CONDITIONS[1:]:
        sweep[c] = []
        for sev in (0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0):
            hits = []
            for mi, m in enumerate(machines[:8]):
                out, _, _ = m.check(c, sev, clip_seed=90_000 + mi)
                hits.append(out["label"] != "healthy")
            sweep[c].append((sev, float(np.mean(hits))))
    print("severity sweep", {c: [h for _, h in v] for c, v in sweep.items()})

    # robustness to background noise
    snr_res = []
    for snr in (30, 20, 10, 5, 0):
        correct, fa = [], []
        for mi in range(8):
            m = Machine(ae, 500 + mi, "pc_fan" if mi % 3 == 0 else "table_fan", snr=snr)
            for ci, c in enumerate(CONDITIONS):
                out, _, _ = m.check(c, 1.0, clip_seed=70_000 + mi * 10 + ci)
                correct.append(out["label"] == LABEL[c])
                if c == "healthy":
                    fa.append(out["label"] != "healthy")
        snr_res.append({"snr": snr, "accuracy": float(np.mean(correct)), "false_alarm": float(np.mean(fa))})
        print(f"  SNR {snr:2d} dB  acc {np.mean(correct):.2f}  false alarms {np.mean(fa):.2f}")

    # degradation timeline for the README ("health dropped from 92 to 61")
    m = machines[1]
    timeline = []
    for day in range(15):
        sev = 0 if day < 4 else min(1.0, (day - 3) * 0.09)
        out, rules, _ = m.check("imbalance" if sev else "healthy", sev, clip_seed=50_000 + day)
        timeline.append({"day": day, "health": out["health"], "label": out["label"], "sev": round(sev, 2)})
    print("timeline", [t["health"] for t in timeline])

    (RES / "pipeline.json").write_text(json.dumps({
        "classes": CLASSES, "conditions": CONDITIONS, "confusion": conf.tolist(),
        "accuracy": acc, "detection_rate": det, "false_alarm_rate": far, "ablation": abl,
        "severity_sweep": sweep, "snr": snr_res, "timeline": timeline,
        "records": records, "n_train_sim": n_sim, "n_train_real": len(train) - n_sim,
    }, indent=1, default=float))

    # ---------------- 3. export ----------------
    (WEB / "models").mkdir(parents=True, exist_ok=True)
    head = ErrorHead(ae).eval()
    dummy = torch.randn(4, 640)
    path = WEB / "models" / "stetho_ae.onnx"
    torch.onnx.export(head, (dummy,), str(path), input_names=["x"], output_names=["err"],
                      opset_version=17, dynamic_axes={"x": {0: "n"}, "err": {0: "n"}}, dynamo=False)
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    xb = np.random.randn(7, 640).astype(np.float32) * 10 - 40
    np.testing.assert_allclose(sess.run(None, {"x": xb})[0], head(torch.from_numpy(xb)).detach().numpy(),
                               rtol=1e-3, atol=1e-4)
    print(f"exported {path.name} ({path.stat().st_size/1024:.0f} KB), onnxruntime parity OK")

    # demo audio: one table fan in every condition
    demo = WEB / "demo"
    demo.mkdir(exist_ok=True)
    p = random_params(7, "table_fan")
    for i, c in enumerate(CONDITIONS):
        p.seed, p.severity = 3000 + i, 1.0
        sf.write(demo / f"{c}.wav", synth_fan(c, p=p), FS, subtype="PCM_16")
    p.seed, p.severity = 2999, 1.0
    sf.write(demo / "baseline.wav", synth_fan("healthy", p=p), FS, subtype="PCM_16")
    torch.save(ae.state_dict(), RES / "stetho_ae.pt")
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
