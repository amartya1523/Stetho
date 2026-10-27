"""Benchmark on the DCASE 2020 Task 2 / MIMII subset fetched by fetch_dcase.py.

Protocol (official): one model per machine ID, trained on normal clips only;
AUC and pAUC (max_fpr=0.1) over that ID's normal vs anomalous test clips.

Variants
  replica : baseline topology, raw log-mel, clip score = mean frame error
  stetho  : + per-ID feature standardisation, cosine LR, score = mean of the
            worst 10 % of frames (short faults are not averaged away)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from stetho.model import clip_scores, train_ae

DATA = Path(__file__).parent / "data" / "dcase"
OUT = Path(__file__).parent.parent / "results"
SEEDS = (0, 1)

# Official DCASE 2020 Task 2 baseline, development set (AUC %, pAUC %)
OFFICIAL = {
    "fan": {"00": (54.41, 49.37), "02": (73.40, 52.47), "04": (61.61, 55.24), "06": (73.92, 52.72)},
    "pump": {"00": (67.15, 56.74), "02": (61.53, 58.10), "04": (88.33, 67.10), "06": (74.55, 62.01)},
    "slider": {"00": (96.19, 81.47), "02": (78.97, 63.68), "04": (94.30, 71.98), "06": (69.59, 49.10)},
    "valve": {"00": (68.76, 51.70), "02": (68.18, 51.83), "04": (74.30, 61.20), "06": (53.90, 48.43)},
}

VARIANTS = {
    "replica": dict(normalize=False, reduce="mean"),
    "stetho": dict(normalize=True, reduce="top10"),
}


def main():
    OUT.mkdir(exist_ok=True)
    results = {}
    parts = OUT / "dcase_parts"
    parts.mkdir(exist_ok=True)
    for f in sorted(DATA.glob("*.npz")):
        machine, mid = f.stem.split("_id")
        cache = parts / f"{f.stem}.json"
        if cache.exists():                      # resumable: one JSON per machine ID
            results.setdefault(machine, {})[mid] = json.loads(cache.read_text())
            continue
        d = np.load(f)
        y = np.r_[np.zeros(len(d["test_normal"])), np.ones(len(d["test_anomaly"]))]
        test = np.concatenate([d["test_normal"], d["test_anomaly"]])
        row = {"n_train": int(len(d["train"])), "n_test": int(len(y))}
        for name, cfg in VARIANTS.items():
            aucs, paucs, curves = [], [], None
            t0 = time.time()
            for seed in SEEDS:
                ae = train_ae(d["train"], epochs=30, normalize=cfg["normalize"], seed=seed)
                s = clip_scores(ae, test, cfg["reduce"])
                aucs.append(roc_auc_score(y, s))
                paucs.append(roc_auc_score(y, s, max_fpr=0.1))
                if curves is None:
                    fpr, tpr, _ = roc_curve(y, s)
                    curves = {"fpr": fpr.round(4).tolist(), "tpr": tpr.round(4).tolist(),
                              "scores_normal": s[y == 0].round(5).tolist(),
                              "scores_anomaly": s[y == 1].round(5).tolist()}
            row[name] = {"auc": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
                         "pauc": float(np.mean(paucs)), "pauc_std": float(np.std(paucs)),
                         "roc": curves}
            print(f"{machine:6s} id{mid} {name:8s} AUC {100*np.mean(aucs):5.1f}±{100*np.std(aucs):.1f}"
                  f"  pAUC {100*np.mean(paucs):5.1f}  ({time.time()-t0:.0f}s)", flush=True)
        row["official"] = {"auc": OFFICIAL[machine][mid][0] / 100, "pauc": OFFICIAL[machine][mid][1] / 100}
        results.setdefault(machine, {})[mid] = row
        cache.write_text(json.dumps(row))
    (OUT / "dcase.json").write_text(json.dumps(results, indent=1))

    print("\nMachine   official AUC/pAUC   replica AUC/pAUC   stetho AUC/pAUC")
    for m, ids in results.items():
        agg = {k: (np.mean([ids[i][k]["auc"] for i in ids]), np.mean([ids[i][k]["pauc"] for i in ids]))
               for k in ("official", "replica", "stetho")}
        print(f"{m:8s}  " + "   ".join(f"{100*a:5.1f} / {100*p:5.1f}  " for a, p in agg.values()))


if __name__ == "__main__":
    main()
