"""Render every README figure from results/*.json and the demo audio."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from stetho.features import log_mel
from stetho.physics import analyze, band_analytic, envelope_spectrum, welch_db
from stetho.synth import random_params, synth_fan

ROOT = Path(__file__).parent.parent
RES = ROOT / "results"
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# validated categorical palette (dark surface), plus a neutral for reference series
TEAL, BLUE, AMBER, RED = "#0f9e89", "#5c6be0", "#c07f12", "#d6405a"
GRAY = "#6b7785"
BG, PANEL, INK, MUTED, GRID = "#070b10", "#0e151d", "#e6eef5", "#8796a5", "#1c2833"
HEAT = LinearSegmentedColormap.from_list("stetho", ["#070b10", "#0b3b3f", "#0f9e89", "#f0b030", "#ff5a6a"])
SEQ = LinearSegmentedColormap.from_list("seq", [PANEL, "#0d4f49", TEAL, "#8ff0de"])

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.titlecolor": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "font.family": "DejaVu Sans", "font.size": 10.5, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2,
})


def pool(f, s, k=8):
    """Max-pool a fine spectrum for display (peaks survive, noise thins out)."""
    n = len(s) // k
    return f[: n * k: k], s[: n * k].reshape(n, k).max(1)


def save(fig, name):
    fig.savefig(OUT / name, dpi=170, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", name)


def title(fig, t, sub=None):
    fig.text(0.01, 1.0, t, fontsize=15, fontweight="bold", va="bottom", ha="left")
    if sub:
        fig.text(0.01, 0.985, sub, fontsize=10.5, color=MUTED, va="top", ha="left")


# ---------------------------------------------------------------- signal anatomy
def signal_anatomy():
    p = random_params(7, "table_fan")
    conds = [("healthy", "Healthy"), ("imbalance", "Imbalance (coin on blade)"),
             ("looseness", "Looseness (loose screw)"), ("bearing", "Bearing wear")]
    p.seed = 2999
    base = analyze(synth_fan("healthy", p=p))
    hint = {"f1": base["f1_hz"], "blades": base["blades"]}
    fig, ax = plt.subplots(3, 4, figsize=(15, 9.2), gridspec_kw={"height_ratios": [1.15, 1, 1]})
    for j, (c, name) in enumerate(conds):
        p.seed, p.severity = 3000 + j, 1.0
        y = synth_fan(c, p=p)
        m = analyze(y, hint)
        f1 = m["f1_hz"]
        lm = log_mel(y)
        a = ax[0, j]
        a.imshow(lm.T, origin="lower", aspect="auto", cmap=HEAT, extent=[0, 10, 0, 128],
                 vmin=lm.max() - 75, vmax=lm.max())
        a.set_title(name, loc="left")
        a.grid(False)
        a.set_xlabel("time (s)")
        if j == 0:
            a.set_ylabel("log-mel band (AE input)")
        fp, pdb = welch_db(y.astype(np.float64))
        a = ax[1, j]
        k = fp <= 1200
        a.plot(fp[k], pdb[k], color=BLUE, lw=1.1)
        for h in range(1, 5):
            a.axvline(h * m["bpf_hz"], color=AMBER, lw=0.8, ls=":", alpha=0.8)
        a.set_xlim(0, 1200)
        a.set_ylim(pdb[k].max() - 70, pdb[k].max() + 5)
        a.set_xlabel("frequency (Hz)")
        if j == 0:
            a.set_ylabel("raw spectrum (dB)")
        a.text(0.98, 0.92, f"BPF {m['bpf_hz']:.0f} Hz", transform=a.transAxes, ha="right", color=AMBER, fontsize=9)
        fe, se = pool(*envelope_spectrum(band_analytic(y.astype(np.float64), 300, 3000)))
        a = ax[2, j]
        k = fe <= 11 * f1
        a.plot(fe[k] / f1, 20 * np.log10(se[k] + 1e-5), color=TEAL, lw=1.1)
        for h in range(1, 11):
            a.axvline(h, color=GRID, lw=0.8)
        a.set_xlim(0, 11)
        a.set_ylim(-60, -5)
        a.set_xlabel("shaft order  (frequency / 1×)")
        if j == 0:
            a.set_ylabel("envelope spectrum (dB)")
        a.text(0.98, 0.9, f"1× mod {20*np.log10(m['mod_1x']+1e-6):.0f} dB", transform=a.transAxes, ha="right",
               color=INK, fontsize=9)
    title(fig, "Anatomy of a fault: same fan, four conditions",
          "Top: what the autoencoder sees. Middle: raw spectrum, dominated by blade-pass tones (dotted). "
          "Bottom: envelope spectrum, where shaft-rate defects become visible.")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "signal_anatomy.png")


def why_envelope():
    p = random_params(7, "table_fan")
    p.seed = 2999
    yh = synth_fan("healthy", p=p).astype(np.float64)
    p.seed = 3001
    yi = synth_fan("imbalance", p=p).astype(np.float64)
    f1 = analyze(yh)["f1_hz"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    for y, col, lab in ((yh, GRAY, "healthy baseline"), (yi, RED, "coin taped to a blade")):
        fp, pdb = welch_db(y)
        k = fp <= 160
        ax[0].plot(fp[k], pdb[k], color=col, lw=1.4, label=lab)
        fe, se = pool(*envelope_spectrum(band_analytic(y, 300, 3000)))
        k = fe <= 160
        ax[1].plot(fe[k], 20 * np.log10(se[k] + 1e-5), color=col, lw=1.4, label=lab)
    for a in ax:
        for h in (1, 2, 3):
            a.axvline(h * f1, color=AMBER, ls=":", lw=1)
        a.set_xlabel("frequency (Hz)")
        a.text(f1, a.get_ylim()[1], " 1×", color=AMBER, va="top", fontsize=10)
    ax[0].set_title("Raw spectrum: 1× sits in the mic roll-off (unreliable on phones)", loc="left")
    ax[0].set_ylabel("dB")
    ax[1].set_title("Envelope spectrum of the 300–3000 Hz band: 1× stands out", loc="left")
    ax[1].legend(loc="upper right")
    fig.tight_layout()
    save(fig, "why_envelope.png")


# ---------------------------------------------------------------- benchmark
def dcase():
    f = RES / "dcase.json"
    if not f.exists():
        print("skip dcase (no results)")
        return
    R = json.loads(f.read_text())
    machines = list(R)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    w = 0.26
    x = np.arange(len(machines))
    for i, (key, col, lab) in enumerate((("official", GRAY, "official DCASE baseline"),
                                         ("replica", BLUE, "baseline re-implemented"),
                                         ("stetho", TEAL, "Stetho AE"))):
        for k, metric in enumerate(("auc", "pauc")):
            vals = [100 * np.mean([R[m][i_][key][metric] for i_ in R[m]]) for m in machines]
            bars = ax[k].bar(x + (i - 1) * w, vals, w - 0.03, color=col, label=lab, zorder=3)
            for b, v in zip(bars, vals):
                ax[k].text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.0f}", ha="center", fontsize=8.5, color=MUTED)
    for k, metric in enumerate(("AUC", "pAUC (FPR ≤ 0.1)")):
        ax[k].set_xticks(x, machines)
        ax[k].set_ylim(40, 100)
        ax[k].set_ylabel(f"{metric} %")
        ax[k].set_title(f"{metric} by machine type (mean over 4 IDs)", loc="left")
        ax[k].grid(axis="x", visible=False)
    ax[0].legend(loc="upper left", ncol=1)
    fig.tight_layout()
    save(fig, "dcase_auc.png")

    fig, ax = plt.subplots(1, len(machines), figsize=(4 * len(machines), 3.9), sharey=True)
    for a, m in zip(np.atleast_1d(ax), machines):
        for i_, col in zip(R[m], (TEAL, BLUE, AMBER, RED)):
            roc = R[m][i_]["stetho"]["roc"]
            a.plot(roc["fpr"], roc["tpr"], color=col, lw=1.6, label=f"id {i_}  AUC {100*R[m][i_]['stetho']['auc']:.0f}")
        a.plot([0, 1], [0, 1], color=GRID, ls="--", lw=1)
        a.set_title(m, loc="left")
        a.set_xlabel("false positive rate")
        a.legend(loc="lower right", fontsize=8.5)
        a.set_aspect("equal")
    np.atleast_1d(ax)[0].set_ylabel("true positive rate")
    fig.tight_layout()
    save(fig, "roc_curves.png")


# ---------------------------------------------------------------- pipeline eval
def pipeline():
    P = json.loads((RES / "pipeline.json").read_text())
    conf = np.array(P["confusion"])
    rows = ["healthy", "imbalance", "misalignment", "looseness", "bearing", "blocked"]
    cols = ["healthy", "imbal.", "misalign.", "loose", "bearing", "airflow", "anomaly"]
    norm = conf / conf.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    ax.imshow(norm, cmap=SEQ, vmin=0, vmax=1)
    ax.grid(False)
    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            if conf[i, j]:
                ax.text(j, i, f"{100*norm[i,j]:.0f}", ha="center", va="center", fontsize=10,
                        color=BG if norm[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(cols)), cols, rotation=0)
    ax.set_yticks(range(len(rows)), rows)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true condition")
    ax.set_title(f"Full pipeline on 12 unseen machines · accuracy {100*P['accuracy']:.0f}%", loc="left")
    save(fig, "confusion.png")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.1))
    a = ax[0]
    names = {"imbalance": "imbalance", "misalignment": "misalignment", "looseness": "looseness",
             "bearing": "bearing", "blocked": "airflow"}
    for (c, v), col, mk in zip(P["severity_sweep"].items(), (RED, BLUE, AMBER, TEAL, "#9aa7b4"),
                               ("o", "s", "^", "D", "v")):
        s, h = zip(*v)
        a.plot(s, np.array(h) * 100, color=col, marker=mk, ms=6, label=names[c])
    a.set_xlabel("fault severity (simulator units)")
    a.set_ylabel("detected %")
    a.set_title("How early is a fault caught?", loc="left")
    a.legend(fontsize=9, loc="lower right")
    a = ax[1]
    snr = P["snr"]
    xs = [d["snr"] for d in snr]
    a.plot(xs, [100 * d["accuracy"] for d in snr], color=TEAL, marker="o", ms=7, label="6-class accuracy")
    a.plot(xs, [100 * d["false_alarm"] for d in snr], color=RED, marker="s", ms=7, label="false alarms")
    a.invert_xaxis()
    a.set_xlabel("background noise SNR (dB)  →  noisier")
    a.set_ylabel("%")
    a.set_ylim(0, 100)
    a.set_title("Robustness to a noisy room", loc="left")
    a.legend(fontsize=9)
    a = ax[2]
    ab = P["ablation"]
    labels = ["autoencoder\nonly", "physics rules\nonly", "fused\n(Stetho)"]
    vals = [ab["ae_only_auc"], ab["physics_only_auc"], ab["fused_auc"]]
    bars = a.bar(labels, np.array(vals) * 100, color=[BLUE, AMBER, TEAL], width=0.55, zorder=3)
    for b, v in zip(bars, vals):
        a.text(b.get_x() + b.get_width() / 2, v * 100 + 1, f"{v:.3f}", ha="center", color=INK)
    a.set_ylim(50, 105)
    a.set_ylabel("healthy-vs-fault AUC %")
    a.set_title("Ablation: which evidence matters", loc="left")
    a.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "robustness.png")

    T = P["timeline"]
    fig, ax = plt.subplots(figsize=(10, 3.4))
    d = [t["day"] for t in T]
    h = [t["health"] for t in T]
    ax.axhspan(80, 100, color=TEAL, alpha=0.08, lw=0)
    ax.axhspan(50, 80, color=AMBER, alpha=0.08, lw=0)
    ax.axhspan(0, 50, color=RED, alpha=0.08, lw=0)
    ax.plot(d, h, color=INK, lw=1.6, zorder=2)
    cols_ = [TEAL if v >= 80 else AMBER if v >= 50 else RED for v in h]
    ax.scatter(d, h, c=cols_, s=55, zorder=3, edgecolors=BG, linewidths=2)
    first_bad = next(i for i, t in enumerate(T) if t["label"] != "healthy")
    ax.annotate(f"day {first_bad}: imbalance flagged", (d[first_bad], h[first_bad]), (d[first_bad] + 1.2, 85),
                color=INK, arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.set_xlabel("day")
    ax.set_ylabel("health")
    ax.set_ylim(0, 102)
    ax.set_title("A fan slowly going out of balance (dust build-up, simulated)", loc="left")
    save(fig, "timeline.png")

    recs = P["records"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    zh = [r["z"] for r in recs if r["cond"] == "healthy"]
    zf = [r["z"] for r in recs if r["cond"] != "healthy"]
    bins = np.linspace(-4, 30, 45)
    ax.hist(np.clip(zh, -4, 30), bins, color=TEAL, alpha=0.85, label="healthy checks")
    ax.hist(np.clip(zf, -4, 30), bins, color=RED, alpha=0.6, label="faulty checks")
    ax.axvline(6, color=AMBER, ls="--", lw=1.4)
    ax.text(6.3, ax.get_ylim()[1] * 0.9, "z₀ = 6  (AE health = 50)", color=AMBER)
    ax.set_xlabel("autoencoder error z-score vs. this machine's baseline")
    ax.set_ylabel("checks")
    ax.legend()
    ax.set_title("The autoencoder alone separates loud faults, not subtle ones", loc="left")
    save(fig, "ae_scores.png")


if __name__ == "__main__":
    signal_anatomy()
    why_envelope()
    dcase()
    pipeline()
