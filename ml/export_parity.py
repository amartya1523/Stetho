"""Write reference features for the JS parity test (web/tests/parity.test.js)."""
import json
from pathlib import Path

import numpy as np

from stetho.features import load_wav, log_mel
from stetho.physics import analyze

ROOT = Path(__file__).parent.parent
DEMO = ROOT / "web" / "public" / "demo"

ref = {}
base = analyze(load_wav(DEMO / "baseline.wav"))
hint = {"f1": base["f1_hz"], "blades": base["blades"]}
for name in ("baseline", "imbalance", "bearing", "blocked"):
    y = load_wav(DEMO / f"{name}.wav")
    lm = log_mel(y)
    ref[name] = {
        "logmel_head": lm[:4].round(4).tolist(),
        "logmel_mean": lm.mean(0).round(4).tolist(),
        "physics": {k: float(v) for k, v in analyze(y, None if name == "baseline" else hint).items()},
    }
out = ROOT / "web" / "tests" / "fixtures" / "parity_ref.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(ref))
print("wrote", out)
