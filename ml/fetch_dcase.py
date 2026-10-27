"""Stream a reproducible subset of DCASE 2020 Task 2 (MIMII-derived) dev data.

The full archives are ~4.4 GB. Zenodo honours HTTP Range requests, so we read
contiguous runs of WAVs out of the remote zip, convert them straight to log-mel and
keep only float16 features (~80 KB/clip instead of 320 KB of audio).

    python fetch_dcase.py --machines fan pump slider valve --train 300 --test 100
"""
from __future__ import annotations

import argparse
import re
import struct
import time
import zipfile
import zlib
from pathlib import Path

import numpy as np
import requests
from remotezip import RemoteZip

from stetho.features import load_wav, log_mel

URL = "https://zenodo.org/records/3678171/files/dev_data_{m}.zip?download=1"
OUT = Path(__file__).parent / "data" / "dcase"


def pick(infos, machine, split, label, mid, k):
    """First k clips of a group in archive order (deterministic, contiguous)."""
    pat = re.compile(rf"{machine}/{split}/{label}_id_{mid}_\d+\.wav$")
    sel = sorted((i for i in infos if pat.search(i.filename)), key=lambda i: i.header_offset)
    return sel[:k]


def fetch_run(url, infos, retries=8):
    """Download the byte span covering `infos` with ONE range request and
    decode every member from its local file header (Zenodo rate-limits
    requests, not bytes)."""
    start = infos[0].header_offset
    end = infos[-1].header_offset + 30 + len(infos[-1].filename) + 1024 + infos[-1].compress_size
    for a in range(retries):
        try:
            r = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=600)
            if r.status_code == 429:
                raise IOError("rate limited")
            r.raise_for_status()
            buf = r.content
            break
        except Exception as e:  # noqa: BLE001
            print("  retry", a, e, flush=True)
            time.sleep(min(60, 5 * 2 ** a))
    else:
        raise RuntimeError("download failed")
    out = []
    for i in infos:
        o = i.header_offset - start
        sig, = struct.unpack("<I", buf[o:o + 4])
        assert sig == 0x04034B50, "bad local header"
        nlen, xlen = struct.unpack("<HH", buf[o + 26:o + 30])
        data = buf[o + 30 + nlen + xlen: o + 30 + nlen + xlen + i.compress_size]
        if i.compress_type == zipfile.ZIP_DEFLATED:
            data = zlib.decompress(data, -15)
        out.append(log_mel(load_wav(data)).astype(np.float16))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machines", nargs="+", default=["fan", "pump", "slider", "valve"])
    ap.add_argument("--ids", nargs="+", default=["00", "02", "04", "06"])
    ap.add_argument("--train", type=int, default=300)
    ap.add_argument("--test", type=int, default=100, help="per label (normal/anomaly)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    for m in a.machines:
        url = URL.format(m=m)
        with RemoteZip(url) as z:
            infos = z.infolist()
        for mid in a.ids:
            dst = OUT / f"{m}_id{mid}.npz"
            if dst.exists():
                print("skip", dst.name); continue
            jobs = {
                "train": pick(infos, m, "train", "normal", mid, a.train),
                "test_normal": pick(infos, m, "test", "normal", mid, a.test),
                "test_anomaly": pick(infos, m, "test", "anomaly", mid, a.test),
            }
            if not jobs["train"]:
                print("no files for", m, mid); continue
            t0 = time.time()
            res = {}
            for key, lst in jobs.items():
                res[key] = np.stack(fetch_run(url, lst))
                res[key + "_names"] = np.array([i.filename for i in lst])
            np.savez_compressed(dst, **res)
            print(f"{m} id{mid}: " + ", ".join(f"{k}={v.shape[0]}" for k, v in res.items()
                                                 if not k.endswith('names'))
                  + f"  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
