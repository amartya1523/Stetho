"""Log-mel front end, identical to the DCASE 2020 Task 2 baseline.

The browser re-implements exactly this in web/src/dsp/mel.js; tests/test_parity
checks both produce the same numbers on the same WAV.
"""
from __future__ import annotations

import io
import sys

import librosa
import numpy as np
import soundfile as sf

SR = 16000
N_FFT = 1024
HOP = 512
N_MELS = 128
CONTEXT = 5                       # stacked frames per AE input vector
DIM = N_MELS * CONTEXT            # 640
EPS = sys.float_info.epsilon


def load_wav(src, sr: int = SR) -> np.ndarray:
    if isinstance(src, (bytes, bytearray)):
        y, fs = sf.read(io.BytesIO(src), dtype="float32", always_2d=True)
    else:
        y, fs = sf.read(src, dtype="float32", always_2d=True)
    y = y[:, 0]
    if fs != sr:
        y = librosa.resample(y, orig_sr=fs, target_sr=sr)
    return y.astype(np.float32)


def log_mel(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """(frames, 128) log-mel in dB: 10*log10(mel + eps)."""
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP,
                                         n_mels=N_MELS, power=2.0, pad_mode="constant")
    return (10.0 * np.log10(mel + EPS)).T.astype(np.float32)


def stack_frames(lm: np.ndarray, context: int = CONTEXT) -> np.ndarray:
    """(frames, 128) -> (frames-context+1, 640) sliding-window vectors."""
    n = lm.shape[0] - context + 1
    idx = np.arange(context)[None, :] + np.arange(n)[:, None]
    return lm[idx].reshape(n, -1)
