"""Dense autoencoder (DCASE 2020 Task 2 baseline topology) + training loop.

    640 -> 128 -> 128 -> 128 -> 128 -> 8 -> 128 -> 128 -> 128 -> 128 -> 640

Trained on healthy sound only. Reconstruction error is the anomaly score:

    s(clip) = 1/T * sum_t || x_t - AE(x_t) ||^2 / D
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .features import DIM, stack_frames


def _block(i: int, o: int):
    return [nn.Linear(i, o), nn.BatchNorm1d(o), nn.ReLU()]


class AE(nn.Module):
    def __init__(self, d: int = DIM, h: int = 128, z: int = 8,
                 mean: np.ndarray | None = None, std: np.ndarray | None = None):
        super().__init__()
        layers = _block(d, h)
        for _ in range(3):
            layers += _block(h, h)
        layers += _block(h, z) + _block(z, h)
        for _ in range(3):
            layers += _block(h, h)
        layers.append(nn.Linear(h, d))
        self.net = nn.Sequential(*layers)
        # optional feature standardisation, baked into the exported graph
        self.register_buffer("mu", torch.zeros(d) if mean is None else torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("sd", torch.ones(d) if std is None else torch.tensor(std, dtype=torch.float32))

    def forward(self, x):
        xn = (x - self.mu) / self.sd
        return xn, self.net(xn)

    def errors(self, x):
        """Per-vector reconstruction error (what the ONNX graph outputs)."""
        xn, rec = self(x)
        return ((xn - rec) ** 2).mean(1)


class ErrorHead(nn.Module):
    """Export wrapper: input (N, 640) log-mel windows -> (N,) errors."""

    def __init__(self, ae: AE):
        super().__init__()
        self.ae = ae

    def forward(self, x):
        return self.ae.errors(x)


def vectors(clips: np.ndarray) -> np.ndarray:
    return np.concatenate([stack_frames(c.astype(np.float32)) for c in clips])


def train_ae(train_clips: np.ndarray, epochs: int = 60, batch: int = 512, lr: float = 1e-3,
             normalize: bool = True, seed: int = 0, verbose: bool = False) -> AE:
    torch.manual_seed(seed)
    np.random.seed(seed)
    X = vectors(train_clips)
    mean = X.mean(0) if normalize else None
    std = X.std(0) + 1e-3 if normalize else None
    ae = AE(mean=mean, std=std)
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Xt = torch.from_numpy(X)
    ae.train()
    for ep in range(epochs):
        perm = torch.randperm(len(Xt))
        tot = 0.0
        for i in range(0, len(Xt), batch):
            xb = Xt[perm[i:i + batch]]
            if len(xb) < 2:
                continue
            xn, rec = ae(xb)
            loss = ((xn - rec) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        sched.step()
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"  epoch {ep:3d} loss {tot / len(Xt):.4f}")
    ae.eval()
    return ae


@torch.no_grad()
def clip_scores(ae: AE, clips: np.ndarray, reduce: str = "mean") -> np.ndarray:
    out = []
    for c in clips:
        e = ae.errors(torch.from_numpy(stack_frames(c.astype(np.float32)))).numpy()
        if reduce == "mean":
            out.append(e.mean())
        elif reduce == "top10":
            out.append(np.mean(np.sort(e)[-max(1, len(e) // 10):]))
        else:
            raise ValueError(reduce)
    return np.array(out)
