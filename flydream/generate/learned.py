"""13A, the amortised inversion: networks "state → video" on the eye's lattice.

Input: the activity of a set of cell types over `T` frames, laid on the 721
columns as channels (B, T, K, 721) — the columnar types are in lattice order
in model zero (checked 2026-09-19), Tm5a's 251 cells are scattered to their
columns with zeros elsewhere. Output: the video (B, T_out, 721), T_out = the
shown frames; the `margin` frames after them are the temporal context.

Two models, the same lattice operations:

- `LinearHexTemporal`: one linear kernel shared over columns — for each
  input type, its value at the column and its 1- or 2-ring neighbours at
  `taps` frames after the frame — plus a bias. Translation-shared, a few
  hundred to a few thousand weights, no nonlinearity. The baseline that
  says whether a nonlinear learned inverse is needed at all.
- `HexTemporalCNN`: the same kernel as a layer, stacked with ReLU and a
  channel width, ~10^5 weights.

Conditions (`CONDITIONS`): `early` (L1, L3), `deep` (T4a-d, T5a-d), `all`
(the 14 ladder types). Training: Adam, MSE on the video, inputs z-scored per
type from the training set. Evaluation: r per frame to the video on the
held-out clips, and the round trip through the frozen brain (`round_trip`):
the generated video is simulated and its state compared with the target per
type, in the per-type normalised error of item 12 — beside the same error
of the Adam inversion on the same clips.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from flydream.decode.hexraster import neighbour_index

LADDER = ["L1", "L3", "Mi1", "Mi4", "Tm5a", "Tm9", "T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"]
CONDITIONS = {"early": ["L1", "L3"], "deep": ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"], "all": LADDER}


# ----------------------------------------------------------------- lattice


def ring_index(rings: int = 1, n: int = 721) -> np.ndarray:
    """(n, m) indices of each column's neighbourhood: itself, then its
    neighbours within `rings` steps; -1 where the lattice ends."""
    nb = neighbour_index(n)
    sets = [{i} for i in range(n)]
    frontier = [{i} for i in range(n)]
    for _ in range(rings):
        new = []
        for i in range(n):
            f = set()
            for j in frontier[i]:
                f.update(int(k) for k in nb[j] if k >= 0)
            f -= sets[i]
            sets[i] |= f
            new.append(f)
        frontier = new
    m = max(len(s) for s in sets)
    out = np.full((n, m), -1, np.int64)
    for i in range(n):
        order = [i] + sorted(sets[i] - {i})
        out[i, :len(order)] = order
    return out


def gather_hex(x: torch.Tensor, ring: torch.Tensor) -> torch.Tensor:
    """(B, T, K, n) -> (B, T, K, n, m): each column's neighbourhood values, 0 off-lattice."""
    valid = (ring >= 0).to(x.dtype)
    g = x[..., ring.clamp(min=0)]
    return g * valid


class HexTemporalLayer(nn.Module):
    """Linear map (B, T, K, n) -> (B, T, C, n): for every column, its ring
    neighbourhood of every input channel at `taps` frames from t on."""

    def __init__(self, k_in: int, c_out: int, ring: np.ndarray, taps: int):
        super().__init__()
        self.register_buffer("ring", torch.as_tensor(ring))
        self.taps, self.m = taps, ring.shape[1]
        self.weight = nn.Parameter(torch.randn(c_out, k_in, self.m, taps) * (1.0 / np.sqrt(k_in * self.m * taps)))
        self.bias = nn.Parameter(torch.zeros(c_out))

    def forward(self, x: torch.Tensor, t_out: int) -> torch.Tensor:
        B, T, K, n = x.shape
        g = gather_hex(x, self.ring)                                    # (B, T, K, n, m)
        # the taps: frames t .. t+taps-1 for each output frame t < t_out (edge-padded at the end)
        pad = max(0, t_out + self.taps - 1 - T)
        if pad:
            g = torch.cat([g, g[:, -1:].expand(B, pad, K, n, self.m)], 1)
        g = g.unfold(1, self.taps, 1)[:, :t_out]                        # (B, t_out, K, n, m, taps)
        return torch.einsum("btknmp,ckmp->btcn", g, self.weight) + self.bias[None, None, :, None]


class LinearHexTemporal(nn.Module):
    def __init__(self, k_in: int, rings: int = 1, taps: int = 5):
        super().__init__()
        self.layer = HexTemporalLayer(k_in, 1, ring_index(rings), taps)

    def forward(self, x: torch.Tensor, t_out: int) -> torch.Tensor:
        return self.layer(x, t_out)[:, :, 0]


class HexTemporalCNN(nn.Module):
    def __init__(self, k_in: int, width: int = 32, depth: int = 3, rings: int = 1, taps: int = 5):
        super().__init__()
        ring = ring_index(rings)
        self.first = HexTemporalLayer(k_in, width, ring, taps)
        self.mid = nn.ModuleList([HexTemporalLayer(width, width, ring, 1) for _ in range(depth - 1)])
        self.last = HexTemporalLayer(width, 1, ring, 1)

    def forward(self, x: torch.Tensor, t_out: int) -> torch.Tensor:
        h = F.relu(self.first(x, t_out))
        for m in self.mid:
            h = F.relu(m(h, t_out)) + h
        return self.last(h, t_out)[:, :, 0]


# ----------------------------------------------------------------- data


def channel_layout(manifest: dict, columns: dict, types: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For `types`: (cell positions in the stored state, channel index, column index) to scatter
    the stored (N, T, cells) states into (N, T, K, 721)."""
    type_of = np.asarray(manifest["type_of_cell"])
    pos, ch, col = [], [], []
    for k, t in enumerate(types):
        where = np.where(type_of == t)[0]
        cols = np.asarray(columns["column_of_cell"][t])
        assert len(where) == len(cols), (t, len(where), len(cols))
        pos.append(where); ch.append(np.full(len(where), k)); col.append(cols)
    return np.concatenate(pos), np.concatenate(ch), np.concatenate(col)


def to_maps(states: np.ndarray, layout, k: int) -> np.ndarray:
    """(N, T, cells) float16 -> (N, T, K, 721) float16 (zeros where a type has no cell)."""
    pos, ch, col = layout
    out = np.zeros((states.shape[0], states.shape[1], k, 721), np.float16)
    out[:, :, ch, col] = states[:, :, pos]
    return out


class ShardSet:
    """The shards of a pairs13 run, loaded into memory as maps for one condition."""

    def __init__(self, root: Path, manifest: dict, columns: dict, types: list[str], indices: list[int]):
        self.types, self.layout = types, channel_layout(manifest, columns, types)
        want = set(indices)
        xs, ys, ids = [], [], []
        for name in manifest["shards"]:
            z = np.load(root / name)
            keep = np.array([i for i, gi in enumerate(z["index"]) if int(gi) in want])
            if len(keep) == 0:
                continue
            xs.append(to_maps(z["states"][keep], self.layout, len(types)))
            ys.append(z["videos"][keep])
            ids.append(z["index"][keep])
        self.x, self.y, self.index = np.concatenate(xs), np.concatenate(ys), np.concatenate(ids)

    def stats(self):
        """Per-type mean and sd over the set, from a float32 pass in chunks."""
        k = self.x.shape[2]
        n = 0; s1 = np.zeros(k); s2 = np.zeros(k)
        for i in range(0, len(self.x), 256):
            xb = self.x[i:i + 256].astype(np.float32)
            s1 += xb.sum((0, 1, 3)); s2 += (xb ** 2).sum((0, 1, 3)); n += xb.shape[0] * xb.shape[1] * xb.shape[3]
        m = (s1 / n).astype(np.float32); sd = np.sqrt(np.maximum(s2 / n - m ** 2, 0)).astype(np.float32) + 1e-6
        return m.reshape(1, 1, k, 1), sd.reshape(1, 1, k, 1)


def pixcorr(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """(N, T, 721) x2 -> (N,) mean per-frame correlation."""
    p = pred - pred.mean(-1, keepdims=True); t = true - true.mean(-1, keepdims=True)
    num = (p * t).sum(-1); den = np.sqrt((p ** 2).sum(-1) * (t ** 2).sum(-1)) + 1e-9
    return (num / den).mean(1)


# ----------------------------------------------------------------- training


def train_model(model: nn.Module, train: ShardSet, val: ShardSet, *, t_out: int, epochs: int, batch: int, lr: float,
                device, log=print) -> dict:
    m, s = train.stats()
    m_t, s_t = torch.as_tensor(m, device=device), torch.as_tensor(s, device=device)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = len(train.x)
    hist = []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(n)
        tot = 0.0
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            x = (torch.as_tensor(train.x[b].astype(np.float32), device=device) - m_t) / s_t
            y = torch.as_tensor(train.y[b, :t_out].astype(np.float32), device=device)
            loss = F.mse_loss(model(x, t_out), y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += float(loss) * len(b)
        sched.step()
        vr = evaluate(model, val, m, s, t_out, device, batch)["r"].mean()
        hist.append({"epoch": ep, "train_mse": tot / n, "val_r": float(vr), "seconds": round(time.time() - t0)})
        log(f"  epoch {ep:2d}  train mse {tot / n:.5f}  val r {vr:.3f}  {time.time() - t0:.0f}s")
    return {"history": hist, "mean": m, "std": s}


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, m, s, t_out: int, device, batch: int = 32) -> np.ndarray:
    model.eval()
    m_t, s_t = torch.as_tensor(m, device=device), torch.as_tensor(s, device=device)
    out = []
    for i in range(0, len(x), batch):
        xb = (torch.as_tensor(x[i:i + batch].astype(np.float32), device=device) - m_t) / s_t
        out.append(model(xb, t_out).clamp(0, 1).cpu().numpy())
    return np.concatenate(out)


def evaluate(model: nn.Module, data: ShardSet, m, s, t_out: int, device, batch: int = 32) -> dict:
    pred = predict(model, data.x, m, s, t_out, device, batch)
    return {"r": pixcorr(pred, data.y[:, :t_out].astype(np.float32)), "pred": pred}


# ----------------------------------------------------------------- the round trip


def round_trip_error(net, videos: np.ndarray, target_states: np.ndarray, cells: np.ndarray, type_of: np.ndarray,
                     types: list[str], dt: float, t_pre: float, margin: int) -> dict:
    """Simulate `videos` (N, T_out, 721) through the frozen brain (the last frame
    held for the margin) and compare the state of `types` with `target_states`
    (N, T, cells): the per-type normalised squared error of item 12, mean over
    types, per clip."""
    from flydream.generate.invert import device_of, simulate
    from flydream.generate.pairs13 import simulate_states

    dev = device_of(net)
    vids = np.concatenate([videos, np.repeat(videos[:, -1:], margin, 1)], 1).astype(np.float16) if margin else videos
    st = simulate_states(net, vids, cells, dt, t_pre, 16).astype(np.float32)
    tg = target_states.astype(np.float32)
    errs = {}
    for t in types:
        sel = np.where(type_of == t)[0]
        var = tg[:, :, sel].var() + 1e-6
        errs[t] = (((st[:, :, sel] - tg[:, :, sel]) ** 2).mean((1, 2)) / var)
    per_clip = np.mean([errs[t] for t in types], 0)
    return {"per_type": {t: float(errs[t].mean()) for t in types}, "per_clip": per_clip, "state": st}
