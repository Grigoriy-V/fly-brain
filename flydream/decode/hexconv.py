"""Rung two of the ladder: a convolutional decoder on the hexagonal lattice.

It exists to answer one question about rung one: is the shape of the
decodability curve a property of the code, or an artefact of insisting the
decoder be linear? So it is deliberately the reference's own architecture
rather than a better one. `DecoderGAVP` (`flyvis/task/decoder.py:190`) is the
only trainable head flyvis ships and the one its pretrained ensemble was
trained with: scatter the 721 hexals into a 31x31 axial map, two hex-masked
5x5 convolutions with batch norm, softplus and dropout, then gather back.

Two properties of that head matter for reading any number it produces.

It is **purely per-frame**: frames are folded into the batch dimension, there
is no temporal filter, no recurrence and no state, so the only cross-frame
coupling is batch norm's own statistics during training. That makes it
comparable to ridge at lag 0 and to nothing else.

The axial map is **not a picture**. It is the sheared rhombus flyvis uses so a
square kernel can be masked into a hex shape; `Conv2dHexSpace` zeroes the
corners of its own weights. Do not read it as an image, and do not use it for
metrics: `hexraster.to_raster` is for that.

The head is small, a few hundred to a few thousand free parameters, but fitting
one per cell type for all 65 is not a local job (AGENTS.md: nothing trains
locally). `fit` here is for a handful of types as a check that the curve holds
its shape; the full sweep belongs on Modal beside roadmap item 3.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as nnf
from torch import nn


class HexDecoder(nn.Module):
    """DecoderGAVP's architecture, taking (samples, frames, channels, hexals) directly.

    Built on flyvis's own `Conv2dHexSpace`, so the hex masking of the kernel is
    the reference's and not a reimplementation, but without the connectome
    plumbing `DecoderGAVP.__init__` needs (it reads its channel count off
    `connectome.output_cell_types` and slices the full activity itself, which
    would force a proxy connectome for every choice of input cell type).
    """

    def __init__(self, in_channels: int, out_channels: int = 1, extent: int = 15,
                 shape=(8,), kernel_size: int = 5, p_dropout: float = 0.5,
                 batch_norm: bool = True, const_weight: float | None = None,
                 normalize_last: bool = True, activation: str = "Softplus"):
        super().__init__()
        from flyvis.task.decoder import Conv2dHexSpace
        from flyvis.utils.hex_utils import get_hex_coords

        pad = (kernel_size - 1) // 2
        u, v = get_hex_coords(extent)
        u, v = u - u.min(), v - v.min()
        self.register_buffer("u", torch.tensor(u, dtype=torch.long), persistent=False)
        self.register_buffer("v", torch.tensor(v, dtype=torch.long), persistent=False)
        self.rows, self.cols = int(u.max() + 1), int(v.max() + 1)
        self.out_channels, self.normalize_last = out_channels, normalize_last

        layers, c_in = [], in_channels
        for c in shape:
            layers.append(Conv2dHexSpace(c_in, c, kernel_size, const_weight=const_weight,
                                         padding=pad))
            if batch_norm:
                layers.append(nn.BatchNorm2d(c))
            layers.append(getattr(nn, activation)())
            if p_dropout:
                layers.append(nn.Dropout(p_dropout))
            c_in = c
        self.base = nn.Sequential(*layers)
        self.head = Conv2dHexSpace(c_in, out_channels + (1 if normalize_last else 0),
                                   kernel_size, const_weight=const_weight, padding=pad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, t, c_in, _ = x.shape
        hex_map = x.new_zeros(n, t, c_in, self.rows, self.cols)
        hex_map[:, :, :, self.u, self.v] = x
        out = self.head(self.base(hex_map.view(n * t, c_in, self.rows, self.cols)))
        if self.normalize_last:
            out = out[:, :self.out_channels] / (nnf.softplus(out[:, self.out_channels:]) + 1)
        return out.view(n, t, self.out_channels, self.rows, self.cols)[:, :, :, self.u, self.v]


def lattice_extent(n_cells: int) -> int:
    """The extent of the regular hex lattice with `n_cells` cells, or an error.

    A regular lattice holds 3k(k+1)+1 cells: 1, 7, 19, 37, ... 721 at extent 15.
    The check is not pedantry: two of flyvis's 65 cell types (Lawf1, Lawf2) are
    strided and have 123 cells, which is not a lattice, and scattering them as
    if they were silently mismatches the shape. Pad them onto the full lattice
    with `pad_to_lattice` first, or leave them to ridge, which needs no
    geometry at all.
    """
    k = 0
    while 3 * k * (k + 1) + 1 < n_cells:
        k += 1
    if 3 * k * (k + 1) + 1 != n_cells:
        raise ValueError(f"{n_cells} cells is not a regular hex lattice "
                         f"(nearest are {3 * (k - 1) * k + 1} and {3 * k * (k + 1) + 1}); "
                         "pad a strided cell type onto the full lattice first")
    return k


def pad_to_lattice(values: np.ndarray, u: np.ndarray, v: np.ndarray, extent: int,
                   fill: float = 0.0) -> np.ndarray:
    """Place a strided type's cells onto the full lattice of `extent`, `fill` elsewhere.

    `u`, `v` are that type's axial coordinates, in the same order as its cells.
    """
    from flyvis.utils.hex_utils import get_hex_coords

    gu, gv = get_hex_coords(extent)
    pos = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(gu, gv))}
    a = np.asarray(values)
    out = np.full((*a.shape[:-1], len(gu)), fill, dtype=np.float32)
    idx = [pos[(int(x), int(y))] for x, y in zip(u, v)]
    out[..., idx] = a
    return out


@dataclass
class Fitted:
    """A trained head and the loss curve that produced it."""

    module: HexDecoder
    x_mean: float
    x_std: float
    epochs: int
    extent: int = 15
    train_loss: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    best_val: float = float("nan")
    n_parameters: int = 0

    def predict(self, x: np.ndarray, batch: int = 64) -> np.ndarray:
        """x: (samples, frames, cells) for one cell type -> (rows, hexals)."""
        self.module.eval()
        xs = (np.asarray(x, dtype=np.float32) - self.x_mean) / self.x_std
        out = []
        with torch.no_grad():
            for i in range(0, len(xs), batch):
                chunk = torch.from_numpy(xs[i:i + batch])[:, :, None, :]
                out.append(self.module(chunk)[:, :, 0].numpy())
        y = np.concatenate(out, axis=0)
        return y.reshape(-1, y.shape[-1])


def fit(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
        *, epochs: int = 40, lr: float = 3e-3, batch: int = 8, seed: int = 0,
        shape=(8,), p_dropout: float = 0.5, weight_decay: float = 0.0,
        extent: int | None = None, verbose: bool = False) -> Fitted:
    """Train one head. Inputs are (samples, frames, cells); targets (samples, frames, hexals).

    The activity is standardised by one scalar mean and one scalar standard
    deviation taken from the training split, not per cell and not per cell type:
    a per-channel rescaling would equalise response amplitude across the
    lattice, which is the same intervention `ridge.py` refuses for the same
    reason. One scalar only makes the optimiser's step size comparable across
    cell types that differ by orders of magnitude in loudness.
    """
    torch.manual_seed(seed)
    xt = np.asarray(x_train, dtype=np.float32)
    extent = lattice_extent(xt.shape[-1]) if extent is None else extent
    mean, std = float(xt.mean()), float(xt.std()) or 1.0
    xtr = torch.from_numpy((xt - mean) / std)[:, :, None, :]
    ytr = torch.from_numpy(np.asarray(y_train, dtype=np.float32))
    xva = torch.from_numpy((np.asarray(x_val, dtype=np.float32) - mean) / std)[:, :, None, :]
    yva = torch.from_numpy(np.asarray(y_val, dtype=np.float32))

    net = HexDecoder(in_channels=1, out_channels=1, extent=extent, shape=shape,
                     p_dropout=p_dropout)
    from flyvis.utils.nn_utils import n_params

    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    best, best_state = float("inf"), None
    out = Fitted(module=net, x_mean=mean, x_std=std, epochs=epochs, extent=extent,
                 n_parameters=int(n_params(net).free))

    for ep in range(epochs):
        net.train()
        order = rng.permutation(len(xtr))
        losses = []
        for i in range(0, len(order), batch):
            sel = order[i:i + batch]
            opt.zero_grad()
            pred = net(xtr[sel])[:, :, 0]
            loss = nnf.mse_loss(pred, ytr[sel])
            loss.backward()
            opt.step()
            losses.append(float(loss))
        net.eval()
        with torch.no_grad():
            vl = float(nnf.mse_loss(net(xva)[:, :, 0], yva))
        out.train_loss.append(float(np.mean(losses)))
        out.val_loss.append(vl)
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in net.state_dict().items()}
        if verbose:
            print(f"    epoch {ep + 1}/{epochs} train {out.train_loss[-1]:.5f} val {vl:.5f}",
                  flush=True)

    if best_state is not None:
        net.load_state_dict(best_state)          # early stopping on the validation split
    out.best_val = best
    return out
