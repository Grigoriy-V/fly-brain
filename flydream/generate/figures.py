"""The inversion ladder: one clip, the video recovered from each stage side by side.

    python -m flydream.generate.figures --tags 2026-09-18_invert_R1_s3 2026-09-18_invert_L1_s3 ... --out 2026-09-18_inversion_ladder

Reads `data/generate/<tag>/recovered.npz` written by `flydream.generate.invert`
(the same clip and frames in every tag) and draws rows = frames, columns =
the true video then one column per tag (the cell types in its meta.json),
each column headed by its PixCorr against the true video and the wrong-target
control's; a GIF with the same columns. The ridge ladder of item 4
(`flydream.decode.figures`) is the picture this one is compared with: same
clip, same stages, a decoder trained on 144 clips there against an
optimiser that saw only this clip's state here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from flydream.model import ROOT
from flydream.decode.hexraster import to_raster

INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def load(tags: list[str]):
    cols = []
    true = None
    for tag in tags:
        d = ROOT / "data" / "generate" / tag
        z = np.load(d / "recovered.npz")
        m = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        if true is None:
            true = z["true"]
        label = ", ".join(m["types"]) if len(m["types"]) <= 2 else f"{m['types'][0]}…{m['types'][-1]} ({len(m['types'])} types)"
        cols.append({"tag": tag, "label": label, "video": z["recovered"], "control": z["control"],
                     "r": float(np.mean(z["s_rec"])), "r_ctrl": float(np.mean(z["s_ctrl"])), "cells": m["cells"]})
    return true, cols


def ladder(true, cols, frames, path: Path, title: str, pix_per_hex: int = 4, with_control: bool = True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows = len(frames) + (1 if with_control else 0)
    n_cols = 1 + len(cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.75 * n_cols + 0.3, 1.7 * n_rows + 1.0),
                             facecolor=SURFACE, squeeze=False)
    def show(ax, vec):
        ax.imshow(to_raster(vec, len(vec), pix_per_hex, fill=np.nan), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    for i, f in enumerate(frames):
        show(axes[i, 0], true[f])
        axes[i, 0].set_ylabel(f"frame {f}", fontsize=8, color=INK2)
        for j, c in enumerate(cols, 1):
            show(axes[i, j], c["video"][f])
    axes[0, 0].set_title("stimulus\n(what the eye saw)", fontsize=8, color=INK)
    for j, c in enumerate(cols, 1):
        axes[0, j].set_title(f"{c['label']}\nr = {c['r']:+.2f} (control {c['r_ctrl']:+.2f})", fontsize=7.5, color=INK)
    if with_control:
        f = frames[len(frames) // 2]
        axes[-1, 0].axis("off")
        axes[-1, 0].text(0.5, 0.5, f"wrong-target\ncontrol, frame {f}", ha="center", va="center", fontsize=8, color=INK2,
                         transform=axes[-1, 0].transAxes)
        for j, c in enumerate(cols, 1):
            show(axes[-1, j], c["control"][f])
    fig.suptitle(title, fontsize=9.5, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, facecolor=SURFACE)
    plt.close(fig)


def clip(true, cols, path: Path, pix_per_hex: int = 4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    vids = [true] + [c["video"] for c in cols]
    names = ["stimulus"] + [c["label"] for c in cols]
    fig, axes = plt.subplots(1, len(vids), figsize=(1.9 * len(vids), 2.4), facecolor=SURFACE)
    ims = []
    for ax, v, n in zip(np.atleast_1d(axes), vids, names):
        ims.append(ax.imshow(to_raster(v[0], v.shape[-1], pix_per_hex, fill=np.nan), cmap="gray", vmin=0, vmax=1,
                             interpolation="nearest"))
        ax.set_title(n, fontsize=7.5, color=INK); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.tight_layout()

    def update(f):
        for im, v in zip(ims, vids):
            im.set_data(to_raster(v[f], v.shape[-1], pix_per_hex, fill=np.nan))
        return ims
    FuncAnimation(fig, update, frames=len(true), blit=True).save(path, writer=PillowWriter(fps=8))
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--out", default="inversion_ladder")
    p.add_argument("--frames", nargs="*", type=int, default=[2, 9, 17])
    p.add_argument("--model", default="")
    a = p.parse_args(argv)
    true, cols = load(a.tags)
    frames = [f for f in a.frames if f < len(true)]
    figdir = ROOT / "reports" / "figures"
    ladder(true, cols, frames, figdir / f"{a.out}.png",
           f"Encoder inversion, stage by stage: the video most compatible with each stage's state"
           + (f"  ({a.model})" if a.model else ""))
    clip(true, cols, figdir / f"{a.out}.gif")
    for c in cols:
        print(f"{c['label']:<28} {c['cells']:>6} cells  r {c['r']:+.3f}  control {c['r_ctrl']:+.3f}")
    print(f"wrote reports/figures/{a.out}.png and .gif")
    return 0


if __name__ == "__main__":
    sys.exit(main())
