"""The generator's picture for the human: two inputs, one row each, the eye's
view first and the video recovered from each stage to its right.

    python tools/fig_generator_two_inputs.py --prefix 2026-09-19_malecns_invert --sample 3 \
        --out 2026-09-19_malecns_generator_two_inputs

Reads `data/generate/<prefix>_<stage>_s<sample>/recovered.npz` (written by
`deploy/modal/generate_app.py` or `flydream.generate.invert`). Row A is the
clip itself and the videos inverted from its state; row B is the wrong-target
control of the same runs, shown with its own input (the control clip named
in meta.json), so the control is a second input row and not a footer
(AGENTS "Artefacts for the human"). Writes a PNG (one frame) and a GIF (the
whole clip, every frame, 8 fps: 40 frames at 20 ms are 0.8 s of the fly's
time shown 6× slower). This is the ad-hoc drawing of 2026-09-19 made
permanent (ROADMAP item 9).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flydream.decode.hexraster import to_raster  # noqa: E402

STAGES = ["R1", "L1", "L3", "Mi1", "Mi4", "Tm5a", "Tm9", "T4a", "T5a", "T4T5"]
SURFACE = "#fcfcfb"


def corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean([np.corrcoef(x, y)[0, 1] for x, y in zip(a, b)]))


def load(prefix: str, sample: int, stages: list[str]):
    cols, meta = [], None
    for s in stages:
        d = ROOT / "data" / "generate" / f"{prefix}_{s}_s{sample}"
        if not d.exists():
            print(f"  {d.name}: missing, skipped"); continue
        z = np.load(d / "recovered.npz")
        m = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        meta = meta or m
        cols.append({"label": s if s != "T4T5" else "T4+T5 (8 types)", "rec": z["recovered"], "ctrl": z["control"],
                     "true": z["true"], "cells": m["cells"]})
    return cols, meta


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", required=True, help="e.g. 2026-09-19_malecns_invert")
    p.add_argument("--sample", type=int, default=3)
    p.add_argument("--stages", nargs="*", default=STAGES)
    p.add_argument("--out", required=True, help="figure name without extension, under reports/figures/")
    p.add_argument("--frame", type=int, default=None, help="frame of the PNG; default: the middle one")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--brain", default=None, help="label; default from the prefix")
    a = p.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from flydream.generate.invert import clip_from_sintel

    cols, meta = load(a.prefix, a.sample, a.stages)
    if not cols:
        print("nothing to draw"); return 1
    face = cols[0]["true"]
    n, dt = face.shape[0], float(meta.get("dt", 0.02))
    ctrl_sample = int(meta["control_sample"])
    other = clip_from_sintel(ctrl_sample, n, dt)
    brain = a.brain or ("MaleCNS (model zero: our wiring, FlyVis parameters transplanted, untrained)"
                        if "malecns" in a.prefix else "FlyVis (flow/0000/000)")
    rows = [(f"ВХОД A: клип {a.sample}", face, "rec"), (f"ВХОД B: клип {ctrl_sample}", other, "ctrl")]
    f0 = a.frame if a.frame is not None else n // 2
    margin = meta.get("margin", 0)
    window = f"{n} кадров по {dt * 1000:.0f} мс = {n * dt:.1f} с мухи" + (f", запас {margin} кадров отброшен" if margin else "")

    def rs(v, f):
        return to_raster(v[f], 721, 4, fill=np.nan)

    def style(ax):
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    figdir = ROOT / "reports" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    W = 1 + len(cols)
    fig, ax = plt.subplots(2, W, figsize=(1.75 * W + 0.6, 4.9), facecolor=SURFACE)
    for i, (name, true, key) in enumerate(rows):
        ax[i, 0].imshow(rs(true, f0), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax[i, 0].set_title(f"{name}\n(что видел глаз, кадр {f0})", fontsize=8)
        for j, c in enumerate(cols, 1):
            ax[i, j].imshow(rs(c[key], f0), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax[i, j].set_title(f"из {c['label']}\nr = {corr(c[key], true):+.2f}", fontsize=8)
        for x in ax[i]:
            style(x)
    fig.suptitle(f"Генератор (инверсия энкодера), мозг {brain}. Строка: вход -> состояние стадии -> выход; "
                 f"r — совпадение выхода со входом строки. {window}", fontsize=8.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(figdir / f"{a.out}.png", dpi=170, facecolor=SURFACE)
    plt.close(fig)

    fig, ax = plt.subplots(2, W, figsize=(1.55 * W + 0.4, 4.0), facecolor=SURFACE)
    ims = []
    for i, (name, true, key) in enumerate(rows):
        vids = [true] + [c[key] for c in cols]
        names = [name + "\nчто видел глаз"] + [f"генератор из {c['label']}" for c in cols]
        for j, (v, nm) in enumerate(zip(vids, names)):
            ims.append((ax[i, j].imshow(rs(v, 0), cmap="gray", vmin=0, vmax=1, interpolation="nearest"), v))
            ax[i, j].set_title(nm, fontsize=7.5); style(ax[i, j])
    fig.suptitle(f"Генератор (инверсия), мозг {brain.split(' (')[0]}: вход -> состояние стадии -> выход; "
                 f"{window}, показано в {1 / (a.fps * dt):.0f}x медленнее", fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    def upd(f):
        for im, v in ims:
            im.set_data(rs(v, f))
        return [im for im, _ in ims]

    FuncAnimation(fig, upd, frames=n, blit=True).save(figdir / f"{a.out}.gif", writer=PillowWriter(fps=a.fps))
    plt.close(fig)
    print(f"wrote reports/figures/{a.out}.png and .gif ({n} frames, {len(cols)} stages, control clip {ctrl_sample})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
