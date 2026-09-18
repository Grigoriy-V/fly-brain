"""Manipulated-state pictures (ROADMAP item 12): one stacked clip per scenario.

    python tools/fig_mix.py --prefix 2026-09-19_malecns

Reads `data/generate/<prefix>_mix_<task>/recovered.npz` written by
`flydream.generate.mix`. Per scenario one GIF and one PNG, one row: the two
clips the states came from ("вход A", "вход B"; the averaged video too for
the interpolation), then one column per task with its r to A and r to B
under it, so the winner of a contest is a number. The control columns
(× 1, all-from-A, all-from-B, the averaged video's state) are the
unmanipulated states beside the manipulated ones.
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

SURFACE = "#fcfcfb"
SCENARIOS = {
    "C": ("Правка усиления: состояние клипа A на всех типах лестницы, один тип умножен",
          ["C_x1", "C_T4a_x0.5", "C_T4a_x2", "C_T5_x0", "C_Mi4_x1.5"],
          {"C_x1": "× 1 (контроль)", "C_T4a_x0.5": "T4a × 0.5", "C_T4a_x2": "T4a × 2", "C_T5_x0": "T5 × 0",
           "C_Mi4_x1.5": "Mi4 × 1.5"}),
    "A": ("Интерполяция: состояние T4a = α·A + (1−α)·B, читается только T4a",
          ["A_T4a_a0", "A_T4a_a0.25", "A_T4a_a0.5", "A_T4a_a0.75", "A_T4a_a1", "A_T4a_avgvideo"],
          {"A_T4a_a0": "α = 0 (B)", "A_T4a_a0.25": "α = 0.25", "A_T4a_a0.5": "α = 0.5", "A_T4a_a0.75": "α = 0.75",
           "A_T4a_a1": "α = 1 (A)", "A_T4a_avgvideo": "состояние от\nусреднённого видео"}),
    "B": ("Гибрид: L1/L3 от одного клипа, T4/T5 от другого",
          ["B_allA", "B_earlyA_motionB", "B_earlyB_motionA", "B_allB"],
          {"B_allA": "всё от A (контроль)", "B_earlyA_motionB": "L1/L3 ← A\nT4/T5 ← B",
           "B_earlyB_motionA": "L1/L3 ← B\nT4/T5 ← A", "B_allB": "всё от B (контроль)"}),
}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", required=True)
    p.add_argument("--scenarios", nargs="*", default=["C", "A", "B"])
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--data", default=str(ROOT / "data" / "generate"))
    p.add_argument("--outdir", default=str(ROOT / "reports" / "figures"))
    a = p.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    def rs(v, f):
        return to_raster(v[f], 721, 4, fill=np.nan)

    def style(ax):
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    figdir = Path(a.outdir)
    figdir.mkdir(parents=True, exist_ok=True)
    for sc in a.scenarios:
        head, names, labels = SCENARIOS[sc]
        cols, meta = [], None
        for nm in names:
            d = Path(a.data) / f"{a.prefix}_mix_{nm}"
            if not d.exists():
                print(f"  {d.name}: missing, skipped"); continue
            z = np.load(d / "recovered.npz")
            m = json.loads((d / "meta.json").read_text(encoding="utf-8"))
            meta = meta or (m | {"clip_a": z["clip_a"], "clip_b": z["clip_b"], "clip_avg": z["clip_avg"]})
            cols.append({"label": labels[nm], "video": z["recovered"], "r_a": m["r_a"], "r_b": m["r_b"], "r_avg": m["r_avg"]})
        if not cols:
            print(f"{sc}: nothing to draw"); continue
        n, dt = len(meta["clip_a"]), float(meta["dt"])
        lead = [(f"вход A: клип {meta['sample_a']}", meta["clip_a"]), (f"вход B: клип {meta['sample_b']}", meta["clip_b"])]
        if sc == "A":
            lead.append(("усреднённое видео\n½(A+B)", meta["clip_avg"]))
        W = len(lead) + len(cols)
        title = (f"Смешанные состояния, мозг MaleCNS (модель нуль). {head}. {n} кадров по {dt * 1000:.0f} мс = "
                 f"{n * dt:.1f} с мухи, показано в {1 / (a.fps * dt):.0f}x медленнее. Под колонкой: r к A / r к B"
                 + (" / r к усреднённому видео" if sc == "A" else ""))
        for kind in ("png", "gif"):
            fig, ax = plt.subplots(1, W, figsize=(1.6 * W + 0.4, 2.9), facecolor=SURFACE, squeeze=False)
            ax = ax[0]
            ims = []
            f0 = n // 2
            vids = [v for _, v in lead] + [c["video"] for c in cols]
            names_ = [ln for ln, _ in lead] + [
                f"{c['label']}\n{c['r_a']:+.2f} / {c['r_b']:+.2f}" + (f" / {c['r_avg']:+.2f}" if sc == "A" else "") for c in cols]
            for j, (v, nm) in enumerate(zip(vids, names_)):
                im = ax[j].imshow(rs(v, f0 if kind == "png" else 0), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                ims.append((im, v))
                ax[j].set_title(nm, fontsize=7.5)
                style(ax[j])
            fig.suptitle(title + (f" Кадр {f0}." if kind == "png" else ""), fontsize=8.5, x=0.01, ha="left", wrap=True)
            fig.tight_layout(rect=(0, 0, 1, 0.8))
            out = figdir / f"{a.prefix}_mix_{sc}.{kind}"
            if kind == "png":
                fig.savefig(out, dpi=170, facecolor=SURFACE)
            else:
                def upd(f, ims=ims):
                    for im, v in ims:
                        im.set_data(rs(v, f))
                    return [im for im, _ in ims]
                FuncAnimation(fig, upd, frames=n, blit=True).save(out, writer=PillowWriter(fps=a.fps))
            plt.close(fig)
        print(f"{sc}: wrote {figdir / (a.prefix + '_mix_' + sc)}.png and .gif ({n} frames, {len(cols)} tasks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
