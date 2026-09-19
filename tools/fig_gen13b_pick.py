"""13B, the two clips for the human: one row each, full resolution, only what
matters (the human, 2026-09-20: vertical sheets, downscaled GIFs and rows
of comparisons are unreadable).

    python tools/fig_gen13b_pick.py --run data/gen13b

1. `..._i2i`: three held-out clips (a dark scene, a bright scene, a
   texture) — what the eye saw beside the generator's video from the
   deep state.
2. `..._knobs`: clip A's state given to the generator in pieces — all
   T4/T5, T4 only, T5 only, T4a only (two seeds), and the shuffled-state
   control.
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


def row(cells: list, title: str, out: Path, frames: int, fps: int, plt, FuncAnimation, PillowWriter, dpi: int = 100):
    """cells: (video, header, footer); one row, cells 3 inches wide."""
    n = len(cells)
    for kind in ("png", "gif"):
        fig, ax = plt.subplots(1, n, figsize=(3.0 * n, 3.9), facecolor=SURFACE, squeeze=False)
        ims, f0 = [], frames // 2
        for j, (vid, head, foot) in enumerate(cells):
            a = ax[0, j]
            a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values():
                sp.set_visible(False)
            im = a.imshow(to_raster(vid[f0 if kind == "png" else 0], 721, 4, fill=np.nan), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ims.append((im, vid))
            a.set_title(head, fontsize=11, pad=8)
            a.set_xlabel(foot, fontsize=10)
        fig.suptitle(title, fontsize=11, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        if kind == "png":
            fig.savefig(out.with_suffix(".png"), dpi=dpi, facecolor=SURFACE)
        else:
            def upd(f, ims=ims):
                for im, v in ims:
                    im.set_data(to_raster(v[f], 721, 4, fill=np.nan))
                return [im for im, _ in ims]
            FuncAnimation(fig, upd, frames=frames, blit=True).save(out.with_suffix(".gif"), writer=PillowWriter(fps=fps), dpi=dpi)
        plt.close(fig)
    print(f"wrote {out}.png / .gif")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=str(ROOT / "data" / "gen13b"))
    p.add_argument("--prefix", default="2026-09-20_malecns_gen13b")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--outdir", default=str(ROOT / "reports" / "figures"))
    a = p.parse_args(argv)
    run = Path(a.run)
    S = json.loads((run / "samples.json").read_text(encoding="utf-8"))
    z = np.load(run / "samples.npz")
    rt, frames = S["roundtrip_per_sample"], S["frames"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    outdir = Path(a.outdir)
    slow = f"{frames} кадров по 20 мс ({frames * 0.02:.1f} с мухи), в {1 / (a.fps * 0.02):.0f}× медленнее"

    def s(key):
        return z[key], rt[key]

    # 1. i2i on held-out clips
    cells = []
    for i, name in ((1473, "тёмная сцена"), (8166, "светлая сцена"), (5268, "текстура")):
        k = f"test_{i}__full__s1__seed0"
        v, r = s(k)
        cells.append((z[f"ref__test_{i}__clip"], f"что видел глаз: {name}", "отложенный клип"))
        cells.append((v, "генератор ← состояние T4/T5", f"прогонка {r:.3f}, r {S['scores'][f'test_{i}__full__s1']['r_clip']:+.2f}"))
    row(cells, f"13B, генератор от состояния мозга (SiT-интерполянт, MaleCNS модель нуль): клипы, которых модель не видела. {slow}.",
        outdir / f"{a.prefix}_i2i", frames, a.fps, plt, FuncAnimation, PillowWriter)
    # 2. knobs on clip A
    ref = z["ref__clip_A__clip_a"]
    cells = [(ref, "что видел глаз: клип A", "вход")]
    for mask, head in (("full", "все 8 типов T4/T5"), ("t4", "только T4a-d"), ("t5", "только T5a-d")):
        v, r = s(f"clip_A__{mask}__s1__seed0")
        cells.append((v, head, f"прогонка {r:.3f}"))
    for k in (0, 1):
        v, r = s(f"clip_A__T4a__s1__seed{k}")
        cells.append((v, f"только T4a, z{k}", f"прогонка {r:.3f}"))
    v, r = s("control_shuffled__full__s1__seed0")
    cells.append((v, "контроль: клетки перемешаны", f"прогонка {r:.3f}"))
    row(cells, f"13B, крутилки: одно состояние (клип A), генератору отдана его часть. Под клеткой — ошибка обратной прогонки "
        f"(видео → мозг → состояние; меньше = совместимее). {slow}.",
        outdir / f"{a.prefix}_knobs_row", frames, a.fps, plt, FuncAnimation, PillowWriter)
    return 0


if __name__ == "__main__":
    sys.exit(main())
