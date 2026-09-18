"""Dreams-lite pictures (ROADMAP item 11): one stacked clip per source.

    python tools/fig_dreams.py --prefix 2026-09-19_malecns --sources eye_noise flash dark_after neuron_noise

Reads `data/generate/<prefix>_dream_<source>_<stage>/recovered.npz` written by
`flydream.generate.dreams`. Per source one GIF and one PNG: row 1 is the
input ("what the eye saw") and, to its right, the video the generator
recovered from each stage's state; row 2 is the shuffled-state control (the
same input, the stage's cells permuted) with the same columns. Under each
column its score: r against the input frame (the clip's last frame in the
dark after a clip), or, for internally generated activity where no picture
went in, the spatial contrast of the recovered video. In the dark after a
clip only the dark frames are shown (the frames the fit read), and a column
with the clip's last frame stands beside the black input for comparison.
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
TITLE = {"eye_noise": "ВХОД: белый шум в глаз (sd {eye_noise_sd})",
         "flash": "ВХОД: серое, вспышка {flash_frames} кадров, серое",
         "dark_after": "ВХОД: клип {dark_after_sample}, затем темнота — показаны только тёмные кадры",
         "neuron_noise": "ВХОД: серое; шум внутри нейронов (sd {neuron_noise_sd})"}


def load(prefix: str, source: str, stages: list[str], data: Path):
    cols, meta = [], None
    for s in stages:
        d = data / f"{prefix}_dream_{source}_{s}"
        if not d.exists():
            print(f"  {d.name}: missing, skipped"); continue
        z = np.load(d / "recovered.npz")
        m = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        meta = meta or m
        cols.append({"label": s if s != "T4T5" else "T4+T5", "rec": z["recovered"], "ctrl": z["control"],
                     "true": z["true"], "read": z["read"], "ref": z["ref"] if "ref" in z else None,
                     "r": m["inversion"], "r_ctrl": m["control"],
                     "c": m["contrast_inversion"], "c_ctrl": m["contrast_control"]})
    return cols, meta


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", required=True, help="e.g. 2026-09-19_malecns")
    p.add_argument("--sources", nargs="*", default=["eye_noise", "flash", "dark_after", "neuron_noise"])
    p.add_argument("--stages", nargs="*", default=STAGES)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--data", default=str(ROOT / "data" / "generate"), help="directory of the <tag>/ folders")
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
    for source in a.sources:
        cols, meta = load(a.prefix, source, a.stages, Path(a.data))
        if not cols:
            print(f"{source}: nothing to draw"); continue
        read = cols[0]["read"]
        inp = cols[0]["true"][read]
        dt = float(meta.get("dt", 0.02))
        n = len(inp)
        lead = [("что видел глаз", inp)]
        if source == "dark_after" and cols[0]["ref"] is not None:
            lead.append(("последний кадр клипа\n(для сравнения)", cols[0]["ref"]))
        picture = meta["inversion"] is not None

        def score(c, key):
            if picture:
                return f"r = {c['r'] if key == 'rec' else c['r_ctrl']:+.2f}"
            return f"контраст {c['c'] if key == 'rec' else c['c_ctrl']:.3f}"

        rows = [(TITLE[source].format(**meta["settings"]), "rec"),
                ("КОНТРОЛЬ: тот же вход, клетки стадии перемешаны", "ctrl")]
        W = len(lead) + len(cols)
        title = (f"Сны-лайт, мозг MaleCNS (модель нуль): состояние стадии -> генератор -> видео. {n} кадров по "
                 f"{dt * 1000:.0f} мс = {n * dt:.1f} с мухи, показано в {1 / (a.fps * dt):.0f}x медленнее. "
                 f"Число под колонкой: {'r к ' + meta['score_against'].replace('input frame', 'входу').replace('last clip frame', 'последнему кадру клипа') if picture else 'контраст выхода (sd по глазу)'}")

        for kind in ("png", "gif"):
            fig, ax = plt.subplots(2, W, figsize=(1.55 * W + 0.4, 5.0), facecolor=SURFACE)
            ims = []
            f0 = n // 2
            for i, (name, key) in enumerate(rows):
                vids = [v for _, v in lead] + [c[key][read] for c in cols]
                names = [ln for ln, _ in lead] + [f"из {c['label']}\n{score(c, key)}" for c in cols]
                # the row's name on its own line above the row, so it never runs into a column title
                ax[i, 0].annotate(name, xy=(0, 1.30), xycoords="axes fraction", ha="left", va="bottom", fontsize=8.5,
                                  fontweight="bold", annotation_clip=False)
                for j, (v, nm) in enumerate(zip(vids, names)):
                    im = ax[i, j].imshow(rs(v, f0 if kind == "png" else 0), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                    ims.append((im, v))
                    ax[i, j].set_title(nm, fontsize=7.5)
                    style(ax[i, j])
            fig.suptitle(title + (f" Кадр {f0}." if kind == "png" else ""), fontsize=8.5, x=0.01, ha="left", wrap=True)
            fig.tight_layout(rect=(0, 0, 1, 0.84), h_pad=3.0)
            out = figdir / f"{a.prefix}_dream_{source}.{kind}"
            if kind == "png":
                fig.savefig(out, dpi=170, facecolor=SURFACE)
            else:
                def upd(f, ims=ims):
                    for im, v in ims:
                        im.set_data(rs(v, f))
                    return [im for im, _ in ims]
                FuncAnimation(fig, upd, frames=n, blit=True).save(out, writer=PillowWriter(fps=a.fps))
            plt.close(fig)
        print(f"{source}: wrote {figdir / (a.prefix + '_dream_' + source)}.png and .gif ({n} frames, {len(cols)} stages)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
