"""GitHub figure: what the layers themselves look like - model zero's activity, cell type by cell type.

    python tools/fig_gh_layers.py --activity <act.npz> --out docs/figures/layers_activity

Stage "model zero". No decoding and no inversion: each panel is one cell type's
activity on the 721-column lattice while the eye watches the clip. Colour is
the deviation from that type's median over the clip (red = depolarised, blue =
hyperpolarised), each type on its own symmetric range, because the types'
voltages are in different units; the sign is what reads. `--activity` is the
frozen model's response to Sintel clip 3 (npz: `video` and one (40, 721) array
per type in flyvis hex order).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from gh_style import BG, INK, LINE, MUTED, font, hex_image, honeycomb, pipeline_strip, save_gif  # noqa: E402

TYPES = [("R1", "photoreceptor", "retina"), ("L1", "ON pathway", "lamina"), ("L3", "OFF pathway", "lamina"),
         ("Mi1", "ON pathway", "medulla"), ("Tm1", "OFF pathway", "medulla"), ("Tm9", "OFF pathway", "medulla"),
         ("T4a", "ON motion", "motion detector"), ("T5a", "OFF motion", "motion detector")]


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import colormaps

    p = argparse.ArgumentParser()
    p.add_argument("--activity", required=True)
    p.add_argument("--out", default=str(ROOT / "docs" / "figures" / "layers_activity"))
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--prefix", default="", help="key prefix in the npz, e.g. 'flow/00_' for the FlyVis network")
    p.add_argument("--model-label", default="the FlyVis reference network, member 000")
    a = p.parse_args(argv)

    A = np.load(a.activity)
    video = A["video"][:40]
    n = len(video)
    cmap = colormaps["RdBu_r"]
    shown = {}
    for t, _, _ in TYPES:
        x = A[a.prefix + t]
        c = x - np.nanmedian(x)
        s = np.nanpercentile(np.abs(c), 99) + 1e-9
        shown[t] = np.nan_to_num(0.5 + 0.5 * np.clip(c / s, -1, 1), nan=0.5)

    comb = honeycomb(5)
    pw, ph = comb[0].shape[1], comb[0].shape[0]
    gap = 22
    per_row = 4
    W = 1216
    gap = (W - 96 - (1 + per_row) * pw) // per_row
    H = 760
    f_title, f_body, f_lab, f_small = font(30, True), font(19), font(19, True), font(16)

    frames = []
    for k in range(n):
        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im)
        y = pipeline_strip(d, 48, 24, W - 96, "model") + 26
        d.text((48, y), "What the layers look like: the network's own activity", font=f_title, fill=INK)
        y += 44
        d.text((48, y), "One cell type per panel, on the 721-column lattice, while the eye watches the clip. Red: "
                        "depolarised, blue: hyperpolarised.", font=f_body, fill=MUTED)
        y += 26
        d.text((48, y), "Photoreceptors copy the picture, lamina cells invert it, the medulla splits it into ON "
                        "and OFF, motion detectors keep moving edges.", font=f_body, fill=MUTED)
        y += 46
        x0 = 48
        d.text((x0, y), "input", font=f_lab, fill=INK)
        d.text((x0, y + 24), "what the eye saw", font=f_small, fill=MUTED)
        im.paste(hex_image(video[k], comb), (x0, y + 52 + (ph + 60) // 2))
        d.line([x0 + pw + gap // 2, y, x0 + pw + gap // 2, y + 2 * (ph + 60) + 40], fill=LINE, width=1)
        for i, (t, role, depth) in enumerate(TYPES):
            r, c = divmod(i, per_row)
            px = x0 + (1 + c) * (pw + gap)
            py = y + r * (ph + 80)
            d.text((px, py), t, font=f_lab, fill=INK)
            d.text((px + d.textlength(t, font=f_lab) + 8, py + 2), depth, font=f_small, fill=MUTED)
            d.text((px, py + 24), role, font=f_small, fill=MUTED)
            im.paste(hex_image(shown[t][k], comb, cmap=cmap), (px, py + 52))
        d.text((48, H - 40), f"Sintel clip, frame {k + 1}/{n} · 20 ms per frame, shown {int(round(50 / a.fps))}× slower"
                             f" · {a.model_label} · each type on its own colour range",
               font=f_small, fill=MUTED)
        frames.append(im)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_gif(frames, out.with_suffix(".gif"), a.fps)
    frames[n // 2].save(out.with_suffix(".png"))
    for s in (".gif", ".png"):
        print(f"{out.with_suffix(s)}  {out.with_suffix(s).stat().st_size / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
