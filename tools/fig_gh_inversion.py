"""GitHub figure: the video recovered from each layer of model zero by inversion.

    python tools/fig_gh_inversion.py --out docs/figures/inversion_by_layer

Stage "inversion" of the pipeline (article part 1, section 4). Reads the runs
`data/generate/2026-09-19_malecns_invert_<type>_s3` (40 frames + 5 margin,
Sintel clip 3, control = clip 10's state as the target). Row 1: the input and
the video recovered from each layer; row 2: the same inversion aimed at another
clip's state, scored against this input. Grey on a fixed 0..1 scale.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from gh_style import (BAD, BG, GOOD, INK, LINE, MUTED, font, hex_image, honeycomb,  # noqa: E402
                      pipeline_strip, save_gif)

LAYERS = [("R1", "R1", "retina"), ("L1", "L1", "lamina"), ("Mi1", "Mi1", "medulla"),
          ("Tm9", "Tm9", "medulla"), ("T4a", "T4a", "motion"), ("T5a", "T5a", "motion")]


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ROOT / "docs" / "figures" / "inversion_by_layer"))
    p.add_argument("--fps", type=int, default=8)
    a = p.parse_args(argv)

    runs = {}
    for key, _, _ in LAYERS:
        d = ROOT / "data" / "generate" / f"2026-09-19_malecns_invert_{key}_s3"
        z = np.load(d / "recovered.npz")
        m = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        runs[key] = (z["true"], z["recovered"], z["control"], m["inversion"], m["control"])
    true = runs["R1"][0]
    n = len(true)
    from flydream.generate.invert import clip_from_sintel
    other = clip_from_sintel(10, n, 0.02, 0)            # clip B: whose activity the control aimed at
    to_b = {key: float(np.mean([corr(c, b) for c, b in zip(runs[key][2], other)])) for key, _, _ in LAYERS}

    comb = honeycomb(5)
    pw, ph = comb[0].shape[1], comb[0].shape[0]
    gap = 22
    cols = 1 + len(LAYERS)
    W = 48 * 2 + cols * pw + (cols - 1) * gap
    H = 700
    f_title, f_body, f_lab, f_small, f_num = font(30, True), font(19), font(19, True), font(16), font(18, True)

    frames = []
    for k in range(n):
        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im)
        y = pipeline_strip(d, 48, 24, W - 96, "invert") + 26
        d.text((48, y), "Inversion: the video recovered from one layer's activity", font=f_title, fill=INK)
        y += 44
        d.text((48, y), "Frozen model zero on the MaleCNS wiring. For each layer, a video is optimised until the "
                        "model reproduces that layer's", font=f_body, fill=MUTED)
        y += 26
        d.text((48, y), "recorded activity. Aimed at clip A's activity it returns clip A; aimed at clip B's, it "
                        f"returns clip B (r to A: {runs['R1'][4]:+.2f}).", font=f_body, fill=MUTED)
        y += 44

        xs = [48 + i * (pw + gap) for i in range(cols)]
        d.text((xs[0], y), "clip A", font=f_lab, fill=INK)
        d.text((xs[0], y + 24), "what the eye saw", font=f_small, fill=MUTED)
        for i, (key, name, depth) in enumerate(LAYERS, start=1):
            d.text((xs[i], y), name, font=f_lab, fill=INK)
            d.text((xs[i], y + 24), depth, font=f_small, fill=MUTED)
        d.line([xs[1] - gap // 2, y, xs[1] - gap // 2, y + 2 * ph + 150], fill=LINE, width=1)
        y += 56

        im.paste(hex_image(true[k], comb), (xs[0], y))
        for i, (key, _, _) in enumerate(LAYERS, start=1):
            im.paste(hex_image(runs[key][1][k], comb), (xs[i], y))
            d.text((xs[i], y + ph + 6), f"r to A = {runs[key][3]:.3f}", font=f_num, fill=GOOD)
        y += ph + 44

        d.text((xs[0], y - 30), "clip B · control", font=f_lab, fill=INK)
        im.paste(hex_image(other[k], comb), (xs[0], y))
        for i, (key, _, _) in enumerate(LAYERS, start=1):
            im.paste(hex_image(runs[key][2][k], comb), (xs[i], y))
            d.text((xs[i], y + ph + 6), f"r to B = {to_b[key]:.3f}", font=f_num, fill=GOOD)
        y += ph + 44

        d.text((48, H - 40), f"Sintel clip, frame {k + 1}/{n} · 20 ms per frame, shown {int(round(50 / a.fps))}× slower · "
                             "r = mean per-frame correlation · run 2026-09-19_malecns_invert_*_s3",
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
