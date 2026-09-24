"""GitHub figure: a video from a brain state - the 13B generator on held-out clips.

    python tools/fig_gh_render.py --out docs/figures/state_to_video

Stage "state -> video" (article part 1, section 8). 13B is a conditional flow
(SiT, 1.4 M parameters): the eight T4/T5 types' activity plus noise z -> 40
frames in 20 Euler steps. Columns: held-out clips. Rows: the clip; 13B from the
clip's state with noise seed 0; the same state with seed 1 (the state, not the
noise, sets the video). The last column is the control on clip A: the same
state and the same z, but with the cells shuffled, gives another video. Run
`2026-09-20_gen13b_samples`, data `data/gen13b/samples.npz`.
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

from gh_style import BAD, BG, GOOD, INK, LINE, MUTED, font, hex_image, honeycomb, pipeline_strip, save_gif  # noqa: E402


def corr(a, b):
    return float(np.mean([np.corrcoef(x, y)[0, 1] for x, y in zip(a, b)]))


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=str(ROOT / "data" / "gen13b"))
    p.add_argument("--clips", nargs="+", default=["test_684", "test_1473", "test_5268", "test_8166", "test_8415"])
    p.add_argument("--out", default=str(ROOT / "docs" / "figures" / "state_to_video"))
    p.add_argument("--fps", type=int, default=8)
    a = p.parse_args(argv)

    z = np.load(Path(a.run) / "samples.npz")
    cols = []
    for c in a.clips:
        ref = z[f"ref__{c}__clip"]
        s0, s1 = z[f"{c}__full__s1__seed0"], z[f"{c}__full__s1__seed1"]
        cols.append((ref, s0, s1, corr(s0, ref), corr(s1, ref)))
    refA = z["ref__clip_A__clip_a"]
    trueA, shufA = z["strength_true__full__s1__seed0"], z["strength_shuffled__full__s1__seed0"]
    ctrl = (refA, trueA, shufA, corr(trueA, refA), corr(shufA, refA))
    n = len(cols[0][0])

    comb = honeycomb(5)
    pw, ph = comb[0].shape[1], comb[0].shape[0]
    W, lab_w = 1216, 180
    m = len(cols) + 1
    gap = (W - 96 - lab_w - m * pw - 30) // (m - 1)
    H = 870
    f_title, f_body, f_lab, f_small, f_num = font(30, True), font(19), font(19, True), font(16), font(17, True)
    rows = [("held-out clip", "what the eye saw"), ("13B from its state", "noise seed 0"),
            ("13B from its state", "noise seed 1")]
    frames = []
    for k in range(n):
        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im)
        y = pipeline_strip(d, 48, 24, W - 96, "render") + 26
        d.text((48, y), "A video from a brain state: the 13B generator", font=f_title, fill=INK)
        y += 44
        d.text((48, y), "A conditional flow takes the activity of the eight T4/T5 motion-detector types plus noise and "
                        "produces 40 frames.", font=f_body, fill=MUTED)
        y += 26
        d.text((48, y), "Two noise seeds give the same video: the state decides it. Control: the same state with its "
                        "cells shuffled.", font=f_body, fill=MUTED)
        y += 60
        xs = [48 + lab_w + i * (pw + gap) for i in range(m - 1)] + [48 + lab_w + (m - 1) * (pw + gap) + 30]
        d.line([xs[0] - 14, y - 26, xs[0] - 14, y + 3 * (ph + 44)], fill=LINE, width=1)
        d.line([xs[-1] - 22, y - 26, xs[-1] - 22, y + 3 * (ph + 44)], fill=LINE, width=1)
        d.text((xs[-1], y - 26), "control, clip A", font=f_small, fill=BAD)
        for r, (name, sub) in enumerate(rows):
            d.text((48, y + ph // 2 - 22), name, font=f_lab, fill=INK)
            d.text((48, y + ph // 2 + 4), sub, font=f_small, fill=MUTED)
            for i, c in enumerate(cols):
                im.paste(hex_image(c[r][k], comb), (xs[i], y))
                if r > 0:
                    d.text((xs[i], y + ph + 4), f"r = {c[r + 2]:.2f}", font=f_num, fill=GOOD)
            im.paste(hex_image(ctrl[r][k], comb), (xs[-1], y))
            if r == 1:
                d.text((xs[-1], y + ph + 4), f"true state r = {ctrl[3]:.2f}", font=f_num, fill=GOOD)
            if r == 2:
                d.text((xs[-1], y + ph + 4), f"shuffled r = {ctrl[4]:+.2f}", font=f_num, fill=BAD)
            y += ph + 44
        d.text((48, H - 40), f"Sintel and procedural clips held out of training · frame {k + 1}/{n} · 20 ms per frame "
                             "· r = correlation with the clip · run 2026-09-20_gen13b_samples", font=f_small, fill=MUTED)
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
