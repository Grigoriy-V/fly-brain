"""Post-format video for post 1: a fly brain model's activity and the video read back out of it.

    python tools/fig_hook_post1.py --out posts/post1_video

For LinkedIn, not for measurement. 4:5 frame (1080 x 1350), dark ground, the
hex lattice as a honeycomb, the style of tools/fig_hook_levels.py. Top: what the
eye saw (Sintel clip 3, "clip A"). Below, four paired columns: each column is a
brain activity on top and the video rebuilt from it below.

- Columns 1-3, inversion: one layer of model zero (R1, Mi1, T4a), the video
  optimised until the frozen model reproduces that layer's activity. Runs
  `2026-09-19_malecns_invert_<type>_s3`. No r on the frame: an inversion of the
  model's own activity reads 0.99-1.00 on every layer, which looks like a typo
  in a post; the pictures carry it.
- Column 4, the 13B generator: the activity of all eight T4/T5 types (T5a
  shown) turned into 40 frames by a conditional flow, no per-clip optimisation.
  Run `2026-09-20_gen13b_samples`, `clip_A__full__s1__seed0`. This clip is from
  13B's training scenes; on held-out clips 13B reads r 0.966 - said on the frame.

`--activity` is written by tools/cmp_layers_flyvis.py (model zero's response to
the same clip, key prefix `malecns_`). Output: mp4 (24 fps container, each frame
repeated to keep --fps) and a png of the middle frame; posts/ is not tracked.
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

from fig_hook_levels import ACCENT, BG, DIM, FG, colorise, font, honeycomb  # noqa: E402

W, H = 1080, 1350
COLS = [("R1", "Photoreceptors", "inversion"), ("Mi1", "Medulla", "inversion"),
        ("T4a", "Motion detectors", "inversion"), ("T5a", "Generator 13B", "one pass")]


def corr(a, b):
    return float(np.mean([np.corrcoef(x, y)[0, 1] for x, y in zip(a, b)]))


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import colormaps

    p = argparse.ArgumentParser()
    p.add_argument("--activity", default=str(ROOT / "data" / "figures" / "act_s3_malecns_flyvis.npz"))
    p.add_argument("--gen", default=str(ROOT / "data" / "gen13b" / "samples.npz"))
    p.add_argument("--out", default=str(ROOT / "posts" / "post1_video"))
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--video-fps", type=int, default=24)
    p.add_argument("--loops", type=int, default=3)
    a = p.parse_args(argv)

    A = np.load(a.activity)
    video = np.clip(A["video"][:40], 0, 1)
    n = len(video)
    rec, score, act = {}, {}, {}
    for t, _, _ in COLS[:3]:
        d = ROOT / "data" / "generate" / f"2026-09-19_malecns_invert_{t}_s3"
        rec[t] = np.clip(np.load(d / "recovered.npz")["recovered"], 0, 1)[:n]
        score[t] = json.loads((d / "meta.json").read_text(encoding="utf-8"))["inversion"]
    g = np.load(a.gen)
    rec["T5a"] = np.clip(g["clip_A__full__s1__seed0"], 0, 1)[:n]
    score["T5a"] = corr(rec["T5a"], video)
    for t, _, _ in COLS:
        x = A["malecns_" + t][:n]
        lo, hi = np.nanpercentile(x, 1), np.nanpercentile(x, 99)
        act[t] = np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)

    gray, heat = colormaps["gray"], colormaps["inferno"]
    big_i, big_m = honeycomb(8)
    sm_i, sm_m = honeycomb(7)
    sw = sm_i.shape[1]
    gap = (W - 2 * 50 - 4 * sw) // 3
    xs = [50 + i * (sw + gap) for i in range(4)]
    f_title, f_sub, f_lab, f_small, f_num, f_arrow = (font(50, True), font(26), font(25, True), font(21), font(24, True),
                                                      font(40, True))
    frames = []
    for k in range(n):
        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im)
        d.text((50, 44), "Reading a video back out", font=f_title, fill=FG)
        d.text((50, 102), "of a fly's brain", font=f_title, fill=ACCENT)
        d.text((50, 172), "A model of the fly visual system, wired by the real connectome.", font=f_sub, fill=DIM)
        d.text((50, 206), "Each column: brain activity, and the video rebuilt from it alone.", font=f_sub, fill=DIM)

        eye = colorise(video[k], big_i, big_m, gray)
        ey = 256
        im.paste(eye, ((W - eye.width) // 2, ey))
        d.text((50, ey + eye.height // 2 - 30), "What the", font=f_lab, fill=FG)
        d.text((50, ey + eye.height // 2 + 2), "eye saw", font=f_lab, fill=FG)

        y = ey + eye.height + 30
        for (t, name, how), x in zip(COLS, xs):
            d.text((x, y), name, font=f_lab, fill=ACCENT if t == "T5a" else FG)
            d.text((x, y + 32), "all T4/T5 types" if t == "T5a" else t, font=f_small, fill=DIM)
        y += 72
        for (t, _, _), x in zip(COLS, xs):
            im.paste(colorise(np.nan_to_num(act[t][k], nan=0.0), sm_i, sm_m, heat), (x, y))
        y += sm_i.shape[0] + 6
        for (t, _, how), x in zip(COLS, xs):
            d.text((x + sw // 2 - 12, y - 6), "↓", font=f_arrow, fill=ACCENT if t == "T5a" else DIM)
            d.text((x + sw // 2 + 18, y + 10), how, font=f_small, fill=ACCENT if t == "T5a" else DIM)
        y += 56
        for (t, _, _), x in zip(COLS, xs):
            im.paste(colorise(rec[t][k], sm_i, sm_m, gray), (x, y))
        y += sm_i.shape[0] + 30
        d.text((50, y), "Inversion: the video is optimised until the frozen model reproduces that activity.",
               font=f_small, fill=DIM)
        d.text((50, y + 28), "13B: a trained flow draws the video from the motion detectors in one go; this clip was",
               font=f_small, fill=DIM)
        d.text((50, y + 56), "in its training scenes, on held-out clips it reads r = 0.966.", font=f_small, fill=DIM)
        bar = int((W - 100) * (k + 1) / n)
        d.rectangle([50, H - 22, 50 + bar, H - 16], fill=ACCENT)
        frames.append(im)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[n // 2].save(out.with_suffix(".png"))
    import imageio.v2 as imageio
    rep = max(1, round(a.video_fps / a.fps))
    with imageio.get_writer(out.with_suffix(".mp4"), fps=a.fps * rep, codec="libx264", quality=8,
                            macro_block_size=2) as wr:
        for _ in range(a.loops):
            for f in frames:
                for _ in range(rep):
                    wr.append_data(np.asarray(f))
    for s in (".png", ".mp4"):
        f = out.with_suffix(s)
        print(f"{f}  {f.stat().st_size / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
