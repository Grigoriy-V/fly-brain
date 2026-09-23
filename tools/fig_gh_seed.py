"""GitHub figure: a clip's own seed, and a random one.

    python tools/fig_gh_seed.py --out docs/figures/seed_known_and_random

Stage "state prior" (article part 2, sections 6 and 10). The hex-local flow of
step 23 maps noise to a brain state (the T4a+T4b block) and runs backwards.
Row 1: six held-out clips from classes the flow never saw. Row 2: each clip ->
frozen brain -> state -> the flow run backwards -> its own seed -> the flow
forwards -> state -> 13B -> video. Row 3: a random seed through the same chain,
tied to no clip. Run `2026-09-21_prior23_seed`, data `data/prior23/seed23.npz`.
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

from gh_style import BAD, BG, GOOD, INK, LINE, MUTED, font, hex_image, honeycomb, pipeline_strip, save_gif  # noqa: E402

SEED, DRAW = "сид от клипа через поток", "свежий розыгрыш"


def corr(a, b):
    return float(np.mean([np.corrcoef(x, y)[0, 1] for x, y in zip(a, b)]))


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=str(ROOT / "data" / "prior23"))
    p.add_argument("--out", default=str(ROOT / "docs" / "figures" / "seed_known_and_random"))
    p.add_argument("--fps", type=int, default=8)
    a = p.parse_args(argv)

    z = np.load(Path(a.run) / "seed23.npz")
    meta = json.loads((Path(a.run) / "seed23.json").read_text(encoding="utf-8"))
    m = len(meta["clip_idx"])
    raw = [z[f"raw__{i}"] for i in range(m)]
    seed = [z[f"video__{SEED}|{i}"] for i in range(m)]
    draw = [z[f"video__{DRAW}|{i}"] for i in range(m)]
    r_seed = [corr(s, r) for s, r in zip(seed, raw)]
    near_draw = meta["groups"][DRAW]["nearest_r"]
    n = len(raw[0])

    comb = honeycomb(5)
    pw, ph = comb[0].shape[1], comb[0].shape[0]
    W = 1216
    lab_w = 150
    gap = (W - 96 - lab_w - m * pw) // (m - 1)
    H = 900
    f_title, f_body, f_lab, f_small, f_num = font(30, True), font(19), font(19, True), font(16), font(18, True)
    rows = [("held-out clip", "what the eye saw", raw, None),
            ("its own seed", "the flow run backwards", seed, [f"r = {v:.2f}" for v in r_seed]),
            ("a random seed", "N(0, I), tied to no clip", draw, None)]

    frames = []
    for k in range(n):
        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im)
        y = pipeline_strip(d, 48, 24, W - 96, "prior") + 26
        d.text((48, y), "A brain state from noise: a clip's own seed, and a random one", font=f_title, fill=INK)
        y += 44
        d.text((48, y), "A flow on the hexagonal lattice maps noise to a T4 brain state; 13B renders the state. Run "
                        "backwards, the flow gives every clip", font=f_body, fill=MUTED)
        y += 26
        d.text((48, y), "its own seed, and that seed brings the clip back. A random seed gives structure, not yet "
                        "a scene.", font=f_body, fill=MUTED)
        y += 48
        xs = [48 + lab_w + i * (pw + gap) for i in range(m)]
        d.line([xs[0] - 14, y, xs[0] - 14, y + 3 * (ph + 54)], fill=LINE, width=1)
        for name, sub, vids, labels in rows:
            d.text((48, y + ph // 2 - 24), name, font=f_lab, fill=INK)
            d.text((48, y + ph // 2 + 2), sub, font=f_small, fill=MUTED)
            for i in range(m):
                im.paste(hex_image(vids[i][k], comb), (xs[i], y))
                if labels:
                    d.text((xs[i], y + ph + 4), labels[i], font=f_num, fill=GOOD)
            y += ph + 54
        d.text((48, y - 20), f"random seeds: nearest of 15,514 training clips r = {near_draw:.2f}, a real held-out "
                             f"clip reads 0.52 — new, not yet a scene", font=f_small, fill=BAD)
        d.text((48, H - 40), f"UCF101, classes never seen in training · frame {k + 1}/{n} · 13B's ceiling on these "
                             f"clips r = 0.88 · run 2026-09-21_prior23_seed", font=f_small, fill=MUTED)
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
