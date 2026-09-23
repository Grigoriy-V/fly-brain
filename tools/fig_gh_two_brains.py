"""GitHub figure: one clip, two brains - FlyVis's own connectome against the MaleCNS build.

    python tools/fig_gh_two_brains.py --out docs/figures/two_brains

Stage "model zero" of the pipeline (article part 1, sections 2 and 4). The same
inversion - optimise the input until the frozen model reproduces one layer's
activity - on the FlyVis reference network (member 000, its own FIB-based
connectome) and on model zero (the same member's parameters on the MaleCNS
wiring), with identical settings: 40 frames + 5 margin, 150 steps, the same
clip, the same control clip. Runs `data/generate/2026-09-24_flyvis_invert_<type>_s3`
(rerun locally for this figure) and `2026-09-19_malecns_invert_<type>_s3`.
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

from gh_style import BG, GOOD, INK, LINE, MUTED, font, hex_image, honeycomb, pipeline_strip, save_gif  # noqa: E402

LAYERS = [("R1", "retina"), ("L1", "lamina"), ("Mi1", "medulla"), ("Tm9", "medulla"), ("T4a", "motion"),
          ("T5a", "motion")]
MODELS = [("flyvis", "FlyVis", "reference connectome"), ("malecns", "MaleCNS", "our build, model zero")]
N = 40


def score(rec, true):
    return float(np.mean([np.corrcoef(a, b)[0, 1] for a, b in zip(rec, true)]))


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ROOT / "docs" / "figures" / "two_brains"))
    p.add_argument("--fps", type=int, default=6)
    a = p.parse_args(argv)

    date = {"flyvis": "2026-09-24", "malecns": "2026-09-19"}
    rec, r = {}, {}
    true = None
    for m, _, _ in MODELS:
        for t, _ in LAYERS:
            z = np.load(ROOT / "data" / "generate" / f"{date[m]}_{m}_invert_{t}_s3" / "recovered.npz")
            tr = z["true"][:N]
            true = tr if true is None else true
            assert np.allclose(tr, true, atol=1e-3), "the two runs must share the input clip"
            rec[m, t] = z["recovered"][:N]
            r[m, t] = score(rec[m, t], tr)

    comb = honeycomb(5)
    pw, ph = comb[0].shape[1], comb[0].shape[0]
    gap = 22
    cols = 1 + len(LAYERS)
    W = 48 * 2 + cols * pw + (cols - 1) * gap
    H = 770
    f_title, f_body, f_lab, f_small, f_num = font(30, True), font(19), font(19, True), font(16), font(18, True)

    frames = []
    for k in range(N):
        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im)
        y = pipeline_strip(d, 48, 24, W - 96, "model") + 26
        d.text((48, y), "One clip, two brains: FlyVis against the MaleCNS build", font=f_title, fill=INK)
        y += 44
        d.text((48, y), "The same FlyVis parameters on two wirings. Each panel: the video recovered from one layer's "
                        "activity by inversion.", font=f_body, fill=MUTED)
        y += 26
        d.text((48, y), "Both brains keep the picture through every layer; the MaleCNS build is the weaker motion "
                        "detector (DSI 0.15 vs 0.39).", font=f_body, fill=MUTED)
        y += 44

        xs = [48 + i * (pw + gap) for i in range(cols)]
        for i, (t, depth) in enumerate(LAYERS, start=1):
            d.text((xs[i], y), t, font=f_lab, fill=INK)
            d.text((xs[i], y + 24), depth, font=f_small, fill=MUTED)
        d.line([xs[1] - gap // 2, y, xs[1] - gap // 2, y + 2 * ph + 150], fill=LINE, width=1)
        y += 56
        top = y
        for row, (m, name, sub) in enumerate(MODELS):
            d.text((xs[1], y), name, font=f_lab, fill=INK if m == "flyvis" else GOOD)
            d.text((xs[1] + d.textlength(name, font=f_lab) + 10, y + 2), sub, font=f_small, fill=MUTED)
            y += 30
            for i, (t, _) in enumerate(LAYERS, start=1):
                im.paste(hex_image(rec[m, t][k], comb), (xs[i], y))
                d.text((xs[i], y + ph + 6), f"r = {r[m, t]:.3f}", font=f_num, fill=INK if m == "flyvis" else GOOD)
            y += ph + 44
        mid = (top + y - 44) // 2 - ph // 2
        d.text((xs[0], mid - 30), "input", font=f_lab, fill=INK)
        im.paste(hex_image(true[k], comb), (xs[0], mid))

        d.text((48, H - 40), f"Sintel clip, frame {k + 1}/{N} · 20 ms per frame · identical settings for both · runs "
                             "2026-09-24_flyvis_invert_*_s3, 2026-09-19_malecns_invert_*_s3", font=f_small,
               fill=MUTED)
        frames.append(im)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_gif(frames, out.with_suffix(".gif"), a.fps)
    frames[N // 2].save(out.with_suffix(".png"))
    for (m, t), v in r.items():
        print(f"  {m:8s} {t:4s} r {v:.3f}")
    for s in (".gif", ".png"):
        print(f"{out.with_suffix(s)}  {out.with_suffix(s).stat().st_size / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
