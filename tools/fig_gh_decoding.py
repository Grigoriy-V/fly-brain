"""GitHub figure: reading the picture back out of one cell type with a linear decoder.

    python tools/fig_gh_decoding.py --out docs/figures/decoding_by_type

Stage "decoding" (article part 1, section 3). Ridge regression from one cell
type's activity (two frames: now and one step back, the window without the
aliasing of ISS-0006) to the 721 pixels of the frame, fitted on Sintel scenes
and scored on the scenes held out whole. Model zero on the MaleCNS wiring
(`data/decode/2026-09-18_decode_sintel_malecns_v9/pairs.npz`). The clip shown is
Sintel clip 3, the same one as docs/figures/inversion_by_layer, so the two
figures sit side by side; that clip is one of the decoder's training scenes,
while r is over the held-out scenes. Control: the same decoder fitted on
activity whose frames are shuffled in time. Grey on a fixed 0..1 scale.
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
from flydream.decode import pairs as P  # noqa: E402
from flydream.decode import ridge as R  # noqa: E402

TYPES = [("R1", "retina"), ("L1", "lamina"), ("Mi1", "medulla"), ("Tm9", "medulla"), ("T4a", "motion"),
         ("T5a", "motion")]
LAGS = (0, 1)


def corr(a, b):
    return float(np.nanmean([np.corrcoef(x, y)[0, 1] for x, y in zip(a, b)]))


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", default=str(ROOT / "data" / "decode" / "2026-09-18_decode_sintel_malecns_v9" / "pairs.npz"))
    p.add_argument("--sample", type=int, default=3, help="clip to show; 3 = the inversion figure's clip")
    p.add_argument("--out", default=str(ROOT / "docs" / "figures" / "decoding_by_type"))
    p.add_argument("--fps", type=int, default=8)
    a = p.parse_args(argv)

    pairs = P.Pairs.load(a.pairs)
    train_i, test_i = P.split_by_group(pairs.groups, 0.2, 0)
    show = a.sample
    shuffled = P.Pairs(stimulus=pairs.stimulus, activity=R.shuffle_time(pairs.activity, 0), groups=pairs.groups,
                       cell_types=pairs.cell_types, index=pairs.index)
    rec, ctrl, r_rec, r_ctrl = {}, {}, {}, {}
    for t, _ in TYPES:
        for src, store, score in ((pairs, rec, r_rec), (shuffled, ctrl, r_ctrl)):
            xtr, ytr, gtr = src.frames(t, train_i, LAGS)
            ok = np.isfinite(xtr).all(1) & np.isfinite(ytr).all(1)
            model = R.fit(xtr[ok], ytr[ok], val_fraction=0.2, seed=0, groups=gtr[ok])
            xte, yte, _ = src.frames(t, test_i, LAGS)
            okt = np.isfinite(xte).all(1) & np.isfinite(yte).all(1)
            score[t] = corr(model.predict(xte[okt]), yte[okt])
            xs, ys, _ = src.frames(t, np.array([show]), LAGS)
            pred = np.full_like(ys, np.nan)
            oks = np.isfinite(xs).all(1)
            pred[oks] = model.predict(xs[oks])
            store[t] = pred
        print(f"  {t:4s} r {r_rec[t]:.3f}   time-shuffled {r_ctrl[t]:+.3f}")
    _, truth, _ = pairs.frames("R1", np.array([show]), LAGS)
    n = len(truth)

    comb = honeycomb(5)
    pw, ph = comb[0].shape[1], comb[0].shape[0]
    gap = 22
    cols = 1 + len(TYPES)
    W = 48 * 2 + cols * pw + (cols - 1) * gap
    H = 700
    f_title, f_body, f_lab, f_small, f_num = font(30, True), font(19), font(19, True), font(16), font(18, True)
    frames = []
    for k in range(n):
        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im)
        y = pipeline_strip(d, 48, 24, W - 96, "decode") + 26
        d.text((48, y), "Decoding: the picture read out of one cell type", font=f_title, fill=INK)
        y += 44
        d.text((48, y), "A linear decoder per cell type of model zero, fitted on Sintel scenes and scored on held-out "
                        "ones. Retina and lamina give the picture", font=f_body, fill=MUTED)
        y += 26
        d.text((48, y), "back whole; deeper types only in part. Control, activity shuffled in time: the scene survives, "
                        "the frame does not.", font=f_body, fill=MUTED)
        y += 44
        xs = [48 + i * (pw + gap) for i in range(cols)]
        d.text((xs[0], y), "clip A", font=f_lab, fill=INK)
        d.text((xs[0], y + 24), "what the eye saw", font=f_small, fill=MUTED)
        for i, (t, depth) in enumerate(TYPES, start=1):
            d.text((xs[i], y), t, font=f_lab, fill=INK)
            d.text((xs[i], y + 24), depth, font=f_small, fill=MUTED)
        d.line([xs[1] - gap // 2, y, xs[1] - gap // 2, y + 2 * ph + 150], fill=LINE, width=1)
        y += 56
        im.paste(hex_image(truth[k], comb), (xs[0], y))
        for i, (t, _) in enumerate(TYPES, start=1):
            im.paste(hex_image(np.nan_to_num(rec[t][k], nan=0.5), comb), (xs[i], y))
            d.text((xs[i], y + ph + 6), f"r = {r_rec[t]:.2f}", font=f_num, fill=GOOD)
        y += ph + 44
        d.text((xs[0], y + ph // 2 - 34), "control", font=f_lab, fill=INK)
        d.text((xs[0], y + ph // 2 - 8), "activity shuffled", font=f_small, fill=MUTED)
        d.text((xs[0], y + ph // 2 + 12), "in time", font=f_small, fill=MUTED)
        for i, (t, _) in enumerate(TYPES, start=1):
            im.paste(hex_image(np.nan_to_num(ctrl[t][k], nan=0.5), comb), (xs[i], y))
            d.text((xs[i], y + ph + 6), f"r = {r_ctrl[t]:+.2f}", font=f_num, fill=BAD)
        d.text((48, H - 40), f"MaleCNS model zero · Sintel, split by scene · frame {k + 1}/{n} · ridge on frames t and t−1 "
                             "· r over the held-out scenes", font=f_small, fill=MUTED)
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
