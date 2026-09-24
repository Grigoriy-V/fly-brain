"""GitHub figure: a static scene drawn from noise, beside real pictures and their states.

    python tools/fig_gh_static.py --out docs/figures/static_draws

Stage "static scene" (article part 2, section 11). The flow of step 29 draws a
static brain state - T4a+T4b, temporal coefficient 0, 721 x 2 = 1,442 numbers -
from noise. There is no trained renderer yet, so every row goes through the
same least-squares map from those 1,442 numbers to a picture, fitted here on
the training split (held-out r is printed; the run recorded 0.941). Rows:
held-out pictures (the window's mean frame), real held-out states through the
map, and fresh draws through the map. The held-out clips are chosen by eye for
recognisable content (`--clips`, corpus indices, picked from the contact sheet of
tools/pick_static_examples.py); the draws are the first six, not picked. A PNG.

Limits, stated on the figure: the map blurs by construction (it draws the
conditional mean), and real states are the local PCA-2048 reconstruction
(89.5 percent of variance) - the true states live on the Modal volume.
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

from gh_style import BG, GOOD, INK, LINE, MUTED, font, hex_image, honeycomb, pipeline_strip  # noqa: E402
from flydream.generate.geom29 import real_static  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ROOT / "docs" / "figures" / "static_draws"))
    p.add_argument("--clips", type=int, nargs="+", default=[51, 6008, 10, 7846, 9351, 10220],
                   help="held-out corpus indices to show, recognisable by eye")
    p.add_argument("--labels", nargs="+", default=["face", "person", "eye", "guitar", "rowing boat", "sky"])
    p.add_argument("--ridge", type=float, default=1.0)
    a = p.parse_args(argv)

    pca = np.load(ROOT / "data" / "prior19" / "pca_ab2048.npz")
    lat = np.load(ROOT / "data" / "prior19" / "pca_ab2048_latent.npz")
    vids = np.load(ROOT / "data" / "corpus18" / "videos.npz", mmap_mode="r")["videos"]
    draw = np.load(ROOT / "data" / "prior29" / "draw29.npz")["draw"]

    def dc(split):
        n = len(lat[f"z_{split}"])
        x = np.concatenate([real_static(pca, {f"z_{split}": lat[f"z_{split}"][i:i + 2000]}, f"z_{split}",
                                        len(lat[f"z_{split}"][i:i + 2000])) for i in range(0, n, 2000)])
        return x.astype(np.float32)

    def frames_of(split):
        idx = lat[f"index_{split}"]
        return np.stack([vids[i, :40].mean(0) for i in idx]).astype(np.float32)

    Xtr, Xte = dc("train"), dc("test")
    Ytr, Yte = frames_of("train"), frames_of("test")
    # the flow was trained on coefficient 0 of the same z-scored DCT maps the PCA was fitted on, so a draw
    # and a real state share one space: no rescaling between them
    print(f"real DC per type: mean {Xtr.reshape(-1, 2, 721).mean((0, 2))}, sd {Xtr.reshape(-1, 2, 721).std((0, 2))}; "
          f"draws: mean {draw.reshape(-1, 2, 721).mean((0, 2))}, sd {draw.reshape(-1, 2, 721).std((0, 2))}")
    A = np.hstack([Xtr, np.ones((len(Xtr), 1), np.float32)])
    W = np.linalg.solve(A.T @ A + a.ridge * np.eye(A.shape[1], dtype=np.float32), A.T @ Ytr)

    def render(x):
        return np.hstack([x, np.ones((len(x), 1), np.float32)]) @ W

    pred = render(Xte)
    r_held = float(np.mean([np.corrcoef(u, v)[0, 1] for u, v in zip(pred, Yte)]))
    print(f"least-squares map, held-out r to the mean frame: {r_held:.3f} (recorded 0.941)")

    pos = {int(c): j for j, c in enumerate(lat["index_test"])}
    held = [pos[c] for c in a.clips]
    k = len(held)
    rows_data = {"picture": Yte[held], "real": render(Xte[held]), "draw": render(draw[:k].astype(np.float32))}

    from flydream.generate import edges18
    nb = np.asarray(edges18.L.neighbour_index(721))
    judged = {key: [edges18.edges(img[None], nb) for img in val] for key, val in rows_data.items()}
    flat = {key: float(np.mean([e["frac_flat"] for e in v])) for key, v in judged.items()}
    coh = {key: float(np.mean([e["neigh_r"] for e in v])) for key, v in judged.items()}
    print("flat fraction:", {kk: round(vv, 3) for kk, vv in flat.items()})
    print("neighbour coherence:", {kk: round(vv, 3) for kk, vv in coh.items()})

    comb = honeycomb(5)
    pw, ph = comb[0].shape[1], comb[0].shape[0]
    Wd, lab_w = 1216, 250
    m = k
    gap = (Wd - 96 - lab_w - m * pw) // (m - 1)
    H = 900
    f_title, f_body, f_lab, f_small = font(30, True), font(19), font(19, True), font(16)
    rows = [("picture", "held-out picture", "the eye's view, window mean", INK),
            ("real", "its real state", "→ the same renderer", INK),
            ("draw", "a fresh draw", "noise → flow → renderer", GOOD)]

    for s in range(1):
        im = Image.new("RGB", (Wd, H), BG)
        d = ImageDraw.Draw(im)
        y = pipeline_strip(d, 48, 24, Wd - 96, "static") + 26
        d.text((48, y), "A static scene from noise: 721 × 2 numbers, no time axis", font=f_title, fill=INK)
        y += 44
        d.text((48, y), "A flow on the hexagonal lattice draws the static part of a T4 brain state from noise. Top two "
                        "rows: real held-out", font=f_body, fill=MUTED)
        y += 26
        d.text((48, y), "pictures and their states; bottom: fresh draws, tied to no clip, through the same renderer.", font=f_body, fill=MUTED)
        y += 72
        xs = [48 + lab_w + i * (pw + gap) for i in range(m)]
        d.line([xs[0] - 14, y - 26, xs[0] - 14, y + 3 * (ph + 40) + 20], fill=LINE, width=1)
        for key, name, sub, colour in rows:
            if key == "draw":
                d.line([48, y - 20, Wd - 48, y - 20], fill=LINE, width=1)
                y += 20
            d.text((48, y + ph // 2 - 34), name, font=f_lab, fill=colour)
            d.text((48, y + ph // 2 - 8), sub, font=f_small, fill=MUTED)
            if True:
                d.text((48, y + ph // 2 + 14), f"flat {100 * flat[key]:.1f} %, coherence {coh[key]:.2f}",
                       font=f_small, fill=MUTED)
            for i in range(m):
                im.paste(hex_image(rows_data[key][i], comb), (xs[i], y))
                if key == "picture":
                    d.text((xs[i], y - 22), a.labels[i], font=f_small, fill=MUTED)
            y += ph + 40
        d.text((48, H - 62), "Renderer: a least-squares map from the 1,442 numbers to a picture "
                             f"(held-out r = {r_held:.2f}); it blurs by construction, so structure is judged, "
                             "not sharpness.", font=f_small, fill=MUTED)
        d.text((48, H - 38), f"Draws: the first {k}, not picked · real states: PCA-2048 reconstruction · UCF101, unseen "
                             "classes · run 2026-09-21_prior29_static_ab", font=f_small, fill=MUTED)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out.with_suffix(".png"))
    print(f"{out.with_suffix('.png')}  {out.with_suffix('.png').stat().st_size / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
