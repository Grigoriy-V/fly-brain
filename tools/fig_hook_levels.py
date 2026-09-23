"""A post-format GIF: a video read back out of a fly brain model, layer by layer.

    python tools/fig_hook_levels.py --out docs/figures/hook_levels

For LinkedIn and the article, not for measurement. 4:5 frame (1080 x 1350), dark
ground, the hex lattice drawn as a honeycomb. Top: what the eye saw. Below, three
layers from the eye inward; each row is that layer's own activity (colour, each
type on its own fixed range over the whole clip) and the video recovered from
that layer alone by inverting the frozen model (grey, fixed 0..1 scale, the
same scale as the input). Numbers are the inversion runs'
`2026-09-19_malecns_invert_<type>_s3` (r to the input, control = another
clip's state as the target).

`--activity` is an npz with `video` (40, 721) and one (40, 721) array per type,
in the flyvis hex order - the frozen model's response to the same clip.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

W, H = 1080, 1350
BG = (11, 13, 17)
FG = (236, 238, 241)
DIM = (140, 146, 156)
ACCENT = (255, 176, 59)
ROWS = [("R1", "Photoreceptors", "the eye itself"),
        ("Tm9", "Medulla", "two synapses in"),
        ("T4a", "Motion detectors", "T4, deepest layer here")]


def font(size: int, bold: bool = False):
    name = "C:/Windows/Fonts/seguisb.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"
    for f in (name, "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()


def honeycomb(pix: int):
    """Pixel -> hexal index on a geometrically regular lattice, plus a mask
    with thin gaps between cells so the lattice reads as a honeycomb."""
    from scipy.spatial import cKDTree
    from flyvis.utils.hex_utils import get_hex_coords, get_hextent

    u, v = get_hex_coords(get_hextent(721))
    s = float(pix)
    x = v * (np.sqrt(3) / 2) * s
    y = (u + v / 2.0) * s
    x0, y0 = x.min() - s, y.min() - s
    w = int(np.ceil(x.max() + s - x0)) + 1
    h = int(np.ceil(y.max() + s - y0)) + 1
    gx, gy = np.meshgrid(np.arange(w) + x0, np.arange(h) + y0)
    d, idx = cKDTree(np.stack([x, y], 1)).query(np.stack([gx.ravel(), gy.ravel()], 1), k=2)
    near, second = d[:, 0].reshape(h, w), d[:, 1].reshape(h, w)
    inside = near <= s / np.sqrt(3) * 1.02
    edge = (second - near) < s * 0.10
    return idx[:, 0].reshape(h, w), inside & ~edge


def colorise(values: np.ndarray, index, mask, cmap) -> Image.Image:
    rgb = (cmap(values[index])[..., :3] * 255).astype(np.uint8)
    rgb[~mask] = BG
    return Image.fromarray(rgb)


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import colormaps

    p = argparse.ArgumentParser()
    p.add_argument("--activity", default=str(ROOT / "data" / "figures" / "act_s3_malecns_flyvis.npz"),
                   help="written by tools/cmp_layers_flyvis.py")
    p.add_argument("--prefix", default="malecns_", help="key prefix in the npz")
    p.add_argument("--out", default=str(ROOT / "docs" / "figures" / "hook_levels"))
    p.add_argument("--fps", type=int, default=10)
    a = p.parse_args(argv)

    A = np.load(a.activity)
    video = np.clip(A["video"][:40], 0, 1)
    rec = {}
    score = {}
    for t, _, _ in ROWS:
        d = ROOT / "data" / "generate" / f"2026-09-19_malecns_invert_{t}_s3"
        rec[t] = np.clip(np.load(d / "recovered.npz")["recovered"], 0, 1)
        score[t] = json.loads((d / "meta.json").read_text(encoding="utf-8"))["inversion"]

    gray, heat = colormaps["gray"], colormaps["inferno"]
    big_i, big_m = honeycomb(9)
    sm_i, sm_m = honeycomb(7)
    act_norm = {}
    for t, _, _ in ROWS:
        x = A[a.prefix + t]
        lo, hi = np.nanpercentile(x, 1), np.nanpercentile(x, 99)
        act_norm[t] = np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)

    f_title, f_sub, f_lab, f_small, f_num = font(50, True), font(27), font(30, True), font(22), font(26, True)
    frames = []
    n = len(video)
    for k in range(n):
        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im)
        d.text((60, 48), "Reading a video back out", font=f_title, fill=FG)
        d.text((60, 106), "of a fly's brain", font=f_title, fill=ACCENT)
        d.text((60, 176), "A model of the fly visual system, wired by the real connectome.", font=f_sub, fill=DIM)
        d.text((60, 210), "Each row: one layer's activity, and the video rebuilt from it alone.", font=f_sub, fill=DIM)

        eye = colorise(video[k], big_i, big_m, gray)
        ex = 430 + (sm_i.shape[1] + 90 + sm_i.shape[1]) // 2 - eye.width // 2
        im.paste(eye, (ex, 258))
        d.text((60, 258 + eye.height // 2 - 34), "What the", font=f_lab, fill=FG)
        d.text((60, 258 + eye.height // 2 + 2), "eye saw", font=f_lab, fill=FG)
        y = 258 + eye.height + 14
        d.text((430, y), "neural activity", font=f_small, fill=DIM)
        d.text((430 + sm_i.shape[1] + 90, y), "rebuilt video", font=f_small, fill=DIM)
        y += 40
        for t, name, sub in ROWS:
            act = colorise(np.nan_to_num(act_norm[t][k], nan=0.0), sm_i, sm_m, heat)
            out = colorise(rec[t][k], sm_i, sm_m, gray)
            d.text((60, y + 36), name, font=f_lab, fill=FG)
            d.text((60, y + 76), sub, font=f_small, fill=DIM)
            ax = 430
            im.paste(act, (ax, y))
            d.text((ax + act.width + 24, y + act.height // 2 - 34), "→", font=f_title, fill=DIM)
            bx = ax + act.width + 90
            im.paste(out, (bx, y))
            d.text((60, y + 118), f"rebuilt at r = {score[t]:.2f}", font=f_num, fill=ACCENT)
            y += act.height + 16
        bar = int((W - 120) * (k + 1) / n)
        d.rectangle([60, H - 22, 60 + bar, H - 16], fill=ACCENT)
        frames.append(im)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pal = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for f in frames]
    pal[0].save(out.with_suffix(".gif"), save_all=True, append_images=pal[1:], duration=int(1000 / a.fps),
                loop=0, optimize=True)
    frames[n // 2].save(out.with_suffix(".png"))
    try:
        import imageio.v2 as imageio
        with imageio.get_writer(out.with_suffix(".mp4"), fps=a.fps, codec="libx264", quality=8,
                                macro_block_size=8) as wr:
            for _ in range(3):
                for f in frames:
                    wr.append_data(np.asarray(f))
    except Exception as e:  # mp4 is a convenience, the gif is the artefact
        print("mp4 skipped:", e)
    for s in (".gif", ".png", ".mp4"):
        f = out.with_suffix(s)
        if f.exists():
            print(f"{f}  {f.stat().st_size / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
