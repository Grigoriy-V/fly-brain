"""Procedural stimuli on the eye's 721-column lattice (ROADMAP item 13A).

    python -m flydream.generate.stimuli --n-per-class 4 --figure

The learned inverse of 13A is trained on pairs (state, video) the frozen
brain makes from videos, and the videos decide what the state manifold
covers: 20k rotated Sintel clips are still 23 scenes. These classes widen
it with the stimuli the physiology literature uses — every one a function
of the column's position (x, y) and time, drawn with random parameters:

    edge      a moving dark/bright edge (direction, speed, polarity)
    bar       a moving bar (width, direction, speed, polarity)
    grating   a drifting sine or square grating (frequency, orientation, speed)
    dots      random dots moving coherently (density, size, direction, speed)
    flow      a 1/f texture expanding, contracting or rotating about the centre
    noise     white or spatially 1/f noise, new each frame or drifting
    flash     grey, a step to a level, grey (onset, duration, level)
    texture   a 1/f texture translating (direction, speed)
    mixture   the sum of two of the above, renormalised

Luminance is in [0, 1] around 0.5, `frames` long (the training window
40 + 5). Each clip records its class and parameters so the split can hold
whole classes out. Coordinates come from flyvis's hex utilities (columns
at u, v in [-15, 15]; `hex_to_pixel` for the cartesian layout).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

CLASSES = ("edge", "bar", "grating", "dots", "flow", "noise", "flash", "texture", "mixture")


def hex_xy(n: int = 721) -> np.ndarray:
    """(n, 2) cartesian column positions, unit spacing, centred."""
    from flyvis.utils.hex_utils import get_hex_coords, get_hextent, hex_to_pixel

    u, v = get_hex_coords(get_hextent(n))
    x, y = hex_to_pixel(np.asarray(u), np.asarray(v))
    xy = np.stack([np.asarray(x, np.float64), np.asarray(y, np.float64)], 1)
    xy = xy - xy.mean(0)
    # unit = one column spacing (hex_to_pixel returns sqrt(3) per step): the eye is then radius 15
    d = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
    return xy / d[d > 0].min()


def _unit(theta: float) -> np.ndarray:
    return np.array([np.cos(theta), np.sin(theta)])


def _pink(xy: np.ndarray, rng: np.random.Generator, beta: float = 1.5, grid: int = 48, span: float = 96.0) -> np.ndarray:
    """A 1/f^beta texture sampled at the columns, values ~ N(0, 1)."""
    f = np.fft.fftfreq(grid)[:, None] ** 2 + np.fft.fftfreq(grid)[None, :] ** 2
    with np.errstate(divide="ignore"):
        amp = np.where(f > 0, f ** (-beta / 2), 0.0)
    img = np.fft.ifft2(np.fft.fft2(rng.standard_normal((grid, grid))) * amp).real
    img = (img - img.mean()) / (img.std() + 1e-9)
    gx = ((xy[:, 0] / span + 0.5) * (grid - 1)).clip(0, grid - 1)
    gy = ((xy[:, 1] / span + 0.5) * (grid - 1)).clip(0, grid - 1)
    return img[gy.round().astype(int), gx.round().astype(int)]


def _texture_field(xy: np.ndarray, rng: np.random.Generator, grid: int = 64) -> tuple[np.ndarray, float]:
    """A periodic 1/f^1.5 texture on a grid, to be sampled at moving positions;
    the grid spans 128 column units, so a texel is two columns wide."""
    f = np.fft.fftfreq(grid)[:, None] ** 2 + np.fft.fftfreq(grid)[None, :] ** 2
    with np.errstate(divide="ignore"):
        amp = np.where(f > 0, f ** -0.75, 0.0)
    img = np.fft.ifft2(np.fft.fft2(rng.standard_normal((grid, grid))) * amp).real
    img = (img - img.mean()) / (img.std() + 1e-9)
    return img, 128.0         # grid, its span in column units (periodic)


def _sample(img: np.ndarray, span: float, pos: np.ndarray) -> np.ndarray:
    grid = img.shape[0]
    g = ((pos / span) % 1.0) * grid
    return img[g[:, 1].astype(int) % grid, g[:, 0].astype(int) % grid]


def clip(cls: str, frames: int, rng: np.random.Generator, xy: np.ndarray | None = None, dt: float = 0.02
         ) -> tuple[np.ndarray, dict]:
    """One clip (frames, 721) in [0, 1] and its parameters."""
    xy = hex_xy() if xy is None else xy
    t = np.arange(frames) * dt
    p: dict = {"class": cls}
    if cls == "mixture":
        a, b = rng.choice([c for c in CLASSES if c != "mixture"], 2, replace=False)
        va, pa = clip(a, frames, rng, xy, dt)
        vb, pb = clip(b, frames, rng, xy, dt)
        v = 0.5 + ((va - 0.5) + (vb - 0.5)) / np.sqrt(2)
        p.update({"parts": [pa, pb]})
        return np.clip(v, 0, 1).astype(np.float32), p
    if cls == "edge":
        th, speed, pol = rng.uniform(0, 2 * np.pi), rng.uniform(20, 45), rng.choice([-1, 1])
        start = -16.0 - rng.uniform(0, 0.2) * speed          # enters the eye within the first 0.2 s
        proj = xy @ _unit(th)
        pos = start + speed * t
        v = 0.5 + 0.5 * pol * np.tanh((proj[None, :] - pos[:, None]) * 2)   # bright side leads (pol) or trails
        p.update(theta=float(th), speed=float(speed), polarity=int(pol), start=float(start))
    elif cls == "bar":
        th, speed, pol, w = rng.uniform(0, 2 * np.pi), rng.uniform(20, 45), rng.choice([-1, 1]), rng.uniform(1.5, 5)
        start = -16.0 - rng.uniform(0, 0.2) * speed
        proj = xy @ _unit(th)
        pos = start + speed * t
        inside = np.abs(proj[None, :] - pos[:, None]) < w / 2
        v = 0.5 + 0.5 * pol * inside
        p.update(theta=float(th), speed=float(speed), polarity=int(pol), width=float(w), start=float(start))
    elif cls == "grating":
        th, sf, tf = rng.uniform(0, np.pi), rng.uniform(0.03, 0.18), rng.uniform(-6, 6)
        square, contrast = bool(rng.random() < 0.4), rng.uniform(0.3, 1.0)
        phase = 2 * np.pi * (sf * (xy @ _unit(th))[None, :] - tf * t[:, None])
        g = np.sin(phase)
        v = 0.5 + 0.5 * contrast * (np.sign(g) if square else g)
        p.update(theta=float(th), spatial_freq=float(sf), temporal_freq=float(tf), square=square, contrast=float(contrast))
    elif cls == "dots":
        n, r, th, speed = int(rng.integers(20, 120)), rng.uniform(0.8, 2.0), rng.uniform(0, 2 * np.pi), rng.uniform(3, 30)
        coh = rng.uniform(0.5, 1.0)
        pos0 = rng.uniform(-20, 20, (n, 2))
        dirs = np.where(rng.random(n)[:, None] < coh, _unit(th)[None, :], rng.uniform(-1, 1, (n, 2)))
        pol = rng.choice([-1, 1])
        v = np.full((frames, len(xy)), 0.5)
        for i, ti in enumerate(t):
            pos = (pos0 + speed * ti * dirs + 20) % 40 - 20
            d2 = ((xy[:, None, :] - pos[None, :, :]) ** 2).sum(-1)
            v[i] += 0.5 * pol * (d2.min(1) < r * r)
        p.update(n=n, radius=float(r), theta=float(th), speed=float(speed), coherence=float(coh), polarity=int(pol))
    elif cls == "flow":
        kind = rng.choice(["expand", "contract", "rotate"])
        rate = rng.uniform(0.3, 1.5) * (1 if kind != "contract" else -1)
        img, span = _texture_field(xy, rng)
        v = np.zeros((frames, len(xy)))
        for i, ti in enumerate(t):
            if kind == "rotate":
                c, s = np.cos(rate * ti), np.sin(rate * ti)
                pos = xy @ np.array([[c, -s], [s, c]]).T
            else:
                pos = xy * np.exp(-rate * ti)
            v[i] = _sample(img, span, pos)
        v = 0.5 + 0.25 * v
        p.update(kind=str(kind), rate=float(rate))
    elif cls == "noise":
        kind, sd = rng.choice(["white", "pink"]), rng.uniform(0.08, 0.25)
        drift = bool(rng.random() < 0.5)
        if kind == "white":
            v = 0.5 + sd * rng.standard_normal((frames, len(xy)))
            if drift:
                keep = rng.uniform(0.6, 0.95)
                for i in range(1, frames):
                    v[i] = 0.5 + keep * (v[i - 1] - 0.5) + np.sqrt(1 - keep ** 2) * (v[i] - 0.5)
        else:
            if drift:
                img, span = _texture_field(xy, rng)
                th, speed = rng.uniform(0, 2 * np.pi), rng.uniform(2, 15)
                v = np.stack([_sample(img, span, xy + speed * ti * _unit(th)) for ti in t])
            else:
                v = np.stack([_pink(xy, rng) for _ in t])
            v = 0.5 + sd * v
        p.update(kind=str(kind), sd=float(sd), drift=drift)
    elif cls == "flash":
        on, dur, level = int(rng.integers(3, frames // 2)), int(rng.integers(1, 12)), float(rng.choice([0.0, 1.0, rng.uniform(0, 1)]))
        v = np.full((frames, len(xy)), 0.5)
        v[on:on + dur] = level
        p.update(onset=on, duration=dur, level=level)
    elif cls == "texture":
        img, span = _texture_field(xy, rng)
        th, speed, contrast = rng.uniform(0, 2 * np.pi), rng.uniform(2, 30), rng.uniform(0.15, 0.35)
        v = 0.5 + contrast * np.stack([_sample(img, span, xy + speed * ti * _unit(th)) for ti in t])
        p.update(theta=float(th), speed=float(speed), contrast=float(contrast))
    else:
        raise ValueError(cls)
    return np.clip(v, 0, 1).astype(np.float32), p


def make_set(n_per_class: int, frames: int, seed: int, classes=CLASSES) -> tuple[np.ndarray, list[dict]]:
    rng = np.random.default_rng(seed)
    xy = hex_xy()
    videos, params = [], []
    for cls in classes:
        for k in range(n_per_class):
            v, p = clip(cls, frames, rng, xy)
            p["index_in_class"] = k
            videos.append(v); params.append(p)
    return np.stack(videos), params


def main(argv=None) -> int:
    from flydream.model import ROOT

    p = argparse.ArgumentParser()
    p.add_argument("--n-per-class", type=int, default=4)
    p.add_argument("--frames", type=int, default=45)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="npz path; default data/stimuli/procedural_<n>_<seed>.npz")
    p.add_argument("--figure", action="store_true", help="also draw one frame per class to reports/figures")
    a = p.parse_args(argv)
    videos, params = make_set(a.n_per_class, a.frames, a.seed)
    out = Path(a.out) if a.out else ROOT / "data" / "stimuli" / f"procedural_{a.n_per_class}x{len(CLASSES)}_s{a.seed}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, videos=videos, classes=np.array([q["class"] for q in params]),
                        params=np.array([json.dumps(q) for q in params]))
    print(f"wrote {out}: {videos.shape}, {videos.nbytes / 1e6:.0f} MB")
    if a.figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from flydream.decode.hexraster import to_raster

        fig, ax = plt.subplots(3, len(CLASSES), figsize=(1.5 * len(CLASSES), 4.6), facecolor="#fcfcfb")
        for j, cls in enumerate(CLASSES):
            i = j * a.n_per_class
            for r, f in enumerate([5, a.frames // 2, a.frames - 6]):
                ax[r, j].imshow(to_raster(videos[i, f], 721, 3, fill=np.nan), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                ax[r, j].set_xticks([]); ax[r, j].set_yticks([])
                for sp in ax[r, j].spines.values():
                    sp.set_visible(False)
                if r == 0:
                    ax[r, j].set_title(cls, fontsize=9)
                if j == 0:
                    ax[r, j].set_ylabel(f"frame {f}", fontsize=8)
        fig.suptitle("Procedural stimuli on the 721-column eye (one example per class, three frames)", fontsize=9, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fp = ROOT / "reports" / "figures" / "2026-09-19_procedural_stimuli.png"
        fig.savefig(fp, dpi=150, facecolor="#fcfcfb"); plt.close(fig)
        print(f"wrote {fp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
