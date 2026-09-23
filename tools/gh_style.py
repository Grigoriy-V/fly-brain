"""House style for the repository's public figures (GitHub README and article).

Light ground, plain type, every figure opens with the project's pipeline as a
strip of stages with the one the figure belongs to highlighted, so a reader
always knows where in the work a picture sits. Hex frames are drawn as a
honeycomb on a geometrically regular lattice; grey video is always on a fixed
0..1 scale (ISS-0012: never normalise a cell on its own).
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BG = (255, 255, 255)
INK = (28, 31, 36)
MUTED = (110, 116, 126)
LINE = (214, 218, 224)
HIGHLIGHT = (37, 99, 235)
HIGHLIGHT_BG = (232, 239, 253)
GOOD = (21, 128, 61)
BAD = (185, 28, 28)

PIPELINE = [
    ("connectome", "connectome"),
    ("model", "model zero"),
    ("decode", "decoding"),
    ("invert", "inversion"),
    ("render", "state → video"),
    ("prior", "state prior"),
    ("static", "static scene"),
]


def font(size: int, bold: bool = False):
    names = (["C:/Windows/Fonts/seguisb.ttf", "C:/Windows/Fonts/arialbd.ttf", "DejaVuSans-Bold.ttf"] if bold
             else ["C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf", "DejaVuSans.ttf"])
    for f in names:
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()


def pipeline_strip(d: ImageDraw.ImageDraw, x0: int, y0: int, width: int, current: str) -> int:
    """Draw the stage strip; returns its bottom y."""
    f = font(17)
    fb = font(17, True)
    n = len(PIPELINE)
    gap = 18
    w = (width - gap * (n - 1)) / n
    h = 34
    for i, (key, label) in enumerate(PIPELINE):
        x = x0 + i * (w + gap)
        on = key == current
        d.rounded_rectangle([x, y0, x + w, y0 + h], radius=7, fill=HIGHLIGHT_BG if on else BG,
                            outline=HIGHLIGHT if on else LINE, width=2 if on else 1)
        ff = fb if on else f
        tw = d.textlength(label, font=ff)
        d.text((x + (w - tw) / 2, y0 + 6), label, font=ff, fill=HIGHLIGHT if on else MUTED)
        if i < n - 1:
            ax = x + w + 4
            d.text((ax, y0 + 5), "›", font=f, fill=LINE)
    return y0 + h


def honeycomb(pix: int):
    """Pixel -> hexal index on a regular lattice, and a mask with thin gaps
    between cells."""
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
    dd, idx = cKDTree(np.stack([x, y], 1)).query(np.stack([gx.ravel(), gy.ravel()], 1), k=2)
    near, second = dd[:, 0].reshape(h, w), dd[:, 1].reshape(h, w)
    inside = near <= s / np.sqrt(3) * 1.02
    if pix < 8:                      # below this the seams alias into stripes
        return idx[:, 0].reshape(h, w), inside
    edge = (second - near) < s * 0.08
    return idx[:, 0].reshape(h, w), inside & ~edge


def hex_image(values: np.ndarray, comb, cmap=None, bg=BG) -> Image.Image:
    """721 values -> honeycomb image. Grey on a fixed 0..1 scale unless a
    colormap is given."""
    index, mask = comb
    v = np.asarray(values, dtype=float)[index]
    if cmap is None:
        g = (np.clip(v, 0, 1) * 255).astype(np.uint8)
        rgb = np.stack([g, g, g], -1)
    else:
        rgb = (cmap(np.clip(v, 0, 1))[..., :3] * 255).astype(np.uint8)
    rgb[~mask] = bg
    return Image.fromarray(rgb)


def save_gif(frames: list[Image.Image], path, fps: int = 8) -> None:
    pal = [f.quantize(colors=96, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for f in frames]
    pal[0].save(path, save_all=True, append_images=pal[1:], duration=int(1000 / fps), loop=0, optimize=True)
