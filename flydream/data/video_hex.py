"""Ordinary video through the fly's eye: a video file → (frames, 721) luminance.

The prior of item 17 learned states from 19 Sintel scenes; the next set is
ordinary video (ROADMAP 18). For a new clip to be comparable with the ones
13A/13B already use, it must pass through **the same eye and the same
processing** as `flyvis`'s `AugmentedSintel`, so every stage here is flyvis's
own call in flyvis's order:

    frames (T, H, W) in [0, 1]
      → `rendering.utils.split`   centre-crop 0.7 of the width, then cut it
                                  into `vertical_splits` overlapping views
      → `rendering.BoxEye`        extent 15, kernel 13 → 721 hexals
      → `augmentation.Interpolate` native framerate → 1/dt, linear
      → `augmentation.HexRotate` / `HexFlip`   (the lattice, not the image)

What is *not* flyvis's is the selection: ordinary video contains cuts, static
tripod shots and black frames, none of which exist in the Sintel set, and all
of which teach a prior the wrong thing. `motion`, `contrast` and `cut_score`
below are the three numbers the corpus builder selects on.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

EXTENT, KERNEL = 15, 13                      # flyvis's boxfilter=dict(extent=15, kernel_size=13)
CROP, SPLITS, WIDTH = 0.7, 3, 1024                        # AugmentedSintel's center_crop_fraction and vertical_splits


def eye(extent: int = EXTENT, kernel_size: int = KERNEL):
    from flyvis.datasets.rendering import BoxEye

    return BoxEye(extent=extent, kernel_size=kernel_size)


LUMA = np.array([0.299, 0.587, 0.114], np.float32)      # PIL's "L" conversion, which flyvis's sample_lum uses


def to_luminance(rgb: np.ndarray) -> np.ndarray:
    """RGB (…, 3) in 0-255 → luminance in [0, 1] exactly as `Image.convert("L")`
    does it, rounding to a byte first. A plain channel mean is *not* the same
    and shifts the whole clip against the Sintel set (checked 2026-09-20:
    r 0.998 with the mean, 1.000 with these weights)."""
    a = np.asarray(rgb, np.float32)
    g = a @ LUMA if a.ndim >= 3 and a.shape[-1] == 3 else a
    return (np.floor(g + 0.5).clip(0, 255) / 255.0).astype(np.float32)


def read_gray(path, *, max_frames: int | None = None, skip: int = 0) -> tuple[np.ndarray, float]:
    """(T, H, W) float32 luminance in [0, 1] and the file's framerate."""
    import imageio.v3 as iio

    meta = iio.immeta(path, plugin="FFMPEG")
    fps = float(meta.get("fps") or 25.0)
    out = []
    for i, frame in enumerate(iio.imiter(path, plugin="FFMPEG")):
        if i < skip:
            continue
        out.append(to_luminance(frame))
        if max_frames is not None and len(out) >= max_frames:
            break
    if not out:
        raise ValueError(f"no frames decoded from {path}")
    return np.stack(out).astype(np.float32), fps


def to_hexals(gray: np.ndarray, fps: float, dt: float, box=None, *, splits: int = SPLITS,
              crop: float = CROP, width: int = WIDTH) -> np.ndarray:
    """(T, H, W) frames → (splits, T', 721) hexal luminance at 1/dt.

    The frame is first resized to `width` pixels across, aspect ratio kept.
    That is Sintel's own width, and it is what makes an object in a new clip
    subtend the same angle and move at the same speed across the eye as one
    in the clips 13B already knows — T4/T5 are tuned to speed, so rendering a
    320-pixel-wide video at its own scale would hand the prior a different
    world, not a bigger one. It is also required: the eye's window is 417
    pixels wide and `split` cannot cut that out of a 224-pixel crop."""
    from flyvis.datasets.augmentation.temporal import Interpolate
    from flyvis.datasets.rendering.utils import split as hsplit

    box = box or eye()
    x = torch.as_tensor(np.asarray(gray, np.float32))
    if width and x.shape[-1] != width:
        h = max(int(box.min_frame_size[0]), int(round(x.shape[-2] * width / x.shape[-1])))
        x = torch.nn.functional.interpolate(x[:, None], size=(h, int(width)), mode="bilinear",
                                            align_corners=False, antialias=True)[:, 0]
    views = hsplit(x, int(box.min_frame_size[1]) + 2 * box.kernel_size, splits, crop)   # (splits, T, H, W)
    hexals = box(views).squeeze(2)                                                      # (splits, T, 721)
    if abs(fps - 1 / dt) > 1e-6:
        hexals = Interpolate(fps, 1 / dt, mode="linear").transform(hexals, dim=1)
    return hexals.cpu().numpy().astype(np.float32)


def augment(hexals: np.ndarray, *, n_rot: int = 0, flip_axis: int | None = None, extent: int = EXTENT) -> np.ndarray:
    """The hex lattice rotated by n_rot·60° and optionally flipped — flyvis's
    own augmentation, so a rotated clip is the same stimulus the network was
    trained to see from another heading."""
    from flyvis.datasets.augmentation.hex import HexFlip, HexRotate

    x = torch.as_tensor(np.asarray(hexals, np.float32))
    single = x.ndim == 2
    if single:
        x = x[None]
    out = []
    rot = HexRotate(extent, n_rot=int(n_rot), p_rot=1.0)
    flip = HexFlip(extent, axis=int(flip_axis or 0), p_flip=1.0, flip_axes=[0, 1, 2, 3])
    for seq in x:
        y = rot(seq[:, None])[:, 0] if n_rot else seq
        if flip_axis is not None:
            y = flip(y[:, None])[:, 0]
        out.append(y)
    y = torch.stack(out).cpu().numpy().astype(np.float32)
    return y[0] if single else y


# ----------------------------------------------------------------- selection


def motion(x: np.ndarray) -> float:
    """Mean |Δ| between consecutive frames in hexal space. Real Sintel clips
    measure 0.014-0.047 here; a tripod shot is an order of magnitude below."""
    return float(np.abs(np.diff(np.asarray(x, np.float32), axis=-2)).mean())


def contrast(x: np.ndarray) -> float:
    """Spatial sd over the whole clip (Sintel clips: 0.07-0.28)."""
    return float(np.asarray(x, np.float32).std())


def cut_score(x: np.ndarray) -> float:
    """Largest frame-to-frame jump over the median one. A hard cut is a flash
    to the whole eye, which the T4/T5 types answer with a transient no real
    motion produces, so clips above the builder's threshold are dropped."""
    d = np.abs(np.diff(np.asarray(x, np.float32), axis=-2)).mean(-1)
    med = float(np.median(d))
    return float(d.max() / (med + 1e-6))


def windows(x: np.ndarray, frames: int, stride: int | None = None) -> list[np.ndarray]:
    """Non-overlapping (by default) clips of `frames` frames from one sequence."""
    stride = stride or frames
    n = x.shape[-2]
    return [x[..., i:i + frames, :] for i in range(0, max(0, n - frames + 1), stride)]


def video_files(root, extensions=(".avi", ".mp4", ".mkv", ".mov", ".webm")) -> list[Path]:
    root = Path(root)
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in extensions)
