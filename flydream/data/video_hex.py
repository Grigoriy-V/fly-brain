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
      → `augmentation.Interpolate` native framerate → 1/dt, **nearest-exact**
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


def view_bounds(width: int, win: int, splits: int) -> list[tuple[int, int]]:
    """`rendering.utils.split`'s own start/stop arithmetic, as bounds instead
    of gathered copies: the views are contiguous slices, and copying two
    417-column blocks of every frame cost more than sampling them."""
    if splits <= 1:
        return [(0, width)]
    out_n = max(win, width // splits)
    overlap = int(np.ceil((out_n * splits - width) / (splits - 1)))
    return [(i * out_n - i * overlap, (i + 1) * out_n - i * overlap) for i in range(splits)]


def hexal_sample(frames: torch.Tensor, box) -> torch.Tensor:
    """(..., H, W) -> (..., 721): `BoxEye`'s own values, computed only where
    they are needed. Any leading batch dimensions are kept.

    `BoxEye.__call__` convolves the whole frame with a 13x13 box and then
    reads 721 points out of it — 169 multiply-adds per pixel for 0.4 % of the
    pixels. A box mean is separable and the receptors sit on a lattice, so the
    same values come from a running sum along x read at the ~31 columns that
    carry a receptor, then along y at the ~61 such rows: O(H·W) instead of
    O(H·W·k²). This was 90 % of the corpus build time, and the values match
    `BoxEye` to 1e-6 (checked in `tests/test_data_helpers.py`).

    The receptor windows lie fully inside the frame (the eye spans 391 of the
    417 rows the caller keeps), so no zero padding enters the sums."""
    k = box.kernel_size
    r = (k - 1) // 2
    x = frames.to(torch.float32)
    H, W = x.shape[-2:]
    c = box.receptor_centers + torch.tensor([H // 2, W // 2])
    xs, x_inv = torch.unique(c[:, 1], return_inverse=True)            # ~31 columns and ~61 rows carry all 721
    ys, y_inv = torch.unique(c[:, 0], return_inverse=True)
    if int((xs - r).min()) < 1 or int((ys - r).min()) < 1 or int((xs + r).max()) >= W or int((ys + r).max()) >= H:
        raise ValueError(f"the receptor windows leave a {H}x{W} frame; the caller must keep at least "
                         f"{int(box.min_frame_size[0]) + 2 * k} rows")
    cx = x.cumsum(-1)                                                 # no padding: the windows lie inside
    rows = cx[..., xs + r] - cx[..., xs - r - 1]                      # (..., H, nx)
    cy = rows.cumsum(-2)
    cols = cy[..., ys + r, :] - cy[..., ys - r - 1, :]                # (..., ny, nx)
    return cols[..., y_inv, x_inv] / (k * k)


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
              crop: float = CROP, width: int = WIDTH, chunk: int = 32,
              resample: str = "nearest-exact") -> np.ndarray:
    """(T, H, W) frames → (splits, T', 721) hexal luminance at 1/dt.

    The frame is first resized to `width` pixels across, aspect ratio kept.
    That is Sintel's own width, and it is what makes an object in a new clip
    subtend the same angle and move at the same speed across the eye as one
    in the clips 13B already knows — T4/T5 are tuned to speed, so rendering a
    320-pixel-wide video at its own scale would hand the prior a different
    world, not a bigger one. It is also required: the eye's window is 417
    pixels wide and `split` cannot cut that out of a 224-pixel crop.

    The crop comes before the resize and covers exactly what the splits use —
    the eye's 417-row window (it samples the centre: `BoxEye.hex_render` adds
    h//2) and the `crop` fraction of the width. For Sintel, whose frames are
    already 1,024 wide, the resize is then the identity and the result is
    unchanged; for a small frame it is the same picture at a fraction of the
    work.

    Time is resampled the way the Sintel clips of 13A/13B actually are:
    **nearest-exact**, not linear. Measured 2026-09-20: a Sintel clip at
    50 Hz repeats every second frame (21 of 39 consecutive pairs are
    identical), because 24 fps is held, not interpolated. The frozen brain
    steps at 20 ms and answers frame-to-frame change, so a smoothly
    interpolated corpus would move differently from everything 13B was
    trained on."""
    from flyvis.datasets.augmentation.temporal import Interpolate

    box = box or eye()
    x = torch.as_tensor(np.asarray(gray, np.float32))
    H, W = x.shape[-2:]
    win = int(box.min_frame_size[1]) + 2 * box.kernel_size                 # 417: the eye's window plus its kernel
    out_w = int(width) if width else W
    out_h = max(win, int(round(H * out_w / W)))
    keep_w = max(win, int(out_w * crop))                                   # what the splits will actually use
    sy, sx = H * win / out_h, W * keep_w / out_w                           # the same box in source pixels
    y0, x0 = int(round(H / 2 - sy / 2)), int(round(W / 2 - sx / 2))
    x = x[:, max(0, y0):max(0, y0) + int(round(sy)), max(0, x0):max(0, x0) + int(round(sx))]
    out = []
    for i in range(0, len(x), chunk):                     # cropped first: the eye needs the centre, not the frame
        c = x[i:i + chunk]
        if c.shape[-2:] != (win, keep_w):
            c = torch.nn.functional.interpolate(c[:, None], size=(win, keep_w), mode="bilinear",
                                                align_corners=False, antialias=True)[:, 0]
        out.append(torch.stack([hexal_sample(c[..., a:b], box) for a, b in view_bounds(c.shape[-1], win, splits)]))
    hexals = torch.cat(out, dim=1)
    if abs(fps - 1 / dt) > 1e-6:
        hexals = Interpolate(fps, 1 / dt, mode=resample).transform(hexals, dim=1)
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
    """Largest frame-to-frame jump against the 90th percentile of the clip's
    own non-zero jumps. A hard cut is a flash across the whole eye, which
    T4/T5 answer with a transient no real motion produces.

    Non-zero, because 25 fps held to 50 Hz makes every second difference
    exactly zero; a high percentile rather than the mean or the median,
    because the cut itself is in the sample and would inflate them. Measured
    on 95 Sintel clips against the same clips spliced in half: at the
    threshold that drops 1 % of clean clips this catches 84 % of the splices,
    where max/mean catches 78 % and max/median 58 %."""
    d = np.abs(np.diff(np.asarray(x, np.float32), axis=-2)).mean(-1)
    nz = d[d > 1e-7]
    if not len(nz):
        return 0.0
    return float(d.max() / (float(np.percentile(nz, 90)) + 1e-6))


def windows(x: np.ndarray, frames: int, stride: int | None = None) -> list[np.ndarray]:
    """Non-overlapping (by default) clips of `frames` frames from one sequence."""
    stride = stride or frames
    n = x.shape[-2]
    return [x[..., i:i + frames, :] for i in range(0, max(0, n - frames + 1), stride)]


def video_files(root, extensions=(".avi", ".mp4", ".mkv", ".mov", ".webm")) -> list[Path]:
    root = Path(root)
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in extensions)
