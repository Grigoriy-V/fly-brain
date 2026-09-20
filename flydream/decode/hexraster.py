"""The hexagonal lattice as a picture, and as a neighbourhood.

FlyVis has no hexal-to-image function. The only hexal-to-2D mapping in the
package is the decoder's axial "map storage" (`flyvis/task/decoder.py:304`), a
31x31 array indexed by (u+15, v+15): a sheared rhombus that exists so a square
convolution kernel can be masked into a hex shape, not a geometrically correct
raster. Everything else hexal-shaped in flyvis is a matplotlib figure, not an
array.

So this module builds the raster: each pixel takes the value of the hexal whose
Voronoi cell covers it, with the lattice support as a mask. Two choices are
fixed here for the life of the metric:

- `pixel_correlation` is computed on the 721-vector, never on the raster.
  Voronoi cells differ by about +-10% in pixel count, which silently weights
  some hexals more than others in a raster Pearson r.
- the raster exists for SSIM, which needs a local neighbourhood, and for
  figures.

`boxeye_aspect=True` reproduces BoxEye's own sampling grid (x = d*v), which is
what the renderer actually read; False gives a geometrically regular lattice.
"""
from __future__ import annotations

import functools

import numpy as np


@functools.lru_cache(maxsize=8)
def raster_map(n_hexals: int = 721, pix_per_hex: int = 3, boxeye_aspect: bool = True):
    """Return (index, mask, shape) for a regular flyvis hex lattice.

    index[i, j] is the hexal whose Voronoi cell covers pixel (i, j); mask is the
    lattice support. Without the mask the border hexals absorb every pixel
    outside the lattice (up to 308 px against about 36 for an interior hexal)
    and corrupt every windowed statistic.
    """
    from scipy.spatial import cKDTree
    from flyvis.utils.hex_utils import get_hex_coords, get_hextent

    u, v = get_hex_coords(get_hextent(n_hexals))
    s = float(pix_per_hex)
    ax = 1.0 if boxeye_aspect else np.sqrt(3) / 2.0
    x = v.astype(float) * ax * s
    y = (u + v / 2.0) * s
    x0, y0 = x.min() - s / 2, y.min() - s / 2
    w = int(round(x.max() + s / 2 - x0)) + 1
    h = int(round(y.max() + s / 2 - y0)) + 1
    gx, gy = np.meshgrid(np.arange(w) + x0, np.arange(h) + y0)
    d, idx = cKDTree(np.stack([x, y], 1)).query(np.stack([gx.ravel(), gy.ravel()], 1))
    return idx.reshape(h, w), (d.reshape(h, w) <= s / np.sqrt(3)), (h, w)


def to_raster(values, n_hexals: int = 721, pix_per_hex: int = 3, fill: float = 0.0):
    """(..., n_hexals) -> (..., H, W). Broadcasts over batch and frame axes."""
    index, mask, _ = raster_map(n_hexals, pix_per_hex)
    img = np.asarray(values)[..., index]
    return np.where(mask, img, fill)


def to_axial(values, fill: float = np.nan):
    """(..., n_hexals) -> (..., rows, cols) in axial coordinates.

    This is flyvis's own storage order (`task/decoder.py:304`): sheared, not
    geometric. Use it to match the reference decoder's layout, not to look at.
    """
    from flyvis.utils.hex_utils import get_hex_coords, get_hextent

    a = np.asarray(values)
    u, v = get_hex_coords(get_hextent(a.shape[-1]))
    u, v = u - u.min(), v - v.min()
    out = np.full((*a.shape[:-1], int(u.max()) + 1, int(v.max()) + 1), fill, dtype=np.float32)
    out[..., u, v] = a
    return out


@functools.lru_cache(maxsize=8)
def axial_coords(n_hexals: int = 721) -> np.ndarray:
    """(n_hexals, 2) int64 axial (u, v) lattice coordinates, as flyvis stores them."""
    from flyvis.utils.hex_utils import get_hex_coords, get_hextent

    u, v = get_hex_coords(get_hextent(n_hexals))
    return np.stack([np.asarray(u, np.int64), np.asarray(v, np.int64)], 1)


@functools.lru_cache(maxsize=8)
def hex_distance(n_hexals: int = 721) -> np.ndarray:
    """(n, n) int64 lattice distance in steps between every pair of columns.

    On an axial hex lattice the number of steps between (u₁,v₁) and (u₂,v₂) is
    (|du| + |dv| + |du+dv|)/2 — the standard cube-coordinate metric, which is 1
    for the six `neighbour_index` neighbours and grows by one per ring."""
    a = axial_coords(n_hexals)
    du = a[:, None, 0] - a[None, :, 0]
    dv = a[:, None, 1] - a[None, :, 1]
    return (np.abs(du) + np.abs(dv) + np.abs(du + dv)) // 2


@functools.lru_cache(maxsize=8)
def ring_of(n_hexals: int = 721) -> np.ndarray:
    """(n,) int64 ring index of each column, 0 at the lattice centre."""
    a = axial_coords(n_hexals)
    c = a - np.rint(a.mean(0)).astype(np.int64)
    return (np.abs(c[:, 0]) + np.abs(c[:, 1]) + np.abs(c[:, 0] + c[:, 1])) // 2


@functools.lru_cache(maxsize=8)
def neighbour_index(n_hexals: int = 721, invalid: int = -1) -> np.ndarray:
    """(n_hexals, 6) int64; columns are E, NE, NW, W, SW, SE, -1 off-lattice.

    The column order is `hex_utils.Hexal.neighbours()`. flyvis's own
    `HexLattice.valid_neighbours()` is ragged, which loses the direction; this
    keeps it, which a spatial smoothness penalty needs.
    """
    from flyvis.utils.hex_utils import get_hex_coords, get_hextent

    u, v = get_hex_coords(get_hextent(n_hexals))
    lut = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(u, v))}
    dirs = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    nb = np.full((len(u), 6), invalid, np.int64)
    for i, (a, b) in enumerate(zip(u, v)):
        for k, (du, dv) in enumerate(dirs):
            nb[i, k] = lut.get((int(a) + du, int(b) + dv), invalid)
    return nb
