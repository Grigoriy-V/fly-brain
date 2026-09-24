

## summary

flyvis has NO hexal→image function. The only hexal→2D mapping in the package is the decoder's axial "map storage" (task/decoder.py:304-305): a 31×31 array indexed directly by (u+15, v+15), which is a sheared rhombus, not a geometrically correct raster — it exists only so Conv2dHexSpace can mask a square kernel into a hex shape. Everything else hexal-shaped goes through plots.hex_scatter, which draws matplotlib RegularPolygon patches (a figure, not an array). The 721-hexal ordering is unambiguous and I verified it programmatically: get_hex_coords(15), BoxEye._receptor_centers, hex_center_coordinates (used by HexEye) and connectome add_strided_nodes with stride (1,1) all emit the identical double loop (u outer -15..15, v inner max(-15,-15-u)..min(15,15-u)), which equals np.lexsort((v,u)) as the identity — so renderer target and connectome activity agree with no permutation needed (63 of 65 cell types have exactly 721 nodes in that order; Lawf1/Lawf2 have 123 and are not output types). There is one in-package rasteriser, HexEye.is_inside — an (n_pixels, 721) bool matrix that I confirmed matches nearest-hexal assignment exactly, but it leaves 38% of pixels uncovered and costs 433 MB at the default ppo=25. My recommendation is to rasterise with a precomputed nearest-hexal Voronoi index (given below, validated end-to-end against a BoxEye render) rather than define SSIM over the hex neighbourhood, and to compute pixel correlation on the raw 721-vector rather than on the raster.

## code_example

"""721-hexal vector -> 2D raster. Validated end-to-end against a BoxEye render:
nearest-hexal assignment reproduces HexEye.is_inside exactly on covered pixels,
and quadrant means of the raster track the source cartesian frame (no transpose,
no mirror). Requires numpy + scipy (both already in .venv).
"""
import numpy as np
from scipy.spatial import cKDTree
from flyvis.utils.hex_utils import get_hex_coords, get_hextent


def hex_raster_map(n_hexals=721, pix_per_hex=3, boxeye_aspect=True):
    """Precompute (index, mask, (H, W)) for a regular flyvis hex lattice.

    index[i, j] = hexal whose Voronoi cell covers pixel (i, j).
    Row increases with (u + v/2), column with v -- the same orientation as
    BoxEye.receptor_centers, so the raster is an upright low-res version of the
    cartesian frame the renderer sampled.

    boxeye_aspect=True  reproduces BoxEye's grid (x = d*v), i.e. faithful to the
                        pixels BoxEye actually read.
    boxeye_aspect=False uses x = d*v*sqrt(3)/2, a geometrically regular lattice.
    Pick one and keep it fixed for the life of the metric.
    """
    u, v = get_hex_coords(get_hextent(n_hexals))
    s = float(pix_per_hex)
    ax = 1.0 if boxeye_aspect else np.sqrt(3) / 2.0
    x = v.astype(float) * ax * s
    y = (u + v / 2.0) * s
    x0, y0 = x.min() - s / 2, y.min() - s / 2
    W = int(round(x.max() + s / 2 - x0)) + 1
    H = int(round(y.max() + s / 2 - y0)) + 1
    GX, GY = np.meshgrid(np.arange(W) + x0, np.arange(H) + y0)
    d, idx = cKDTree(np.stack([x, y], 1)).query(np.stack([GX.ravel(), GY.ravel()], 1))
    # mask == the lattice support; WITHOUT it border hexals absorb everything
    # outside (up to 308 px vs ~36 interior) and corrupt every window statistic.
    return idx.reshape(H, W), d.reshape(H, W) <= s / np.sqrt(3), (H, W)


def hex_to_raster(values, index, mask=None, fill=0.0):
    """values (..., n_hexals) -> (..., H, W). Broadcasts over batch/frame axes."""
    img = np.asarray(values)[..., index]
    return img if mask is None else np.where(mask, img, fill)


def hex_to_axial_map(values, fill=0.0):
    """flyvis' own storage (task/decoder.py:304-305): (..., 721) -> (..., 31, 31).
    Sheared, not geometric -- use only if you also mask the window (see
    Conv2dHexSpace, task/decoder.py:160-165)."""
    a = np.asarray(values)
    u, v = get_hex_coords(get_hextent(a.shape[-1]))
    u, v = u - u.min(), v - v.min()
    out = np.full((*a.shape[:-1], u.max() + 1, v.max() + 1), fill, dtype=float)
    out[..., u, v] = a
    return out


def hex_neighbour_index(n_hexals=721, invalid=-1):
    """(n_hexals, 6) int64; columns are E, NE, NW, W, SW, SE -- the order of
    Hexal.neighbours() (hex_utils.py:420). -1 where the neighbour is off-lattice.
    Verified element-for-element identical to HexLattice(extent=15).
    valid_neighbours(), but fixed-shape and direction-preserving (that one is
    ragged) and ~1000x faster to build."""
    u, v = get_hex_coords(get_hextent(n_hexals))
    lut = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(u, v))}
    dirs = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    nb = np.full((len(u), 6), invalid, np.int64)
    for i, (a, b) in enumerate(zip(u, v)):
        for k, (du, dv) in enumerate(dirs):
            nb[i, k] = lut.get((int(a) + du, int(b) + dv), invalid)
    return nb


# ---- how to use it for the two metrics --------------------------------------
# Build ONCE, reuse for every prediction/target pair:
INDEX, MASK, SHAPE = hex_raster_map(721, pix_per_hex=3)   # -> (94, 94)

def pixel_correlation(pred_hex, targ_hex):
    """Compute on the 721-vector, NOT the raster: Voronoi cells differ by ~+-10%
    in pixel count, which shifts raster Pearson r (measured -0.0606 vs -0.0642
    on random data)."""
    a = np.asarray(pred_hex).ravel(); b = np.asarray(targ_hex).ravel()
    return float(np.corrcoef(a, b)[0, 1])

def hex_ssim(pred_hex, targ_hex, data_range=1.0, win_size=9):
    """SSIM on the rasterised pair. win_size must span >= 3 hexals, else the
    window sits inside one flat hexal, local variance is 0 and SSIM saturates.
    At pix_per_hex=3, win_size=9 covers 3 hexals. NOTE: scikit-image is NOT
    installed in this venv -- `uv add scikit-image` first."""
    from skimage.metrics import structural_similarity
    p = hex_to_raster(pred_hex, INDEX, MASK, fill=0.0)
    t = hex_to_raster(targ_hex, INDEX, MASK, fill=0.0)
    return structural_similarity(t, p, data_range=data_range, win_size=win_size)


if __name__ == "__main__":
    import torch
    from flyvis.datasets.rendering import BoxEye
    H0 = W0 = 391
    _, xx = np.mgrid[0:H0, 0:W0]
    frame = 0.3 + 0.5 * xx / W0
    frame[:80, :80] = 1.0
    frame[300:340, 120:260] = 0.0
    hexvals = BoxEye(15, 13)(torch.tensor(frame[None, None], dtype=torch.float32))
    hexvals = hexvals.numpy().reshape(-1)              # (721,)

    idx, mask, shp = hex_raster_map(721, 6)
    img = hex_to_raster(hexvals, idx, mask, fill=np.nan)
    print("raster", shp, "coverage %.3f" % mask.mean(),
          "hexals hit", len(np.unique(idx[mask])))     # (187, 187) 0.704 721
    print("axial ", hex_to_axial_map(hexvals).shape)   # (31, 31)
    nb = hex_neighbour_index()
    print("nbrs", nb.shape, np.bincount((nb >= 0).sum(1)))  # (721,6) ... 3:6 4:84 6:631
    batch = np.random.rand(4, 7, 721)
    print("batched", hex_to_raster(batch, idx, mask).shape)  # (4, 7, 187, 187)

## api

[
 {
  "name": "get_hex_coords",
  "signature": "get_hex_coords(extent: int, astensor: bool = False) -> Tuple[NDArray, NDArray]",
  "file": ".venv/Lib/site-packages/flyvis/utils/hex_utils.py:15",
  "returns": "(u, v), each int64 shape (get_num_hexals(extent),). extent=15 -> (721,), (721,). Range -15..15 each, with -15 <= u+v <= 15.",
  "notes": "THE canonical ordering. Double loop: `for q in range(-extent, extent+1): for r in range(max(-extent,-extent-q), min(extent,extent-q)+1)`. Verified: sort_u_then_v_index(u,v) == np.arange(721), i.e. the order IS lexsort by u then v."
 },
 {
  "name": "ActivityDecoder.__init__ / DecoderGAVP.forward  (the axial 'map storage')",
  "signature": "self.u, self.v = get_hex_coords(connectome.config.extent); self.u -= self.u.min(); self.v -= self.v.min(); self.H, self.W = self.u.max()+1, self.v.max()+1  ->  x_map[..., self.u, self.v] = x",
  "file": ".venv/Lib/site-packages/flyvis/task/decoder.py:46 (coords), 304-305 (scatter), 323-325 (gather back)",
  "returns": "(n_samples, n_frames, in_channels, 31, 31) for extent=15; 721 of the 961 cells written, the other 240 left at 0.",
  "notes": "This is the ONLY hexal-to-2D-array code in flyvis and the closest thing to a 'hex_to_square'. It is axial map storage (https://www.redblobgames.com/grids/hexagons/#map-storage), i.e. a sheared lattice: in this array (u+1,v-1) is a diagonal that IS a hex neighbour while (u+1,v+1) is a diagonal that is NOT (hex distance 2). Conv2dHexSpace compensates by masking the kernel; a plain square SSIM window on it would not."
 },
 {
  "name": "Conv2dHexSpace.__init__ (hex-shaped mask for a square kernel in axial storage)",
  "signature": "u, v = get_hex_coords(kernel_size // 2); u -= u.min(); v -= v.min(); mask = np.zeros(self.weight.shape); mask[:, :, u, v] = 1",
  "file": ".venv/Lib/site-packages/flyvis/task/decoder.py:160-165",
  "returns": "mask of shape (out_ch, in_ch, kernel_size, kernel_size) with 1 at the get_num_hexals(kernel_size//2) hex positions.",
  "notes": "Reusable directly as the hexagonal window for a hex-masked SSIM on the 31x31 axial map, if you go that route. kernel_size must be odd."
 },
 {
  "name": "hex_to_pixel",
  "signature": "hex_to_pixel(u, v, size: float = 1, mode: Literal['default','flat','pointy'] = 'default') -> Tuple[NDArray, NDArray]",
  "file": ".venv/Lib/site-packages/flyvis/utils/hex_utils.py:45",
  "returns": "(x, y) float arrays, same shape as u/v. mode='default': x = 1.5*v, y = -sqrt(3)*(u + v/2). Note `size` is IGNORED in 'default' mode.",
  "notes": "The only hexal -> continuous (x,y) function. Its 'default' mode y is SIGN-FLIPPED relative to the renderer (BoxEye uses y = +d*(u+v/2)), and its x scale (1.5) is not the renderer's. Used only by hex_scatter for plotting. Do not mix conventions."
 },
 {
  "name": "BoxEye / BoxEye._receptor_centers / BoxEye.hex_render",
  "signature": "BoxEye(extent: int = 15, kernel_size: int = 13); __call__(sequence, ftype='mean', hex_sample=True) -> Tensor",
  "file": ".venv/Lib/site-packages/flyvis/datasets/rendering/eye.py:29 (class), 71-89 (centers), 153 (hex_render), 169-170 (the actual indexing)",
  "returns": "__call__ -> (samples, frames, 1, 721). receptor_centers: torch.long (721, 2) as (y, x) with y = kernel_size*(u + v/2), x = kernel_size*v. min_frame_size = [391, 391] for the defaults.",
  "notes": "Verified numerically: receptor_centers ordering == get_hex_coords(15) ordering, y == (13*(u+v/2)).astype(int64), x == 13*v. Image row increases with (u+v/2), image column with v -- this is the orientation your raster should match. NOTE the grid is anisotropic (x should be v*d*sqrt(3)/2 for a regular lattice); the correct line is commented out at eye.py:88."
 },
 {
  "name": "HexEye / HexEye.is_inside  (the one real rasteriser in the package)",
  "signature": "HexEye(n_ommatidia=721, ppo=25, monitor_height_px=None, monitor_width_px=None, device=..., dtype=torch.float16); .is_inside",
  "file": ".venv/Lib/site-packages/flyvis/datasets/rendering/eye.py:225 (class), 282-292 (is_inside), utils.py:211 (is_inside_hex)",
  "returns": "is_inside: bool tensor (monitor_h*monitor_w, n_ommatidia). For 721 at ppo=7 -> (47089, 721), 34 MB; at the default ppo=25 -> (600625, 721), 433 MB.",
  "notes": "img = (is_inside.float() @ values).reshape(H, W) is a correct upright raster -- I verified it agrees 100% with nearest-hexal-centre assignment on all covered pixels. Caveats: only 61.9% of pixels are covered (gaps between hexagons; zero overlap), and it works only because monitor_height_px == monitor_width_px by default (the x/y swap at eye.py:283-286 cancels the x-major product() flattening at eye.py:271-278 only for a square monitor)."
 },
 {
  "name": "hex_center_coordinates",
  "signature": "hex_center_coordinates(n_hex_area: int, img_width: int, img_height: int, center: bool = True) -> Tuple[NDArray, NDArray, Tuple[float, float]]",
  "file": ".venv/Lib/site-packages/flyvis/datasets/rendering/utils.py:175",
  "returns": "(xs, ys, (dist_w, dist_h)); xs = dist_w*v (+ W//2), ys = dist_h*(u + v/2) (+ H//2); dist_w = img_width/(2n+1), n = floor(sqrt(n_hex_area/3)).",
  "notes": "Same double loop as get_hex_coords, so same ordering. This is the cleanest existing hexal -> (x, y) pixel-position function that agrees with the renderer's orientation (unlike hex_to_pixel)."
 },
 {
  "name": "ConnectomeFromAvgFilters.nodes / add_strided_nodes / layer_index",
  "signature": "add_strided_nodes(seq, typ, extent, strides=(u_stride, v_stride)); connectome.nodes.layer_index[cell_type] -> int64 indices; connectome.nodes.u / .v -> int32 (45669,)",
  "file": ".venv/Lib/site-packages/flyvis/connectome/connectome.py:326 (add_strided_nodes), 235-241 (nodes dict), 266-271 (layer_index), 262-264 (central_cells_index)",
  "returns": "layer_index[ct] is (721,) for 63 of 65 cell types; nodes.u[layer_index[ct]] == get_hex_coords(15)[0] exactly.",
  "notes": "Verified for all 65 types: 63 give (True, True) against get_hex_coords(15); only Lawf1 and Lawf2 have 123 nodes (strided), and neither is an output_cell_type, so a decoder reading output types is unaffected. LayerActivity (utils/activity_utils.py:204) slices activity by exactly these indices, so network activity is in the same order as the renderer's target."
 },
 {
  "name": "Hexal.neighbours / HexLattice.valid_neighbours",
  "signature": "Hexal(u, v, value).neighbours() -> 6-tuple of Hexal, CCW from east; HexLattice(extent=15).valid_neighbours() -> tuple of 721 ragged index tuples",
  "file": ".venv/Lib/site-packages/flyvis/utils/hex_utils.py:420 (neighbours), 397-418 (E/NE/NW/W/SW/SE), 825 (_get_neighbour_indices), 834 (valid_neighbours)",
  "returns": "valid_neighbours(): 721 tuples, lengths 6 (631 hexals), 4 (84), 3 (6). ~0.9 s to build for extent 15.",
  "notes": "YES, the package exposes nearest neighbours. Direction order is E=(+1,0), NE=(0,+1), NW=(-1,+1), W=(-1,0), SW=(0,-1), SE=(+1,-1). But the return is RAGGED -- missing border neighbours are dropped, not padded -- so you cannot recover direction from position at the border. I verified my fixed-shape (721, 6) table with -1 padding is identical to valid_neighbours() element-for-element on all 721 hexals."
 },
 {
  "name": "get_num_hexals / get_hextent",
  "signature": "get_num_hexals(extent) -> int  (1 + 3*extent*(extent+1)); get_hextent(num_hexals) -> int  (floor(sqrt(num_hexals/3)))",
  "file": ".venv/Lib/site-packages/flyvis/utils/hex_utils.py:218, 233",
  "returns": "get_num_hexals(15) == 721; get_hextent(721) == 15.",
  "notes": "Use get_hextent to derive extent from a vector length, as quick_hex_scatter and HexScatter do."
 },
 {
  "name": "pad_to_regular_hex / crop_to_extent / sort_u_then_v_index",
  "signature": "pad_to_regular_hex(u, v, values, extent, value=np.nan) -> (u_pad, v_pad, values_pad); crop_to_extent(u, v, color, max_extent); sort_u_then_v_index(u, v) -> NDArray",
  "file": ".venv/Lib/site-packages/flyvis/utils/hex_utils.py:145, 308, 269",
  "returns": "pad_to_regular_hex broadcasts over leading axes: values (..., n) -> (..., get_num_hexals(extent)).",
  "notes": "Useful for going between a partial hexal set (e.g. a receptive field) and the full 721 lattice. sort_u_then_v_index is the canonical-order operator."
 },
 {
  "name": "rotation_permutation_index / flip_permutation_index / rotate_Nx60",
  "signature": "rotation_permutation_index(extent: int, n_rot: int) -> Tensor; flip_permutation_index(extent: int, axis: Literal[1,2,3]) -> Tensor",
  "file": ".venv/Lib/site-packages/flyvis/datasets/augmentation/utils.py:37, 111, 52",
  "returns": "(get_num_hexals(extent),) int permutation applied directly to the last axis of a hexal tensor.",
  "notes": "Independent confirmation that the hexal vector is in get_hex_coords order: these are built as sort_u_then_v_index(*rotate(get_hex_coords(extent))) and applied straight to the 721-axis by HexRotate/HexFlip (datasets/augmentation/hex.py:25, 110)."
 },
 {
  "name": "hex_scatter / quick_hex_scatter / HexScatter",
  "signature": "hex_scatter(u, v, values, max_extent=None, fig=None, ax=None, figsize=(1,1), ..., mode='default', origin='lower', ...) -> (Figure, Axes, (Line2D, ScalarMappable)); quick_hex_scatter(values, cmap=..., **kw)",
  "file": ".venv/Lib/site-packages/flyvis/analysis/visualization/plots.py:153, 605; animations/hexscatter.py:19",
  "returns": "A matplotlib Figure/Axes -- NOT an array. Draws one RegularPolygon patch per hexal (plots.py:325-345).",
  "notes": "quick_hex_scatter(values) is the one-liner for eyeballing a 721-vector: it infers extent via get_hextent and coords via get_hex_coords. There is no figure->array helper in the package, so do not try to route metrics through this."
 },
 {
  "name": "MultiTaskSintel / MultiTaskSintelDataset targets",
  "signature": "MultiTaskSintel(tasks=['flow'], boxfilter=dict(extent=15, kernel_size=13), ...); stored keys `<seq>/lum` (frames, 1, 721), `<seq>/flow` (frames, 2, 721), `<seq>/depth` (frames, 1, 721)",
  "file": ".venv/Lib/site-packages/flyvis/datasets/sintel.py:70-71 (render), 116-161 (storage), 227-272 (dataset), 658/691 (cartesian_* reloads)",
  "returns": "All targets already in 721-hexal space, in canonical order, produced by BoxEye.",
  "notes": "The decoder target is a hexal vector, not an image. cartesian_lum/cartesian_flow/cartesian_depth do NOT un-render -- they reload the original frames from disk. There is no inverse rendering anywhere in flyvis."
 },
 {
  "name": "l2norm / epe / correlation (flyvis' own metrics)",
  "signature": "l2norm(y_est, y_gt); epe(y_est, y_gt); correlation(x, x_pred, epsilon=0); quick_correlation_one_to_many(x, Y, zero_nans=True)",
  "file": ".venv/Lib/site-packages/flyvis/task/objectives.py:10, 25; analysis/correlation.py:31, 52",
  "returns": "scalars / arrays; all reduce over the hexal axis directly.",
  "notes": "flyvis never rasterises to compute a metric, and contains no SSIM. So there is no in-package precedent to match either way -- the choice is yours."
 }
]

## gotchas

[
 "There is no hex_to_square and no to_image anywhere in flyvis -- I grepped the whole package. The task's premise that hex_utils.py has them is wrong. hex_rows(n_rows, n_columns, eps, mode) (hex_utils.py:80) is NOT related: it returns pixel coordinates for a rectangular n_rows x n_columns grid of hexes, not for the 721 lattice.",
 "The decoder's 31x31 axial map (task/decoder.py:304-305) is sheared map storage, not a raster. In that array the diagonal (u+1, v-1) IS a hex neighbour while the diagonal (u+1, v+1) is at hex distance 2. A plain 3x3 square window there has 8 members of which only 6 are true neighbours -- a 25% error in the window support. flyvis compensates with Conv2dHexSpace's hex mask (decoder.py:160-165); an unmasked SSIM window would not.",
 "BoxEye's receptor grid is ANISOTROPIC: eye.py:85-88 gives y = kernel_size*(u + v/2), x = kernel_size*v. A regular hex lattice needs x = kernel_size*v*sqrt(3)/2, so BoxEye's sampling is stretched horizontally by 2/sqrt(3) = 1.155. The author knew -- the correct line is commented out at eye.py:88. Consequence: `boxeye_aspect=True` gives a raster faithful to the cartesian pixels BoxEye actually read (187x187 at pix_per_hex=6); `False` gives a geometrically regular lattice (187x163). Both are defensible; mixing them across runs is not.",
 "hex_to_pixel(mode='default') (hex_utils.py:45) does NOT match the renderer: it returns y = -sqrt(3)*(u + v/2) (sign-flipped) and x = 1.5*v (different scale), and it silently IGNORES its `size` argument in 'default' mode. It exists for hex_scatter's plotting only. Use hex_center_coordinates (rendering/utils.py:175) or BoxEye.receptor_centers if you want the renderer's orientation.",
 "BoxEye stores receptor_centers as torch.long, so 13*(u + v/2) is TRUNCATED TOWARD ZERO -- odd-v rows land 0.5 px off, with the rounding direction depending on sign. Sub-pixel; harmless for metrics but it means receptor_centers is not exactly an affine image of (u, v).",
 "HexEye.is_inside is a working rasteriser (img = (is_inside.float() @ values).reshape(H, W)), and I verified it matches nearest-hexal-centre assignment on 100% of covered pixels -- but only 61.9% of pixels are covered (gaps between hexagons; zero overlap), so 38% of the image is holes. It also costs 34 MB at ppo=7 and 433 MB at the default ppo=25, and it is only correct because monitor_height_px == monitor_width_px: is_inside_hex is called with the points' x/y swapped (eye.py:283-286), which cancels the x-major `product(arange(W), arange(H))` flattening (eye.py:271-278) only for a square monitor. Pass unequal monitor dims and it breaks silently.",
 "ALWAYS apply the lattice-support mask to the raster. Without it, each of the 90 border hexals absorbs the entire region outside the lattice -- I measured up to 308 pixels for a border hexal versus ~36 for an interior one. With the mask (d <= pix_per_hex/sqrt(3)) every hexal gets 31-37 pixels, interior and border alike, so window statistics are effectively unweighted.",
 "Do NOT compute pixel correlation on the raster. Voronoi cell sizes still vary by about +-10%, so raster Pearson r != hexal Pearson r (I measured -0.0606 vs -0.0642 on random 721-vectors). Compute correlation on the raw 721-vectors and use the raster only for SSIM, which genuinely needs a grid.",
 "SSIM window sizing is the real trap. A 7x7 Gaussian window at pix_per_hex=6 sits entirely inside one flat hexal: local variance is 0, the structure term degenerates and SSIM saturates near 1. Size win_size so the window spans at least 3 hexals (win_size >= 2*pix_per_hex+1), or keep pix_per_hex at 2-3. Any SSIM number you report is meaningless without stating pix_per_hex, win_size and the mask/fill convention.",
 "HexLattice(extent=15).valid_neighbours() (hex_utils.py:834) does give correct neighbour indices -- I verified it agrees element-for-element with my table -- but it returns a RAGGED tuple-of-tuples with off-lattice neighbours dropped rather than padded, so you cannot recover the direction from the tuple position at the border, and it takes ~0.9 s to build for extent 15 (it iterates Hexal objects). Prefer a fixed (721, 6) table with -1 padding.",
 "Two of 65 connectome cell types are strided and have only 123 nodes: Lawf1 and Lawf2. Every other type has exactly 721 in get_hex_coords(15) order. Neither is in output_cell_types, so a decoder reading output types is safe -- but never assume `layer_index[ct].shape == (721,)` without checking, and never assume node indices are contiguous per type.",
 "scikit-image is NOT installed in .venv (scipy 1.18.1, torch 2.14.0+cpu, matplotlib 3.11.2 are). You need `uv add scikit-image` or a hand-rolled SSIM over scipy.ndimage.gaussian_filter.",
 "flyvis computes no image metrics on rasters at all: l2norm and epe (task/objectives.py:10, 25) reduce over the hexal axis directly, and analysis/correlation.py works on flat vectors. There is no SSIM in the package, so there is no in-package convention to match -- and no published flyvis SSIM number to compare against.",
 "flyvis has no inverse rendering. MultiTaskSintel.cartesian_lum / cartesian_flow / cartesian_depth (sintel.py:658, 691 and nearby) reload the ORIGINAL frames from disk via sample_lum/sample_flow/sample_depth -- they do not un-render a hexal vector. If you want 'reconstruction vs. original image', you rasterise; there is no shortcut."
]

## unknowns

[
 "Whether this project's own axial coordinates (flydream/data/columns_roi.py, optic_lobe.py, export.py use hex1/hex2 from the FlyWire/optic-lobe `assignedOlHex1/Hex2`) share flyvis' u/v axis convention, origin and sign. The FlyWire values are 1-based and centred differently. I did not establish a mapping -- if you plan to move activity between the two coordinate systems you must verify this separately, and a wrong mapping would be a silent 60-degree rotation or reflection, not a crash.",
 "The absolute retinotopic orientation. Both eye.py:87 and rendering/utils.py:203 carry the same unresolved comment, 'either must be negative or origin must be upper', flagging a sign ambiguity the flyvis authors left open. I verified internal consistency (renderer -> hexal vector -> my raster preserves the source frame's up/down and left/right, checked by quadrant means on an asymmetric test image) but NOT correctness against the fly's real visual field. If a figure needs to be right way up biologically, confirm against a published flyvis figure.",
 "Which target this project's decoder actually uses. I confirmed sintel stores lum as (frames, 1, 721) and flow as (frames, 2, 721), both BoxEye-rendered at extent 15, but nothing in flydream currently references flyvis or 721 -- so I could not check what the reconstruction target is here.",
 "The right pix_per_hex / win_size for this project. There is no flyvis precedent and no published hex-SSIM baseline, so absolute SSIM values will not be comparable to any external number regardless of what you pick. I validated that the machinery works; I did not calibrate it.",
 "Whether area-weighted (anti-aliased) rasterisation would change the SSIM ranking versus the nearest-neighbour rasterisation I tested. Nearest-neighbour produces hard hexagonal blocks, which inflates the local-structure term at small window sizes; I did not measure the difference against a smooth interpolant.",
 "Whether the 2 strided types (Lawf1, Lawf2, 123 nodes) matter for anything downstream in this project. They are not output cell types, so the decoder path is clear, but I did not check other consumers."
]