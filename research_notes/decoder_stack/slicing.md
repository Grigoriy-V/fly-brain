

## summary

Slicing flyvis activity by cell type across all columns works and I verified it end-to-end by actually running the package (not reading only). `Network.simulate()` returns one dense tensor over ALL neurons — shape `(n_samples, n_frames, 45669)` float32 — and per-type slices come either from `LayerActivity(activity, connectome, keepref=True)["T4a"]` or, identically (bit-exact, asserted), from `activity[:, :, np.asarray(connectome.nodes.layer_index["T4a"][:])]`. The default connectome (fib25-fib19_v2.2.json, extent=15) has 65 cell types (not 64) over 721 columns; 63 types have one cell per column (721) and exactly two — Lawf1 and Lawf2 — are strided (u%3==0 & v%2==0) with 123 cells, so their slices come back as `(B, T, 123)`. Within a type the node indices are one contiguous block ordered by u ascending then v ascending, which for stride-(1,1) types is byte-identical to `hex_utils.get_hex_coords(15)` (asserted), so the (u,v) lattice coordinate of cell k is just `nodes.to_df()[nodes.type==t][["u","v"]]` row k, and the central cell sits at position 360 (61 for Lawf). Two real blockers for using FlyVis as a baseline on this machine: (1) building the connectome fresh on Windows always crashes in datamate (`_write_h5` unlinks an open h5 handle → WinError 32) — it only works here because I built a valid `ConnectomeFromAvgFilters_0003` cache under a monkeypatch; (2) all the public `*_responses` helpers hardwire `cell_index="central"`, so the cached/ensemble response machinery gives you the central cell only — all-columns data requires `generic_responses(..., cell_index=None)` or your own `simulate` loop. Memory is modest for the activity itself (17.42 MiB per sample per 100 frames) but `simulate(initial_state="auto")` peaks at ~650 MiB per sample because `steady_state` holds 100 states each caching a (B, 1.51M-edge) tensor.

## code_example

"""Verified runnable: .venv/Scripts/python.exe <this file>
Saved at a local scratch directory (final_snippet.py)
Output:
  activity (2, 100, 45669) 36535200 bytes
  T4a (2, 100, 721) Lawf1 (2, 100, 123)
  central_all (2, 100, 65)
  dense T4a (31, 31) dense Lawf1 (11, 15)
  padded Lawf1 (100, 721) non-nan/frame 123
  chunked {'T4a': (2, 100, 721), 'T5a': (2, 100, 721)} 1153600 bytes kept
"""
import numpy as np, torch
import flyvis                                    # sets torch default device + datamate root
from flyvis import Network
from flyvis.utils.activity_utils import LayerActivity
from flyvis.utils import hex_utils

net = Network()                                  # default connectome: extent=15
cx  = net.connectome
H   = net.stimulus.n_input_elements               # 721 photoreceptor hexals

# ---------- 1. simulate: (n_samples, n_frames, n_cells=45669) ----------
B, T, dt = 2, 100, 1 / 100
movie = torch.rand(B, T, 1, H)                    # (samples, frames, 1, hexals) -- 4-D required
with torch.no_grad():
    activity = net.simulate(movie, dt=dt)         # torch.Tensor (B, T, 45669)

# ---------- 2. per-cell-type slice over ALL columns ----------
# (a) high level -- keepref=True is MANDATORY (keepref=False raises AttributeError)
la = LayerActivity(activity, cx, keepref=True)
t4a         = la["T4a"]                           # (B, T, 721)
lawf1       = la["Lawf1"]                         # (B, T, 123)  <- strided type
central_t4a = la.central.T4a                      # (B, T)       central cell only
central_all = la.central[:]                       # (B, T, 65)   one cell per type

# (b) low level -- bit-exact identical, no wrapper object
idx_t4a = np.asarray(cx.nodes.layer_index["T4a"][:])   # int64, len 721, contiguous
assert torch.equal(activity[:, :, idx_t4a], t4a)

# ---------- 3. (u, v) lattice coordinate per cell of a type ----------
nodes = cx.nodes.to_df()                          # index, role, type(str), u, v
sub   = nodes[nodes.type == "T4a"]                # rows already in layer_index order
u, v  = sub.u.values.astype(int), sub.v.values.astype(int)
assert np.array_equal((u, v), hex_utils.get_hex_coords(15))   # exact for stride-(1,1) types

# ---------- 4. hex lattice -> dense (rows, cols) image ----------
def to_dense(u, v, values):
    """values: (..., n_cells_of_type) -> (..., Hrows, Wcols) axial image, NaN off-lattice."""
    du = np.unique(np.diff(np.unique(u))); dv = np.unique(np.diff(np.unique(v)))
    su = int(du[0]) if du.size else 1;     sv = int(dv[0]) if dv.size else 1
    r = (u - u.min()) // su; c = (v - v.min()) // sv
    img = np.full((*values.shape[:-1], r.max() + 1, c.max() + 1), np.nan, np.float32)
    img[..., r, c] = np.asarray(values)
    return img

frame   = to_dense(u, v, t4a[0, 50].numpy())                          # (31, 31), 721 finite
sub_l   = nodes[nodes.type == "Lawf1"]
frame_l = to_dense(sub_l.u.values.astype(int), sub_l.v.values.astype(int),
                   lawf1[0, 50].numpy())                               # (11, 15), 123 finite

x, y = hex_utils.hex_to_pixel(u, v)               # true hex centres for scatter/patch plots

# pad a strided type back onto the full 721-hexal lattice (NaN elsewhere)
pu, pv, padded = hex_utils.pad_to_regular_hex(
    sub_l.u.values, sub_l.v.values, lawf1[0].numpy(), 15)              # padded: (T, 721)

# ---------- 5. memory-lean: never hold the full tensor ----------
from flyvis.network.network import simulation as simulation_ctx
want  = ["T4a", "T5a"]
index = {t: torch.as_tensor(np.asarray(cx.nodes.layer_index[t][:])) for t in want}
CHUNK = 10
out   = {t: [] for t in want}
with torch.no_grad(), simulation_ctx(net):
    state = net.steady_state(1.0, dt, B)          # compute ONCE and reuse (~588 MiB peak)
    for s in range(0, T, CHUNK):
        chunk = movie[:, s:s + CHUNK]
        net.stimulus.zero(B, chunk.shape[1])
        net.stimulus.add_input(chunk)
        states = net(net.stimulus(), dt, state=state, as_states=True)
        acts   = torch.stack([st.nodes.activity for st in states], dim=1)
        for t in want:
            out[t].append(acts[:, :, index[t]].clone())   # .clone() so the chunk can be freed
        state = states[-1]                                 # carry the state across chunks
        del states, acts
out = {t: torch.cat(v, dim=1) for t, v in out.items()}
# verified bit-exact vs the monolithic simulate(): max abs diff 0.0


## api

[
 {
  "name": "Network.simulate",
  "file": ".venv/Lib/site-packages/flyvis/network/network.py:628",
  "signature": "simulate(movie_input: Tensor, dt: float, initial_state='auto', as_states=False, as_layer_activity=False)",
  "returns": "torch.Tensor (n_samples, n_frames, 45669) float32 on CPU-or-default-device. movie_input MUST be 4-D (n_samples, n_frames, 1, 721) or it raises ValueError. With as_layer_activity=True returns LayerActivity(..., keepref=True) already .cpu()'d. With as_states=True returns a list of n_frames AutoDeref states.",
  "notes": "Returns ALL neurons — no cell-type or central subsetting. Verified: (2,20,45669) and (1,100,45669). initial_state='auto' calls steady_state(1.0, dt, batch_size) internally; pass a precomputed state to skip that (see gotchas)."
 },
 {
  "name": "LayerActivity",
  "file": ".venv/Lib/site-packages/flyvis/utils/activity_utils.py:204",
  "signature": "LayerActivity(activity, connectome, keepref: bool = False, use_central: bool = True)",
  "returns": "dict subclass; la[\"T4a\"] / la.T4a -> (n_samples, n_frames, 721); la[\"Lawf1\"] -> (n_samples, n_frames, 123). Preserves the input's type (torch.Tensor in -> torch.Tensor out).",
  "notes": "THE call the task asks for. keepref=True is mandatory when use_central=True — the default keepref=False raises AttributeError. Slicing is `activity[..., dict.__getitem__(self, key)]` at activity_utils.py:90-92, so any leading axes are preserved."
 },
 {
  "name": "LayerActivity.central / CentralActivity",
  "file": ".venv/Lib/site-packages/flyvis/utils/activity_utils.py:118",
  "signature": "la.central.T4a  |  la.central[:]  |  la.central[[\"T4a\",\"T5a\"]]",
  "returns": "la.central.T4a -> (n_samples, n_frames) (scalar per frame, the u=v=0 cell). la.central[:] -> (n_samples, n_frames, 65), one central cell per type in `unique_cell_types` order.",
  "notes": "CentralActivity.__setattr__ (activity_utils.py:183) detects activity.shape[-1] != 65 and eagerly materializes the (B,T,65) central slice, forcing keepref=True on itself. Small (26 kB for B=1,T=100) but it is a real extra allocation on every LayerActivity construction."
 },
 {
  "name": "CellTypeActivity.update",
  "file": ".venv/Lib/site-packages/flyvis/utils/activity_utils.py:71",
  "signature": "la.update(new_activity)",
  "returns": "None; rebinds the activity so the same indexer can be reused across batches.",
  "notes": "Canonical indexer-only pattern used inside flyvis: `ix = LayerActivity(None, connectome, keepref=True)` then `ix.update(act)` per batch (see stimulus_responses_currents.py:64 and optimal_stimuli.py:141). Verified working."
 },
 {
  "name": "connectome.nodes.layer_index",
  "file": ".venv/Lib/site-packages/flyvis/connectome/connectome.py:267",
  "signature": "connectome.nodes.layer_index[cell_type][:]  ->  np.ndarray int64",
  "returns": "int64 node indices into the last axis of the activity tensor. len 721 for 63 types, 123 for Lawf1/Lawf2. Always a CONTIGUOUS ascending block (verified np.all(np.diff(idx)==1) for R1, T4a, Lawf1).",
  "notes": "Built as `np.nonzero(nodes.type == cell_type)[0]` (connectome.py:269). It is a datamate Directory of ArrayFiles, NOT a plain dict — you must use `[:]` to materialize; `.items()` works and keys are decoded str. This is the direct-indexing route: `activity[:, :, idx]` is bit-exact equal to `LayerActivity[...]` (asserted torch.equal)."
 },
 {
  "name": "connectome.central_cells_index",
  "file": ".venv/Lib/site-packages/flyvis/connectome/connectome.py:262",
  "signature": "connectome.central_cells_index[:]  ->  np.ndarray int64, shape (65,)",
  "returns": "One node index per cell type (u==0 & v==0), ordered exactly like `unique_cell_types` (verified list equality). First values: [360, 1081, 1802, 2523, ...].",
  "notes": "Every type including the strided Lawf1/Lawf2 has a (0,0) cell, so this is length 65 with no gaps."
 },
 {
  "name": "connectome.nodes.to_df",
  "file": ".venv/Lib/site-packages/flyvis/connectome/connectome.py:235",
  "signature": "connectome.nodes.to_df()  ->  pandas.DataFrame",
  "returns": "columns ['index','role','type','u','v']; dtypes int64, StringDtype, StringDtype, int32, int32. 45669 rows. `layer_index` is NOT a column (it is a sub-Directory).",
  "notes": "The (u,v) lookup: `sub = df[df.type == 'T4a']; u, v = sub.u.values, sub.v.values` — rows come out in exactly layer_index order, so u[k], v[k] is the lattice coordinate of column k of the sliced activity. `type` is already decoded to str here, unlike `connectome.nodes.type[:]` (bytes |S8)."
 },
 {
  "name": "connectome.unique_cell_types",
  "file": ".venv/Lib/site-packages/flyvis/connectome/connectome.py:182",
  "signature": "connectome.unique_cell_types[:].astype(str)",
  "returns": "65 names: R1..R8, L1..L5, Lawf1, Lawf2, Am, C2, C3, CT1(Lo1), CT1(M10), Mi1..Mi15, T1, T2, T2a, T3, T4a-d, T5a-d, Tm*, TmY*.",
  "notes": "Raw dtype is |S8 bytes — always .astype(str). input_cell_types = R1..R8 (8); output_cell_types = 34 T/Tm/TmY types; node roles split 5768 input / 15387 intermediate / 24514 output."
 },
 {
  "name": "add_strided_nodes",
  "file": ".venv/Lib/site-packages/flyvis/connectome/connectome.py:326",
  "signature": "for u in range(-n, n+1): for v in range(max(-n,-n-u), min(n,n-u)+1): if u % u_stride == 0 and v % v_stride == 0",
  "returns": "Defines the node ordering: u ascending outer, v ascending inner, hexagon-clipped. Node ids are assigned sequentially per type, so each type's ids are contiguous.",
  "notes": "This exact loop is duplicated in hex_utils.get_hex_coords (hex_utils.py:15), which is why the two orders match for stride (1,1). Python's `%` on negatives is non-negative, so u=-15 passes u%3==0."
 },
 {
  "name": "hex_utils.get_hex_coords",
  "file": ".venv/Lib/site-packages/flyvis/utils/hex_utils.py:15",
  "signature": "get_hex_coords(extent: int, astensor=False) -> (u, v)",
  "returns": "Two arrays of length get_num_hexals(extent) = 1+3*e*(e+1); 721 for extent 15. Verified np.array_equal with T4a's (u,v).",
  "notes": "Use this for any stride-(1,1) type instead of touching the dataframe. It does NOT describe Lawf1/Lawf2."
 },
 {
  "name": "hex_utils.pad_to_regular_hex",
  "file": ".venv/Lib/site-packages/flyvis/utils/hex_utils.py:145",
  "signature": "pad_to_regular_hex(u, v, values, extent, value=np.nan) -> (u_padded, v_padded, values_padded)",
  "returns": "Lifts a strided type onto the full lattice. Verified: Lawf1 (100, 123) -> (100, 721) with 123 finite entries per frame. Works with arbitrary leading axes (last axis must be cells).",
  "notes": "The right way to render Lawf1/Lawf2 on the same 721-hexal canvas as everything else."
 },
 {
  "name": "hex_utils.hex_to_pixel",
  "file": ".venv/Lib/site-packages/flyvis/utils/hex_utils.py:45",
  "signature": "hex_to_pixel(u, v, size=1, mode='default') -> (x, y)",
  "returns": "mode='default' gives x = 1.5*v, y = -sqrt(3)*(u + v/2). For T4a: x in [-22.5, 22.5], y in [-25.98, 25.98].",
  "notes": "Use for true hexagonal scatter/patch rendering; `mode='default'` is what plots.hex_scatter assumes."
 },
 {
  "name": "hex_utils.get_num_hexals / get_hextent",
  "file": ".venv/Lib/site-packages/flyvis/utils/hex_utils.py:218",
  "signature": "get_num_hexals(15) == 721 ; get_hextent(721) == 15",
  "returns": "int",
  "notes": "get_hextent is floor(sqrt(n/3)) and is WRONG for strided types: get_hextent(123) == 6 but get_num_hexals(6) == 127. Anything that infers geometry from the cell count silently mis-renders Lawf1/Lawf2."
 },
 {
  "name": "plots.hex_scatter / quick_hex_scatter",
  "file": ".venv/Lib/site-packages/flyvis/analysis/visualization/plots.py:153",
  "signature": "hex_scatter(u, v, values, max_extent=None, fill=False, cmap=..., ...) ; quick_hex_scatter(values, **kw)",
  "returns": "(Figure, Axes, (Line2D|None, ScalarMappable))",
  "notes": "quick_hex_scatter (plots.py:605) infers u,v via get_hextent(len(values)) — safe only for 721-cell types. For Lawf1/Lawf2 call hex_scatter with explicit u, v."
 },
 {
  "name": "animations.HexScatter",
  "file": ".venv/Lib/site-packages/flyvis/analysis/animations/hexscatter.py:76",
  "signature": "HexScatter(hexarray, u=None, v=None, ...)  # hexarray (n_samples, n_frames, 1, n_cells)",
  "returns": "Animation object",
  "notes": "The canonical flyvis pattern for 'one cell type over all columns' is at analysis/animations/network.py:154 — `voltage = self.responses[cell_type][:, :, None]` (inserting the channel axis) plus explicit `u, v` from the nodes dataframe. If u/v are omitted it falls back to get_hex_coords(get_hextent(n)) (hexscatter.py:120-121), which breaks for strided types."
 },
 {
  "name": "stimulus_responses.generic_responses",
  "file": ".venv/Lib/site-packages/flyvis/analysis/stimulus_responses.py:111",
  "signature": "generic_responses(network_view_or_ensemble, dataset, dataset_config, default_dataset_cls, t_pre, t_fade_in, batch_size, cell_index='central')",
  "returns": "xr.Dataset with dims (network_id, sample, frame, neuron) and coords cell_type/u/v attached to the `neuron` dim (stimulus_responses.py:237-243).",
  "notes": "Pass cell_index=None for ALL 45669 neurons, or a layer_index array for one type. This is the nicest all-columns container because it carries cell_type/u/v as coords. Every public helper (flash_responses:263, moving_edge_responses:292, naturalistic_stimuli_responses:357, ...) omits the argument, so they all cache CENTRAL-ONLY responses."
 },
 {
  "name": "Stimulus",
  "file": ".venv/Lib/site-packages/flyvis/network/stimulus.py:111",
  "signature": "net.stimulus.zero(n_samples, n_frames) ; net.stimulus.add_input(x) ; net.stimulus()",
  "returns": "buffer torch.zeros((n_samples, n_frames, 45669)) — stimulus.py:161. net.stimulus.n_input_elements == 721, n_nodes == 45669.",
  "notes": "A SECOND full-size tensor, same bytes as the activity output. It is reused across calls when (n_samples, n_frames) is unchanged. Also exposes its own str-keyed `layer_index` and `central_cells_index` dicts (stimulus.py:118-127) if you want plain dicts rather than the datamate Directory."
 },
 {
  "name": "Network.steady_state",
  "file": ".venv/Lib/site-packages/flyvis/network/network.py:548",
  "signature": "steady_state(t_pre, dt, batch_size, value=0.5, state=None, grad=False, return_last=True)",
  "returns": "AutoDeref state (the last one)",
  "notes": "Implemented as `self(..., as_states=True)[-1]` (network.py:585), i.e. it materializes int(t_pre/dt) states before discarding all but one. Measured peak for t_pre=1.0, dt=1/100, B=1: +588 MiB RSS. Compute it once and pass it as `initial_state` / `state`."
 },
 {
  "name": "nn_utils.simulation",
  "file": ".venv/Lib/site-packages/flyvis/utils/nn_utils.py:11",
  "signature": "with simulation(network): ...",
  "returns": "context manager",
  "notes": "Sets network.training=False and requires_grad=False on all params, restoring afterwards. Needed if you drive net.forward/net.__call__ directly instead of via simulate() (simulate asserts both at network.py:685)."
 }
]

## gotchas

[
 "BROKEN DEFAULT CONSTRUCTOR: `LayerActivity(activity, connectome)` -- the exact example in the module docstring (activity_utils.py:8) -- raises `AttributeError: 'weakref.ReferenceType' object has no attribute 'shape'`. LayerActivity.__setattr__ (activity_utils.py:279-282) wraps the tensor in a weakref BEFORE handing it to CentralActivity.__setattr__, which then calls `value.shape[-1]` (activity_utils.py:185). Verified matrix: keepref=True/use_central=True OK, keepref=True/use_central=False OK, keepref=False/use_central=True CRASH, keepref=False/use_central=False OK. Always pass keepref=True.",
 "SILENT None: with keepref=False (and use_central=False so it constructs at all), `activity` is held as a weakref. Once the tensor goes out of scope, `la.T4a` returns `None` instead of raising -- CellTypeActivity.__getattr__ (activity_utils.py:79-81) early-returns None on a dead ref. Verified.",
 "DOCUMENTED FEATURE THAT DOES NOT EXIST: the LayerActivity docstring advertises 'virtual types' -- `a['L2+L4']` as the sum of individuals (activity_utils.py:237-241). It raises `ValueError: L2+L4`; there is no '+' handling anywhere in __getattr__. Verified.",
 "MULTI-TYPE AXIS POSITION: `la[['T4a','T4b']]` returns shape (B, T, 2, 721) -- the type axis is inserted at -2 via np.stack of the index arrays (activity_utils.py:83-86), not concatenated into the cell axis. Only works when the types have equal cell counts; mixing T4a (721) with Lawf1 (123) will fail in np.stack.",
 "65 TYPES, NOT 64. fib25-fib19_v2.2.json defines 65 node types. Total nodes = 63*721 + 2*123 = 45669 (not 721*64 = 46144).",
 "STRIDED TYPES: exactly Lawf1 and Lawf2, pattern ['stride', [3, 2]] -> nodes where u%3==0 and v%2==0 -> 123 cells (u in [-15,15] step 3, v in [-14,14] step 2). Every other type is ['stride', [1,1]] -> 721. Their slice is (B, T, 123), and their central cell is at position 61 within the type (vs 360 for 721-cell types).",
 "`hex_utils.get_hextent(123)` returns 6, but `get_num_hexals(6)` is 127 != 123. Any code that infers lattice geometry from the cell count -- including `quick_hex_scatter` (plots.py:605-619) and `HexScatter` when u/v are omitted (hexscatter.py:115, 120-121) -- silently mis-renders Lawf1/Lawf2. Pass explicit u, v, or lift them with `pad_to_regular_hex(..., extent=15)` first.",
 "`connectome.nodes.layer_index` is a datamate `Directory` of ArrayFiles, NOT a dict of arrays. `layer_index['T4a']` is a lazy h5 accessor -- you must slice `[:]` (and wrap in np.asarray) before using it as an index. `.items()` works and keys are already decoded str.",
 "BYTES vs STR: `connectome.unique_cell_types[:]` and `connectome.nodes.type[:]` are numpy |S8 / |S12 BYTES -- compare with b'T4a' or call .astype(str). `connectome.nodes.to_df()` decodes them to pandas StringDtype. Mixing the two is the easiest way to get an empty selection.",
 "MEMORY TRAP -- steady_state dominates, not the activity tensor. `simulate(initial_state='auto')` peaks at +650 MiB RSS for B=1/T=100, of which +588 MiB is `steady_state(1.0, dt, B)`: it runs forward with `as_states=True` and holds int(t_pre/dt)=100 states, each of which caches a dereferenced (B, n_edges=1513231) float32 edge tensor (5.77 MiB each) inside AutoDeref._cache (tensor_utils.py:68-77). Compute the steady state ONCE and pass it as `initial_state`/`state`. Measured: doing so drops the per-chunk loop to ~0 extra RSS.",
 "The stimulus buffer is a second full-size tensor: `torch.zeros((n_samples, n_frames, 45669))` at stimulus.py:161 -- same bytes as the activity output, so budget 2x. It is reused only while (n_samples, n_frames) stay constant.",
 "`import flyvis` calls `torch.set_default_device(cuda if available)` at __init__.py:13-14. Every tensor you create after the import silently lands on the GPU. On this machine torch.cuda.is_available() is False, so it is CPU -- but that changes the moment a CUDA build is installed.",
 "WINDOWS BLOCKER -- a fresh connectome build always fails. `datamate.io._write_h5` (.venv/Lib/site-packages/datamate/io.py:153-181) opens the h5 with mode='w', hits KeyError on f['data'], then calls `path.unlink()` at line 174 while its own handle `f` is still open -> `PermissionError: [WinError 32]`. Reproduced with a clean FLYVIS_ROOT_DIR. The only reason it works in the repository root today is that I built a valid cache under a monkeypatch; the fix is either that monkeypatch (close f before unlinking) or shipping/copying a prebuilt connectome directory.",
 "Stale cache dirs are present in the venv: .venv/Lib/site-packages/flyvis/data/connectome/ contains ConnectomeFromAvgFilters_0000, _0001, _0002 (all half-written, `status: stopped` in _meta.yaml -- _0001/_0002 are my failed attempts; I was denied permission to delete them) and _0003 (`status: done`, the working one, 45669 nodes). datamate resolves to _0003 because it skips non-'done' dirs, so the broken ones are inert -- but they should be deleted.",
 "`datamate.set_root_context(path)` does NOT redirect the connectome despite the `@root(..., precedence=2)` docs claiming context wins (connectome.py:108, context.py:62-72). Use the `FLYVIS_ROOT_DIR` environment variable instead, set BEFORE `import flyvis` (__init__.py:45-54, and it also honours a .env via dotenv at __init__.py:18).",
 "CENTRAL-ONLY CACHE: every public response helper -- flash_responses (stimulus_responses.py:263), moving_edge_responses (:292), moving_bar_responses, naturalistic_stimuli_responses (:357), central_impulses_responses (:386), spatial_impulses_responses -- calls generic_responses WITHOUT cell_index, so it defaults to 'central' (:41, :119) and the joblib-cached xr.Dataset has 65 neurons, not 45669. For all columns call `generic_responses(..., cell_index=None)` directly (it will not hit the existing cache) or drive `network.simulate` / `network.stimulus_response` yourself.",
 "`net.forward(x, dt, state)` returns the stacked activity but NOT the final state, so chunked simulation must use `as_states=True` and take `states[-1]` to carry state across chunks. Verified bit-exact against a monolithic simulate() (max abs diff 0.0) with CHUNK=10.",
 "A dense (2*extent+1, 2*extent+1) axial image is only ~75% full: 721 of 31*31=961 cells are on-lattice; the rest must stay NaN (the hexagon inscribed in the rhombus). For Lawf1 the dense grid is (11, 15) with 123 of 165 filled. The (u,v)->(row,col) map `r=(u-u.min())//u_stride, c=(v-v.min())//v_stride` is injective in both cases (verified)."
]

## unknowns

[
 "Whether the pretrained ensemble checkpoints (NetworkView / Ensemble / flyvis download-pretrained) load on this machine -- I only used a randomly initialized `Network()` (free=734, fixed=2959 params). I did not attempt any network download.",
 "Whether the connectome I built with the `_write_h5` monkeypatch (ConnectomeFromAvgFilters_0003) is identical to an officially built one. Node/edge construction is pure numpy/json so it should be deterministic, but I could not cross-verify against a reference build.",
 "Exact per-tensor accounting for the `as_states=True` path. My 588/650 MiB figures are RSS sampled at 2 ms from a background thread, so they include PyTorch allocator retention; the 5.77 MiB-per-state edge-cache figure is computed from n_edges=1513231, not measured directly.",
 "GPU behaviour and GPU memory -- torch.cuda.is_available() is False here, so nothing about device placement in simulate(), LayerActivity's .cpu() call at network.py:692, or CUDA peak memory was exercised.",
 "Whether `flyvis` has a supported way to make the public *_responses helpers keep all columns (e.g. a config flag I did not find). I only established that `generic_responses(cell_index=...)` is the parameter and that no public wrapper forwards it.",
 "Timing at realistic scale: my longest run was B=2, T=100 at dt=1/100 (~seconds on CPU). I did not measure throughput for a full Sintel-scale sweep, which is the number that actually decides whether all-column simulation is affordable as a baseline.",
 "Why datamate's root precedence does not behave as documented (set_root_context ignored). I worked around it with FLYVIS_ROOT_DIR rather than tracing _directory_from_config."
]