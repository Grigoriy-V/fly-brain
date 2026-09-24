

## summary

The naturalistic dataset is `flyvis.datasets.sintel.AugmentedSintel` (subclass of `MultiTaskSintel`), which pre-renders Sintel through a `BoxEye(extent=15, kernel_size=13)` filter into exactly 721 hexals and caches every sequence in RAM as float32. One sample is a dict, not a tuple: `{"lum": (n_out, 1, 721), "flow": (n_out, 2, 721)}` float32, where `n_out = ceil(19 / (24*dt))` — 40 frames at dt=1/50, 80 at dt=1/100 (19 raw frames at Sintel's 24 fps, resampled). Sintel itself is NOT in the pretrained download you already fetched; it must sit at `<FLYVIS_ROOT_DIR>/SintelDataSet/{training/final,training/flow,test}`, and `FLYVIS_ROOT_DIR` is not currently set in .env, so flyvis resolves root_dir to site-packages/flyvis/data and cannot see data/flyvis at all. The rendered 721-value hexal input comes back for free as the first element of every `Network.stimulus_response(...)` yield — shape (batch, frames, 1, 721) float32 numpy, frame-aligned 1:1 with the response (batch, frames, 45669), because the t_pre/t_fade_in frames go into the initial state only and never into the output. AugmentedSintel is already deterministic with its own defaults (contrast/brightness/noise/gamma std all None, p_flip=p_rot=0, geometric augmentation baked per-sample into `flip_ax`/`n_rot` columns), so you must NOT set augment=False — that short-circuits `get_item` and silently skips resampling. I verified all shapes, dtypes, determinism, the split logic and the paired extraction against pretrained network flow/0000/000; one hard Windows blocker (datamate's `_write_h5` leaves an h5py handle open before `unlink()`, WinError 32) stops the Sintel rendering step and needs the monkeypatch included in the snippet.

## code_example

"""Paired (rendered hexal stimulus, network activity) samples from flyvis on Sintel.
VERIFIED end-to-end against pretrained flow/0000/000 using a synthetic 2-scene
Sintel tree; full file at
a local scratch directory (flyvis_pairs_final.py)

Prereqs:
  1) .env must gain   FLYVIS_ROOT_DIR=data/flyvis
     (flyvis reads it at import via dotenv; flyvis/__init__.py:45-58)
  2) Sintel must live at data/flyvis/SintelDataSet/ with
         training/final/<scene>/frame_XXXX.png
         training/flow/<scene>/frame_XXXX.flo
         test/
     It is NOT part of results_pretrained_models.zip. Let
     flyvis.datasets.sintel_utils.download_sintel() fetch
     http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip, or unzip it there
     yourself. First dataset construction then renders
     data/flyvis/renderings/RenderedSintel_0000 (one-off, slow).
"""
# ---------------------------------------------------------------- WINDOWS ONLY
# datamate 1.0.0 io.py:153 _write_h5 calls path.unlink() while an h5py handle on
# that path is still open -> PermissionError WinError 32 during rendering.
# Reproduced and fixed by this patch (verified).
import sys
if sys.platform == "win32":
    from pathlib import Path
    import h5py as h5
    import numpy as _np
    import datamate.io as _dio
    import datamate.directory as _ddir

    def _write_h5(path: Path, val) -> None:
        val = _np.asarray(val)
        f = None
        try:
            f = h5.File(path, libver="latest", mode="w")
            if "data" not in f or f["data"].dtype != val.dtype:
                raise ValueError()
            f["data"][...] = val
            f.swmr_mode = True
        except Exception:
            if f is not None:
                try:
                    f.close()          # <- the fix: release before unlink
                except Exception:
                    pass
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.is_dir():
                path.rmdir()
            elif path.exists():
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
            f = h5.File(path, libver="latest", mode="w")
            f["data"] = val
            f.swmr_mode = True
        f.close()

    _dio._write_h5 = _write_h5
    _ddir._write_h5 = _write_h5
# -----------------------------------------------------------------------------

import numpy as np
import torch

import flyvis
from flyvis.datasets.sintel import AugmentedSintel

DT = 1 / 50   # 0.02 s == the dt the pretrained flow ensemble was trained at

# ------------------------------------------------------------------- dataset
# Deterministic by construction: AugmentedSintel defaults contrast_std /
# brightness_std / gaussian_white_noise / gamma_std to None (identity), sets
# p_flip=p_rot=0, and bakes geometric augmentation into each sample (flip_ax /
# n_rot columns). temporal_split=True bins every sequence into fixed 19-frame
# chunks. KEEP augment=True: augment=False short-circuits get_item
# (sintel.py:970) and skips the 24 fps -> 1/dt resampling (19 raw frames).
dataset = AugmentedSintel(
    tasks=["lum", "flow"],       # "lum" = rendered input, "flow" = optic-flow target
    n_frames=19,
    dt=DT,
    temporal_split=True,
    interpolate=True,            # linear interpolation of the flow target
    flip_axes=[0],               # [0] = no flip;     [0, 1] doubles the dataset
    n_rotations=[0],             # [0] = no rotation; [0..5] multiplies it by 6
    augment=True,
    vertical_splits=3,
    center_crop_fraction=0.7,
    boxfilter=dict(extent=15, kernel_size=13),   # -> 721 hexals
)
print(f"dataset: {len(dataset)} sequences, dt={dataset.dt}, "
      f"original_framerate={dataset.original_framerate}, t_pre={dataset.t_pre}")
print(dataset.arg_df.head().to_string())

sample = dataset[0]                                    # dict, not a tuple
for k, v in sample.items():
    print(f"  {k:5s} {tuple(v.shape)} {v.dtype}  [{v.min():.3f}, {v.max():.3f}]")
# ->   lum   (40, 1, 721) torch.float32  [0.413, 0.578]
# ->   flow  (40, 2, 721) torch.float32  [-0.122, 0.127]

# --------------------------------------------- scene-level train/test split
# Split on the Sintel scene so no scene leaks across the split (row-level random
# splits leak: the same scene reappears as 3 vertical x N temporal x 12 geometric
# variants). dataset.arg_df has name / original_index for exactly this.
scenes = np.array([n.split("_split_")[0].split("_", 2)[2]
                   for n in dataset.arg_df.name])
uniq = np.unique(scenes)
rng = np.random.default_rng(0)
rng.shuffle(uniq)
n_test = max(1, int(round(0.2 * len(uniq))))
test_scenes = set(uniq[:n_test])
test_idx = np.flatnonzero([s in test_scenes for s in scenes])
train_idx = np.flatnonzero([s not in test_scenes for s in scenes])
print(f"train {len(train_idx)} / test {len(test_idx)}; held out {sorted(test_scenes)}")
# flyvis' own paper split (MultiTaskSintel-shaped ONLY -- see gotchas):
#   train_i, val_i = MultiTaskSintel(...).original_train_and_validation_indices()

# -------------------------------------------------- paired stimulus/response
nv = flyvis.NetworkView("flow/0000/000")      # any pretrained ensemble member
net = nv.init_network(checkpoint="best")

for stim, resp in net.stimulus_response(
    dataset,
    dt=DT,
    indices=train_idx,        # these samples, in this order
    t_pre=0.0,                # grey pre-stimulus: state only, not in the output
    t_fade_in=2.0,            # 2 s contrast ramp-in: state only, not in the output
    batch_size=4,
    default_stim_key="lum",
):
    # stim: (batch, frames, 1, 721) float32 numpy -- THE RENDERED HEXAL INPUT,
    #       the exact tensor the photoreceptors saw. Use as the decoder target.
    # resp: (batch, frames, 45669) float32 numpy -- every cell, frame-aligned 1:1.
    print("stim", stim.shape, stim.dtype, "resp", resp.shape, resp.dtype)
    central = net.connectome.central_cells_index[:]           # (65,) one per type
    print("central-only resp", np.take(resp, central, axis=-1).shape)
    break
# -> stim (3, 40, 1, 721) float32 resp (3, 40, 45669) float32
# -> central-only resp (3, 40, 65)

# the optic-flow target for the same samples (stimulus_response drops non-"lum" keys)
flow = torch.stack([dataset[int(i)]["flow"] for i in train_idx[:4]])
print("flow target", tuple(flow.shape), flow.dtype)          # (4, frames, 2, 721)


# ------------------------------- alternative: xarray via compute_responses ----
# Call compute_responses DIRECTLY (not through NetworkView) to skip joblib caching.
from flyvis.analysis.stimulus_responses import compute_responses
cn = nv.network(checkpoint="best", lazy=True); cn.init()
res = compute_responses(
    cn, AugmentedSintel,
    dict(tasks=["lum"], interpolate=False, boxfilter={"extent": 15, "kernel_size": 13},
         temporal_split=True, dt=DT, indices=None, n_rotations=[0], flip_axes=[0]),
    batch_size=2, t_pre=0.0, t_fade_in=2.0,
    cell_index=None,          # None -> all 45669 cells; 'central' -> 65
)
hexal_frames = res.stimulus.values[:, :, 0, :]   # (n_samples, n_frames, 721) float32
activity     = res.responses.values[0]           # (n_samples, n_frames, n_cells)
print(hexal_frames.shape, activity.shape)
# res coords carry name / original_index / vertical_split_index /
# temporal_split_index / frames / flip_ax / n_rot per sample.


## api

[
 {
  "name": "AugmentedSintel",
  "file": ".venv/Lib/site-packages/flyvis/datasets/sintel.py:733 (__init__ at :780)",
  "signature": "AugmentedSintel(n_frames=19, flip_axes=[0,1], n_rotations=[0,1,2,3,4,5], build_stim_on_init=True, temporal_split=False, augment=True, dt=1/50, tasks=['flow'], interpolate=True, all_frames=False, random_temporal_crop=False, boxfilter={'extent':15,'kernel_size':13}, vertical_splits=3, contrast_std=None, brightness_std=None, gaussian_white_noise=None, gamma_std=None, center_crop_fraction=0.7, indices=None, unittest=False, **kwargs)",
  "returns": "torch Dataset. dataset[i] -> Dict[str, torch.Tensor]: 'lum' (n_out,1,721) float32 in ~[0,1]; 'flow' (n_out,2,721) float32; 'depth' (n_out,1,721) if requested. n_out = ceil(n_frames/(24*dt)) = 40 @ dt=1/50, 80 @ dt=1/100. len(dataset) = n_temporal_splits * len(flip_axes) * len(n_rotations).",
  "notes": "THE naturalistic dataset. `tasks` must include 'lum' to get the rendered input in the sample dict ('lum' is always rendered and cached, but only returned if listed). It calls super().__init__ with p_flip=0, p_rot=0 and passes flip/rot through _build deterministically. **kwargs is silently DISCARDED (never forwarded to super) — misspelled args are no-ops. `indices` subsets cached_sequences and arg_df after _build."
 },
 {
  "name": "AugmentedSintel.arg_df",
  "file": ".venv/Lib/site-packages/flyvis/datasets/sintel.py:885",
  "returns": "pandas.DataFrame with columns ['name','original_index','vertical_split_index','temporal_split_index','frames','flip_ax','n_rot']; len(arg_df) == len(dataset). 'name' is e.g. 'sequence_00_alley_1_split_00'; 'frames' is the RAW (19) count, not n_out.",
  "notes": "This is your split key. Sample ordering is itertools.product(sequences, flip_axes, n_rotations) — sequence outermost, rotation innermost (verified). After `indices=`, arg_df keeps its ORIGINAL index labels (e.g. 0,3,5) while positional order is 0..len-1."
 },
 {
  "name": "MultiTaskSintel",
  "file": ".venv/Lib/site-packages/flyvis/datasets/sintel.py:182 (__init__ at :229)",
  "signature": "MultiTaskSintel(tasks=['flow'], boxfilter={'extent':15,'kernel_size':13}, vertical_splits=3, n_frames=19, center_crop_fraction=0.7, dt=1/50, augment=True, random_temporal_crop=True, all_frames=False, resampling=True, interpolate=True, p_flip=0.5, p_rot=5/6, contrast_std=0.2, brightness_std=0.1, gaussian_white_noise=0.08, gamma_std=None, _init_cache=True, unittest=False, flip_axes=[0,1], sintel_path=None)",
  "returns": "Same dict shape as AugmentedSintel. len == n_scenes * vertical_splits (69 for real Sintel: 23 scenes x 3).",
  "notes": "The training-time dataset (the pretrained flow nets used exactly this with dt=0.02, flip_axes=[0,1,2,3], p_rot=0.5, contrast_std=0.2, brightness_std=0.1, gaussian_white_noise=0.08 — see data/flyvis/results/flow/0000/000/_meta.yaml). Random per-call augmentation by default. Unlike AugmentedSintel, augment=False here STILL resamples to 1/dt (verified: 40 frames at dt=1/50)."
 },
 {
  "name": "RenderedSintel",
  "file": ".venv/Lib/site-packages/flyvis/datasets/sintel.py:47 (__init__ at :68)",
  "signature": "RenderedSintel(tasks=['flow'], boxfilter={'extent':15,'kernel_size':13}, vertical_splits=3, n_frames=19, center_crop_fraction=0.7, unittest=False, sintel_path=None)",
  "returns": "datamate Directory at <FLYVIS_ROOT_DIR>/renderings/RenderedSintel_0000 with sequence_<i>_<scene>_split_<j>/{lum,flow[,depth]}.h5; lum (frames,1,721), flow (frames,2,721), frames = N_scene_files - 1.",
  "notes": "Built automatically on first MultiTaskSintel/AugmentedSintel construction; one-off and slow. Only renders scenes where len(files)-1 >= n_frames. `sintel_path` is captured into the datamate config (verified in _meta.yaml), so the absolute Sintel path is part of the directory identity — moving your data root forces a full re-render."
 },
 {
  "name": "download_sintel",
  "file": ".venv/Lib/site-packages/flyvis/datasets/sintel_utils.py:295",
  "signature": "download_sintel(delete_if_exists=False, depth=False) -> Path",
  "returns": "Path == flyvis.sintel_dir == <FLYVIS_ROOT_DIR>/SintelDataSet",
  "notes": "Fetches http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip (and, for depth, .../jwulff/sintel/MPI-Sintel-depth-training-20150305.zip) and unzips in place. Existence test requires training/, test/, training/flow/ (and training/depth/ when depth=True). Line 325 does `assert not sintel_zip.exists()` — a leftover partial zip makes it die on a bare AssertionError."
 },
 {
  "name": "flyvis.sintel_dir / renderings_dir / results_dir / resolve_root_dir",
  "file": ".venv/Lib/site-packages/flyvis/__init__.py:45-58",
  "returns": "root_dir = Path(os.getenv('FLYVIS_ROOT_DIR', <site-packages>/flyvis/data)); sintel_dir = root_dir/'SintelDataSet'; renderings_dir = root_dir/'renderings'; results_dir = root_dir/'results'",
  "notes": "Resolved at import, after dotenv.load_dotenv(find_dotenv(usecwd=True)). Verified: with FLYVIS_ROOT_DIR=data/flyvis, sintel_dir = data\\flyvis\\SintelDataSet and it does NOT exist. Your .env sets only NEUPRINT_TOKEN, so you must add FLYVIS_ROOT_DIR=data/flyvis."
 },
 {
  "name": "Network.stimulus_response",
  "file": ".venv/Lib/site-packages/flyvis/network/network.py:712",
  "signature": "stimulus_response(stim_dataset, dt, indices=None, t_pre=1.0, t_fade_in=0.0, grad=False, default_stim_key='lum', batch_size=1)",
  "returns": "generator yielding (stimulus, responses) numpy float32: stimulus (batch, frames, 1, 721) — the rendered hexal input verbatim; responses (batch, frames, 45669) — all cells. Verified shapes exactly.",
  "notes": "THE way to get paired data. t_pre/t_fade_in frames are folded into the initial state only, so stimulus and responses are frame-aligned 1:1. Picks stim = sample[default_stim_key], so flow/depth never reach the output. MUTATES stim_dataset.dt = dt in place (line 744) — verified: dataset[0]['lum'] went 40 -> 80 frames afterwards. Uses IndexSampler(indices), so output order == indices order."
 },
 {
  "name": "Network.simulate",
  "file": ".venv/Lib/site-packages/flyvis/network/network.py:628",
  "signature": "simulate(movie_input, dt, initial_state='auto', as_states=False, as_layer_activity=False)",
  "returns": "Tensor (batch, frames, 45669); or List[AutoDeref] if as_states; or LayerActivity if as_layer_activity.",
  "notes": "Lower-level alternative when you already hold the hexal movie. Requires movie_input.ndim == 4 == (sample, frame, 1, hexals) or raises ValueError. initial_state='auto' = steady_state after 1 s of grey 0.5. Warns if dt > 1/50."
 },
 {
  "name": "compute_responses",
  "file": ".venv/Lib/site-packages/flyvis/analysis/stimulus_responses.py:34",
  "signature": "compute_responses(network: CheckpointedNetwork, dataset_class, dataset_config: dict, batch_size, t_pre, t_fade_in, cell_index='central')",
  "returns": "xr.Dataset. Verified: stimulus (sample, frame, channel=1, hex_pixel=721) float32; responses (network_id=1, sample, frame, neuron) float32 with neuron=65 for cell_index='central', 45669 for cell_index=None. Coords: sample=arange(len) plus every arg_df column (name, original_index, vertical_split_index, temporal_split_index, frames, flip_ax, n_rot).",
  "notes": "`ds.stimulus.values[:, :, 0, :]` is the (n_samples, n_frames, 721) decoder target. It RE-INSTANTIATES the dataset from dataset_config, so pass a config dict, not an object. Calling it directly does no caching; via NetworkView it is joblib-cached."
 },
 {
  "name": "naturalistic_stimuli_responses",
  "file": ".venv/Lib/site-packages/flyvis/analysis/stimulus_responses.py:357 (also NetworkView method at network_view.py:381)",
  "signature": "naturalistic_stimuli_responses(network_view_or_ensemble, dataset=None, dt=1/100, batch_size=4, indices=None)",
  "returns": "xr.Dataset as above, plus coords time/cell_type/u/v/u_in/v_in/network_name/checkpoints added by generic_responses.",
  "notes": "Default config is AugmentedSintel(tasks=['lum'], interpolate=False, boxfilter={'extent':15,'kernel_size':13}, temporal_split=True, dt=dt, indices=indices) with t_pre=0.0, t_fade_in=2.0. NO cell_index parameter -> responses are ALWAYS the 65 central cells. Useless for a spatial (721-hexal) decoder; use stimulus_response or compute_responses(cell_index=None) instead."
 },
 {
  "name": "generic_responses",
  "file": ".venv/Lib/site-packages/flyvis/analysis/stimulus_responses.py:111",
  "signature": "generic_responses(network_view_or_ensemble, dataset, dataset_config, default_dataset_cls, t_pre, t_fade_in, batch_size, cell_index='central')",
  "returns": "xr.Dataset concatenated over network_id.",
  "notes": "The joblib-cached wrapper (cache at <network_dir>/__cache__, ignore=['batch_size']). If you pass a dataset INSTANCE it uses dataset.config.to_dict(), which for AugmentedSintel contains p_flip/p_rot/resampling — keys AugmentedSintel.__init__ swallows into **kwargs and drops. Supports cell_index=None, but caching 45669 cells x 2268 samples x 80 frames would be ~33 GB."
 },
 {
  "name": "MultiTaskSintel.original_train_and_validation_indices / sintel_utils.original_train_and_validation_indices",
  "file": ".venv/Lib/site-packages/flyvis/datasets/sintel.py:724 -> sintel_utils.py:241",
  "returns": "(train_indices, val_indices) as lists of ints, matched by scene name against 17 train and 6 validation scene names.",
  "notes": "The paper split. Hardcodes val_indices.remove(37); val_indices.remove(38) at sintel_utils.py:290-291 — valid only for the 69-row MultiTaskSintel arg_df. On AugmentedSintel it removes the wrong rows or raises ValueError. The 23 scene names here are the code evidence that Sintel training has 23 usable scenes."
 },
 {
  "name": "MultiTaskDataset.get_random_data_split",
  "file": ".venv/Lib/site-packages/flyvis/datasets/datasets.py:159 -> utils/dataset_utils.py:186",
  "signature": "get_random_data_split(fold, n_folds, shuffle=True, seed=0)",
  "returns": "(train_seq_index, val_seq_index) numpy arrays of row indices.",
  "notes": "Row-level, NOT scene-level. On AugmentedSintel this leaks: the same scene reappears as 3 vertical x N temporal x 12 geometric variants. Split on arg_df['name'] or arg_df['original_index'] instead."
 },
 {
  "name": "MultiTaskSintel.augmentation / .augment setter / .apply_augmentation",
  "file": ".venv/Lib/site-packages/flyvis/datasets/sintel.py:483 / :521 / :542",
  "returns": "context manager / property setter / Dict[str, torch.Tensor]",
  "notes": "Order is noise -> jitter -> flip -> rotate -> piecewise_resample for 'lum'; flip -> rotate -> linear_interpolate (or piecewise) for targets. piecewise_resample.augment is tied to `resampling` and linear_interpolate.augment to `interpolate` — NOT to `augment` (lines 538-539), which is why MultiTaskSintel(augment=False) still resamples."
 },
 {
  "name": "AugmentedSintel.get_item",
  "file": ".venv/Lib/site-packages/flyvis/datasets/sintel.py:953",
  "signature": "get_item(key, pad_to_length=None)",
  "returns": "Dict[str, torch.Tensor]",
  "notes": "if self.augment: apply_augmentation(..., n_rot=0, flip_axis=0) -> resampled. else: raw cached sequence, 19 frames at 24 fps, NO resampling (verified: (19,1,721) vs (40,1,721))."
 },
 {
  "name": "BoxEye",
  "file": ".venv/Lib/site-packages/flyvis/datasets/rendering/eye.py:29 (__init__ at :47)",
  "signature": "BoxEye(extent=15, kernel_size=13)",
  "returns": "callable; .hexals == 721 (verified), .min_frame_size == tensor([391, 391]) (verified), .receptor_centers (721, 2) long",
  "notes": "extent=15 -> 3*15^2+3*15+1 = 721 hexals. Rendering splits each Sintel frame to width min_frame_size[1] + 2*kernel_size = 417 px (verified). lum uses ftype='mean', flow uses ftype='sum' per channel, depth ftype='median'."
 },
 {
  "name": "connectome.central_cells_index / nodes.type / input_cell_types",
  "file": ".venv/Lib/site-packages/flyvis/network/network_view.py:107 (nv.connectome)",
  "returns": "central_cells_index (65,) int; nodes.type (45669,) bytes->str with 65 unique types; input_cell_types == ['R1'..'R8']; (nodes.type=='R1').sum() == 721",
  "notes": "All verified against flow/0000/000. The 721 R1 cells are exactly the hexal lattice, and u/v of those R1 cells are what generic_responses exposes as the u_in/v_in coords for plotting a hexal frame."
 },
 {
  "name": "DecoderGAVP / ActivityDecoder",
  "file": ".venv/Lib/site-packages/flyvis/task/decoder.py:190 / :23",
  "signature": "DecoderGAVP(connectome, shape, kernel_size=5, const_weight=0.001, n_out_features=None, p_dropout=0.5); forward(activity)",
  "returns": "activity (n_samples, n_frames, 45669) -> (n_samples, n_frames, out_channels, 721)",
  "notes": "flyvis' own flow decoder: shape=[8,2] -> 2 output channels x 721 hexals. For stimulus reconstruction use shape=[..., 1]. nv.init_decoder(checkpoint='best') recovers the trained flow decoder."
 },
 {
  "name": "temporal_split_cached_samples / temporal_split_sequence",
  "file": ".venv/Lib/site-packages/flyvis/datasets/sintel_utils.py:97 / :137",
  "returns": "(list of per-split sample dicts, repeats array)",
  "notes": "splits = int(round(n_frames_seq / max_frames)); each split is exactly max_frames long and splits OVERLAP when the sequence length is not a multiple (e.g. 49 frames -> [0:19],[15:34],[30:49]). Do not treat temporal splits from one scene as independent test samples."
 }
]

## gotchas

[
 "FLYVIS_ROOT_DIR is not set anywhere in the repository root (.env has only NEUPRINT_TOKEN, env.example doesn't mention it). Without it flyvis/__init__.py:47 resolves root_dir to .venv/Lib/site-packages/flyvis/data and your downloaded data/flyvis is invisible: no pretrained results, no connectome, no renderings. Add FLYVIS_ROOT_DIR=data/flyvis to .env (dotenv is loaded at import, flyvis/__init__.py:18).",
 "Sintel is NOT in the pretrained download. flyvis_cli/download_pretrained_models.py:13-25 lists only results_pretrained_models.zip and results_umap_and_clustering.zip. Confirmed: data/flyvis/SintelDataSet does not exist, and renderings/ holds only RenderedFlashes_0000 and RenderedOffsets_0000 — no RenderedSintel. You need MPI-Sintel-complete.zip unpacked into <root>/SintelDataSet (training/final + training/flow + test), then a one-off local render.",
 "WINDOWS BLOCKER, reproduced: rendering RenderedSintel dies with PermissionError WinError 32 on the second h5 write inside each sequence folder. datamate 1.0.0 io.py:153 _write_h5 opens h5py in mode='w', hits KeyError on f['data'], and in the except branch calls path.unlink() with that handle STILL OPEN — harmless on Linux, fatal on Windows. Fix with the monkeypatch in the snippet (patch BOTH datamate.io._write_h5 and datamate.directory._write_h5, since directory.py:578 imported the name), or pre-render under WSL. A failed render leaves renderings/RenderedSintel_0000 with status: stopped, which you must delete before retrying.",
 "AugmentedSintel(augment=False) is the WRONG way to get determinism: get_item (sintel.py:953-970) short-circuits and returns the raw cached sequence — 19 frames at Sintel's 24 fps instead of the resampled n_out (verified: (19,1,721) vs (40,1,721)). Worse, with temporal_split=False the raw lengths vary per scene and DataLoader collation fails for batch_size>1. The defaults already give determinism: contrast_std / brightness_std / gaussian_white_noise / gamma_std all default to None (identity, verified torch.equal on repeated reads) and p_flip=p_rot=0. Keep augment=True. Note MultiTaskSintel(augment=False) behaves differently and DOES resample.",
 "Network.stimulus_response MUTATES the dataset: network.py:744 does stim_dataset.dt = dt, which flows through MultiTaskSintel.__setattr__ -> update_augmentation and re-targets both resamplers. Verified: after calling with dt=1/100, dataset[0]['lum'] returned 80 frames instead of 40. Never share one dataset instance across sweeps at different dt without re-setting dataset.dt.",
 "naturalistic_stimuli_responses has NO cell_index parameter (stimulus_responses.py:357-380), so generic_responses' default 'central' applies and you ALWAYS get just the 65 central cells — one per cell type, no spatial structure. Useless as an encoder for reconstructing a 721-hexal frame. Use net.stimulus_response (all 45669) or compute_responses(..., cell_index=None).",
 "Memory: full activity is ~14.6 MB per sample (80 frames x 45669 x float32). The default AugmentedSintel augmentation (flip_axes=[0,1] x n_rotations=[0..5] = 12x) gives ~2268 samples -> ~33 GB of activity and ~0.5 GB for the stimulus array alone. Stream via net.stimulus_response, or cut flip_axes=[0], n_rotations=[0], or pass indices=.",
 "AugmentedSintel.__init__ takes **kwargs (sintel.py:802) and NEVER forwards them to super(). Misspelled or unsupported arguments are silent no-ops. This also bites generic_responses: when you hand it a dataset instance it round-trips dataset.config.to_dict(), which contains p_flip / p_rot / resampling — keys AugmentedSintel silently drops, so resampling=False would be lost.",
 "AugmentedSintel.pad_nans is broken: sintel.py:941 assigns data = {} BEFORE iterating data.items(), so get_item(key, pad_to_length=N) returns an empty dict (verified: {}). Harmless via DataLoader (__getitem__ never passes pad_to_length) but fatal if you call get_item directly with padding.",
 "original_train_and_validation_indices (sintel_utils.py:241-292) hardcodes val_indices.remove(37); val_indices.remove(38). Those positions are only meaningful for the 69-row MultiTaskSintel arg_df. Applied to AugmentedSintel it silently removes the wrong rows or raises ValueError. MultiTaskDataset.get_random_data_split is row-level and also leaks scenes across the split, since each scene reappears as 3 vertical x N temporal x 12 geometric variants. Group by arg_df['name'] or arg_df['original_index'].",
 "Temporal splits from one scene OVERLAP: temporal_split_sequence (sintel_utils.py:137-161) computes splits = round(n_frames/19), and split() re-uses frames when the length is not a multiple — e.g. 49 frames -> [0:19], [15:34], [30:49]. Do not treat temporal splits from the same scene as independent test samples.",
 "Only the 'lum' key reaches the network. stimulus_response does stim = stim[default_stim_key] (network.py:764), so flow/depth are discarded and the xarray from compute_responses has no target variable. Pull targets from the dataset separately by index: dataset[i]['flow'] -> (n_out, 2, 721). Sample order is preserved because stimulus_response uses IndexSampler(indices).",
 "Flow target units are not physical: sample_flow (sintel_utils.py:61-79) returns pixel/image_height with the y axis inverted, and the box filter aggregates it with ftype='sum' (sintel.py:130-136), so the 2x721 values are summed-over-kernel arbitrary units, not deg/s. Under hex rotation/flip the 2-vector is rotated by the matching 2x2 matrix (hex.py:74-77, 176-179) but the summation happened pre-rotation.",
 "RenderedSintel's datamate config includes the ABSOLUTE sintel_path (verified in _meta.yaml). Move or rename the data root and the entire Sintel rendering is rebuilt from scratch under a new RenderedSintel_NNNN.",
 "compute_responses' joblib key (via generic_responses / NetworkView.*_responses, cached under <network_dir>/__cache__ with ignore=['batch_size']) covers only the checkpoint, dataset class, dataset_config dict, t_pre, t_fade_in and cell_index — NOT FLYVIS_ROOT_DIR or sintel_path. A result computed against one Sintel copy is silently reused for another. (I deliberately called compute_responses directly during verification to avoid poisoning your real cache with synthetic-Sintel results.)",
 "arg_df index labels survive `indices=` subsetting (verified: indices=[0,3,5] -> arg_df.index == [0,3,5]) while compute_responses sets the xarray 'sample' coord to arange(len(dataset)). Call .reset_index(drop=True) if you rely on positional alignment. Sample ordering is product(sequence, flip_axes, n_rotations) — sequence outermost, rotation innermost.",
 "`frames` in arg_df is the RAW frame count (19), not the returned n_out (40 at dt=1/50, 80 at dt=1/100). n_out = ceil(n_frames / (24*dt)) via Interpolate (temporal.py:46-53); verified 40 / 80 / 159 for dt = 1/50, 1/100, 1/200.",
 "flyvis calls torch.set_default_device(cuda) at import (flyvis/__init__.py:13-14), so every tensor you create after `import flyvis` lands on GPU by default. Also expect a torch UserWarning about non-tuple multidimensional indexing from CropFrames.transform (temporal.py:123) on torch >= 2.8.",
 "t_fade_in is expensive: it integrates an extra int(t_fade_in/dt) steps per batch (100 extra frames at t_fade_in=2.0, dt=1/50) and is recomputed for every batch. Drop it to 0 and rely on t_pre grey steady-state if you need throughput; but then the first ~100 ms of activity is a transient your decoder will see.",
 "RenderedSintel skips scenes with len(files)-1 < n_frames (sintel.py:88), so raising n_frames silently shrinks the dataset. sintel_meta applies the same filter when building meta.lum_paths."
]

## unknowns

[
 "Exact sequence counts for the real Sintel data — Sintel is not on disk here, so I verified with a synthetic 2-scene tree. MultiTaskSintel = n_scenes x vertical_splits and n_scenes = 23 is code-evidenced (sintel_utils.py:249-276 lists 17 train + 6 validation scene names), giving 69. For AugmentedSintel(temporal_split=True, n_frames=19) the count is len(flip_axes) x len(n_rotations) x sum over the 69 vertical splits of max(1, round((N_scene_files-1)/19)). With the standard Sintel frame counts (19 scenes at 50 files, ambush_2 at 21, ambush_4 at 33, ambush_6 at 20, market_6 at 40) that is 189 x 12 = 2268 for the defaults — an estimate from my knowledge of Sintel, NOT verified from this code or disk.",
 "Download size and duration of MPI-Sintel-complete.zip, and whether http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip and the separate depth URL still resolve — I did not attempt any download.",
 "Wall-clock time and disk footprint of building RenderedSintel_0000 for the full 23 scenes. My synthetic 2-scene render took ~1.5 s for 6 splits on CPU, which says nothing useful about the real 1000+ frame render.",
 "Whether the Windows _write_h5 patch is sufficient for the full real render. It fixes the exact failure I reproduced, and the failure mode is path/handle-based rather than data-dependent, but I could not run the real render to confirm no further Windows issues appear.",
 "Whether the joblib/xarray_dataset_h5 Memory backend in NetworkView writes cleanly on Windows — I deliberately bypassed it (calling compute_responses directly) to avoid polluting your real cache, so that path is untested here.",
 "Whether depth rendering works: tasks=['depth'] needs the separate depth zip and I never exercised sample_depth or ftype='median'.",
 "GPU vs CPU throughput for a full sweep. My runs were on whatever device torch picked at import; I did not measure per-sample simulation cost at scale.",
 "Whether the pretrained flow decoder (nv.init_decoder) is a useful initialization for a stimulus-reconstruction decoder — I read the DecoderGAVP shapes but did not run it."
]