

## summary

The flyvis normalisation constant is one scalar per (model, cell type): the root-mean-square of that model's **central-column** voltage over the whole AugmentedSintel naturalistic dataset, computed as `1/sqrt(n_samples*n_frames) * ||responses||` over the sample and frame axes (`analysis/response_norms.py:432`). Two variants ship: `norm` (raw voltage) and `rectified_norm` (max(v,0) first). For the released ensemble they ship inside the wheel at `D:/ML/Fly_Brain/.venv/Lib/site-packages/flyvis/data/responses_norm.h5` — one HDF5 group per ensemble, currently only `flow/0000`, 50 models x 65 cell types, all recorded from `chkpt_00000`; I verified the local `flow/0000/000` checkpoint's SHA256 matches the shipped hash, so the constants are valid for the member at `D:/ML/Fly_Brain/data/flyvis/results/flow/0000/000`. Nothing in the package divides by it automatically: it is an opt-in `norm=` argument to the `moving_bar_responses.peak_responses` family and to `moving_edge_currents.MovingEdgeCurrentView.divide_by_given_norm`, while `flash_response_index`, `angular_tuning`, `optimal_stimuli`, clustering, and — decisively — flyvis's own trained `DecoderGAVP` do not touch it. `DecoderGAVP.forward` (`task/decoder.py:285`) feeds `relu(raw voltages)` straight into a `BatchNorm2d`, i.e. the reference decoder learns its per-channel scale from data rather than dividing by a precomputed constant. **Recommendation: raw voltages** (optionally rectified, to match the baseline), because that is what the FlyVis baseline itself does, and because the per-cell-type constant spans 76x within model 000 and ~2863x on average per model (up to 124,756x) — dividing by it is not a cosmetic rescale but an intervention that equalises exactly the response-amplitude variable your layer-wise decodability comparison is trying to measure. So yes: taking FlyVis as the baseline is genuinely simpler here — the norm is not in the decode path and needs no decision to get started; it only becomes a decision if you deliberately want to ask the amplitude-blind question.

## code_example

"""Load the flyvis response-normalisation constants for one pretrained member.

Verified to run: D:/ML/Fly_Brain/.venv/Scripts/python.exe with
FLYVIS_ROOT_DIR=D:/ML/Fly_Brain/data/flyvis
Output: (1, 1, 1, 65) float32 / T4a 0.21942745 L1 1.6164393 / spread 76.414055

Deliberately bypasses Ensemble.responses_norm(): on Windows that path returns
None and silently re-simulates the whole Sintel dataset (see gotchas).
"""
from pathlib import Path
from flyvis.analysis import response_norms as rn

MODEL_DIR = Path("D:/ML/Fly_Brain/data/flyvis/results/flow/0000/000")
ENSEMBLE = "flow/0000"        # forward slashes: the HDF5 group name
MEMBER = "flow/0000/000"      # forward slashes: the stored model name

norms = rn.read_response_norms(rn.PRECOMPUTED_FILE, ENSEMBLE)
assert norms is not None, f"{ENSEMBLE} not in {rn.PRECOMPUTED_FILE}"

# Refuse foreign constants: the shipped values must come from THIS checkpoint.
ckpt = MODEL_DIR / "chkpts" / "chkpt_00000"
assert norms.covers([MEMBER], [ckpt.name], [rn.checkpoint_hash(ckpt)]), \
    "stored constants were computed from a different checkpoint"

v = norms.select([MEMBER])                       # (1, 1, 1, 65) float32
norm_by_cell_type = dict(zip(norms.cell_types, v[0, 0, 0]))

print(v.shape, v.dtype)
print("T4a", norm_by_cell_type["T4a"], "L1", norm_by_cell_type["L1"])
print("spread across cell types:",
      max(norm_by_cell_type.values()) / min(norm_by_cell_type.values()))

# --- how a decoder should NOT and SHOULD use it -------------------------------
# NOT: responses = responses / v          # equalises the very amplitude
#                                         # difference the decodability map measures
# DO:  feed raw (or relu'd) voltages, exactly like flyvis's own DecoderGAVP,
#      and let the decoder own its scaling:
#        - ridge: tune lambda per cell type by CV (scale-equivariant, so the
#          answer is then invariant to any per-channel rescale anyway)
#        - hex-CNN: put a BatchNorm as the first layer, which is literally what
#          task/decoder.py:260-261 does
#      Report norm_by_cell_type ALONGSIDE the decodability score as a covariate,
#      never as a divisor.

## api

[
 {
  "name": "_norm (definition of the constant)",
  "file": "D:/ML/Fly_Brain/.venv/Lib/site-packages/flyvis/analysis/response_norms.py:432",
  "signature": "_norm(responses: np.ndarray, rectified: bool = False) -> np.ndarray",
  "returns": "(n_models, n_cell_types) float32. RMS over axes (1,2) of responses shaped (n_models, n_samples, n_frames, n_cell_types): 1/sqrt(n_samples*n_frames) * np.linalg.norm(responses, axis=(1,2)). NaNs -> 0; rectified=True applies np.maximum(responses, 0) first.",
  "notes": "Module-private but this is the whole definition. No baseline/resting-potential subtraction, so it is RMS of the absolute voltage including any tonic offset - not the std of fluctuations. That is why L1/L2/Lawf1/Am have norm ~1.6 but rectified_norm exactly 0.0."
 },
 {
  "name": "PRECOMPUTED_FILE",
  "file": "D:/ML/Fly_Brain/.venv/Lib/site-packages/flyvis/analysis/response_norms.py:57",
  "signature": "PRECOMPUTED_FILE: Path = flyvis.package_dir / \"data\" / \"responses_norm.h5\"",
  "returns": "Path -> D:/ML/Fly_Brain/.venv/Lib/site-packages/flyvis/data/responses_norm.h5 (57,480 bytes; in the wheel RECORD at flyvis-1.2.0.dist-info/RECORD:57)",
  "notes": "This is where the precomputed constants ship - inside the package, NOT under a results/ directory. Contains exactly one top-level group: 'flow'. Group 'flow/0000' holds datasets model_names(50), cell_types(65), checkpoints(50), checkpoint_hashes(50), norm(50,65), rectified_norm(50,65), plus attrs dataset_config and flyvis_version='1.1.4.dev6+g13a7705ec.d20260509'."
 },
 {
  "name": "ENSEMBLE_FILE_NAME",
  "file": "D:/ML/Fly_Brain/.venv/Lib/site-packages/flyvis/analysis/response_norms.py:60",
  "signature": "ENSEMBLE_FILE_NAME: str = \"responses_norm.h5\"",
  "returns": "str. The per-ensemble cache written to <ensemble_dir>/responses_norm.h5 (path built by ensemble_file(), response_norms.py:311).",
  "notes": "For this project that would be D:/ML/Fly_Brain/data/flyvis/results/flow/0000/responses_norm.h5. It does not currently exist."
 },
 {
  "name": "ResponseNorms",
  "file": "D:/ML/Fly_Brain/.venv/Lib/site-packages/flyvis/analysis/response_norms.py:63",
  "signature": "@dataclass ResponseNorms(ensemble_name, model_names, cell_types, checkpoints, norm, rectified_norm, dataset_config='', checkpoint_hashes=[])",
  "returns": "Dataclass. .norm and .rectified_norm are (n_models, n_cell_types) float32; .cell_types is the 65-entry list in connectome.unique_cell_types order (R1..TmY18).",
  "notes": "Everything is keyed by model NAME, never position."
 },
 {
  "name": "ResponseNorms.select",
  "file": "D:/ML/Fly_Brain/.venv/Lib/site-packages/flyvis/analysis/response_norms.py:172",
  "signature": "select(model_names: List[str], rectified: bool = False) -> np.ndarray",
  "returns": "(n_models, 1, 1, n_cell_types) float32 - broadcastable against responses shaped (network_id, sample, frame, neuron).",
  "notes": "Raises KeyError for a name not stored. Names must use '/' separators as stored in the file ('flow/0000/000')."
 },
 {
  "name": "ResponseNorms.covers",
  "file": "D:/ML/Fly_Brain/.venv/Lib/site-packages/flyvis/analysis/response_norms.py:106",
  "signature": "covers(model_names, checkpoints, checkpoint_hashes=None) -> bool",
  "returns": "bool. True only if every model is stored AND its checkpoint SHA256 matches (falls back to comparing checkpoint file names when either side has no hash).",
  "notes": "This is the guard that stops foreign constants being applied to a retrained checkpoint. Verified: covers(['flow/0000/000'], ['chkpt_00000'], [sha256 of the local file]) -> True; with a wrong hash -> False."
 },
 {
  "name": "read_response_norms",
  "file": "D:/ML/Fly_Brain/.venv/Lib/site-packages/flyvis/analysis/response_norms.py:243",
  "signature": "read_response_norms(path: Path, ensemble_name: str) -> Optional[ResponseNorms]",
  "returns": "ResponseNorms or None (None if the file or the named HDF5 group is absent).",
  "notes": "THE portable entry point on Windows. Call it with the literal forward-slash group name 'flow/0000'."
 },
 {
  "name": "load_response_norms",
  "file": "D:/ML/Fly_Brain/.venv/Lib/site-packages/flyvis/analysis/response_norms.py:361",
  "signature": "load_response_norms(ensemble: Ensemble) -> Optional[ResponseNorms]",
  "returns": "ResponseNorms or None. Tries <ensemble_dir>/responses_norm.h5 then PRECOMPUTED_FILE, requiring covers() on the current checkpoints.",
  "notes": "BROKEN ON WINDOWS - returns None for flow/0000 because ensemble.name is 'flow\\\\0000' while the HDF5 group is 'flow/0000'. Empirically verified (see gotchas)."
 },
 {
  "name": "compute_response_norms",
  "file": "D:/ML/Fly_Brain/.venv/Lib/site-packages/flyvis/analysis/response_norms.py:385",
  "signature": "compute_response_norms(ensemble: Ensemble) -> ResponseNorms",
  "returns": "ResponseNorms for every member, from network_view.naturalistic_stimuli_responses() per model.",
  "notes": "The expensive fallback: simulates the full AugmentedSintel dataset for every uncached model. This is what silently fires on Windows when load_response_norms() returns None."
 },
 {
  "name": "responses_norm",
  "file": "D:/ML/Fly_Brain/.venv/Lib/site-packages/flyvis/analysis/response_norms.py:483",
  "signature": "responses_norm(ensemble, rectified=False, force_recompute=False, store=True) -> np.ndarray",
  "returns": "(n_models, 1, 1, n_cell_types) in ensemble.names order.",
  "notes": "store=True by default: newly computed constants are written into the ensemble directory as a side effect."
 },
 {
  "name": "Ensemble.responses_norm",
  "file": "D:/ML/Fly_Brain/.venv/Lib/site-packages/flyvis/network/ensemble.py:840",
  "signature": "Ensemble.responses_norm(self, rectified=False, force_recompute=False, store=True) -> np.ndarray",
  "returns": "(n_models, 1, 1, 65). Thin delegate to analysis.response_norms.responses_norm (ensemble.py:864).",
  "notes": "This is the ONLY public accessor in the package. There is no per-model equivalent."
 },
 {
  "name": "NetworkView - no norm API",
  "file": "D:/ML/Fly_Brain/.venv/Lib/site-packages/flyvis/network/network_view.py:1",
  "signature": "(none)",
  "returns": "N/A - grep for 'norm' in network_view.py returns zero hits.",
  "notes": "Answers the task question directly: there is no 'norm' property or method on NetworkView. To get the constant for a single pretrained member you either build a one-member Ensemble or read the HDF5 directly (the latter is what I recommend on Windows)."
 },
 {
  "name": "checkpoint_hash",
  "file": "D:/ML/Fly_Brain/.venv/Lib/site-packages/flyvis/analysis/response_norms.py:323",
  "signature": "checkpoint_hash(path) -> str",
  "returns": "SHA256 hex digest, or '' if unreadable.",
  "notes": "Local D:/ML/Fly_Brain/data/flyvis/results/flow/0000/000/chkpts/chkpt_00000 -> d0e42857e738d0315897c2d50fde9eb3fb1a3fb1f071d55dfc13d53d72bccc3f, identical to the shipped hash for flow/0000/000. The shipped constants are therefore valid for this download."
 },
 {
  "name": "checkpoint_identity",
  "file": "D:/ML/Fly_Brain/.venv/Lib/site-packages/flyvis/analysis/response_norms.py:343",
  "signature": "checkpoint_identity(ensemble) -> Tuple[List[str], List[str]]",
  "returns": "(checkpoint file names, SHA256s) in ensemble.names order.",
  "notes": "Uses network_view.network(checkpoint='best', lazy=True).checkpoint."
 },
 {
  "name": "moving_bar_responses.peak_responses  (DIVIDES by norm)",
  "file": "D:/ML/Fly_Brain/.venv/Lib/site-packages/flyvis/analysis/moving_bar_responses.py:41",
  "signature": "peak_responses(dataset: xr.Dataset, norm: xr.DataArray = None, from_degree=None, to_degree=None) -> xr.DataArray",
  "returns": "Peak (max over 'frame') of time-masked, rectified responses; divided by norm at moving_bar_responses.py:74-75 only if norm is not None.",
  "notes": "norm defaults to None, so the package does NOT normalise unless the caller passes it. Note it rectifies at line 71 (clip(min=0)) - so if you pass a norm here, the matching variant is rectified_norm, which is 0 for several cell types."
 },
 {
  "name": "moving_bar_responses.peak_responses_angular / direction_selectivity_index / preferred_direction (pass norm through)",
  "file": "D:/ML/Fly_Brain/.venv/Lib/site-packages/flyvis/analysis/moving_bar_responses.py:123",
  "signature": "peak_responses_angular(...norm=None...) :123 ; direction_selectivity_index(...norm=None...) :155 ; preferred_direction(...norm=None...) :515",
  "returns": "Complex-valued peak responses / DSI / preferred direction.",
  "notes": "All three only forward norm to peak_responses (lines 142, 176, 536). DSI and preferred direction are ratios/angles, so a per-cell-type scalar cancels out of them anyway - passing norm there is near-inert by construction."
 },
 {
  "name": "moving_edge_currents.MovingEdgeCurrentView.divide_by_given_norm (DIVIDES by norm)",
  "file": "D:/ML/Fly_Brain/.venv/Lib/site-packages/flyvis/analysis/moving_edge_currents.py:181",
  "signature": "divide_by_given_norm(self, norm: CellTypeArray) -> MovingEdgeCurrentView",
  "returns": "New view with responses and ALL input currents divided by the norm of the TARGET cell type (lines 197-214).",
  "notes": "Requires a CellTypeArray (nodes_edges_utils.py:233), not a bare ndarray - raises ValueError otherwise. Nothing inside the package calls it; it is for figure notebooks, which do not ship in the wheel."
 },
 {
  "name": "analyses that do NOT use the norm",
  "file": "D:/ML/Fly_Brain/.venv/Lib/site-packages/flyvis/analysis/flash_responses.py:25",
  "signature": "flash_response_index(...) :25 (flash_responses.py) ; angular_tuning(...) :637 (moving_bar_responses.py) ; optimal_stimuli.py (zero 'norm' hits) ; clustering.umap_embedding :432 ; validation.py",
  "returns": "N/A",
  "notes": "FRI is (on_peak - off_peak)/(on_peak + off_peak + 1e-16), scale-invariant by construction, and takes no norm argument. angular_tuning's own docstring (moving_bar_responses.py:661-662) says 'Not normalized, so that tunings of different models or reductions can be compared on a common scale'. clustering embeds ONE cell type at a time with metric='correlation' (scale-invariant) - so it never rescales across cell types either."
 },
 {
  "name": "DecoderGAVP.forward - the FlyVis baseline decoder",
  "file": "D:/ML/Fly_Brain/.venv/Lib/site-packages/flyvis/task/decoder.py:285",
  "signature": "forward(self, activity: torch.Tensor) -> torch.Tensor",
  "returns": "(n_samples, n_frames, out_channels, n_hexals)",
  "notes": "DECISIVE for the recommendation. Line 297: x = nnf.relu(self.dvs_channels.output) - raw voltages, rectified, NO division by any norm. The first learned op is nn.BatchNorm2d (decoder.py:251 inside the conv stack, and decoder.py:260-261 when shape has no hidden layers), i.e. the reference decoder learns a per-channel scale+shift from data. in_channels = len(connectome.output_cell_types) = 34 (T1..TmY18)."
 },
 {
  "name": "stimulus_responses.generic_responses / compute_responses",
  "file": "D:/ML/Fly_Brain/.venv/Lib/site-packages/flyvis/analysis/stimulus_responses.py:111",
  "signature": "generic_responses(..., cell_index: Optional[np.ndarray|str] = \"central\") :111 ; compute_responses(..., cell_index=\"central\") :34",
  "returns": "xr.Dataset with 'responses' dims ('network_id','sample','frame','neuron').",
  "notes": "cell_index='central' -> np.take(responses, connectome.central_cells_index, axis=-1) at stimulus_responses.py:56,72. So the 65 'neuron' entries are one central cell per type, and the norm is a CENTRAL-COLUMN statistic, not a whole-hex-field one."
 },
 {
  "name": "naturalistic_stimuli_responses (the stimulus the norm is measured on)",
  "file": "D:/ML/Fly_Brain/.venv/Lib/site-packages/flyvis/analysis/stimulus_responses.py:357",
  "signature": "naturalistic_stimuli_responses(network_view_or_ensemble, dataset=None, dt=1/100, batch_size=4, indices=None)",
  "returns": "xr.Dataset of AugmentedSintel responses; NetworkView wrapper at network/network_view.py:381.",
  "notes": "Config recorded in the shipped file: {'tasks': ['lum'], 'interpolate': false, 'boxfilter': {'extent': 15, 'kernel_size': 13}, 'temporal_split': true, 'dt': 0.01, 'indices': null}. indices=null means ALL Sintel sequences - see the leakage gotcha."
 },
 {
  "name": "CLI: flyvis responses-norm",
  "file": "D:/ML/Fly_Brain/.venv/Lib/site-packages/flyvis_cli/analysis/responses_norm.py:1",
  "signature": "flyvis responses-norm task_name=flow ensemble_id=0000 [--force] [--export]",
  "returns": "Writes <ensemble_dir>/responses_norm.h5; --export additionally writes into PRECOMPUTED_FILE (registered in flyvis_cli/flyvis_cli.py:40).",
  "notes": "--export is described in the script's own help as 'For maintainers preparing a release of an ensemble' - do not run it, it mutates the installed package."
 }
]

## gotchas

[
 "WINDOWS BUG, CONFIRMED EMPIRICALLY: Ensemble.responses_norm() and load_response_norms() never find the shipped constants on this machine. model_path_names (network/ensemble.py:882) builds names with os.path.sep, so ensemble.name == 'flow\\\\0000' and ensemble.names == ['flow\\\\0000\\\\000'], while the HDF5 group and stored model names use '/'. Measured: read_response_norms(PRECOMPUTED_FILE, e.name) -> None, read_response_norms(PRECOMPUTED_FILE, 'flow/0000') -> ResponseNorms(flow/0000, 50 models, 65 cell types), load_response_norms(e) -> None. Consequence: responses_norm() (response_norms.py:506-510) falls through to compute_response_norms() and silently simulates the full AugmentedSintel dataset for every member, then writes into your results dir because store=True. Always go through read_response_norms(PRECOMPUTED_FILE, 'flow/0000') with literal forward slashes.",
 "rectified_norm IS EXACTLY ZERO for several cell types: for member 000, L1, L2, Lawf1 and Am all have rectified_norm == 0.0 (their central voltage is negative across the whole of Sintel), and TmY14 is 0.0062, Tm4 0.0089. Across the shipped 50x65 table, 66 of 3250 rectified entries are < 1e-12. Dividing by rectified_norm gives inf/NaN. There is no epsilon guard anywhere - peak_responses just does `rectified / norm` (moving_bar_responses.py:75).",
 "THE LEAK YOU ASKED ABOUT IS REAL BUT IT IS A SCALE LEAK, NOT A LABEL LEAK, AND ITS DAMAGE IS TO THE COMPARISON, NOT TO THE SCORE. Dividing by the per-cell-type constant makes every cell type's Sintel RMS equal 1 by construction. Measured spread in the shipped table: 76x between the loudest and quietest cell type within member 000 (T2 2.965 vs Tm4 0.0388); mean 2863x per model across the ensemble, max 124,756x; the smallest single entry is flow/0000/016 C2 at 1.0e-4. So the norm is not a cosmetic rescale - it is a 3-to-5-orders-of-magnitude per-channel reweighting. A per-channel scalar is invertible, so information content (and hence a CV-tuned ridge's R^2, since ridge is scale-equivariant and the optimal lambda just scales by s^2) is unchanged. But with a FIXED lambda shared across cell types - the obvious shortcut when building a 65-entry map - normalising completely rewrites the ranking: raw voltages penalise quiet channels (they need huge weights), normalised voltages let them compete evenly. The same applies to a hex-CNN with fixed init scale and learning rate. FLAG IT: per-cell-type rescaling equalises response amplitude, which is one of the plausible mechanistic reasons a cell type is more decodable. If you normalise, you have silently switched from 'which cell type carries more recoverable stimulus information' to 'which cell type's waveform shape is more informative, amplitude aside'. That is a legitimate but different question and must be stated, not defaulted into.",
 "SECONDARY, GENUINE TRAIN/VAL LEAK: the shipped constants were computed with indices=null (dataset_config attr in the h5), i.e. over ALL Sintel sequences, while the networks themselves were trained on a 4-fold sequence split (task/tasks.py:79-84 via utils/dataset_utils.py:186 get_random_data_split). So the constants mix each member's training and held-out sequences. It is only one float per cell type, but if your decoder is evaluated on Sintel you would be rescaling with a statistic that saw the eval frames. If you want scale invariance, compute it from the decoder's own TRAIN split, not from this file.",
 "THE NORM IS A CENTRAL-COLUMN STATISTIC. generic_responses defaults to cell_index='central' (stimulus_responses.py:119) and compute_responses takes connectome.central_cells_index (stimulus_responses.py:54-56, 72), so the 65 'neuron' entries are one central cell per type. Your decoders read back to the 721-hexal input, presumably from the full hex field per cell type. Applying a central-column RMS to a whole field is an extrapolation the package never makes.",
 "THE NORM'S PURPOSE IS CROSS-MODEL, NOT CROSS-CELL-TYPE, COMPARABILITY. select() returns shape (n_models, 1, 1, n_cell_types) to make the 50 ensemble members' amplitudes commensurate with each other for tuning-curve figures. Nothing in the package or its docstrings proposes it as a way to compare cell types to each other. angular_tuning's docstring (moving_bar_responses.py:661) is explicit that it leaves things unnormalised so different models sit on a common scale.",
 "NO BASELINE SUBTRACTION ANYWHERE. Network.stimulus_response (network/network.py:754) starts from steady_state(t_pre, dt, value=0.5) and returns absolute voltages; _norm does not centre them. So `norm` is RMS about zero including any tonic offset, not the std of fluctuations - which is exactly why L1's norm is 1.62 while its rectified norm is 0. If your decoder wants a variance-like quantity, this file is not it.",
 "ONLY flow/0000 SHIPS. The precomputed file's single top-level group is 'flow'; only 'flow/0000' exists under it. Any other task or ensemble id triggers the full recompute path. All 50 entries record checkpoint 'chkpt_00000', which is the only checkpoint present in this local download.",
 "best-checkpoint resolution emits a warning on this install: \"epe not in ...results/flow/0000/000/validation, but 'loss' is. Falling back to 'loss'.\" Harmless here because each member has exactly one checkpoint, but checkpoint_identity() depends on best_checkpoint_fn, so on a download with several checkpoints a different loss file would change which checkpoint is hashed and covers() would then reject the shipped constants.",
 "FLYVIS_ROOT_DIR must be set or flyvis.results_dir resolves to <site-packages>/flyvis/data/results and Ensemble loading fails with StopIteration at ensemble.py:186 (observed). Set FLYVIS_ROOT_DIR=D:/ML/Fly_Brain/data/flyvis.",
 "divide_by_given_norm demands a CellTypeArray (utils/nodes_edges_utils.py:233) and raises a bare ValueError for an ndarray, and it divides a target cell's INPUT CURRENTS by the TARGET's norm (the code comments this explicitly at moving_edge_currents.py:45-46 of that method) - so currents from different sources become mutually comparable but are no longer in their own units.",
 "Do not run `flyvis responses-norm --export`: it writes into the installed package's responses_norm.h5 (flyvis_cli/analysis/responses_norm.py:77-79). The plain invocation is safe and writes only to the ensemble directory."
]

## unknowns

[
 "I could not verify the module docstring's '~30 minutes of naturalistic stimuli' claim by measurement - that would require instantiating AugmentedSintel and simulating. I only read the config recorded in the shipped file: tasks=['lum'], boxfilter extent 15 / kernel 13, temporal_split=true, dt=0.01, indices=null.",
 "The installed dist-info METADATA contains no release notes and no 'norm' text, so I could not read the v1.2.0 release notes the task refers to. I established the shipping claim from the module docstring (response_norms.py:11-15) plus the wheel RECORD entry (flyvis-1.2.0.dist-info/RECORD:57) - both first-hand, but not from the release notes themselves.",
 "Which published FlyVis figures actually passed norm= to peak_responses. The wheel ships only flyvis/ and flyvis_cli/ - no examples/ or notebooks/ directory - so there is no in-package call site to inspect. My statement that nothing in the package normalises automatically is verified; my inference about intended figure usage is inference.",
 "I verified the checkpoint SHA256 match only for flow/0000/000 (d0e42857...). I did not hash the other 49 members, so I cannot assert covers() would return True for the whole local ensemble.",
 "Whether the 76x / 2863x amplitude spread I measured is specific to Sintel. These constants are stimulus-distribution-dependent; the per-channel scale spread under whatever stimulus set your decoders train on could differ. I did not measure it on any other stimulus.",
 "Whether the project's ridge decoders will tune lambda per cell type or share one. That choice decides whether normalisation is inert (per-cell-type CV) or decisive (shared fixed lambda), and I could not determine it from ROADMAP.md item 4, which names ridge / hex-CNN / encoder inversion but not the regularisation protocol."
]