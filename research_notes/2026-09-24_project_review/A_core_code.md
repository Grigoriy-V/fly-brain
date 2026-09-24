# Review A: core library (model, train, data, decode, config, core tests)

Reviewer A, 2026-09-24. Read-only review of `flydream/model/`, `flydream/train/`,
`flydream/data/`, `flydream/decode/`, `config.toml` and the tests that cover them.
The checks were short local Python runs with `.venv/Scripts/python.exe` on the
existing export `data/ol/filters_R_w5wk50m500oc.json` and FlyVis member
`flow/0000/000`. No network, no Modal, no file edits except this note. The scratch
scripts live in the session scratchpad, not in the repository.

## Checked and correct (not findings)

- **Transplant rescale** (`zero.transplant`, `zero.total_n_syn`). The rescale is correct
  and it is applied at every call site: `zero.main`, `init_state.main`,
  `decode/map.py:228` and `generate/invert.py:68`. All four use `rescale=True` and
  the config cap. The pretrained config (`data/flyvis/results/flow/0000/000/_meta.yaml`)
  has `syn_count` `requires_grad: false`, so `syn_count` comes from the connectome
  and does not need to be carried over. In FlyVis's dynamics, `syn_count` is the mean
  `n_syn` per (src, tar, du, dv), which is the same value `total_n_syn` sums. Measured
  after transplant (member 000, v9 export):
  - Per-pair drive (gain × total per target cell) on the 477 matched pairs:
    p50 0.091, p90 0.676, p99 1.93, max 4.17.
  - FlyVis's own: p50 0.097, p90 0.691, p99 2.21, max 4.17.
  - The 198 defaulted pairs get the unrescaled FlyVis median gain (0.0181). Their
    drive is at most 0.144, below FlyVis's p90, so the default does no harm on this
    export.
- **Decoder data handling.**
  - `Pairs.frames`: the lag alignment is correct, and so are the dropped edge frames.
  - Ridge: the penalty is chosen on held-out training groups, and the final model is
    refitted on train only.
  - The subset control's column mapping across lags is correct.
  - `shuffle_samples` on the Sintel pairs: k = 160 of 189, and 0 samples are paired
    with activity from their own scene.
- **Signs.** The export's signs agree with FlyVis on every pair both carry (0
  disagreements). 42 of those pairs, all into R1..R6, are signed from MaleCNS NT
  because of the pooled key, but they agree.
- **Repository contents.** No data, weights or secrets are committed in this area.
  `columns_roi.py` reads `NEUPRINT_TOKEN` from `.env` only.

## Findings

### A1 — high — Am striding makes L3 a hyperpolarised sublattice (the likely cause of ISS-0015's L3 dots)

- **Where:** `flydream/data/export.py:310-316` (stride from density; only R1-R8 are exempt), `:135` ("src" normalisation), `flydream/model/zero.py:122` (the cap then multiplies it).
- **Statement:** MaleCNS has 47 Lai fragments for Am (ISS-0005 a, a reconstruction gap). As a result:
  - the export places Am at stride [5, 4], one cell per 20 columns;
  - each Am cell carries a filter onto L3 with a strong centre;
  - the transplant multiplies that gain by the cap (×3; the asked factor is 7.8).

  So L3 cells that sit exactly on Am positions get ~3× FlyVis's Am inhibition. The others get almost none. FlyVis's L3 gets the same −0.563 everywhere.
- **Evidence (run):**
  - Per-cell Am→L3 input after transplant: MaleCNS mean −0.171, sd 0.335, min −1.563. FlyVis: −0.563 at every cell (sd 0).
  - Across all source→target pairs, per-target input CV over interior cells (ring ≤ 10): Am→L3 is the single outlier at 1.85. Next are Mi15→Mi4 at 0.97 (only 52 % of Mi4 cells receive Mi15, stride [2, 1]) and Tm5c→Mi4 at 0.73. FlyVis has CV 0 for all of them.
  - At the grey steady state (value 0.5, 2 s), L3 on the 37 Am positions averages **−0.548**; every other L3 cell sits at 0.021 ± 0.038. **All 37 of the 37 most hyperpolarised L3 cells sit on an Am position.**
- **Failure scenario:** every generator state comes from model zero (ISS-0015 "Costs"). L3 feeds Tm9 and Tm9 feeds T5, so the OFF half carries a periodic artefact of a reconstruction gap. ISS-0015 still records its cause as "unknown".
- **Fix:**
  - Treat Am like the photoreceptors. Its MaleCNS count is not its density, so tile it [1, 1] and normalise its filter per target column so that each L3 gets FlyVis-like uniform input.
  - More generally, for a sparse source, do not concentrate a per-source-cell filter on a sublattice.
  - Add a check that fails when per-target input CV exceeds a bound (see A12). Then look at Mi15→Mi4 and Tm5c→Mi4 as well.
- **Confidence:** confirmed by running, both the connectivity and the steady state. That this is the pattern in ISS-0015's Sintel picture is plausible (same lattice); I did not re-render that figure.

### A2 — medium — the column count for striding is the largest type (Tm3, 1,037), not the columns (~892); threshold and stride set are written in code

- **Where:** `flydream/data/export.py:310` (`n_columns = max(...)  # ~ one per column`), `:165` (`STRIDES`), `:171` (`density >= 0.7`).
- **Statement:** Tm3 has more than one cell per column, so every density is underestimated by 892/1,037. The density→stride threshold (0.7) and the candidate strides are constants in code, not settings in `config.toml`.
- **Evidence (run):**
  - Tm3 has 1,037 cells; L1, L2, L3, C3 and Mi1 have 892, 893, 892, 892 and 887.
  - With 892 as the reference, 11 types change stride:
    - TmY5a 678 and TmY18 694: [2, 1] → [1, 1];
    - Tm5c and TmY3: [3, 1] → [2, 1];
    - Tm5b, TmY4, TmY10: [2, 2] → [3, 1];
    - Lawf1, Lawf2, Tm16, TmY13: [3, 2] → [2, 2].
- **Failure scenario:** model zero's wiring (cell counts, and which types can be outputs) depends on an arbitrary choice that is not recorded. TmY5a and TmY18 are dropped from `output_units` as "strided" only because of it.
- **Fix:** derive `n_columns` from a columnar reference, for example the median over L1/L2/L3/C3/Mi1 or the number of distinct column ROIs. Name that reference, the 0.7 threshold and the stride set in `config.toml [data]` with the reason. The change needs a new export tag (A3).
- **Confidence:** confirmed by running.

### A3 — medium — the export's identity ignores `normalisation` and `min_filter_syn`, so a changed setting silently rewrites the "same" export

- **Where:** `flydream/data/export.py:187-205` (`filters_tag`, `filters_path`), `:302`, `flydream/model/zero.py:317`, `flydream/decode/map.py:209`.
- **Statement:** the `filters_path` docstring says "a changed setting reads a different file and never a silently re-exported one". But the tag only encodes `min_weight`, `weak_pair_*` and `output_units_columnar_only`.
- **Evidence (run):** with `normalisation = "tar"`, and separately with `min_filter_syn = 1.0`, `filters_path` still returns `filters_R_w5wk50m500oc.json`.
- **Related overwrites:**
  - `filter_comparison_<side>.csv` has no tag and is overwritten by every export.
  - The default run directories `<date>_step2_zero_<side>` and `<date>_decode_<stimuli>` are opened with `exist_ok=True`, so a same-day rerun overwrites them.
- **Failure scenario:**
  - A re-export with another normalisation overwrites the file that v9, `init_w5wk50m500oc_m000.pt` and every generator run name as their connectome. The file name no longer pins the content.
  - This breaks AGENTS "never silently overwrite a measured run". `content_addressed` only protects datamate's cache.
- **Fix:** put a hash of the whole `[data]` table, or every key, into the tag. Refuse to overwrite an existing export whose content differs. Tag the comparison CSV the same way.
- **Confidence:** confirmed by running.

### A4 — medium — the SSIM on the raster has a ~0.3 floor from the zero fill outside the lattice

- **Where:** `flydream/decode/metrics.py:80-81` (`to_raster(..., fill=0.0)` for both images).
- **Statement:** 26 % of the 94×94 raster lies outside the hexagon. Both images are 0 there, so SSIM is ≈ 1 in that region and near the border. The mean over the raster is therefore high for any prediction.
- **Evidence (run, `pix_per_hex=3`, `win=9`):**

  | Case | SSIM |
  |---|---|
  | Two independent random hexal frames (sd 0.18 around 0.5) | 0.262 |
  | Constant-mean prediction | **0.320** (a blind flat guess beats a random frame) |
  | Pixels outside the mask, independent frames | 0.903 |
  | Interior pixels, independent frames | 0.062 |

- **Failure scenario:** `reports/2026-09-18_step4_decoder_stack.md:67` reads SSIM control 0.670 against real 0.769 and puts the weak discrimination down to local structure between neighbouring hexals. Part of it is this floor, which also rewards blur over detail.
- **Fix:** average the SSIM map only over pixels whose whole window lies inside the mask (erode the mask by `win//2`), or use NaN fill with a masked mean. Rerun the column if SSIM is quoted again.
- **Confidence:** confirmed by running.

### A5 — medium — "held-out scenes" are Sintel shots whose sibling shots of the same set stay in training

- **Where:** `flydream/decode/pairs.py:117-134` (`scene_labels` keeps `alley_1`, `alley_2`, ... as separate groups), `config.toml:163-166` (`held_scenes`, and the comment "so the test is about kinds of video the model never saw").
- **Evidence (run on `data/decode/2026-09-18_decode_sintel_flyvis/pairs.npz`):**
  - There are 23 groups. The seed-0 test set is ambush_5, bandage_2, cave_2, shaman_3 and sleeping_2.
  - Each has a sibling in training: ambush_2/4/6/7, bandage_1, cave_4, shaman_2, sleeping_1.
  - The same holds for 13A's `held_scenes`: bamboo_2/bamboo_1, cave_4/cave_2, market_6/market_2,5, temple_3/temple_2.
- **Failure scenario:** held-out scores (the decoder map, 13A/13B test numbers) are read against a claim of unseen scenes. Siblings share set, characters, lighting and textures, so the numbers can overstate generalisation. The size of the effect is not measured.
- **Fix:** group by set (strip the trailing `_\d+`, which gives 12 sets), or narrow the claim to "held-out shots".
- **Confidence:** confirmed by running (the group lists). The effect on scores is plausible only.

### A6 — low — expanding pooled R1-R6 copies the pool's per-cell filter to each of six nodes (6× synapse mass)

- **Where:** `flydream/data/export.py:326-352`.
- **Statement:**
  - On the target side, with "src" normalisation, each of R1..R6 receives the full per-source-cell count into the whole R1-R6 pool. For example, Am→R1..R6 is 19.62 synapses per Am cell on each of the six edges.
  - On the source side, 887 R1-R6 cells (~1 per column) become six nodes per column, each with the per-cell filter.
- **Evidence:** the export's edges into R1..R6 (Am 19.62 ×6, L4 4.61 ×6, L2 2.22 ×6), and n_cells_malecns 887 for each expanded node.
- **Failure scenario:** the transplant's per-pair rescale absorbs this where the factor is inside the cap. Where the factor is capped, the pathway stays off:
  - L4→R2 asks for 0.223 and gets 1/3, so it stays about 1.5× FlyVis;
  - R1..R6→Am asks for 30-34 and gets 3.

  A training run from flyvis's own initialisation (the "from scratch" control) would inherit the raw 6×.
- **Fix:** divide by the number of expanded members on the side where the MaleCNS count is not per member.
- **Confidence:** confirmed by reading the code and the export values.

### A7 — low — the mini-network test says it is isolated in a temporary root, but it writes into the project's `data/flyvis`

- **Where:** `tests/test_mini_network.py:4-5,24`.
- **Evidence:**
  - `data/flyvis/connectome/ConnectomeFromAvgFilters_000b/_meta.yaml` names `file: D:\ML\Fly_Brain\tests\fixtures\mini_connectome.json`.
  - flyvis fixes `root_dir` at first import (`flyvis/__init__.py:54`), and the connectome class's root at class definition. Setting `FLYVIS_ROOT_DIR` inside the fixture has no effect once `flydream.model` or `flydream.data.export` (collected earlier) has imported flyvis.
  - `.venv/Lib/site-packages/flyvis/data/connectome/` also holds builds. They come from runs where flyvis was imported first without the env var.
- **Failure scenario:** the offline rule is not broken (no network), but the test writes outside tmp and its docstring is false.
- **Fix:** set `FLYVIS_ROOT_DIR` in a `tests/conftest.py` before any import, or build under `datamate.set_root_context(tmp)`.
- **Confidence:** confirmed by reading the cache.

### A8 — low — `diverged()` does not do what its docstring says

- **Where:** `flydream/model/zero.py:243-247`.
- **Statement:** the docstring says "or NaN already in the first frame". The code checks only `isinf` and `nanmax(|r|) > 1e6`. An all-NaN response gives `nanmax` = NaN, and NaN > 1e6 is False, so it counts as not diverged.
- **Failure scenario:**
  - The run is not reported as diverged.
  - `score_fri` then counts `sign(NaN) == known` as a polarity disagreement.
  - The "all members stable" reading would miss it.
- **Confidence:** confirmed by reading; not observed in a run.

### A9 — low — config keys read nowhere, and limits written in code

- **Unread keys:** `[flyvis] version` and `[flyvis] ensemble` are read nowhere.
- **Hard-coded ensemble path:** "flow/0000" is written in `zero.py:328`, `init_state.py:47`, `decode/map.py:41,225` and `generate/invert.py:66`.
- **Other limits in code, not in `config.toml`:**
  - `ridge.ALPHAS` (−9..6, `ridge.py:38`);
  - `identification(min_std=1e-3)` (`metrics.py:102`);
  - the divergence bound 1e6 (`zero.py:247`);
  - the protocol dt and speeds (`zero.py:304-305`);
  - `video_corpus.SINTEL_BAND`, which is a fallback only (the manifest shows the derived band was used), and `max − min > 0.05` (`video_corpus.py:42,53`).
- **Fix:** read the ensemble from config, or delete the keys. Name the rest in config with the reason.
- **Confidence:** confirmed by grep.

### A10 — low — dead code

- `export.choose_orientation` (`export.py:148-162`) is never called; `main` uses `orientation.choose`. `hex_symmetries` (`:46`) is used only by that dead function, and `tests/test_data_helpers.py` tests dead code.
- `flydream/decode/hexconv.py` has no caller outside tests. `docs/PROJECT_MAP.md` calls rung two "built" with "the full sweep … a Modal job", but no app or tool invokes it.
- **Confidence:** confirmed by grep.

### A11 — low — decode runs do not record where their pairs came from

- **Where:** `flydream/decode/map.py:242-251,280`.
- **Evidence:** the lag runs (`data/decode/2026-09-18_decode_sintel_lag_*`) have no `pairs.npz`, so they reused another run's file. Their `meta.json` names `model` from the CLI and nothing about the pairs source.
- **Failure scenario:** `--model malecns --pairs-from <a FlyVis run>` would label FlyVis activity as MaleCNS without any error. `--n-samples` is also ignored when a cache exists.
- **Fix:** store the pairs file path and hash, and the pairs' own model, in `meta.json`. Refuse a model mismatch.
- **Confidence:** confirmed by reading.

### A12 — low — test gap under a published result

- **Statement:** model zero (post 1, every generator state) depends on `export.malecns_filters`, pooled expansion, striding (`export.main`) and `orientation.choose`. None of them has a test.
- **Transplant tests:** these use stand-in networks, so nothing ties `total_n_syn` to FlyVis's actual `syn_count`. I checked that link by hand above, and it holds.
- **Fix:** A1 and A6 would have been caught by a fixture test asserting that per-target input from a sparse source is uniform and that pooled expansion conserves synapse mass.
- **Confidence:** confirmed by reading.

### A13 — low — a URL is written outside `sources.py`

- **Where:** `flydream/data/fetch_sintel.py:31`.
- **Statement:** this writes the Sintel URL, against the contract in `sources.py:1-4` and PROJECT_MAP that `sources.py` is "the only place a URL is written".
- **Confidence:** confirmed by grep.

## Gaps

- I did not re-render ISS-0015's Sintel-frame figure, so A1 connects the L3 dots to Am by connectivity and the grey steady state, not by the same picture. I did not examine Tm1's sign or where Tm9/T5a's stripes come from, beyond Tm9 receiving L3.
- I did not run the test files (the supervisor runs the suite). I checked none of `tests/fixtures/mini_connectome.json`'s numbers beyond the test's own claims.
- I did not check the MaleCNS column-ROI and hex conventions against neuPrint (no network). `optic_lobe`'s side inference for photoreceptors under `--edges-only --min-weight 1` may use a different photoreceptor set from `neurons_R.parquet`; not checked.
- I did not check whether `orientation.choose`'s chosen M is right beyond the published DS centroids. I did not check the flyvis-side meaning of `(du, dv)`; a global sign flip would be absorbed by M.
- `video_hex`/`video_corpus`: I relied on `test_video_hex_sampler_matches_flyvis_boxeye` for BoxEye equivalence and did not recheck the crop or resize geometry against flyvis on a real clip.
- `train/`: I checked only that the ReLU adapter equals the reference formula, and the benchmark logic. Training is paused, and no training result was reviewed.
- I did not measure how much A2, A5 or A6 moves any recorded number.
