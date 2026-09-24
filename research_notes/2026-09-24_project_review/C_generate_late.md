# Review C: `flydream/generate/` steps 18-29

Reviewer C, 2026-09-24. Read-only. No network, no Modal, nothing priced. Every
check below ran locally on the owner's machine in under two minutes. The
throwaway scripts are in the session scratchpad, not in the repository. Evidence
is quoted from the code and from saved artefacts under `data/` (gitignored).

Scope: `assigned18, blend18, check18, classcmp18, edges18, pca18, reach18,
selfinv18, trunc18, vae18, vaeval18, walk18, why18, cut19, fix19, hexcov19,
pca19, pcaval19, seed19, tprofile19, back21, floors22, inside22, latent22,
resid22, accept23, fixdraw23, hexflow23, seed23, static23, still23, geom29`, and
their tests. I also read the Modal functions `pca19`, `blockae22` and `resid22`
in `deploy/modal/generate_app.py`, because the code in `resid22.py` is only half
of the pipeline.

Known issues are not re-reported. ISS-0009 through ISS-0014 are listed under
Gaps with their current status.

---

## C1 - HIGH - the "local residual" and "lattice autoencoder" of step 22 never saw the lattice

- **Where:** `deploy/modal/generate_app.py:866-867` (blockae22) and `:934-935`
  (resid22) flatten `maps[:, :, ch]`. `flydream/generate/resid22.py:115` and
  `:128`, then `generate_app.py:884` and `:955`, reshape that flat vector as
  `(n_cols, c_in)`. `flydream/generate/latent22.py:176` and `:197` repeat the
  same reshape in the local evaluation.
- **Statement:** the maps file is laid out as (coefficient, type, column). The
  code reinterprets that memory as (column, channel) without a permute. Each
  "column" row the hex convolutions and the √3 pooling see is therefore 32
  consecutive columns of one channel. The networks were not local on the
  lattice, and the block AE did not decimate the lattice.
- **Evidence:**
  - The maps are built as `c = np.empty((len(x), int(dct_k), k, x.shape[3]))`
    (`generate_app.py:1409` and `:493`) and fed in as
    `mm = maps_all[ids][:, :, ch]; split[subset] = mm.reshape(len(mm), -1)`.
    The training step then does `e = E[idx].float().reshape(batch, n_cols, c_in)`.
  - Demo (build an index array in the (16, 2, 721) layout, then apply the same
    reshape):
    ```
    row 0  : columns [0 1 2 3 4 5] ... channels [0]
    row 22 : columns [704 705 706 707 708 709] ... channels [0 1]
    rows whose 32 entries come from one single lattice column: 0
    HexConv treats row 360 and row 391 as lattice neighbours; they hold columns [705 706 707].. and [255 256 257]..
    ```
  - `tests/test_resid22.py::test_training_reduces_the_residual_on_a_tiny_problem`
    builds `E` from a `(64, 721, 2)` array, which is the layout `train` expects.
    That is why the test cannot catch the layout the Modal code actually feeds.
  - `geom29.py:21-24` warns about exactly this trap for the static state.
- **Failure scenario:** step 22 recorded that "a learned local residual" (22.4:
  r 0.820 at 4,420 numbers) and "a block autoencoder" (22.5: r 0.656 at 1,928
  numbers) lost to plain PCA. The step-22 report §4, ROADMAP Done 22 and the
  published article part 2 §8 all carry this. The negative result belongs to a
  scrambled, non-local network, so it says nothing about locality. The local
  numbers are internally consistent, because the evaluation reshapes the same
  way, and that is why nothing looked off.
- **Fix:** convert with `mm.permute(0, 3, 1, 2).reshape(N, 721, 16*len(ch))`
  before training, and apply the inverse permute when writing back in
  `latent22.py:181,202`. Add a test that feeds a (coef, type, col) index array
  through the Modal preparation and asserts that each row holds one column.
  Re-reading step 22 needs a rerun of both arms (priced; not proposed here).
- **Confidence:** confirmed by reading and by the layout demo. No rerun was
  done, so the size of the effect is unknown.

## C2 - HIGH - "rotation 19° → 72°" compares two representations; 72° is what a pure Gaussian map gives

- **Where:** `flydream/generate/accept23.py:87-89` and
  `fixdraw23.py:95-97` compute the transport angle in the raw block space.
  Step 22.7 (`reports/2026-09-21_the_draw_distribution.md` §4) computed it in
  the whitened PCA-1536 latent.
- **Statement:** the transport angle is set mostly by the covariance spectrum
  of the space it is measured in, not by what the flow learned:
  - A flow that learned only the second moment has the map x = Σ^½ε. In
    whitened space (Σ = I) that map rotates by 0°. In the raw block space it
    rotates by arccos(tr Σ^½ / √(D·tr Σ)).
  - Flow 22 was measured in the first space (19°) and flow 23 in the second
    (72°), so the two numbers are not comparable.
- **Evidence:** from the saved block eigenvalues (`data/prior19/pca_ab2048.npz`,
  `eigvals_all`, N = 13,555):
  ```
  Gaussian (second-moment-only) map x=S^1/2 eps in raw block space: cos 0.3224 -> 71.2 deg
  accept23 transport {... 'cos': 0.3082, 'degrees': 72.05}
  whitened-space Gaussian map: x=eps -> 0 deg
  ```
  - Against its own Gaussian baseline, flow 22 rotated 19° beyond identity.
    Flow 23 rotated about 1° beyond the Gaussian map.
  - The angle is also built from mean norms rather than per sample
    (`cos = (rb²+ra²-(moved·ra)²)/(2·ra·rb)`), which is a second approximation.
- **Failure scenario:** ROADMAP Done 23 says "the transport from a 19-degree
  rotation to 72 where a random one is 90 - the near-identity that made
  interpolation render as a double exposure is gone". The step-23 report §3
  (item 1) says the same. The public `docs/articles/part2_en.md:374` prints
  "rotation | 19° | **72°** | 90° for a random one" and then states "The double
  exposure is gone." That conclusion rests only on this number. A linear map in
  raw space would also render a slerp midpoint as a blend of its endpoints.
- **Fix:**
  - Report the angle beside the Gaussian-map angle computed in the same
    representation, or drop it.
  - Test the double-exposure claim directly: render the slerp midpoint of two
    seeds and compare it with the 0.5 blend of the two endpoint renders.
- **Confidence:** confirmed by computation from the saved spectrum and
  `accept23.json`.

## C3 - HIGH - `tprofile19` takes the first 256 training rows; the recorded "59 percent high" is against an unrepresentative slice

- **Where:** `flydream/generate/tprofile19.py:80`,
  `zt = torch.as_tensor(np.asarray(z["z_train"][:a.n], np.float32))`.
- **Statement:** the latent rows are in corpus-index order, which is not
  random. The first 256 training rows have sd 0.767, where the whole training
  latent has 1.000 by construction and random 256-subsamples have
  1.011 ± 0.033. The "ideal" end point 0.767, and the overshoot measured
  against it, belong to that slice, not to the distribution the flow was
  trained on.
- **Evidence:**
  ```
  train (13555, 2048) sd 0.99997 ; first 256 sd 0.7671 ; radius first-256 32.7 vs all 41.06
  random 256 subsample sd 1.0113 +- 0.0329     (20 draws)
  index_train[:20] = [136 138 139 140 ...]      (ordered)
  data/prior19/tprofile19.json: sd_data 0.7671, from_draw['1.0'] 1.2195
  ```
  - The run record `reports/runs.jsonl` (`2026-09-21_prior19_tprofile`) says
    "256 held-out preimages … against the data's 0.767 … ends 59 percent high".
    The code uses training rows, not held-out ones.
  - Step-19 report §9 carries the same number.
- **Failure scenario:** the draw ends 22% above the training sd (1.220 against
  1.000), not 59%. At t = 0.5 the gap is 12% (0.795 against 0.707), not 26%.
  The "exposure bias" diagnosis that step 20 was built on is about half the
  recorded size. The qualitative shape survives: the draw departs at t ≈ 0.1.
- **Fix:** `rng.choice(len(z_train), n, replace=False)`, name the split
  honestly, and re-record the run under a new id.
- **Confidence:** confirmed by running.

## C4 - MEDIUM - the same head-slice pattern in `geom29` (test reference) and `fix19` (velocity profile)

- **Where:** `flydream/generate/geom29.py:67` (`idx = np.arange(n) if rng is
  None`) called at `:91` for `z_test`. `flydream/generate/fix19.py:171`
  (`zt = z_test[:m]`).
- **Statement:** the held-out column of the step-29 geometry table is the first
  256 test rows, not a sample of the test split. The report's reading
  "held-out differs from training on all rows … the same difference as
  ISS-0010" is largely an artefact of that slice.
- **Evidence:** 20 random 256-subsamples of `z_test` through the same
  `real_static` and `geometry`:

  | | first 256 (recorded) | random test 20×256 | train 20×256 (recorded) |
  |---|---|---|---|
  | kurtosis | 2.777 | 3.049 ± 0.088 | 3.161 ± 0.106 |
  | radius | 38.72 | 37.62 ± 0.18 | 37.61 ± 0.22 |
  | common share | 0.667 | 0.693 ± 0.008 | 0.707 ± 0.010 |
  | pair r p99 | 0.517 | 0.514 ± 0.009 | 0.575 ± 0.019 |
  | pair r max | 0.998 | 0.893 ± 0.064 | 0.984 ± 0.012 |

  - Radius does not differ at all, and kurtosis and common share differ about
    1σ, not 3σ and 6σ. Only pair p99 is really lower on the test split.
- **Failure scenario:** `reports/2026-09-21_step29_a_scene_as_one_picture.md`
  §6 (lines 216-220) states a split difference that is mostly not there. The
  fix19 "preimage" velocity trajectories (step-20 report §3, the 0.07× and
  2.48× ratios) start from 2-3 classes rather than from the held-out split.
- **Fix:** always sample rows at random, and state the split beside every
  reference number.
- **Confidence:** confirmed by running (geom29). The fix19 part is confirmed
  by reading only; its numbers were not re-measured.

## C5 - MEDIUM - every cut/block/PCA arm passes through DCT-16, the "whole state" row does not

- **Where:**
  - `flydream/generate/inside22.py:193,201` projects every arm onto 16 DCT
    functions, but the reference row is `groups = {"полное состояние": real}`
    (`:198`, 40 raw frames).
  - `cut19.py`, `back21.py`, `seed19.py` and `pcaval19.py` do the same through
    `R.to_model_space`, with `real` as the ceiling.
- **Statement:** DCT-16 truncation alone costs about 0.04 of r_to_raw, and the
  tables attribute that cost to the cut being tested.
- **Evidence:** on the same six clips `[2665, 6008, 5069, 2713, 51, 9449]`,
  same 13B seed:

  | state rendered by 13B | r_to_raw | source |
  |---|---|---|
  | all 8 types, 40 frames | 0.952 | `inside22.json` and my run |
  | all 8 types, DCT-16 | 0.914 | my run (scratch `dct16.py`) |
  | T4a+T4b, 40 frames | 0.916 | `static23.json`, "настоящий клип" |
  | T4a+T4b, DCT-16 | 0.884 | `seed23.json` |

  - The project's own run `2026-09-20_prior18_dct_ceiling` already recorded
    0.912 for K = 16.
- **Failure scenario:**
  - ROADMAP "The object … r 0.884 against 0.952 for the whole state (22)".
  - Step-22 §1 table and §8 "Объект уменьшен … ценой 0,952 → 0,884".
  - Step-22 §7's self-correction ("правильное сравнение — 0,952 против
    0,884").
  - Step 22.6's ceiling 0.916 beside PCA rows at 0.809.

  In the same representation, dropping six types costs 0.914 → 0.884 (or
  0.952 → 0.916 without DCT). That is about half the recorded cost. It is also
  why step 23's ceiling (0.884) and step 22.6's ceiling (0.916) differ for the
  same block.
- **Fix:** put a "full state, DCT-16" row in every such table, or DCT the
  reference row too, and say which representation each ceiling is in.
- **Confidence:** confirmed by running.

## C6 - MEDIUM - `sdproj` normalises by the batch sd: rendered draws depend on their batch and are not the draws whose geometry is reported

- **Where:** `flydream/generate/fix19.py:98-101`
  (`have = float(x.std()); x = x * (want / have)`, over the whole batch).
  `seed19.py:210-211` renders the six draws as one batch of 6, while
  `:212-214` measures the arm's geometry on batches of 256.
- **Statement:** each draw's output depends on which other draws share its
  batch. The six rendered draws are rescaled by a six-sample statistic. The
  geometry reported for the arm (sd 1.000, radius 42.6) is not theirs, and the
  sd match is imposed by construction.
- **Evidence:** same six ε (seed 9000), flow `flow_pca2048_b256.pt`,
  `sdproj,steps=100`:
  ```
  radius when integrated as a batch of 6  : [64.5 41.9 31.4 15.4 49.2 52.3]
  radius when integrated in a batch of 256: [70.6 54.9 42.4 22.6 60.  62.1]
  ratio [0.913 0.763 0.742 0.68  0.821 0.842]
  ```
- **Failure scenario:** step-20 report §4 ("розыгрыш + sdproj … ворота
  0,0983/0.0933, контраст 0,153 при 0,152 у сырого видео") and §1 ("Геометрию
  розыгрыша удалось починить полностью") read a batch-6 statistic as a property
  of the sampler. The contrast match is partly forced by the six-sample
  normalisation.
- **Fix:** estimate the per-step scale once on a large batch and apply it as a
  fixed schedule per sample. Render draws taken from the batch whose geometry
  is reported.
- **Confidence:** confirmed by running.

## C7 - MEDIUM - the "sharpness 0-100" scale (flat fraction) is unbounded and contrast-blind

- **Where:** `flydream/generate/edges18.py:59,73`. Differences are divided by
  their own sd, and `frac_flat = (|d| < 0.25·s).mean()`.
- **Statement:** a near-empty grey field scores far above real video, and a
  clip at 1% contrast scores the same as at full contrast. The calibrated
  scale (noise 0, raw video 100) is therefore neither bounded nor monotone in
  "scene-ness". It can be bought exactly the way step 26 found kurtosis can.
- **Evidence:** `describe` from `edges18`, scale `(ff − 0.197)/(0.498 − 0.197)·100`:
  ```
  raw clip                       frac_flat 0.405  scale 69/100   sd 0.2581
  grey + one dark 6-col blob     frac_flat 0.987  scale 263/100  sd 0.0273
  grey + noise 1e-3              frac_flat 0.198  scale 0/100    sd 0.0010
  raw clip * 0.01 + 0.5          frac_flat 0.405  scale 69/100   sd 0.0026
  ```
- **Failure scenario:** "sharpness of a draw 11 → 22, ceiling 66" (ROADMAP 23,
  article part 2 §10) and step 29's "flat fraction 27.1 against 27.1". A
  generator collapsing toward a flat field with a few edges would read as
  sharper. The reports do say the judge "does not separate a scene from a
  smooth field", but they do not say it rewards emptiness.
- **Fix:** always print it beside sd/contrast and beside a blank-field floor,
  and cap or replace it with a bounded, contrast-aware statistic.
- **Confidence:** confirmed by running.

## C8 - MEDIUM - nearest-neighbour novelty uses the whole corpus, including the scored clip, and a floor from another code path

- **Where:**
  - `seed19.py:132,281`, `seed23.py:88,145`, `pcaval19.py:72,120`,
    `cut19.py:177,231`, `inside22.py:183,234`, `still23.py:72,118` all use
    `bank_all = videos[:, :frames]`, all 15,514 clips.
  - `floors22.py:78-79` excludes the own row only for the real-clip arm.
- **Statement:** AGENTS requires the distance to the nearest *training* video,
  measured in the same code path as the real clip. Here the bank includes val
  and test clips and the clip itself, and the "real held-out clip" floor is
  measured on raw video rather than on a 13B render.
- **Evidence:**
  - In `seed23.json` the fresh draw's nearest clips are
    `[(7011,'train'),(8693,'train'),(11501,'train'),(10307,'test'),(4665,'val'),(15009,'train')]`;
    `seed19_b256.json` also has one test and one val.
  - The "real state" group in seed23 reads 0.889 with `nearest_idx` equal to
    its own clip ids, so that value is a self-match.
  - Same-path floor, computed by me: 13B render of the real held-out block
    against the bank without its own clip gives **0.564**, against the
    recorded raw-video floor 0.521.
  - In `floors22.json` the time-shuffled clip's "nearest 0.662" matches its
    own source clip in 4 of 6 cases (`[14772, 6008, 5069, 7428, 51, 9449]`
    against `clip_idx [2665, 6008, 5069, 2713, 51, 9449]`).
- **Failure scenario:** "not a copy: nearest 0.485 where a real held-out clip
  reads 0.521" (ROADMAP 23, article part 2) compares a render with a raw clip.
  The honest reference is 0.564, which strengthens "not a copy". The
  per-group nearest columns for real and seed groups are self-matches, and the
  0.662 shuffle floor is not a floor.
- **Fix:**
  - Use a training-only bank.
  - Measure the floor as a render of a real held-out state with its own row
    excluded.
  - Exclude the source clip for derived arms.
- **Confidence:** confirmed by running.

## C9 - MEDIUM - the hex patch tokenisation is not equivariant

- **Where:** `flydream/generate/hexflow23.py:105` (members ordered
  `[centre] + sorted(column index)`) with a single shared `self.inp` (`:213`)
  and `self.out` (`:224`) over the `P·T·K` slot features.
- **Statement:**
  - Kuhn matching gives each centre two arbitrary neighbours, and slots are
    ordered by column index, not by lattice direction.
  - Slot 1 and slot 2 therefore mean different geometric neighbours in
    different patches, while the shared linear layers read them as the same
    feature.
  - The relative attention bias is equivariant, but the tokens are not.
- **Evidence:** offsets (axial, member minus centre) per patch:
  ```
  distinct slot->offset patterns: 17
  150 ((0, -1), (0, 1))   20 ((0, 1), (1, 0))   12 ((0, 1), (1, -1))   12 ((0, -1), (1, -1)) ...
  ```
  91 of 241 patches (38%) differ from the majority pattern.
- **Failure scenario:** the step-23 report §1, ROADMAP 23 and article part 2
  §10 present the design as "equivariance: one rule everywhere" and cite Kamb
  & Ganguli for it. The measured gains of step 23 and 29 were obtained with a
  position-dependent tokeniser, so any conclusion about equivariance is
  untested.
- **Fix:** use a translation-consistent tiling. On the √3 sublattice every
  interior centre can take the same two offsets, such as (0,−1),(0,+1); edge
  patches get masked slots. Alternatively, order slots by direction and add a
  test that asserts one offset pattern for interior patches.
- **Confidence:** confirmed by running. The effect on the recorded numbers is
  unknown.

## C10 - MEDIUM - the locality curve (0.876 at one step) comes from a rank-2048 truncation that has not converged

- **Where:** `flydream/generate/hexcov19.py:105`,
  `S = (Bb * lam) @ Bb.T` from the top-2048 basis.
- **Statement:** truncation drops the least spatially coherent variance, which
  inflates correlations. The value keeps falling as k grows.
- **Evidence:** the same weighting as `hexcov19`, varying k only:
  ```
  k= 256  corr d=1 0.943  d=8 0.420
  k= 512  corr d=1 0.926  d=8 0.362
  k=1024  corr d=1 0.902  d=8 0.326
  k=2048  corr d=1 0.876  d=8 0.302
  ```
- **Failure scenario:** step-20 report §6, ROADMAP 20-21 and article part 2
  §10 cite "0.876 at one step against 0.267 shuffled" as a measured property
  of the data. It is an upper bound of unknown size, and the shuffled control
  (0.267, which is the mean over all pairs) is inflated the same way. The
  qualitative locality claim probably survives. The docstring names the
  approximation but not the direction of its bias.
- **Fix:** compute the curve from real states (a few hundred corpus states, or
  the states on the volume), or report the k-trend beside the number.
- **Confidence:** confirmed by running (the trend). The full-rank value was
  not measured.

## C11 - MEDIUM - `seed23` has no control in its own run

- **Where:** `flydream/generate/seed23.py:123-125`. The groups are real block,
  clip seed and fresh draw only.
- **Statement:** AGENTS requires a video from a sampled state to be shown
  beside a state known to be unreachable (shuffled state / noise in the
  types), from the same run. seed19 has `--controls` (N(0, I) without the
  flow, radius-from-data). seed23, which produced the step-23 video numbers,
  has neither.
- **Failure scenario:** the step-23 report §4 judges the fresh draw
  (sharpness 22, nearest 0.485) against floors taken from run
  `2026-09-21_prior22_floors`, which was measured on raw videos in a different
  code path (see C8). There is no same-run number showing what an unreachable
  block renders to.
- **Fix:** add "N(0, I) block, no flow" and "real block with columns shuffled"
  groups, rendered with the same z.
- **Confidence:** confirmed by reading.

## C12 - LOW - config keys read nowhere; step 18-29 settings are buried in code

- **Where:** `seed23.py:173`, `still23.py:140` and `static23.py:155` call
  `settings().get("gen13b", {})`, but `config.toml` has no
  `[generate.gen13b]` table. `invert.settings()` returns `[generate]`, so `g`
  is `{}` and `frames/margin/dt/t_pre` silently fall back to code defaults.
  They coincide with the config today.
- **Statement:** no settings for steps 18-29 are in `config.toml`, although
  AGENTS names sampler steps explicitly. Missing examples:
  - flow Euler steps: 20 in seed19, 100 in seed23/accept23/fixdraw23. The
    report showed this changes the radius 63.1 → 53.5.
  - 13B steps = 20.
  - the number of draws (256), subsample repeats (40 and 20), k = 1536.
  - the seeds 1000/4242/7700/9000.
  - the hexflow `lattice/radius/n_global`.
  - the sdproj/vscale/shift ladders and the fixdraw radii.
- **Fix:** add a `[generate.prior]` table with reasons, and read `[generate]`
  directly in the three scripts.
- **Confidence:** confirmed by reading.

## C13 - LOW - ISS-0011 has spread: more same-name helpers with different semantics

- **Where:**
  - `geom29.py:47` `geometry` (uncentred radius, common share, pair r),
    against `pca19.py:135` and `accept23.py:45` `geometry`.
  - `seed19.py:56` `geometry_of` (wraps `pca19.geometry`) against `seed23.py:50`
    `geometry_of` (own formula, no sd-axis fields).
  - `accept23.py:39` `kurtosis_of` (per-coordinate array) against
    `geom29.py:41` `kurtosis_of` (scalar mean).
- **Statement:** these are the same trap ISS-0011 describes. No recorded
  number differs today.
- **Fix:** one metrics module.
- **Confidence:** confirmed by reading.

## C14 - LOW - `cut19` "+complete" replaces the given half too

- **Where:** `flydream/generate/cut19.py:129`, `return mu + c @ B.T, resid`.
  The whole state is overwritten by its rank-2048 LS fit.
- **Statement:** "given T4 + completed T5" is actually "PCA-2048 projection
  inferred from T4". The given half carries a 31.3% residual (step-20 §5), so
  the arm does not measure "specify half, derive the rest".
- **Failure scenario:** step-20 §1 "Часть состояния можно не задавать, если её
  достраивать" and back21's `types=T4+complete` row (r 0.976) describe a
  different operation. The conclusion probably holds or is conservative.
- **Fix:** keep `x[:, mask]` exact and fill only `~mask`.
- **Confidence:** confirmed by reading.

## C15 - LOW - back21 per-type r has no floor and pools clips, time and cells

- **Where:** `flydream/generate/back21.py:58-60` computes r over the flattened
  (clips × time × cells) of one type. The headline 0.961 is also a mean over
  8 types, 4 of which were given, not restored.
- **Evidence:** from `back21.npz` (T5c, real states), a *different* clip's
  state against the real state gives pooled r = 0.447 (pairwise 0.17-0.82).
  The restored T5 types alone read 0.944-0.959.
- **Failure scenario:** "a zeroed half restored at r 0.961" reads as if the
  floor were 0. The restoration is real (0.95 against 0.45), but the number
  needs its floor and should be the restored types only.
- **Fix:** report the restored-types-only r beside the cross-clip floor.
- **Confidence:** confirmed by running.

---

## Gaps

- **Not rerun:**
  - C1: resid22 and blockae22 with the corrected layout (priced GPU runs).
  - C9: the hex flow with an equivariant tokeniser (priced).
  - C10: the full-rank locality curve (needs states on the volume).
  - C3: the corrected step-20 sweep.

  The direction of each effect is shown; its size on the recorded conclusions
  is not.
- **Only skimmed, not traced line by line:** `assigned18`, `classcmp18`,
  `reach18`, `walk18`, `why18`, `vae18`. In these and in the other step-18
  modules I checked the split selection, the nearest bank and the 13B batching.
  No new defect was found beyond ISS-0009.
- **Known issues confirmed still present, not re-reported:**
  - ISS-0009: 13B noise batch 8 in `assigned18:111`, `blend18:104`,
    `pca18:117`, `reach18:121`, `trunc18:89`, `vaeval18:119`, `walk18:110`.
  - ISS-0010: `seed19.py:111,176,183`, where `latent_data` and `draw_scale`
    are still taken from `z_test`.
  - ISS-0011: duplicate `nearest` in `blend18.py:44` and `vaeval18.py:47`.
- **Checked and found sound:**
  - The PCA fit sees only the training split (`generate_app.py:754-776`).
  - The DCT coefficient z-scoring uses training statistics only
    (`generate_app.py:1417-1422`).
  - The hex flow and `geom29` use the correct (coef, type, col) layout.
- **An onset-transient concern about static23 was tested and dropped.** A
  still frame shown from grey keeps its T4 deviation from the grey response
  unchanged over 3 s (0.40 at frames 0-5 and at 110-150), so the static
  picture is not an onset artefact.
- **Nothing was checked on Modal.** Statements about the maps layout rest on
  the builder code (`generate_app.py:493,1409`) and on the local
  `pca_ab2048.npz` meta shape `[16, 2, 721]`.
- **Not done:** the LS "renderer" for step 29 lives in `tools/` and was not
  reviewed. I did not run any test file; the five test files read are offline
  (no data, no network).
