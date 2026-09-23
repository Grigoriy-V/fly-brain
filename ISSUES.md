# Issues

Observed defects, one entry per defect. This file is not a plan and
authorizes nothing; `ROADMAP.md` alone orders work. Evidence lives in
`reports/`, the entry links to it.

## Rules

- A defect is observed behaviour, not a suspicion: a wrong number, a broken
  export, a model that fails a validation it passed, a mapping row that
  contradicts its source. A missing capability belongs in the roadmap; a
  false claim of one belongs here.
- New entry: next free number, never reused; put it at the top of **Open**
  and a row at the top of the catalog.
- Status names the defect, not the effort: `open`, `mitigated` (harm reduced,
  defect still there), `fixed` (verified by a rerun), `won't fix`. A closed
  entry stays, shortened, under **Closed**; a defect that comes back moves up
  to **Open** with a "seen again" line and keeps its full body.
- `Cause` stays "unknown" until proven. One defect per entry. `Costs` says
  what it does to a result or a reader; there is no severity word.
- A defect in a third-party input (a dataset table, a community mapping, a
  reference implementation) is recorded here with its source named; it is
  not fixed upstream by this project.
- Fields: Status, Seen, Costs, Reproduce, Cause, Evidence, Related.

## Catalog

| Id | Status | Defect | Related |
|---|---|---|---|
| ISS-0015 | open | model zero's OFF pathway is spatially broken: on a Sintel clip L3's activity is a regular lattice of isolated strongly hyperpolarised cells over a near-flat field where FlyVis's L3 is a smooth image, and Tm9 and T5a downstream carry periodic diagonal stripes (neighbour roughness 0.58 and 0.46 against FlyVis's 0.38 and 0.18); Tm1 has the wrong sign (correlation with the frame +0.93 against FlyVis's -0.78) | roadmap 2, ISS-0002, ISS-0005 |
| ISS-0014 | open | the round trips of 13A and 13B are normalised by different variances - 13A divides by the variance over all eight target states together (`learned.py:249`), 13B by clip A's own variance (`roundtrip13.py:130`, `generate_app.py:1831`) - so 13A's CNN 0.029 and 13B's 0.031 on the same eight clips are not on one scale, and "SiT equals the 13A CNN" (13B report, 13B design, article part 1) is unsupported; an approximate rescale puts the CNN near 0.033 | roadmap 13A, 13B |
| ISS-0013 | open | 13B's "shuffled state" control scores the video made from the shuffled state against clip A's TRUE state (`generate_app.py:1860`, target taken from `j[1]` at 1886-1888; `states["control_shuffled"]` is never used as a target), so its 0.957 measures how far a foreign video is from clip A, not whether the video obeys its own condition; step 14's control of the same name scores against the shuffled state and reads 19.1, twenty times larger | roadmap 13B, 14 |
| ISS-0012 | open | today's still-picture grids normalise every cell to mean 0.5 and sd 0.19 before display, which equalises exactly the contrast the captions beside them report - the original of clip 6008 has sd 0.226 and was shown at 0.19, and draws at sd 0.122 were shown at the same 0.19 as the reference at 0.182, so the draws looked better and the originals worse than they are | roadmap 29 |
| ISS-0011 | open | three functions named `nearest` live in `flydream/generate/`: `vaeval18.py:47` and `blend18.py:44` are line-for-line the same Pearson correlation over a clip bank, while `baseline17.py:67` is a different algorithm under the same name - cosine similarity with no mean-centring, a batched signature and a `skip` mask - so importing the wrong one silently changes the scale a novelty claim is read on | roadmap 17-22 |
| ISS-0010 | open | a fresh draw's geometry is scored against the held-out split, which the flow was never trained to match: at k = 1536 the held-out latent has per-axis sd 0.829 [0.206 .. 1.075] and radius 29.67 against exactly 1.000 [0.999 .. 1.000] and 35.21 on training, so the draw's radius overshoot was reported as 44 % when it is 20 %, and `--draw-scale -1` shrinks the rendered state by 16 % toward the wrong reference | roadmap 19-22 |
| ISS-0009 | open, narrowed | 13B's sampling noise is drawn per batch, so "one z per clip in every cell" holds only when the batch boundaries fall on the group boundaries; at batch 8 and groups of 6 clips each cell of a figure got a different z, which moves a six-clip gate by about a tenth of its value (the same arm read 0.3818 in cut19 and 0.4162 in back21) | roadmap 19-21 |
| ISS-0008 | fixed in code, rerun pending | the class embedding was never trained: nn.Embedding starts at N(0,1) (row norm 11.3 against 29.8 for the timestep embedding it is added to, fifty times DiT's 0.02) and AdamW decayed the table like any weight, so after 20,000 steps the two independently trained tables agree on class geometry at r = +0.001 - both still their initial noise, which voids 18.4e and 18.12 as tests of conditioning | roadmap 18, ISS-0007 |
| ISS-0007 | open | the corpus split holds out whole classes, so 10 of 110 labels have no training clips and their embedding rows are never trained - yet the gate draws sampling labels uniformly over all 110, conditioning about 9.1 % of every conditional run on a row at its initialisation | roadmap 18 |
| ISS-0006 | open | a decoder window of even-spaced lags ([0 2 4], [0 2 4 6 8]) aliases Sintel's 24→50 Hz frame hold: taps two steps apart sit in one phase of the two-step hold, the fit averages two regimes and the reconstruction alternates frame by frame (L3 per-frame corr 0.94/0.92/0.94/0.89/0.93/0.86…); consecutive lags [0 1 2 3 4] remove it; the sweep at those two windows is being redone | roadmap 4 |
| ISS-0005 | mitigated (b fixed, a third-party) | the right-lobe export carries about thirty times less total input than FlyVis on lamina pairs through Am (a: MaleCNS has 49 Lai fragments for ~750 Am, a reconstruction gap) and on the feedback pairs Tm2→L2 and Mi4→Tm2 (b: the min_weight cut, fixed by the weak-pair exception; v9 DSI 0.152 against v7's 0.109, all members stable) | roadmap 1, ISS-0003 |
| ISS-0004 | open | the step-4 stage curve (R → L → Mi/Tm → T4/T5) is a decoder artefact: at a 60 ms window instead of lag 0, T5a's control-corrected score goes 0.057 → 0.625 and the order inverts; stage explains R² 0.317 of the map; read from the best of 50 members on one split | roadmap 4 |
| ISS-0003 | mitigated | `transplant` copied FlyVis's `syn_strength` raw, though it is a gain divided by that connectome's own synapse count; on MaleCNS the copy underweighted T5's main drive and overweighted its inhibition, and step 2's DSI 0.020 was at least partly that (member 000 → 0.161 corrected) | roadmap 2, 3; DECISIONS 2026-09-18 |
| ISS-0002 | open, third-party input | the right CT1 (bodyId 10157, "Roughly traced") has 56,517 outgoing rows in the weights table but only 8 to typed right-lobe neurons, so every CT1 pathway FlyVis carries (CT1 → T4/T5, 14–40 synapses per column) is absent from the export | roadmap 1, 2 |
| ISS-0001 | open, third-party input | MaleCNS's column tag on Tm4 disagrees with the column holding most of its synapses in 47% of right-lobe Tm4 cells; every other tagged type agrees ≥99.3% | roadmap 1 |

---

## Open

### ISS-0015 — model zero's OFF pathway is spatially broken (L3 dots, Tm9/T5a stripes, Tm1 sign)

- **Status:** open. Seen 2026-09-24 while drawing each layer's activity for a
  public figure; never looked at as a picture before.
- **Seen:** on Sintel clip 3, frame 20, member 000, model zero against the
  FlyVis network with the same parameters on its own connectome:
  - **L3** is a regular sublattice of isolated, strongly hyperpolarised cells
    on a near-flat field; FlyVis's L3 is a smooth OFF image of the scene;
  - **Tm9** (L3's main target) and **T5a** (downstream) carry periodic
    diagonal stripes; neighbour roughness 0.58 and 0.46 against FlyVis's
    0.38 and 0.18 (mean |cell - its six neighbours' mean| / spatial sd);
  - **Tm1** has the opposite sign: correlation with the frame +0.93 against
    FlyVis's -0.78, i.e. it behaves as an ON cell;
  - the ON side is clean: R1, L1, Mi1 match FlyVis in roughness and sign.
- **Costs:** every state the generator line uses comes from this model, so
  the OFF half of T4/T5 (T5) is built on a periodic artefact. It is a
  candidate cause of the weak T5 selectivity (DSI 0.01-0.03, article part 1
  section 2), which was put down to the missing CT1 (ISS-0002) and weak
  Tm9 -> T5. Flash polarity (0.922) did not catch it: a full-field flash
  cannot see a spatial pattern.
- **Reproduce:** `python tools/cmp_layers_flyvis.py` (local, ~40 s).
- **Cause:** unknown. Candidates, none checked: the export's per-type
  sampling of L3 or its R1-R6 inputs (a stride would draw exactly a regular
  sublattice), the column offsets of L3 -> Tm9, the transplant's capped gain
  on lamina pairs (ISS-0005 a), Tm1's input signs.
- **Evidence:** `reports/figures/2026-09-24_malecns_layers_vs_flyvis.png`
  (top FlyVis, bottom model zero, L1 L3 Mi1 Tm1 Tm9 T4a T5a).
- **Related:** roadmap 2; ISS-0002, ISS-0005.

### ISS-0014 — 13A and 13B round trips are normalised differently and compared as equal

- **Status:** open. Seen 2026-09-23 in the number check of the master article,
  part 1.
- **Seen:** 13A's round trip divides the squared error by the variance of all
  eight target states pooled (`flydream/generate/learned.py:249`,
  `var = tg[:, :, sel].var()`); 13B's divides by the variance of the one clip
  being scored (`flydream/generate/roundtrip13.py:130`,
  `deploy/modal/generate_app.py:1831`). Per-type variance ratios between the
  two range 0.60 (T4a) to 1.64 (T5d).
- **Costs:** "13B's round trip equals the 13A CNN's" (13B report, 13B design,
  article part 1 before its revision) compares two scales. An approximate
  local rescale onto 13B's scale gives CNN ≈ 0.033, linear ≈ 0.039, inversion
  ≈ 0.009 - SiT slightly better than the CNN, not equal. The rescale held the
  last frame for the 5-frame margin because the pairs13 shards are not local,
  so it is approximate.
- **Reproduce:** score the eight 13A test clips' CNN videos with
  `roundtrip13.py` and compare with `data/train13/summary.json`.
- **Cause:** known - two functions written for two steps, never unified.
- **Evidence:** the verification pass of 2026-09-23 (article part 1, § 8).
- **Related:** ISS-0011 (the same class of defect: one name, two metrics).

### ISS-0013 — 13B's shuffled-state control is scored against the wrong state

- **Status:** open. Seen 2026-09-23 in the number check of the master article,
  part 1.
- **Seen:** the control job is added as
  `add("clip_A", ..., maps=maps_of(st_sh), tag="control_shuffled")`
  (`deploy/modal/generate_app.py:1860`); the round-trip target is looked up by
  the job's first field, `state_of[k] = j[1]` = "clip_A" (1886-1888), so the
  video made FROM the shuffled state is compared with clip A's TRUE state.
  `states["control_shuffled"] = st_sh` (1861) is never used as a target.
- **Costs:** the recorded 0.957 (`data/gen13b/samples.json`,
  `control_shuffled__full__s1`) measures the distance of a foreign video from
  clip A - a zero state against clip A reads 0.854 by the same route - and not
  whether the generator obeyed the shuffled condition. Step 14 builds a
  control of the same name correctly and gets 19.1, so the project carries
  two numbers under one name, twenty times apart. The article printed 0.957
  beside r 0.966 as if it were a correlation; it now uses the same-z output
  correlation (0.03) instead. The control was also computed on clip A alone,
  four seeds, not on the eight held-out clips it stood beside.
- **Reproduce:** read the job list in `generate_app.py` around 1855-1890.
- **Cause:** known - the target is keyed by clip name, and the control reused
  clip A's name.
- **Evidence:** `reports/2026-09-20_step13b_generative_decoder.md:83`,
  `reports/2026-09-20_step14_controllable_generator.md:50`.
- **Related:** roadmap 13B, 14.

### ISS-0012 — the picture grids equalise the contrast their own captions compare

- **Status:** open. Seen 2026-09-21, found by the human: "куда исчез контраст?
  это же оригинал". Affects every still-picture grid made today
  (`fig_pic23.py`, `fig_pic23k16.py` and the inline grids for 29); the video
  figures are unaffected, they draw raw values on a fixed 0..1 scale.
- **Seen:** each cell is displayed as `(v - v.mean()) / v.std() * 0.19 + 0.5`,
  so every image is forced to mean 0.5 and sd 0.19 whatever its real contrast.
- **Costs:** it hides the difference the captions are about. Clip 6008's frame
  has sd 0.226 and its 40-frame mean 0.211 - averaging costs only 1.07x - yet
  both were shown at 0.19 with the deep blacks pulled to grey. Worse, in the
  four-row grid of 29 the reference reads sd 0.182 and the draws 0.122, and
  both were displayed at 0.19: the draws were flattered and the reference was
  flattened, in the same figure whose caption compared them. Recorded numbers
  in `reports/runs.jsonl` are unaffected - they were computed before display.
- **Reproduce:** `reports/figures/2026-09-21_malecns_contrast6008.png` shows
  the same frame as it is and after the normalisation, side by side.
- **Cause:** known - the normalisation was added so cells of different
  absolute scale could be compared, which is the wrong trade when the caption
  is about scale.
- **Evidence:** `reports/figures/2026-09-21_malecns_static29_true.png` is the
  same grid drawn on a fixed 0..1 scale.
- **Related:** roadmap 29. The fix is to draw on a fixed scale and to say the
  per-row sd in the caption, as the video figures already do.

### ISS-0011 — three different functions are called `nearest`, and one of them is a different metric

- **Status:** open. Seen 2026-09-21 in the metric audit of 22.8, by a
  delegated read of every metric path in the seven measurement scripts.
- **Seen:** `flydream/generate/vaeval18.py:47-58` and
  `flydream/generate/blend18.py:44-56` hold the same body (Pearson r: subtract
  the mean, divide by the sd, chunked matmul over the bank, return the best
  index and value); only a docstring differs. Both were added 2026-09-20, the
  second a copy of the first rather than an import.
  `flydream/generate/baseline17.py:67-82` is a **different function with the
  same name**: it takes a batch of queries, L2-normalises without subtracting
  the mean (cosine similarity, not Pearson), accepts a `skip` leave-one-out
  mask and returns three arrays instead of a pair.
- **Costs:** every live measurement imports the `vaeval18` copy
  (`seed19.py:53`, `cut19.py:56`, `pcaval19.py:42`, `inside22.py:50`,
  `floors22.py`), so no recorded number is wrong today. The cost is the trap:
  `nearest_r` is the only number behind every "not a copy" claim in this
  project, and one import line decides whether it is a correlation or a
  cosine. The two disagree on any data with a non-zero mean, which every hex
  raster has. A leave-one-out floor (22.8) needs exactly the `skip` argument
  that only the wrong-metric copy provides, and it had to be worked around by
  zeroing a bank row instead.
- **Seen again 2026-09-21**, same trap, different name: `pca19.geometry()` and
  `accept23.geometry()` are two different functions under one name in two
  modules, computing overlapping but not identical statistics. Found by the
  delegated audit of the 26 sweep. No number is wrong today; the cost is the
  same as above.
- **Reproduce:** `grep -rn "def nearest\|def geometry" --include=*.py flydream/`.
- **Cause:** known — duplication at the time of writing, then a same-name
  function with different semantics in an older module.
- **Evidence:** `reports/2026-09-21_step22_a_smaller_object.md` § 7;
  `reports/2026-09-21_the_draw_distribution.md` § 8.
- **Related:** roadmap 17-22. The fix is one shared module with one Pearson
  `nearest` that takes `skip`, and a rename of the cosine one to what it is.

### ISS-0010 — a draw is scored against the held-out split, not the one the flow was trained on

- **Status:** open. In `flydream/generate/seed19.py`; `pcaval19.py` and every
  figure, caption and message built on its `draw_many` block carry the number.
  Seen 2026-09-21 while auditing the metrics before more training, on the human's
  question whether the metrics are right.
- **Seen:** `latent_data` is `P.geometry(z_test)` (line 111) over `z_test`, the
  ten held-out UCF101 classes (line 107). `radius_needed` (line 176), the
  `--draw-scale -1` correction (line 183) and the log lines that call it "у
  данных" (189, 192-193) all read from it, while the flow is trained on
  `z_train`. The same function already loads `z_train` for `sd_train` in the
  `fixes` branch (line 204), so both references sit in one function and disagree.
- **Costs:** the reference is a different distribution, not another sample of the
  same one. At k = 1536 the training latent is whitened exactly (per-axis sd
  1.000, range 0.999-1.000, radius 35.21) and the held-out split is not (0.829,
  range 0.206-1.075, radius 29.67), because ten unseen classes sit off the basis
  fitted on training. The draw's radius overshoot was therefore reported as 44 %
  (42.59 against 29.67) when against the right reference it is 20 % (42.59
  against 35.50 +- 1.09 on matched 256-point subsamples). `--draw-scale -1` does
  not only misreport: it rescales the state that goes to the renderer, by 0.836
  instead of 1.000, a 16 % shrink.
- **Reproduce:** load `data/prior19/pca_ab2048_latent.npz`, take the first 1536
  columns, and compare `z_train`, `z_val` and `z_test` on per-axis sd, radius and
  kurtosis.
- **Cause:** known — `latent_data` was written for part A of the seed test, the
  preimage geometry, where the held-out split is the right reference because
  those are the latents being inverted; it was then reused as the target for the
  forward draw, where it is not.
- **Evidence:** `reports/2026-09-21_the_draw_distribution.md` §§ 2-3; run
  `2026-09-21_prior22_flow_ab1536`.
- **Related:** roadmap 19-22; ISS-0007 (the same class-level split, other harm).
  The fix is to score the draw against `z_train` and to say which split every
  reference number comes from.

### ISS-0009 — 13B's noise is batched, so a figure's cells are not drawn with the same z

- **Status:** open, narrowed. Fixed 2026-09-21 in `back21.py`, `inside22.py`,
  `latent22.py`, and now `seed19.py`, `cut19.py`, `pcaval19.py` (batch =
  `n_clips`, the group size). Still present in nine scripts of the 17-18 era,
  whose steps are closed and whose numbers are recorded as they were measured:
  `assigned18.py`, `blend18.py`, `noise17.py`, `pca18.py`, `reach18.py`,
  `samples17.py`, `trunc18.py`, `vaeval18.py`, `walk18.py`. They are not fixed
  blind: each groups its jobs differently and the batch has to be checked per
  script. Seen 2026-09-21 while re-measuring the cut arms.
- **Seen:** the render loop re-seeds one generator per batch —
  `for i in range(0, len(names), 8): gg = Generator().manual_seed(1000 + seed)`
  — and `gen13b.sample` draws `torch.randn(B, T, n, generator=gg)`, so element
  *j* of a batch gets the *j*-th noise slab. With groups of 6 clips and a batch
  of 8, a group straddles batch boundaries and its clips take slabs 0-5 in one
  cell and 2-7 in the next. The captions of 19.1, 19.3, 19.5, 20 and 21a all
  say "один z у 13B во всех клетках"; that is true only within a batch.
- **Costs:** part of the difference between two cells of a figure is a
  different sampling noise rather than a different state. Measured size: the
  same arm (`types=T4`) with the same code and seed reads gate 0.3818 in
  `cut19` (batch 8) and 0.4162 in `back21` (batch 6), and the no-cut control
  0.0114 against 0.0124 — about a tenth of the value on six clips. Every
  conclusion of 19-21 survives at that size, but a difference of that order
  between two cells is not evidence.
  A worse case than the measured tenth was found 2026-09-21 in `blend18` (the
  interpolation arc of 19.5's predecessor): 5 alphas x 2 pairs = 10 jobs at
  batch 8, so both endpoints (alpha 0 and alpha 1) fall on noise slabs 0-1 and
  the three middle alphas take slabs 2-7. The endpoints are z-matched to each
  other and the middle is matched to neither, which is exactly the comparison
  the arc is read for. The dip it shows (flat fraction 36.8 and 53.0 at the
  ends against 28.7-29.3 in the middle) is therefore not evidence of its own
  size; what survives is the human's visual reading of the midpoint as a
  0.5-alpha double exposure and the 19.0-degree transport measured in
  `reports/2026-09-21_the_draw_distribution.md` section 4, neither of which
  depends on the sampling noise.
- **Reproduce:** render any group whose size does not divide the batch and
  compare a cell's video against the same state rendered alone.
- **Cause:** known — the generator is re-seeded per batch, and the noise index
  inside a batch is the element's position.
- **Evidence:** `reports/2026-09-21_state_restoration.md` § 2;
  `flydream/generate/back21.py` (the fixed form: batch = group).
- **Related:** roadmap 19-22; the fix is one line per script (batch = the
  group size, or draw the noise per clip outside the loop).

### ISS-0008 — the class embedding was never trained: N(0, 1) init and weight decay on the table

- **Status:** fixed in code 2026-09-20, **not yet verified by a rerun**. Seen
  2026-09-20 on roadmap 18 (18.4e, 18.12).
- **Seen:** `SiTStates` built its label table with plain
  `nn.Embedding(n_classes, 128)`, whose default initialisation is N(0, 1) —
  a row norm of **11.3** against **29.8** for the timestep embedding it is
  added to. DiT initialises the same table at std 0.02 (row norm 0.23,
  **fifty times smaller**) so that, with adaLN-Zero, the model starts
  unconditional and grows the label signal only where it pays. Second, AdamW
  applied `weight_decay=0.01` to that table like any other weight, while DiT
  trains with no decay at all. Measured on the two trained checkpoints:

  | | per-class deviation ‖y − ȳ‖ | vs ‖te‖ |
  |---|---|---|
  | 18.4e (`corpus_dct16_cls_c`) | 10.37 | 0.32 |
  | 18.12 (`…_cls_g_c`) | 10.10 | 0.34 |
  | an untrained N(0, 1) table | 11.32 | — |

  The rows ended at 0.89 of their initial random norm, which is what decay
  alone would do over 20,000 steps at lr 1e-3. And the geometry learned
  nothing: the pairwise-cosine matrices of the **two independently
  initialised and independently trained** tables agree at
  **r = +0.0014**, against −0.0124 for a random matrix as control, with
  identical spread (sd 0.091 / 0.090 / 0.089). Both tables are, to
  measurement precision, still their initial noise.
- **Costs:** every class-conditional result of item 18 is void as a test of
  conditioning. 18.4e ("classes change nothing at the gate") and 18.12 ("the
  label carries nothing, ≈0 ≡ 1") measured a model whose class vectors were
  frozen random noise, so **whether the labels carry anything has not in fact
  been tested**. The guidance machinery built on top (18.12) is sound and its
  arithmetic is verified, but it was amplifying the difference between a
  field and a field plus fixed noise.
- **Reproduce:** load either checkpoint with `prior17.load`, take
  `y_emb.weight[:n_classes]`, subtract the column mean and compare the row
  norms with `t_mlp(t_embedding(t, 128))`; then correlate the pairwise-cosine
  matrices of the two checkpoints.
- **Cause:** known, and mine. `SiTStates.__init__` never set the table's
  initialisation, and `train` put every parameter in one decay group. With a
  large random per-class bias added to `te`, the cheapest thing the network
  can learn is to be blind to that subspace — after which no gradient reaches
  the table and decay is the only force left on it.
- **Fix:** `nn.init.normal_(self.y_emb.weight, std=0.02)`, and the table in
  its own AdamW group with `weight_decay=0.0`. Unconditional arms have no
  `y_emb`, so every earlier arm of item 18 stays comparable. Pinned by
  `tests/test_generate.py::test_prior17_label_table_starts_small_and_escapes_weight_decay`.
  **A rerun of the conditional arm is what would close this.**
- **Evidence:** `reports/2026-09-20_step18_5_width_and_the_representation_floor.md`
  §18.12 and §18.14; `data/prior18/corpus_dct16_cls_c.pt`,
  `data/prior18/corpus_dct16_w192_lr1e3_cls_g_c.pt`.
- **Related:** roadmap 18; ISS-0007 (the other defect in the same arm).

---

### ISS-0007 — the gate samples labels the prior was never trained on

- **Status:** open. Seen 2026-09-20 on roadmap 18 (18.4e, 18.12).
- **Seen:** the corpus split holds out **whole classes** for the test side
  (`pairs13(held_classes=…)`), so 10 of the 110 labels — ApplyEyeMakeup,
  Basketball, CliffDiving, Drumming, HorseRace, JumpRope, PlayingGuitar,
  Rowing, SkyDiving, TaiChi — have **zero** training clips and their
  embedding rows never receive a gradient. `samples17.run` nevertheless draws
  the sampling label uniformly over all 110
  (`y = torch.randint(0, n_cls, (n_samples,))`), so about **9.1 %** of every
  conditional gate run is conditioned on a row still at its
  `nn.Embedding` initialisation. Confirmed on the 18.12 run: sample 14 of 16
  drew `CliffDiving`.
- **Costs:** roughly one sample in sixteen of every conditional arm carries a
  large random vector added to its timestep embedding, which is noise in
  exactly the arms being compared (18.4e, 18.12). It does **not** explain
  18.12's null result — an untrained row makes the conditional field differ
  *more* from the unconditional one, not less, so the measured effect is if
  anything inflated — but any further conditioning work must draw labels only
  from the trained set, and the two arms already measured should be re-read
  without those samples.
- **Reproduce:** `python /tmp/lab.py` pattern — take
  `data/corpus18/pairs_manifest.json` and the `meta` of
  `data/corpus18/videos.npz`, label each clip with `_clip_label`, and count
  per class over `split["train"]`; the minimum is 0 for 10 classes. The drawn
  labels of a run are in its summary under `sampled_classes`.
- **Cause:** known. Held-out classes are the point of the split (the gate
  needs unseen clips), but the sampler's label draw was written before the
  prior was ever class-conditional and was never restricted to the classes
  that have training data.
- **Evidence:** `reports/2026-09-20_step18_5_width_and_the_representation_floor.md`
  §18.12; `data/prior18/samples18_corpus_dct16_w192_lr1e3_cls_g_c_g1.json`
  (`sampled_classes`); `data/prior18/corpus_dct16_w192_lr1e3_cls_g_c_train.json`
  (`class_names`).
- **Related:** roadmap 18; ISS-0004 (a measurement artefact of the same kind).

---

### ISS-0006 — even-spaced decoder lags alias the stimulus frame hold

- **Status:** open; `config.toml [decode] lag_windows` changed to consecutive
  lags, the 80 ms and 160 ms windows of the sweep being recomputed. Seen
  2026-09-18 (night) by the human in the 80 ms ladder clip
  (`2026-09-18_decode_sintel_flyvis_ladder_lag_0_2_4.gif`): L3 flickers.
- **Seen:** Sintel is 24 fps resampled to the model's 50 Hz, so each source
  frame is held for about two steps and the stimulus changes mostly on even
  steps (mean frame-to-frame change 0.027 on even, 0.013 on odd steps). With
  lags (0, 2, 4) the decoder's taps sit in one phase of that hold; the fit
  averages the two regimes and the prediction alternates: L3 per-frame
  PixCorr on five test clips 0.94 / 0.92 / 0.94 / 0.89 / 0.93 / 0.86 / 0.92
  / 0.85 …, prediction jump even→odd 0.031 against odd→even 0.043. At lag 0
  the curve declines smoothly; at (0, 1, 2, 3, 4) the jumps are 0.0385 /
  0.0378 and the alternation is gone (mean corr 0.896). L3 shows it most
  (an integrating cell whose response to each frame update is what the
  40 ms taps read); Tm5a less.
- **Costs:** the sweep's 80 ms and 160 ms points (report §10, ISS-0004)
  were measured with aliased windows; the stage-level conclusion may hold
  but the per-type numbers at those windows are not the final ones.
- **Reproduce:** the script in the session transcript of 2026-09-18 night;
  `flydream.decode.map --lags 0 2 4` against `--lags 0 1 2 3 4` on
  `2026-09-18_decode_sintel_flyvis/pairs.npz`, per-frame correlation on
  the test clips.
- **Cause:** lags were chosen as a reach in milliseconds without regard to
  the stimulus's own update period; a window must sample every step it
  spans.
- **Related:** roadmap 4; ISS-0004.

### ISS-0005 — the export's lamina and feedback pairs carry ~30× less total input than FlyVis

- **Status:** open. Seen 2026-09-18 while diagnosing why the corrected
  transplant (ISS-0003) sent two of three members to infinity.
- **Seen:** total synapses a target cell receives from a source type,
  FlyVis against `data/ol/filters_R.json`: R1–R6→Am 36 against 1.2, Am→T1
  63 against 3.3, Tm2→L2 7.3 against 0.3, Mi4→Tm2 5.4 against 0.4, Am→L3
  14 against 1.6. Columnar motion pairs agree: Mi1→T4a 64 against 60,
  L2→Tm2 123 against 128, Tm9→T5a 47 against 28. 63 of 395 matched pairs
  differ more than fivefold by this statistic.
- **Costs:** a transplant that preserves total input (ISS-0003) multiplies
  these pairs' gains by up to 34 and the network runs away; a transplant
  that does not leaves them thirty times too weak. Either way model zero's
  lamina is not FlyVis's lamina, and every result read through Am, T1 or
  the Tm2/L2 feedback is suspect.
- **Reproduce:** `zero.total_n_syn` on both networks; the comparison is
  printed by the diagnostic in `reports/2026-09-18_step4_decoder_stack.md`
  §9 (correction).
- **Cause:** two causes, separated 2026-09-18 evening against the raw
  tables (`data/malecns/*weights*.feather`, `data/ol/neurons_R.parquet`).
  (a) **Am is a third-party reconstruction gap, like CT1.** FlyVis's Am is
  one cell per column (721). MaleCNS names it `Lai` (the bridge renames it
  by `olmatching.tsv`), and the right lobe holds **49** Lai bodies, with
  the source table's own note "many OL Lai are fragmented". The 49
  fragments carry every R→Am, Am→T1 and Am→L3 synapse the export sees, so
  those pairs are about fifteenfold short by cell count alone; the
  photoreceptor side adds the rest, because MaleCNS pools R1–R6 as one
  class (`R1-R6`, 887 cells) that the export expands into six types. No
  threshold or column choice recovers cells that are not in the
  reconstruction. (b) **The feedback pairs are a threshold artefact.**
  Tm2→L2 has 1,421 synapses over 670 rows at weight ≥1 and 234 over 42 rows
  at ≥5: `min_weight` 5 removes 84% of the pair; per target cell that is
  2.2 at ≥1 against the export's 0.3 and FlyVis's 7.3. Mi4→Tm2 likewise
  (1,887 → 338). Columnar motion pairs lose about 20% to the same cut
  (Tm9→T5a 41.2 → 32.2 per cell, Mi1→T4a 72.2 → 65.1) and are not the
  problem. Fix for (b): a per-pair floor, keeping every row of a pair
  whose FlyVis total is small; for (a): record as third-party input, and
  decide with the human whether the Am pathways borrow FlyVis's own
  filters (as a named exception, like a CT1 restoration would be) or stay
  absent and are named in every report that reads through the lamina.
- **Evidence:** `data/runs/2026-09-18_step2_zero_R_v6_rescaled/*.json`
  (the `capped_pairs` list once v7 lands), `reports/runs.jsonl`
  (`step2.transplant`).
- **Related:** ISS-0003, ISS-0002 (CT1 is the same class of defect on a
  single cell), roadmap 1 and 5.

### ISS-0004 — the step-4 stage curve is a decoder artefact

- **Status:** open; the figure is withdrawn from the report and the map is
  being redone. Seen 2026-09-18 by the adversarial audit (six auditors and
  a judge, `research_notes/audit_2026-09-18/`), confirmed by the project
  agent on the same `pairs.npz`.
- **Seen:** with the decoder's window changed from lag 0 to lags (0, 2, 4)
  at dt 0.02 s, i.e. 60 ms, and the time-shuffle control refit at the same
  window, the control-corrected PixCorr goes T5a 0.057 → 0.625, T5d 0.111 →
  0.534, TmY15 0.054 → 0.417, T4a 0.141 → 0.345, while L1 0.336 → 0.300 and
  R1 0.579 → 0.358. The four types at the bottom of the curve end above L1.
  Across 11 types, corr(lag-0 score, gain from the window) = −0.909. The
  frame component at lag 0 separates the published sustained class (L3,
  Mi4, Mi9, Tm9: 0.596 ± 0.048) from the transient class (L1, L2, L4, Mi1,
  Tm1–4: 0.390 ± 0.093) with no overlap across three stages: the map read
  temporal-filter identity, not stage. Stage explains R² = 0.317 of the map
  (Chen et al. 2024, the cited benchmark, R > 0.9); 50% of the variance is
  within stage; hop distance from the photoreceptors alone explains 0.367.
- **Costs:** the advertised first figure (ROADMAP item 4, "the curve R → L
  → Mi/Tm → T4/T5") does not exist in that form; the ladder picture sent on
  2026-09-18 is a true picture of what a per-frame linear decoder recovers
  and a false one of what each stage carries.
- **Reproduce:** `config.toml [decode] lags = [0, 2, 4]`, rerun
  `flydream.decode.map --stimuli sintel --cache` on the cached pairs.
- **Cause:** `lags = [0]` was chosen to match FlyVis's own per-frame head
  and never swept; cells with slow membranes carry the frame's luminance
  in their present voltage, transient cells carry it in their recent
  voltage, and a lag-0 decoder can read only the first. Three further
  defects compound it, each verified by the project agent: the map is read
  from `flow/0000/000`, which is the best of the 50 members by validation
  loss (Spearman(id, loss) = 1.000; member ids are a ranking); the
  sample-shuffle "floor" of 0.117 is the PixCorr of the training-mean
  image, the same for every type (SD 0.003), so it is not a per-type null;
  and seed 0 is the minimum of six scene splits for both Mi1 (0.414 against
  a mean of 0.605) and T5a (0.188 against 0.492).
- **Evidence:** `data/decode/2026-09-18_decode_sintel_flyvis/map.csv`,
  `by_stage.csv`; the audit's own refits in `research_notes/audit_2026-09-18/`.
- **Sweep (2026-09-18, four windows on the same pairs, all 65 types,
  `flydream.decode.sweep`, `data/decode/2026-09-18_sintel_lag_sweep/`):**
  median frame component per stage at 0 / 20 / 80 / 160 ms: photoreceptors
  0.579 / 0.404 / 0.358 / 0.349, lamina 0.435 / 0.366 / 0.300 / 0.255,
  medulla Mi 0.498 / 0.509 / 0.392 / 0.249, Tm/TmY 0.438 / 0.314 / 0.326 /
  0.256, T4/T5 0.221 / 0.384 / 0.511 / 0.427. Spearman(score at 0 ms, gain
  to 160 ms) = −0.827 over 65 types: whatever reads best at one window reads
  worst at another. Median raw PixCorr peaks at 80 ms for every stage but
  the lamina, so 80 ms is the window the map is redone at, reported with
  the 0 ms column beside it. Figure `reports/figures/2026-09-18_sintel_lag_sweep.png`.
- **Related:** roadmap 4; `reports/2026-09-18_step4_decoder_stack.md` §9.

### ISS-0003 — the transplant copied a connectome-relative gain across connectomes

- **Status:** mitigated. The basis is corrected and capped, and the v7 run
  (`2026-09-18_step2_zero_R_v7_total_cap3`) is stable on all three members:
  MaleCNS T4/T5 DSI 0.101 / 0.150 / 0.075 (mean 0.109) against the raw
  copy's 0.020 and FlyVis's 0.547 / 0.344 / 0.281 (mean 0.391);
  preferred-direction error 35.7° / 10.2° / 8.2° against FlyVis's 11.5° /
  14.4° / 69.1°; flash polarity 0.933 / 0.933 / 0.967 against 0.938 / 0.875
  / 0.906. The gap to FlyVis went from 20× to 3.6×; what remains is
  attributed, not fixed: the 114 capped pairs (ISS-0005) and the missing
  CT1 (ISS-0002). Seen 2026-09-18 by the audit, verified in flyvis's source
  and by measurement by the project agent.
- **Seen:** `flyvis/network/initialization.py:501` sets `syn_strength =
  scale / <n_syn>` for the pair in that connectome and `dynamics.py:165`
  forms `weight = sign * n_syn * syn_strength`; `zero.transplant` copied
  the value by type-pair key. On the raw copy, by total input per target
  cell, Tm9→T5a–d (T5's main excitatory drive) came out 1.5–1.7× too weak
  and TmY15→T4/T5 (inhibitory) about 2× too strong. Rerun with the gain
  rescaled: member 000's T4/T5 DSI 0.020 → 0.161 (FlyVis 0.547) with
  preferred-direction error 7.0° against FlyVis's own 11.5°; members 001
  and 002 diverged, because the first rescale used the mean per edge, which
  reached 34× on the pairs of ISS-0005.
- **Costs:** step 2's headline (DSI 0.020 against 0.391) was at least partly
  this function and not MaleCNS; it is the number DECISIONS cites for
  putting item 4 before item 3, and the recorded reason is corrected there.
- **Reproduce:** `python -m flydream.model.zero --models 0 --no-rescale`
  against the default.
- **Cause:** known: the parameter's definition was not read before it was
  copied. Fixed basis: `syn_strength * total_src / total_dst` over
  `zero.total_n_syn`, bounded by `config.toml [model] rescale_cap`.
- **Evidence:** `reports/runs.jsonl` (`step2.transplant`),
  `data/runs/2026-09-18_step2_zero_R_v6_rescaled/`, the v7 run.
- **Related:** ISS-0005, ISS-0002; DECISIONS 2026-09-18 (the order of 4
  and 3, corrected); roadmap 2, 3.

### ISS-0002 — the right CT1's partners are not typed optic-lobe neurons

- **Status:** open; a property of MaleCNS v1.0 as released (status "Roughly
  traced"), not of this code. Seen 2026-09-18 on roadmap 2.
- **Seen:** `connectome-weights` rows with `body_pre = 10157` (CT1, right,
  the only CT1 body on that side): 56,517 rows at any weight, of which 8
  reach a typed right-lobe neuron (LT33 284 synapses, OLVC3 13 at weight
  ≥5); no row reaches a T4, T5, Tm9 or Mi1, which in FlyVis receive 7–40
  CT1 synapses per column. The CT1 has a column ROI (31, 24) like any
  columnar cell.
- **Costs:** the export has no CT1 edges, so model zero lacks the CT1
  inhibitory compartments FlyVis has in every column (CT1(M10) onto T4,
  CT1(Lo1) onto T5); the missing input is one candidate for the weak T4/T5
  selectivity seen in the transplant.
- **Reproduce:** the diagnostic in `reports/2026-09-18_step2_model_zero.md`
  §3 (filter `body_pre == 10157` in the weights table and join to
  `data/ol/neurons_R.parquet`).
- **Cause:** unknown. Plausible: the giant CT1 is split into many untyped
  fragments in v1.0 and the typed body holds only part of the arbor; or its
  postsynaptic partners in the medulla/lobula are the unproofread fragments
  (postsynaptic completion 42%). neuPrint `fetch_adjacencies` on 10157
  would show the partner bodies' status.
- **Evidence:** `reports/2026-09-18_step2_model_zero.md`.
- **Related:** roadmap 1 export, roadmap 2; the neuron count 1 per side in
  `data/bridge/types.csv`.

### ISS-0001 — Tm4's column tag is not the column of its synapses

- **Status:** open; a property of the MaleCNS v1.0 annotation, not of this
  code. Seen 2026-09-18 on roadmap 1.
- **Seen:** right optic lobe, 833 Tm4 cells with `assignedOlHex1/2`: the
  column ROI (`ME_R_col_*`) holding most of the cell's synapses equals the
  tag in 52.9% of cells; the partner-median inference agrees with the tag in
  36.3%. For the other 14 tagged types both methods agree with the tag in
  ≥99.3% (C2 99.5%, Tm20 99.4%, Tm9 99.3%, the rest ≥99.8%).
- **Costs:** whichever column is used, Tm4's offsets to its partners are
  shifted by about one column relative to the other types; a filter
  `Tm4 -> X` or `X -> Tm4` carries that shift. Tm4 is an input to T5 and to
  several Tm/TmY types.
- **Reproduce:** `python -m flydream.data.optic_lobe` with
  `data/ol/roi_columns.parquet` present; the "roi exact by type" line.
- **Cause:** unknown. The tag was likely assigned by a different anchor
  (Tm4's lobula terminal, or the column of its main input) than the medulla
  synapse mass; the MaleCNS release notes do not say how the 15 types were
  tagged.
- **Evidence:** `reports/2026-09-18_step1_data.md` §3; `data/ol/neurons_R.parquet`.
- **Related:** roadmap 1; the choice of ROI columns in `DECISIONS.md`
  (none yet; a step-2 question).

---

## Closed

Shortened to what a later reader needs; the linked report has the rest.
