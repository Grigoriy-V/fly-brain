# Roadmap

**Updated:** 2026-09-21

**Project status:** the chain `video → frozen brain → T4/T5 state → generator →
video → frozen brain` works end to end and is measured, so the project can
render any valid brain state and cannot yet invent one. The goal since
2026-09-20 (the human): **new video without a source clip**, through generated
brain states — `ε ~ N(0, I) → noise2state → state → 13B → video`. The
acceptance criterion is his and overrides every metric
(`reports/2026-09-20_the_seed_problem.md` §5): a result counts only as a
**clip against the raw corpus video**, and blur is accepted for a first
version if scenes appear and a fresh draw gives a new, meaningful video.
Since 2026-09-21 that goal has a narrower first target, also his: **a scene as
one still picture**, not a clip - "добиться сцены не с видео а просто
картинкой, а движение меня пока не парят" - over a static state of 721 x 2
numbers.

**Current approved step:** 29.2, the eight-slice static flow (queue below).
The human fixed its composition 2026-09-21 ("1 и 4 берём на след тест") and
excluded from it the trained renderer, the other six types and rotation
augmentation. The run itself is **not yet authorized**: it is priced and asked
for separately, as every priced run is.

Observed defects are in `ISSUES.md`, which is not a plan and authorizes
nothing. `docs/PROJECT_MAP.md` and `docs/OPERATIONS_MAP.md` describe the
system and operations, `docs/ARTEFACTS.md` how a result is delivered,
`AGENTS.md` holds execution rules, `DECISIONS.md` preserves approved durable
choices. This file alone owns current work, order and authorization.

## Current state

- **Data:** MaleCNS v1.0 and the FlyVis 1.2.0 ensemble under `data/` with
  hashes in `data/manifest.json`; the right optic lobe as
  `data/ol/filters_R_w5wk50m500oc.json`. The training corpus is ordinary video
  (UCF101, 15,514 clips from 12,411 source files, split by class), built by
  `corpus18`; states for it in `gen18/maps_dct16.npz` on the Modal volume.
  Settings in `config.toml`.
- **Model zero:** FlyVis dynamics on the MaleCNS export with a member's
  parameters transplanted by type (`flydream/model/zero.py`). Frozen; it is
  both the encoder and the verifier. Training it is paused (3').
- **Generator 13B:** the conditional flow model state → video
  (`flydream/generate/gen13b.py`, `data/gen13b/sit.pt`, mirrored on the
  volume). Trained on the item-13 corpus, frozen, not a bottleneck.
- **The object:** the T4a+T4b block, 721 columns x 32 channels = 23,072
  numbers, with the other six types declared absent through 13B's own type
  mask; r 0.884 against 0.952 for the whole state (22).
- **The static object** (since 29): coefficient 0 of that same block, 721
  columns x 2 types = 1,442 numbers, with no time axis at all. It is **not**
  721 x 32 - the human, twice: "никаких 721*32 быть не может, в чём тогда
  смысл выкидывания времени". It is read off the states already on the volume;
  nothing is re-simulated.
- **First stage:** PCA over the training states, whitened
  (`flydream/generate/pca19.py`); PCA-2048 over the whole state and PCA-1536
  over the block, basis and latents on the volume under `prior19/`. Acceptance
  in `pcaval19.py`. Step 23 drops it: a linear basis keeps no lattice for a
  local network to use.
- **noise2state:** flow matching over the whitened latent, 16 tokens of 128
  (`prior17.py` with `train17(latent_tokens=…, latent_k=…)`); checkpoints under
  `prior19/` (2,048 of the whole state) and `prior22/` (1,536 of the block).
  Since 23 the flow also runs on the lattice itself (`hexflow23.py`,
  `train17(backbone="hex")`), checkpoints `prior23/` over the block and
  `prior29/` over the static state. The live problem is here.
- **Rendering a static state:** 13B cannot do it. It goes blank on a state
  that is constant in time and no synthetic residue rescues it - the best of
  four crutches reads r 0.351 against a 0.941 ceiling, and the donor arm
  renders the donor's picture sharply, so the content it reads lives in the
  temporal coefficients. The provisional judge is a least-squares matrix from
  the 1,442 numbers to the window's mean frame, r 0.941 on held-out clips from
  unseen classes; it blurs, because least squares draws the conditional mean.
  A trained renderer is queue 30 and **does not exist in any form**: no
  function on Modal, no module under `flydream/generate/`.
- **Measurement:** `seed19.py` (the seed test: preimage geometry, a clip's own
  seed, a fresh draw, controls without the flow, interpolation, `--fix` for a
  corrected sampler), `tprofile19.py` (where the sampler leaves the truth),
  `fix19.py` (the sampler knobs and the velocity profile), `cut19.py` and
  `inside22.py` (how far the state can be cut), `latent22.py` (the PCA ladder
  and the compression routes), `hexcov19.py` (covariance against hex distance),
  `back21.py` (what the chain restores), `floors22.py` (the floor and range of
  every judge), `accept23.py` (kurtosis, radius and transport angle against
  matched training subsamples), `seed23.py` (the seed test without PCA),
  `still23.py` and `static23.py` (whether one still picture survives the
  chain), `fixdraw23.py` (corrections at draw time), `vaeval18.py`,
  `roundtrip13.py`, figures under `tools/fig_*.py`.
- **Compute:** the owner's machine (32 cores, 102 GB, CPU) for anything that
  fits in ten minutes; Modal T4 (`flydream-generate`, `flydream-train`,
  `flydream-decode`) for GPU work, called through `tools/modal_call.py`.
- **Tests:** 132 offline tests passing (2026-09-21, 26 s), no download, no
  Modal, no credential.

## Done

Closed items, one line each, evidence in the linked report.

- **Research, the landscape and the plan** (2026-09-18).
  `reports/Коннектом мухи и план проекта.md`,
  `reports/Обучение коннектомных сетей и ускорение.md`.
- **1, data** (2026-09-18): MaleCNS core files, the FlyVis ensemble, the right
  optic lobe export. `reports/2026-09-18_step1_data.md`.
- **2, model zero** (2026-09-18): FlyVis dynamics on the MaleCNS export,
  parameters transplanted by type. `reports/2026-09-18_step2_model_zero.md`.
- **3, training priced** (2026-09-18): T4 smoke, packing, batch sweep.
  `reports/2026-09-18_step3_training_options.md`,
  `reports/2026-09-18_training_optimization_bench.md`.
- **4, the decoder ladder** (2026-09-18/19): ridge per type, the hex-conv
  head, the lag sweep. `reports/2026-09-18_step4_decoder_stack.md`.
- **8-9, the generator by encoder inversion and its window** (2026-09-19).
  `reports/2026-09-19_step9_window_margin.md`.
- **11, dreams-lite** (2026-09-19): inversion of states no clip caused.
  `reports/2026-09-19_step11_dreams_lite.md`.
- **12, manipulated states** (2026-09-19).
  `reports/2026-09-19_step12_manipulated_states.md`.
- **13A, amortised inversion** (2026-09-20): pairs from the frozen brain, the
  learned decoder. `reports/2026-09-19_step13a_amortised_inversion.md`.
- **13B, the conditional flow generator** (2026-09-20): the decoder this whole
  track uses. `reports/2026-09-19_step13b_design.md`,
  `reports/2026-09-20_step13b_generative_decoder.md`.
- **14, what can be prompted** (2026-09-20): random, edited and composed
  states; the first video with no source clip.
  `reports/2026-09-20_step14_controllable_generator.md`.
- **17, the brain-state prior** (2026-09-20): the free unconditional baseline,
  the flow over states, the DCT representation, the noise inversion.
  `reports/2026-09-20_step17_state_prior.md`,
  `reports/2026-09-20_step17_0_unconditional_baseline.md`,
  `reports/2026-09-20_step17_1b_scene_dct_prior.md`,
  `reports/2026-09-20_step17_3b_noise_inversion.md`.
- **18, the corpus of ordinary video and the arms over it** (2026-09-20):
  every lever tried on the flow over states, and the seed problem stated in
  the human's words with the list of what is eliminated.
  `reports/2026-09-20_step18_corpus_of_ordinary_video.md`,
  `reports/2026-09-20_step18_3_prior_on_the_corpus.md`,
  `reports/2026-09-20_step18_4_throughput_and_learning_rate.md`,
  `reports/2026-09-20_step18_5_width_and_the_representation_floor.md`,
  `reports/2026-09-20_the_seed_problem.md`.
- **Research, how VAEs are actually trained** (2026-09-21): why the first
  stage may be linear and why nobody asks a latent to be Gaussian.
  `reports/2026-09-21_research_how_vaes_are_trained.md`.
- **Research, the path to a video generator** (2026-09-21): six scouts on
  sample complexity, the overshoot, non-neural samplers, a hex-local prior,
  conditioning and off-the-shelf generators; nine candidate routes, none
  chosen. `reports/2026-09-21_research_path_to_a_video_generator.md`.
- **19, a linear first stage and a flow over its latent** (2026-09-21): PCA on
  the card and its acceptance, the flow over the latent in two recipes, the
  seed test, the time profile, the controls without the flow and the
  interpolation between two seeds. Runs `2026-09-21_prior19_*`; code
  `flydream/generate/{pca19,pcaval19,seed19,tprofile19}.py`.
  `reports/2026-09-21_step19_linear_first_stage.md`.

- **20-21, the sampler fix and the state cuts** (2026-09-21): the overshoot is
  a time-skew and the literature's constant-divisor fixes miss it; a projection
  onto the measured trajectory curve repairs the geometry and improves the gate
  2.2x without reaching a scene; a dropped half of the state costs nothing when
  completed and is unusable when zeroed; 128 PCA components carry the gate as
  well as 2,048; the state's covariance is local (0.876 at one step against a
  0.267 shuffled control). Runs `2026-09-21_prior19_{fix_sweep,seed_fix,cut,hexcov}`;
  code `flydream/generate/{fix19,cut19,hexcov19}.py`.
  `reports/2026-09-21_step20_sampler_and_cuts.md`.

- **21c, state restoration through the chain** (2026-09-21): the gate in 21a
  measured distance to the requested state, which for a cut state is
  impossible by construction; against the real state the chain
  state -> 13B -> video -> brain gives back a zeroed half at r 0.961 (error
  0.5581 -> 0.0846, own floor 0.0124), restoring exactly what the video
  determines. Run `2026-09-21_prior19_back`; code `flydream/generate/back21.py`.
  `reports/2026-09-21_state_restoration.md`.

- **22, a smaller object** (2026-09-21): inside T4 space is cheap to cut and
  time is not; the four directions are unequal and the horizontal pair a+b
  carries almost everything (0.884 against 0.904 for all four, at half the
  numbers); decimating the lattice costs as much as cropping the field of view,
  and 13B does not fill the gaps itself; the PCA ladder over the block is flat
  (0.806 at 1,536 against 0.824 at 2,048); and three routes around plain PCA -
  channel compression, a learned local residual, a block autoencoder - all lost
  to it at a comparable budget. The flow over the block's 1,536 coordinates
  trained for 0.05 dollars and a fresh draw is still not a scene. Runs
  `2026-09-21_prior19_inside` and `2026-09-21_prior22_*`, 0.13 dollars in all;
  code `flydream/generate/{inside22,latent22,resid22}.py`.
  `reports/2026-09-21_step22_a_smaller_object.md`.

- **22.7-22.8, the metric audit** (2026-09-21, on the human's stop before more
  training): the flow learned the latent's second moment and not its fourth -
  per-coordinate kurtosis 4.11 against 8.31 +- 1.17 on matched training
  subsamples, a Gaussian being 2.97 - and it rotates a point by 19 degrees
  against 90 for a random rotation, which is why an interpolation renders as a
  double exposure. Every judge got its floor: `r_to_raw` between two different
  real clips is 0.002 +- 0.100, so a draw's 0.017 was never evidence;
  `nearest_r` for a real held-out clip is 0.521 against the draw's 0.526, so
  "not a copy" is true and weak; on a noise-to-raw-video scale the renderer's
  own ceiling is 71 and a fresh draw is 11. Found ISS-0010 and ISS-0011, closed
  ISS-0009 in the three live scripts. Runs
  `2026-09-21_prior22_{draw_distribution,floors}`, free; code
  `flydream/generate/floors22.py`, `tools/fig_dist22.py`.
  `reports/2026-09-21_the_draw_distribution.md`.

- **23, a hex-local prior over the block** (2026-09-21): the flow runs on the
  T4a+T4b block itself, 721 x 32 with the lattice intact - 3x patchify to 241
  patches of exactly three columns, attention restricted to the six
  sublattice neighbours, a relative position bias shared across positions
  instead of a learned per-token one, and four register tokens as the global
  channel. Every measured number moved: per-coordinate kurtosis of draws 4.11
  to 5.36 against 8.31 +- 1.17 on matched training subsamples (21 to 45
  percent of the way from noise to data), the radius overshoot 20 to 6.4
  percent, and the transport from a 19-degree rotation to 72 where a random
  one is 90 - the near-identity that made interpolation render as a double
  exposure is gone. A clip's own seed now returns its clip at 0.884, exactly
  the ceiling, because dropping PCA removed the lossy stage. A fresh draw
  doubled its sharpness, 11 to 22 of 100 on the scale calibrated in 22.8, and
  is still not a scene; it is not a copy either (nearest 0.485 where a real
  held-out clip reads 0.521). One new defect: the inverse lands real blocks at
  radius 136.2 against a shell of 151.9. Runs `2026-09-21_prior23_*`, 0.51
  dollars on one T4 at 99.9 percent utilisation, acceptance local and free;
  code `flydream/generate/{hexflow23,accept23,seed23}.py`.

- **26, corrections at draw time** (2026-09-21, delegated, local, free):
  **nothing works, and the acceptance metric turned out to be gameable.**
  Seven arms over the drawn noise - the shell set to the inverse's own radius
  136.2, then 140, 145 and 151.9, and a plain scale of 0.90 and 0.95. At
  D = 23,072 the two knobs are the same knob, since `randn` concentrates
  within well under a percent of sqrt(D). Two arms raise kurtosis above the
  measurement noise (145 to 7.11 and scale 0.95 to 7.17 against the control's
  5.36, where the training reference is 8.31 +- 1.17) - but their radius in
  the PCA basis falls to 12.39 and 11.57 against the data's 35.50, a threefold
  UNDERSHOOT, and their energy outside the PCA subspace drops from 16.3 to
  about 6 percent where real blocks sit at 14.7. Both say the output collapsed
  toward the corpus mean, and kurtosis rises mechanically when most
  coordinates sit at zero. The arm that implements the measured diagnosis
  directly, drawing at radius 136.2, is the worst of the sweep on kurtosis.
  So the item-20 precedent holds again: a geometric gain is not a generator
  gain. **Consequence for the acceptance metric: per-coordinate kurtosis must
  always be read beside the radius, because shrinking the output buys kurtosis
  for free.** Run `2026-09-21_prior23_fixdraw`; code
  `flydream/generate/fixdraw23.py`.

- **23.1-23.2, is there a still picture in this state space** (2026-09-21):
  yes, and better than a clip. 23.1 asked it wrongly by zeroing 15 of 16
  temporal coefficients of a real state, which hands 13B a state no clip ever
  caused; the human called that invalid and it is withdrawn. 23.2 changes the
  stimulus instead: a single frame held for the whole window goes stimulus ->
  frozen brain -> state -> 13B at r 0.953 against 0.916 for the real moving
  clip, contrast 0.128 against the stimulus's 0.130, and the render is
  genuinely motionless, frame-to-frame 0.998 against 1.000. The still frame's
  state carries the same energy as a clip's, so "T4 and T5 are motion
  detectors, therefore no static picture" is false. The same frame drifting a
  lattice step per frame is worse, 0.611. Runs
  `2026-09-21_prior23_{still,static}`; code
  `flydream/generate/{still23,static23}.py`.

- **29a, a picture out of a state without 13B** (2026-09-21, local, free):
  the static content is there and it is LINEAR. A least-squares matrix from
  the 1,442-number DC to the window's mean frame reads r 0.941 on held-out
  clips from ten unseen classes, against 0.021 for predicting the corpus mean
  and 0.002 for shuffled pairs; from the whole block to one single frame,
  0.902 against the DC's 0.825, so the instantaneous part of the state is what
  a single frame needs. What it cannot do is sharpness - flat fraction 24.3
  against the target's 45.1 - because least squares draws the conditional mean
  over every picture compatible with the state. The same run killed the
  crutch family: 13B on DC plus zeros 0.257, plus the mean still residue
  0.351, plus another still's residue 0.137 (and it renders the DONOR sharply),
  plus unit noise 0.045, against a 0.941 ceiling. Runs
  `2026-09-21_prior23_{pic,pic_k16,resfix}`; code `tools/fig_pic23*.py`.

- **29, the flow over the static state** (2026-09-21): trained on the DC of
  the states already on the volume - no new brain pass, since the DC of a
  moving clip and the state a still frame causes agree at r 0.922. Hex-local,
  3x patchify, 10,000 steps at batch 256 and lr 2e-3, 2,512 s on one T4 at
  99.0 percent utilisation, **0.42 dollars**, 0 nan, val 0.2160 against a
  zero-velocity control of 2.0. Through the least-squares renderer a draw
  matches a real held-out state exactly on flat fraction, 27.1 against 27.1,
  and nearly on neighbour coherence, 0.928 against 0.949, where the no-flow
  N(0,I) control reads 19.5 percent and 0.492 - the first time in this line
  that a draw is not separable from a real state on a structural judge.
  Diversity is right too: pairwise correlation between draws -0.007 mean and
  0.819 max against 0.008 and 0.758 between real pictures. Two things bound
  it. The renderer blurs, so matching a real state THROUGH IT is a much lower
  bar than being a scene; and the target distribution is nearly Gaussian
  (training DC kurtosis 3.17 +- 0.10 against a Gaussian's 2.98), because a
  40-frame average is driven to normality, so the kurtosis gate that governed
  22-23 is vacuous here. The human's verdict by eye: closer to a scene than
  anything before it, and not yet a scene. Run `2026-09-21_prior29_static_ab`;
  code `flydream/generate/hexflow23.py` with `train17(coef=1)`.
  **Reports for 23, 26 and 29 are not written yet**; the numbers above and in
  `reports/runs.jsonl` are the record until they are.

- **29.1, why one frame and not forty** (2026-09-21, local, free): the DC is
  a 40-frame average and that is what makes it Gaussian and blurred. A single
  temporal slice of the same state is genuinely sparser, per-coordinate
  kurtosis 5.26 against the DC's 3.17, and it renders its own frame at r
  0.93-0.96 (clip 6008) and 0.88-0.92 (clip 51) for frames 5 to 35. Frame 0 is
  broken, r -0.936 and -0.890, a flat grey field - a window-edge artefact - so
  slices start at frame 5. Measured with a throwaway script that was not kept:
  the numbers stand, the figures are in `reports/figures/*slices8*.png`, and
  29.2 re-derives them in committed code.

## Queue

One item at a time; the human's word starts each, and each is priced when it
is proposed. Two levers have been spent: the sampler (20, not the blocker)
and the size of the object (22, nothing moved). Two have paid: the shape of
the model (23, every number moved once the lattice was kept) and the shape of
the target (29, a draw that a structural judge cannot separate from a real
state). What remains on this branch is sharpness - the renderer - and a target
that is not Gaussian, which is 29.2. Evidence:
`reports/2026-09-21_step22_a_smaller_object.md` and
`reports/2026-09-21_the_draw_distribution.md`, with the full ladder of routes
in `reports/2026-09-21_research_path_to_a_video_generator.md`.

29.2. **Eight slices per clip, not one average** - **the agreed next step**,
    the human 2026-09-21: "1 и 4 берём на след тест". The flow's target stops
    being the 40-frame average and becomes a single temporal slice of the
    state, eight per clip at frames 5, 10, ... 35 (frame 0 is a broken
    window edge, 29.1). Two things follow at once: the object stays 721 x 2,
    and the corpus grows eightfold without one new simulation, which is item
    27's lever at zero cost. The reason is measured, not aesthetic - a slice
    has kurtosis 5.26 against the average's 3.17, so the target stops being
    Gaussian and the acceptance gate stops being vacuous. Excluded from this
    test by the human, explicitly: the trained renderer, the other six types,
    rotation augmentation. Not yet authorized: it is a T4 training run and is
    priced when it is proposed.

30. **A generative renderer, static state to picture** - not approved, and
    the human kept it out of 29.2 ("отрисовщик не"). Priced 2026-09-21 at
    **0.15 to 0.50 dollars**, most likely about 0.25, 15 to 50 minutes on one
    T4. Input 1,442 numbers, output one 721-value picture, no time axis; the
    data is free and needs no brain pass - states in `gen18/maps_dct16.npz`,
    frames in `gen18/maps_deep.npz`, both on the volume, and the DCT-compact
    file carries no videos, so the two halves of a pair come from two files.
    It is needed because 13B cannot take a 721 x 2 state and no crutch
    rescues it (29a), and because the least-squares stand-in blurs by
    construction. Two things stand between it and a run: **no training code
    exists** - no function on Modal, no module under `flydream/generate/` -
    and the objective is undecided, since a squared-error renderer would
    reproduce exactly the conditional-mean blur it is meant to remove, which
    makes this a generative model and not a bigger matrix. Open beside it:
    the human cannot yet confirm by eye that a still picture is correctly
    extracted from a state, because every picture shown to him passed through
    a local PCA-2048 reconstruction (89.5 percent of variance); the true
    states are on the volume and that contamination is local to the laptop.

27. **Corpus four to six times over** - the human, 2026-09-21: "скорее всего
    будет следующим". The reason this was deferred no longer holds: it was
    parked because the published scaling transitions were measured on
    convolutional nets over spatial data and ours was a flat net over a PCA
    vector. Since 23 it is a local net over the lattice. The clips exist as
    unused rotations of the corpus; the cost is re-simulating their states
    through the frozen brain and retraining, and is priced when proposed.
    29.2 takes the same lever eightfold for nothing on the static branch, so
    this item is now about the moving one, or about going past eightfold.

28. **More capacity and more steps** - the human, 2026-09-21: "пока в ожидании
    после 2". Width 192 to 384 and depth 6 to 10-12 with a longer schedule;
    3.68 M parameters is small, and validation was flat over the last 2,500
    steps so steps alone at this schedule buy little. The diagnostic that
    separates this from 27 is the train-validation gap, 0.3464 against 0.3963,
    13 percent - mild, so neither lever is clearly the one. Acceptance is free
    now (kurtosis beside radius, 22.8 and 26), so this reads in an hour.

24. **Conditioning, two-stage.** Draw the DC part of the state, then the
    motion given it. 29 built the first half of this and measured it; what is
    untried is the second, the motion conditioned on a drawn static state. Kept behind the above because its measured gains come
    from fine continuous conditions, and its cost is memorisation, which then
    has to be measured beside every sample. The human asked for an
    explanation of it 2026-09-21 before deciding.

Waiting, not in the order above:

25. **The state space as an interface**, the claim item 17 was accepted on:
    interpolation and editing in state space as a product, not as a
    diagnostic. The machinery exists (`seed19.py --controls`); what is missing
    is a fair novelty test — pairs of two structured clips, r to each parent
    separately, and the path at several alpha.

## Not started

Recorded, not approved, not begun. One line each.

- **Restoring the types that 22 cut** (T4a+T4b back to all four T4, or to all
  eight) - **declined by the human 2026-09-21**: "4 смысла нет, востановление
  идеальное, а сцену строить вряд-ли мешает". The numbers agree: a clip's own
  seed returns its clip at exactly the ceiling, and the ceiling is 66 of 100
  where a fresh draw is 22, so the ceiling binds nothing.
- **The time axis inside the architecture** - raised 2026-09-21, not yet a
  proposal: the human asked what it would actually mean and the honest answer
  is that it is a diagnosis first, not an architecture change. Time is 16 DCT
  coefficients carried as plain channels; the draws are over-smooth in time,
  which in DCT terms means the high-index coefficients are under-produced.
  Measuring the per-coefficient variance of draws against the data says
  whether that is so, and the lever if it is would be `loss_weight_p`, which
  already exists (18.6), not a new backbone.
- **An external generator as the scene or corpus source** — unbounded clips,
  a weaker claim; route G of the same report.
- **3', fine-tuning MaleCNS from the transplanted weights** (the human,
  2026-09-19: "не сейчас"). Commands in
  `reports/2026-09-18_step3_training_options.md` §5б.
- **16, the dream source inside the model:** spontaneous drive, the optic-lobe
  loops the export dropped, central-brain drive. Design
  `reports/2026-09-20_step16_dream_source_design.md`; the human's revision
  `docs/ideas/step16_dream_source_revised.md` takes precedence where they
  differ. Not fully formed, not fully discussed (the human, 2026-09-20).
- **10, a dense MaleCNS export** (every type on all 721 columns), and **a
  64×64 raster of the hexals** — both cosmetic, local work.
- **5, both eyes, every column, the missing biophysics** — for HDRI / 360°
  input, not for clips.
- **6, the central brain and the state knobs** — new export, long training,
  validation.
- **7, dreams, the full protocol** — the rigorous form of item 11.
- **15, HDRI / 360° panoramas as a stimulus source** (the human, 2026-09-19):
  unlimited clips with exact optic flow from rotations.
- **C, colour as an ML task** (the human, 2026-09-19): MaleCNS has the types,
  there are no parameters to transplant; until then colour in an output is a
  display overlay and says so.
- **Deeper than T4/T5 with generic dynamics**
  (`docs/ideas/malecns_shiu_lif_baseline_idea.md`): collides with
  `DECISIONS.md` 2026-09-18 on the model class, so that decision is revisited
  first.
- **Multi-level conditioning of the generator:** early, deep and central
  states together as the condition, after a deeper model exists.
- **Paper-grade rigour:** ensembles, several splits, per-type nulls, subset
  curves, validation against the 26 physiology studies. Code exists and is
  tested; measurements stopped 2026-09-18.
- **The connectome as a computational substrate** — parked 2026-09-18; if
  revisited, a degree-preserving shuffled connectome must do measurably worse.
- **One-liners, recorded only:** the three-connectome comparison; a second
  training objective; the embodied loop in MuJoCo; inversion on real
  recordings; photoreceptor columns from lamina cartridges; the left lobe
  export; the fitted-dynamics parameter source; ablations on model zero
  (ISS-0002); per-target filter normalisation (measured unstable, `"src"`
  stands).

## How this file is kept

- **Only approved work.** A conclusion the human has not approved in words is
  a draft and belongs in `reports/`, or in `docs/ideas/` if it is his own
  note.
- **State and order, not reasoning.** No results, options, comparisons or
  research; those go to `reports/`, durable choices to `DECISIONS.md`.
- **One entry per item, a few lines, plus links.** Evidence lives in the
  report it links to and is not summarized twice.
- **Done is a list of outcomes**, not a history of how they were reached.
- **Queue is an order, not a list.** Unfinished work returns as its own queue
  item instead of staying as a caveat inside a closed one.
- **No budget here.** A price belongs to the step it buys and is stated when
  that step is proposed.
- **Short beats complete.** If this file needs a table of contents, cut it.
