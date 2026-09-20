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

**Current approved step:** 20 and 21 are measured
(`reports/2026-09-21_step20_sampler_and_cuts.md`): the sampler's overshoot is
fixed and was not the blocker, half the state can be left unspecified if it is
completed rather than zeroed, and the state's covariance is local — which is
what chose the order below. Next: 22, on the human's word.

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
- **First stage:** PCA-2048 over the training states, whitened
  (`flydream/generate/pca19.py`); basis and latents on the volume under
  `prior19/`. Acceptance in `pcaval19.py`.
- **noise2state:** flow matching over the 2,048-dimensional latent, 16 tokens
  of 128 (`prior17.py` with `train17(latent_tokens=…)`); two checkpoints under
  `prior19/`. The live problem is here.
- **Measurement:** `seed19.py` (the seed test: preimage geometry, a clip's own
  seed, a fresh draw, controls without the flow, interpolation, `--fix` for a
  corrected sampler), `tprofile19.py` (where the sampler leaves the truth),
  `fix19.py` (the sampler knobs and the velocity profile), `cut19.py` (how far
  the state can be cut), `hexcov19.py` (covariance against hex distance),
  `vaeval18.py`, `roundtrip13.py`, figures under `tools/fig_*.py`.
- **Compute:** the owner's machine (32 cores, 102 GB, CPU) for anything that
  fits in ten minutes; Modal T4 (`flydream-generate`, `flydream-train`,
  `flydream-decode`) for GPU work, called through `tools/modal_call.py`.
- **Tests:** 122 offline tests passing (2026-09-21, 23 s), no download, no
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

## Queue

One item at a time; the human's word starts each, and each is priced when it
is proposed. The order below is what 20 and 21 measured, not a preference:
the sampler is repaired and was not the blocker, so what remains is the shape
of the problem (how many numbers are drawn) and the shape of the model (what
the field can generalise from). Evidence:
`reports/2026-09-21_step20_sampler_and_cuts.md`, with the full ladder of
routes in `reports/2026-09-21_research_path_to_a_video_generator.md`.

22. **A smaller latent.** 128-512 PCA components instead of 2,048: the same
    net on a problem 4-16 times smaller at the same N = 13,555, and the gate
    through the chain is already measured as no worse (0.0334 at 128 against
    0.0406 at 2,048). One training run, judged by the seed test.

23. **A hex-local prior.** Column positions and a neighbour window in the
    flow, plus a global channel — 21b measured both halves: locality is real
    (0.876 at one step) and a window alone would miss the 0.27 background.
    This is the only published mechanism of novelty for a diffusion model.

24. **Conditioning, two-stage.** Draw the DC part of the state, then the
    motion given it. Kept behind 22-23 because its measured gains come from
    fine continuous conditions, and its cost is memorisation, which then has
    to be measured beside every sample.

Waiting, not in the order above:

25. **The state space as an interface**, the claim item 17 was accepted on:
    interpolation and editing in state space as a product, not as a
    diagnostic. The machinery exists (`seed19.py --controls`); what is missing
    is a fair novelty test — pairs of two structured clips, r to each parent
    separately, and the path at several alpha.

## Not started

Recorded, not approved, not begun. One line each.

- **More corpus and more capacity for the flow** — the earlier queue item 20
  (four to six times the states from the unused rotations, the same capacity
  retrained, then capacity). Deferred 2026-09-21: the published transitions
  were measured on convolutional nets over spatial data, so data alone is not
  the gap — routes B and C of
  `reports/2026-09-21_research_path_to_a_video_generator.md`.
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
