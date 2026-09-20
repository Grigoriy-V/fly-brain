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

**Current approved step:** 19 is built and measured
(`reports/2026-09-21_step19_linear_first_stage.md`): the first stage is
linear, the seed is drawable, a clip's own seed returns that clip, a fresh
draw does not yet give a scene. The order of what follows was reset on
2026-09-21 by `reports/2026-09-21_research_path_to_a_video_generator.md`.
Next: 20, on the human's word.

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
  seed, a fresh draw, controls without the flow, interpolation),
  `tprofile19.py` (where the sampler leaves the truth), `vaeval18.py`,
  `roundtrip13.py`, figures under `tools/fig_*.py`.
- **Compute:** the owner's machine (32 cores, 102 GB, CPU) for anything that
  fits in ten minutes; Modal T4 (`flydream-generate`, `flydream-train`,
  `flydream-decode`) for GPU work, called through `tools/modal_call.py`.
- **Tests:** 106 offline tests passing (2026-09-21, 23 s), no download, no
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

## Queue

One item at a time; the human's word starts each, and each is priced when it
is proposed. The evidence behind this order, and the full ladder of nine
routes with their sources, is
`reports/2026-09-21_research_path_to_a_video_generator.md`; the measured state
of the blocker is `reports/2026-09-21_step19_linear_first_stage.md` §§ 7–11.
The task is unconditional 64x64 video generation in other coordinates, so the
order is: free fixes and free measurements first, then the one paid route the
measurements select.

20. **The sampler, not the model.** A fresh draw overshoots the radius 1.55x
    and leaves the ideal trajectory at t = 0.1, which is the published
    exposure-bias signature; both fixes are one scalar on the existing
    checkpoint. Then the seed test. Free, local.

21. **Two free measurements.** How far the state can be cut before the video
    stops holding (types, columns, DCT, PCA components) — and whether column
    covariance decays with hex distance, which is what decides between 22 and
    23. Free, local.

22. **Conditioning, two-stage.** Draw the DC part of the state (8 types x 721
    columns), then the motion given it; the conditioned objective is the one
    with measured gains and it keeps unconditional sampling available.

23. **Hex-local attention in the flow.** Column positions and a neighbour
    window instead of PCA tokens, to reach the only published mechanism of
    novelty. Taken if 21 shows covariance is local.

Waiting, not in the order above:

24. **The state space as an interface**, the claim item 17 was accepted on:
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
