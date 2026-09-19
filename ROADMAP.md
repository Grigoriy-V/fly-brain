# Roadmap

**Updated:** 2026-09-19

**Project status:** the deliverable is a working generator of clips from
the model's internal states ("что снится мухе" is the title; the human,
2026-09-18 night: not a paper, an ML product). Steps 1 (data), 2 (model
zero on MaleCNS), 4 (the decoder ladder) and 8 (the generator by encoder
inversion) are done at their minimal shape; the generator exists on both
brains as the check "state → the clip that caused it". Current branch:
**dreams and visual data** — states no clip caused, manipulated states, a
learned generator. Everything that is not on that branch is kept in
"Beyond the branch" so it is not lost. Research behind the plan:
`reports/Коннектом мухи и план проекта.md`.

**Rules of the day (DECISIONS 2026-09-18 night):** local CPU by default;
Modal only for a GPU-bound job or a ≥4× gain; a training run within $0.50
and only from the transplanted weights; one substrate (MaleCNS) at a time;
one artefact per message with its text first (`AGENTS.md`, "Artefacts for
the human").

**Current approved step:** 13B (design approved 2026-09-20); 9, 11, 12 and
13A are done. Each priced run starts on the human's word.

Observed defects are in `ISSUES.md`, which is not a plan and authorizes
nothing. `docs/PROJECT_MAP.md` and `docs/OPERATIONS_MAP.md` describe the
system and operations; `AGENTS.md` holds execution rules; `DECISIONS.md`
preserves approved durable choices. This file alone owns current work, order
and authorization.

## Current state

- **Data:** MaleCNS v1.0 core files and the FlyVis 1.2.0 ensemble under
  `data/` with hashes in `data/manifest.json`; the right optic lobe exported
  as `data/ol/filters_R_w5wk50m500oc.json` (60 types, 677 type pairs,
  31,526 neurons, 1.39 M edges; weak-pair exception, outputs restricted to
  the 18 columnar types); Sintel under `data/flyvis/SintelDataSet`.
  Settings in `config.toml [data]`.
- **Model:** model zero (`flydream/model/zero.py`): FlyVis's network on the
  MaleCNS export with a member's parameters transplanted by type, gain
  rescaled to total input per target and capped (v9: DSI 0.152 against
  FlyVis's 0.391, flash polarity and direction at or above FlyVis, stable
  on three members). Not trained on MaleCNS; training is paused (3').
- **Decoders:** `flydream/decode/`: ridge per cell type (penalty chosen on
  held-out scenes; float32 SVD), hex-conv head, metrics, raster, the map
  driver, the lag sweep, the ensemble aggregation (beyond the branch). The
  window is part of the measurement (ISS-0004), consecutive lags only
  (ISS-0006).
- **Generator:** `flydream/generate/invert.py`: encoder inversion — a
  20 × 721 video optimised through the frozen network to reproduce one
  stage's state (Adam, TV prior, grey start); `run_ladder` for all stages
  in one model load; `deploy/modal/generate_app.py` runs it on a T4
  (10 stages in ~10 min, ~$0.10); `flydream/generate/figures.py` and the
  two-input figures of 2026-09-19 draw it; the window is `frames + margin`
  (40 + 5), so the last shown frames are constrained (item 9, done).
- **Compute:** the owner's machine (32 cores, 102 GB, CPU) for everything
  that fits in hours; Modal T4 (`flydream-train`, `flydream-decode`,
  `flydream-generate`) for GPU-bound jobs; `tools/modal_watch.py` to watch.
- **Measurement:** decoder ladders at 0 and 80 ms on both brains
  (`reports/figures/2026-09-19_decoder_ladders_flyvis_vs_malecns.gif`); the
  generator recovers clips 3 and 10 from every stage with r 0.92-1.00 on
  both brains (`2026-09-19_*_generator_two_inputs.*`,
  `2026-09-19_generator_forest_flyvis_vs_malecns.gif`). Training priced:
  batch 16 saturates a T4 at 14 samples/s, `stats_relu` gives 1.17×
  (`reports/2026-09-18_step3_training_options.md`). Tests: 64 offline.

## Done

- **Research: the landscape and the plan** (2026-09-18).
  `reports/Коннектом мухи и план проекта.md`; how connectome models are
  trained and sped up: `reports/Обучение коннектомных сетей и ускорение.md`.
- **The records** (2026-09-18): this file, `AGENTS.md`, `DECISIONS.md`,
  `ISSUES.md`, `docs/`.
- **1, the data step** (2026-09-18): MaleCNS core files and the FlyVis
  ensemble fetched with a manifest; the type bridge (61/65); home columns
  from neuPrint column ROIs; hex axes aligned to FlyVis; filters exported in
  FlyVis shape (0.869 of FlyVis type pairs, Spearman 0.752, sign agreement
  0.951); weak-pair exception and columnar outputs so the reference task
  runs. `reports/2026-09-18_step1_data.md`, ISS-0001, ISS-0002, ISS-0005.
- **2, model zero** (2026-09-18): FlyVis dynamics on the MaleCNS right lobe
  with transplanted parameters; the transplant corrected to preserve total
  input per target with a cap (ISS-0003): DSI 0.152 (v9) against FlyVis's
  0.391, flash polarity and direction at or above FlyVis, stable on three
  members. `reports/2026-09-18_step2_model_zero.md`, §9 of the step-4 report.
- **4, the decoder ladder** (2026-09-18/19): ridge per cell type with the
  time-shuffle and sample-shuffle controls; the first stage curve withdrawn
  by its own audit (ISS-0004: the order depends on the decoder's window);
  even-spaced lags alias the frame hold (ISS-0006); closed at the minimal
  shape — ladders at 0 and 80 ms on both brains, one member, one split.
  `reports/2026-09-18_step4_decoder_stack.md`.
- **8, the generator by encoder inversion** (2026-09-19): a video optimised
  through the frozen network reproduces one stage's state; from every
  stage, including T4/T5 alone, the clip that caused the state comes back
  (r 0.92-1.00) and a state from another clip gives that other clip; on a
  T4 in minutes. Figures `reports/figures/2026-09-19_*generator*`.
- **9, the window's end constrained** (2026-09-19): the generator fits
  `frames + margin` (config [generate]: 40 + 5) and shows `frames`; the
  margin is the next chunk of the scene. The end-of-clip blur is gone
  (last-frame r T5a 0.66 → 0.92, T4+T5 0.80 → 1.00), mean r unchanged
  (0.93-1.00), whole 0.8 s clip. One T4 ladder, ≈ $0.17.
  `reports/2026-09-19_step9_window_margin.md`,
  `reports/figures/2026-09-19_malecns_generator_two_inputs_40f.gif`.
- **10', the ladder batched** (2026-09-19): all 20 tasks of a ladder in
  one pass, per-task masked loss; 948 s → 176 s on a T4 (7.5× on the
  optimisation), 66 % utilisation, videos identical to batch 1; a plateau
  stop with a floor. A ladder of one clip ≈ $0.05. §"Item 10'" of
  `reports/2026-09-19_step9_window_margin.md`.
- **11, dreams-lite** (2026-09-19): the generator on states no clip
  caused — noise into the eye, a flash, the dark after a clip, noise inside
  the neurons — from every stage beside a shuffled-state control, one
  batch per source on a T4 (≈ $0.15). Eye noise comes back (r 0.91-1.00,
  Tm5a 0.53, T5a 0.31; control ≈ 0); the flash by its timing without
  texture; a faint after-image for a few dark frames from L3 onward;
  internal noise as quiet ripple. The shuffled state is explained by
  diagonal stripes from Mi4/Tm9/T4a — the wiring's own texture.
  `reports/2026-09-19_step11_dreams_lite.md`,
  `reports/figures/2026-09-19_malecns_dream_*.gif`.
- **12, manipulated states** (2026-09-19): per-type normalised loss, 15
  tasks in one T4 batch (≈ $0.05). Gain edits of one type leave the clip
  (r_A 0.95-1.00) with a residual 2,000-8,000× the control's — unreachable
  states; the mix of T4a states is the state of the mixed video (r 0.99 at
  α = 0.5, monotone in α); hybrids are a contest that T4/T5 win
  (0.51/0.62 and 0.88/0.15). The three-source hybrid waits for a balanced
  readout. `reports/2026-09-19_step12_manipulated_states.md`,
  `reports/figures/2026-09-19_malecns_mix_{C,A,B}.gif`.
- **3, training priced** (2026-09-18): T4 smoke, packing, batch sweep and
  the Codex session's optimisation benchmark; the reference schedule
  (~$10-15 per member) is over budget and was withdrawn; what remains is
  3' below. `reports/2026-09-18_step3_training_options.md`,
  `reports/2026-09-18_training_optimization_bench.md`.

## Queue: the dreams and visual-data branch

One item at a time; the human's word starts each. Order: **13**; then the paused items of this branch when the human says so.

13. **Level C, a learned generator — in two layers** (the human and a
    second agent's review, 2026-09-19; the goal stays the human's: "видео
    ген модель, которая работает от состояния мозга, а не от шума + клип
    енкодеров"). A small deterministic network trained on reachable pairs
    is an *amortised inversion*, not a dreamer: outside the reachable set
    a pretty output proves nothing, so the state's contribution is
    measured before any prior is added.

    **13A, amortised inversion (deterministic).**
    - *Pairs* the frozen brain makes: video → state, video is the truth by
      construction. Videos: Sintel with flyvis's geometric augmentations
      **plus procedural stimuli** (moving edges, bars, gratings, dots,
      optic flow, noise, flashes, drifting textures, mixtures) — the
      count is unbounded, the diversity is the stimulus distribution, and
      20k rotated Sintel clips are still 23 scenes. 10-20k clips of 40 + 5
      frames, one T4 pass (≈ $0.3); split by scene and by stimulus class,
      whole classes held out.
    - *Three input conditions, one architecture:* `early` (L1 + L3), `deep`
      (T4a-d + T5a-d) — **the one that matters for the project** — and
      `all` (the ladder's 14 types) as the upper bound.
    - *Three models on the same data:* a **linear hex-temporal decoder**
      with shared weights (type × neighbouring columns × temporal taps →
      luminance at the column; not a flattened ridge), a small nonlinear
      hex + temporal CNN (T4, ≈ $0.3-0.5, the three conditions in one
      batch), and the item-8 Adam inversion as the ceiling. Answers
      whether a nonlinear learned inverse is needed at all.
    - *Scores.* Held-out clips: r to the video. States with no video (11,
      12): **the round trip** `state → generator → video → frozen brain →
      state′` and its compatibility error beside the *same* error of the
      Adam inversion (the best reachable answer); agreement with the
      inversion; stability across models and seeds; the shuffled-state
      control; how much the output depends on the state rather than the
      learned prior. The inversion's residual is kept as the signal
      "no input explains this state" and is never hidden by a picture.
    - *Deliverable:* per state, columns inversion / linear / CNN × the
      three conditions, r where a video exists and the round-trip error
      under every column.

    *13A status (2026-09-20), measured, complete:* data and the three
    conditions on held-out scenes and classes — deep: linear r 0.93 / round
    trip 0.036, CNN 0.96 / 0.029, inversion 1.00 / 0.009; the deep state
    reads out in one pass, the nonlinearity matters only there; the learned
    models sit hundredths of r below the inversion, with a larger round-trip
    error (the amortisation gap). The round trip on the 16 states of 11-12
    (rebuilt on the worker): on reachable states (clips, whole-state mixes,
    the dreams) the decoders' round trip is within 2-4× of the inversion's
    (deep CNN 0.006-0.022 vs 0.001-0.008) and 100-300× below the
    shuffled-state control (≈ 3); on unreachable states (gain edits, hybrids)
    every method's error is large and the decoders diverge from the
    inversion (deep, T4a × 0.5: linear 2.4, CNN 0.5, inversion 0.2).
    `reports/2026-09-19_step13a_amortised_inversion.md` §Round trip on 11-12.
    Open option, not a gate for 13B: more CNN epochs via `--resume`.

    **13B, the generative decoder** (approved 2026-09-20; design
    `reports/2026-09-19_step13b_design.md`, revision 2):
    `deep state (T4a-d + T5a-d) + type mask + z → conditional flow model →
    video → frozen brain → compatibility`. Only the 8 T4/T5 channels
    (early/all were 13A's controls); SiT-style linear interpolant (flow
    matching) as the objective, the backbone (hex-temporal ResNet vs SiT
    transformer over the 721 columns) chosen by a 200-step benchmark;
    structured type masks in training (full / T4 / T5 / one type / one
    direction / random subset / unconditional) for classifier-free guidance
    and "knobs"; all 40 frames at once; no brain-consistency term in the
    first version. Order, each run on its own word: (a) multimodality of the
    inversion from several random starts (≈ $0.05) — decides whether sample
    spread is a goal; (b) benchmark (≈ $0.04); (c) training ≤ $0.50, with
    the optimised loop the design lists; (d) samples + round trip (≈ $0.05).
    Scores: median round trip over seeds (main), best, spread, r to the
    clip; the conditioning-strength test (same z: true / shuffled / zero
    state) is mandatory; the inversion is the reference, not a target to
    beat. Tests: held-out clips, item-12 edits and mixes, single-type
    prompts, eye noise and neuron noise, shuffled-state control.

    *13B status (2026-09-20), measured:* (a) no multimodality at the full
    deep state (8 random starts → one solution); (b) SiT backbone, 0.057
    s/step; (c) SiT 128 × 4, 20k steps, val 0.0100, ≈ $0.22; (d) held-out
    clips: median round trip 0.031 / r 0.966 (13A CNN 0.029 / 0.960,
    inversion 0.009), control 0.96; knobs work (T4 only 0.033, T5 only
    0.112, T4a only 0.309 with seed spread appearing there); guidance > 1
    hurts; on gain edits the prior wins as with the CNN (0.29-1.47 vs
    inversion 0.19-0.86); the strength test passes (r 0.03 / 0.10).
    `reports/2026-09-20_step13b_generative_decoder.md`. Open options, none
    started: a brain-consistency term on x̂₁, a wider model / compile,
    prompts written by hand without a clip.

**Paused in this branch (the human, 2026-09-19: "не сейчас"):**

- **3', fine-tuning MaleCNS from the transplanted weights.** Not from
  scratch, not the reference schedule. Within $0.50 on Modal: two members ×
  1,200 iterations at batch 16 with `--variant stats_relu` (1-core
  container), checkpoints at 0/900/1200, validated like model zero (DSI,
  flash, direction) — enough to see whether training moves the biology;
  or 25,000 iterations for the same money on a cheaper provider (Vast.ai
  T4 ~$0.07/h) if the human opens one. Bought only if the generator on
  model zero turns out visibly worse than on FlyVis, or when the human
  wants a trained MaleCNS model. Commands and prices:
  `reports/2026-09-18_step3_training_options.md` §5б.
- **10, a dense MaleCNS export: every type on all 721 columns.** Types with
  fewer than 721 cells (15 of 33 output types; Tm5a 251 at [3,1]) tiled
  onto the empty columns with the type's shared filters, FlyVis's
  one-cell-per-column assumption in place of the data's count; removes the
  Tm5a/T5a lattice in the pictures; every artefact from it says which cells
  were tiled in. A day of local work; not a priority (cosmetic).
- **A 64×64 / 128×128 raster of the hexals** for the outputs (a rescale of
  the 721 values, no information added); the honest picture. Minutes.

## Beyond the branch (kept, not on the path)

Work that is not part of dreams and visual data. Nothing here is refuted or
cancelled; it waits for the human's word.

- **5, both eyes, every column, the missing biophysics.** ~880 columns per
  eye and the left eye; photoreceptor temporal filter with
  luminance-dependent speed, R1-6 gap junctions, contrast adaptation, each
  validated (Pang et al. 2024, Drews et al. 2020). Needed for HDRI / 360°
  input and two-eyed behaviour, not for clips.
- **6, the central brain and the state knobs** — the human's "как муха
  отреагировала бы на виз-данные": LC types, optic glomeruli, the central
  complex from MaleCNS in one model; trained on a behavioural task (looming,
  tracking) or fitted to whole-brain data (DANDI 000727); knobs: octopamine
  gain, R5 slow-wave gating, mean luminance. Gives reflexes and the drive
  for smarter dreams. A large item: new export, training beyond $0.50,
  validation. Reading for it: whole-brain spiking implementations (Shiu et
  al. 2024, community ports), `research_notes/`.
- **7, dreams, the full protocol** (the rigorous form of item 11): decoding
  in the dark against the stimulus history; internally generated activity
  with the fraction of variance in the stimulus subspace measured before
  anything is decoded and a shuffled-connectivity control; most-compatible
  stimulus with its compatibility score. The second paper's material.
- **Paper-grade rigour:** the ensemble map (ten members × five splits;
  `tools/map_local.py`, `deploy/modal/decode_app.py`,
  `flydream.decode.ensemble`), per-type nulls, subset curves, the
  consecutive-lag four-window sweep on more than one member, from-scratch
  training controls, validation against the 26 physiology studies, the
  reference training schedule. Code exists and is tested; measurements
  stopped 2026-09-18 night.
- **The connectome as a computational substrate** (an image generator or a
  language model made of fly wiring). Parked 2026-09-18 ("я бы начал не с
  LLM"); if revisited, a degree-preserving shuffled connectome must do
  measurably worse or nothing was shown.
- **14, colour as an ML task** (the human, 2026-09-19). FlyVis reads one
  luminance in all eight photoreceptor types and has no colour pathway;
  MaleCNS has the types (R7p/y, R8p/y, Dm8a/b, Dm9) but no parameters to
  transplant. Task: give the model a UV/blue/green input and learn the
  colour pathway's parameters on a task or a fit; until then colour in
  outputs is a display overlay from the source clip and says so. Weeks,
  training beyond $0.50; beyond the branch.
- **15, HDRI / 360° panoramas as the stimulus source and training data**
  (the human, 2026-09-19). A virtual fly camera inside a panorama (Poly
  Haven, CC0, thousands free) rendered onto the 721 ommatidia with its
  orientation as a parameter: unlimited clips with exact optic flow from
  rotations — the reference task's training signal, against Sintel's 23
  scenes. Translation needs scenes with depth (3D environments); HDR range
  needs item 5's luminance adaptation, otherwise the panorama is tone-mapped
  to 0..1. A day for the renderer; then the stimulus set for 11/12 and the
  data for 3'. Beyond the branch until the renderer exists, then it joins it.
- **16, the dream source inside the model** — reformulated by the human
  2026-09-20 ("источник мозговой активности внутри модели"; the one task
  after 13B): the brain must produce T4/T5 states with no video, 13B turns
  them into video. Design `reports/2026-09-20_step16_dream_source_design.md`
  (draft): 16.1 structured spontaneous drive into model zero as is; 16.2
  close the optic-lobe loops the export filter dropped (LPi, Dm, Pm, TmY16/19a,
  Y — 25 % of the synapses onto the ladder and 91 % of T4/T5's output go to
  types model zero lacks), stability and validation as gates; 16.3 central
  brain drive. Original note (2026-09-19) kept below.
  **Status 2026-09-20: not yet fully formed, not yet fully discussed** (the
  human). The human's revision `docs/ideas/step16_dream_source_revised.md`
  is the companion to the design report and takes precedence where they
  differ (parameters of the added types as ranges with a sensitivity sweep,
  sign from neurotransmitter annotations; loops ON/OFF and shuffled-topology
  controls; 16.1 is an artificial-drive baseline, not "the brain generated
  the state"; result levels A/B/C). Deferred; the discussion resumes on
  the human's word before any build.
  (the human, 2026-09-19: "чтобы сам мозг стал нейронной моделью-генератором";
  "вернёмся к ней позже"). Items 8-13 generate *beside* the brain: an
  optimiser or a separate network turns a state into a video. The
  alternative is inside it: the MaleCNS export carries top-down and
  between-layer feedback edges that model zero, shaped like FlyVis
  (feed-forward, input → output), does not use. Connect them, drive the
  deep types (T4/T5, or a manipulated state from item 12) and read the
  picture off the early layers (L1/L3 are the video to r 0.99 for the
  ridge decoder) — no second model, the wiring itself makes the image, as
  in the biological account of dreaming. Before building: count the
  feedback edges onto the ladder's types in the export and their weight;
  a network with feedback must be checked for stability at dt 0.02 (a
  spectral or a long-run test) before any picture is read. Deliverable:
  one clip beside the item-8 inversion of the same state and a
  shuffled-feedback control (same edges, permuted targets). Beyond the
  branch until 13A is measured.
- **One-liners, recorded only:** the three-connectome comparison (FlyVis,
  FlyWire, MaleCNS on one map); a second training objective; the embodied
  loop (the model as the eyes of a MuJoCo fly); inversion on real
  recordings (Pang et al. 2024); a MaleCNS LIF brain with a FlyVis front
  end; photoreceptor columns from lamina cartridges; the left optic lobe
  export; colour (R7/R8, Dm8/Dm9 — FlyVis reads one luminance; MaleCNS has
  the types); a 360° input path; the fitted-dynamics parameter source;
  ablations on model zero (CT1 restored, ISS-0002); per-target / per-pair
  filter normalisation (measured unstable at step 1; `"src"` stands).

## How this file is kept

- **Only approved work.** A conclusion the human has not approved in words is
  a draft and belongs in `reports/`, not here.
- **State and order, not reasoning.** No options, comparisons, prices or
  research; those go to `reports/`, durable choices to `DECISIONS.md`.
- **One entry per item, a few lines, plus links.** Evidence lives in the
  report it links to and is not summarized twice.
- **Done is a list of outcomes**, not a history of how they were reached.
- **Queue is an order, not a list.** Unfinished work returns as its own queue
  item instead of staying as a caveat inside a closed one.
- **Two groups outside the queue:** "Paused in this branch" for dreams /
  visual-data work the human stopped for now; "Beyond the branch" for
  everything else, kept so it is not lost.
- **Short beats complete.** If this file needs a table of contents, cut it.
