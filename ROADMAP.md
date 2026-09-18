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

**Current approved step:** 9, then 11 → 12 → 13 in order; each build starts
on the human's word.

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
  two-input figures of 2026-09-19 draw it. Known defect: the last 2-3
  frames of the window are constrained by a fitted margin (item 9, done).
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
- **3, training priced** (2026-09-18): T4 smoke, packing, batch sweep and
  the Codex session's optimisation benchmark; the reference schedule
  (~$10-15 per member) is over budget and was withdrawn; what remains is
  3' below. `reports/2026-09-18_step3_training_options.md`,
  `reports/2026-09-18_training_optimization_bench.md`.

## Queue: the dreams and visual-data branch

One item at a time; the human's word starts each. Order: **10' → 11 → 12 →
13**; then the paused items of this branch when the human says so.

10'. **The generator uses the card to the full.** (The human, 2026-09-19,
    while the item-9 ladder ran one task at a time: "точно нужно сделать".)
    All tasks of a ladder — every stage's inversion and its wrong-target
    control, 20 videos — are optimised in one batch through one simulation,
    the loss masked per task to its cells; a stop on plateau (the fit
    changes by less than a setting over 20 steps; R1 was flat from step 50
    of 150) in `config.toml [generate]`. Expected 4-8× on a T4 (batch 16
    saturated it in training); if memory is the limit with speed headroom
    left, an L4 (AGENTS "Human gates"). Measured before/after: seconds per
    ladder and the card's utilisation, in the item-9 report. The pictures
    must not change (same r within 0.01).

11. **Level A, dreams-lite: generation from states that no clip caused**
    (the human, 2026-09-19: "наконец-то мы говорим о том что я хотел").
    Input removed; the state comes from one of four sources, each its own
    clip beside a shuffled-state control (same numbers, permuted across
    cells or time): (a) noise into the eye (white, or 1/f like natural
    scenes — the choice is a setting and part of the result); (b) a flash or
    a slow drift; (c) the dark after a clip, the after-effect; (d)
    **internally generated activity** — the eye sees grey, noise is injected
    into the neurons' own dynamics (a noise term in the Euler step, all
    types or one type at a time, amplitude in `config.toml`), so the picture
    is from the wiring, not from the input's statistics. The item-8
    generator turns each state into a clip. MaleCNS model zero, a T4,
    minutes per clip. Deliverable: one stacked clip per source, input row
    and control row shown.

12. **Level B, manipulated states.** Mix two clips' states across stages (a
    face in L3, a forest in T4/T5), amplify one type, interpolate between
    two states; generate with the item-8 generator; each clip beside the
    unmanipulated states it was made from. MaleCNS, a T4.

13. **Level C, a learned generator.** A network "state → video" trained on
    pairs the model produces without limit, for one-pass generation from any
    state, with a generative prior for resolution if wanted; every output
    checked against the inversion of 11/12 for what is from the brain and
    what from the network. Price and shape proposed before it is built.

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
