# Roadmap

**Updated:** 2026-09-18

**Project status:** steps 1 (data) and 2 (model zero) are built, measured
and reported; step 4 (the decodability map) is built and under correction
after an adversarial audit of 2026-09-18 (`ISSUES.md` ISS-0003 to ISS-0005).
The research report (`reports/Коннектом мухи и план проекта.md`, 2026-09-18)
surveyed the connectome datasets, FlyVis and the whole-brain simulators,
decoding methods and the biology of fly sleep. The human's word of 2026-09-18
fixed the shape: MaleCNS v1.0 is the canonical connectome from the first
step, FlyVis is the reference model and the source of the starting
parameters, training runs on Modal (`DECISIONS.md`, 2026-09-18). The human's
word of the same day fixed the goal: the generative inverse model is the
load-bearing deliverable, "что снится мухе" is the title, and the sleep
chapter is a second stage after it.

**Current approved step:** 4, the decodability map, in progress; its first
figure was withdrawn by its own audit and is being redone with a lag sweep,
several ensemble members and several splits (see item 4's state line). Next
in the order: 3 (training on Modal), on the human's word. The transplant is
verified stable on three members (v7, 2026-09-18: MaleCNS T4/T5 DSI 0.109
against the raw copy's 0.020 and FlyVis's 0.391; direction and flash
polarity at or above FlyVis); the remaining gap is attributed to ISS-0005
and ISS-0002, which are step-1 export work and come before the priced run,
so training does not pay to learn around a defect.

Observed defects are in `ISSUES.md`, which is not a plan and authorizes
nothing. `docs/PROJECT_MAP.md` and `docs/OPERATIONS_MAP.md` describe the
system and operations; `AGENTS.md` holds execution rules; `DECISIONS.md`
preserves approved durable choices. This file alone owns current work, order
and authorization.

**Bounded engineering work approved in the current chat (2026-09-18):**
prepare the training optimization benchmark (device-side activity statistics,
ReLU before edge gathering, actual-iteration timing and a profiler), with
offline equivalence checks. Implementation and 62 offline tests are ready;
one paired T4 measurement was explicitly approved and started in this chat
(app `ap-Fr0liJXeN4XNScXl9pn1zx`). No second worker is authorized.
This does not authorize full training
or change the scientific order above. Evidence and command:
`reports/2026-09-18_training_optimization_bench.md`.
Cross-agent handoff: `reports/2026-09-18_training_optimization_handoff.md`.

## Current state

- **Data:** the three MaleCNS v1.0 core Feather files and the FlyVis 1.2.0
  pretrained ensemble are under `data/` with hashes in `data/manifest.json`;
  neuPrint `male-cns:v1.0` reachable with the token in `.env`. The right
  optic lobe is exported as `data/ol/filters_R.json` (55 nodes, 1,234
  type-pair edges, FlyVis shape) and as per-neuron tables
  (`data/ol/neurons_R.parquet`, `edges_R_w5.parquet`); the type bridge is
  `data/bridge/types.csv` (61 of 65 FlyVis nodes matched). Settings in
  `config.toml [data]`.
- **Model:** model zero (`flydream/model/zero.py`): FlyVis's deep
  mechanistic network on the MaleCNS export with a pretrained member's
  parameters transplanted by type; 31,526 neurons, 829k edges at the
  density strides. Not trained on MaleCNS yet.
- **Decoders:** `flydream/decode/` (rung one of the ladder): paired stimulus
  and per-cell-type activity split by scene, ridge with the penalty tuned per
  cell type, PixCorr / R2 / SSIM on the lattice / identification, the
  hexagonal lattice as a raster, and a driver writing one row per cell type
  per control. Model and connectome are arguments (`--model malecns` runs the
  same map on the MaleCNS export). Settings in `config.toml [decode]`.
  Neither the hexagonal convolutional decoder nor encoder inversion is built.
- **Environment:** `.venv` by `uv` on Python 3.12, `pyproject.toml`; torch
  CPU locally, plus scipy and scikit-image for the raster metrics.
  `pytest -q`: 34 offline tests, on a synthetic miniature connectome and
  synthetic decoder data.
- **Compute:** the owner's Windows machine for data preparation and small
  runs; Modal for training, ensembles and inversion batches (the Modal
  patterns of `D:/ML/local-multimodal-agent/deploy/modal/` are the reference
  when that step comes).
- **Measurement:** model zero validated against FlyVis (flash polarity 0.911
  against 0.906; T4/T5 preferred direction within 45 degrees in 0.79 against
  0.83, but DSI 0.020 against 0.391), `reports/2026-09-18_step2_model_zero.md`.
  The decoder stack is checked but has no decodability result yet: on the
  moving-edge protocol the sample-shuffle control scores what the real fit
  scores (0.903 against 0.902, the control at least as high in 55 of 65
  types), because every condition runs one schedule and the frame index alone
  determines the stimulus. The map needs stimuli that are not synchronised to
  each other; see `reports/2026-09-18_step4_decoder_stack.md`. Targets stay
  FlyVis's (ON/OFF selectivity for 32 cell types, T4/T5 direction selectivity
  against the 26 studies Lappalainen et al. used) and the decodability metrics
  of the references (PixCorr, SSIM on the hexagonal lattice, identification,
  spatiotemporal correlation for video).

## Done

- **Research: the landscape and the plan** (2026-09-18).
  `reports/Коннектом мухи и план проекта.md`; notes in
  `research_notes/Коннектом мухи и план проекта/`.
- **The records** (2026-09-18): this file, `AGENTS.md`, `DECISIONS.md`,
  `ISSUES.md`, `docs/`, shaped after the owner's harness repository.
- **1, the data step** (2026-09-18): MaleCNS core files and the FlyVis
  ensemble fetched with a manifest; the type bridge (61/65; Mi3, Mi11, Mi12,
  Tm28 unmatched); home columns from neuPrint column ROIs for 50,906 of
  51,875 right-lobe neurons (0.969 exact against the 15 tagged types; Tm4
  0.529, ISS-0001); hex axes aligned to FlyVis by M = (−1, 0; 1, −1),
  confirmed by the T4/T5 input centroids; filters exported in FlyVis shape:
  0.869 of FlyVis type pairs recovered, Spearman 0.752, sign agreement
  0.951. Open: photoreceptor columns come from partners (no lamina column
  ROIs), the left lobe not exported, no synthetic fixture yet.
  `reports/2026-09-18_step1_data.md`.
- **2, model zero** (2026-09-18): FlyVis dynamics on the MaleCNS right lobe
  with members 000–002's parameters transplanted by type. Flash polarity
  0.911 against FlyVis's own 0.906 (30 and 32 known types × 3); T4/T5
  preferred directions within 45° in 0.79 of cases against 0.83, but the
  selectivity index 0.020 against 0.391; the shuffled-strength control
  diverged in every member. Established on the way: per-source filter
  normalisation is the only one stable under transplant; lattice strides
  from cell density (TmY4 runaway otherwise); FlyVis signs where the pair
  exists; CT1 absent (ISS-0002); datamate Windows and cache fixes.
  `reports/2026-09-18_step2_model_zero.md`.

## Queue

One item at a time; the human's word starts each. Order proposed 2026-09-18,
preliminary; 4 moved ahead of 3 on the human's word 2026-09-18
(`DECISIONS.md`, the decoder stack on FlyVis first).

4. **The decodability map.** For every cell type of the optic lobe, three
   decoders back to the 721-hexal rendered input: ridge, a hexagonal
   convolutional decoder, and encoder inversion (Bauer et al.) with the
   ensemble. Static frame and optic flow decoded separately. Controls:
   time-shuffled activity, random neuron subsets, quality against neuron
   count, the spread across the ensemble. Built and first reported on the
   pretrained FlyVis ensemble (validated, no training needed), with model
   zero on the MaleCNS export beside it as the connectome comparison; the
   same code runs on the trained MaleCNS ensemble when 3 lands.
   Deliverable: the curve R → L → Mi/Tm → T4/T5 with its controls; the
   first paper's figure.

   State 2026-09-18 (evening): ridge, the hex-conv head, the metrics, the
   raster, the driver and the figures are built and tested (47 tests); the
   wiring is confirmed (photoreceptors 1.000). The moving-edge protocol was
   ruled out by its own control. The first Sintel map (65 types, member 000,
   one split, lag 0) was produced and then withdrawn by the adversarial
   audit: its stage curve inverts under a 60 ms window (ISS-0004). What the
   map does show, and what survives, is that at lag 0 the frame component
   separates sustained from transient cell types with no overlap, i.e. it
   reads temporal-filter identity. Remaining for this item, in order: the lag
   sweep with matched controls on the cached pairs; ten members and five
   splits with per-type intervals; a per-type null in place of the floor;
   the figure regenerated and labelled with its window; then the same on
   MaleCNS (after ISS-0003 is verified stable), then inversion (rung three).
   Report: `reports/2026-09-18_step4_decoder_stack.md` (§9 carries the
   correction).

3. **Training on Modal.** The optic-flow task of Lappalainen et al. on
   Sintel, the same loss, schedule and augmentations, on the MaleCNS model;
   an ensemble sized by the budget (the human's word on the count). Validate
   against the 26-study targets. Deliverable: the trained ensemble on a
   Volume, the validation table, the cost.

   State 2026-09-18 (evening): the options report is written
   (`reports/2026-09-18_step3_training_options.md`), the Modal app exists
   (`deploy/modal/train_app.py`, never run), the transplanted init state is
   saved, and the export was changed so the reference task can run on it
   (inputs tile every column; outputs restricted to the 18 columnar types,
   `output_units_columnar_only`; weak-pair exception with a 500-synapse
   floor). The Modal account is the second one (profile `grigoriy98smile`);
   the GPU is T4, L4 the alternative, nothing above (the human,
   2026-09-18); the export, the init state and Sintel are on the Volume
   `flydream-data` (uploads approved the same day). The app packs N members
   per card (`smoke_packed`, `train_packed`). Gates in order: `smoke` on T4
   (minutes, under a dollar: the per-iteration price), `smoke_packed` at
   N = 2, 4, 8 (the packing factor), two members for the reference schedule
   (one from the transplant, one from scratch), then an ensemble sized by the
   human as `ceil(N / pack)` cards. Pre-conditions met 2026-09-18 evening:
   v9 (model zero on the refined export) is stable on three members (DSI
   0.200 / 0.191 / 0.065, flash 0.93); the T4 smoke measured 0.434 s/iter,
   i.e. 30 h and about $18 per member alone on a card; the packing factor
   is being measured (`smoke_packed` 2 / 4 / 8).

5. **Both eyes, every column, the missing biophysics.** Extend to ~880
   columns per eye and the left eye; add, one at a time and each validated
   (Pang et al. 2024 L1/L2 data, Drews et al. 2020 contrast curves): a
   photoreceptor temporal filter with luminance-dependent speed, R1–6 gap
   junctions, contrast adaptation. Repeat the map; report what each addition
   changes.

6. **The central brain and the state knobs.** Visual projection neurons and
   their targets from MaleCNS (LC types, the optic glomeruli, then the central
   complex) in one differentiable model, no graded-to-spiking seam. Training
   objective: a behavioural task (looming, object tracking) or a fit to
   whole-brain data (DANDI 000727). State knobs: octopamine gain on
   Mi1/Tm3/Mi4/Mi9 and T4/T5, R5 slow-wave gating on EPG, mean luminance.
   Deliverable: the map continued into the central brain; the modulation
   sweeps.

7. **Dreams.** Three experiments, each with its control, in this order:
   decoding in the dark (stimulus, input removed, decode the following
   seconds against the stimulus history); internally generated activity
   (input removed, noise or slow-wave drive to the central complex, the
   fraction of variance in the stimulus subspace measured before anything is
   decoded, shuffled-connectivity control); most-compatible stimulus by
   inversion with the compatibility score reported. Deliverable: the second
   paper's material, framed as "stimuli compatible with internally generated
   states".

## Parked

Kept, not cancelled: work the 2026-09-18 change of course moved off the path,
with the reason, so a later reader can pick it up deliberately rather than
rediscover it. Nothing here is refuted; it is out of the way.

- **The connectome as a computational substrate (an image generator or a
  language model made out of fly wiring).** Parked on the human's word
  2026-09-18: "я бы начал не с LLM". This was the first spark as the project
  agent had read it, and the reading was wrong. The human's "image generator"
  means the generative inverse model, which is the decoder ladder's third rung
  and now the load-bearing item, not the connectome used as a substrate for
  an unrelated task. If it is ever revisited, the control it needs is stated:
  a connectome shuffled to the same degree distribution must do measurably
  worse, or the experiment has shown nothing.
- **Sleep and spontaneous activity as the project's culmination.** Not parked,
  resequenced: the human confirmed 2026-09-18 that "что снится мухе" is a
  marketing title and the chapter is a second stage, after the generator
  works. The scientific content stays (decoding internally generated activity
  against the orthogonality control); only its position moved.
- **Per-target and per-pair filter normalisation** (`config.toml [data]
  normalisation`). Measured and set aside at step 1: per-target sent TmY15 to
  1.2e5 against FlyVis's 1.4, per-pair produced NaN. `"src"` stands. Kept
  because a different substrate may behave differently.

## Not started

Recorded, not approved, not begun. One line each.

- **The three-connectome comparison:** the same map on the FlyVis
  connectome, MaleCNS and the FlyWire optic lobe (Codex Visual Columns Map).
- **A second training objective:** an ensemble trained on input
  reconstruction and one on looming detection, to separate what the wiring
  determines from what the task does.
- **The embodied loop:** the model as the eyes of a MuJoCo fly (flybody or
  NeuroMechFly v2), so spontaneous activity is locomotor as in biology.
- **Inversion on real recordings:** most-compatible stimuli for the Pang et
  al. 2024 L1/L2 voltage data.
- **The substrate experiments:** moved to Parked 2026-09-18. They were
  recorded here as "the human's first spark", which misread it; see Parked.
- **A sanity run of a MaleCNS LIF brain with a FlyVis front end** (community
  code) to price the graded-to-spiking seam before item 6.
- **Whole-brain spiking implementations to reuse at item 6** (read
  2026-09-18, none trains or decodes anything; all are the Shiu et al. LIF
  model or a port of it): `chaobrain/drosophila_whole_brain_snn_simulation`
  (FlyWire v630 on JAX/BrainState, Apache-2.0) is the one worth testing for
  surrogate-gradient support, since item 6 needs one differentiable model
  rather than a graded-to-spiking seam; `eonsystemspbc/fly-brain` (FlyWire
  v783, six backends incl. Brian2CUDA/PyTorch/GeNN, cross-backend ground
  truth by Jaccard overlap and firing-rate correlation) is a ready
  benchmark harness but **GPL-2.0-or-later**, so its code cannot enter this
  repository without relicensing it; `philshiu/Drosophila_brain_model` (MIT)
  is the original; `testinganything/accurate-fly-brain` (MaleCNS v1.0 LIF
  with video injected into LC4/LPLC2 and other visual projection neurons)
  has no licence and no validation, and injecting at the projection-neuron
  level skips the optic-lobe computation this project exists to measure.
- **Photoreceptor columns from lamina cartridges:** no column ROIs exist for
  the lamina in MaleCNS; a cartridge assignment from synapse positions
  (`syn-points`, 12.7 GB, a gate) or from L1–L3 partners with an offset
  check.
- **The left optic lobe export** (`side = "L"`).
- **Colour.** All eight photoreceptor types in FlyVis read one luminance
  channel: measured 2026-09-18, their response patterns correlate with R1's
  at 0.86 to 1.00 and differ only by fitted gain and time constant, so there
  is no spectral information in the model at all and R1–R6 plus R8 come back
  from the map as near-duplicates at 1.000. Adding it means spectrally
  distinct R7/R8 channels, chromatic stimuli, and per-stage colour
  decodability. Raised by the human 2026-09-18; no prior connectome-
  constrained colour model is known to the project yet (being checked).
- **A 360-degree input path.** The 721-column lattice is 31 by 31 ommatidia
  at 5.8 degrees each, so it spans about 174 degrees: approximately one whole
  compound eye, not a patch. A flat video therefore fills only part of it,
  and flyvis's own pipeline box-filters a flat frame without spherical
  geometry. An equirectangular (HDRI or 360 video) sampler that maps
  ommatidium directions to panorama coordinates is the geometrically correct
  input, and becomes fully meaningful at item 5 with both eyes.
- **The fitted-dynamics parameter source.** A preprint reported to the
  project 2026-09-18 (August 2026, unverified) is said to fit synaptic
  strengths and time constants of a whole FlyWire model to real whole-brain
  calcium activity with connectome-fixed topology and signs. If it is real
  and its artefacts are usable, it is a better source of dynamics than
  transplanting from flyvis, which was fitted to an optic-flow task and not
  to recordings. Verification in progress before anything is planned on it.
- **Ablations on model zero** before or beside training: CT1 restored from
  neuPrint partners (ISS-0002), the 56 defaulted pairs, the four absent
  types, the filter-scale mismatch on wide-field TmY types; which of them
  carries the lost T4/T5 selectivity.
- **Protocols on GPU:** the flash and edge protocols run on Modal beside
  training, so a full ensemble is minutes, not hours.
- **Community projects worth reading before the steps they touch** (from
  `townie/awesome-fruit-fly`, read 2026-09-18; all hobby code, claims
  unverified, treat as leads not facts): `AbijahKaj` optic-lobe steering
  renders 1,771 column directions from both MaleCNS lobes — a cross-check
  for our column inference and for the left-lobe export;
  `oskarmalmwiklund/swat-or-buy` and `jerryjliu/fly_ocr` decode from a
  frozen MaleCNS (retina/lamina, and as a character reservoir) — the
  nearest prior art to items 4 and the substrate experiment;
  `suanmiao/fly-self-driving` trains a 165k-neuron / 25.6M-synapse MaleCNS
  network from pixels, so training at our scale is demonstrably feasible;
  `nftechie/doomfly` is the whole-brain-LIF-with-photoreceptor-input
  architecture and reports failing its own visual validation gates.

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
- **Short beats complete.** If this file needs a table of contents, cut it.
