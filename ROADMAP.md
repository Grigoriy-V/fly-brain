# Roadmap

**Updated:** 2026-09-19

**Project status (2026-09-19):** steps 1 (data), 2 (model zero on
MaleCNS), 4 (the decoder ladder) and 8 (the generator by encoder inversion)
are done at their minimal shape; the deliverable — a working generator of
clips from the model's internal states — exists on both brains as the check
"state → the clip that caused it". Current step: 9, generation from states
no clip caused. The research behind the plan is
`reports/Коннектом мухи и план проекта.md`.

**Change of course, 2026-09-18 night (the human: "да, меняй").** The human
is not writing a paper; the deliverable is a working generator of images
from internal states, first on FlyVis (trained, free), then on the MaleCNS
model zero. Paper-grade rigour (ensembles, several splits, from-scratch
controls, the 26-study validation, price calibration of training) is parked
below; training on MaleCNS is optional, only from the transplanted weights,
only within $0.50 per run, and only if the generator on model zero turns out
visibly worse than on FlyVis.

**Order (2026-09-19):** 9 (the end of the window constrained) → 11 (level
A, dreams-lite: four sources of a state no clip caused, each a stacked clip
beside its shuffled-state control) → 12 (level B, manipulated states) → 13
(level C, a learned one-pass generator checked against inversion). All on MaleCNS model
zero; a T4 for minutes per clip. **Later, not a priority:** a 64×64 /
128×128 raster of the hexals; item 10 (dense export); 3' (fine-tune within
$0.50) only if model zero's generator is visibly worse than FlyVis's.
**Parked:** ensembles, sweeps, the 26-study validation, item 7's full
protocol, items 5 and 6.

**Current approved step:** 9, then 11-13 in order (the human, 2026-09-19:
"ну пусть так будет"; the start of each build waits for the human's word). Items 4 and 8 are closed at their minimal shape: the decoder ladders
at 0 and 80 ms and the generator's "state → the clip that caused it" check
on both brains. The transplant is verified stable on three members (v9:
MaleCNS T4/T5 DSI 0.152 against FlyVis's 0.391; direction and flash
polarity at or above FlyVis) and is the MaleCNS model for everything below
until a fine-tune is bought.

Observed defects are in `ISSUES.md`, which is not a plan and authorizes
nothing. `docs/PROJECT_MAP.md` and `docs/OPERATIONS_MAP.md` describe the
system and operations; `AGENTS.md` holds execution rules; `DECISIONS.md`
preserves approved durable choices. This file alone owns current work, order
and authorization.

The training-optimisation benchmark of the Codex session (device-side
statistics, ReLU before the gather) is measured and closed: 1.17× on a T4,
equivalent within CUDA noise (`reports/2026-09-18_training_optimization_bench.md`);
`--variant stats_relu` is the setting any future training uses.

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
  on three members). Not trained on MaleCNS; training is optional (3').
- **Decoders:** `flydream/decode/`: ridge per cell type (penalty chosen on
  held-out scenes; float32 SVD), hex-conv head, metrics, raster, the map
  driver, the lag sweep, the ensemble aggregation (parked). The window is
  part of the measurement (ISS-0004), consecutive lags only (ISS-0006).
- **Generator:** `flydream/generate/invert.py`: encoder inversion — a
  20 × 721 video optimised through the frozen network to reproduce one
  stage's state (Adam, TV prior, grey start); `run_ladder` for all stages
  in one model load; `deploy/modal/generate_app.py` runs it on a T4
  (10 stages in ~10 min, ~$0.10); `flydream/generate/figures.py` and the
  two-input figures of 2026-09-19 draw it. Known defect: the last 2-3
  frames of the window are unconstrained and blur.
- **Compute:** the owner's machine (32 cores, 102 GB, CPU) for everything
  that fits in hours; Modal T4 only for a GPU-bound job or a ≥4× gain;
  training within $0.50 (DECISIONS 2026-09-18 night).
- **Measurement:** decoder ladders at 0 and 80 ms on both brains
  (`2026-09-19_decoder_ladders_flyvis_vs_malecns.gif`); the generator
  recovers clips 3 and 10 from every stage with r 0.92-1.00 on both brains
  (`2026-09-19_*_generator_two_inputs.*`,
  `2026-09-19_generator_forest_flyvis_vs_malecns.gif`). Tests: 64 offline.

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

**Closed 2026-09-19 at their minimal shape (moved here from the queue; the state lines inside say what was done and what was parked):**

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
   reads temporal-filter identity. Done since (night of 2026-09-18): the lag
   sweep with matched controls on the cached pairs, four windows 0 / 20 /
   80 / 160 ms (`flydream.decode.sweep`, report §10, ISS-0004): the frame
   component's order by stage depends on the window (Spearman −0.83 between
   a type's score at 0 ms and its gain to 160 ms), raw PixCorr peaks at 80
   ms for every stage but the lamina, so the map is reported at 80 ms with
   the 0 ms column beside it; the ladder regenerated and labelled with its
   window (`figures --lags`). **Closed 2026-09-19 at the minimal shape:**
   the decoder ladders at 0 and 80 ms (consecutive lags, ISS-0006) on
   FlyVis and on MaleCNS model zero, one member, one split
   (`reports/figures/2026-09-19_decoder_ladders_flyvis_vs_malecns.gif`).
   The ensemble map, the per-type null and the four-window sweep with
   consecutive lags are parked (the sweep's [0..4] and [0..8] windows were
   stopped mid-run and not repeated). Report:
   `reports/2026-09-18_step4_decoder_stack.md` (§9 the correction, §10 the
   aliased sweep).

8. **The generator: encoder inversion.** Given the state of one or several
   cell types at a moment, find the stimulus (721 hexals, then a 64×64 or
   128×128 image through the hex raster) that the encoder — the FlyVis
   network, later the MaleCNS model zero — maps closest to that state:
   gradient descent on the input through the differentiable network (Bauer
   et al. 2026), initialised from the ridge decoder's guess, with a
   smoothness prior; the ridge reconstruction is the baseline it must beat,
   the time-shuffle control the floor. Run first on CPU on short clips (the
   network's step is ~0.4 s per frame at batch 1 on this machine); moved to
   a T4 only if that is ≥4× faster for the batch that matters. Deliverable:
   the same ladder picture as item 4 with an "inversion" column per stage,
   a clip, and the compatibility score per frame; then the same on the
   MaleCNS model zero. Report: `reports/<date>_step8_generator.md`.

   State 2026-09-19: done as the check "state → the clip that caused it"
   on both brains (10 stages, r 0.92–1.00, clips 3 and 10, T4, ~$0.20;
   figures `2026-09-19_*_generator_two_inputs.*`,
   `2026-09-19_generator_forest_flyvis_vs_malecns.gif`). The generator is a
   video optimiser conditioned on a brain state (20 frames × 721 hexals at
   once, grey start, 150 Adam steps, TV prior); its known defect is the
   blurred last 2–3 frames of the window. The human (2026-09-19): this is
   verification, not yet generation; generation is what follows.


## Queue

One item at a time; the human's word starts each. Order as of 2026-09-19:
**9 → 11 → 12 → 13**; later, not a priority: the hexal raster at
64/128 px, 10, 3'; parked: 7 (full protocol), 5, 6, and item 3 as the
reference schedule. Items 4 and 8 are closed at their minimal shape. Item
bodies below keep their original text; the state lines say what of each
still applies.

9. **The end of the window is constrained.** The generator fits 25 frames
   and shows the first 20, so the last shown frames have a future that the
   recorded activity reflects; the item-8 defect (blurred frames 18-20)
   should disappear. Change in `flydream.generate.invert` (buffer length
   and the slice on save), one rerun of the MaleCNS ladder on a T4
   (~10 min, ~$0.10). Deliverable: the two-input clip of 2026-09-19 redrawn
   without the blur, first/middle/last frames checked before sending.

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

10. **A dense MaleCNS export: every type on all 721 columns** — recorded,
    **not a priority** (the human, 2026-09-19: "вернёмся к ней позже"; it
    replaces a fact of the data with FlyVis's one-cell-per-column
    assumption, so it is a cosmetic choice for the pictures). Every
    type with fewer than 721 cells in the right lobe (15 of the 33 output
    types, e.g. Tm5a 251 cells at [3,1], TmY15 [3,3], Tm30 [4,4]) is
    tiled onto the empty columns with the type's shared filters, the way
    FlyVis assumes one cell per column; the transplant renormalises total
    input per target cell; model zero is rebuilt and checked stable
    (flash, DSI); the decoder and generator ladders on MaleCNS are redrawn
    and the Tm5a/T5a lattice should be gone. Local CPU, about a day; the
    generator on a T4 for cents. Every artefact from the dense export says
    which cells were tiled in; the sparse export stays as the alternative
    (`config.toml [data]` setting, new export tag).


## Parked

Kept, not cancelled: work the 2026-09-18 change of course moved off the path,
with the reason, so a later reader can pick it up deliberately rather than
rediscover it. Nothing here is refuted; it is out of the way.

- **Paper-grade rigour (2026-09-18 night, the human: not a paper).** The
  ensemble map (ten members × five splits × windows, `tools/map_local.py`,
  `deploy/modal/decode_app.py`, `flydream.decode.ensemble`), per-type nulls,
  the random-subset curves on every run, the four-window sweep on more than
  one member, the from-scratch training control, validation against the 26
  physiology studies, and the reference schedule of training (250k
  iterations, ~$10-15 per member). All the code exists and is tested; the
  measurements are stopped. Picked up only on the human's word.
- **Training on MaleCNS as item 3.** Reduced to 3': an optional fine-tune
  from the transplanted weights within $0.50 (about 1,200 iterations at batch
  16 with `stats_relu` on a T4, `reports/2026-09-18_step3_training_options.md`
  §5б), or longer on a cheaper provider (Vast.ai T4 ~$0.07/h) if the human
  opens one. Bought only if the generator on model zero is visibly worse
  than on FlyVis.

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

**Moved out of the queue 2026-09-19 (the human: the generator is the goal; these stay as written, for later):**

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
   i.e. 30 h and about $18 per member alone on a card. Measured since (night
   of 2026-09-18, `reports/2026-09-18_step3_training_options.md` §4-5a):
   packing two members per card gains only 1.2-1.5× and is dominated by a
   larger batch; throughput saturates at batch 16 (14 samples/s, ~$12 per
   member for the reference's 10^6 samples), batch 64 and above cannot run
   (fewer than 64 training clips); two source-level changes to flyvis's
   step (activity statistics on the device, ReLU before the gather;
   `flydream/train/optimizations.py`, benchmark of the Codex session) give
   1.17× and are equivalent within CUDA's own noise, so a member at batch 16
   with `--variant stats_relu` costs ~$10 of GPU (~$15 with the container).
   The research report `reports/Обучение коннектомных сетей и ускорение.md`
   ranks the remaining levers (plateau stop, fp16, kernel fusion) with their
   scientific risk. **Next gate:** two members for the reference schedule
   (one from the transplant, one from scratch), batch 16, `stats_relu`,
   ~$30 for the pair; the human's word is pending.

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

   State 2026-09-18 (night): resequenced as "dreams, lite" right after item
   8: input removed, noise or a slow drive fed in, the generator of item 8
   run on the resulting states, each clip beside its noise-input control.
   The full three-experiment protocol stays parked.


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
