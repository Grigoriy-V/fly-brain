# Decisions

Approved durable scientific, architectural and scope choices, and why they
were made. Not a roadmap, a current-state map or evidence: `ROADMAP.md` owns
work, `docs/` owns the current system, `reports/` owns measurements. Read an
entry when a map links it, when its reason matters, or when the choice is
being reconsidered. A decision is a draft until the human approves it in
words.

Entries are in date order. Each has Decision, Why, Consequences, and a
Supersedes line where one applies. A superseded entry stays, shortened, and
says what replaced it.

## Catalog

| Date | Decision | Standing |
|---|---|---|
| 2026-09-18 | MaleCNS v1.0 is the canonical connectome from the first step | standing |
| 2026-09-18 | FlyVis is the reference model and the source of the starting parameters, not the substrate | standing |
| 2026-09-18 | The model class is a rate-based deep mechanistic network with per-cell-type parameters | standing |
| 2026-09-18 | Training, ensembles and inversion batches run on Modal; nothing trains locally | standing |
| 2026-09-18 | Decoding climbs a ladder: ridge, hexagonal convolution, encoder inversion; diffusion only after | standing |
| 2026-09-18 | "Dream" is the project's name; a decoded image is the most compatible stimulus, reported beside its control | standing |
| 2026-09-18 | The order is data, model zero, training, decodability map, extensions, central brain, dreams | preliminary |
| 2026-09-18 | The records are shaped after the owner's harness repository | standing |
| 2026-09-18 | The project agent delegates on Opus and Sonnet, never Fable, within stated limits | standing |
| 2026-09-18 | The decoder stack is built and first reported on FlyVis; MaleCNS stays the canonical substrate | standing, reason corrected |
| 2026-09-18 | The generative inverse model is the deliverable; the sleep chapter is a second stage; the substrate spark is parked | standing |
| 2026-09-18 | A transplanted gain preserves total input per target cell, is capped, and a capped pair is an export defect | standing |
| 2026-09-18 | A decodability number is reported over a lag sweep, several ensemble members and several splits, against a per-type null | standing |

---

## 2026-09-18 — MaleCNS v1.0 is the canonical connectome from the first step

Decision (the human, 2026-09-18): every model in this project is built on
the MaleCNS v1.0 connectivity (Janelia FlyEM, `male-cns:v1.0`, CC-BY 4.0),
starting with the right optic lobe and growing to both eyes, the visual
projection neurons and the central brain. No prototype is built on the FlyVis
connectome to be ported later.

Why: MaleCNS is the only complete CNS with both optic lobes including the
lamina, explicit column coordinates, neurotransmitter predictions and a
cross-match to FlyWire for 97.5% of neurons; no peer-reviewed model of
activity exists on it, so the first result of the project is the difference
between the averaged connectome FlyVis was built from (FIB-25, FIB-19, tiled
over 721 columns) and an individual, complete one. Starting on FlyVis and
porting later would have meant two index spaces and a migration
(`reports/Коннектом мухи и план проекта.md` §2, §6, and the chat of
2026-09-18).

Consequences: every neuron is addressed by its MaleCNS `bodyId` and its
hexagonal column from day one; FlyWire is a cross-check on a female, never a
substrate; the caveats travel with the data (postsynaptic completion 42%,
column tags on 15 of ~282 optic-lobe types, released tables at confidence
0.5) and every report on MaleCNS names the synapse-weight threshold it used.

## 2026-09-18 — FlyVis is the reference model and the source of the starting parameters, not the substrate

Decision (the human, 2026-09-18): FlyVis (TuragaLab/flyvis 1.2.0, MIT) is
used for three things: its code (rendering, dynamics, training loop,
`LayerActivity`, the decoder head), its validated behaviour as the target
of comparison (ON/OFF selectivity for 32 types, T4/T5 direction selectivity),
and its pretrained ensemble as the initialisation of the MaleCNS model:
per-cell-type resting potentials and time constants and per-type-pair
synapse gains transplanted by cell-type name, medians for what has no match.

Why: MaleCNS gives wiring and no dynamics; the 734 free parameters of the
FlyVis class have to come from somewhere, and FlyVis's are attached to cell
types and type pairs that MaleCNS shares (55–59 of 65 by community tables).
The transplant gives a running model on the full connectome on the first day
and makes the first measurement the effect of the connectome alone.

Consequences: the type bridge is a first-class artefact of the data step
with its unmatched rows listed; the transplant is followed by retraining on
the same task before any decodability result is claimed; FlyVis's version and
ensemble name (`flow/0000`) are recorded with every run that used them.

## 2026-09-18 — The model class is a rate-based deep mechanistic network with per-cell-type parameters

Decision: neurons are point units with a linear membrane and a threshold-
linear output, synapses instantaneous with sign fixed by neurotransmitter
and strength shared per source-type/target-type pair times the measured
synapse count, as in Lappalainen et al. 2024. The central brain, when it
comes, joins the same differentiable model; there is no graded-to-spiking
seam between an optic-lobe model and a LIF brain.

Why: the class is validated on the fly optic lobe, differentiable end to end
(which encoder inversion needs), and small in free parameters, so a
connectome swap is measurable. Every community project that stitched a
graded FlyVis front end onto a spiking MaleCNS brain reports losing direction
selectivity at the seam (`reports/Коннектом мухи и план проекта.md` §3, §6).

Consequences: spiking is an experiment, not the substrate; additions (gap
junctions, adaptation, photoreceptor filters, neuromodulator gains) enter as
named terms of the same equations with a setting each, validated one at a
time (roadmap 5, 6).

## 2026-09-18 — Training, ensembles and inversion batches run on Modal; nothing trains locally

Decision (the human, 2026-09-18): the owner's machine prepares data and runs
short checks; every training run, ensemble simulation and inversion batch is
a Modal Function on a GPU, with data and checkpoints on Modal Volumes. The
patterns (Apps, Volumes, secrets from `.env`, scale-to-zero) are taken from
the owner's harness repository (`D:/ML/local-multimodal-agent/deploy/modal/`)
when the training step is reached.

Why: there is no local GPU sized for a 250k-iteration ensemble; the owner
already runs Modal in other projects and has the patterns.

Consequences: every priced run is a human gate, per action, with the price
stated first (`AGENTS.md`); `docs/OPERATIONS_MAP.md` names the Apps, Volumes
and how a run is read back; a run's identity is its date and config hash.

## 2026-09-18 — Decoding climbs a ladder: ridge, hexagonal convolution, encoder inversion; diffusion only after

Decision: every decodability number is first a ridge regression from a cell
type's activity to the 721-hexal rendered input; then a small convolutional
decoder on the hexagonal lattice; then inversion of the model itself by
gradient descent on the input with the ensemble as the prior. A diffusion or
transformer decoder is built only after the three agree on the shape of the
curve, and never alone.

Why: at single-neuron resolution linear decoders are within 0.01 of
nonlinear ones (retina, mouse V1), and the best video reconstruction from
neurons (Bauer et al. 2026) is encoder inversion; the model here is the
encoder, so inversion is exact up to ill-posedness. Semantic metrics (CLIP)
mean nothing for a fly; the metrics are PixCorr, SSIM on the lattice,
identification, spatiotemporal correlation.

Consequences: the target of every decoder is the rendered hexal input, not
the source pixels; every inversion reports its compatibility score
(correlation of re-predicted with target activity); the decodability map is
reported per decoder, not as one number.

## 2026-09-18 — "Dream" is the project's name; a decoded image is the most compatible stimulus, reported beside its control

Decision (the human, 2026-09-18): the project keeps its title. In every
report, figure and abstract, an image decoded from activity without a
stimulus is "the stimulus most compatible with this internal state under this
encoder". The dream experiments run in a fixed order with a control each:
decoding in the dark against the stimulus history; internally generated
activity with the fraction of variance in the stimulus subspace measured
first and a shuffled-connectivity control; inversion with the compatibility
score. Sleep in the model means removed or attenuated input, low octopamine
gain, slow-wave gating of central targets and centrifugal drive into the
optic lobe; any replay inside the optic lobe is labelled a hypothesis.

Why: no experiment shows visual replay in fly optic lobes during sleep;
optic-lobe responses are unchanged in sleep and gating sits in the central
brain (Van De Poll & van Swinderen 2025); a feed-forward optic-lobe model
with its input removed decays to rest. Spontaneous activity in mouse V1 is
nearly orthogonal to the stimulus subspace (Stringer et al. 2019), so a
decoder applied blindly draws a picture from anything.

Consequences: roadmap 7 is last and depends on 6; the first spark (an image
generator from brain states) is this chapter, the substrate and LLM sparks
are separate experiments in "Not started" with their own control.

## 2026-09-18 — The order is data, model zero, training, decodability map, extensions, central brain, dreams

Decision (the human, 2026-09-18, preliminary): the queue of `ROADMAP.md`
in that order, each step delivering a number beside its control before the
next starts.

Why: each step is publishable on its own (the connectome-swap effect, the
layer-wise map of an insect visual system, the modulation sweeps) and none
is blocked by the one after it; the dreams depend on a central brain that
exists and is validated.

Consequences: roadmap items 1–7; the order can be changed by the human's
word, in the roadmap, with the reason here.

## 2026-09-18 — The records are shaped after the owner's harness repository

Decision (the human, 2026-09-18): `AGENTS.md`, `CLAUDE.md`, `ROADMAP.md`,
`DECISIONS.md`, `ISSUES.md`, `docs/` and `reports/` follow the rules and
structure of `D:/ML/local-multimodal-agent`, adapted to a research project:
a run log instead of a work log, a measurement beside its control as the
unit of done, priced Modal runs as the human gate.

Why: the rules there were arrived at over a month of work with agents and
hold the same failure modes (claims without evidence, drafts becoming rules,
priced runs started on an implied yes).

Consequences: canonical documents in English, reports in the language they
were requested in; `.claude/settings.json` as a mechanical backstop, not a
source of rules.

## 2026-09-18 — The decoder stack is built and first reported on FlyVis; MaleCNS stays the canonical substrate

Decision (the human, 2026-09-18, after step 2): roadmap item 4 moves ahead
of item 3. The decoders and the inversion are built model-independently and
their first numbers come from the pretrained FlyVis ensemble, which needs no
training; model zero on the MaleCNS export runs beside it as the connectome
comparison. Training the MaleCNS ensemble on Modal (item 3) follows, and the
same code then produces the map on it. MaleCNS remains the canonical
connectome of the project (2026-09-18, unchanged); this is an order, not a
change of substrate.

Why: step 2 measured T4/T5 selectivity of 0.020 on the transplanted MaleCNS
model against 0.391 on FlyVis, so a decoder read from model zero's motion
pathway would be reading near-silence and could not be told from a broken
decoder. The 50 FlyVis members are validated against 26 studies and already
on disk, so the stack is debugged against activity known to be meaningful,
and the FlyVis-versus-MaleCNS comparison on the same map comes for free.
Nothing from items 1 and 2 is discarded: the export, the bridge, the columns
and model zero are all inputs to this item.

Correction (2026-09-18, later the same day, after the audit): the number
0.020 was at least partly a defect in the transplant, not a property of
MaleCNS (`ISSUES.md` ISS-0003): the gain copied across connectomes is
defined relative to each connectome's own synapse counts, and once rescaled
member 000 gives 0.161 with a better preferred-direction error than FlyVis
itself. The order stands, for the reason that survives: a decoder is still
best debugged against activity validated against recordings, and the FlyVis
members are that. The reason as first recorded does not stand and is not to
be cited.

Consequences: `flydream/decode/` takes a model and a connectome as
arguments, never a hard-coded one; every decodability figure names which
model produced it; the extension of the map into the central brain stays
item 6's deliverable, not this item's.

## 2026-09-18 — The generative inverse model is the deliverable; the sleep chapter is a second stage; the substrate spark is parked

Decision (the human, 2026-09-18, in words): "«что снится мухе» — это
маркетинговое название"; the task is to extract visual data from the brain
at each stage, understand what of the world survives there, and turn the
internal state back into an image — "изображение, которое generative inverse
model считает наиболее совместимым с текущим внутренним состоянием её
мозга". Dreams stay ("если мы можем реально получить реальные сны — это
интересно… можно вторым этапом"); the connectome-as-substrate idea is not
the goal ("я бы начал не с LLM").

Why: the project agent had read the first spark as "use the connectome as a
computational substrate" and rated it the weakest leg; the human's
description names the inverse model (rung three of the decoder ladder)
instead, which the 2026-09-18 API scouting verified works on the stock
package. Colour and the full ladder into the central brain both require the
full connectome, which reinforces the substrate choice.

Consequences: roadmap order becomes map → inversion and generator → training
on Modal → extensions (colour, 360° input) → central brain → spontaneous
activity; the substrate spark moves to Parked with its control named; visual
output (figures, clips) is part of every run, not a separate step; the
generator ships with two mandatory controls (mismatched conditioning, and
re-encoding the output through the model for a compatibility score) and its
scored deliverable is the 721-hexal reconstruction, any larger render being
labelled a visualisation.

## 2026-09-18 — A transplanted gain preserves total input per target cell, is capped, and a capped pair is an export defect

Decision: `zero.transplant` moves FlyVis's `syn_strength` onto another
connectome as `syn_strength × total_src / total_dst`, where `total` is the
synapses a target cell receives from the source type (`zero.total_n_syn`),
bounded to `[1/cap, cap]` by `config.toml [model] rescale_cap`; pairs that hit
the cap are written to the run and recorded as defects of the export, never
corrected by gain.

Why: FlyVis defines the parameter as `scale / <n_syn>` for the pair in its own
connectome and forms the weight as `sign × n_syn × syn_strength`, so the raw
number is a gain relative to that connectome's counts (`ISSUES.md`
ISS-0003). The mean per edge is the wrong basis because the two connectomes
spread a pair over different numbers of offsets (Tm9→T5a: 4.1× by mean, 1.7×
by total). Uncapped, the total basis multiplies the lamina pairs of ISS-0005
by up to 34 and two of three members diverge.

Consequences: step 2's numbers are re-measured (v7 run); a capped pair is a
task for step 1, not a knob; `--no-rescale` reproduces the old behaviour for
comparison and nothing else.

## 2026-09-18 — A decodability number is reported over a lag sweep, several ensemble members and several splits, against a per-type null

Decision: no per-cell-type decodability figure is written into a report
unless it is (a) measured over a sweep of the decoder's temporal window with
the controls refit at the same window, (b) taken from at least ten ensemble
members with a per-type interval, (c) averaged over several scene splits,
and (d) compared to a per-type null (the time shuffle), the sample shuffle
being reported as what it is: the training-mean image's score.

Why: the first Sintel map's stage curve inverted under one config line
(`ISSUES.md` ISS-0004); the member it was read from is the best of 50 by
validation loss, the split was the worst of six for the two types at the
bottom, and the "floor" was the same number for every type.

Consequences: `config.toml [decode]` gains `lags` as a list to sweep,
`members` and `splits`; the ladder figure is regenerated from the sweep and
labelled with its window; the first figure of 2026-09-18 is kept in
`reports/figures/` with its caveat and is not cited.

## 2026-09-18 — The project agent delegates on Opus and Sonnet, never Fable, within stated limits

Decision (the human, 2026-09-18): the project agent chooses, per task,
which model a subagent runs on and how many run at once; subagents run on
Opus or Sonnet by the task's weight and never on Fable unless the human
names it for that task. Research, reading, checks and blind judgement are
delegated; records, priced runs, decisions and the answer to the human are
not. One extra round per step, six researchers and one writer per research
request, cost recorded. The rule itself: `AGENTS.md`, Delegation.

Why: the human works in Fable and does not want its cost multiplied across
subagents; the first research request of the project ran five researchers
and a writer (~630k subagent tokens) with no rule saying what they may do.

Consequences: the `Agent` calls name a model; a subagent's output is data
checked by the project agent; the step's report carries the delegation's
cost.
