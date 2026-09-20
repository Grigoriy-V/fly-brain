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
| 2026-09-20 | The state prior moves to a VAE with a 2,048-dimensional latent; the seed becomes drawable by construction | standing |
| 2026-09-18 | Training runs on T4 (L4 the alternative, nothing above), packed several members per card; every changed result ships with a picture | standing |
| 2026-09-20 | The next stage is new video from a sampled brain state: a prior over reachable T4/T5 states, gated by the round trip and by novelty of both state and video (the free unconditional baseline measured first) | standing; sub-step design is the agent's |
| 2026-09-20 | `AGENTS.md` holds rules only; artefact rules live in `docs/ARTEFACTS.md`; `docs/ideas/` is the human's input, not a plan | standing |
| 2026-09-20 | The $0.50 cap on a training run is lifted; the prior's training set is rebuilt from ordinary video, and 13B may be retrained on it | standing |

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

## 2026-09-18 — Training runs on T4, packed several members per card; every changed result ships with a picture

Decision (the human, 2026-09-18, in words): "берём T4, L4 держим как
альтернативу, выше нельзя"; "надо подумать как нагрузить карту максимально
эффективно, параллельные запуски"; uploads to Modal Volumes approved
("можешь грузить всё на модал"); and the standing rule "после каждого
значимого шага, когда есть изменения, я хочу видеть визуальные данные".

Why: the model is small (31,526 nodes, 1.4M edges, about a gigabyte of graph
per iteration) and sequential over 40 time steps, so a single member leaves a
card mostly idle and a bigger card buys little; the cost that matters is
dollars per member-iteration, which packing lowers and which `smoke_packed`
measures at N = 1, 2, 4, 8 rather than assumes. The picture rule: the
project's product is images and clips, and the human judges by them.

Consequences: `FLYDREAM_GPU` defaults to `T4`; `deploy/modal/train_app.py`
runs N members as N processes in one container after a warm-up that builds
the shared caches; the ensemble is scheduled as `ceil(N / pack)` cards; every
step that changes a result ends with a figure or clip sent to the human in
the same turn (`flydream.decode.figures`, `flydream.decode.sweep`, a
loss-and-validation figure for training); priced calls still need the
human's yes per action with the price stated.

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

## 2026-09-18 — Modal is used only for a measured ≥4× speed-up or for a GPU; training is priced in single dollars

Decision (the human, 2026-09-18, night): "если доп скорости нет, модал не
надо использовать, модал используем если есть х4 или лучше х6 ускорение,
либо если надо гпу. Сразу скажу никаких $30 на обучение будет и быть не
может." Modal runs a job only when it needs a GPU or when it is at least
four times (better six) faster wall-clock than the owner's machine (32
cores, 102 GB); a training proposal costs a few dollars at most.

Why: the pilot of the ensemble map on Modal priced 100 CPU-bound ridge jobs
at ~$30 for a wall-clock gain that the local machine gives by running
several map processes at once; the reference schedule at ~$10–15 per
member is over the project's budget for a single run.

Consequences: the decodability map (ten members, five splits, two windows)
runs locally with `tools/map_local.py` (N processes at once); member
simulation stays local too (624 s per member on CPU against ~2 min plus
transfer on a T4, no ≥4× gain after moving 1.3 GB per member back);
`deploy/modal/decode_app.py` is kept for a GPU-bound case only. Training
options are re-priced under single-digit dollars: a short schedule from the
transplant with checkpoints and the plateau read off the validation loss
(`reports/2026-09-18_step3_training_options.md`); the two-member reference
schedule is withdrawn as a proposal. Supersedes the blanket "decode on Modal"
line of the same day in `docs/OPERATIONS_MAP.md`.

## 2026-09-18 — The deliverable is a working generator, not a paper; rigour is parked, training is optional

Decision (the human, 2026-09-18, night; "да, меняй"): the human is an ML
practitioner, not a scientist, and the goal is a generator of images from the
model's internal states ("что снится мухе" as the title), not a publication.
The order becomes: the map at its minimal shape (one member, one split, two
windows) → the generator by encoder inversion on FlyVis → the same on the
MaleCNS model zero → dreams, lite → an optional fine-tune from the
transplanted weights within $0.50, only if model zero's generator is visibly
worse. Ensembles, several splits, from-scratch controls, the 26-study
validation and the reference training schedule are parked.

Why: the day of 2026-09-18 went into price calibration (three smokes, packing,
batch sweep, a benchmark), a four-window sweep and an ensemble map — all
required by a paper-shaped contract and none of it moving the generator; the
human: "мне кажется ты тратишь мои время и деньги".

Consequences: `AGENTS.md` "Project" and "Primary principle" rewritten (one
time-shuffle control per picture, no ensemble gates, references as a source
of methods, not a standard); `ROADMAP.md` reordered with item 8 (the
generator) as the current step and a "Parked" entry listing the rigour;
local CPU is the default, Modal only for a GPU or a ≥4× gain, a training run
within $0.50 and only from the transplant; the running ensemble map was
stopped, the consecutive-lag sweep of member 000 finishes item 4.

## 2026-09-20 — The next stage is new video from a sampled brain state: a prior over reachable T4/T5 states, gated by the round trip and by novelty of both state and video

**Decided by the human** (2026-09-20, in words: "я ставлю чёткую задачу: я
хочу получать новые видео"; design documents
`docs/ideas/brain_state_prior_new_video_generation.md` and
`docs/ideas/full_project_architecture_brain_to_video.md`): the current track is
to generate video **without a source clip**, through brain states the project
generates rather than reads, and, long term, to use the neural-state space as
the interface through which deeper and internal brain activity will drive the
generator:
`noise → state prior → T4/T5 state → 13B → video → frozen brain → round trip`.

**Designed by the agent** (a draft the human has not ruled on line by line; it
stands until they say otherwise): the order 17.0 → 17.4 of `ROADMAP.md`, with
the free unconditional-13B baseline measured **before** any training; a
flow-matching model over states on 13B's own interpolant and `SiTColumns`
backbone as the first prior, with an autoencoder plus a latent flow held as
the fallback (the choice of flow before VAE is the agent's, not the human's);
and the gates — the round trip of the sample, the nearest training **state**,
the nearest training **video**, diversity of both, and what motion the brain
reads back.

Why: item 14 measured that numbers written into the eight T4/T5 types by hand
land outside what the brain can reach (round trip 1.9-2.3 against 0.027 for a
clip in the same table), so states cannot simply be invented; what is missing
is a *learned* source of states. Generating from a state no single clip caused
already works (item 11's noise and flash states, 14.2's region compositions at
0.100-0.150), so the prior is not needed to make "a new video" as such — the
unconditional 13B branch already produces video from z alone, which is why
17.0 measures it first and for free. The prior earns its $0.22 by making the
state space itself samplable, interpolable and editable, and by being the
place a state from a deeper level or from the model's own activity can later
enter. It is also the cheapest such component available: the 9,468 states are
already on the volume, 13B and the frozen brain stay untouched, and the run
fits the $0.50 training cap.

Consequences: `ROADMAP.md` item 17 with its sub-steps and gates is the current
approved step, each priced run on the human's word; the round trip stays the
unit of evidence on this track (`docs/PROJECT_MAP.md`, Boundaries); novelty is
reported as a nearest-neighbour distance for the state **and** for the video,
never as an impression, and without the video distance a result is stated as
"generated without a source clip", not as "a video that exists in no clip"
(`AGENTS.md`, Primary principle); every control in a table is rebuilt in that
table's own code path, because the two existing shuffled-state controls (13B's
0.96 and item 14's 19.1) are not on one scale; "dream" language stays out of
this track, because the source of the state is an artificial prior. The dream
source inside the model (item 16) is a separate, deferred item; it *may* reuse
this prior to bring a spontaneous state onto the reachable manifold, but that
projection mechanism is not defined by an unconditional flow and has to be
designed and measured on its own (ROADMAP 17.3). Extends 2026-09-18 ("the
generative inverse model is the deliverable"): the deliverable now includes a
learned source of states, not only the inverse.

## 2026-09-20 — `AGENTS.md` holds rules only; artefact rules live in `docs/ARTEFACTS.md`; `docs/ideas/` is the human's input, not a plan

Decision (the human, 2026-09-20): `AGENTS.md` is the file of working rules and
carries the project's description in four lines at most; how a result is
delivered to the human moves out of it into `docs/ARTEFACTS.md`, which is
canonical and as binding; the compute-and-money rules are grouped in one
section instead of being spread over "How to work" and "Human gates"; and
`docs/ideas/` is where the human's own notes and drafts live — input to a
step, never a plan, and never rewritten by an agent.

Why: the rules file had grown a project summary, a reference list and the
artefact scheme, so a reader looking for a rule read three paragraphs of
context first, and the artefact scheme — the part that is corrected most often
— was buried inside it. Keeping the human's notes in `docs/ideas/` untouched
separates what the human wrote from what an agent concluded, which is the
distinction `DECISIONS.md` exists to protect.

Consequences: `AGENTS.md` shrank to rules with pointers; `docs/ARTEFACTS.md`
is new and is listed in `AGENTS.md` Context as always-read; references in code
and settings to the old section names were updated (`config.toml`,
`flydream/generate/invert.py`, `flydream/decode/ensemble.py`,
`tools/fig_generator_two_inputs.py`); `CLAUDE.md` and `README.md` list the new
document. No rule was weakened, added or removed in the move; the artefact
rules gained only the layout for a state that no video caused, which item 14
had already required in practice.

## 2026-09-20 — The $0.50 cap on a training run is lifted; the prior learns from ordinary video, and 13B may be retrained on it

Decision (the human, 2026-09-20, in words): "правило «$0.50 на обучение»
больше нет"; "можно закладывать переобучение 13б, но сначала проверить его на
новом приоре"; "датасет выбери сам, можешь скачивать"; "моих видео нет".

Three things follow. **(1)** A training run is no longer capped at $0.50 per
run. Nothing else about money changes: local is still the default, Modal is
still only for a GPU or a measured ≥4× gain, a GPU is still used to the full,
a function still asks for the minimum cpu and memory, and **every priced run
still needs the human's explicit permission before it starts, with what it
does, how long it takes, the price and the exact command stated first**. The
cap was a ceiling on the size of one run; it was never the permission.
**(2)** The prior's training set stops being 19 Sintel scenes: it is rebuilt
from ordinary video, with procedural stimuli kept as a minority for motion
coverage. **(3)** Retraining 13B on that set is allowed, but only after the
existing 13B is measured against the new prior — if it renders the new prior's
states at the round trip it reaches on Sintel states, it is not retrained.

Why: 17.1b measured the cause of the gap. The prior samples reach a round trip
of 0.142 where a real clip reaches 0.006 and the representation itself allows
0.009, and 17.3b's inversion showed real states sit at a noise radius of 1.07
and 1.29 where the prior's own draws sit at 1.000 ± 0.014 — the density is
next to the real states, not on them. 1,695 states of 19 scenes is the
smallest suspect, and the next one after it is capacity (width 192-256, ≈
$0.4-0.9 per run), which the old cap forbade outright.

Consequences: `ROADMAP.md` loses the cap wherever it was quoted as a live
constraint (17.1, the capacity option, 3', item 6) and gains the dataset step;
the prior is retrained on the new set and read against the same gates so the
new number is comparable with 0.142; the check of 13B against the new prior is
a gate of that step, and its outcome decides whether 13B is retrained.
`AGENTS.md` keeps the permission rule unchanged.

## 2026-09-20 — The state prior moves to a VAE with a 2,048-dimensional latent

**Decision.** The unconditional flow over 92,288-dimensional states is replaced
by a learned encoder and decoder with a 2,048-dimensional latent and a KL term
toward N(0, I). A flow over that latent may or may not be needed and is decided
during the build. Approved by the human, 2026-09-20 ("VAE на 2 048").

**Why.** Three measurements, all in
`reports/2026-09-20_the_seed_problem.md`:

1. **Flow matching never looks at the inverse direction.** Its loss contains no
   term evaluating where real data inverts to, which is why the preimages of
   real scenes sit at radius 252.4 (per-axis sd 0.833) while every draw lands
   on the shell at 303.8 — 73 standard deviations away. The instrument was
   checked: a state the prior itself made from a known standard draw inverts
   back to sd 1.001 at 0.0-0.4 % error. A KL term is precisely a term on that
   quantity.
2. **Pinning the pair by hand does not work (18.20).** One fixed noise per
   training clip for a whole run returns its own clip 0 times out of 8; 2.6 M
   parameters cannot hold 13,555 arbitrary point-to-point assignments, so the
   model falls back on the same average field. A learned encoder chooses the
   latent itself, which is learnable by construction. $0.36 settled this.
3. **2,048 dimensions carry enough (18.22).** PCA fitted on 2,400 states and
   measured on 600 held out, reconstructions rendered against the raw corpus
   clip: k = 2,048 reaches 0.890, i.e. 91.1 % of its own ceiling against 95.8 %
   for the working DCT K = 16 representation. This is the linear bound; a
   learned encoder beats it.

**Consequences.**
- If the KL does its work, **z is drawable from N(0, I) by construction**: the
  seed problem is dissolved rather than patched, and a separate prior over
  states may not be needed.
- The acceptance criterion does not change and overrides every metric: a drawn
  z, decoded, rendered, judged as a **clip against the raw corpus video** (the
  human, 2026-09-20). The gate, sparseness, neighbour correlation and pixel
  correlation have each now been shown to disagree with the picture.
- Closed by this decision and not to be reopened without new evidence: classes
  as a condition, guidance, noise scaling, radius normalisation, the shared
  preimage direction, K = 32, a fixed hand-assigned coupling, and buying
  dimension by DCT truncation below K = 16.
- Not committed: the encoder and decoder architecture over the hex lattice, the
  reconstruction/KL balance, and whether a flow over the latent is needed.
