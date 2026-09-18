# What Does a Fly Dream Of?

Russian: «Что снится мухе?»

A connectome-constrained model of the fruit fly visual system on the MaleCNS
connectome, a map of how much of the visual world stays decodable at each
stage of the fly's visual pathway, and a decoder that turns the model's
internal states back into the stimulus most compatible with them. The last
chapter asks what that decoder produces when the model runs with no visual
input. The title is a metaphor; the method is not.

The idea is [`what_does_a_fly_dream_of.md`](what_does_a_fly_dream_of.md). The
research behind the plan, in Russian, is
[`reports/Коннектом мухи и план проекта.md`](reports/Коннектом%20мухи%20и%20план%20проекта.md)
with its primary-source notes under `research_notes/`.

## The shape (2026-09-18)

- **Connectome:** MaleCNS v1.0 (Janelia FlyEM, male brain + both optic lobes
  + VNC, 166,691 neurons, CC-BY 4.0), from the first step. FlyWire is a
  cross-check on a female; BANC has no lamina and is not used.
- **Model class:** the deep mechanistic network of FlyVis (Lappalainen et
  al., Nature 2024): rate-based point neurons, connectivity and synaptic
  signs fixed by the connectome, a few hundred free parameters per cell type
  and type pair, trained on optic flow. FlyVis's pretrained ensemble is the
  initialisation and the validation target; its connectome (FIB-25 and
  FIB-19 tiled over 721 columns) is not the substrate.
- **Decoding:** ridge → hexagonal convolution → inversion of the model by
  gradient descent (Bauer et al., eLife 2026); every number beside a control.
- **Compute:** data preparation locally; training, ensembles and inversion
  batches on Modal.
- **Framing:** a decoded image is "the stimulus most compatible with this
  state under this encoder". Nothing is a percept.

## Layout

```text
what_does_a_fly_dream_of.md   the idea, as written
AGENTS.md / CLAUDE.md         how work is done here, by a person or an agent
ROADMAP.md                    the only plan
DECISIONS.md                  approved durable choices and why
ISSUES.md                     observed defects
docs/                         PROJECT_MAP (system shape), OPERATIONS_MAP (data, Modal, runs)
reports/                      evidence, dated; research reports; runs.jsonl
research_notes/               literature notes behind each research report
data/                         (not committed) connectome tables, stimuli, activity, checkpoints
flydream/                     the package: data (MaleCNS export), model (model zero), decode (the ladder, figures)
tools/                        run_log (the only writer of reports/runs.jsonl)
tests/                        47 offline tests on a synthetic miniature connectome and synthetic decoder data
deploy/modal/                 (to come at roadmap 3) Modal apps for training and batches
```

## Where things are decided

[`ROADMAP.md`](ROADMAP.md) is the only plan. [`DECISIONS.md`](DECISIONS.md)
holds the durable choices and why. [`ISSUES.md`](ISSUES.md) holds the defects,
observed, whether or not anyone means to fix them. [`AGENTS.md`](AGENTS.md)
is how work is done here.

## Key references

- MaleCNS: https://male-cns.janelia.org/ ; Berg et al., bioRxiv 2025.10.09.680999, Cell 2026
- FlyVis: https://github.com/TuragaLab/flyvis ; Lappalainen et al., Nature 634:1132 (2024)
- Optic lobe inventory: Nern et al., Nature 641:1225 (2025)
- Whole-brain LIF: Shiu et al., Nature 634:210 (2024)
- Encoder inversion: Bauer, Margrie & Clopath, eLife 105081 (2026)
- Layer-wise decodability: Chen et al., PLOS Comput Biol 20:e1012297 (2024)
- Decoding spontaneous activity: Horikawa & Kamitani, Science 340:639 (2013); Front Comput Neurosci 11:4 (2017)
- Fly sleep and vision: Raccuglia et al., Nature 2025; Van De Poll & van Swinderen, J Exp Biol 228 (2025)
