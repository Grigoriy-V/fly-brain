# What Does a Fly Dream Of?

A model of the fruit fly's visual system wired by the **MaleCNS connectome**, and a
generator that turns the model's internal state into **video** — then an attempt to
make video that has no source clip at all, from a brain state drawn out of a learned
distribution.

```
video → fly eye (721 hexagonal columns) → optic-lobe model on the MaleCNS wiring
      → T4/T5 motion-detector state → conditional flow (SiT) → video
      → back through the same frozen model to check it

noise → flow on the hexagonal lattice → brain state → the same generator → new video
```

Everything here is a **model**, not a recording of a fly. A video recovered from a
state is the stimulus most compatible with that state in this model, not what a fly
sees; "dream" in the title is the project's question, not a claim.

The whole path, stage by stage, with every number and every mistake found on the
way, is in the master article:
[part 1, from the connectome to video from a brain state](reports/2026-09-19_master_article_draft.md) ·
[part 2, generating without a source clip](reports/2026-09-23_master_article_part2_draft.md)
(Russian for now; English versions follow).

## The pipeline

Every figure below opens with this strip and highlights its own stage.

### 1. The connectome, brought into a trainable network

MaleCNS gives individual neurons and synapses; FlyVis (Lappalainen et al., Nature
2024) is a trained network built on cell types on a hexagonal lattice. We
exported the right optic lobe into FlyVis's format: every reconstructed neuron gets
a home column from where its synapses sit (matching MaleCNS's own column tag in
96.9 % of 13,267 checked cells), and connections become
`source type → target type → column offset`. The model has 60 cell types, 31,526
units and 1.35 M edges; 61 of FlyVis's 65 types have a MaleCNS counterpart.

![The MaleCNS export: neurons on the eye's columns, type pairs against FlyVis](docs/figures/connectome_export.png)

### 2. The brain model — and the bug its first validation found

We transplanted FlyVis's trained parameters onto the MaleCNS wiring. The first
version kept ON/OFF polarity but its motion detectors were nearly blind to direction:
**DSI 0.020 against 0.391** for FlyVis. The cause was in our transfer, not in the
connectome: FlyVis's synaptic gain is normalised by the synapse counts of *its own*
connectome, so copying it onto a different wiring silently rescaled every input.
Rescaled by total input, with a revised export, the model reaches DSI 0.152 — a
working but weaker motion detector than FlyVis, with the OFF pathway (T5) as the
weak point.

What each layer does with a scene — photoreceptors copy it, lamina cells invert it,
the medulla splits it into ON and OFF, motion detectors keep only moving edges:

![Activity of eight cell types while the eye watches a clip](docs/figures/layers_activity.gif)

The same parameters on two wirings, compared by how well each layer's activity
gives the input back:

![One clip, two brains: FlyVis against the MaleCNS build](docs/figures/two_brains.gif)

### 3. Reading the picture back: decoding and inversion

Two ways to get the video back from one layer's activity, on the same brain, the same
clip and the same six layers. A **linear decoder** reads the picture perfectly from
the retina and the lamina and only partly from deeper types:

![A linear decoder per cell type](docs/figures/decoding_by_type.gif)

**Inversion** — freeze the model and optimise the input until the model reproduces
that layer's activity — recovers it from every layer (r 0.93–1.00). The control is
the same procedure aimed at another clip's activity: it returns that other clip.
The picture is still there in the deep layers; a linear readout just cannot get it.

![Inversion: the video recovered from each layer](docs/figures/inversion_by_layer.gif)

### 4. A video from a brain state

Inversion needs hundreds of optimisation steps per clip. **13B** is a conditional
flow model (SiT, 1.4 M parameters, trained for $0.22 on one T4) that maps the eight
T4/T5 types' activity plus noise to 40 frames in 20 steps: **r 0.966** to the source
video on held-out clips. Two noise seeds give the same video; with the same noise and
the cells shuffled, the video is gone (r between outputs 0.03) — the state, not the
generator's prior, makes the picture.

![13B: a video from a brain state](docs/figures/state_to_video.gif)

### 5. A brain state from noise

To make video without a source clip, a second generator has to learn the
distribution of brain states. It took several architectures to get there; the one
that works is a flow on the hexagonal lattice itself — patches of exactly three
columns by Kuhn matching, local attention, a shared relative position bias and
register tokens. Run backwards, it gives every clip its own seed, and that seed
brings the clip back (r 0.88, exactly the renderer's ceiling on these clips). A random
seed gives new video (nearest of 15,514 training clips r 0.49, where a real held-out
clip reads 0.52) with structure, not yet a scene.

![A clip's own seed, and a random one](docs/figures/seed_known_and_random.gif)

### 6. A static scene from noise

The current target is narrower: one still picture instead of a clip — 721 × 2
numbers, the static part of the T4 state, with the time axis removed from the task.
Through the same stand-in renderer, fresh draws match real states on the structural
judges (flat-field share 25.3 % against 25.3 %, neighbour coherence 0.92 against
0.94). Whether they are scenes is not proven yet: that needs a generative renderer
(the one used here is least squares and blurs by construction) and a scene metric
borrowed from image generation.

![Static draws beside real pictures and their states](docs/figures/static_draws.png)

## Metrics that lied, and how we found out

- The **round trip through the brain** — the project's first judge — gave a nearly
  blank grey field a better score than a real clip: compatibility with the brain is
  not content. The acceptance criterion became a clip judged against raw video.
- **Per-coordinate kurtosis** can be bought by collapsing the output toward the
  mean; it is now always read beside the radius.
- The flow first learned the second moment of the state distribution and not the
  fourth — and rotated points by 19° where a random rotation gives 90°, which is
  why its interpolations looked like double exposures.
- Every judge now has a measured floor, every result a control from the same run,
  and a pass that re-derived every number of the reports from the saved artefacts
  found and corrected nine discrepancies. Defects live in [ISSUES.md](ISSUES.md).

## Engineering: many experiments on one T4

The whole project cost about **$13** of cloud GPU.

- Independent tasks are packed into one batch axis until the card is full: T4
  utilisation 98–99.9 % on every training run, each run $0.05–0.51.
- **Batched inversion**: ten layers and ten controls optimised together instead of
  one after another, ~7× faster, videos identical to 7 × 10⁻⁴.
- **A fly-eye renderer 60× faster** than the reference implementation and
  bit-for-bit equal to FlyVis's own dataset (running sums read only at the ~31
  columns and ~61 rows that carry a receptor).
- **PCA of 13,555 × 92,288** states through the Gram matrix on the card: 43 s, $0.02.
- **Profiling before optimising**: the "launch-bound" hypothesis was refuted (2.4 ms
  of fixed cost in a 54.7 ms step); `torch.compile` took the step to 38.5 ms.
- **Exact flow inversion** by four fixed-point iterations per Euler step: error
  0.243 → 0.0058.

## Known limitations

- Model zero is a weaker motion detector than FlyVis, and its OFF pathway carries a
  spatial defect found by drawing each layer against FlyVis
  ([ISS-0015](ISSUES.md)); every state in the generator line comes from this model.
- Generation from noise produces structure, not yet scenes; the renderer for static
  states is a least-squares stand-in.
- The corpus is UCF101 through the fly's eye (721 columns, about 31 × 31), so every
  picture is small by construction.

## Running it

Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest -q
```

146 offline tests on a synthetic miniature connectome: no downloads, no cloud, no
credentials. MaleCNS is fetched through neuPrint (`NEUPRINT_TOKEN` in `.env`, see
`env.example`); GPU steps run on Modal. Step-by-step commands are in
[`docs/OPERATIONS_MAP.md`](docs/OPERATIONS_MAP.md), and every figure above is made
by a script in [`tools/`](tools/) named in its docstring.

## Layout

```text
flydream/        the package: data (MaleCNS export), model (model zero), decode, generate (generators, priors)
deploy/modal/    Modal functions on a T4: training, decoding, generation
tools/           run log, figure scripts
tests/           offline tests
docs/figures/    the figures of this README and the article
reports/         step reports, the article, runs.jsonl (one record per measured outcome)
research_notes/  the literature behind the research reports
ROADMAP.md       the current plan · DECISIONS.md why things are as they are · ISSUES.md defects found
```

Data, activity and checkpoints are not in the repository.

## Data and licences

Code: MIT ([LICENSE](LICENSE)). MaleCNS v1.0 (Janelia FlyEM) — CC-BY 4.0. FlyVis
1.2.0 — MIT. UCF101 — research use. Sintel — through FlyVis's dataset cache.

## References

- MaleCNS: https://male-cns.janelia.org/ ; Berg et al., bioRxiv 2025.10.09.680999, Cell 2026
- FlyVis: https://github.com/TuragaLab/flyvis ; Lappalainen et al., Nature 634:1132 (2024)
- Optic lobe inventory: Nern et al., Nature 641:1225 (2025)
- Encoder inversion: Bauer, Margrie & Clopath, eLife 105081 (2026)
- Layer-wise decodability: Chen et al., PLOS Comput Biol 20:e1012297 (2024)
- SiT: Ma et al., ECCV 2024 · flow matching: Lipman et al., ICLR 2023
- Decoding spontaneous activity: Horikawa & Kamitani, Science 340:639 (2013)
