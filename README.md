# What Does a Fly Dream Of?

A model of the fruit fly's visual system wired by the **MaleCNS connectome**, and
generators that turn the model's internal state into **video** — ending in an
attempt to make video that no clip ever caused, from a brain state drawn out of a
learned distribution.

Everything here is a **model**, not a recording of a fly. A video recovered from a
state is the stimulus most compatible with that state in this model, not what a fly
sees; "dream" in the title is the project's question, not a claim.

The full story, stage by stage, with every number and every mistake found on the
way: [part 1, from the connectome to video from a brain state](reports/2026-09-19_master_article_draft.md) ·
[part 2, generating without a source clip](reports/2026-09-23_master_article_part2_draft.md)
(Russian for now; English versions follow).

## Goal

Take a real wiring diagram of a fly's brain and use it, not a generic network, as the
computational substrate of a visual model — then run the model backwards.

Two questions drive the project:

1. **How much of the visual world can be read back out of each stage of a
   connectome-constrained visual system?** From photoreceptors to motion
   detectors: where does the representation stop being a picture?
2. **What visual content is most compatible with an internal state that no
   stimulus caused?** If you can render any brain state as video, can you invent
   a new state and render that?

The deliverable is a working generator, built and measured as an ML system: every
result stands beside a control from the same run, and every claim is bounded by
what was measured.

## Tasks

| # | task | status |
|---|---|---|
| 1 | Export the MaleCNS optic lobe into a trainable network format | done |
| 2 | Run FlyVis's trained dynamics on that wiring and validate it ("model zero") | done, with a known defect |
| 3 | Map what is decodable at each stage of the visual pathway | done |
| 4 | Recover a video from a single layer's activity by inverting the frozen model | done |
| 5 | Amortise that inversion into a one-step generator (state → video) | done |
| 6 | Learn a distribution of brain states and draw new ones (noise → state) | structure yes, scenes not yet |
| 7 | Narrow the target to one still scene; build its renderer and a scene metric | in progress |

## Questions we had to answer along the way

**Can a connectome from one fly run the trained parameters of a model built on
another?** Yes, but not by copying. The first transplant kept ON/OFF polarity and
left the motion detectors blind to direction (DSI 0.020 against 0.391 for FlyVis).
The cause was ours: FlyVis's synaptic gain is divided by the synapse counts of its
own connectome, so copying it onto different wiring rescaled every input. Rescaled by
total input, the model reaches DSI 0.152 — working, and a weaker motion detector
than FlyVis.

**Is visual information lost as it goes deeper?** Our first "ladder" said yes, and
it was wrong twice. On a simple moving edge the shuffled control scored as high as
the real pairs — the protocol measured how repeatable the stimulus was. On natural
scenes, the ranking of layers flipped with the decoder's time window (T5a:
0.057 → 0.625), and one window aliased the frame rate of the movie. What is true: a
**linear** readout gets the picture whole from the retina and the lamina and only
partly from deeper types.

**Is the picture gone from the deep layers, or just not linearly readable?** Not
gone. Freezing the model and optimising the input until it reproduces one layer's
activity recovers the clip from every layer, motion detectors included
(r 0.93–1.00); the same procedure aimed at another clip's activity returns that other
clip.

**Can inversion be done in one step?** Yes. A conditional flow (SiT) maps the
T4/T5 state plus noise to 40 frames in 20 steps at r 0.966 on held-out clips. With the
same noise and the cells shuffled the video is gone (r between outputs 0.03): the
state, not the generator's prior, makes the picture.

**Does the generator obey any state you give it?** No. Random activity, permuted
direction channels and a hand-drawn stripe are largely ignored; spatial compositions
of real responses work. Run in a closed loop (state → video → brain → state …) each
step stays compatible while the content drifts away (r 0.99 → 0.26 in 11 passes).

**Can we invent a state no clip caused?** A flow over states learns their
structure — temporal and spatial correlations within a few hundredths of real ones —
and a clip's own seed, found by running the flow backwards, brings the clip back. A
random seed does not give a scene. The reason is geometric: in 92,288 dimensions,
Gaussian noise lives on a thin shell and the seeds of real clips sit 73 standard
deviations inside it. No seed on that shell gives a scene; the model has to learn the
map, not the points.

**Is it the size of the object or the shape of the model?** Shrinking the state
fourfold changed nothing. Keeping the hexagonal lattice in the architecture changed
every number at once (below).

**Are our metrics telling the truth?** Not always — see [Metrics that lied](#metrics-that-lied).

**Is there a still picture in a motion detector's state?** Yes. A single frame held
still for the whole window goes stimulus → brain → state → video at r 0.953, better
than a moving clip. "T4/T5 are motion detectors, so there is no static picture" is
false — which opened the current direction: one still scene instead of a clip.

## What the brain model looks like

Each panel is one cell type on the 721-column lattice while the eye watches a clip:
photoreceptors copy the scene, lamina cells invert it, the medulla splits it into ON
and OFF, motion detectors keep only moving edges.

![Activity of eight cell types while the eye watches a clip](docs/figures/layers_activity.gif)

## Generating from noise

The flow that finally learned the state distribution works on the hexagonal lattice
itself. Run backwards, it gives every clip its own seed, and that seed brings the
clip back at r 0.88 — exactly the renderer's ceiling on these clips. A random seed
gives new video (its nearest of 15,514 training clips is at r 0.49, where a real
held-out clip reads 0.52) with structure, not yet a scene.

![A clip's own seed, and a random one](docs/figures/seed_known_and_random.gif)

The current target is one still scene: 721 × 2 numbers, the static part of the T4
state, with time removed from the task. Through the same stand-in renderer, fresh
draws match real states on structure (flat-field share 25.3 % against 25.3 %,
neighbour coherence 0.92 against 0.94). Whether they are scenes is not proven: the
renderer here is least squares and blurs by construction, and there is no scene
metric yet — both are the next steps.

![Static draws beside real pictures and their states](docs/figures/static_draws.png)

## What we built

### Connectome export

- **Home column per neuron** from where its synapses sit, not from a tag — matching
  MaleCNS's own tag in 96.9 % of 13,267 checked cells, and defined for cells the tag
  misses.
- **Axis alignment by search**: the map from MaleCNS's hexagonal axes to FlyVis's is
  chosen among 104 integer lattice transforms by how well the two connectomes'
  filters agree, and reported through the direction-selective inputs so a reader
  can see it, not only a score.
- **Filters as `type → type → column offset`**, the format FlyVis trains on; lattice
  stride per type from its real cell density, a separate rule for weak pairs, and
  outputs restricted to types with a columnar map. 60 types, 31,526 units, 1.35 M
  edges.

### Transplanting trained parameters

- Gains rescaled by **total input** rather than copied, with a cap on the correction,
  after the copy was found to rescale every input silently. Every export and
  transplant version stayed runnable, so each fix is measured against the previous
  one.

### Reading and inverting

- Per-type **ridge decoders** with two controls each (frames shuffled in time,
  samples shuffled), scene-level splits, and a hexagonal convolution decoder.
- **Encoder inversion** through the frozen recurrent network with a total-variation
  prior and an **end margin**: a frame's activity depends on what comes after it, so
  five extra frames are optimised and dropped (last frame of T5a 0.66 → 0.92).
- **Batched inversion**: twenty independent tasks — ten layers and their ten controls
  — in one batch axis with per-task masks and losses. ~7× faster, videos identical to
  7 × 10⁻⁴.

### Generators

- **13B**, a SiT conditional flow over the T4/T5 state with a **type mask as a native
  input**: "this type is not given" instead of "this type is zero" (T4 alone
  0.837 → 0.904).
- **A compact state**: temporal DCT, 16 of 40 coefficients, each z-scored; K chosen by
  measuring the round trip on real states (K = 8 costs 13×, K = 16 is the knee).
- **Exact flow inversion**: four fixed-point iterations per Euler step take the
  error from 0.243 to 0.0058, so every clip has a recoverable seed.
- **A hex-local flow**: tokens are patches of exactly three neighbouring columns,
  assigned by **Kuhn matching** (nearest-centre assignment gave patches of 1–5 and
  broke equivariance); attention restricted to the six sublattice neighbours; a
  relative position bias shared across positions; four register tokens as the
  global channel. Against the same flow over PCA: kurtosis 4.11 → 5.36 (of 8.31 in the
  data), radius overshoot 20 % → 6.4 %, transport 19° → 72°, sharpness 11 → 22 of 100.

### Data

- **Our own fly-eye renderer** for ordinary video, **60× faster** than the reference
  (running sums read only at the ~31 columns and ~61 rows that carry a receptor) and
  bit-for-bit equal to FlyVis's own dataset (r = 1.000000). Checking it caught a
  wrong temporal law: frames are held, not interpolated (0.998 otherwise).
- **15,514 UCF101 clips** through the fly's eye, split **by class**, so every
  generalisation number is on ten classes the models never saw.

### Compute

- **Local first**: data, measurement and all judging run on a CPU workstation; the
  cloud GPU (Modal, one T4) is used only for training and batched optimisation.
- **Full card or no card**: independent tasks are packed into one batch until the GPU
  is saturated — 98–99.9 % utilisation on every training run, $0.05–0.51 each, about
  **$13 for the whole project**.
- **Minimum container**: 1 CPU and 12 GB beside the T4, and a GPU function does GPU
  work only; rendering and data assembly run elsewhere.
- **Profile, then optimise**: the "launch-bound" hypothesis was refuted (2.4 ms of
  fixed cost in a 54.7 ms step); `torch.compile` took the step to 38.5 ms. For the
  brain model itself, the edge gather/scatter was profiled and trimmed (1.16×).
- **PCA of 13,555 × 92,288** states through the Gram matrix on the card: 43 s, $0.02.
- **One record per measured outcome** (`reports/runs.jsonl`) with its run id, cost and
  control, so any number in the article traces back to a run.

## Metrics that lied

- The **round trip through the brain** gave a nearly blank grey field a better score
  than a real clip: compatibility with the brain is not content. The acceptance
  criterion became a clip judged against raw video.
- **Per-coordinate kurtosis** can be bought by collapsing the output toward the mean;
  it is now always read beside the radius.
- The flow first learned the **second moment** of the state distribution and not the
  fourth, and rotated points by 19° where a random rotation gives 90° — which is why
  its interpolations looked like double exposures.
- A pass that re-derived every number of the reports from the saved artefacts found
  and corrected nine discrepancies. Every judge now has a measured floor and every
  result a control from the same run; defects live in [ISSUES.md](ISSUES.md).

## Known limitations

- Model zero is a weaker motion detector than FlyVis, and its OFF pathway carries a
  spatial defect found by drawing each layer against FlyVis
  ([ISS-0015](ISSUES.md)); every state in the generator line comes from this model.
- Generation from noise gives structure, not yet scenes; the static renderer is a
  least-squares stand-in.
- The eye has 721 columns (about 31 × 31), so every picture is small by construction.
- No autonomous source of activity inside the model yet — the "dream" proper.

## Running it

Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest -q
```

146 offline tests on a synthetic miniature connectome: no downloads, no cloud, no
credentials. MaleCNS is fetched through neuPrint (`NEUPRINT_TOKEN` in `.env`, see
`env.example`); GPU steps run on Modal. Commands are in
[`docs/OPERATIONS_MAP.md`](docs/OPERATIONS_MAP.md); every figure is made by a script
in [`tools/`](tools/) named in its docstring.

## Layout

```text
flydream/        the package: data (MaleCNS export), model (model zero), decode, generate (generators, priors)
deploy/modal/    Modal functions on a T4: training, decoding, generation
tools/           run log, figure scripts
tests/           offline tests
docs/figures/    figures of this README and the articles
reports/         step reports, the articles, runs.jsonl (one record per measured outcome)
research_notes/  the literature behind the research reports
ROADMAP.md       the plan · DECISIONS.md why things are as they are · ISSUES.md defects found
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
