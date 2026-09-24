# What Does a Fly Dream Of? Part 1. From the MaleCNS connectome to video from a brain state

*Grigoriy Voyakin, September 2026.* [Русская версия](2026-09-19_article_part1_ru.md).
Code, run records and every figure are in the
[fly-brain](https://github.com/Grigoriy-V/fly-brain) repository. Part 2,
[generation without a source clip](2026-09-23_article_part2_ru.md),
is in Russian for now.

## The project in brief

Two questions: how does a video signal pass through the fly's visual system,
and can the state of that system be used to **create video**? To answer them we:

- wired the **right optic lobe** of the MaleCNS connectome into the network
  architecture of FlyVis and ran FlyVis's dynamics and parameters on it;
- recorded the model's responses at different levels of the visual pathway;
- built four ways back from activity to video: a linear readout, optimisation
  of the input through the frozen model, a fast convolutional network, and a
  conditional generative model (SiT).

```
video → eye (721 hexagonal columns) → optic-lobe model on the MaleCNS wiring
      → state of a level (R, L, Mi/Tm, T4/T5)
      → linear readout / input inversion / CNN / SiT → video
      → another pass through the same model
```

**What is new here for an ML engineer:**

1. exporting a connectome into filters on a hexagonal lattice with explicit
   column geometry, and transferring trained parameters onto someone else's
   wiring - including the transfer bug we had to find;
2. a map of decodability by level in which controls twice overturned a
   neat conclusion;
3. video inversion through a frozen recurrent network, and batching it;
4. a conditional flow "brain state → video", and a check of where it obeys
   its condition and where it does not.

**Scope of the claims.** This is a model constrained by the MaleCNS wiring,
not a simulation of the whole nervous system and not a recording of a living
fly's brain. A video recovered from a state is the stimulus most compatible
with that state in this model, not what the fly sees. The "dream" in the
title is the project's question, not its result.

**One measure for the whole article.** Any video obtained from a state can be
run through the same frozen model again, and the state it produces compared
with the one requested. We call this error the **round trip**: the smaller
it is, the more compatible the video is with the state. The round trip
measures compatibility, not content, and below there are cases where it
misleads (sections 8 and 9).

## 1. Two descriptions of the brain had to be made compatible

MaleCNS gives connections between individual neurons; FlyVis gives a ready
dynamical architecture, a set of cell types and trained parameters. Putting
"the connectome into FlyVis" did not mean inserting a weight table. It meant
reconciling type names, the hexagonal coordinates of columns, the direction
of the axes, signs and the scale of inputs.

- **Types:** 61 of FlyVis's 65 node types have a match in MaleCNS.
- **Columns:** a neuron's home column is found from where its synapses sit.
  On 13,267 tagged cells of 15 types it agrees with MaleCNS's own column tag
  in 96.9 % of cases.
- **Filters:** connections are aggregated into `source type → target type →
  column offset`, the format FlyVis reads.
- **Density:** types with fewer cells than columns get a sparser lattice
  stride set by their density. Without it the wide-field TmY4 would sit in
  every column, get three times as many neighbours as it really has, and the
  loop TmY4 → TmY5a → T2a would drive the network to 10⁸.

![Fig. 1](../docs/figures/connectome_export.png)

*Fig. 1. Left: reconstructed MaleCNS neurons of three columnar types, placed
in their home columns. Right: the total input of each type pair in the FlyVis
connectome against the same pair in the MaleCNS export; pairs present in only
one connectome sit on the axes. Script `tools/fig_gh_connectome.py`.*

The differences between the connectomes did not go away. Of FlyVis's 571
type pairs, 496 are in the export (0.869). The rank correlation of their
total weights is 0.752 on the first export and 0.69 on the current one; the
signs agree in 95 % of pairs. Missing cells and pathways - above all CT1 and
part of the lamina - remain known gaps of the reconstruction.

The export changed twice afterwards: weak pairs got a rule of their own, and
the outputs of FlyVis's standard task were limited to types with a columnar
map. **The current model:** 60 types, 31,526 nodes on 721 columns, 1.35 M
edges. Nodes are the model's cells on the lattice, not reconstructed neurons:
the right lobe of MaleCNS itself has 51,875 of those.
[Data report](2026-09-18_step1_data.md) (in Russian).

## 2. "Model zero": the dynamics ran, and validation found our own bug

FlyVis's parameters were transferred by cell type and by type pair onto the
MaleCNS wiring. The check is FlyVis's standard protocols on three members of
its ensemble: the polarity of the response to a flash, and the direction
selectivity (DSI) of the T4/T5 motion detectors to a moving edge.

The first version kept ON/OFF polarity, but T4/T5 selectivity was almost
zero: **DSI 0.020 against FlyVis's 0.391.** The cause was in our own
transfer: FlyVis's `syn_strength` is a gain divided by the number of synapses
in *its* connectome. Copying the coefficient onto MaleCNS as is
underweighted T5's main drive and overweighted its inhibition.

| version | what changed | DSI, members 000 / 001 / 002 | mean DSI | mean direction error |
|---|---|---|---|---|
| v5 | direct copy | 0.034 / 0.026 / 0.000 | 0.020 | 29.9° |
| v7 | gain rescaled by the pair's total input, correction factor capped | 0.101 / 0.150 / 0.075 | 0.109 | 18.0° |
| **v9 (current)** | plus an exception for weak pairs in the export | 0.200 / 0.191 / 0.065 | **0.152** | 39.2° |
| FlyVis, same members | - | 0.547 / 0.344 / 0.281 | 0.391 | 31.7° |

![Fig. 2](../docs/figures/dsi_by_version.png)

*Fig. 2. T4/T5 direction selectivity on the MaleCNS wiring, transfer version
by version, beside the FlyVis member the parameters came from. Script
`tools/fig_gh_charts.py`, runs `2026-09-18_step2_zero_R_*`.*

All three members are stable and flash polarity is kept (0.922 against
FlyVis's 0.906). Against v7 the current version is a trade: selectivity is
up 40 %, and the mean error of the preferred direction has doubled.

**The ON pathway transferred; the OFF pathway did not.** On the best member,
001, the four T4 subtypes reach a DSI of 0.37 against FlyVis's 0.61, with a
direction error of 5.5° against 9.8°. T5 on members 001 and 002 is barely
selective (DSI 0.006-0.020), and on member 000 it is selective but points
49-144° away from the known direction; these cells make the mean error. To
be fair, FlyVis's own T5 on members 001 and 002 is also much weaker than its
T4.

**The OFF-pathway defect is visible by eye.** When we drew every type's
activity next to the same FlyVis network, it turned out that L3 in model zero
is a regular lattice of isolated, strongly hyperpolarised cells on an almost
flat field, Tm9 and T5a carry periodic diagonal stripes, and Tm1 responds
with the opposite sign. The ON pathway (R1, L1, Mi1) matches FlyVis. A flash
cannot see this: a full field is indifferent to a spatial pattern. The cause
is not established; the candidates are the sparse lattice stride from
section 1, the incomplete lamina, the capped transfer coefficients on lamina
pairs, and the signs of Tm1's inputs. Every state the generators use below
and in part 2 was taken from this model, so the T5 in them carries this
defect ([ISS-0015](../ISSUES.md)).

The lesson: a DSI of 0.020 could not be sold as a property of the MaleCNS
connectome - at least in part it was a scaling bug in the transfer, and what
remains is partly explained by a defect that only a picture found.
[Step 2 report](2026-09-18_step2_model_zero.md) (in Russian; the v7 and v9
table by subtype is its section 7), the cause of the first bug - ISS-0003 in
[ISSUES](../ISSUES.md).

## 3. What can be read back out of the different levels

![Fig. 3](../docs/figures/layers_activity.gif)

*Fig. 3. What the levels look like: the activity of eight cell types on the
721-column lattice while the eye watches a Sintel clip. Red is
depolarisation, blue is hyperpolarisation. This is the FlyVis reference
network (member 000): its OFF pathway is clean, and it is what model zero is
compared with in section 2. Script `tools/fig_gh_layers.py`.*

We recorded the activity of photoreceptors, lamina, medulla and the T4/T5
motion detectors. The first tool is ridge regression per cell type: the
brightness of 721 columns from the activity of one type, beside controls
that break the correspondence between activity and frame.

**The first protocol measured the wrong thing.** On a simple moving edge, a
control with shuffled conditions scored 0.903 against 0.902 for the correct
pairs: the protocol measured how repeatable the stimulus was, not the
information in the state. We moved to Sintel split by scene and built a
hexagonal convolutional decoder.

**The second "ladder" did not become a conclusion either.** The curve
"information is lost from the retina inwards" turned out to be a property of
the decoder's window. On the best member of the FlyVis ensemble the
control-corrected correlation of T5a rises from 0.057 at lag 0 to 0.625 with
a window of lags 0, 2, 4 frames - and the order of the levels flips. That
window itself carried an artefact: sparse lags fall on the same phase of
Sintel's frame hold (24 frames per second, raised to 50), and the
reconstruction alternates from frame to frame. With consecutive lags 0, 1
the same T5a gives 0.495.

![Fig. 4](../docs/figures/decoder_window.png)

*Fig. 4. How much of the frame a linear decoder reads from each stage beyond
a time-shuffled control, as a function of its window. FlyVis network,
member 000; windows of every second lag (80 and 160 ms) hit the frame-hold
artefact and are drawn as hollow points. Script `tools/fig_gh_charts.py`.*

![Fig. 5](../docs/figures/decoding_by_type.gif)

*Fig. 5. Linear readout of model zero: a ridge decoder per type on frames t
and t−1, fitted on Sintel scenes, r computed on held-out scenes. Bottom row -
the control: the same decoder fitted on activity shuffled in time. Script
`tools/fig_gh_decoding.py`, pairs `2026-09-18_decode_sintel_malecns_v9`.*

On model zero, with a window of consecutive lags, the picture is this: the
retina and lamina give the frame back whole (r 1.00), Mi1 0.99, Tm9 and T4a
0.83, T5a 0.71. The time-shuffled control sits at 0.53-0.63 for every type:
it knows the scene (what is where) but not the frame. So a linear readout of
the early types sees the frame with a margin of about 0.4, and of T5a with
only 0.12.

Hence a careful conclusion: **a linear readout does weaken with depth, but
that does not mean the picture is lost in the deep layers.** The next section
recovers it from every level. We do not claim a monotonic loss of visual
information. [Decoder report](2026-09-18_step4_decoder_stack.md) (in
Russian), ISS-0004 and ISS-0006.

## 4. From reading a state to video: input inversion

The model is frozen, a target state of the chosen level is set, and the input
video is optimised until another pass produces that state. This is **encoder
inversion**, not a trained video model.

![Fig. 6](../docs/figures/inversion_by_layer.gif)

*Fig. 6. Inversion of model zero. Top row: the video recovered from one
level's activity and its correlation with clip A. Bottom row - the control:
the same inversion aimed at the state of another clip, B, returns clip B.
Fixed 0..1 grey scale. Script `tools/fig_gh_inversion.py`, runs
`2026-09-19_malecns_invert_*_s3`.*

On one Sintel clip and ten levels - from R1 to T4+T5 - inversion returned
the clip that caused the state at r 0.93-1.00. The control is another clip's
state as the target: its result scores r −0.20 to −0.22 against the original
clip and 0.94-1.00 against its own clip on the levels shown. Inversion
returns what it is aimed at, not one and the same "average video".

The same inversion with the same settings on the FlyVis reference network
gives almost the same numbers; the difference shows only on T5a - 0.971 for
FlyVis against 0.933 for model zero, which is exactly the OFF pathway of
section 2.

![Fig. 7](../docs/figures/two_brains.gif)

*Fig. 7. One clip, two brains: the same inversion on the FlyVis connectome
(top) and on the MaleCNS build (bottom), the same parameters of member 000,
40 frames plus a 5-frame margin, 150 steps. Script
`tools/fig_gh_two_brains.py`.*

A recurrent network has no future context, so the last frames of a window
are recovered worse. A margin of five frames at the end of the window lifted
the last frame of T5a from 0.66 to 0.92, and of T4+T5 from 0.80 to 1.00. The
comparison is not clean: the window length changed together with the margin
(20 → 40 frames).

The exact statement of the result: we found a stimulus compatible with the
state of **this model** under this optimiser and regularisation. Later
(section 7) the same inversion on eight held-out clips gave r 0.999-1.000 for
three groups of types. [Report](2026-09-19_step9_window_margin.md), code
`flydream/generate/invert.py`.

## 5. Engineering: where a GPU is needed and how to fill it

**Training the MaleCNS model itself was not needed.** A training loop on the
full export takes about 17 s per iteration on a loaded CPU against
0.36-0.43 s on a T4 at batch 4. Fine-tuning is postponed: the generator does
not need it as long as model zero does not limit it.
[Training options report](2026-09-18_step3_training_options.md).

**Batched inversion is the main speed-up of this part.** Ten levels and ten
control targets used to be optimised one after another. We packed 20
independent tasks into one batch axis, each with its own cell mask and its
own loss.

| | one at a time | batched |
|---|---|---|
| time on a T4 | 948 s | 127 s (optimisation) / 176 s (whole ladder) |
| price per run | ≈ $0.17 | ≈ $0.05 |
| video difference from the sequential version | - | at most 0.0007 on a 0-1 range |

The speed-up is **7.1-7.5×** on the optimisation and **5.4-5.8×** on the
whole ladder; a range rather than one number because the timing of the
sequential run does not separate the optimisation from the whole ladder. The
video correlations agree to three decimals. This speeds up not training but
*optimising video through the brain*.
[Timing table](2026-09-19_step9_window_margin.md#item-10-the-ladder-batched-same-day).

![Fig. 8](../docs/figures/speedups.png)

*Fig. 8. Left: one inversion ladder on a T4, one task at a time and batched.
Right: the throughput of model zero's training step by batch size. Script
`tools/fig_gh_charts.py`, runs `2026-09-19_generate_malecns_s3_40f5_batch20`
and `2026-09-18_step3_smoke_batch_t4`.*

**Optimising the training step of the MaleCNS model**, in case it is ever
needed:

- two processes on one T4 gave 1.2-1.5×; four and eight did not survive
  warm-up;
- batch 16 raised throughput to ~14 samples/s, with a plateau after it;
- the profile showed the step is eaten by gather (20 %), scatter_add (12 %)
  and per-edge multiplications (20 %);
- moving diagnostic reductions onto the GPU and applying ReLU before the
  gather gave 1.164× and 1.172× in two paired repeats (1.136 → 0.973 s per
  iteration at batch 16).

The losses agree within tolerance; the final states of the decoder head
formally do not (a difference of up to 4.3e-4 where two reference runs
differ by 3.0e-4). Thirty iterations do not test convergence: this is a
measured code optimisation, not proof of equivalent training.
[Benchmark](2026-09-18_training_optimization_bench.md).

**The GPU rule.** Every GPU function does GPU work only and requests the
minimum CPU and memory; independent tasks are packed into one pass until
the card is full. This part of the project cost a few dollars of GPU time;
the engineering of part 2 - packing tasks up to 99 % T4 utilisation,
profiling the step, our own eye renderer - is collected in its section 12.

## 6. Inverting states that no ordinary clip caused

We fed the model noise into the eye, a flash, darkness after a clip, and
noise into the cells' activity, then inverted each level's response beside a
shuffled state.

- **Noise in the eye:** recovery depends on the level - from r 0.31 for T5a
  to 1.00 for R1 and Mi1; T4+T5 together reach 0.98. The second dip is Tm5a,
  0.53. The shuffled control is near zero.
- **Darkness after a clip:** some levels show a faint, brief trace (contrast
  at most 0.008), black by frame 20. There is no bright "dream picture".
- **Noise in the cells:** faint ripples.

![Fig. 9](../docs/figures/inversion_chart.png)

*Fig. 9. Inversion across all ten levels of model zero: a Sintel clip
(section 4) and white noise in the eye, each beside its control. Script
`tools/fig_gh_charts.py`, runs `2026-09-19_malecns_invert_*_s3` and
`2026-09-19_malecns_dream_eye_noise_*`.*

[Report](2026-09-19_step11_dreams_lite.md).

Then we edited states already recorded:

- **Amplifying one type.** T4a × 2 barely changes the video (r 0.99 to the
  original), but the round trip grows thousands of times (0.492 against
  6e-5). The "T4a knob" cannot be turned while keeping the rest.
- **Mixing the states of two clips.** A T4a state averaged over two clips
  gives a video almost identical to the pixel-wise mean of the two clips
  (r 0.99). This speaks to the linearity of the map "video → T4a state", not
  to a new intermediate scene: a pixel-wise mean is a double exposure. The
  scale matters: clip A alone already scores 0.81 against that mean. In
  part 2 the same property resurfaces as a defect of the generator.
- **Hybrids of levels** (early types from one clip, T4/T5 from another) are
  a contest, and T4/T5 win: in one variant the video resembles the clip given
  to T4/T5 at 0.88 and the other at 0.15. The round trip stays at 0.15.

So even before a trained generator, the boundary of a reachable state was
visible. [Report](2026-09-19_step12_manipulated_states.md).

## 7. Amortised inversion: a linear model and a CNN

Inversion takes hundreds of steps per target. We built `video → state` pairs
from Sintel and procedural stimuli - 9,468 clips: motion, flashes, noise,
textures; whole scenes and stimulus classes held out for testing - and
trained two fast decoders for three groups of types (early L1/L3, deep
T4a-d/T5a-d, and all 14 types):

- a **linear model** with a shared hexagonal-temporal convolution;
- a small non-linear **hex + temporal CNN**.

On the deep group and 1,996 held-out clips the per-frame correlation is
**0.927** for the linear model and **0.960** for the CNN. The round trip on
eight held-out clips, drawn evenly across the test set: 0.036 for the linear
model, 0.029 for the CNN and 0.009 for slow inversion. A control with
shuffled cells gives 2.9-4.5.

This is a cheap one-step approximation of the inverse map - **amortised
inversion** - not proof that the CNN follows any arbitrary state. These
numbers have no time-shuffle control.
[13A report](2026-09-19_step13a_amortised_inversion.md).

## 8. SiT: generation conditioned on neurons

Next came a conditional flow (SiT, 1.40 M parameters): eight T4/T5 types, a
mask of available types and noise `z` give all 40 frames of a video in 20
Euler steps. 20,000 training steps on a T4, ≈ $0.22. This generator, **13B**,
is the renderer of all of part 2.

![Fig. 10](../docs/figures/state_to_video.gif)

*Fig. 10. 13B on held-out clips. Top: the clip; below: video from its T4/T5
state under two different noises. Right: the control on clip A - the same
state and the same noise, with the cells shuffled, give another video.
Script `tools/fig_gh_render.py`, run `2026-09-20_gen13b_samples`.*

| | value |
|---|---|
| r to the original video, 8 held-out clips | **0.966** |
| same `z`, true state against shuffled state: r between the outputs | **0.03** |
| video from the shuffled state: r to the clip | 0.00 |
| round trip | 0.031 |
| spread over seeds with the full state | r 0.96-0.99 between samples |
| spread over seeds conditioned on T4a only (one clip, 4 seeds) | r 0.82 |

The state makes the picture, not the model's prior: with one `z`, switching
to the shuffled state gives a different video. The full T4/T5 state nearly
fixes the clip; a condition on T4a alone leaves the generator noticeable
freedom.

**Where the condition is not obeyed.** On strong hand edits of T4/T5 the
round trip stays large (0.29-1.47): SiT's prior beats the condition.
"Generative model" here means a way of producing video, not a guarantee of
obeying any condition.

Two caveats on the numbers:
- **The round trip measures the video's compatibility with the brain, not
  its content.** In part 2 (section 5) it scores an almost blank grey field
  better than a real clip. Here it is read only beside a visual check.
- **The round trips of 13A and 13B are normalised differently** (over all
  eight clips against one clip), so the CNN's 0.029 and SiT's 0.031 cannot be
  compared directly. An approximate rescale onto a common scale puts the CNN
  near 0.033 ([ISS-0014](../ISSUES.md)).

[13B report](2026-09-20_step13b_generative_decoder.md).

## 9. The generator's limits: new conditions and a closed loop

What happens if the condition is not the state of a single clip?

The numbers in this section are means over two generator noises.

- **Random T4/T5 activity** gives video, but weak compatibility: a round trip
  of 1.85 for white and 2.33 for structured noise against 0.029 for the state
  of an ordinary clip.
- **Permutations and rotations of the direction channels** did not become
  knobs: 2.2-9.5. Time reversal gives 1.0 and 5.5, amplifying one direction
  0.68.
- **A hand-drawn T4a stripe** is almost ignored: the video is grey. Yet its
  round trip is 0.095 - lower than any successful condition below. This is
  the first place where it shows that the round trip cannot tell an obeyed
  condition from an ignored one.

**Spatial compositions of the model's real responses** - rightward motion in
the left half and leftward in the right, motion next to expansion, upward
motion in a window - give plausible video that exists in no clip (round trip
0.10-0.14). How well the condition is met is so far visible only by eye and
only on the first seed: a crude metric, "which T4 direction dominates the
video", matches the requested one in two cases out of six (three
compositions × two seeds). This is a narrow way of specifying a new state,
not a language of neural prompts.

![Fig. 11](../docs/figures/round_trips.png)

*Fig. 11. 13B's round trip on conditions no clip caused. The ignored stripe
(orange) looks more compatible than every composition. Script
`tools/fig_gh_charts.py`, run `2026-09-20_prompts14`.*

**The closed loop** `state → SiT → video → model → state → …`: every
adjacent pair is compatible (round trip 0.02-0.05), but the content drifts -
r to the starting clip falls from 0.99 to 0.26 over 11 passes. These are the
dynamics of the *pair* of generator and encoder, not an attractor of the
fly's brain.

![Fig. 12](../docs/figures/closed_loop.png)

*Fig. 12. The closed loop from three starts: the video's similarity to the
starting clip by pass, and the round trip of each pass. Script
`tools/fig_gh_charts.py`, run `2026-09-20_prompts14_loop`.*

[Step 14 report](2026-09-20_step14_controllable_generator.md).

## 10. What we have, and where part 2 begins

A working end-to-end loop:

`MaleCNS wiring → optic-lobe model → responses by level →
linear readout / inversion / CNN / SiT → video → another pass through the model`.

The project's contribution is not a claim to have uncovered fly vision but a
built pipeline and its checks: a connectome export with explicit column
geometry, a parameter-transfer bug found and fixed, decoding by level with
controls, generation from T4/T5, and batching of an expensive inversion.

The negative results matter as much:
- a protocol that measured the stimulus's repeatability instead of
  information;
- a ladder of levels created by the decoder's window;
- a scale bug in the parameter transfer;
- a spatial defect of the OFF pathway that no protocol caught and only a
  picture of the layers showed;
- arbitrary states the generator refuses to obey;
- the drift of the closed loop.

**Limitations.** Model zero is a weaker motion detector than FlyVis, and its
OFF pathway is broken (ISS-0015); one side of one male, one synapse
threshold, parameters transferred rather than trained on this wiring. Every
number comes from one data split and from one or three ensemble members.

**The main limit of this part is the source of the state.** We can render any
reachable state, but every one of them was taken from some video. Part 2 is
about learning the distribution of states and drawing new ones from it.

## Reproducing

All figures are in [`docs/figures/`](../docs/figures/) and are built locally
by the scripts in `tools/` from saved runs:

```bash
uv run python tools/fig_gh_connectome.py   # fig. 1
uv run python tools/fig_gh_layers.py       # fig. 3
uv run python tools/fig_gh_decoding.py     # fig. 5
uv run python tools/fig_gh_inversion.py    # fig. 6
uv run python tools/fig_gh_two_brains.py   # fig. 7
uv run python tools/fig_gh_render.py       # fig. 10
uv run python tools/fig_gh_charts.py       # fig. 2, 4, 8, 9, 11, 12
```

Every number in the article leads to a step report in `reports/` (step
reports are in Russian) and to a record in [`reports/runs.jsonl`](runs.jsonl)
with its run id; known defects are in [`ISSUES.md`](../ISSUES.md).

## References

- Lappalainen J. K. et al. Connectome-constrained networks predict neural
  activity across the fly visual system. *Nature* 634, 1132-1140 (2024). -
  the FlyVis architecture, parameters and protocols.
- Berg S. et al. Sexual dimorphism in the complete connectome of the
  *Drosophila* male central nervous system. *Cell* (2026). - the MaleCNS v1.0
  connectome, Janelia FlyEM, CC-BY 4.0.
- Bauer J., Margrie T. W., Clopath C. Movie reconstruction from mouse visual
  cortex activity. *eLife*, reviewed preprint 105081 (2026). - video
  reconstruction by encoder inversion.
- Chen Y. et al. Decoding dynamic visual scenes across the brain hierarchy.
  *PLOS Computational Biology* (2024), doi:10.1371/journal.pcbi.1012297. -
  decodability by level and its controls.
- Ma N. et al. SiT: Exploring flow and diffusion-based generative models with
  scalable interpolant transformers. *ECCV* (2024). - the conditional flow of
  13B.
- Butler D. J. et al. A naturalistic open source movie for optical flow
  evaluation. *ECCV* (2012). - the Sintel video.
