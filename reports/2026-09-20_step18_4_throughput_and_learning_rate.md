# Step 18.4: seven arms on one control — capacity was the limit, and all four predictions taken from the literature pointed the wrong way

Date 2026-09-20. Agent: Claude (Fable). Every option 18.3 left open, plus the
two the research produced, run as arms of one sweep against a single control.
Code `deploy/modal/generate_app.py::{bench17, dct_maps, train17}`,
`flydream/generate/prior17.py` (compile, class conditioning),
`flydream/generate/samples17.py` (a label per sample); figures
`tools/fig_bench17.py`, `tools/fig_lr18.py`, `tools/fig_arms18.py`,
`tools/fig_scaling18.py`.
**Cost: $1.93 on ten Modal containers**; every gate below is local CPU, $0.
Checkpoints on `flydream-runs:/prior18/`, local copies in `data/prior18/`.

The headline in one line: **width 192 moves the main gate 0.095 → 0.034**,
against a representation floor of 0.021 and a real held-out clip's 0.012.

## The whole sweep

Every arm is the 18.3 configuration with **one** thing changed, 20,000 steps,
batch 32, seed 0, the same maps file, the same code path, `torch.compile` on.
The gate is the round trip of 16 samples through the frozen brain.

| arm | validation | **round trip** | novelty (r to the nearest training video) |
|---|---|---|---|
| 18.3 as published (no compile) | 0.5064 | 0.095 | +0.35 |
| **control**: 3e-4, w128, K16, compiled | 0.5052 | **0.090** | +0.36 |
| classes (110 labels) | 0.4944 | 0.098 | +0.36 |
| lr 6e-4 | 0.4264 | **0.080** | +0.40 |
| lr 1e-3 | 0.4225 | **0.076** | +0.42 |
| **width 192** | **0.2977** | **0.034** | +0.45 |
| 60,000 steps | 0.4046 | **0.074** | +0.38 |
| K = 32 | 0.9835 ‡ | 0.527 | +0.12 |
| lr 1e-4 (18.4b, uncompiled) | 0.7328 | 0.591 | +0.16 |
| *the representation's floor* (a real state at DCT-16) | — | *0.021* | +0.52 |
| *a real held-out clip* | — | *0.012* | +0.55 |
| *control: a clip state, columns permuted* | — | *1.313* | +0.12 |
| *control: white noise in the types* | — | *2.177* | +0.06 |

‡ K = 32's validation loss is against a different target (32 coefficients, not
16) and is not comparable to the rest; only its gate is.

## 18.4a Where the training step goes — the launch-bound reading is wrong

`single_gpu_throughput.md` read our 56.4 ms/step against a generous compute
floor of 15-23 ms and called the job "compatible with launch-bound, not proof
of it; profile." Profiled, one container, eight variants, 60 timed steps each:

| variant | batch | ms/step | samples/s |
|---|---|---|---|
| **base — 18.3's own loop** | 32 | **54.72** | 584.8 |
| without the per-step `loss.item()` | 32 | 54.81 | 583.8 |
| `torch._foreach_` EMA instead of the per-parameter loop | 32 | 54.94 | 582.5 |
| both | 64 / 128 / 256 | 106.08 / 209.72 / 418.13 | 603.3 / 610.3 / 612.2 |
| **`torch.compile(mode="reduce-overhead")`** | 32 | **38.47** | **831.8** |

**Affine fit: 2.44 ms fixed + 1.6229 ms per sample.** At batch 32 the fixed
part is **4.5 %** of the step, and an 8× batch buys **+4.9 %**. The step is
paid per sample, not per launch — the hypothesis is refuted, and with it the
plan that followed from it ("batch first, then compile, then L4"). Neither of
the two suspects in our own code costs anything measurable; with 2.44 ms of
fixed cost there is no room for them to.

`torch.compile` is the one real lever: **1.42× on the step, 1.23× end to end**
(920 s against 1,128 s for the same 20,000 steps, data loading included). The
training runs use `mode="default"` rather than `"reduce-overhead"`, because
`validate` swaps the EMA weights into the same module between steps and CUDA
graphs do not tolerate that; by the same affine fit, the graphs can be worth at
most 4.5 %.

*Not claimed:* even compiled, 1.20 ms/sample is still 1.7-2.6× the note's
estimated floor. That floor was computed for dense matmuls; ours is attention
over 721 tokens at width 128 with head dimension 32, which is memory-bound.
Where the remaining factor sits was not measured.

## 18.4b-c The learning rate — the optimum is above ours, not below it

The note flagged our 3e-4 at batch 32 as **8.5-24× above** the extrapolation of
DiT/SiT's 1e-4 @ 256 down to our batch. Three arms settle the direction:

| lr | validation | round trip |
|---|---|---|
| 1e-4 — the extrapolated value | 0.7328 | 0.591 |
| 3e-4 — ours | 0.5052 | 0.090 |
| 6e-4 | 0.4264 | 0.080 |
| 1e-3 | 0.4225 | 0.076 |

**Monotone, and pointing the opposite way from the literature.** The
extrapolated 1e-4 is 6.6× worse than our own setting, and the series is still
improving at 1e-3, so the optimum is at or above 1e-3 and **is not bracketed**.
It is also flattening: 3e-4 → 6e-4 buys 0.010, 6e-4 → 1e-3 buys 0.004.

**Length is the same lever, bought dearer.** The 60,000-step arm (same 3e-4)
settles 18.3's open question — validation was still falling at 20,000, and
running three times longer does converge: −0.41 % over the last 10,000 steps,
landing at validation 0.4046 and a gate of **0.074**. Put the four arms in
terms of the product **lr × steps** and they collapse onto one curve:

| lr × steps (units of 1e-4 × 1,000 steps) | arm | round trip | cost |
|---|---|---|---|
| 60 | 20k at 3e-4 (control) | 0.090 | $0.18 |
| 120 | 20k at 6e-4 | 0.080 | $0.17 |
| 180 | **60k at 3e-4** | **0.074** | **$0.47** |
| 200 | **20k at 1e-3** | **0.076** | **$0.17** |
| *60* | *20k at 3e-4, width 192* | *0.034* | *$0.24* |

60k at 3e-4 and 20k at 1e-3 are **interchangeable** (0.074 against 0.076 at 180
against 200 units) — and the learning rate buys that product **2.8× cheaper**
than steps do. Width 192 sits at the *same* 60 units as the control and scores
0.034, entirely off this curve: capacity is a different axis, and it dominates
the one these four share.

The 1e-4 arm also shows why novelty is never reported alone. Its samples are
*less* correlated with each other and *further* from the nearest training video
(+0.16 against +0.36) — both would read as gains in isolation. Beside a round
trip of 0.591 they mean the opposite: states that are near nothing the brain
can read back.

## 18.4d Capacity — the largest move in item 18, and the reading it overturns

**Width 192 (2.64 M parameters): the gate goes 0.090 → 0.034.** The distance to
what the representation itself allows falls from 4.3× (0.090 / 0.021) to
**1.6×**, so the split of the error changes character:

| | 18.3 | control | width 192 |
|---|---|---|---|
| round trip | 0.095 | 0.090 | **0.034** |
| the floor (DCT-16) | 0.021 | 0.021 | 0.021 |
| **the prior's own share** | 0.074 | 0.069 | **0.013** |

The prior's own error is down 5.3×, and **the floor now dominates**: two thirds
of what is left belongs to the representation, not the model.

This contradicts the reading recorded the same morning in
`reports/2026-09-20_research_video_generation_and_training.md` §3, where five
independent lines (DiT's params-per-example ratio, EDM2-XS's, the absent upturn
in our validation curve, an extrapolated depth-to-width transition, Hyper-DP3)
were read together as "growing is safe, and nothing says capacity is the
cause". Growing was safe. Capacity *was* the cause. The five lines were not
wrong about their own papers; none of them studied this architecture at this
size on this modality, which is what the note said about itself, and what the
$0.24 measurement was for.

**K = 32 is the same finding from the other side.** Doubling the coefficients
lowers the floor — the file keeps 99.913 % of the state energy against 99.32 %
for K = 16 — and the gate gets **5.9× worse** (0.527 against 0.090). At width
128 the model is already the binding constraint, so handing it twice the target
dimension makes it relatively smaller still; its own error swamps everything
the floor gives back. K is worth revisiting **after** capacity, not instead of
it.

## 18.4f The scaling view — our points do not fall on one diagonal

The human, on seeing the sweep: the shape recalls an LLM training plot, where
a change of variable puts everything on one straight line. Plotted the way
those are — loss against compute `C = 6·N·B·S`, log-log — the seven arms say
no, and the way they fail is the finding (`tools/fig_scaling18.py`, $0, from
runs already paid for).

**Three arms sit at the same compute and are 20 % apart.** The control, 6e-4
and 1e-3 differ only in learning rate: same N, same batch, same steps, so
`C = 5.29e12` for all three, and validation 0.5052 / 0.4264 / 0.4225. In the
LLM plots this variation is absent because the rate is tuned per point; ours is
not, so compute alone cannot be the x-axis here.

**Width leaves the line; length stays on it.** Fit a power law through the four
width-128 arms — slope −0.096 on validation, −0.089 on the gate, which is the
familiar range — and extrapolate to width 192's compute:

| | compute | predicted | measured | |
|---|---|---|---|---|
| validation | 1.01e13 | 0.4225 | **0.2977** | **−30 %** |
| round trip | 1.01e13 | 0.0765 | **0.034** | **−56 %** |

The 60,000-step arm has **1.6× more compute** than width 192 and is worse on
both (0.4046 and 0.074). So at equal or greater compute, parameters beat steps
here — the same shape as Chinchilla's finding that a fixed budget has an
optimal split between model and data, with us on the under-parameterised side
of it. That is also why K = 32 sits far *above* the line: it adds target
dimension without adding capacity.

**What this is not.** Seven points over a 3× range of compute, one seed each,
with the learning rate untuned per point; scaling laws are fitted over four to
six orders of magnitude. The fitted line is a ruler for reading the width-192
residual off, not a law, and the exponent should not be quoted as one.

## 18.4e Classes — the one published precedent that did not transfer

A class-conditional prior over 110 labels (101 UCF101 actions + 9 procedural
kinds), conditioned DiT's way: the label embedding added to the timestep
embedding, modulating every block through adaLN-Zero, no dropout and no null
class — matching the precedent, which is conditioning alone without guidance.

**No effect: 0.098 against the control's 0.090**, inside the spread
(0.068-0.131 against 0.072-0.110); validation 0.4944 against 0.5052.

The precedent was clean and large (same net, same budget, no guidance either
side: FID 26.21 → 10.94). The research note named the gap in advance: no study
covers a label only *loosely coupled* to the signal, which is exactly what a
UCF101 action class is to a T4/T5 motion trajectory. "Archery" does not
constrain which directions the eight motion types take. The mechanism the
precedent relies on is absent here, and the measurement says so.

**Is the label read at all?** A gate averaged over sixteen samples from sixteen
noises cannot tell "the label is ignored" from "the label moves the sample but
not its compatibility". `flydream/generate/classcmp18.py` separates them: **one
noise vector, four ways out of it** — through the unconditional prior and
through the conditional one with three labels — with 13B rendering all four
from the same z, so the only thing that differs is the prior's response to the
label. Repeated on four noises (local CPU, 4 × 20 s, $0):

| | unconditional | «Surfing» | «Archery» | «procedural:grating» |
|---|---|---|---|---|
| round trip, mean of 4 noises | 0.093 | **0.081** | 0.103 | **0.136** |
| consistent across noises | — | better in 4 / 4 | worse in 3 / 4 | worst in 4 / 4 |

Correlation between the videos of two different labels on the same noise:
**+0.91** (0.88-0.93 per noise). Two different *noises* give +0.02. So:

- **the label is read** — the cells are not identical, and the ordering repeats
  in every noise;
- **it displaces the sample, it does not choose it.** The noise decides which
  video this is; the label adjusts it. Whatever the label carries, it is a
  small fraction of what the noise carries.
- **and what it carries is not motion.** The one class that should have helped
  most — a drifting grating, the canonical T4/T5 stimulus — is the **worst** of
  the four in every noise, while "Surfing" beats the unconditional prior in
  every noise. A label that encoded the motion in the clip would do the
  opposite. Over 110 classes these cancel, which is exactly the null the gate
  reported.

`reports/figures/2026-09-20_malecns_classcmp18.gif` (frames 0/20/39 of every
cell viewed before sending).

## What this changes

1. **Capacity first.** Width 192 alone is worth more than every other lever
   tested put together, and the prior is no longer the dominant term.
2. **The learning rate is not bracketed.** 1e-3 is the best measured point and
   the series has not turned. What the four length/rate arms share is the
   product lr × steps; the rate is the cheap way to buy it, and **more steps
   is the expensive way** — $0.47 for what $0.17 already gives.
3. **Two levers are dead for now:** classes (no effect — and the same-noise
   diagnostic shows why: the label displaces the sample by r 0.09 where the
   noise decides the rest, and what it carries is not motion) and K = 32 (much
   worse at this width).
4. **`torch.compile` is free money**: 1.23× end to end, so every future run of
   this shape costs about 80 % of what it did.
5. **Every literature-derived prediction we tested failed** — launch-bound
   (refuted), the extrapolated learning rate (6.6× worse), capacity "not the
   cause" (it was), conditioning worth 2.4× (worth nothing here). Four for
   four. What the literature *did* buy us is the three levers it closed for
   free — logit-normal timesteps already optimal, no t-weighting lever for
   velocity + MSE, our EMA rate in range — plus the SD3 sign-convention trap it
   warned about. The notes are worth their cost as a map of what not to try;
   they are not worth trusting as a prediction at this scale. Each of these
   four cost between $0.03 and $0.24 to settle, which is the argument for
   measuring rather than reading.

## What this does not tell us

- **One seed per arm.** The control's own 0.090 against 18.3's 0.095 is the
  scale of one seed plus kernel fusion, so differences of that size (classes,
  0.098) are not differences.
- **Nothing was combined.** Width 192 and lr 1e-3 each helped alone; whether
  they add is untested, and so is K = 32 at a width that can carry it.
- **The 60,000-step arm is not in this report.** It was still running when the
  sweep was written up and is being finished separately.
- **Novelty falls as the gate improves** (+0.36 → +0.45 against a real clip's
  +0.55): better priors sit closer to their training videos. That is expected
  and measured, not a problem, but it is the number to watch if the gate keeps
  improving — and per the bound recorded in 18.3, it means "not a near-copy",
  never "not memorised".

## Cost

Ten containers, all cpu 1 / 12 GB except the maps builder: bench17 120.6 s
≈ $0.03; lr 1e-4 1,243 s ≈ $0.24; `dct_maps` cpu 2 / 24 GB 106.5 s ≈ $0.03;
then the sweep — control 920 s ≈ $0.18, 6e-4 863 s ≈ $0.17, 1e-3 842 s ≈ $0.17,
width 192 1,215 s ≈ $0.24, K = 32 1,033 s ≈ $0.23, classes 845 s ≈ $0.17,
60,000 steps 2,479 s ≈ $0.47. GPU
utilisation 97-98 % on every training arm. Gates, figures and analysis local
CPU, $0. **Item 18 total ≈ $2.51.**
