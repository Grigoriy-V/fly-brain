# Step 18.5: the two levers compose, the prior reaches the representation's floor, and K = 32 is not a capacity problem

Date 2026-09-20. Agent: Claude (Fable). The two options 18.4 left open, run as
one pair at the width that 18.4 found to be the dominant lever. Code
`deploy/modal/generate_app.py::train17` (unchanged), gate
`flydream/generate/samples17.py` (unchanged); figure `tools/fig_width18.py`.
**Cost: $0.48 on two Modal containers**; both gates local CPU, $0.
Checkpoints on `flydream-runs:/prior18/`, local copies in `data/prior18/`.

Two lines: **width 192 + lr 1e-3 puts the round trip at 0.0224 against a
representation floor of 0.0209** — the model is no longer what limits it. And
**K = 32 at width 192 is still 6.6× worse than K = 16**, which is *more* than
the 5.9× it cost at width 128, so capacity was never what it lacked.

## The arms

Both are the 18.4 configuration with the stated change, 20,000 steps, batch
32, seed 0, `torch.compile` on, one T4 each (cpu 1 / 12 GB), run in parallel.

| arm | validation | **round trip** | spread | floor at its own K | novelty |
|---|---|---|---|---|---|
| control: 3e-4, w128, K16 | 0.5052 | 0.0895 | 0.072-0.110 | 0.0209 | +0.36 |
| lr 1e-3, w128 | 0.4225 | 0.0762 | 0.058-0.083 | 0.0209 | +0.42 |
| width 192, 3e-4 | 0.2977 | 0.0341 | 0.024-0.051 | 0.0209 | +0.45 |
| **width 192 + lr 1e-3** | **0.2480** | **0.0224** | 0.014-0.039 | 0.0209 | +0.45 |
| K = 32, w128, 3e-4 | 0.9835 ‡ | 0.5268 | 0.447-0.748 | 0.0115 | +0.12 |
| **K = 32, width 192, 3e-4** | **0.7360** ‡ | **0.2243** | 0.193-0.278 | 0.0115 | +0.35 |
| *a real held-out clip* | — | *0.0119* | | | *+0.55* |

‡ a K = 32 validation loss is against 32 coefficients, not 16; it is
comparable only to the other K = 32 arm.

Runs: 1,180.3 s at 95.8 % GPU (combined) and 1,228.7 s at 96.6 % (K = 32),
$0.23 and $0.25.

## 18.5a The two levers compose, and the composition lands on the floor

Rate and capacity were found separately in 18.4 and lie on different axes.
Run together they **multiply, to within the noise of the measurement**:

| | validation | multiplier against the control |
|---|---|---|
| control | 0.5052 | — |
| lr 1e-3 alone | 0.4225 | ×0.836 |
| width 192 alone | 0.2977 | ×0.589 |
| **both** | **0.2480** | ×0.491 |

Independent factors would give ×0.836 × ×0.589 = ×0.493, i.e. **0.2490**
against a measured **0.2480** — 0.4 % apart. No interference in either
direction: no saturation, and no synergy either.

**At the gate the arithmetic stops, because the gate has a floor.** The same
product predicts 0.0291; measured is 0.0224. Read as the *prior's own share*
— the round trip minus what the representation itself costs — the picture is
plain:

| | round trip | floor (DCT-16) | the prior's own share |
|---|---|---|---|
| control | 0.0895 | 0.0209 | 0.0687 |
| width 192 | 0.0341 | 0.0209 | 0.0133 |
| **width 192 + lr 1e-3** | **0.0224** | 0.0209 | **0.0015** |

**The claim this licenses, and no more:** at DCT-16 the prior's own error is
now below what sixteen samples can resolve. It is *not* "the prior is
solved" — 0.0015 is a difference of two medians whose own spreads are
0.014-0.039 and 0.010-0.041, so the honest statement is that the two are
indistinguishable, not that the remainder is 0.0015. Nor is it "as good as a
real clip": a real held-out clip still scores 0.0119, and the floor sits
above it.

The consequence is operational: **further work on the model cannot move this
number at DCT-16.** Whatever is spent on width, rate or steps from here is
spent under a ceiling the representation sets.

Novelty did not degenerate to buy this: nearest-training-video r is +0.452
against +0.449 for width 192 alone and +0.554 for a real clip, and the samples
still differ from one another (median pairwise video r -0.002). One thing to
watch: the absolute distance to the nearest training video fell from 2.29 to
1.83 while the correlation held.

## 18.5b K = 32 is not short of capacity — it is short of a sane loss weighting

18.4d found K = 32 5.9× worse than K = 16 at width 128 and read it as
capacity: "at width 128 the model is already the binding constraint, so
handing it twice the target dimension makes it relatively smaller still". The
test of that reading is to hand it the capacity. Done, at the same 3e-4 on
both sides:

| | K = 16 | K = 32 | ratio |
|---|---|---|---|
| width 128 | 0.0895 | 0.5268 | **5.9×** |
| width 192 | 0.0341 | 0.2243 | **6.6×** |

Both improved in absolute terms — width helps K = 32 too (0.527 → 0.224, and
its novelty recovers from +0.12 to +0.35, so its samples are no longer near
nothing). But **the ratio did not shrink**; it widened slightly. A capacity
explanation predicts the opposite. 18.4d's reading is refuted by the
measurement it asked for.

**What the coefficients are actually worth.** Each DCT coefficient is z-scored
over the corpus before training, so every coefficient enters the loss with
equal weight. Their energies are not equal:

| | share of state energy in the **top half** of the coefficients | share of the loss |
|---|---|---|
| K = 16 (indices 8-15) | 3.50 % | 50 % |
| K = 32 (indices 16-31) | **0.45 %** | 50 % |

At K = 32 half the training signal is spent on a tail carrying 0.45 % of the
state — a **112× over-weighting**, against 14× at K = 16. The standard
deviations fall from 5.10 at index 0 to 0.055 at index 31, monotonically.
Splitting the K = 32 validation loss on that assumption is consistent: 0.7360
over 32 coefficients with the low half near a K = 16 arm's 0.248 implies ≈1.22
on the upper half, i.e. that half is close to unlearnable.

*This is a hypothesis with an arithmetic motivation, not a measured cause.*
It is consistent with every number above and it explains why capacity does not
help, but nothing here rules out a different mechanism.

## What this leaves open

Two options, neither run, both cheap, and they are now the only way the round
trip moves at all:

1. **Weight the loss by the coefficients' energy** instead of z-scoring every
   coefficient to unit variance (or keep the z-scoring for conditioning and
   apply an energy weight in the loss). A code change in
   `flydream/generate/prior17.py`, no new dataset. One arm at K = 32, width
   192 to test it: ≈ $0.25. If the hypothesis in 18.5b holds, this is what
   unlocks the lower floor.
2. **A width ladder** (256 / 384 / 512) to find where capacity stops paying.
   Note that at K = 16 it now has **nowhere to go** — the floor is reached —
   so it is only meaningful *after* a K that leaves room, which is what
   option 1 is for.

The K = 40 question the human raised is answered without a run: the time axis
is 40 frames, so K = 40 is the complete orthonormal basis and the ceiling of
that axis, and at K = 32 the floor (0.0115) is already below a real clip's own
round trip (0.0119). There is nothing left to buy with more coefficients; the
problem is what we do with the ones we have.
