# Step 17.1-17.2: a prior over T4/T5 states — trained, and it does not reach the manifold

Date 2026-09-20. Agent: Claude (Fable). Design: `ROADMAP.md` item 17, from the
human's documents `docs/ideas/brain_state_prior_new_video_generation.md` and
`docs/ideas/full_project_architecture_brain_to_video.md`. Code
`flydream/generate/prior17.py` (the model and its loop),
`deploy/modal/generate_app.py::train17` (the priced run),
`flydream/generate/samples17.py` (the gates, local), figure
`tools/fig_prior17.py`. **Cost: one T4 run, 1,170 s, ≈ $0.23** (17.1) plus
local CPU, 41 s, $0 (17.2). Data `data/prior17/`, checkpoint
`flydream-runs:/prior17/state_flow.pt`; clip
`reports/figures/2026-09-20_malecns_prior17.{gif,png}` (frames 0/20/39 of
every cell viewed).

## Question

13B renders `state → video`; it has no model of `p(state)`, so a state could
only ever be read off a video. 17.1 learns that distribution and 17.2 asks the
gate: does a state sampled from it survive the round trip
`state → 13B → video → frozen brain → state′` the way a real state does?

## 17.1 What was trained

Flow matching in state space, the same linear interpolant as 13B, with the
state as the data and no condition: token = column (721), features = that
column's 40 frames × 8 T4/T5 types, learned position, adaLN-Zero by t
(`SiTStates`, 1,428,032 parameters at width 128 × depth 4 — the size 13B's own
benchmark priced). Data: the 6,725 training states of 13A/13B, in exactly the
representation 13B is conditioned in (z-scored per type,
`flydream-runs:/gen13b/maps_deep.npz`), so a sample is fed to 13B unchanged.
Loop: 13B's optimised one (states on the GPU in fp16, AMP + GradScaler, fused
AdamW, warmup + cosine, EMA 0.999, gradient clipping), 20,000 steps at batch
32.

| | |
|---|---|
| machine | one T4, cpu 1, memory 12 GB, **GPU utilisation 99.2 %** |
| time / cost | 1,170 s / ≈ $0.23 (13B's measured rate: 1,117 s = $0.22) |
| loss | 1.741 → 0.754 |
| validation (EMA, fixed t grid) | 1.038 (2k) → 0.806 (5k) → 0.784 (10k) → 0.777 (15k) → **0.776** (20k) |

The validation curve is flat from ~12k steps: more steps of this model would
not have changed the result.

## 17.2 The gates

16 states from the prior (20 Euler steps) → 13B (full mask) → video → the
frozen brain, with every control rebuilt in the same code path so the table
shares one scale: three held-out clips' own states (the reachable reference),
white noise and structured noise in the types (item 14.0's constructions) and
a clip state with its columns permuted.

| state given to 13B | round trip | nearest training video, r | notes |
|---|---|---|---|
| **from the prior** (16 samples) | **1.19** (1.06-1.37) | +0.12 (max +0.19) | — |
| real held-out clip (Sintel temple_3) | 0.061 | +0.20 | the reachable reference |
| real held-out clips (procedural texture, 2) | 0.004, 0.006 | +0.68, +0.74 | |
| control: white noise in the types | 2.19 | +0.05 | 14.0's construction |
| control: structured noise (τ 100 ms, ring 1) | 3.04 | +0.30 | |
| control: a clip state, columns permuted | 1.77 | +0.16 | |

Diversity (pairwise correlation): the prior's videos 0.022 against 0.012 for
real clips, its states 0.32 against 0.72 — the samples are *more* different
from each other than real states are. Direction: the brain reads the state's
own direction back in 13 of 16 samples. The videos themselves are speckle:
frame-to-frame change 0.280 against 0.057 for a real clip's video (and 0.038
for 17.0's unconditional samples).

**The main gate fails.** A sampled state is not a reachable state: its round
trip, 1.19, sits beside a *shuffled* clip state's 1.77 and far from a real
clip's 0.006-0.061. Novelty and diversity are irrelevant while that holds — a
state nothing can produce is new in the same way noise is new.

## Why it fails (measured, not guessed)

- **Not the sampler.** 20 / 50 / 100 Euler steps give round trip 1.23 / 1.34 /
  1.40 on the same four samples: more steps make it slightly worse, so this is
  not integration error.
- **Not the plumbing.** A real clip's state, passed through the same
  map → state conversion and the same scoring, gives 0.006 — the 13B-level
  number. The path is verified end to end.
- **The samples are too white.** Within-sample structure, against real states:
  temporal lag-1 correlation **0.56** against 0.99, one-ring spatial
  correlation **0.34** against 0.80, cross-type coupling **0.16** against 0.35
  (white noise: 0.00 / 0.00 / 0.00). The prior learned perhaps half the
  structure of a brain state and none of its smoothness.
- **The velocity field is right near the data and mean-like away from it.**
  Denoising a real state at t = 0.9 gives x̂₁ with correlation **0.993** to the
  truth and the right structure (lag-1 0.98, ring 0.79); at t = 0.1 the
  correlation is **0.615** and the structure is gone (0.52 / 0.37). Along an
  actual sampling trajectory the clean estimate x̂₁ never leaves that regime:
  lag-1 0.49 → 0.56, ring 0.33 → 0.34 from the first step to the last. The
  model predicts something close to the conditional *mean* where the noise
  dominates, and integration from pure noise therefore lands beside the
  manifold, not on it.

That is the signature of a model that is too small (or a representation too
large) for the distribution it is asked to carry: 230,720 dimensions per state
against 1.4 M parameters and 6,725 examples.

## What this does and does not change

- 13B, the frozen brain, and every measured result of 13A/13B/14 and 17.0 are
  untouched: this step added a model beside them and it did not pass.
- "New video without a source clip" still stands where 17.0 left it —
  unconditional 13B makes video that is not in the training set, as texture
  rather than as a scene.
- What is still missing is what the track is for: a state one can **sample**
  and that the brain would accept. The prior in this form does not give it.

## Options, for the human to choose (none started)

1. **More capacity, same design.** Width 192-256 or depth 6 at 12-20k steps.
   The plateau says the current model is at its limit, not undertrained; a
   wider one is the direct test. Cost: ≈ $0.4-0.9 for one run, which touches
   or breaks the $0.50-per-run rule — needs the human's word on both the
   money and the exception.
2. **The documented fallback** (the human's §6): compress the state and put
   the flow in the compact space — either a small autoencoder over states, or
   a fixed basis that matches the measured smoothness (a temporal DCT keeping
   ~8 of 40 coefficients cuts the dimension fivefold at ~1 % of the variance
   lost, and needs no fitting). Cheaper per run than (1) and attacks the
   measured cause directly.
3. **Stop the prior line here.** Keep 17.0 as the answer to "video without a
   source clip", and spend the next money on item 16 or on a larger and more
   varied state set (17'), where the manifold is denser and a prior has more
   to learn from.

## Cost

17.1: one T4, cpu 1 / 12 GB, 1,170 s, GPU utilisation 99.2 %, ≈ $0.23.
17.2 and every diagnostic above: local CPU, 41 s + a few minutes, $0.
Total for item 17 so far: **≈ $0.23**.
