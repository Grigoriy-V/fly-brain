# Step 17.1b: the prior on scene states in a compact basis — the speckle is gone

Date 2026-09-20. Agent: Claude (Fable). Design: the human, 2026-09-20 — "текущий
dataset на ~76 % procedural, а задача — VIDEO GENERATOR": train the prior on
the Sintel states only, in a compact temporal representation, and compare with
17.1 by round trip and by eye. Code `flydream/generate/prior17.py` (DCT
helpers, `sample_states`), `deploy/modal/generate_app.py::train17`
(`--sources17 sintel --dct17 16`), evaluation `flydream/generate/samples17.py`,
figure `tools/fig_prior17.py --compare`. **Cost: one T4, 676 s, ≈ $0.14**;
everything else local CPU, $0. Data `data/prior17/`, checkpoint
`flydream-runs:/prior17/scene_dct16.pt`; clip
`reports/figures/2026-09-20_malecns_prior17b.{gif,png}` (frames 0/20/39 of
every cell and of eight samples viewed).

## What changed against 17.1

| | 17.1 | 17.1b |
|---|---|---|
| training states | 6,725 (5,030 procedural, 1,695 scenes) | **1,695 scene states only** |
| representation | 40 frames × 8 types × 721 columns (230,720) | **temporal DCT, 16 of 40 coefficients** (92,288), each coefficient z-scored |
| model | SiTStates 128 × 4, 1,428,032 par | SiTStates 128 × 4, 1,378,688 par |
| schedule | 20,000 steps, batch 32 | 12,000 steps, batch 32 |
| machine | T4, cpu 1 / 12 GB, 1,170 s, util 99.2 % | T4, cpu 1 / 12 GB, **676 s**, util 98.3 % |
| loss / validation | 1.741 → 0.754 / 0.776 | 1.837 → 0.506 / 0.566 |

**Why 16 coefficients and not 8** (measured locally before the run, on real
states): band-limiting a *real* state to K coefficients costs a round trip of
0.58 at K = 8, 0.20 at K = 12, **0.045 at K = 16**, 0.021 at K = 32, against
0.019 for the full state. K = 8 would have capped the whole test 13× above a
real clip; K = 16 keeps 99.55 % of the energy and is the knee.

## The gates (local CPU, 41 s, $0, same code path for every row)

| state given to 13B | round trip | nearest training video, r |
|---|---|---|
| **17.1b: from the scene/DCT prior** (16 samples) | **0.142** (0.119-0.176) | +0.41 (max +0.55) |
| 17.1: from the first prior (16 samples) | 1.191 (1.056-1.370) | +0.12 |
| a real held-out clip's own state | 0.006 (Sintel scene 0.061) | +0.68 |
| the representation's own ceiling (a real state, same DCT) | 0.009 (0.007-0.097) | +0.70 |
| control: white noise in the types | 2.187 | +0.05 |
| control: structured noise | 3.043 | +0.30 |
| control: a clip state, columns permuted | 1.770 | +0.16 |

Structure of the sampled states, against real ones:

| | temporal lag-1 | one-ring spatial | cross-type coupling | video frame-to-frame |
|---|---|---|---|---|
| 17.1 | 0.57 | 0.34 | 0.16 | 0.280 |
| **17.1b** | **0.98** | **0.77** | **0.31** | **0.026** |
| real states / clips | 0.99 | 0.80 | 0.35 | 0.057 |
| what the DCT-16 basis imposes by itself (white noise band-limited) | 0.77 | −0.00 | 0.01 | — |

Diversity: pairwise correlation between sampled states 0.75 (17.1: 0.32; real
states 0.72) and between their videos 0.029 (17.1: 0.022; real clips 0.012).

## What it shows

- **The main gate moves 8×**: 0.142 against 17.1's 1.19. A sampled state is no
  longer in the same class as a shuffled state (1.77) — it is an order of
  magnitude closer to a real one, though still ~20× above a real clip's 0.006
  and ~15× above the ceiling its own representation allows (0.009).
- **The structure is learned, not imposed.** The basis alone would give
  temporal 0.77 and nothing spatial (−0.00) or across types (0.01); the
  samples have 0.98 / 0.77 / 0.31, within a few hundredths of real states on
  all three. That is the model, not the compression.
- **The speckle is gone.** In the clip, 17.1's sample is salt-and-pepper at
  every frame; 17.1b's is smooth moving light and dark structure with oriented
  texture, and its video changes 0.026 per frame against 17.1's 0.280 (a real
  clip: 0.057 — the samples are now *smoother* than real video, not rougher).
  Judged honestly: oriented structure and large moving regions, **not scenes
  with objects**.
- **The videos moved toward the training distribution**: nearest training
  video r +0.41 against 17.1's +0.12 (a real held-out clip sits at +0.68), and
  the samples stay different from each other.

## What this does not tell us

Two things changed at once — the data (scenes only) and the representation
(DCT-16) — so this run cannot say which did the work. What it does separate:
the representation is not the remaining limit (its ceiling is 0.009, the
samples are at 0.142), and the temporal smoothness could have been imposed by
the basis while the spatial and cross-type structure could not.

## Where the remaining error is

The gap 0.142 → 0.009 is the prior itself: the model still has to place the
right pattern in the right columns, and a band-limited real state shows that
the basis can carry that. 1,695 training states of 19 scenes is a small set
for it; the human's plan — a large set of ordinary video with procedural
stimuli only as a minority — attacks exactly this.

## Cost

One T4 run, cpu 1 / 12 GB, 676 s (worker 722 s), GPU utilisation 98.3 %,
≈ $0.14. The K sweep, the evaluation and every diagnostic: local CPU, $0.
Item 17 so far: ≈ $0.37.
