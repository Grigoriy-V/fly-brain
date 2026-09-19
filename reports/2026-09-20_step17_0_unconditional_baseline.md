# Step 17.0: the unconditional baseline — what 13B already does from z alone

Date 2026-09-20. Agent: Claude (Fable). Design: `ROADMAP.md` item 17.0, the
free baseline that runs **before** a state prior is trained. Code
`flydream/generate/baseline17.py`, figure `tools/fig_baseline17.py`. Model:
the trained 13B generator (`data/gen13b/sit.pt`, SiT 128 × 4, EMA weights, 20
Euler steps) and the frozen MaleCNS model zero; nothing was trained, nothing
ran on Modal. **Local CPU, 35 s, $0.** Data
`data/baseline17/{summary.json, baseline17.npz}`; clip
`reports/figures/2026-09-20_malecns_baseline17.{gif,png}` (frames 0/20/39 of
every cell viewed before sending).

## Question

13B saw the unconditional mask on 5 % of its training steps, so it can draw
without being given a state: `z → 13B (mask "none") → video`. Before spending
anything on a prior over states, the plain question is whether that already
produces **new video without a source clip** — and what the frozen brain makes
of it.

No round trip is scored here. Nothing was conditioned, so the state read back
from a generated video would be compared with itself; compatibility is not the
baseline's claim. What the baseline claims is novelty and diversity, and both
are measured against the same numbers for real video.

## Method

- **The bank.** 13B's own training videos were rebuilt locally in the order of
  the 13A/13B manifest: 2,268 augmented Sintel clips from
  `pairs13.sintel_videos` (flyvis's rendering cache, 2 s) plus the 7,200
  procedural clips saved in `data/pairs13/procedural_800_s0.npz` — 9,468 in
  total, checked against `manifest["meta"]` at four index boundaries.
  `manifest["split"]` then gives the 6,725 videos 13B trained on and the 1,996
  it never saw.
- **Samples.** 16 unconditional videos (one batch, zero condition, zero mask,
  20 Euler steps, seed 1000).
- **The brain.** Every video (last frame held for the 5-frame margin, as
  everywhere in items 11-14) simulated through model zero from the same grey
  steady state; the deep state is the 8 T4/T5 types, the grey state is the
  baseline for `direction_energy`.
- **Novelty.** For each video, the nearest of the 6,725 training videos by
  correlation over the 40 × 721 hexals, with a normalised squared distance
  beside it. The scale is the same number for 16 **held-out real clips**: a
  genuine video the generator never saw also has a nearest training video, and
  that is what "not a copy" looks like.
- **The metric's own check.** 13B conditioned on four training clips' states
  (full mask) must come back nearest to those very clips. Without this a
  novelty number proves nothing.

## Numbers

| | generated from z (16) | real held-out clips (16) |
|---|---|---|
| nearest training video, r: median | **+0.53** | +0.52 |
| nearest training video, r: max | +0.63 | +0.85 |
| nearest training video, normalised distance (median) | 2.9 | 2.0 |
| pairwise r between videos (mean) | −0.005 | −0.012 |
| pairwise r between their brain states (mean) | 0.95 | 0.92 |
| video mean / sd | 0.34 / 0.13 | 0.41 / 0.16 |
| frame-to-frame change | **0.038** | 0.017 |
| direction the brain reads (T4 argmax) | d in 15 of 16 | d in 12 of 16 |
| T4/T5 energy above grey, per type (mean) | −0.26 −0.25 −0.14 +0.11 / +0.15 +0.16 +0.13 +0.43 | −0.15 −0.14 −0.09 +0.06 / +0.11 +0.12 +0.09 +0.35 |

Metric check: **4 of 4** state-conditioned samples have their own source clip
as the nearest training video (r to the source +0.985 sleeping_1, +0.942
market_5, +0.992 a grating, +0.976 a bar; normalised distance 0.017-0.116).
The metric detects copying when copying happens.

## What it shows

- **Unconditional 13B already makes video that is not in the training set.**
  Its nearest training video sits at r +0.53, the same place a genuine
  held-out clip's nearest sits (+0.52), and no sample reaches the +0.85 that a
  real clip reached. Samples are as unlike each other as real clips are
  (pairwise r ≈ 0 for both). So "a new video without a source clip" is
  already true today, before any prior.
- **But what comes out is the prior's moving texture, not a scene.** The clip
  shows it: drifting light and dark blobs, no edges, no objects, while the
  nearest training video beside it — and the real clip in the same row — have
  structure. The numbers agree: the generated videos change twice as fast
  frame to frame (0.038 against 0.017), are darker and lower in contrast
  (0.34 / 0.13 against 0.41 / 0.16), and one drifts in mean brightness across
  the clip (sample #4: 0.35 → 0.47). This is the same texture item 14 saw when
  random states were fed in (14.0).
- **The states these videos produce are reachable by construction**, and they
  are strong: the per-type energy above grey has the same sign pattern as real
  clips' and about 1.5× the magnitude. The brain reads downward motion (T4d)
  in 15 of 16 — but it also reads it in 12 of 16 **real** clips, so that bias
  belongs to the stimulus set and the model, not to the generator.
- **State diversity is high for both** (pairwise r 0.95 generated, 0.92 real):
  at this measure a state is dominated by structure common to every clip, so
  it separates nothing. A sharper state metric is needed in 17.2 and it will
  be the distance to the *nearest training state*, on the full state set that
  lives on the volume.

## What this means for 17.1

The baseline removes one justification for the prior and leaves the real one
standing. "Can we make a new video without a source clip" is answered — yes,
weakly, as texture. What unconditional sampling cannot give is a **state you
chose**: there is no handle here, no interpolation, no editing, and nothing a
state from elsewhere (a deeper level, item 16's spontaneous activity) could
enter. That is what 17.1's prior over states is for, and 17.2 must show its
samples beat this row on the thing that matters — structure that the brain
reads back as a compatible state — not merely on being new.

## Limits

- 16 samples and 16 real clips; the medians are stable but the maxima are
  single draws.
- The novelty search is over the 6,725 training videos of this split only; a
  sample could still resemble a val/test video, which is not a copying failure
  but is not measured here.
- The normalised distance is divided by the query's own energy, so a dark
  sample's distance (up to 16.9) says as much about its contrast as about its
  novelty; the correlation column is the one to read.
- No state novelty here: the training states are the 5.5 GB `maps_deep.npz` on
  the Modal volume. 17.2 measures that where the file lives.

## Cost

Local CPU only (35 s for the measurement, plus figure rendering); $0. No
Modal, no training, no download.
