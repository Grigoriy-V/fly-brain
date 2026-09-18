# Step 11: dreams-lite — generation from states that no clip caused

Date 2026-09-19. Agent: Claude (Fable). Code: `flydream/generate/dreams.py`,
`tools/fig_dreams.py`, `deploy/modal/generate_app.py::dreams`, commit `f088059`
and the follow-up (temporal score).

## Question

The item-8/9 generator turns a stage's state back into the clip that caused it.
Does it produce a picture from a state that no natural clip caused, and is that
picture read from the state's structure (against a shuffled-state control) or
from its statistics alone?

## Method

Four sources of state (`config.toml [generate.dreams]`), each inverted from every
stage of the ladder (R1, L1, L3, Mi1, Mi4, Tm5a, Tm9, T4a, T5a, T4+T5) beside its
**shuffled-state control**: the same target activity with the stage's cells
permuted by one fixed permutation (numbers kept, cell-to-place map destroyed).
20 tasks per source in one batch (item 10'), 150 steps with the plateau stop,
window 40 + 5 frames (85 for the dark after a clip).

- `eye_noise`: white noise into the eye, luminance 0.5 + N(0, 0.15), new draw
  each frame.
- `flash`: grey; white on frames 10-14; grey.
- `dark_after`: Sintel clip 3 for 40 frames, then 40 frames of black; the fit
  reads the dark frames only.
- `neuron_noise`: the eye sees grey; N(0, 0.05) added to every cell's activity
  after every Euler step of the target simulation (`simulate_noisy`); the
  inversion runs through the noiseless network.

Scores: PixCorr per frame against the input frame (eye noise), against the
clip's last frame (dark after), correlation over frames of the mean luminance
r(t) (flash: a uniform input has no spatial score), and the recovered video's
spatial contrast (sd over hexals, mean over frames) for every source.

## Run

`modal run deploy/modal/generate_app.py --model malecns --dream-sources
eye_noise,flash,dark_after,neuron_noise`, T4, 611 s of GPU for 80 tasks, ≈ $0.15.
Data `data/generate/2026-09-19_malecns_dream_<source>_<stage>/`. Clips
`reports/figures/2026-09-19_malecns_dream_<source>.{gif,png}`, first/middle/last
frames and the flash frame viewed before sending.

| source | GPU s | utilisation | plateau at step |
|---|---|---|---|
| eye_noise | 80 | 76 % | 107 |
| flash | 101 | 80 % | 136 |
| dark_after | 216 | 87 % | 150 (none) |
| neuron_noise | 76 | 74 % | 100 |

## Numbers

**Eye noise** (r to the input; control r; contrast out / control):

| R1 | L1 | L3 | Mi1 | Mi4 | Tm5a | Tm9 | T4a | T5a | T4+T5 |
|---|---|---|---|---|---|---|---|---|---|
| 1.00 | 0.94 | 0.94 | 1.00 | 0.96 | 0.53 | 0.91 | 0.95 | 0.31 | 0.98 |
| 0.00 | 0.00 | 0.00 | −0.01 | 0.00 | 0.01 | 0.00 | 0.00 | 0.00 | −0.01 |
| .12/.12 | .09/.11 | .07/.16 | .14/.17 | .13/.30 | .08/.09 | .09/.25 | .12/.34 | .04/.18 | .12/.13 |

**Flash** (r(t); control r(t); contrast out / control): r(t) 0.98-1.00 from every
stage, contrast 0.000-0.006 (the input's is 0); control r(t) 0.97-1.00 except
T4+T5 0.60, contrast 0.03-0.32. The shuffle keeps each cell's time course, so
the control recovers the flash's timing too; what it adds is the spatial texture.

**Dark after the clip** (r of the dark frames to the clip's last frame; control):

| R1 | L1 | L3 | Mi1 | Mi4 | Tm5a | Tm9 | T4a | T5a | T4+T5 |
|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 0.01 | 0.45 | −0.01 | 0.07 | 0.02 | 0.42 | 0.14 | 0.24 | 0.28 |
| 0.14 | 0.14 | 0.10 | 0.00 | −0.06 | −0.07 | 0.06 | −0.11 | 0.04 | −0.01 |

Recovered contrast 0.000-0.008: the after-image is faint and lives in the first
dark frames (a silhouette from L3, Tm9, T4a, T5a, T4+T5 at frame 0 of the dark,
black by frame 20).

**Noise inside the neurons** (contrast out / control): R1 .040/.054, L1
.052/.072, L3 .060/.148, Mi1 .061/.117, Mi4 .097/.289, Tm5a .050/.065, Tm9
.070/.238, T4a .067/.329, T5a .043/.178, T4+T5 .056/.130. Fine grey ripple of
low contrast from every stage, blurred patches from T5a; the control is 2-5×
more contrasted.

## What the pictures show

- The generator reads the state's structure: eye noise comes back from every
  stage but Tm5a and T5a, and the shuffled state gives r ≈ 0 everywhere.
- A shuffled state of Mi4, Tm9, T4a (and, weaker, L3, T4+T5) is explained by
  **diagonal stripes** in every source. That texture is not in any input; it is
  what the wiring needs to see for cells to hold values their neighbours do not
  share. It is the same lattice seen in the Tm5a/T5a decoder ladders.
- After the input goes dark the state keeps a faint trace of the last frame for
  a few frames, from L3 onward, not in R1/L1/Mi1.
- Internally generated activity is explained by quiet ripple, quieter than what
  the shuffled state needs.

## Claim

"Dream" is the project's name, not a claim. Every recovered video is the
stimulus most compatible with a state under this encoder; what was removed is
the input (grey or black), what was added is noise (into the eye or into the
cells) or a flash, and the control shows what the same numbers give without
their structure. One model (MaleCNS model zero), one seed, one setting per
source.

## Cost

Modal T4 ≈ $0.15. Local: CPU drawing only.
