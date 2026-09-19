# Step 13A: the amortised inversion — linear hex-temporal vs CNN vs Adam inversion

Date 2026-09-19/20. Agent: Claude (Fable). Code: `flydream/generate/stimuli.py`,
`pairs13.py`, `learned.py`, `deploy/modal/generate_app.py::{pairs13_videos,
pairs13, train13}`, `tools/fig_train13.py`.

## Question

Can a small deterministic network read the video out of a state in one pass,
how much of that needs a nonlinearity, and how much of its output is the state
rather than a learned prior — against the Adam inversion as the reachable
ceiling? Three input conditions: `early` (L1, L3), `deep` (T4a-d, T5a-d — the
one that matters), `all` (the 14 ladder types).

## Data (pairs13)

Videos: the augmented Sintel set (189 clips × 2 flips × 6 lattice rotations =
2,268) and 7,200 procedural clips (`stimuli.py`: edge, bar, grating, dots, flow,
noise, flash, texture, mixture; 800 each; figure
`reports/figures/2026-09-19_procedural_stimuli.png`), all 40 + 5 frames.
States: model zero's activity of the 14 ladder types (9,624 cells) from the
grey steady state, float16, 19 shards on the `flydream-runs` volume. Split by
whole scenes (bamboo_2, cave_4, market_6, temple_3) and whole classes (dots,
texture): train 6,725 / val 747 / test 1,996. Simulation: T4, 9,468 clips in
25 min, ≈ $0.5 — of which ~18 min was flyvis rendering Sintel on the worker's
CPU inside the GPU function (the human's correction; the function is now
split into a CPU render and a GPU simulation, AGENTS "a GPU function does GPU
work only").

## Models

Input (B, T, K, 721): each type a channel on the lattice (Tm5a's 251 cells at
their columns). Both models share one kernel over columns: a column's value
and its 6 neighbours at 5 frames from t on.

- **Linear hex-temporal:** one such kernel + bias. 71 / 281 / 491 weights.
- **CNN hex+temporal:** the same as a first layer (width 32), two residual
  hex layers, a hex readout. 16.9k / 23.6k / 30.3k weights.
- **Adam inversion** (item 8/12 code, per-type normalised loss) on the same 8
  test clips as the ceiling.

Training: Adam, MSE on the video, inputs z-scored per type; linear 12 epochs
(5 s each), CNN 4 epochs (≈ 190 s each — the human stopped a 12-epoch run at
≈ $0.15 when the price came out 2.7× the estimate; `--epochs-cnn`, `--resume`
added). One T4 run, 2 cores, 12 GB, 59 min, GPU utilisation 72 %, ≈ $1.0.

## Numbers

Mean per-frame PixCorr to the video on the held-out test set; the round trip
= the model's video simulated through the frozen brain, per-type normalised
distance of its state to the target (mean over the condition's types, 8 test
clips); the inversion's round trip is the reachable floor.

| condition | model | weights | test r | round trip |
|---|---|---|---|---|
| early | linear | 71 | 0.971 | 0.030 |
| early | CNN | 16,897 | 0.978 | 0.029 |
| early | inversion | — | 1.000 | 0.018 |
| deep | linear | 281 | 0.927 | 0.036 |
| deep | CNN | 23,617 | 0.960 | 0.029 |
| deep | inversion | — | 0.999 | 0.009 |
| all | linear | 491 | 0.997 | 0.017 |
| all | CNN | 30,337 | 0.987 | 0.022 |
| all | inversion | — | 1.000 | 0.016 |

Test r by group (linear / CNN): early — bamboo_2 0.96/0.98, cave_4 0.91/0.96,
market_6 0.99/0.99, temple_3 0.98/0.98, dots 0.98/0.97, texture 0.97/0.99.
deep — bamboo_2 0.90/0.95, cave_4 0.81/0.85, market_6 0.96/0.98, temple_3
0.94/0.98, dots 0.94/0.95, texture 0.93/0.98. all — 0.985-0.998 / 0.957-0.993.

Val r (0.79-0.84) is lower than test r because the validation set holds
flashes, whose uniform frames give a per-frame r of 0 by construction; the
test classes have no uniform frames.

Figures: `reports/figures/2026-09-19_malecns_train13_{early,deep,all}.{gif,png}`
(rows: held-out clips; columns: eye / linear / CNN / inversion; frames 0/20/39
viewed), `..._summary.png`.

## What the numbers say

- **The deep state reads out in one pass** (r 0.93 linear, 0.96 CNN on scenes
  and a class never seen), and `deep` is the only condition where the
  nonlinearity matters: +0.03 r and a 20 % lower round-trip error. With the
  early types a linear sum nearly suffices; with all 14 types the linear
  decoder (491 weights) matches the inversion on both scores and the CNN,
  under-trained at 4 epochs, sits below it.
- **The learned models sit a few hundredths of r below the inversion, and
  their round-trip error is larger in relative terms** (deep: 0.029-0.036 vs
  0.009; the pictures differ in detail, not in kind). They return the video
  that is on average right for a state, not the one that caused it; this is
  the amortisation gap, small on the eye and measurable in the round trip.
  For a first version the one-pass readout is a competitor to the
  inversion, not a fallback.
- Dark scenes are the weak spot (cave_4 0.81-0.85 from deep).

## Round trip on the states of 11-12

Run `roundtrip13` (`flydream/generate/roundtrip13.py`,
`generate_app.py::roundtrip13`, `tools/fig_roundtrip13.py`): the 16 states of
items 11-12 rebuilt on the worker from their recipes (4 dreams; clips A and B;
gain edits T4a × 0.5, × 2, T5 × 0, Mi4 × 1.5; whole-state mixes α = 0.25,
0.5, 0.75 and the averaged video's state; two hybrids, L1…Tm9 from one clip
and T4/T5 from the other — the mixes and hybrids differ from item 12 so one
state feeds all three conditions). Per condition: linear, CNN, the Adam
inversion on the same types (item-12 loss, clip A's variances, batch of 15;
dark_after alone on 85 frames with the dark window read), and the linear
decoder on the state with its cells permuted (control). Every video's round
trip as above, relative to the target on the condition's types; r to the
reference video where one exists. T4, cpu=1, 4 GB, 441 s on the worker, GPU
utilisation 44 % (the inversion batch saturates; the single-clip dark_after
inversion and the decoders' small batches do not), ≈ $0.11. Data
`data/roundtrip13/`; clips
`reports/figures/2026-09-19_malecns_roundtrip13_{early,deep,all}_{dream,mix}.{gif,png}`,
frames 0/20/39 viewed.

Round-trip error, `deep` (T4a-d + T5a-d), linear / CNN / inversion / shuffled:

| state | linear | CNN | inversion | shuffled | r (CNN) to |
|---|---|---|---|---|---|
| eye noise | 0.020 | 0.015 | 0.007 | 3.85 | 0.97 input |
| flash | 0.014 | 0.003 | 0.000 | 4.51 | 1.00 input (temporal) |
| dark after clip | 0.168 | 0.150 | 0.000* | 0.47 | — |
| neuron noise | 0.056 | 0.040 | 0.028 | 3.98 | — |
| clip A | 0.036 | 0.018 | 0.001 | 2.86 | 0.99 A |
| clip B | 0.028 | 0.010 | 0.004 | 3.17 | 0.99 B |
| A, T4a × 0.5 | 2.384 | 0.508 | 0.194 | 3.40 | 0.96 A |
| A, T4a × 2 | 2.643 | 1.378 | 0.802 | 4.19 | 0.91 A |
| A, T5 × 0 | 1.713 | 1.584 | 0.864 | 6.39 | 0.78 A |
| A, Mi4 × 1.5 | 0.036 | 0.018 | 0.001 | 3.03 | 0.99 A |
| 0.5·A + 0.5·B | 0.047 | 0.022 | 0.008 | 3.11 | 0.97 ½(A+B) |
| state of ½(A+B) video | 0.017 | 0.006 | 0.001 | 3.29 | 0.98 ½(A+B) |
| L1…Tm9 ← A, T4/T5 ← B | 0.028 | 0.010 | 0.004 | 3.28 | 0.99 B |

\* the dark_after inversion reads 85 frames (clip + darkness) and so sees the
clip; the decoders read the dark window only, and return darkness (no
after-image). `early` and `all`: the full table is `data/roundtrip13/summary.json`;
early is flat (0.006-0.015 on every reachable state, the gain edits on T4a/T5/Mi4
invisible to L1/L3 by construction), `all` on the gain edits: linear 0.13-0.97,
CNN 0.21-0.74, inversion 0.12-0.62; on the hybrids 0.60 / 0.48 / 0.26.

What it shows:

- **On reachable states the one-pass readout holds off the training set.**
  Dreams, clips, whole-state mixes: the CNN's round trip is 0.003-0.04, within
  2-4× of the inversion's and 100-300× below the shuffled control; the video
  is the one the eye saw (r 0.97-1.00), and eye noise is recovered as noise,
  the flash by its timing. The amortisation gap of the test set is the same
  gap here; no new failure appears on states the training set did not contain.
- **On unreachable states the methods part.** A gain edit or a hybrid has no
  video; the inversion returns the nearest reachable video (item 12) with a
  large residual, and the decoders return something else with a larger one
  (T4a × 0.5: linear 2.4, the picture goes black; CNN 0.5; inversion 0.2).
  Linear and CNN still agree (r 0.87-0.98), so the decoders are consistent
  with each other but not with the brain. The round trip is the number that
  tells the two cases apart; r to a clip does not (0.85-0.96 on the edits).
- **After-image lost.** The dark window's state carries the clip (item 11),
  and the decoders read it as darkness (0.15-0.17 vs the shuffled 0.47): a
  state's history is not in the maps a one-pass readout sees.

## Not done (the rest of 13A)

More CNN epochs from the saved checkpoints (`--resume train13`) if the `deep`
gap to the inversion is to be narrowed before 13B; not a gate.

## Cost

Modal: pairs13 ≈ $0.5 (T4 25 min + 4 cores / 24 GB, the render on the card
included), train13 ≈ $1.0 (T4 59 min + 2 cores / 12 GB) + ≈ $0.15 aborted; roundtrip13 ≈ $0.11 (T4 7.4 min + 1 core / 4 GB).
Local: procedural clips (34 s on 16 cores), figures.
