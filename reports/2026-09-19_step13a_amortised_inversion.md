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
- **The learned models are 2-3× less compatible with the state than the
  inversion** (deep: 0.029-0.036 vs 0.009). They return the video that is on
  average right for a state, not the one that caused it; this is the
  amortisation gap, and the number the round trip was built to show.
- Dark scenes are the weak spot (cave_4 0.81-0.85 from deep).

## Not done (the rest of 13A)

The round trip on the states of items 11-12 (dreams, mixes) — the states with
no video — was not part of this run; the checkpoints are on the volume
(`/runs/train13/<cond>_<kind>.pt`) and a short GPU job (≈ $0.05) applies them
to those states beside the inversion's round trip, the shuffled-state control
and the model-to-model agreement. Also open: more CNN epochs from the saved
checkpoints (`--resume train13`) if the `deep` gap to the inversion is to be
narrowed before 13B.

## Cost

Modal: pairs13 ≈ $0.5 (T4 25 min + 4 cores / 24 GB, the render on the card
included), train13 ≈ $1.0 (T4 59 min + 2 cores / 12 GB) + ≈ $0.15 aborted.
Local: procedural clips (34 s on 16 cores), figures.
