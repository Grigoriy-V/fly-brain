# Step 13B: the generative decoder — deep state + type mask + z → video

Date 2026-09-20. Agent: Claude (Fable). Design (approved by the human):
`reports/2026-09-19_step13b_design.md` revision 2. Code
`flydream/generate/gen13b.py`, `deploy/modal/generate_app.py::{maps13b,
bench13b, multi_init13b, train13b, sample13b}`, `tools/fig_gen13b.py`.

## Question

Can one generative model return, from the deep state (T4a-d + T5a-d) and a
type mask, a video compatible with that state — with the interface of
generative AI (i2i, prompt by types, knobs, guidance, seed) — and how
compatible is it beside 13A's one-pass CNN and the Adam inversion?

## Data and model

Pairs13 shards → maps of the 8 T4/T5 types (9,468 × 40 × 8 × 721, float16,
z-scored per type over the training split; CPU container, 2 cores / 12 GB,
116 s, 5.5 GB on `flydream-runs:/gen13b/`). The same split as 13A (whole
scenes and classes held out).

Objective: SiT-style linear interpolant (flow matching), velocity MSE,
logit-normal t. Model: `SiTColumns` — 721 column tokens, features = the 40
frames of x_t + the masked state's 8 × 40 values + 8 mask bits, learned
position per column, 4 adaLN-Zero blocks, width 128, 4 heads; 1.40 M
parameters. Structured masks in training: full 40 %, T4 only 10 %, T5 only
10 %, one type 15 %, one direction 10 %, random subset 10 %, unconditional
5 %. Sampling: Euler, 20 steps, classifier-free guidance s, z = the initial
noise (one generator per seed, so one z is shared across states).

Training loop as designed: data on the GPU in float16, AMP fp16 +
GradScaler, fused AdamW (lr 3e-4, wd 0.01), warmup 100 + cosine, EMA
0.999, batch 32, grad clip 1. `torch.compile` not used (not benchmarked
after the backbone choice). Validation: the interpolant loss with the EMA
weights on 256 val clips, fixed noise and t grid.

## Runs

| run | what | machine | time | util | cost |
|---|---|---|---|---|---|
| (a) multi_init | inversion of 4 held-out deep states from grey + 8 random starts (sd 0.25), batch 36, 150 steps | T4, 1 core / 6 GB | 253 s | 82 % | ≈ $0.05 |
| (b) bench | backbones at batch 32 (6,000 steps by a flag error) | T4, 1 core / 12 GB | 358 s | 98 % (SiT) | ≈ $0.08 |
| (c) train | SiT 128 × 4, 20,000 steps | T4, 1 core / 12 GB | 1,117 s | 98 % | ≈ $0.22 |
| (d) sample | 156 samples in batches of 32 + one round-trip batch | T4, 1 core / 8 GB | 81 s (+ 1 failed start) | 47 % | ≈ $0.07 |

`data/gen13b/{multi_init,bench,sit_train,samples}.json`, `samples.npz`;
`reports/runs.jsonl`.

**(a) No multimodality at the full deep state.** The 8 random starts
(mean |init − grey| 0.19) converge to the grey start's solution within
|Δ| 1e-4 to 1e-3, pairwise r 1.000, identical round trips (0.005 / 0.006 /
0.014 / 0.103 on the four clips). Sample spread is therefore not a goal at
the full mask; it is expected only under partial masks and on unreachable
states.

**(b) SiT is the backbone.** 0.057 s/step at batch 32, peak 1.9 GB, GPU
98 %; loss 0.118 → 0.025 over 6,000 steps. The hex-ResNet (64 × 6) ran out
of memory at batch 32 (fp32 neighbour gather ≈ 1.6 GB per layer); not
pursued, SiT being cheap enough.

**(c) Training.** Train loss 0.118 → 0.0225 (noisy at the floor of the
velocity target), validation loss 0.0127 (step 8,500) → 0.0100 (step
20,000), still falling slowly at the end.

## (d) Numbers

Round trip = the sample simulated through the frozen brain from grey (last
frame held 5 frames), per-type squared error to the target on T4/T5 over
clip A's variances, mean over the 8 types (13A's metric). Median over 4
seeds; "spread" = mean r between the 4 samples; r to the reference video.

| state (mask full, s = 1) | median rt | best | spread r | r to ref | 13A CNN rt | inversion rt |
|---|---|---|---|---|---|---|
| held-out clips, mean of 8 | 0.031 | 0.029 | 0.96-0.99 | 0.966 | 0.029 | 0.009 |
| clip A / clip B | 0.027 / 0.013 | 0.024 / 0.011 | 0.99 | 0.99 / 0.98 | 0.018 / 0.010 | 0.001 / 0.004 |
| A, T4a × 0.5 / × 2 | 0.287 / 0.963 | 0.282 / 0.952 | 0.99 | 0.94 / 0.94 (A) | 0.508 / 1.378 | 0.194 / 0.802 |
| A, T5 × 0 | 1.470 | 1.453 | 0.99 | 0.98 (A) | 1.584 | 0.864 |
| 0.5·A + 0.5·B | 0.033 | 0.032 | 0.99 | 0.98 (½(A+B)) | 0.022 | 0.008 |
| state of ½(A+B) video | 0.010 | 0.009 | 0.99 | 0.98 | 0.006 | 0.001 |
| L1…Tm9 ← A, T4/T5 ← B | 0.013 | 0.011 | 0.99 | 0.98 (B) | 0.010 | 0.004 |
| eye noise | 0.021 | 0.020 | 0.92 | 0.86 (input) | 0.015 | 0.007 |
| neuron noise | 0.045 | 0.044 | 0.96 | — | 0.040 | 0.028 |
| shuffled state (control) | 0.957 | 0.895 | 0.85 | 0.00 | — | — |

Knobs on clip A's state:

| mask / guidance | median rt | best | spread r | r to A |
|---|---|---|---|---|
| all T4/T5, s = 1 | 0.027 | 0.024 | 0.99 | 0.99 |
| only T4a | 0.309 | 0.239 | 0.82 | 0.78 |
| only T4a-d | 0.033 | 0.032 | 0.99 | 0.98 |
| only T5a-d | 0.112 | 0.106 | 0.92 | 0.87 |
| T4a + T5a | 0.149 | 0.135 | 0.91 | 0.89 |
| all, s = 2 | 0.035 | 0.033 | 1.00 | 0.98 |
| all, s = 4 | 0.052 | 0.045 | 1.00 | 0.97 |

Conditioning strength (one z): r between the sample on the true state and
on the shuffled state 0.03, on the zero state 0.10 — the state, not the
prior, makes the picture. Figures
`reports/figures/2026-09-20_malecns_gen13b_{test,states,knobs,strength}.{gif,png}`
(frames 0/20/39 viewed).

## What the pictures show

- **The generator is a one-pass inversion with an interface.** On held-out
  clips and on every reachable state of items 11-12 its round trip equals
  the 13A CNN's (0.031 vs 0.029 on the 8 test clips; r 0.966 vs 0.960),
  3× the Adam inversion's, 30× under the shuffled control. The video is the
  one the eye saw (r 0.95-0.99), eye noise comes back as noise, the whole-
  state mix as the averaged video. Seeds agree (r 0.96-0.99), as (a)
  predicted: at the full mask there is nothing left for z to choose.
- **Knobs work as knobs.** Dropping types costs compatibility in the order
  of the information removed: T4 only ≈ full (0.033), T5 only 0.112, one
  direction 0.149, one type 0.309 — and the seeds diverge exactly there
  (spread r 0.82 at T4a only): z chooses what the state no longer fixes.
  T4 alone carries the frame; T5 alone gives a darker, blurrier
  reconstruction.
- **Guidance above 1 does not help** (s = 2: 0.035, s = 4: 0.052 vs 0.027)
  — the conditional branch already follows the state; pushing further
  over-sharpens. s = 1 is the setting.
- **On unreachable states the prior wins, as with the CNN.** For gain
  edits the generator returns clip A almost unchanged (r 0.94-0.98) with a
  large round trip (0.29 / 0.96 / 1.47); the inversion, which has no
  prior, gets closer (0.19 / 0.80 / 0.86) by leaving the clip. The
  generator's answer to "T4a × 2, all else equal" is the video whose state
  is nearest under its prior — the honest answer, with its price in the
  number — but it does not search the reachable set the way the inversion
  does. Seeds agree here too (0.99): the prior, not z, decides.
- **The prior does not swallow the state.** Same z, three states: r 0.03 /
  0.10 between the outputs; the shuffled state gives noise (0.96), the
  zero state gives flat grey.

## What is left in 13B (options, none started)

1. A brain-consistency term on x̂₁ (design § 5, last paragraph): the one
   thing that would move the generator on unreachable states toward the
   inversion; ≈ 2-3× the training price.
2. `torch.compile` and a wider model were not benchmarked; training is
   cheap enough (≈ $0.22) that a 2× model is affordable if the deep gap to
   the inversion (0.03 vs 0.01) is to be narrowed.
3. Prompts by hand (a state written by types rather than taken from a
   clip) were tested only as masks of clip A's state; a state drawn
   without any clip is the next knob.

## Cost

Modal ≈ $0.47 in total (maps $0.05, (a) $0.05, (b) $0.08, (c) $0.22, (d)
$0.07); local: figures.
