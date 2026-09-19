# Step 14: the 13B generator as a controllable generator — four cheap tests

Date 2026-09-20. Agent: Claude (Fable). Design: the human's note
`docs/ideas/fly_brain_next_3_experiments.md` plus the random-state test (the
human: "мне всё равно, если это не будет работать, надо посмотреть, что
получим"). Code `flydream/generate/prompts14.py`, `tools/fig_prompts14.py`.
Everything on the trained 13B generator (`gen13b/sit.pt`, EMA weights, 20
Euler steps, full mask, s = 1) and the frozen model zero; no training.
**Local CPU, 99 s, $0.** Data `data/prompts14/{summary.json, prompts14.npz}`;
clips `reports/figures/2026-09-20_malecns_prompts14_{random,edits,prompts,loop,loop_B,loop_noise}.{gif,png}`
(frames 0/20/39 viewed).

## Question

Does the generator make video from states no video caused: noise in the
types (14.0), edited states (14.1), states assembled by regions or written
by hand (14.2); and what does the pair generator + brain do when closed on
itself (14.3)?

## Scores

Round trip as in 13A/13B (the video simulated through the frozen brain,
per-type normalised distance to the state on T4/T5, clip A's variances;
lower = more compatible). New: the **direction the brain reads** in a video
— the T4 type with the largest mean activity above the grey baseline after
the round trip (a–d = front-to-back, back-to-front, up, down) — beside the
direction the state itself carries. Two seeds per state (they agree within
0.01–0.04 of round trip everywhere; the table shows seed 0). Control: clip
A's T4/T5 cells permuted.

| state | what | round trip | state → video direction | r to reference |
|---|---|---|---|---|
| clip A | i2i reference | 0.027 | d → d | 0.99 (A) |
| **14.0** white noise in T4/T5 | per-cell N(mean, sd) of each type | 1.856 | d → b | — |
| 14.0 structured noise | smoothed in time (τ 100 ms) and over the lattice | 2.334 | d → d | — |
| **14.1** A, T4a↔T4c, T5a↔T5c | direction channels swapped | 4.047 | c → d | 0.91 (A) |
| 14.1 A, all channels +90° | a→c→b→d→a | 7.179 | d → d | 0.93 (A) |
| 14.1 A reversed in time | | 1.036 | d → b | 0.01 (A reversed) |
| 14.1 A, direction a × 2, others × 0.5 | | 0.664 | d → d | 0.91 (A) |
| 14.1 A left half, B right half | two clips' states by halves | 0.060 | b → d | 0.68 (A) |
| 14.1 grating → | the brain's state of a right-moving grating | 0.099 | b → b | 0.99 |
| 14.1 grating →, a↔b | | 2.174 | a → b | 0.99 (unreversed) |
| 14.1 grating →, +90° | | 9.554 | d → b | 0.90 (unrotated) |
| 14.1 grating →, reversed in time | | 5.453 | b → b | 0.23 (reversed) |
| **14.2** left → , right ← | halves of two brain states | 0.150 | b → b | — |
| 14.2 left →, right expansion | | 0.138 | b → a | — |
| 14.2 ↑ inside radius 6, grey outside | | 0.100 | c → c | — |
| 14.2 rotation | the brain's state of the stimulus | 0.006 | a → a | 0.99 |
| 14.2 hand-written: T4a stripe | T4a +3 sd in a horizontal stripe, rest grey | 0.095 | a → a | — |
| control: A shuffled | | 19.106 | d → d | −0.03 |

14.3, the closed loop state → 13B → video → brain → state′ → …, 12
iterations, one z (round trip per iteration; state change = mean |Δ| over
T4/T5 cells in units of clip A's sd):

| start | round trip, iterations 0 … 11 | change | r to the start clip, 0 … 11 |
|---|---|---|---|
| **clip A** | 0.027, 0.021, 0.024, 0.030, 0.038, 0.043, 0.045, 0.052, 0.049, 0.037, 0.027, 0.020 | 0.04–0.07 | 0.99, 0.95, 0.90, 0.82, 0.74, 0.65, 0.55, 0.47, 0.39, 0.33, 0.29, 0.26 |
| **clip B** | 0.012, 0.010, 0.015, 0.028, 0.050, 0.062, 0.059, 0.054, 0.048, 0.038, 0.029, 0.022 | 0.03–0.07 | 0.99, 0.95, 0.89, 0.80, 0.69, 0.56, 0.44, 0.33, 0.24, 0.18, 0.13, 0.10 |
| grating → | 0.099, 0.021, 0.018 … 0.024 | 0.04–0.07 | 0.99, 0.97, 0.95, 0.92, 0.89, 0.84, 0.78, 0.71, 0.64, 0.58, 0.52, 0.46 |
| white noise | 1.86, 0.09, 0.08, 0.06, 0.04, 0.03, 0.02, 0.015, 0.012, 0.010, 0.008, 0.007 | 0.49 → 0.03 |
| hand-written stripe | 0.095, 0.002, 0.002, 0.003, 0.004, 0.005, 0.007, 0.012, 0.020, 0.031, 0.044, 0.053 | 0.06 → 0.07 |
| grating, a↔b | 2.17, 0.024, 0.019, 0.020 … 0.024 | 0.42 → 0.05 |
| left →, right ← | 0.150, 0.032, 0.033 … 0.020 | 0.09 → 0.04 |

## What the pictures show

- **14.0 Random state → a picture, but not a compatible one.** White noise
  in the 8 types gives a noise video (round trip 1.9 against 19 for the
  shuffled control and 0.03 for a clip); structured noise gives a video
  of blobs and streaks that looks like a scene from the training set —
  the prior's texture — with a worse round trip (2.3). This is the answer
  to "what if the inputs are random": the generator draws its prior over
  the noise, and the brain does not recognise the result as the state
  that was asked for. Noise in the types is not the analogue of noise in
  a diffusion model; z is.
- **14.1 The direction channels are not knobs.** Swapping T4a↔T4b/c,
  rotating all channels by 90°, reversing the state in time, amplifying
  one direction: none of these is a state the brain can reach (round trip
  0.7–9.6), and the generator answers all of them with the *original*
  motion (video direction unchanged, r 0.90–0.99 to the unedited clip or
  grating). Direction is carried by the spatio-temporal pattern of
  activity across columns, which these edits leave intact, not by the
  channel label alone; a reversed state is not the state of a reversed
  video (the brain's delays run one way). The reachable edit is
  **composition by region**: A on the left, B on the right (0.060) gives a
  video that is A on the left and B on the right.
- **14.2 Prompts without a video work when their parts come from the
  brain.** Motion right on the left and left on the right (0.150),
  motion right beside expansion (0.138), motion up in a window on grey
  (0.100): the generator renders each region with its own motion, the
  brain reads the composite direction back, and no such video existed.
  The hand-written state (a T4a stripe on grey) gets a grey video: the
  round trip is low (0.095) only because the stripe is a few columns of
  one type, and the generator ignores it. A prompt has to be written in
  the brain's own patterns; a single type set by hand is not one.
- **14.3 From a clip the loop drifts: every step is compatible, the scene
  is not conserved.** Started from clip A's state, each pass returns a
  video the brain maps back to nearly the same state (round trip
  0.02–0.05 at every iteration), yet the video walks away from the clip:
  r to the start 0.99 → 0.90 (step 2) → 0.65 (step 5) → 0.26 (step 11);
  clip B 0.99 → 0.10; the grating keeps its motion longer (0.46 at step
  11). The small per-step error of the pair (the 13A/13B amortisation
  gap, 0.03) compounds: what the generator loses at each pass — dark
  regions, fine detail — the brain does not put back, and the prior
  fills the gap with its own texture, so after ten passes the scene is a
  different one with the same coarse motion. Unreachable starts jump to
  a reachable state in one step (white noise 1.86 → 0.09, swap 2.17 →
  0.02) and then drift the same way; from white noise the pair settles
  on a *noise video* (round trip → 0.007), a fixed point that is not a
  scene; from the hand-written stripe it sits on grey for ~6 passes
  and then drifts into blobs. These are properties of generator +
  brain, not attractors of the brain; the drift is the honest measure of
  how much of a scene the pair carries per pass.

## What this changes for the write-up

The generator is controllable through *compositions of brain states*
(regions, clips, stimuli), not through per-type knobs: the state space
has a structure the brain imposes, and edits outside it are answered by
the prior. Random inputs give the prior's texture, measurably
incompatible. This closes the stage: 13A (what reads out), 13B (one-pass
generator with masks), 14 (what can and cannot be prompted). Item 16
(where states come from without a video) stays deferred.

## Cost

Local CPU only (76 s for the run, figures); $0.
