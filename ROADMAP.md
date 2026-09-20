# Roadmap

**Updated:** 2026-09-20.

This file alone owns current direction, order and authorization. Rules:
`AGENTS.md` and `docs/ARTEFACTS.md`. System: `docs/PROJECT_MAP.md`.
Operations: `docs/OPERATIONS_MAP.md`. Durable choices: `DECISIONS.md`.
Defects: `ISSUES.md` (not a plan, authorizes nothing). Evidence: `reports/`.

## Where the project stands

The chain `video → frozen brain → T4/T5 state → generator → video → frozen
brain` works and is measured end to end: model zero on the MaleCNS export
(step 2), the decodability ladder (4), encoder inversion (8-12), the learned
decoder 13A, the conditional flow generator 13B, and item 14 on what can and
cannot be prompted. 13B turns a state into video by integrating its flow in 20
Euler steps (the one-pass read-out is 13A's CNN, not this one), at a round trip
of 0.031 against the inversion's 0.009 and its own shuffled-state control of
0.96.

What is missing is a **generative source of states**. Generating from a state
that no single clip caused is already measured: eye noise, a flash, the dark
after a clip and noise inside the neurons (item 11), and in 14.2 states
composed by region — motion right on the left half and left on the right,
motion up in a window on grey — which 13B rendered as asked and the brain read
back at a round trip of 0.100-0.150, with no source clip behind them. What the
project cannot do is **sample** a valid state: 13B's training distribution is
`video → brain → T4/T5`, no model of that distribution exists, and numbers
written into the eight types by hand land outside what the brain can reach
(14.0: 1.9-2.3 against 0.027 for a clip in the same table).

Two shuffled-state controls appear in the records and are **not on one scale**:
13B's, computed inside its sampling job (`sample13b`), is 0.96; item 14's,
built in `prompts14.py` with its own reference variances and state set, is 19.1.
Each sets the scale inside its own table only; the reason for the gap between
the two constructions has not been re-derived, so the two numbers are never
compared with each other or quoted as one baseline.

## Current track: new video without a source video

The human, 2026-09-20: "я ставлю чёткую задачу: я хочу получать новые видео".
Learn the distribution of reachable T4/T5 states, sample it from noise, render
with 13B, verify with the frozen brain:

```text
noise → state prior → T4/T5 state → 13B → video → frozen brain → round trip
```

The human's design documents for this track:
`docs/ideas/brain_state_prior_new_video_generation.md` and
`docs/ideas/full_project_architecture_brain_to_video.md`.

**Why a prior over states and not only "a new video".** 13B was trained with
an unconditional mask on 5 % of its steps, so it can already draw video from z
alone, and a video the brain has never seen produces a reachable state by
construction; 17.0 measures how far that alone gets us, for free, before
anything is trained. The prior's own value is different and is the reason it
is worth $0.22: it makes the **neural-state space itself a generative
interface** — sampling and editing in state space rather than in pixels,
interpolation between two brain states, variations around one — and it is the
interface a state from somewhere else can enter later: a deeper level of the
model, or the model's own spontaneous activity (item 16). A mechanism that
pulls an off-manifold state onto the learned manifold would serve item 16
directly, but it does not come free with a flow over states and has to be
defined and tested on its own (17.3).

This is not the dream chapter: the source of the state is an artificial prior,
not the brain's own activity. The dream source inside the model (item 16)
stays deferred.

## Item 17, the brain-state prior — measured, its gates open

Goal: video generated without a source clip, from states the project produced
rather than read, at the round trip of a real clip — and a state space that can
be sampled, interpolated and edited on its own. Nothing here retrains 13B,
changes the frozen brain or needs a new export.

**What is whose.** The human decided the track: new video without a source
clip, through generated brain states, with the neural-state space as the
interface to deeper and internal brain activity later (2026-09-20, in words).
The shape below is the agent's design from the human's documents: the free
unconditional baseline first, a flow over states before any VAE, an
autoencoder plus a latent flow held as the fallback, and the gates as listed.
It stands until the human says otherwise, and each priced run starts on the
human's word with the price stated first.

**17.0 The unconditional baseline — free, and first.** `z → 13B with
mask "none" → video → frozen brain → state`. 13B already saw that mask on 5 %
of its training steps, so this costs nothing but sampling and one simulation
and it answers the plain question before any training: *can the system already
make new video without a source clip, and what states does that video produce?*
Measured: diversity of the generated videos (pairwise correlation), the nearest
training video to each (correlation and normalised distance), the distribution
of the resulting brain states (per-type energy, the T4 direction the brain
reads). No round trip is scored here: the state is read *from* the generated
video, so comparing it with itself is zero by construction — the baseline's
claim is about novelty and diversity of video, not about compatibility, and
the report says so. These numbers are what every 17.2 result is read against.
Local CPU, $0, no new training code — only a sampling and scoring script.

*17.0 status (2026-09-20), measured, local CPU 35 s, $0:* unconditional 13B
already makes video that is not in the training set — the nearest of the 6,725
training videos sits at r +0.53 (max +0.63), where a genuine held-out clip's
nearest sits at +0.52 (max +0.85), and the samples are as unlike each other as
real clips are (pairwise r ≈ 0 for both). The novelty metric was checked on
the generator itself: 4 of 4 state-conditioned samples land on their own
source clip (r 0.94-0.99). What comes out is the prior's moving texture, not a
scene: twice the frame-to-frame change of a real clip (0.038 vs 0.017), darker
and flatter (mean 0.34 / sd 0.13 vs 0.41 / 0.16), no edges or objects. The
states are reachable by construction and about 1.5× the per-type energy of
real clips', with the same sign pattern; the downward-motion bias (T4d in 15
of 16) is the stimulus set's, not the generator's (12 of 16 real clips too).
Pairwise state correlation separates nothing at this level (0.95 vs 0.92), so
17.2 measures state novelty as the distance to the nearest training state
instead. `reports/2026-09-20_step17_0_unconditional_baseline.md`,
`reports/figures/2026-09-20_malecns_baseline17.gif`. What this leaves for
17.1: not "make a new video" — that is answered — but a state you can choose,
interpolate and edit, and an entry point for a state from elsewhere.

**17.1 The prior (training).** Flow matching over states, not a VAE: the same
linear interpolant and the same `SiTColumns` backbone as 13B
(`flydream/generate/gen13b.py`), with the state as the data (8 T4/T5 types ×
40 frames on 721 columns, z-scored per type) and no condition. Training data
is already on the volume — the 9,468 state maps of 13A/13B
(`flydream-runs:/gen13b/maps_deep.npz`, stored as 9,468 × 45 × 8 × 721 and
cropped to 40 frames on load as 13B does), split by scene exactly as 13B split
them, so a sample can be compared against held-out states. The optimised loop
of 13B (data on the GPU in fp16, AMP, fused AdamW, warmup + cosine, EMA 0.999,
batch 32) carries over. A local CPU smoke of the code and the loader first
($0); then one T4 run, ~20k steps, ≈ 1,100 s, ≈ $0.22.
Fallback if the samples fail the gates: an autoencoder over states plus a flow
in its latent (§6 of the human's document), not a prettier picture.

*17.1 status (2026-09-20), measured, one T4 1,170 s, ≈ $0.23:* `SiTStates`
128 × 4, 1,428,032 parameters, 20k steps at batch 32 on the 6,725 training
states, GPU utilisation 99.2 %, cpu 1 / 12 GB; loss 1.741 → 0.754, validation
1.038 → 0.776 and flat from ~12k steps. The checkpoint is
`flydream-runs:/prior17/state_flow.pt` (and `data/prior17/`).
`reports/2026-09-20_step17_state_prior.md`.

**17.2 Sampling and the gates.** Sampled states → 13B (full mask, s = 1,
20 Euler steps) → video → frozen brain → state′, all samples batched into one
pass on the card. Measured per sample:
- **round trip** of the sample, the main gate: `state → 13B → video → brain →
  state′` against the sampled state;
- **state novelty**: the nearest training state (correlation and normalised
  distance) — a sample that reproduces a training state is not a new state;
- **video novelty**: the nearest training video to the generated one, on the
  same 40 × 721 hexals (correlation and normalised distance). Without this
  number the result is stated as "generated without a source clip" and never
  as "a video that exists in no clip";
- **diversity**: pairwise correlation between sampled states and between their
  videos, on the scale of the pairwise correlation of real clips' states and
  videos;
- **what the brain reads**: the T4 direction energy after the round trip
  (`prompts14.direction_energy`), so a sample is described by the motion it
  carries.
Controls, **rebuilt inside this run's own code path** so every number in the
table shares one scale (the 0.96 / 19.1 pair above is what happens otherwise):
a held-out clip as the reachable reference, white and structured noise in the
types (14.0's construction), a shuffled clip state, and 17.0's unconditional
samples as the "no state at all" row. Local CPU if sampling and the round trips
fit in minutes as they did in item 14 ($0); otherwise one T4 pass, ≈ $0.05.

*17.2 status (2026-09-20), measured, local CPU 41 s, $0:* **the main gate
fails.** 16 sampled states give a round trip of 1.19 (1.06-1.37) where a
held-out clip's own state gives 0.006-0.061 and a *shuffled* clip state 1.77
(white noise 2.19, structured noise 3.04) — a sampled state is not a reachable
state, so novelty (+0.12 against a real clip's +0.68) and diversity (states
0.32 against 0.72) do not count for anything yet. Measured causes: not the
sampler (50/100 Euler steps give 1.34/1.40), not the plumbing (a real state
through the same path gives 0.006); the samples are too white — temporal
lag-1 0.56 against 0.99, ring-1 0.34 against 0.80, cross-type coupling 0.16
against 0.35 — and the velocity field is right near the data (denoising a real
state at t = 0.9: r 0.993, structure 0.98/0.79) and mean-like away from it
(t = 0.1: r 0.615, structure 0.52/0.37), so a trajectory from pure noise never
enters the structured region. `reports/2026-09-20_step17_state_prior.md`,
`reports/figures/2026-09-20_malecns_prior17.gif`. **State novelty against the
full training set is still unmeasured** (the states live in the 5.5 GB
`maps_deep.npz` on the volume; `sample17` in the Modal app measures it there,
≈ $0.10) — it is pointless until a sample passes the round trip.

*17.1b status (2026-09-20), measured, one T4 676 s ≈ $0.14 + local CPU $0:*
the human's diagnostic — **scene states only** (1,695 Sintel states of the
6,725; the set was 76 % procedural and the task is a video generator) in a
**compact temporal basis** (DCT, 16 of 40 coefficients, each z-scored). K was
chosen by a free local sweep first: band-limiting a *real* state costs a round
trip of 0.58 at K = 8, 0.20 at K = 12, 0.045 at K = 16 against 0.019 for the
full state, so K = 8 would have capped the test. Result: **round trip 0.142**
(0.119-0.176) against 17.1's 1.19 — 8× better, no longer in the class of a
shuffled state (1.77), still ~20× above a real clip (0.006) and ~15× above
what this representation allows (0.009). The structure of the samples now
matches real states — temporal lag-1 0.98, one-ring spatial 0.77, cross-type
coupling 0.31 against 0.99 / 0.80 / 0.35 — and it is learned, not imposed: the
basis alone gives 0.77 / −0.00 / 0.01. The speckle is gone (video
frame-to-frame 0.026 against 17.1's 0.280, a real clip's 0.057); what appears
is oriented structure and large moving regions, **not scenes with objects**.
Video novelty +0.41 against 17.1's +0.12 (a real clip +0.68), samples still
differ from each other. Two factors moved at once (data and basis), so this
run does not separate them.
`reports/2026-09-20_step17_1b_scene_dct_prior.md`,
`reports/figures/2026-09-20_malecns_prior17b.gif`. Next per the human's plan,
not started: a large set of ordinary video with procedural stimuli as a
minority, then the prior retrained on it; a wider SiT is explicitly **not** to
be run before that.

**The human's decision, nothing started:** (1) more capacity, same design
(width 192-256 or depth 6; ≈ $0.4-0.9 per run — the per-run cap was lifted
by the human on 2026-09-20, the permission-per-run gate was not); (2) the documented fallback — compress the state and put the flow
in the compact space (a small autoencoder, or a fixed temporal-DCT basis that
matches the measured smoothness and needs no fitting), cheaper per run and
aimed at the measured cause; (3) stop the prior line, keep 17.0 as the answer
to "video without a source clip" and spend the next money on item 16 or on a
denser state set (17').

**17.3 The prior as a control surface** (local, $0, only after 17.2 passes).
Two parts, and only the first is defined today:
- **Defined, and measured 2026-09-20** (local CPU, 35 s, $0,
  `flydream/generate/noise17.py`, `prior17.invert` / `to_noise` / `slerp`):
  the prior's flow runs backwards, so the noise a path is drawn through is not
  limited to states the prior sampled — a **real** clip's state has its own
  noise. The inversion is exact (noise → state → noise, r 0.99998 with four
  fixed-point iterations per step; the explicit backward step loses a quarter
  of the vector), a clip's state survives the trip at r 0.96 with round trip
  0.072 / 0.191 against 0.023 / 0.038 for the real states, and the noises of
  two clips mixed on the sphere give a monotone morph (r to A +0.96 → +0.41,
  to B +0.37 → +0.96) whose every point stays reachable (0.17-0.22 against
  1.585 for a shuffled state). The midpoint is the farthest of the row from
  the training set (nearest training video +0.59 against +0.99 / +0.96 at the
  ends). It also gives the first *coverage* number: a real state's noise sits
  at ‖ε‖²/D = 1.07 and 1.29 where a Gaussian is 1.000 ± 0.014, an unreachable
  state at 4.16 — the prior is near, not on, the real states.
  `reports/2026-09-20_step17_3b_noise_inversion.md`,
  `reports/figures/2026-09-20_malecns_prior17b_noise.gif`.
- **Not defined, to be designed and tested as its own task:** pulling an
  off-manifold state (14.0's random state, the hand-written stripe, later a
  spontaneous state from item 16) onto the learned manifold. An unconditional
  flow `noise → state` is a sampler, not a projector, so this needs a
  mechanism and a check of its own: optimising z so that the flow's output
  matches the target state, an inversion / partial-noising-and-denoising
  procedure, or an autoencoder whose encoder gives the compact latent (the
  same component as the 17.1 fallback). Which one, and whether the result is
  still reachable by the brain, is measured before this is called a
  projection; item 16 depends on the answer, so it is not assumed here.
  *First evidence (2026-09-20, local CPU, $0, `flydream/generate/clip17.py`,
  `prior17.refine`):* the partial-noising route works on a real state — clip A
  (the face clip of items 8-14) taken to noise level t and finished by the
  17.1b prior gives round trip 0.054 / 0.093 / 0.134 at t = 0.8 / 0.6 / 0.3
  with the video still correlating +0.97 / +0.95 / +0.88 to the clip (the real
  state: 0.023 at r +0.99; a pure sample: 0.145 at r −0.17). So the prior can
  rewrite a state by a chosen amount and keep it reachable. It is *not* yet
  shown to pull an **off-manifold** state (14.0's noise, the hand-written
  stripe, item 16's spontaneous activity) onto the manifold — that is the
  measurement 17.3 still needs.
  `reports/figures/2026-09-20_malecns_prior17b_clipA.gif`.

**17.4 Report and artefacts.** `reports/<date>_step17_brain_state_prior.md`
with the runs and the controls, 17.0's baseline beside 17.2's samples in one
table; `reports/runs.jsonl` per measured outcome; row clips per
`docs/ARTEFACTS.md`: state map / video / what the brain read, with the round
trip and both nearest-neighbour distances under each, the controls and the
17.0 baseline as their own rows with their own maps.

**What counts as done.** (1) The sampled states' round trip is close to a
held-out clip's and far above the noise states, on this run's own scale;
(2) both nearest-neighbour distances — state and video — show the samples are
not copies of the training set; (3) samples differ from each other; (4) the
clips are watchable; (5) the result is stated against 17.0: what the prior
adds over what unconditional 13B already does. A beautiful clip without (1) is
the 13B prior's texture, not a brain state (14.0 measured exactly that); a
clip that passes (1) but not (2) is a training clip remade.

**Limits stated in the report, not discovered by the reader.** The prior
learns the distribution of *our* clips (Sintel's 23 scenes with flyvis's
augmentations, plus 13A's procedural stimuli), so a "new" state is new inside
that distribution; and 13B's own amortisation gap (0.031 against the
inversion's 0.009) bounds how compatible any sample's video can be.

**17', more states — superseded by item 18 below.** The human's document §16
asked for 100k+ clips covering translation, rotation, looming, optic flow,
several local motions, occlusion and dynamic textures. Item 18 is that, at the
size the measurements justify and from ordinary video rather than from
generated stimuli.

## Current approved step: 18, the corpus of ordinary video

The human, 2026-09-20: "правило «$0.50 на обучение» больше нет"; "можно
закладывать переобучение 13б, но сначала проверить его на новом приоре";
"датасет выбери сам, можешь скачивать" (`DECISIONS.md`, same date). Item 17's
prior learned 1,695 states from **19 Sintel scenes**, and two measurements
point at the set rather than at the method: the samples sit at a round trip of
0.142 where the representation itself allows 0.009 (17.1b), and real states
invert to a noise radius of 1.07-1.29 where the prior's own draws sit at
1.000 ± 0.014 (17.3b) — the density is beside the real states, not on them.

**18.0 The corpus** (local CPU, $0 apart from the download). Source: **UCF101**
(13,320 clips, 101 action classes, 320×240, ~25 fps, 6.6 GB, research use) —
chosen for the number of *distinct* scenes per gigabyte, which is the quantity
17.1b lacked; its low resolution costs nothing, because the eye is 721 hexals.
Every clip passes through `flydream/data/video_hex.py`, which is flyvis's own
chain in flyvis's order (centre-crop 0.7 → `split` → `BoxEye` extent 15 /
kernel 13 → linear resampling to 1/dt → `HexRotate`/`HexFlip`), so a new clip
is the same kind of object as a Sintel one. Two rules are ours:
- **the frame is resized to Sintel's 1,024 px across, aspect kept**, so an
  object subtends the same angle and moves at the same speed across the eye
  (T4/T5 are speed-tuned; rendering a 320-px video at its own scale would be a
  different world, not a bigger one);
- **selection, not collection**: ordinary video carries cuts (a flash across
  the whole eye), tripod shots (no motion at all) and flat frames, none of
  which exist in Sintel. `motion`, `contrast` and `cut_score`
  (`flydream/data/video_corpus.py`) keep a window only inside the band the
  real Sintel clips occupy, and `--probe` measures that band on the corpus
  before any threshold is fixed.
Target ≈ 24,000 clips of 45 frames from ≈ 6,000 distinct videos — about 13×
17.1b's independent content — with one hex rotation per window, cycled, for
direction balance. Procedural stimuli stay as a minority (≈ 20 % of the final
set, reusing 13A's 7,200) for motion-space coverage; the human's own framing
(2026-09-20) is ordinary video first, procedural as a small part.
*Verification of the chain (2026-09-20, free):* rendering Sintel's own frames
through it reproduces flyvis's own item at r = 0.998, slope 1.002, offset
−0.0003; the residual (sd 0.018, temporally anti-correlated) is a sub-frame
interpolation phase, not a difference of eye or scaling.

**18.1 The states** (one T4, ≈ 20 min, ≈ $0.20, the human's word first).
The same function that built 13A's pairs: videos on the volume → frozen brain
→ the 8 T4/T5 types × 45 frames × 721 columns, z-scored per type with the
**13B statistics** so the old and new states share one scale. Held-out split
by *class*, as 13A held out scenes.

**18.2 The two checks before anything is retrained** (local CPU, $0).
(a) The corpus against Sintel: motion, contrast, brightness and the state
statistics (lag-1, one-ring, cross-type), so a difference in the prior later
is attributable. (b) **13B on the new distribution**: ~16 held-out corpus
clips → their real states → 13B → the round trip, against 0.023-0.038 for
Sintel clips in the same code path. If 13B renders the new states as well as
the old ones, it is not retrained.

**18.3 The prior retrained** (done: one T4, 1,128 s, ≈ $0.22).
Same `SiTStates`, same DCT-16 basis, on the new corpus, 20,000 steps; then
17.2's gates and 17.3b's noise radius in the same code path, so the numbers
land beside 0.142 and 1.07-1.29.

**18.4 13B retrained — not needed.** *Measured 2026-09-20 (18.2b): 13B renders
a state that came from ordinary video at a round trip of 0.016 against 0.011
on its own Sintel states, with an unreachable state at 1.048.* The human's
condition is not met, so 13B is not retrained and its ≈ $0.22-0.40 is not
spent.

**Status, all measured 2026-09-20.**

| | result |
|---|---|
| 18.0 corpus | **15,514 clips** = 12,411 of 13,320 UCF101 files (93 % passed the derived band, 0 failed) + 3,103 procedural (20 %); 1.03 GB; built in 543 s locally, $0. The eye chain reproduces flyvis exactly (r = 1.000000) and is 25× cheaper than `BoxEye`. |
| 18.1 states | 605 s on a T4 at batch 32, utilisation 69 %; maps (15,514, 45, 8, 721) on 13B's per-type scale, compact DCT-16 file (2.86 GB) keeping 99.32 % of the energy; 0 dead clips. ≈ $0.15. |
| 18.2 checks | corpus vs Sintel: motion 0.0248 / 0.0118, contrast 0.272 / 0.241, state lag-1 0.975 / 0.982, one-ring 0.875 / 0.866, cross-type 0.348 / 0.349. 13B: 0.016 / 0.011 / 1.048. $0. |
| 18.3 prior | **round trip 0.095** (0.077-0.117) against 17.1b's 0.142; real clip 0.012, DCT-16 ceiling 0.021, shuffled 1.313, white noise 2.177. Video novelty +0.35 against +0.55 for a real clip (bank = the corpus). Noise radius of a real state 1.110 / 0.970 against Gaussian 1.000 ± 0.014 (was 1.073 / 1.290); a clip survives the trip through its own noise at r 0.98-0.99, and mixtures of two clips score 0.050-0.066 — better than unconditional samples. One T4, 1,128 s, utilisation 98.5 %, ≈ $0.22. |
| 18.4a throughput | the step is **per-sample bound, not launch-bound**: 2.44 ms fixed + 1.6229 ms per sample, so 4.5 % of a batch-32 step is fixed cost and an 8× batch buys **+4.9 %**. Removing the per-step `loss.item()` (54.81) and fusing the EMA loop (54.94) change nothing against the base 54.72. **`torch.compile(reduce-overhead)` 38.47 ms = 1.42×**, warmup 41 s once — a 20k-step run becomes ≈ 810 s, $0.22 → $0.16. One T4, 120.6 s, ≈ $0.03. |
| 18.4b learning rate | **1e-4 is 6.2× worse than our 3e-4** on the main gate (0.591 against 0.095) and 0.7328 against 0.5064 on validation, with identical controls. One T4, 1,243 s, utilisation 97.5 %, ≈ $0.24. |
| 18.4c the sweep's control | 18.3's configuration with `torch.compile`: gate **0.090**, validation 0.5052, 920 s against 1,128 s (**1.23× end to end**). The 0.095 → 0.090 difference is the measured size of fusion plus one seed, and the reason every arm below is compared to this run. ≈ $0.18. |
| 18.4c higher learning rates | 6e-4 → **0.080**, 1e-3 → **0.076**, against the control's 0.090. With 18.4b the series 1e-4 / 3e-4 / 6e-4 / 1e-3 = 0.591 / 0.090 / 0.080 / 0.076 is monotone: **the optimum is at or above 1e-3 and is not bracketed.** ≈ $0.34 for both. |
| 18.4d **capacity** | **width 192 (2.64 M par): the gate goes 0.090 → 0.034**, validation 0.5052 → 0.2977. The gap to the representation's floor falls from 4.3× to **1.6×**, and the prior's own share of the error from 0.069 to **0.013** — the floor now dominates. The largest single move in item 18. One T4, 1,215 s, ≈ $0.24. |
| 18.4d K = 32 | the floor is lower (99.913 % of the energy kept against 99.32 %) and the gate is **5.9× worse**: 0.527. At width 128 the model already binds, so twice the target dimension makes it relatively smaller. Revisit **after** capacity, not instead of it. Maps file by `dct_maps` (cpu 2 / 24 GB, 106 s, ≈ $0.03) built **without touching** the 9 GB original. One T4, 1,033 s, ≈ $0.23. |
| 18.4c 60,000 steps | **0.074** against the control's 0.090, validation 0.4046 — and converged (−0.41 % over the last 10,000 steps), which settles 18.3's open question. The four rate/length arms collapse onto the product **lr × steps**: 60 / 120 / 180 / 200 units give 0.090 / 0.080 / 0.074 / 0.076, so 60k at 3e-4 and 20k at 1e-3 are interchangeable — and the rate buys that product **2.8× cheaper** than steps. Width 192 sits at the same 60 units as the control at 0.034, off this curve entirely. One T4, 2,479 s, utilisation 95.0 %, ≈ $0.47. |
| 18.4e classes | 110 labels (101 UCF101 + 9 procedural kinds), DiT-style label embedding: **no effect**, 0.098 against 0.090, inside the spread. The published precedent (conditioning alone, FID 26.21 → 10.94) does not transfer — the research note named the reason in advance: no study covers a label only loosely coupled to the signal. One T4, 845 s, ≈ $0.17. |
| 18.5a **width 192 + lr 1e-3** | the two levers of 18.4 run together: gate **0.0224** against a DCT-16 floor of **0.0209**, validation 0.2480. The prior's own share falls 0.0133 → **0.0015**, below what 16 samples resolve — the two medians are indistinguishable, which is the claim, not "solved". On validation the levers **multiply to within 0.4 %** (×0.836 · ×0.589 predicts 0.2490). Novelty holds +0.452 (clip +0.554). **Consequence: no model-side lever can move the round trip at DCT-16.** One T4, 1,180 s, utilisation 95.8 %, ≈ $0.23. |
| 18.5b **K = 32 at width 192** | capacity is **not** what K = 32 lacked: the K=32/K=16 ratio is 5.9× at width 128 and **6.6×** at width 192 — it did not shrink, which refutes 18.4d's capacity reading. Both improve absolutely (0.527 → **0.224**) and novelty recovers +0.12 → +0.35. Measured alternative, offered as a hypothesis: per-coefficient z-scoring gives the top 16 of 32 coefficients **50 % of the loss** while they carry **0.45 % of the state energy** (112× over-weighting; 14× at K = 16). One T4, 1,229 s, utilisation 96.6 %, ≈ $0.25. |
| 18.6 **the loss weight** | 18.5b's hypothesis **confirmed**: weighting the loss by `sd^1` across the coefficient axis takes K = 32 at width 192 from **0.2243 to 0.0346, 6.5×**, while the unweighted validation moves 2.4 % (0.7360 → 0.7185) — by validation alone this reads as a null result. `sd^2` (the raw state space) was rejected before spending: it gives coefficient 0 80.4 % of the loss and an effective K of 1.54 of 32. **Not** done: K = 32's floor 0.0115 stays out of reach, own share 0.0231 against 0.0015 for K = 16, and **K = 16 remains better overall (0.0224)**. Two changes at once (weight and rate); the rate alone was predicted at 0.147 before the run. One T4, 1,215 s, utilisation 96.5 %, ≈ $0.25. |

`reports/2026-09-20_step18_corpus_of_ordinary_video.md`,
`reports/2026-09-20_step18_3_prior_on_the_corpus.md`,
`reports/2026-09-20_step18_4_throughput_and_learning_rate.md`,
`reports/2026-09-20_step18_5_width_and_the_representation_floor.md`,
`reports/2026-09-20_options_after_the_floor.md` (**варианты, не план**),
`reports/figures/2026-09-20_malecns_check18.gif`,
`..._prior18.gif`, `..._new_video18.gif`, `..._bench17.png`, `..._lr18.png`,
`..._arms18.png`, `..._scaling18.png`, `..._width18.png`, `..._weight18.png`,
`..._classcmp18.gif`, `..._cmp18.gif`, `..._13b18.gif`, `..._why18.png`, `..._steps18.png`, `..._edges18.png`, `..._axis18.png`, `..._new384.gif`,
`..._new_video18_w192.gif`.

**What the numbers answer.** Data was the limit, not the method: the same
model and the same code move the gate 0.142 → 0.095 and the coverage 1.290 →
0.970. What they do not give is a scene: the samples are structured moving
texture, 8× a real clip's round trip.

**Where the remaining error sits, and the options** (nothing started, the
human decides). *The options below concern the round trip, which 18.5a drove
to the floor of DCT-16; the options for what to do **after** that — 13B, an
embedding in place of the class label, a scene metric, a learned temporal
latent — are written up as options in
`reports/2026-09-20_options_after_the_floor.md`, none of them started.* Of the 0.095, **0.021 is the floor** — a real state
band-limited to the same DCT-16 scores exactly that — and **0.074 is the
prior**. So:
1. ~~**The floor**~~ — **measured twice. 18.4d: K = 32 is 5.9× worse** at
   width 128, read then as capacity. **18.5b tested that reading and it is
   wrong**: at width 192 the penalty is 6.6×, no smaller. The floor is now
   what binds — 18.5a's 0.0224 sits on the 0.0209 of DCT-16 — but raising K
   costs more than the floor it buys, and the open lead is the loss weighting
   (the top half of the K = 32 coefficients takes 50 % of the loss for 0.45 %
   of the energy), not a bigger model — **measured 2026-09-20 (18.6): that
   weighting is worth 6.5× at the gate**, and K = 32 still does not beat
   K = 16 (0.0346 against 0.0224), so the lower floor remains out of reach.
   K = 40 needs no run: the time axis is 40 frames, and K = 32's floor
   (0.0115) is already below a real clip's own round trip (0.0119).
2. ~~**The learning rate**~~ — **closed by measurement 2026-09-20 (18.4b)**.
   The research flagged our 3e-4 at batch 32 as 8.5-24× above the
   extrapolation of DiT/SiT's 1e-4 @ 256. Run as a paired arm, nothing else
   changed: **1e-4 gives a round trip of 0.591 against 0.095, 6.2× worse**,
   and a validation loss of 0.7328 against 0.5064, flatter at the end rather
   than merely late. The literature's value does not transfer; ours stands.
   ≈ $0.24 spent, the question does not need asking again.
   *Open follow-up, untested:* two points are not a sweep, and the shape of
   the curves is consistent with an optimum **above** 3e-4.
3. ~~**Classes**~~ — **measured 2026-09-20 (18.4e): no effect** (0.098 against
   0.090, inside the spread). The risk named when this was proposed — a label
   only loosely coupled to the signal, with no study covering that case — is
   what happened. ≈ $0.17 spent, the option is closed.
4. ~~**Capacity**~~ — **measured 2026-09-20 (18.4d) and it was the answer.**
   Width 192 moves the gate 0.090 → 0.034 and cuts the prior's own share of
   the error from 0.069 to 0.013. The literature reading written the same
   morning ("growing is safe, nothing says capacity is the cause") was right
   about safety and wrong about cause; ≈ $0.24 settled it.
5. ~~**More steps**~~ — **measured 2026-09-20 (18.4c): it works and it is the
   expensive way.** 60,000 steps converge at a gate of 0.074, but 20,000 at
   lr 1e-3 reach 0.076 for **$0.17 against $0.47**. Length and rate buy the
   same product; the rate is 2.8× cheaper per unit of it.
6. **Free, local, no GPU**: an MMD permutation test (validated at 5-10 samples
   per group), the same checks in a PCA-32 feature space (rankings stable at
   N = 50) and Carlini's neighbourhood-relative threshold in place of our
   arbitrary correlation cutoff.
Also measured and usable now: the path between two real states in noise space
scores 0.050-0.066, better than sampling from scratch — a practical route to
new-but-reachable video that needs no further training.
**A bound on our own claim**, from the same research: nearest-neighbour
distance tests do not detect instance-level memorisation, so 18.3's +0.35
against a real clip's +0.55 means "not a near-copy", never "not memorised".
Sources, prices and the research's own cost:
`reports/2026-09-20_research_video_generation_and_training.md`, notes in
`research_notes/2026-09-20_video_generation_and_training/`.

## Current state of the system

- **Data:** MaleCNS v1.0 core files and the FlyVis 1.2.0 ensemble under
  `data/` with hashes in `data/manifest.json`; the right optic lobe exported
  as `data/ol/filters_R_w5wk50m500oc.json` (60 types, 677 type pairs, 31,526
  neurons, 1.39 M edges; weak-pair exception, outputs restricted to the 18
  columnar types); Sintel under `data/flyvis/SintelDataSet`. Settings in
  `config.toml [data]`.
- **Model:** model zero (`flydream/model/zero.py`), FlyVis's network on the
  MaleCNS export with a member's parameters transplanted by type, gain
  rescaled to total input per target and capped (v9: DSI 0.152 against
  FlyVis's 0.391, flash polarity and direction at or above FlyVis, stable on
  three members). Not trained on MaleCNS; training is paused (3').
- **Decoders:** `flydream/decode/` — ridge per cell type, hex-conv head,
  metrics, raster, the map driver, the lag sweep, the ensemble aggregation
  (beyond the track). The window is part of the measurement (ISS-0004),
  consecutive lags only (ISS-0006).
- **Generator:** `flydream/generate/` — `invert.py` (encoder inversion, the
  reference answer), `gen13b.py` (the conditional flow generator, checkpoint
  `data/gen13b/sit.pt` and `flydream-runs:/gen13b/sit.pt`), `roundtrip13.py`
  (the round trip on built states), `prompts14.py` (random, edited and
  composed states, the closed loop).
- **Compute:** the owner's machine (32 cores, 102 GB, CPU) for everything that
  fits in hours; Modal T4 (`flydream-train`, `flydream-decode`,
  `flydream-generate`) for GPU-bound work; `tools/modal_watch.py` to watch.
- **Tests:** 77 offline tests passing (2026-09-20, 23 s), no download, no
  Modal, no credential.
- **Spend recorded in `reports/runs.jsonl`:** $3.35 in total, of which 13A
  ≈ $1.61 (the pairs, the three decoders, the round trip on 11-12) and 13B
  ≈ $0.42; item 14 ran locally at $0.

## Done

- **Research: the landscape and the plan** (2026-09-18).
  `reports/Коннектом мухи и план проекта.md`; training and speed-up of
  connectome models: `reports/Обучение коннектомных сетей и ускорение.md`.
- **The records** (2026-09-18): `ROADMAP.md`, `AGENTS.md`, `DECISIONS.md`,
  `ISSUES.md`, `docs/`; `docs/ARTEFACTS.md` split out 2026-09-20.
- **1, data** (2026-09-18): MaleCNS core files and the FlyVis ensemble with a
  manifest; the type bridge (61/65); home columns from neuPrint column ROIs;
  hex axes aligned to FlyVis; the export in FlyVis shape (0.869 of FlyVis type
  pairs, Spearman 0.752, sign agreement 0.951); the weak-pair exception and
  columnar outputs. `reports/2026-09-18_step1_data.md`; ISS-0001, ISS-0002,
  ISS-0005.
- **2, model zero** (2026-09-18): FlyVis dynamics on the MaleCNS right lobe
  with transplanted parameters, corrected to preserve total input per target
  with a cap (ISS-0003). DSI 0.152 (v9) against FlyVis's 0.391, flash polarity
  and direction at or above FlyVis, three members stable.
  `reports/2026-09-18_step2_model_zero.md`.
- **4, the decoder ladder** (2026-09-18/19): ridge per cell type with
  time-shuffle and sample-shuffle controls; the first stage curve withdrawn by
  its own audit (ISS-0004), even-spaced lags alias the frame hold (ISS-0006);
  closed at the minimal shape, ladders at 0 and 80 ms on both brains.
  `reports/2026-09-18_step4_decoder_stack.md`.
- **8, the generator by encoder inversion** (2026-09-19): from every stage,
  T4/T5 alone included, the clip that caused the state comes back (r
  0.92-1.00); a state from another clip gives that other clip; minutes on a
  T4. Figures `reports/figures/2026-09-19_*generator*`.
- **9, the window's end constrained** (2026-09-19): the generator fits
  `frames + margin` (40 + 5) and shows `frames`; last-frame r T5a 0.66 → 0.92,
  T4+T5 0.80 → 1.00, mean r unchanged. ≈ $0.17.
  `reports/2026-09-19_step9_window_margin.md`.
- **10', the ladder batched** (2026-09-19): 20 tasks in one pass with a
  per-task masked loss, 948 s → 176 s on a T4 (7.5×), 66 % utilisation,
  identical videos; a ladder ≈ $0.05. Same report, §"Item 10'".
- **11, dreams-lite** (2026-09-19): inversion of states no clip caused — noise
  into the eye, a flash, the dark after a clip, noise inside the neurons —
  beside a shuffled-state control, ≈ $0.15. Eye noise comes back (r 0.91-1.00;
  Tm5a 0.53, T5a 0.31; control ≈ 0); the flash by its timing without texture;
  a faint after-image for a few dark frames from L3 onward; internal noise as
  quiet ripple; the shuffled state draws the wiring's own diagonal stripes.
  `reports/2026-09-19_step11_dreams_lite.md`.
- **12, manipulated states** (2026-09-19): per-type normalised loss, 15 tasks
  in one T4 batch (≈ $0.05). Gain edits of one type leave the clip in place
  (r_A 0.95-1.00) with a residual 2,000-8,000× the control's — unreachable
  states; a mix of T4a states is the state of the mixed video (r 0.99 at
  α = 0.5, monotone in α); hybrids are a contest T4/T5 win.
  `reports/2026-09-19_step12_manipulated_states.md`.
- **13A, amortised inversion** (2026-09-20): pairs from the frozen brain
  (Sintel plus procedural stimuli), three input conditions, linear
  hex-temporal and CNN decoders against the inversion. Deep: linear r 0.93 /
  round trip 0.036, CNN 0.96 / 0.029, inversion 1.00 / 0.009 — the deep state
  reads out in one pass and the nonlinearity matters only there. On the 16
  states of 11-12: reachable states within 2-4× of the inversion and 100-300×
  below the shuffled control; on unreachable ones every method's error is
  large and the decoders diverge from the inversion.
  `reports/2026-09-19_step13a_amortised_inversion.md`.
- **13B, the conditional flow generator** (2026-09-20): design
  `reports/2026-09-19_step13b_design.md` (revision 2), result
  `reports/2026-09-20_step13b_generative_decoder.md`. No multimodality at the
  full deep state (8 random starts → one solution); SiT backbone at 0.057
  s/step; SiT 128 × 4 trained 20k steps, val 0.0100, $0.22; held-out clips
  median round trip 0.031 / r 0.966 (control 0.96); the knobs work (T4 only
  0.033, T5 only 0.112, T4a only 0.309 with seed spread appearing there);
  guidance > 1 hurts; the conditioning-strength test passes (r 0.03 / 0.10);
  gain edits are answered by the prior, as with the CNN.
- **14, what can be prompted** (2026-09-20, local CPU, $0):
  `reports/2026-09-20_step14_controllable_generator.md`. Random states give
  the prior's texture (1.9 / 2.3 against control 19 and clip 0.03); the
  direction channels are not knobs (swaps, rotation, time reversal,
  amplification: 0.7-9.6 with the motion unchanged); compositions of brain
  states by region are reachable and render as asked (0.06-0.15, the brain
  reads the composite direction back); a hand-written T4a stripe is ignored;
  the closed loop from a clip drifts — every pass compatible (0.02-0.05) but
  r to the clip 0.99 → 0.26 in 11 passes — and noise is a fixed point. This
  step is why item 17 exists.
- **3, training priced** (2026-09-18): T4 smoke, packing, batch sweep,
  optimisation benchmark; batch 16 saturates a T4 at 14 samples/s,
  `stats_relu` 1.17×; the reference schedule (~$10-15 per member) is over
  budget and withdrawn. `reports/2026-09-18_step3_training_options.md`,
  `reports/2026-09-18_training_optimization_bench.md`.

## Paused in this track

- **3', fine-tuning MaleCNS from the transplanted weights** (the human,
  2026-09-19: "не сейчас"). Two members × 1,200 iterations at
  batch 16 with `--variant stats_relu`, checkpoints at 0/900/1200, validated
  like model zero. Bought only if a generator on model zero turns out visibly
  worse than on FlyVis, or when the human wants a trained MaleCNS model.
  Commands and prices: `reports/2026-09-18_step3_training_options.md` §5б.
- **10, a dense MaleCNS export: every type on all 721 columns.** Types with
  fewer than 721 cells tiled onto the empty columns with the type's shared
  filters; removes the Tm5a/T5a lattice from the pictures; every artefact says
  which cells were tiled in. A day of local work, cosmetic.
- **A 64×64 / 128×128 raster of the hexals** for the outputs: a rescale of the
  721 values, no information added. Minutes.

## Beyond the track

Kept, not on the path; nothing here is refuted, and each waits for the
human's word.

- **16, the dream source inside the model** — the brain produces T4/T5 states
  with no video and 13B turns them into video. Design (draft)
  `reports/2026-09-20_step16_dream_source_design.md`: 16.1 structured
  spontaneous drive into model zero as is; 16.2 close the optic-lobe loops the
  export filter dropped (LPi, Dm, Pm, TmY16/19a, Y — 25 % of the synapses onto
  the ladder, and 91 % of T4/T5's output goes to types model zero lacks), with
  stability and validation as gates; 16.3 central-brain drive. **Status
  2026-09-20: not yet fully formed, not yet fully discussed** (the human).
  The human's revision `docs/ideas/step16_dream_source_revised.md` is the
  companion to that report and takes precedence where they differ (parameters
  as ranges with a sensitivity sweep, sign from neurotransmitter annotations;
  loops ON/OFF and shuffled-topology controls; 16.1 is an artificial-drive
  baseline, not "the brain generated the state"; result levels A/B/C).
  Deferred; the discussion resumes on the human's word before any build.
  Item 17 may feed it — a spontaneous state brought onto the learned manifold
  before it reaches 13B — but only once 17.3 defines and measures such a
  projection; an unconditional flow over states does not provide one.
- **5, both eyes, every column, the missing biophysics.** ~880 columns per
  eye and the left eye; photoreceptor temporal filter with luminance-dependent
  speed, R1-6 gap junctions, contrast adaptation, each validated (Pang et al.
  2024, Drews et al. 2020). Needed for HDRI / 360° input and two-eyed
  behaviour, not for clips.
- **6, the central brain and the state knobs:** LC types, optic glomeruli, the
  central complex from MaleCNS in one model; trained on a behavioural task or
  fitted to whole-brain data (DANDI 000727); knobs — octopamine gain, R5
  slow-wave gating, mean luminance. A large item: new export, a long training
  run, validation.
- **7, dreams, the full protocol** (the rigorous form of item 11): decoding in
  the dark against the stimulus history; internally generated activity with
  the fraction of variance in the stimulus subspace measured before anything
  is decoded, and a shuffled-connectivity control.
- **Deeper than T4/T5 with generic dynamics:** the human's note
  `docs/ideas/malecns_shiu_lif_baseline_idea.md` — MaleCNS wiring with
  Shiu-style LIF dynamics as a baseline that reaches LPi/VS/HS, the visual
  projection neurons and the central brain, with the FlyVis-based model kept
  as the calibrated visual reference. Unapproved, unpriced; it collides with
  DECISIONS 2026-09-18 on the model class, so it needs that decision revisited
  first.
- **Multi-level conditioning of the generator** (the human's architecture
  document §13): early, deep and central states together as the condition
  instead of T4/T5 alone. After 17 and after a deeper model exists.
- **C, colour as an ML task** (the human, 2026-09-19; recorded as "14" that
  day, renamed here to avoid the collision with step 14). FlyVis reads one
  luminance in all eight photoreceptor types and has no colour pathway;
  MaleCNS has the types (R7p/y, R8p/y, Dm8a/b, Dm9) but no parameters to
  transplant. Until then colour in an output is a display overlay from the
  source clip and says so.
- **15, HDRI / 360° panoramas as the stimulus source** (the human,
  2026-09-19): a virtual fly camera inside a panorama (Poly Haven, CC0)
  rendered onto the 721 ommatidia with its orientation as a parameter —
  unlimited clips with exact optic flow from rotations, against Sintel's 23
  scenes. A day for the renderer; it would serve 17' directly.
- **Paper-grade rigour:** the ensemble map (ten members × five splits), per-type
  nulls, subset curves, the consecutive-lag sweep on more than one member,
  from-scratch training controls, validation against the 26 physiology
  studies, the reference training schedule. Code exists and is tested;
  measurements stopped 2026-09-18 night.
- **The connectome as a computational substrate** (an image generator or a
  language model made of fly wiring). Parked 2026-09-18; if revisited, a
  degree-preserving shuffled connectome must do measurably worse or nothing
  was shown.
- **One-liners, recorded only:** the three-connectome comparison (FlyVis,
  FlyWire, MaleCNS on one map); a second training objective; the embodied loop
  (the model as the eyes of a MuJoCo fly); inversion on real recordings (Pang
  et al. 2024); photoreceptor columns from lamina cartridges; the left optic
  lobe export; the fitted-dynamics parameter source; ablations on model zero
  (CT1 restored, ISS-0002); per-target / per-pair filter normalisation
  (measured unstable at step 1; `"src"` stands).

## How this file is kept

- **Only approved work.** A conclusion the human has not approved in words is
  a draft and belongs in `reports/`, or stays in `docs/ideas/` if it is the
  human's own note.
- **State and order, not reasoning.** No options, comparisons, prices or
  research; those go to `reports/`, durable choices to `DECISIONS.md`.
- **One entry per item, a few lines, plus links.** Evidence lives in the
  report it links to and is not summarized twice.
- **Done is a list of outcomes**, not a history of how they were reached.
- **Unfinished work returns as its own item**, never as a caveat inside a
  closed one.
- **Two groups outside the track:** "Paused in this track" for work on this
  track the human stopped for now, "Beyond the track" for everything else.
- **Short beats complete.** If this file needs a table of contents, cut it.
