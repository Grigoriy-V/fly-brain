# I. ML review of the state-generation track (steps 16-29)

Reviewer: Claude (Opus), subagent of the 2026-09-24 whole-project review.
Scope: models, representations, objectives and evaluations of steps 16-29,
judged against "a working generator the owner can look at", not against
paper-grade rigour. Method: read `ROADMAP.md`, `docs/PROJECT_MAP.md`, the 21
step and research reports of this track, the code they used; ran eight
read-only checks on saved local arrays (CPU, each under 20 s, $0; code in the
appendix). Nothing in the repository was changed except this file.

## 0. Verdict in six lines

1. Most of the track's money (the `2026-09-20_prior18_*` runs: $7.19 of
   ~$9.2 recorded) went to sweeps selected by a judge that measures
   reachability, not scenes, and to a "seed" diagnosis that could not move a
   learned density; a distribution-level judge was proposed on 2026-09-20 as
   "first, $0" and is still not built.
2. Two representation detours (flat PCA latent 19-22; a VAE asked to make its
   latent N(0, I), 18.24) were predictable from literature the track later
   cited and from its own 18.5b loss-weighting finding.
3. The one architectural lever that paid - lattice locality (23) - is the
   standard inductive bias for spatial data; it came only after the whole
   17.1-22 sequence (69 run records, 31 of them priced, in `runs.jsonl`) had
   been spent on flat or global models.
4. Step 29's static flow is a real result at the state level (this review's
   two-sample tests: its draws are indistinguishable from training states
   where a covariance-matched Gaussian is not), but part of its reported
   renderer-level parity is a function of an unrecorded renderer ridge.
5. The static state is, to R^2 0.974, a linear function of the window's mean
   frame: at the static level the brain is a near-invertible linear change of
   basis, so "generate a static brain state" is almost "generate a 721-pixel
   grey picture". That makes the image-space generator the natural ceiling
   control and makes true still-frame states (brain pass measured at $0.15
   for the whole corpus) the natural target.
6. The resumed order is logically sound; the cheapest improvement is to
   replace "verify the conversion" (29.3) with "simulate true still-frame
   states", add a free state-level two-sample test now, and give 30 a
   covariance-matched Gaussian control.

## 1. Map of the track

| step | question | outcome (source) |
|---|---|---|
| 16 | where would a spontaneous "dream" source come from | design only; 25 % of input to the ladder and 91 % of T4/T5 output go to types model zero lacks; parked (`reports/2026-09-20_step16_dream_source_design.md` §2) |
| 17.0 | does unconditional 13B already make new video | yes, as texture; nearest-train r +0.53 vs real +0.52 (`..._step17_0_unconditional_baseline.md` §Numbers) |
| 17.1 | flow over full states (230,720 numbers, 1.4 M params, 6,725 states) | samples unreachable, round trip 1.19 vs real 0.006 (`..._step17_state_prior.md` 53-72) |
| 17.1b | DCT-16, Sintel only | round trip 0.142, speckle gone (`..._step17_1b_scene_dct_prior.md` 32-50) |
| 17.3b | invert the flow | exact to 0.6 %; real preimages at radius 1.07-1.29 (`..._step17_3b_noise_inversion.md` 53-70) |
| 18.0-18.3 | UCF101 corpus, 13,555 train states | round trip 0.095; split by class (`..._step18_3_prior_on_the_corpus.md` 28-38) |
| 18.4-18.6 | lr, width, K, classes, loss weight | width 192 + lr 1e-3 reaches the DCT floor 0.0224/0.0209; K=32 needs sd^1 weighting (`..._step18_5_...md` 36-73, 142-188) |
| 18.7-18.13 | why no scene | round trip penalises contrast; flat-fraction/coherence pair invented; w384 best 27 % vs real 46.6 % (`..._step18_5_...md` 308-324, 365-396, 638-676) |
| 18.14-18.16 | classes | first null was an embedding-init bug; fixed labels still do not pay (`..._step18_5_...md` 705-847) |
| 18.17-18.19 | where scenes are in noise | real preimages 66-81 sigma inside the shell; radius and common-direction fixes fail; round trip scores a grey field 0.0070 below a real clip's 0.0119 (`..._step18_5_...md` 870-926, 999-1026) |
| 18.20-18.24 | "seed problem": fixed coupling, preimage pairs, VAE with N(0,I) latent | 0/8; 8/8 but no generalisation; VAE loses to linear PCA 0.687 vs 0.890 (`reports/2026-09-20_the_seed_problem.md` §§ 4, 9-11) |
| research 09-21 | how VAEs are trained; path to a generator | nobody asks a latent to be N(0,I); locality is the known mechanism of novelty (`..._research_how_vaes_are_trained.md` §1; `..._research_path_to_a_video_generator.md` §§1, 4) |
| 19 | PCA-2048 whitened + flow over 16 x 128 tokens | seed returns clip (0.825 = PCA ceiling); fresh draw overshoots radius 1.55x, no scene (`..._step19_linear_first_stage.md` §§ 5, 8) |
| 20 | fix the sampler | geometry fixed by batch-sd projection, gate 2.2x better, still no scene; covariance is local 0.876 vs 0.267 (`..._step20_sampler_and_cuts.md` §§ 2-4, 6) |
| 21c | what the chain restores | Phi = brain o 13B restores a dropped T5 to r 0.979 with an honest mask (`..._state_restoration.md` § 9) |
| 22 | smaller object | T4a+T4b block (23,072) carries 0.884 of 0.904; PCA over the block does not help the flow (`..._step22_a_smaller_object.md` §§ 2, 5, 8) |
| 22.7-22.8 | metric audit | kurtosis 4.11 vs 8.31 +- 1.17; judges' floors measured; sharpness 11/100 (`..._the_draw_distribution.md` §§ 3, 8-9) |
| 23 | hex-local flow on the block | kurtosis 5.36, sharpness 22/100, seed returns clip at 0.884; $0.51 (`..._step23_hex_local_prior.md` §§ 3-4) |
| 26 | draw at the preimage radius | collapses to the corpus mean; kurtosis is buyable by shrinkage (`..._step26_corrections_at_draw_time.md` §§ 2-4) |
| 29 | a still picture from a 721 x 2 static state | still frame -> brain -> 13B at 0.953; 13B cannot render DC-only states; LS map DC -> mean frame r 0.941; static flow $0.42 matches real on flat fraction through LS (`..._step29_a_scene_as_one_picture.md` §§ 2-5) |
| 29.1 | why a slice | slice kurtosis 5.26 vs DC 3.17 (unverified script) (same, § 7) |

## 2. Checks run by this review

All on saved local arrays: `data/prior19/pca_ab2048{,_latent}.npz` (the block
PCA and its train/test latents), `data/corpus18/videos.npz`,
`data/prior29/draw29.npz` (256 static draws). Real static states are the
local PCA-2048 reconstruction, as in `flydream/generate/geom29.py:64-69`;
the true states on the volume were not read. Code in the appendix.

- **C1. The static state is almost a linear function of the picture.** A
  least-squares map from the window's mean frame (frames 0-39) to the 1,442
  DC numbers explains **R^2 0.974** of held-out DC variance (1,246 clips from
  ten unseen classes); the forward map DC -> mean frame reproduces the
  recorded r (0.938 here, 0.941 in `runs.jsonl` `2026-09-21_prior23_pic`).
  Frames 5-44 give 0.929 / 0.955, so frames 0-39 are the aligned window.
- **C2. Renderer-level parity depends on the renderer's ridge.** Through the
  LS map (first 128 of each set):

  | ridge | held-out r | real: flat / coherence | flow draws | Gaussian (train mean + cov) |
  |---|---|---|---|---|
  | 1e-6 | 0.938 | 25.7 % / 0.939 | 21.3 % / 0.841 | 19.6 % / 0.938 |
  | 1 | 0.939 | 25.8 / 0.940 | 27.8 / 0.898 | 19.7 / 0.939 |
  | 30 | 0.941 | 26.3 / 0.942 | 27.1 / 0.925 | 19.7 / 0.943 |
  | 300 | 0.943 | 26.9 / 0.946 | 26.5 / 0.941 | 19.7 / 0.948 |

  The recorded "27.1 against 27.1, coherence 0.928 against 0.949"
  (`runs.jsonl` line 150) is reproduced at ridge ~30, a setting recorded
  nowhere (the script was not kept, step-29 report § 9). Draws carry 1.5 % of
  their centred energy outside the top 200 real-covariance directions
  against 0.2 % for real states, which an unregularised map amplifies.
  Two readings follow: **coherence parity is explained by second-order
  statistics alone** (the Gaussian matches it at every ridge); **flat-fraction
  parity is not** (Gaussian 19.7 % at every ridge, flow 26.5-27.8 %) - that
  is the part of step 29 that shows learned non-Gaussian structure.
- **C3. State-level two-sample tests.** In the top 64 directions of the real
  DC covariance (95.3 % of its variance), n = 256 per side, against 256
  training states: draws MMD^2 p = 0.35, classifier two-sample test accuracy
  0.47 (MLP) / 0.49 (boosted trees); covariance-matched Gaussian p = 0.007,
  0.55 / 0.61; held-out classes p = 0.003, 0.51 / 0.56; train-vs-train control
  p = 0.88, 0.49 / 0.53. Memorisation: median nearest-training distance of a
  draw 5.83 against 5.86 for a training state to its other neighbours
  (held-out 7.59); NN1/NN2 < 0.5 for 0 of 256 draws. So the static flow's
  draws are new samples of the training distribution at this n, and a
  Gaussian with the same covariance is not.
- **C4. The right floor for the flow-matching loss is the Gaussian, not zero
  velocity.** For `x_t = (1-t) eps + t x1`, the best linear predictor of
  `v = x1 - eps` leaves `lambda / ((1-t)^2 + t^2 lambda)` per eigen-direction
  (averaged over `validate`'s t grid, `flydream/generate/prior17.py:270`).
  Whitened latent: **1.626** (flows 1.324-1.368); block with the training
  spectrum: **0.471** (hex flow 0.396); static DC from the reconstruction:
  **>= 0.181** (a lower bound, since reconstruction shrinks small
  eigenvalues; static flow 0.216). The reports read every val loss against
  the zero-velocity 2.0 ("80 % explained", `runs.jsonl` line 140), which says
  nothing about non-Gaussian learning; against the Gaussian every flow is
  ~16-19 % better and the static one is undetermined locally.
- **C5. The "19 degrees to 72 degrees" transport claim is a change of space,
  not of regime.** `accept23.py:85-89` measures the angle in each flow's own
  input space. In a whitened latent the Gaussian flow map is the identity
  (0 degrees), so 19 degrees is "slightly more than Gaussian". In the raw
  block the Gaussian map `mu + Sigma^(1/2) eps` with the training spectrum
  already gives **71.6 degrees** (law-of-cosines form as in `accept23.py`;
  77 degrees by Monte Carlo on the top 2,048 directions) against the hex
  flow's 72.0. ROADMAP:208-211 and step-23 report § 3 read it as the end of
  the near-identity defect; the like-for-like evidence for step 23 is
  kurtosis in one basis and sharpness through 13B, not the angle.
- **C6. The kurtosis gate mostly measured clip energy.** Per-coordinate
  kurtosis of the training latent (k = 1536, n = 256 x 40) is 8.31 +- 1.17
  (reproduces `the_draw_distribution.md` § 3); after normalising each state
  to the same radius it is **4.10 +- 0.16**. About 79 % of the excess over a
  Gaussian is one scalar per clip (how energetic the clip is), not spatial
  sparsity of scenes as read in § 5 of that report.
- **C7. Whitening re-created the 18.5b loss pathology.** In PCA-2048 the
  bottom half of the components carries 6.9 % (full state) and 8.5 % (block)
  of the variance and, once whitened, 50 % of the flow's loss - the same
  mismatch 18.5b measured for DCT tails (3.5 % energy, 50 % loss at K = 16;
  `..._step18_5_...md` 104-118), found the day before step 19.
- **C8. The network is almost untrained where the sampler starts.** Training
  draws `t = sigmoid(N(0,1))` (`flydream/generate/gen13b.py:215-217`), so
  t < 0.05 is seen in ~0.16 % of steps; the sampler integrates from t = 0
  (`prior17.py:211-216`). The measured velocity norm at t = 0 is 2.2 (fresh
  draw) and 7.5 (preimage) (`..._step20_sampler_and_cuts.md` 93-99), where
  the exact optimum `E[x1] - eps` has norm about sqrt(2048) = 45. This is a
  hypothesis for the "time-skew" 20 measured, not a proven cause.

## 3. Decisions

### D1. The round trip through the frozen brain as the gate (17.2-18.19)

- **What.** `state -> 13B -> video -> brain -> state'`, error normalised by
  the reference variance, was the only acceptance number for generated
  states until the owner struck it (`reports/2026-09-20_the_seed_problem.md`
  § 5); the contract made it "the unit of evidence on the generator track"
  (`docs/PROJECT_MAP.md:195-205`).
- **Fit to task.** For reconstruction (13A/13B) a target exists and the round
  trip is a fair cycle-consistency test. For unconditional generation it
  measures only whether 13B can find a video the brain maps back to the
  state, i.e. reachability. Low-information states are easy to reach, so it
  prefers bland: +18 % contrast gave a 34 % worse gate (18.8, lines 308-324);
  best arm below its own representation floor (18.13, 666-672); a near-empty
  grey field 0.0070 against a real clip's 0.0119 (18.17, 909-914); a smooth
  gradient 0.0119 (18.19, 1024-1026).
- **Circularity.** Not logical circularity - the brain is frozen and
  independent of the prior - but 13B was trained as this brain's inverse on
  the real-state manifold, so `Phi = brain o 13B` is a near-projection onto
  reachable states (`..._state_restoration.md` §§ 5-6: real states are fixed
  points at 0.0124). Anything in Phi's image passes, textures included. It is
  a filter (reject the unreachable), never a score.
- **Fit to goal.** Poor: the goal is a scene judged by eye; the gate cannot
  see "scene vs texture" (options_after_the_floor.md 26-30 says so itself).
- **Better alternative.** From 17.2 on, a distribution test against a
  covariance-matched Gaussian and against held-out states: MMD with
  permutation (Gretton et al. 2012, JMLR 13:723) or a classifier two-sample
  test (Lopez-Paz & Oquab, ICLR 2017, arXiv:1610.06545), in state space and
  on rendered pictures; the round trip kept as a pass/fail filter. $0, local,
  minutes (C3). The track's own research listed MMD as "$0, local" on
  2026-09-20 (`..._research_video_generation_and_training.md` §6, row in §7).
- **Verdict.** Defensible at 17.2; **mistake** to keep it as the selection
  metric after 18.8. Information was available: that low-order or
  consistency metrics disagree with sample quality is standard (Theis, van
  den Oord & Bethge, ICLR 2016, arXiv:1511.01844).

### D2. Full-state representation and model sizing (17.1-18.13)

- **What.** DCT-16 of 40 frames x 8 types x 721 columns = 92,288 numbers,
  each coefficient z-scored; SiT with one token per column, 1.4 M -> 8.8 M
  parameters.
- **Fit to task.** DCT was a good cheap choice (17.1b: 8x gate, 99.3 % of the
  energy). Two costs: equal weighting of coefficients (18.5b/18.6), and 128
  channels per site, an order above the 4-16 at which latent generative
  models are published (`..._research_path_to_a_video_generator.md` § 7.1,
  Yao et al. arXiv:2501.01423). 17.1's 1.4 M parameters for 230,720 numbers
  was sized to 13B's benchmark, not to the task; 18.4 found width the
  dominant lever (0.090 -> 0.034, `..._step18_4_...md` 117-148).
- **Fit to goal.** Neutral: the sweeps optimised the round trip (D1); 18.10
  shows gate and structure mostly agree (r -0.79, lines 427-431), so the
  capacity finding survives, the K = 32 line (~$1.3) does not matter.
- **Better alternative.** Fewer channels per site from the start (what 22
  arrived at), and scale width before sweeping K.
- **Verdict.** Defensible at the time; the sweeps were cheap ($1.93, 18.4)
  and each literature prediction was settled by measurement.

### D3. The "seed problem" line (18.17-18.24)

- **What.** Reading real preimages at 0.83 of the shell ("73 sigma inside",
  `the_seed_problem.md` § 3) as a seed defect, then fixed random noise per
  state (18.20), the model's own preimages as fixed pairs (18.23), and a VAE
  whose latent must be N(0, I) (18.24, three beta runs).
- **Fit to task.** The informative number was the per-axis sd 0.83: the
  model's pushforward is broader than the data - a model-fit property, as
  18.17 itself concluded (lines 916-921). "73 sigma" is correct arithmetic
  but in D = 92,288 any per-axis mismatch is dozens of sigma (the shell's
  width is 0.71). Independent coupling is the standard, correct coupling of
  flow matching (Lipman et al. arXiv:2210.02747); a fixed random pairing is an
  arbitrary regression target (0/8, as the agent's pre-run forecast warned,
  § 4); using the model's own ODE pairs is reflow (Liu et al.
  arXiv:2209.03003), which straightens paths and inherits the model's
  distribution - it learned 8/8 pairs and did not generalise (§ 9). Asking
  the VAE's KL to make the latent N(0, I) is the recipe the industry does not
  use (two-stage: Rombach et al. arXiv:2112.10752; Dai & Wipf
  arXiv:1903.05789), as the track's own later research found
  (`..._research_how_vaes_are_trained.md` §§ 1, 9).
- **Fit to goal.** None of it could move the density; ~$2.69 and the rest of
  2026-09-20 (`the_seed_problem.md` 359-362).
- **Better alternative.** Read the sd as "under-fit model", go straight to
  capacity / representation / inductive bias; a flow's temperature knob was
  already tested (18.17) and is the only sampler-level lever the literature
  offers (Kingma & Dhariwal, Glow, arXiv:1807.03039).
- **Verdict.** **Mistake**, knowable from literature at the time. The framing
  came from the owner's hypothesis ("проблема в сидах", § 4); the agent's
  forecast named the risks, but the line was run to completion.

### D4. PCA-2048 whitened latent and a flat flow over 16 x 128 tokens (19-22)

- **What.** A linear, whitened first stage; the flow sees 16 tokens that are
  arbitrary chunks of the PCA spectrum.
- **Fit to task.** It beat the small VAE (0.825 vs 0.687, step 19 § 5) for
  $0.02. But (a) whitening puts half the loss on components holding 7-9 % of
  the variance (C7, the 18.5b pathology); (b) in whitened coordinates a
  Gaussian model is the identity map, so the flow's job is only the
  non-Gaussian remainder, which a 2.5 M flat net barely touched (19 degrees,
  C5; loss 16 % under the Gaussian, C4); (c) attention over spectrum chunks
  has no locality or equivariance to exploit.
- **Fit to goal.** The lattice - where the literature locates the mechanism
  of novelty (Kamb & Ganguli arXiv:2412.20292; locality in DiT
  arXiv:2410.21273; permuted-pixel control in Kadkhodaie et al.
  arXiv:2310.02557) - was destroyed; the path report said so the same day
  after the step was built (§ 4 and table row E, lines 384-473, 1086).
- **Better alternative.** Keep the lattice from the first compression: SD-type
  latents are spatial grids, not vectors (the VAE report's own table, § 1);
  compress channels per column, not the layout (research § 7.2, 710-757).
- **Verdict.** **Detour, predictable** from the project's own 18.5b and from
  literature available on 2026-09-21; cheap in money ($0.39 for 19 + 22) and
  it yielded durable by-products: the restoration operator Phi (21c), the
  honest-mask finding (T4 alone 0.904), the T4a+T4b block (22).

### D5. Sampler fixes (20) and draw-time corrections (26)

- **What.** Step count, epsilon scaling, grid shift, and `sdproj` - a
  per-batch rescaling to the ideal sd curve (`fix19.py:85-102`); 26: seven
  noise radii on the hex flow.
- **Fit to task.** 20 closed the sampler hypothesis for free (gate 0.2164 ->
  0.0933, still 8x a real state). `sdproj` makes every sample depend on its
  batch and made both video judges worse (sharpness 11 -> 7,
  `the_draw_distribution.md` § 9); it is a diagnostic, not a generator part.
  26 repeated 18.17's result on a new model; its useful product is that
  kurtosis is buyable by shrinkage. Missed: C8 - the endpoint t ~ 0 is almost
  untrained under logit-normal sampling.
- **Better alternative.** Replace the network's velocity below a small t_min
  by the analytic Gaussian velocity (exact at t = 0: `E[x1] - eps`), or
  train with a uniform/heavy-tailed t mixture; free to test on saved
  checkpoints.
- **Verdict.** 20 defensible (free, decisive); 26 low-value repetition
  (free).

### D6. A smaller object: the T4a+T4b block (21c-22)

- **What.** Declare T5 and T4c/d absent through 13B's type mask; 23,072
  numbers at a 0.884 ceiling (`..._step22_a_smaller_object.md` § 2).
- **Fit.** Cuts channels per site 128 -> 32, the direction the research
  supported; cheap ($0.13) and measured. Caveats: the a+b choice reflects
  UCF101's horizontal motion (§ 9); the exact a+b-only mask appears in about
  0.04 % of 13B's training steps (1/256 of the 10 % "random" mode,
  `flydream/generate/gen13b.py:34, 57-61`). It also moves the
  branch onto the side ISS-0015 calls clean (`ISSUES.md:63`), though T4 has
  OFF-side inputs (Mi9, fed by L3 in the fly - Takemura et al. 2017) whose
  state in model zero was not checked.
- **Verdict.** **Right call.** Its negative result ("size did not help") was
  measured through the PCA flow, the actual bottleneck, so it says little
  about object size.

### D7. Hex-local flow on the lattice (23)

- **What.** `hexflow23.py`: 3x patchify by Kuhn matching (239 patches of 3,
  2 of 2), radius-2 local attention, shared relative-position bias, 4
  registers (`hexflow23.py:115-166, 194-225`); 3.68 M parameters, $0.51.
- **Fit.** Matches the measured local covariance (0.876 at one step vs 0.267
  shuffled, step 20 § 6) and the literature's locality mechanism. Like-for-
  like gains: kurtosis in one basis 4.11 -> 5.36 (partly clip energy, C6),
  sharpness 11 -> 22 of 100, the clip's own seed at the 0.884 ceiling. The
  72-degree transport is not evidence (C5). Train/val gap 13 % (ROADMAP:
  489-491) and 13,555 states sit at the published memorisation-generalisation
  edge for small U-Nets (K ~ 8,000-16,000, research § 1).
- **Verdict.** **Right call, late**: 17.1 already had column tokens and
  global attention; locality could have entered there.

### D8. The judges (18.9, 22.7-22.8, 23, 26, 29)

- **What.** Flat fraction + neighbour coherence (18.9, 18.12), a noise-to-raw
  sharpness scale (22.8), per-coordinate kurtosis + radius (22.7), nearest
  training video.
- **Fit.** Honest and increasingly well controlled (floors in 22.8 are good
  practice), but all low-order: flat fraction is gamed by salt-and-pepper
  (18.12, lines 572-590) and equals 19.7 % for any Gaussian field (C2);
  kurtosis is mostly clip energy (C6), buyable by shrinkage (26) and vacuous
  on the DC (29 § 5); nearest-neighbour r cannot tell a scene from a smooth
  field (22.8 § 8) and cannot detect memorisation (van den Burg & Williams
  arXiv:2106.03216, cited in 18.3).
- **Timing.** A scene metric was proposed on 2026-09-20 as option D / option
  2 "first, $0, unblocks everything" (`..._options_after_the_floor.md`
  129-139; `..._options_after_the_classes.md` 63-73, 127-130); it became
  queue item 29.4 only on 2026-09-23 and does not exist.
- **Verdict.** **Mistake of ordering**: the judge should have preceded the
  18.x sweeps. A distribution test (C3) needs no renderer and no pretrained
  network and would have separated "Gaussian texture" from data from 17.1b
  onwards.

### D9. The static branch (29)

- **What.** Object = DCT coefficient 0 of T4a+T4b of moving-clip states
  (1,442 numbers, no re-simulation, owner's constraints ROADMAP:53-57);
  judge = LS map DC -> mean frame; flow = the hex flow at k = 1, $0.42.
- **Fit to task.** Excellent for speed: smallest object, near-Gaussian
  target, cheap flow, and at the state level it works (C3: passes MMD/C2ST
  where a Gaussian fails, no copies). Three limits the reports did not
  separate:
  1. the target is a 40-frame mean, blurred by construction (target flat
     39.2 % vs one frame 45.1 % vs raw video 49.8 %, step 29 § 4) - no
     renderer can return sharpness the DC does not carry;
  2. the DC is not a state any stimulus causes, which is exactly why 13B
     goes blank on it (the 21c logic: an unreachable order); the true
     still-frame state renders at 0.953 and 13B reads the picture from its
     temporal coefficients (the donor arm, § 3) - so the chosen object threw
     away the part the existing renderer uses;
  3. the renderer-level parity depends on an unrecorded ridge (C2), and
     coherence parity is second-order only.
- **Fit to goal.** By C1 the static state is ~a linear image of the mean
  frame, so a static-state prior is an image prior in a linear basis: the
  fly brain contributes a change of coordinates at this level. That is fine
  for the owner's first target, but it means the image-space generator is
  the ceiling control, and "scene appears" must be shown against it.
- **Better alternative.** True still-frame states: the held frame is already
  rendered (`videos.npz`), a corpus brain pass is 605 s / $0.15 on a T4
  (`runs.jsonl` `2026-09-20_pairs18m_maps18`), so eight frames per clip is
  about $1.2; 13B is then a renderer with a measured 0.953 ceiling, either
  on generated full still states or on the DC plus a residue predicted from
  it (the docstring at `deploy/modal/generate_app.py:1066-1069` claims 76 %
  predictable - no run records it).
- **Verdict.** **Right call** for the owner's narrowed target, with the object
  choice **still open**: "no re-simulation" saved $0.15 and cost a renderer.

### D10. The framing: stimulus-driven states of a model with a broken OFF path

The priors learn `p(state)` = the corpus pushed through model zero. No
internal dynamics enter (item 16 parked), so the object is a video/image
prior in brain coordinates; ISS-0015 puts a periodic artefact into T5 (used
in 17-21) and possibly, via OFF inputs, into T4. Alternatives, one line
each; costs from measured project rates unless marked:

| alternative | expected effect | rough cost | gives up against the idea |
|---|---|---|---|
| A. Off-the-shelf image/video generator -> eye -> brain | valid novel states by construction; clean pairs for any renderer; immediate demo | stills: cents per 1,000 (research § 6, A100-derived, not T4-measured; SD-Turbo is non-commercial, use an Apache model); brain pass $0.15 per 15.5k frames | the brain determines nothing in the picture (research § 6, 663-669); no evidence the state space is a generative latent |
| B. Image-space flow on the same mean frames (721 grey hexals), mapped to states by C1's linear map or by simulation | the ceiling control: tells whether 13.5k 40-frame means at 721 hexals can look like scenes at all | ~$0.4 (29's run shape) | as a product, the same as A; as a control, nothing |
| C. Joint state+picture model (one flow over [picture, state] or a shared latent) | renderer and prior in one; sharpness from the picture half; state consistency checkable by simulation | ~$0.4-0.5, one T4 run | the scene is invented jointly, not in state space |
| D. Lattice flow with the renderer in the loop | optimises the owner's axis; with the linear LS map a picture-space loss is just a Mahalanobis weight `W W^T` on the state loss - a free code change like `loss_weight_p`; with 30, guidance at draw time | $0.4 per retrain; guidance free | nothing; Goodhart risk - keep a held-out judge |
| E. Conditioning on scene content | the literature's largest measured lever: per-sample representation conditioning, RCG -51 to -82 % FID (arXiv:2312.03701); clusters ~10x classes (arXiv:2403.00570). Class labels measured null here (18.16) | embeddings ~$0.25-0.5 (unmeasured, options_after_the_classes 84-85) + ~$0.5 training | none if the embedding is itself drawn (two-stage); memorisation must be shown beside every sample (arXiv:2310.02664) |
| F. More data earlier (27; rotations) | moves N past the small-model transition; 29.2 gives 8x free. In state space only mirror + a<->b swap is plausibly exact for the a+b pair (60-degree rotations map a/b onto c/d, which were dropped), and only approximately on MaleCNS | re-simulation $0.15 per corpus pass, x4-6 | nothing |
| G. True still-frame states + 13B as renderer (D9) | removes 29.3's uncertainty and most of 30 | ~$0.15-1.2 | the owner's "1,442 numbers only" unless the residue is predicted |

## 4. Top lessons (ranked)

1. **Build the judge before the model, and make it distributional.** A
   two-sample test (MMD/C2ST now; KID on a corpus-trained feature net for
   pictures, Binkowski et al. arXiv:1801.01401) against held-out states and a
   Gaussian baseline costs $0 and separates what the round trip could not
   (C3). The track optimised a reachability filter through most of 18.x
   (D1, D8).
2. **Carry the Gaussian baseline everywhere.** Covariance-matched Gaussian
   draws, the Gaussian-optimal loss (C4) and the Gaussian transport angle
   (C5) are the "no learning" floor; several headline numbers (80 %
   explained, 72 degrees, coherence parity) are at or near it.
3. **Keep the inductive bias that matches the data.** Lattice locality was
   the one lever that moved everything (23); flattening it (19-22) was
   predicted by literature and by the project's own 18.5b (C7).
4. **Diagnose the model, not the seed.** An over-broad pushforward is a fit
   problem; coupling, radius, sampler and KL fixes cannot move a learned
   density (18.17-18.24, 20, 26; ~$2.7 and most of a working day).
5. **Pick the target so the renderer and the ceiling exist.** DC of moving
   clips is blurred and unreachable; true still states cost $0.15 per pass
   and keep a 0.953 renderer (D9).
6. **Measure how linear the representation is before modelling it.** R^2
   0.974 picture -> static state (C1) says the static problem is image
   generation in disguise; run the image-space control (alternative B).
7. **Record every free parameter of a judge and keep the script.** The
   renderer ridge moves draws' flat fraction 21 -> 27 % (C2); r 0.922, the
   slice numbers and the 76 % residue exist in no artefact.

## 5. Options for the resumed order (29.3 -> 30 -> 29.4 -> 29.2 -> 31)

The order's logic holds: 30 trained on converted states inherits any
conversion error. As options, not decisions:

- **Option 1 - keep the order, sharpen each item (smallest change).**
  29.3 as planned but also record (a) LS reconstruction sharpness from the
  still state's DC vs its full 16 coefficients, (b) 13B on DC + residue
  predicted from the DC by ridge (tests the 76 % docstring), both local on
  the same held-out frames; add the free state-level MMD/C2ST now (C3) with
  the covariance-matched Gaussian as the one control; give 30 two controls
  in its first check (Gaussian states and column-shuffled states through the
  renderer - if they also look like scenes, the scene is the renderer's);
  record the renderer ridge.
- **Option 2 - replace the conversion by simulation.** Simulate true still
  states for the held-out set locally (29.3 becomes a direct comparison),
  then for the corpus at ~$0.15 per frame per clip (8 frames ~$1.2) - this
  is 29.2's eightfold data with sharper, reachable targets. Render with 13B
  (0.953 ceiling); 30 becomes needed only if the owner keeps the 1,442-number
  object and the residue prediction fails.
- **Option 3 - add the image-space ceiling control before 30.** Train the
  same hex flow on the mean frames (or on single frames) at ~$0.4. If its
  draws do not look like scenes to the owner, the ceiling is the data
  (13.5k blurred 721-hexal pictures), not the state representation, and
  the next lever is data/target (29.2, 27), not a renderer.
- **On 29.4.** Split it: the state-level part is free and usable now; the
  picture-level part (KID + precision/recall, Kynkaenniemi et al.
  arXiv:1904.06991) needs 30 or Option 2's renders; a feature net trained on
  the corpus pictures avoids adapting an ImageNet network to 721 grey hexals.
- **On 31.** For the T4a+T4b branch the exposure is indirect; money is not
  the issue (re-simulation $0.15, priors $0.4-0.5 each), human work is. A
  free first check - does repairing L3 move T4a/T4b DC maps at all - would
  tell whether 31 must precede the static branch's retraining.

## 6. Gaps

- All my checks use the local PCA-2048 reconstruction of real states (89.5 %
  of held-out block variance); the true states on the volume were not read.
  C1's R^2, C2's ridge dependence and C4's static floor could shift on true
  states; C4's static value is a lower bound only.
- C3 is at n = 256 in the top 64 directions; a larger n or all 1,442
  directions may separate draws from real (state-space neighbour r: draws
  0.809, real 0.869, Gaussian 0.861 - unresolved because the reconstruction
  is smoother than true states).
- C5 uses the training spectrum with null directions at zero and the same
  law-of-cosines approximation as `accept23.py`; the flow's output has energy
  in those directions (16.3 % outside PCA-1536).
- C8 is a hypothesis: I did not evaluate the saved checkpoints' velocity
  against `E[x1] - eps` or retrain.
- Not verified: the 76 % residue claim, r 0.922 DC-vs-still, the slice
  numbers (all without artefacts per the reports); whether model zero's L3
  reaches T4a/b through its OFF-side inputs.
- Costs for alternatives A and E are unmeasured estimates from the reports
  cited; none of the literature numbers were re-checked online in this
  review (all are cited from the track's own research reports or are
  standard references named by arXiv id).
- Recorded track cost: $8.87 in `runs.jsonl` for runs keyed prior18/19/22/
  23/29/corpus18/pairs18 dated >= 2026-09-20 (my sum; prior18 alone $7.19),
  plus ~$0.37 for 17.1/17.1b from their reports.

## Appendix: the reviewer's checks (reproducible from the repo root)

```python
import numpy as np, sys; sys.path.insert(0, ".")
from flydream.generate.edges18 import edges
from flydream.decode import hexraster as L
p = np.load("data/prior19/pca_ab2048.npz"); z = np.load("data/prior19/pca_ab2048_latent.npz")
B, mu, lam = p["basis"][:1442].astype(float), p["mean"][:1442].astype(float), p["lam"].astype(float)
dc = lambda s: (z[f"z_{s}"] * np.sqrt(lam)) @ B.T + mu            # coef 0 = first 1,442 entries
Xtr, Xte = dc("train"), dc("test")
V = np.load("data/corpus18/videos.npz", mmap_mode="r")["videos"]
Ytr = np.stack([V[i, :40].astype(np.float32).mean(0) for i in z["index_train"]])
Yte = np.stack([V[i, :40].astype(np.float32).mean(0) for i in z["index_test"]])
# C1: picture -> DC, held-out R^2
Wi = np.linalg.lstsq(np.c_[Ytr, np.ones(len(Ytr))], Xtr, rcond=None)[0]
R2 = 1 - ((np.c_[Yte, np.ones(len(Yte))] @ Wi - Xte) ** 2).sum() / ((Xte - Xtr.mean(0)) ** 2).sum()
# C2: ridge LS renderer, judged with edges() on draws / real / Gaussian(train mean, cov)
A = np.c_[Xtr, np.ones(len(Xtr))]; W = np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ Ytr)
ev, U = np.linalg.eigh(np.cov(Xtr, rowvar=False)); G = Xtr.mean(0) + (rng.standard_normal((256, 1442)) * np.sqrt(ev.clip(0))) @ U.T
# C3: project on top-64 eigen-directions, whiten; MMD^2 (RBF, median width, 300 permutations);
#     sklearn MLPClassifier((128,)) / HistGradientBoostingClassifier, 5-fold CV accuracy
# C4: Gaussian-optimal FM loss = mean_t sum_i lam_i/((1-t)^2+t^2 lam_i) / D, t = linspace(.05,.95)
#     static: eigvals of cov(Xtr); block: p["eigvals_all"]/13554 with D = 23,072; whitened: lam_i = 1
# C5: cos = sum(sqrt(lam_block)) / (sqrt(D) * sqrt(|mean|^2 + sum(lam_block)))  -> 71.6 degrees
# C6: kurtosis of z_train[:, :1536] subsamples (n=256 x 40) raw vs each row scaled to radius sqrt(1536)
```
