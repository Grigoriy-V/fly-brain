# What Does a Fly Dream Of? Part 2. From rendering a state to generation without a source clip

*Grigoriy Voyakin, September 2026.* [Русская версия](part2_ru.md).
Code, run records and every figure are in the
[fly-brain](https://github.com/Grigoriy-V/fly-brain) repository.
[Part 1](part1_en.md) ends at step 14: the chain
`video → frozen brain → T4/T5 state → generator → video` works and is
measured. This part is about the next task: **getting a new video that has no
source clip**, through a drawn brain state. It is an account of what was
built, what was measured, what turned out to be false, and where we stand:
structure has appeared in the generated pictures, and a scene is not proven
yet.

Every number comes from `reports/runs.jsonl` and the step reports; where a
number has no artefact behind it, the text says so.

## In brief

By the end of part 1 the project could **render any reachable brain state**
and could not **invent a new one**. Generation without a clip needs a second
generator, further up the chain:

`ε ~ N(0, I) → noise2state → state → 13B → video`

Over four days and about **$9.3** on Modal we went from a flow over 230,720
numbers to a flow over 1,442 numbers on the hexagonal lattice. Along the way
we:

- learned the structure of the states almost to the level of real ones;
- found that the project's main judge **was lying**, and replaced it;
- caught our own metrics in an error twice;
- narrowed the task from a video to **one still picture**.

For the first time in this line, a draw cannot be told from a real state by
two structural judges - the flat-field fraction and neighbour coherence:
structure has appeared in the generated pictures. The caveats are those of
section 11: the judges look through a blurring renderer, and the target is
nearly Gaussian. Whether this structure is a scene is not proven yet - that
needs a proper renderer and a "this is a scene" metric.

**The scope of the claims is unchanged.** A drawn state is not the fly
brain's own activity, and a video made from it is not a dream. It is the
video most compatible with the state under this encoder. "New" is a
measurement: the distance to the nearest training clip, beside the same
number for a real held-out clip.

## 1. The task, and the criterion that overruled every metric

13B (part 1, section 8) is a conditional flow "state → video". It has no
model of the state distribution `p(state)`, so a state could only be taken
from a video. The task of part 2 is to learn that distribution and draw from
it.

The acceptance criterion outranks every metric: **a result counts only as a
clip against the raw corpus video.** Blur is acceptable in a first version if
scenes appear and a fresh draw gives a new, meaningful video. Section 5
explains why: before this criterion we trusted a wrong judge four times in a
row.

## 2. A free baseline: what 13B does with no condition at all

Before training a prior we asked whether the task was already solved. 13B saw
an "empty" condition mask on 5 % of its training steps, so it can draw from
noise alone. Sixteen such videos, locally, 35 seconds, $0.

The novelty metric was first tested on itself: 13B conditioned on the states
of four training clips finds **its own** clip as the nearest in 4 cases out of
4 (r 0.94-0.99). So it does catch copying.

| | from noise (16) | real held-out (16) |
|---|---|---|
| nearest training video, r (median) | +0.53 | +0.52 |
| frame-to-frame change | 0.038 | 0.017 |

The videos are new in the same sense as a real held-out clip is new - and
they are not scenes. Hence the need for a prior.
[Report 17.0](../../reports/2026-09-20_step17_0_unconditional_baseline.md) (step reports are in Russian).

## 3. The first prior over states: a flow, but no manifold

**17.1.** Flow matching directly in state space, on 13B's recipe: a token is
a column (721), features are 40 frames × 8 types, 1.43 M parameters. One T4,
1,170 s, ≈ $0.23, 99.2 % card utilisation.

The judge at the time is the round trip: state → 13B → video → brain →
state′.

| state fed to 13B | round trip |
|---|---|
| drawn from the prior | **1.19** |
| a real held-out clip | 0.004-0.061 |
| control: the clip's columns shuffled | 1.77 |
| control: white noise in the types | 2.19 |

A drawn state sits next to the shuffled one, not the real one. The video is
"salt and pepper". [Report 17.1](../../reports/2026-09-20_step17_state_prior.md).

**17.1b.** Three quarters of 17.1's training states came from procedural
stimuli, while the goal is a video generator. The prior was retrained on
scenes only, in a compact temporal basis: DCT, 16 of 40 coefficients, each
z-scored (92,288 numbers). K = 16 was chosen by measurement, not by eye: on a
real state K = 8 gives a round trip of 0.58, K = 16 gives 0.045, the full
state 0.019.

| | lag-1 in time | ring neighbours | type coupling | video frame change |
|---|---|---|---|---|
| 17.1 | 0.57 | 0.34 | 0.16 | 0.280 |
| **17.1b** | **0.98** | **0.77** | **0.31** | **0.026** |
| real | 0.99 | 0.80 | 0.35 | 0.057 |
| what the DCT-16 basis itself gives on white noise | 0.77 | −0.00 | 0.01 | - |

The last row is the control: the basis alone creates neither spatial
structure nor type coupling. The model learned them. The round trip fell
eightfold, to 0.142 - but there are no scenes, only oriented texture and
large moving blobs. [Report 17.1b](../../reports/2026-09-20_step17_1b_scene_dct_prior.md).

![Fig. 1](../figures/first_priors.gif)

*Fig. 1. Every cell is a 13B video. Top: from a held-out clip's state; below:
from 13B's empty condition, from a draw of prior 17.1 and from a draw of
17.1b. Script `tools/fig_gh_part2.py`, runs `2026-09-20_baseline17`,
`2026-09-20_prior17_*`.*

## 4. The flow run backwards: every clip has its own noise

The prior is a deterministic map from noise to state. Running the Euler steps
backwards finds, for any state, the noise it would have been drawn from.

A plain backward step loses a quarter of the vector (error 0.243). **Four
fixed-point iterations** per step give an error of 0.0058 and r 0.99998 - the
inversion is exact. Everything that follows rests on this.

- A real clip → its noise → back: the state returns at r 0.964; a face in
  the clip stays a face.
- The noises of two clips are uncorrelated (−0.013), although their states
  correlate at +0.48: the prior decorrelates what it models.
- The preimages of real states lie **just outside** the typical set of
  Gaussian noise; a shuffled state lies four times further out.

![Fig. 2](../figures/flow_inversion.gif)

*Fig. 2. Top: clip A, 13B from its state, the same state through its own noise
and back, and a control with a shuffled state. Bottom: a straight line between
the noises of clips A and B; the midpoint matches the average of the two
videos at r 0.85 - a double exposure. Script `tools/fig_gh_part2.py`, data
`data/prior17/noise17_local`.*

[Report 17.3b](../../reports/2026-09-20_step17_3b_noise_inversion.md).

## 5. A corpus of ordinary video, training levers, and the judge that lied

**The corpus (18).** Nineteen Sintel scenes are too few. We built a corpus
from UCF101: 12,411 clips of ordinary video passed the filter out of 13,320
files, plus 3,103 procedural clips - 15,514 in all. The split is **by class**,
so testing is on ten classes training never saw. The fly's eye for it is our
own implementation of the FlyVis chain:

- checked on Sintel frames against FlyVis's own dataset: **r = 1.000000**,
  maximum difference 2.6e-4 (the precision of the float16 cache);
- linear interpolation in time instead of holding frames would have given
  0.998 - not rounding but the wrong temporal law, and we caught it;
- the eye renderer runs **60 times** faster (5.46 → 0.09 s per file): instead
  of convolving the whole 417×417 frame, running sums read only at the ~31
  columns and ~61 rows that hold a receptor. The whole corpus in 9 minutes
  instead of 57.

[Report 18](../../reports/2026-09-20_step18_corpus_of_ordinary_video.md).

**The levers (18.3-18.5).** Step 18.4 ran seven arms against one control -
learning rate, length, width, number of coefficients, class conditioning
(Fig. 3). The table shows only the winning chain:

| step | round trip |
|---|---|
| 17.1b, Sintel only | 0.142 |
| 18.3, UCF101 corpus | 0.095 |
| 18.4, width 192 | 0.034 |
| 18.5, width 192 + lr 1e-3 | **0.0224** |
| representation floor (a real state in DCT-16) | 0.0209 |

All four predictions taken from the literature pointed the wrong way; width
turned out to be the lever. Reports
[18.3](../../reports/2026-09-20_step18_3_prior_on_the_corpus.md),
[18.4](../../reports/2026-09-20_step18_4_throughput_and_learning_rate.md),
[18.5](../../reports/2026-09-20_step18_5_width_and_the_representation_floor.md).

**And then the judge failed.** The round trip reached the representation
floor, while the videos stayed clouds. On inspection the gate was giving good
marks to bad output:

- it penalised contrast;
- it went below the representation floor;
- it gave 0.0070 - better than a real clip - to **an almost blank grey
  field**;
- it gave 0.0119, level with a real clip, to an almost uniform gradient.

The round trip measures a video's compatibility with the brain, and a grey
field is compatible with anything. From that day the gate was removed from
its post as judge, and the criterion of section 1 applies: a clip against the
raw video.

![Fig. 3](../figures/levers_and_judge.png)

*Fig. 3. Left: the round trip of every arm of 18.3-18.5 beside the
representation floor. Right: an almost blank grey field scores better than a
real clip. Script `tools/fig_gh_part2.py`, runs `2026-09-20_prior18_*`.*

## 6. The seed problem

The statement is simple, and we kept it fixed: as long as one cannot feed a
seed into noise2state and get a video like a real clip, there is no result.
Everything ruled out along the way is recorded
[separately](../../reports/2026-09-20_the_seed_problem.md), so as not to return to it.

| where the noise comes from | r to the real clip |
|---|---|
| 13B straight from the real state (ceiling) | +0.969 |
| **the clip's preimage, set by hand** | **+0.957** |
| an ordinary draw N(0, I) | −0.060 |

**By hand it works; by drawing it does not.** The cause is geometry (Fig. 4).
Gaussian noise in 92,288 dimensions lies on a sphere of radius 303.8 and
width 0.707, while the preimages of real states lie at 252.4, **73 standard
deviations inside**. A seed that gives a scene does not exist: a seed indexes
a draw on the shell, and scenes are not on the shell. The instrument is
checked: a state drawn by the prior itself inverts back exactly onto the
shell.

What we tried and ruled out:

| what | result |
|---|---|
| a shorter draw | worse |
| an arbitrary noise pinned to each clip | its own clip 0 times out of 8 |
| the clip's **own preimage** pinned to it | 8 out of 8 - learned, but does not generalise |
| minibatch-OT | empty analytically: the choice of pairing moves 0.33 % of the cost |
| class conditioning + guidance | worse structure |
| more steps | fixes the geometry, does not touch the picture |
| more width | moves the picture, does not touch the geometry |

The last two rows together mean: **the geometry of the preimage and the
quality of a sample are two different diseases.**

![Fig. 4](../figures/seed_shell.gif)

*Fig. 4. Six held-out clips from unseen classes. Top to bottom: 13B from the
real state; the same state from its own noise (radius under each cell: for
this prior 232-283 against a shell of 303.8); the same noise pushed out onto
the shell √D; an ordinary draw. Prior 18.5, script `tools/fig_gh_part2.py`,
run `2026-09-20_prior18_reach_noise_geometry`.*

**VAE.** The KL term is exactly what flow matching lacks: it looks at where
real data are encoded. A column-wise VAE, latent 721 × 3.
- The "a seed can be drawn" half is **met for the first time**: the latents
  of held-out clips sit at radius 46.1 ± 1.99 against √D = 46.5.
- The "a draw gives a scene" half is not. The decisive test with KL switched
  off showed it is not a matter of balance: from a clean code the VAE gives
  0.687, while **plain PCA of the same dimension gives 0.890**. The loss is
  not to a transformer but to linear algebra.

## 7. A linear first stage, and what it showed

**19.** PCA to 2,048 components over all 13,555 training states, whitened,
with the flow over its latent. The decomposition through the 13,555 × 13,555
Gram matrix, right on the card: 43 seconds on a T4, **$0.02**; `eigh` took
21 s instead of the expected minutes.
[Report 19](../../reports/2026-09-21_step19_linear_first_stage.md).

**20.** The sampler's overshoot is not a constant factor but a **skew in
time**: the velocity field is 0.07 of the true one at t = 0.4 and about 2.5 at
t = 0.6-0.8. That is why the literature's fixes with a constant divisor do not
catch it. Projecting onto the measured trajectory fixes the geometry and
improves the gate 2.2 times - **and gives no scene**. For the second time a
win in geometry was not a win for the generator.
[Report 20](../../reports/2026-09-21_step20_sampler_and_cuts.md).

**21c-21d.** Two useful properties of the chain:
- **The chain restores what was thrown away.** Half a state, zeroed and run
  through `state → 13B → video → brain` with an honest type mask (next
  point), returns with an error of 0.0433 against 0.5581 before the chain -
  12.9 times smaller. The first measurement, through the full mask, gave
  0.0846 and understated the effect.
  [Report](../../reports/2026-09-21_state_restoration.md).
- **13B has a type mask, and it must be used honestly.** Every run before
  this fed the full mask over a zeroed half, i.e. told the model "the type is
  there and it is flat". With an honest mask T4 alone gives r 0.904 against
  0.837 for completing the state by PCA.

## 8. A smaller object - and nothing moved

**22.** If the task is hard, the object can be made smaller. Inside T4:
- space cannot be cut: narrowing the field of view from 721 to 127 columns
  drops the correlation to the raw video over the whole frame from 0.904 to
  0.238 (a first reading of "0.011 lost" was computed on the remaining zone
  only and was a mistake);
- time is expensive to cut: half the temporal coefficients at the same number
  of coordinates give 0.444 against 0.721;
- T4's four directions are not equal, and the horizontal pair **T4a+T4b**
  carries almost everything: 0.884 against 0.904 for all four, at half the
  numbers.

From this step the object is the T4a+T4b block, 721 columns × 32 channels =
23,072 numbers. The PCA ladder over it is flat (0.806 at 1,536 components
against 0.824 at 2,048). Three ways around PCA - compressing channels, a
learned local residual, an autoencoder of the block - lost to it at a
comparable budget. A flow over 1,536 coordinates was trained for $0.05; a
fresh draw is still not a scene.
[Report 22](../../reports/2026-09-21_step22_a_smaller_object.md).

## 9. An audit of the metrics before the next training run

After one judge had already turned out to be false, we stopped training and
rechecked every metric: what it measures, what its floor is, and whether it
is computed on the wrong distribution.
[Report](../../reports/2026-09-21_the_draw_distribution.md).

**The flow learned the second moment of the distribution, not the fourth.**
The spread of the draws' radius is almost right, while the per-coordinate
kurtosis is 4.11 against 8.31 ± 1.17 on training subsamples of the same size
(a Gaussian gives 2.97). The data are heavy-tailed and sparse; the draws are
almost Gaussian.

**The flow barely rotates a point**: 19° where a random rotation gives 90°.
The map is close to the identity. Hence the observation that interpolating
between two clips looks like a double exposure: in an almost linear map,
"between" means "the average".

**Every judge gets its floor:**

| judge | floor | what it means |
|---|---|---|
| r of a draw to the raw video | 0.002 ± 0.100 between two different real clips | the earlier 0.017 was never evidence |
| nearest training clip | 0.521 for a real held-out clip | "not a copy" - true, and weak |
| sharpness, scale "noise = 0, raw video = 100" | renderer ceiling about 66 for the T4a+T4b block | a fresh draw - 11 |

Found along the way: one script compared the draws not with the training
latent but with the test one (ISS-0010); because of it the radius overshoot
read as 44 % where it is 20 %. Three functions with the same name, `nearest`,
computed different things (ISS-0011).

## 10. A flow on the lattice: everything measured moved

The conclusion of the audit and of step 22 is one: **a linear first stage
destroys the geometry.** A PCA coordinate is a global mode over all columns;
neighbourhood cannot be expressed in it, and no local architecture can be
built on top. Yet the state's covariance is local: 0.876 at one lattice step
against 0.267 for a shuffled control.

**23.** The flow runs directly on the 721 × 32 block, the lattice intact, no
PCA ([report](../../reports/2026-09-21_step23_hex_local_prior.md), code
`flydream/generate/hexflow23.py`):

| element | choice |
|---|---|
| token | a patch of three neighbouring columns on the √3×√3 sublattice |
| partition | Kuhn matching: **239 patches of exactly 3 columns and 2 of 2** |
| attention | a local mask, 7 neighbours |
| position | a shared table of relative offsets, no absolute position |
| global link | 4 register tokens |

Why a matching rather than the nearest centre: naive assignment gives patches
of 1 to 5 columns, tokens get different weight, and the equivariance the
whole design was built for breaks.

Training: 12,000 steps, batch 256, the learning rate raised tenfold
(2e-4 → 2e-3) after a free local probe, **99.9 % T4 utilisation** against
37.4 % for the previous run, 3,047 s, **$0.51**.

| | flow over PCA (22) | **flow on the lattice (23)** | data |
|---|---|---|---|
| kurtosis | 4.11 | **5.36** | 8.31 ± 1.17 |
| path from noise to data | 21 % | **45 %** | - |
| radius overshoot | 20 % | **6.4 %** | - |
| rotation | 19° | **72°** | 90° for a random one |
| a clip's own seed returns the clip | 0.809 | **0.884 = ceiling** | - |
| sharpness of a draw (noise 0 / video 100) | 11 | **22** | ceiling 66 |

The double exposure is gone. A clip's own seed returns its clip exactly at
the ceiling, because no lossy stage is left between the request and the
answer. There is still no scene.

![Fig. 5](../figures/hex_flow.png)

*Fig. 5. The same T4a+T4b block and the same data: a flow over PCA-1536 (grey)
and a flow on the hex lattice (green), each beside the data or the ceiling.
Script `tools/fig_gh_part2.py`, runs `2026-09-21_prior22_*`,
`2026-09-21_prior23_accept`, `2026-09-21_prior23_seed`.*

![Fig. 6](../figures/seed_known_and_random.gif)

*Fig. 6. The flow on the lattice. Top: held-out clips from unseen classes;
middle: their own seeds, run forwards through the flow and 13B; bottom: random
seeds. Script `tools/fig_gh_seed.py`, run `2026-09-21_prior23_seed`.*

**26 - corrections at draw time, and a metric that can be bought.** Seven
arms over the radius of the drawn noise, locally, $0. Two arms lifted
kurtosis to 7.11 and 7.17 - the best result of the line, read on its own. But
their output radius fell three times **below** the data (12.4 and 11.6
against 35.5), and the energy outside PCA from 16.3 to 6.5 and 6.1 %. The
output collapsed toward the corpus mean, and kurtosis rises mechanically when
most coordinates sit at zero. The rule that survived the step: **kurtosis is
always read beside the radius.**
[Report 26](../../reports/2026-09-21_step26_corrections_at_draw_time.md).

![Fig. 7](../figures/bought_kurtosis.png)

*Fig. 7. The seven arms of step 26 on radius-kurtosis axes; the bands are the
data's range. The red arms raised kurtosis by dropping the radius. Script
`tools/fig_gh_part2.py`, data `data/prior23/fixdraw23.json`.*

## 11. The turn to statics: a scene as one picture

The task is narrowed: the scene is not a video but one still picture; motion
does not matter for now. The object is **721 × 2** (the block's zeroth
temporal coefficient), not 721 × 32: the point of narrowing is to remove time
from the task entirely, not to shrink it. The bet is not on the size of the
object - step 22 showed size buys nothing - but on a whole factor of
variation leaving the task.
[Report 29](../../reports/2026-09-21_step29_a_scene_as_one_picture.md).

**Is there a still picture in this space at all?** Tested by changing the
stimulus: a single frame held for the whole window goes through
`stimulus → brain → state → 13B` at **r 0.953**, better than a moving clip
(0.916), and the rendering is genuinely still. The argument "T4/T5 are
motion detectors, so there is no static picture" is false.

**13B cannot render a state that is constant in time**, and no crutch saves
it:

| what is fed to 13B beside the 721 × 2 | r to the picture |
|---|---|
| zeros in the other coefficients | 0.257 |
| the mean residue of still states | 0.351 |
| the residue of **another** still picture | 0.137 |
| unit noise | 0.045 |
| ceiling: the real still state | 0.941 |

The most informative arm is the third: 13B draws a sharp picture, but the
**donor's** picture. The content 13B reads lives in the temporal
coefficients, although they hold only 4 % of the energy.

![Fig. 8](../figures/static_chain.png)

*Fig. 8. Left: what 13B makes of a 721 × 2 state with each crutch, and what a
least-squares map gives. Right: draws of the static flow through that map,
beside a real state and a control with no flow. Script
`tools/fig_gh_part2.py`, runs `2026-09-21_prior23_resfix`,
`2026-09-21_prior23_pic`, `2026-09-21_prior29_static_ab`.*

**But the picture is in the static state, and it is linear.** A least-squares
matrix from the 1,442 numbers to the window's mean frame gives **r 0.941** on
held-out clips of unseen classes; the controls give 0.021 (the corpus mean)
and 0.002 (shuffled pairs). Least squares cannot give sharpness, and does
not: it draws the conditional mean over all pictures compatible with the
state - 27.1 % flat field against the target's 39.2 %.

**29 - a flow over the static state.** Trained on the DCT coefficient 0 of
states already recorded, without a single new brain simulation. 10,000
steps, batch 256, 2,512 s, 99.0 % T4 utilisation, **$0.42**. The judges go
through the same least-squares map, 64 draws:

| | draw | real state | control: N(0, I), no flow |
|---|---|---|---|
| flat-field fraction | **27.1 %** | 27.1 % | 19.5 % |
| neighbour coherence | **0.928** | 0.949 | 0.492 |
| contrast | 0.128 | 0.181 | - |
| nearest training picture | 0.753 | 0.706 | - |

**For the first time in this line a draw does not differ from a real state
on two structural judges**, while the control differs grossly. That far and
no further: contrast falls 29 % short, and a draw lies closer to the training
corpus than a real held-out state does.

Two limits without which this cannot be read:
- the renderer blurs, and matching a real state **through it** is a much
  lower bar than being a scene;
- the target is almost Gaussian: the mean over 40 frames pulls toward
  normality (kurtosis 3.17 against 2.98 for a Gaussian), so the kurtosis
  judge is empty here.

The second limit suggests the next step: **a single temporal slice** of the
state is sparser than the mean (kurtosis 5.26 against 3.17) and renders into
its own frame at r 0.88-0.96 for frames 5 to 35; frame 0 is broken - an
artefact of the window's edge. A caveat: these numbers come from a one-off
script that was not kept and have not yet been rechecked with code in the
repository; that is the first item of the next step.

**The visual judgement** says the same as the table and no more: the shapes
are closer to a scene than anything before, but from pictures that went
through a blurring renderer one cannot tell whether the flow is good or bad.
Even a real state is hard to recognise through it.

![Fig. 9](../figures/static_draws.png)

*Fig. 9. Top: the pictures of six held-out clips (the window's mean frame),
chosen by eye as recognisable; middle: their real static states through the
least-squares map; bottom: the first six draws of the static flow, not
picked, through the same map. The numbers on the figure are computed on
these six clips, with a map refitted by the script (r 0.939 instead of 0.941)
and on a local reconstruction of the states, so they differ slightly from the
table above (flat field 25.3 % instead of 27.1 %); the table is the run
record over 64 draws. Script `tools/fig_gh_static.py`.*

## 12. The engineering of part 2

The thread running through this part is how to run many cheap experiments
on one T4 without paying for idle time.

| choice | what it gave |
|---|---|
| **the "GPU at 100 %" rule**: independent tasks in one batch axis | 98-99.9 % T4 utilisation in every training run; 37.4 → 99.9 % at step 23 |
| **the minimum container request**: 1 CPU / 12 GB beside the T4 | a core ≈ $0.19/h and a GB ≈ $0.024/h against $0.59/h for the card itself; $0.05-0.51 per run |
| **a GPU function does GPU work only** | rendering and data assembly on a CPU or locally; the card starts from a finished file on the volume |
| **profiling instead of guessing** (18.4a) | the "kernel-launch bound" hypothesis refuted: 2.44 ms of fixed cost per step out of 54.7; `torch.compile` gave 54.7 → 38.5 ms |
| **our own eye renderer** | 60× faster, matches FlyVis at r 1.000000 (difference at most 2.6e-4 - the float16 cache's precision) |
| **DCT-16 and per-coefficient z-scoring** | 230,720 → 92,288 numbers per state at 99.55 % of the energy; the knee chosen by measurement |
| **PCA through the Gram matrix on the card** | 43 s and $0.02 instead of the estimated minutes |
| **flow inversion by fixed point** | error 0.243 → 0.0058 at 4 model calls per step |
| **patching the hex lattice by Kuhn matching** | equal tokens, equivariance without absolute position |
| **13B's type mask as a native input** | "type not given" instead of "type is zero"; 0.837 → 0.904 |

All of part 2 on Modal cost about **$9.3**; all judging, every figure and
most experiments ran locally on a CPU, for free.

## 13. Mistakes caught along the way

For an ML engineer this section may be more useful than the results. We
recorded every mistake in [ISSUES](../../ISSUES.md) or in the step's report.

- **The gate lied** (section 5): compatibility with the brain is not a scene;
  a grey field is compatible with everything.
- **Comparing against the wrong distribution** (ISS-0010): draws were checked
  against the test latent instead of the training one, and a 20 % overshoot
  read as 44 %.
- **Three `nearest` functions** with different maths under one name
  (ISS-0011).
- **13B's noise was shared per batch** (ISS-0009): the cells of one figure
  were not drawn with the same z, and an interpolation showed a dip that was
  partly its own artefact.
- **Unequal patches** (23): the first version of the partition broke the
  equivariance.
- **The display ate the contrast** (ISS-0012): picture grids normalised every
  cell to one mean and spread, equalising exactly the contrast the captions
  compared: a high-contrast clip looked grey.
- **Recomputing report numbers from artefacts**: nine discrepancies, among
  them "the lr raised fivefold" instead of tenfold, a number with no artefact
  (15.8 % instead of 16.3 %) and two overstatements in the result's favour.
  All corrected with a mark.

The general lesson: **summary statistics disagreed with the picture six
times**, and every time the picture was right. That is why every judge now has
a floor, every result has a control from the same run, and the last word
belongs to a visual judgement against the raw video.

## 14. Where we are now

**The structure of the generated images has shifted a lot.** Draws of the
static flow match real states on the flat-field fraction and neighbour
coherence, the no-flow control separates grossly, and the diversity is within
the data's range. Structure that no earlier generator of this line had has
appeared. The indirect metrics all moved the right way at once.

**A scene is not proven - for three reasons, and none of them is "it did not
work":**

1. **There is no proper renderer.** The picture now comes from a least-squares
   matrix, which draws the conditional mean and blurs by construction. Even a
   real state is hard to recognise through it, so a scene cannot be judged by
   it. What is needed is a generative "state → picture" renderer, checked
   first on real states against real frames.
2. **There is no "this is a scene" metric.** Every judge of this line is
   structural and indirect: flat-field fraction, coherence, kurtosis, radius,
   nearest training clip. None answers whether the picture is meaningful. That
   needs an evaluation from conventional image generation - an FID/KID-like
   comparison of distributions and precision/recall on the features of a
   pretrained network - computed on the renderings beside the same number
   for real states.
3. **This is the first test of static generation.** One run, 10,000 steps,
   $0.42, with a target - the window's mean - that pulls toward a Gaussian.
   The bottleneck may be the data (eight temporal slices per clip give an
   eightfold corpus with no new simulation, and a non-Gaussian target) or the
   architecture (width, depth, training length); neither has been varied yet.

There is also a set of judges with measured floors and known ways of being
fooled, a flow on the hex lattice that carries over from video to statics
unchanged, and a clear order: re-measure in code the two numbers the static
branch rests on; train a renderer; add a scene evaluation; then a flow over
temporal slices.

## Reproducing

All figures are in [`docs/figures/`](../figures/) and are built locally
by the scripts in `tools/` from saved runs:

```bash
uv run python tools/fig_gh_part2.py      # figs. 1-5, 7, 8
uv run python tools/fig_gh_seed.py       # fig. 6
uv run python tools/fig_gh_static.py     # fig. 9
```

Every number in the article leads to a step report in `reports/` (step
reports are in Russian) and to a record in [`reports/runs.jsonl`](../../reports/runs.jsonl)
with its run id; known defects are in [`ISSUES.md`](../../ISSUES.md).

## References

- Lipman Y. et al. Flow matching for generative modeling. *ICLR* (2023). -
  training the priors over states.
- Ma N. et al. SiT: Exploring flow and diffusion-based generative models with
  scalable interpolant transformers. *ECCV* (2024). - the 13B generator.
- Kingma D. P., Welling M. Auto-encoding variational Bayes. *ICLR* (2014). -
  the VAE of section 6.
- Tong A. et al. Improving and generalizing flow-based generative models with
  minibatch optimal transport. *TMLR* (2024). - minibatch-OT in section 6.
- Ho J., Salimans T. Classifier-free diffusion guidance. arXiv:2207.12598
  (2022). - class conditioning and guidance in section 6.
- Kuhn H. W. The Hungarian method for the assignment problem. *Naval Research
  Logistics Quarterly* 2, 83-97 (1955). - partitioning the lattice into
  patches in section 10.
- Heusel M. et al. GANs trained by a two time-scale update rule converge to a
  local Nash equilibrium. *NeurIPS* (2017). - FID, section 14.
- Kynkäänniemi T. et al. Improved precision and recall metric for assessing
  generative models. *NeurIPS* (2019). - precision/recall, section 14.
- Soomro K., Zamir A. R., Shah M. UCF101: A dataset of 101 human actions
  classes from videos in the wild. arXiv:1212.0402 (2012). - the corpus of
  section 5.
- Lappalainen J. K. et al. Connectome-constrained networks predict neural
  activity across the fly visual system. *Nature* 634, 1132-1140 (2024). -
  the optic-lobe model, its eye and the Sintel data.
