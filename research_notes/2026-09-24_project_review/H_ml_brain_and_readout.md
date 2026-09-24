# H - ML review: the brain model and the readout side of the chain

Reviewer H (ML design of model zero, the decoders, the encoder inversion and
the 13B generator). 2026-09-24. Read-only review: no Modal, no download, no
file written except this note. Three short local checks were run in memory
with `.venv\Scripts\python.exe` on data already under `data/` (code and raw
output in Appendix A). They are **review evidence, not project runs**: they
are not in `reports/runs.jsonl`, use 2-48 clips, one ensemble member and one
seed, and should be re-derived in committed code before anything is built on
them.

Scope is judged against the project's own goal - a working generator, not a
paper (`AGENTS.md` "The project, in four lines"; `DECISIONS.md` 2026-09-18
"The deliverable is a working generator"). The prior side (items 17-29
flows, PCA, samplers) belongs to another reviewer and is touched only where
it meets the encoder or the renderer.

## 0. Summary

- **As ML, the readout side works, and for a structural reason:** the frozen
  encoder is deterministic, close to linear and close to invertible. Evidence
  from the project's own runs: the state of the averaged video equals the
  average of two states (inversion of 0.5·A + 0.5·B returns ½(A+B) at r 0.99,
  `reports/2026-09-19_step12_manipulated_states.md`); a linear hex-temporal
  decoder with 281 weights reads the video out of T4/T5 at r 0.927 and one with
  491 weights out of 14 types at 0.997 (`2026-09-19_train13_*`,
  `reports/2026-09-19_step13a_amortised_inversion.md`); eight random starts of
  the inversion converge to one video, pairwise r 1.000
  (`2026-09-20_gen13b_multi_init`); least squares reads a picture from 1,442
  static numbers at r 0.941 (`2026-09-21_prior23_pic`); the state is larger
  than its input - 1.13x a 20 x 64 x 64 source clip
  (`reports/2026-09-21_research_path_to_a_video_generator.md` §7.1), 3.2x the
  40 x 721 hexal video it is computed from (92,288 / 28,840). Every readout rung therefore succeeds, and 13B was a cheap and right
  choice for the moving track.
- **The weak points are not in the readout models but around them:** (1) what
  the encoder is - model zero is FlyVis member 000 transplanted, whose T5 is
  uncorrelated with FlyVis's T5 on natural input and whose T4 is almost
  sustained and space-time separable (review checks 1-2); (2) a protocol
  convention - every state starts from a grey steady state, so every state
  carries an onset transient that 13B learned to read, which made its failure
  on a constant-in-time state foreseeable; (3) the encoder used as the only
  judge - the round trip certifies reachability, not content, and the 18.x
  arm sweep optimised it while the draws stayed bland.
- **New in this review:** (a) a per-type fidelity table of model zero against
  FlyVis (30 s, would have caught ISS-0015 on day one); (b) model zero's
  still-frame T4a+T4b response is 96 % rank-1 separable and equals its own
  fixed point at r 0.999, while FlyVis's is 63 % separable and its fixed point
  carries almost no picture (T4a-to-frame r 0.04) - the static branch's object
  exists *because of* this encoder's defects; (c) the premise of 29.3 (r 0.922
  between a moving clip's DC and a still frame's state) reads 0.70 ± 0.22
  against frame 0 and 0.82 ± 0.14 against frame 20 on 48 UCF101 clips;
  (d) a fifth "residue" arm that was never tried - DC times a per-type
  temporal kernel - reconstructs a true still state at relative error 0.036,
  against 0.093-0.102 for the arms that were tried; whether 13B then renders
  it is a free local test.

---

## 1. Model zero

### Built and measured

- FlyVis 1.2.0 dynamics on the MaleCNS right-lobe export
  (`data/ol/filters_R_w5wk50m500oc.json`, 60 types, 31,526 units, 721
  columns), per-type and per-type-pair parameters of a FlyVis member
  transplanted by name, gain rescaled by total input per target cell and
  capped at 3 (`flydream/model/zero.py:122` `transplant`, `:101`
  `total_n_syn`; `config.toml:53` `rescale_cap`; `DECISIONS.md` 2026-09-18
  "A transplanted gain preserves total input").
- Validation: flash polarity (FRI) and moving-edge DSI/PD per member
  (`zero.py:275` `score_fri`, `:285` `score_dsi`). Mean T4/T5 DSI v5 0.020
  (`2026-09-18_step2_zero_R_v5`) → v7 0.109 (`..._v7_total_cap3`) → v9 0.152
  (`..._v9_wk50m500oc`) against FlyVis 0.391; flash polarity 0.922 against
  0.906; shuffled-strength control diverges in 3/3
  (`reports/2026-09-18_step2_model_zero.md` §2, §7).
- Per member in v9 (`step2` §7): member 001 T4 DSI 0.369 (FlyVis 0.611 on
  the same member), T5 0.006-0.020; **member 000 T4a/b/c/d 0.050, 0.074,
  0.145, 0.000 (mean ≈ 0.067); T5 selective (0.070-0.728) with preferred-
  direction errors 144°, 6°, 59°, 49°**.
- **The generator line uses member 000.** `load_network("malecns")` defaults
  to member 0 (`flydream/generate/invert.py:65`); every Modal entry point
  passes `model="malecns"` (`deploy/modal/generate_app.py:107,138,191,365,
  1553,1701,1779,1959`); the decode run records `"model": "malecns:0"`
  (`data/decode/2026-09-18_decode_sintel_malecns_v9/meta.json`). The member
  is not a `config.toml` setting. The article's "on the best member, 001, T4
  reaches 0.37" (`docs/articles/part1_en.md:120-122`) describes a member the
  generator never used.
- Training (3') was priced and never run: 1,200 iterations x 2 members for
  $0.47, ≈$1.5 per 6,000 iterations, ≈$15 per member at the reference sample
  budget (`reports/2026-09-18_step3_training_options.md` §5a, §5б;
  `2026-09-18_bench_150043_689575_229624b2a4` 0.973 s/iter at batch 16);
  declined 2026-09-19 "не сейчас" (`ROADMAP.md` Not started).
- ISS-0015 (`ISSUES.md`): L3 a lattice of isolated hyperpolarised cells,
  Tm9/T5a stripes, Tm1 sign flipped; found 2026-09-24, six days after the
  model was frozen, by drawing layers - the flash protocol could not see it.

### Review check 1 - per-type fidelity against FlyVis (Appendix A.1)

Same member-000 parameters, model zero against FlyVis on its own connectome,
Sintel clips 3 and 10, correlation of per-cell temporal fluctuations over
frames 5-39 (`r_fluct`) and of the spatial map at frame 20 (`r_space`):

| type | r_fluct | r_space | note |
|---|---|---|---|
| R1-R8, L1, L2 | 1.00 | 0.83-1.00 | input side faithful |
| Mi1 / Tm3 / Mi4 / Mi9 / Tm9 | 0.80 / 0.87 / 0.89 / 0.82 / 0.83 | 0.59-0.79 | T4/T5 inputs mostly faithful in time |
| L3 | 0.67 | **0.22** | spatial defect (ISS-0015) |
| **T4a / T4b / T4c / T4d** | **0.87 / 0.55 / 0.60 / 0.80** | 0.54-0.68 | the static object is T4a+T4b |
| **T5a / T5b / T5c / T5d** | **-0.20 / +0.03 / -0.10 / -0.07** | -0.06..+0.10 | **T5 is not FlyVis's T5 at all** |
| **Tm1** | **-0.61** | -0.62 | sign flipped (ISS-0015) |
| C3 | -0.33 | -0.16 | not in ISS-0015 |

Neighbour roughness at frame 20 of clip 3 (the ISS-0015 measure) adds that
T4's flank inputs are also rough in model zero: **Mi4 0.58 against 0.34, Mi9
0.56 against 0.39**, while its central inputs are clean (Mi1 0.29/0.32, Tm3
0.21/0.17). So ISS-0015 plausibly reaches T4 through Mi4/Mi9 as well as T5
through Tm9 - consistent with member 000's near-zero T4 DSI. One clip, one
frame: indicative only.

### Review check 2 - what T4a+T4b does with a still frame (Appendix A.2-A.3)

48 random clips of `data/corpus18/videos_000.npz` (UCF101 + procedural),
frame 20 held for the window, grey-baseline subtracted, T4a+T4b:

| | model zero (member 000) | FlyVis member 000 |
|---|---|---|
| still response, energy in the rank-1 (space x time) component | **0.964** [0.956-0.986] | 0.631 [0.518-0.829] |
| onset-window DC vs the fixed point after 2 s (24 frames) | **r 0.999**, amplitude 1.04 | r 0.960, 0.97 |
| T4a map vs the frame: onset DC / fixed point | +0.70 / **+0.69** | +0.29 / **+0.04** |

Model zero's T4a/T4b behave like a sustained, space-time separable luminance
filter: a held frame gives one spatial map times one temporal kernel, and the
state barely changes after the onset. FlyVis's T4 is transient and
inseparable; at its fixed point it carries almost no picture. This is the
mechanism behind 23.2's refutation of "T4/T5 are motion detectors, so no
static picture" (`reports/2026-09-21_step29_a_scene_as_one_picture.md` §2):
the refutation holds **for this encoder** and probably not for a
FlyVis-faithful T4.

### Fit to the step's task

Step 2's task was "the first running model on MaleCNS, and the effect of the
connectome alone" (`DECISIONS.md` 2026-09-18 "FlyVis is the reference model
and the source of the starting parameters"). The transplant did that in a day
for $0 and exposed a real bug (ISS-0003). Right call. The validation protocol
was too coarse for it: flash FRI and edge DSI are summary numbers over a full
field, and a spatial defect passes both (ISS-0015 "Costs"). Both networks
were loaded side by side on day one, so the per-type comparison of check 1
was available at no cost.

### Fit to the goal

For a working generator, the encoder needs to be deterministic, informative
and fixed; its biological fidelity does not decide whether the chain works.
The project's own diagnosis agrees: the bottleneck of the moving track was
the prior, not the brain or 13B (`2026-09-20_prior18_13b_diagnostic`;
`reports/2026-09-20_options_after_the_floor.md` variant A). So freezing
model zero was right for the generator. Its defects bite in two places:
(1) every public claim about "motion detectors" - member 000's T4 is barely
direction-selective and its T5 does not track FlyVis's T5 (checks 1-2); and
(2) the static branch, whose object - a picture in T4a+T4b's static state -
exists because model zero's T4 is sustained and separable (check 2). A
repair (item 31) may therefore remove the static branch's object.

### Better alternatives, with cost

1. **Choose the member by the target, not by FlyVis rank** (free to decide;
   ≈$0.85 and a day of work to redo downstream: corpus re-simulation ≈$0.15 by
   the pass in `2026-09-20_pairs18_wrong_model`, 13B maps + train ≈$0.27
   `step13b` report, static flow $0.42 `2026-09-21_prior29_static_ab`). Member
   000 was picked as "the best of 50" on FlyVis (`config.toml` [decode]
   comment); on MaleCNS its T4 is the weakest of the three measured. Worth
   doing only bundled with item 31, since both force the same re-simulation.
2. **Per-type fidelity against the reference on natural input as a standing
   validation** (check 1; free, 30 s local). Would have found ISS-0015 and the
   T5 problem on 2026-09-18.
3. **Activity distillation instead of, or before, the flow-task fine-tune**:
   fit model zero's 734 parameters so each type's response on Sintel matches
   the FlyVis member's (MSE per type, same differentiable simulator). It is a
   dense, well-posed target for "transplant faithfully", and the residual
   mismatch localises wiring defects for item 31. Per-iteration cost as the
   flow task (0.973 s at batch 16 on a T4); iteration count unknown - an
   estimate of 2-5k iterations, $0.5-1.5, is a guess, not a measurement.
4. **Fine-tuning on FlyVis's own objective** (3'): priced, never run. It can
   plausibly restore direction selectivity, but 734 per-type parameters cannot
   repair a spatially wrong wiring pattern (L3 dots, ISS-0015), so it is not a
   substitute for diagnosing the export. Note that `DECISIONS.md` 2026-09-18
   ("the transplant is followed by retraining on the same task before any
   decodability result is claimed") lapsed silently when rigour was parked;
   the public article makes decodability claims on the untrained transplant.

### Verdicts

| decision | verdict |
|---|---|
| transplant by type for step 2 | right call |
| gain rescale by total input, cap 3 (v7); weak-pair exception (v9) | right call; v9 is a stated trade (DSI +40 %, PD error x2) |
| validation by flash FRI + edge DSI only | mistake in hindsight (information available on day one) |
| model zero = member 000, hard-coded default | mistake for the claims, neutral for the generator; still open |
| not fine-tuning (3') | defensible for the goal; the decodability-claim condition lapsed |
| freeze model zero and build every prior on its states | right for the generator; the static branch now depends on its defects - still open with 31 |

---

## 2. Decoders

### Built and measured

Ridge per type with the penalty tuned per type on a relative grid, controls
`shuffle_time` and `shuffle_samples` (`flydream/decode/ridge.py`); the hex-conv
rung from FlyVis's `DecoderGAVP` (`flydream/decode/hexconv.py`); metrics,
raster, map driver (`reports/2026-09-18_step4_decoder_stack.md` §2).
- Edge protocol: real 0.902, condition-shuffle 0.903 - the protocol measured
  frame number, not the state (`2026-09-18_decode_edges_flyvis`; step 4 §4).
- Sintel, FlyVis member 000: the stage curve inverts with the decoder window
  (T5a control-corrected 0.057 → 0.625), Spearman -0.827 between lag-0 score
  and gain to 160 ms (step 4 §9-10; ISS-0004); even lags alias Sintel's frame
  hold (ISS-0006).
- Model zero, consecutive lags t and t-1: R1 1.00, Mi1 0.99, Tm9/T4a 0.83,
  T5a 0.71, time-shuffle 0.53-0.63 (`docs/articles/part1_en.md:192-195`,
  `tools/fig_gh_decoding.py`; not in `runs.jsonl` - the logged lag-0 values
  are T4a 0.735, T5a 0.686, Mi1 0.90, `2026-09-18_decode_sintel_malecns_v9`).
- The hex-conv rung has no measured result in any report or run record; the
  13A CNN (its own `HexTemporalLayer`) superseded it.

### Fit to the step's task

Ridge per type plus a time-shuffle null is the standard first rung, and the
per-type penalty (instead of normalising by FlyVis's 76-124,756x constants) is
a sound choice with a test behind it (step 4 §2). The edge protocol was a
design mistake that its own control caught the same day - good practice.
The lag-window result is a real finding and fed the 5-frame kernels of 13A.

### Fit to the goal

For the generator the map was a detour: the deliverable needed a readout
that works, not a stage curve over ten members and five splits. About 3 M
subagent tokens went into it (step 4 §7 and §9: 1.08 M + 1.45 M + 0.48 M)
before the human reset the contract on 2026-09-18 night (`DECISIONS.md`
"The deliverable is a working generator"). One transfer to the current
branch matters: **for a static target the time-shuffle is not a null.** It
keeps the scene and destroys only the frame order, which is why it reads
0.53-0.63; for the static branch the right null is a between-clip shuffle,
which 29a used (0.002, `2026-09-21_prior23_pic`).

### Better alternative

For the generator goal: one ridge on T4/T5 with a 5-frame consecutive window,
time shuffle and between-clip shuffle, one member, one split - an hour of CPU
instead of a day and ~3 M tokens. The information to choose this was not
available until the human's 2026-09-18 night reset.

### Verdicts

| decision | verdict |
|---|---|
| ridge per type, per-type penalty, time-shuffle null | right call |
| edge protocol first | mistake, caught by its own control the same day |
| full decodability map (lag sweep, ensemble, audit) | defensible at the time (paper-shaped contract); not needed for the goal |
| hex-conv rung built, never measured | minor waste; superseded by 13A |

---

## 3. Encoder inversion

### Built and measured

Video (frames x 721 hexals) optimised by Adam through the frozen network to
match a type's activity, TV prior 0.02, 150 steps, lr 0.05, window 40 + 5
margin, all tasks batched (`flydream/generate/invert.py:146,189`;
`config.toml` [generate]). r 0.93-1.00 across the ladder, wrong-target control
-0.20 (`2026-09-19_generate_malecns_s3_40f5`,
`reports/2026-09-19_step9_window_margin.md`); ≈$0.05 per ladder batched
(`..._batch20`). Dreams-lite and manipulated states with a shuffled-state
control (`reports/2026-09-19_step11_dreams_lite.md`, `..._step12_...`).
Multi-init: eight starts, pairwise r 1.000 (`2026-09-20_gen13b_multi_init`).

### Fit to the step's task

As an existence proof and as the ceiling for learned readouts, right, cheap and
well-controlled (the item-11 shuffled-state control, r ≈ 0, is the good one).
The multi-init test before 13B is exemplary: it decided in advance that z has
nothing to choose at the full mask (`reports/2026-09-20_step13b_generative_decoder.md` (a)).

### As "the ground truth for readability"

Overstated. In a noiseless, deterministic, near-linear encoder, inversion r
measures injectivity and conditioning, not how much of the picture a noisy
reader could use. Two numbers in the project already said so: the wrong-target
control is simply the correlation between clips 3 and 10 (-0.20 in every row
of step 9), not a null for the method; and T5a inverts at r 0.933 while its
spatial-gradient energy is 0.26-0.35 of the true clip (step 9 "Numbers") - r
is dominated by coarse layout. The project adopted a detail metric only at
22.8 (`reports/2026-09-21_the_draw_distribution.md` §9, sharpness 71/100 for
13B from a real state). The honest claim is "the video is recoverable from
this encoder's T4/T5 state", which is what `DECISIONS.md` 2026-09-18 ("Dream
is the project's name") already requires.

### Better alternative

Report a detail metric (gradient energy or flat fraction on the calibrated
scale) beside r from step 9 on; it was measured there and dropped. Free. A
noise-robustness control (inversion from a target with neuron noise at a
stated SNR) would separate "readable" from "invertible", but that is
paper-grade and not needed for the goal.

### Verdicts

| decision | verdict |
|---|---|
| pixel inversion by Adam through the frozen model | right call |
| inversion r as evidence that "the picture survives" | defensible, overstated; detail metric dropped |
| multi-init before designing 13B | right call |

---

## 4. Generator 13B

### Built and measured

`SiTColumns` (`flydream/generate/gen13b.py:176-201`): 721 column tokens, each
token the 40 frames of x_t plus the masked state's 8 x 40 values plus 8 mask
bits through one linear layer to width 128, learned absolute position, four
adaLN-Zero blocks with global attention, 1.40 M parameters; SiT linear
interpolant, logit-normal t (`:215`), Euler 20 steps, CFG. Trained 20,000
steps at batch 32 on the pairs13 corpus - 189 Sintel clips x 12 flips/rotations
plus 7,200 procedural clips, split by scene and class - for ≈$0.22, val loss
"still falling slowly at the end" (`2026-09-20_gen13b_sit_train`; 13B report
(c)). Held-out r 0.966 on 8 clips, round trip 0.031
(`2026-09-20_gen13b_samples`). **Not retrained on UCF101**: it renders corpus
states at round trip 0.016 against 0.011 on Sintel
(`2026-09-20_corpus18_build`; `reports/2026-09-20_step18_corpus_of_ordinary_video.md`).
Knobs: T4 alone 0.033 ≈ full 0.027, T5 alone 0.112 (13B report, knobs table).
Renderer ceiling on the calibrated scale: sharpness 71 of 100, neighbour
coherence 0.815 against the raw video's 0.784, i.e. it over-smooths
(`reports/2026-09-21_the_draw_distribution.md` §9). Static state: DC + zeros
0.257, + mean residue 0.351, + another still's residue 0.137 (rendered sharply,
of the donor), + noise 0.045, against 0.941 for the true still state
(`2026-09-21_prior23_resfix`); a still frame's true state renders at 0.953
(`2026-09-21_prior23_static`). Evaluation defects: ISS-0013 (shuffled control
scored against the wrong state), ISS-0014 (13A/13B round trips on different
scales).

### Fit to the step's task

Flow matching with the SiT interpolant, structured type masks and EMA was a
modern, cheap (≈$0.47 all in), well-benchmarked design (13B design §5-6).
Conditioning on T4/T5 alone was right for the story and kept L1/L3 from
short-cutting (13B design §2). Given the multi-init result, 13B at the full
mask is effectively a learned inverse; its generativity pays off on partial
masks and unreachable states, where it degrades more gracefully than the
deterministic decoders (T4a x 0.5: linear 2.38, CNN 0.508, 13B 0.287, 13A and
13B reports). The backbone was chosen because the hex-ResNet ran out of memory
at batch 32 in an fp32 gather, not on quality (13B report (b)) - an
engineering reason; later the prior found locality decisive (step 23).

### Fit to the goal

Right for the moving track: the project's own diagnosis cleared 13B as the
bottleneck (`2026-09-20_prior18_13b_diagnostic`). Two limits matter now:
- **effectively conditioned on T4**: T5 adds little (knobs) and is not
  FlyVis-like (check 1), so the "T4/T5 → video" story is really "T4 → video";
- **the renderer ceiling is 71/100** and over-smoothed; val loss was still
  falling at 20 k steps, so part of that is under-training. Unmeasured.

### Was the failure on a static state foreseeable?

Yes, from the project's own evidence by 2026-09-20.
1. **Every training state starts from a grey steady state**
   (`flydream/generate/pairs13.py:129` `steady_state(t_pre, ..., value=0.5)`;
   `config.toml` `t_pre = 1.0`) and no procedural class holds a textured
   image still (`flydream/generate/stimuli.py:11-19`; flash is uniform, every
   other class moves). So every state 13B saw carries the grey→scene onset,
   and a state constant in time never occurs.
2. **13B reads the picture from that onset.** In model zero a held frame's
   T4a+T4b state is 91 % DC energy (check 2: DC-repeated relative error
   0.093), yet DC alone renders at r 0.257 against 0.953 for the full still
   state (29 §3); "energy is not information here" (29 §3) has a mechanism: the
   content 13B uses sits in the onset transient.
3. **Learned renderers answer unreachable states with their prior** - gain
   edits return the original clip, a hand-written stripe returns grey
   (`reports/2026-09-20_step14_controllable_generator.md` 14.1-14.2; 13B report
   "On unreachable states the prior wins"). A DC-only state is one more
   unreachable state.
The failure itself cost $0 (local). The real cost is the object: the static
branch was defined as a statistic of moving states (DCT-0), which no stimulus
produces, so it has no renderer and no ground truth (hence 29.3 and 30).

### A missed arm: the multiplicative residue (review check, Appendix A.2)

The four arms tried supply the temporal residue additively or from elsewhere.
Because model zero's still response is ≈ one spatial map times one temporal
kernel (check 2), the residue can be predicted from the DC itself:
`state(t) ≈ h_type(t) · DC`, with `h` fitted per type on a few true still
states. On 16 held-out UCF101 frames (h fitted on 32) this reconstructs the
true 40-frame still state at **relative error 0.036, r 0.972**, against 0.102
for the additive mean residue and 0.093 for the DC repeated (the arms of
`prior23_resfix`). Whether 13B renders the reconstructed state near its 0.94
ceiling is untested; it is a free local run of minutes (13B inference on CPU
ran in 99 s in step 14). On FlyVis's T4 the same trick fails (0.441), so it is
a property of this encoder.

### Better alternatives, with cost

1. **Burn-in instead of a grey onset** for states meant to be stationary: run
   the clip for N frames before the recorded window (or start from the first
   frame held) so that no state carries the window edge; this removes 29.1's
   "frame 0 broken" slice artefact (ROADMAP 29.1) at +N/45 simulation cost.
   Applies at the next re-simulation (item 31 or a new corpus); not worth a
   re-simulation on its own.
2. **A still class in 13B's training set** (held Sintel frames, held textures;
   ≈10 % of pairs13) - retrain ≈$0.27. It would make still states in-
   distribution; it does not rescue a DC-only object, which stays unreachable.
3. **Renderer quality**: 100 sampler steps instead of 20 (free, local; 18.7
   found steps an untouched lever for prior samples, `2026-09-20_prior18_why18`)
   and a wider/longer 13B (≈$0.5-1 for 2x width, 40 k steps; effect unmeasured)
   - only if the moving branch resumes.

### Verdicts

| decision | verdict |
|---|---|
| conditional flow matching (SiT interpolant, masks, EMA, CFG) | right call |
| SiTColumns over hex-ResNet (chosen on memory, not quality) | defensible; whether locality lifts the 71/100 ceiling is open |
| condition on T4/T5 only | right for the story; effectively T4 |
| 40-frame window from grey, no burn-in, no still class | defensible at the time (inherited from inversion); a mistake for the static branch in hindsight |
| 1.4 M params, 20 k steps, val still falling | defensible (cheap); under-trained, open |
| not retraining on UCF101 | right call (measured) |
| rendering DC with 13B (23.1, 29a) | foreseeable failure, cheap; the multiplicative-residue arm was missed |

---

## 5. The frozen model as its own verifier

The round trip `state → 13B → video → frozen brain → state′` became "the unit
of evidence" (`docs/PROJECT_MAP.md` Boundaries; `DECISIONS.md` 2026-09-20).

- **Not circular for what it tests.** An inverse must be checked against its
  forward model; for 13B and for "is this sampled state reachable", the round
  trip is the right quantity.
- **Blind by construction to two things.** (1) The encoder's own defects:
  ISS-0015 passes every round trip. (2) Content in the encoder's near-null
  space: T4 is a spatial low-pass, so a bland video with the right coarse
  layout round-trips well. The project measured both effects itself:
  generated states rendered as "smooth blobs" at round trips 0.020/0.028
  against 0.014/0.009 for real clips ("a direct demonstration that the round
  trip does not measure content", `2026-09-20_prior18_13b_diagnostic`); the best
  arm reached 0.0166, *below* a real clip's DCT-16 floor 0.0209 ("the gate
  rewards bland states", `2026-09-20_prior18_dct16_w384_gate`); and the closed
  loop kept a round trip of 0.02-0.05 at every pass while r to the start clip
  fell 0.99 → 0.26 (clip A) and 0.99 → 0.10 (clip B) in 11 passes, converging
  from white noise to a noise video at 0.007 (step 14, 14.3).
- **Scales were not unified** (ISS-0013, ISS-0014), so "13B equals the 13A
  CNN" and "30x under the shuffled control" do not stand as written.

Verdict: right as a reachability check; a mistake as the main quality gate of
the 18.x arms (sixteen gate records `2026-09-20_prior18_*` with metric `roundtrip_median_prior_samples`), which optimised a
number that could not see what the human wanted. An independent picture judge
(29.4) should have come with the round trip, not after it. For 29.4, two
small-sample tools fit the setting better than FID: KID, unbiased at small n
(Binkowski et al., "Demystifying MMD GANs", ICLR 2018), and a classifier
two-sample test, which needs no pretrained network - a small classifier
trained to tell real renders from drawn ones, accuracy 0.5 = indistinguishable
(Lopez-Paz & Oquab, "Revisiting Classifier Two-Sample Tests", ICLR 2017).

---

## Top lessons (ranked)

1. **The encoder is near-linear and near-invertible, so a state prior is a
   picture/video prior in linear coordinates.** The brain gives no compression
   (the state is 3.2x the hexal video it encodes; research 2026-09-21 §7.1) and no semantic
   abstraction at T4 level. The scene problem is an unconditional generation
   problem at N ≈ 13.5 k, and budgets and expectations should come from that
   literature (research 2026-09-21 §7.3), not from the brain.
2. **Validate a frozen component against its reference on natural input, per
   type, on day one.** Flash and DSI summaries missed ISS-0015 for six days;
   a 30-s per-type correlation shows T5 at r ≈ 0 and Tm1 at -0.61 (check 1).
3. **The verifier cannot be the only judge when it is the encoder.** The round
   trip rewarded bland states for a whole sweep (§5); a content judge must be
   in place before a sweep, not after it.
4. **Protocol conventions become the data distribution.** The grey onset made
   the window edge part of every state; 13B learned to read the picture from
   it, and a constant-in-time state became unreachable (§4). Decide the
   simulation protocol for the object you will generate, not for the one you
   inverted first.
5. **The static branch's object is a property of this encoder.** Model zero's
   T4 is sustained and separable (fixed point = onset DC at r 0.999, T4a-frame
   0.69); FlyVis's T4 fixed point carries almost no picture (0.04). Any
   encoder change (item 31, a member switch, fine-tuning) can remove the
   object.
6. **Select frozen components by the criterion of the task they serve.**
   Member 000 was chosen by FlyVis rank; on MaleCNS its T4 is the weakest
   measured (≈0.067 against 0.369 for member 001), and the choice sits in a
   code default, not in `config.toml`.
7. **Carry a detail metric beside r from the first picture.** Step 9 measured
   T5a's gradient energy at 0.26-0.35 of the true clip at r 0.933; the project
   needed until 22.8 to make sharpness a standing judge.

---

## Implications for the resumed order 29.3 → 30 → 29.4 → 29.2 → 31 (options, not decisions)

- **29.3 - expect the fallback, and make the fallback the plan.** The review
  check (48 UCF101 clips, model zero, T4a+T4b) gives corr(DC of the moving
  clip, DC of its frame held still) **0.697 ± 0.224 against frame 0 and
  0.822 ± 0.135 against frame 20**, control (another clip) -0.001 - well below
  the 0.922 the branch rests on (29 §5, never in an artefact). Options:
  (a) run 29.3 as written, pairing against the middle or mean frame rather than
  frame 0; (b) skip the conversion and build the static corpus from **true
  still states**: in model zero a held frame's state ≈ its fixed point
  (r 0.999), so the object is exactly 721 x 2 numbers with ground truth by
  construction. Local cost measured here ≈ 0.4 s per 45-frame clip per model
  on the owner's CPU at batch 48, i.e. ≈1.7 h for one frame of each of the
  15.5 k corpus clips, ≈14 h for eight per clip; free. Or ≈$0.15 per 15.5 k on
  a T4 by analogy with `2026-09-20_pairs18_wrong_model`.
- **Before 30 - a free test that might defer it.** Feed true still DCs, and the
  existing prior29 draws, through `h_type(t) · DC` into the existing 13B on six
  held-out clips; compare with the 0.941/0.953 ceilings of 29a/23.2. If it gets
  close, a sharp first renderer exists without new training code; if not, 30
  proceeds as priced (0.15-0.50 dollars, ROADMAP 30).
- **30 - train on what the prior will produce.** If 30 is built, its pairs are
  free and unlimited (any image → held → brain) - a conditional, nearly
  deterministic renderer is not data-limited the way the prior is. Train it on
  true still states, not on converted moving DCs, so it inherits no conversion
  error; an optional brain-consistency term only enforces reachability (§5).
- **29.4 - pick small-sample judges.** KID and a classifier two-sample test
  (§5) work at 64-256 draws, need no adaptation of an ImageNet network, and
  read beside real states through the same renderer and the no-flow control.
- **29.2 - slices of moving states are the conversion problem eight times.**
  A slice at frame k mixes the motion history; eight *held* frames per clip give
  the eightfold corpus with ground truth (option (b) above). The kurtosis
  argument (slice 5.26 against DC 3.17, 29.1) was measured on slices by a
  script that was not kept; the kurtosis of true still states is unmeasured.
- **31 - decide its position knowingly, because it interacts with the static
  object.** Check 2 says a FlyVis-like T4 would carry almost no static picture.
  Options: (a) run 31's diagnosis first (local, free) to learn whether the fix
  touches T4 (Mi4/Mi9 roughness suggests it might); (b) keep model zero frozen
  for the static branch and state every claim as "this encoder"; (c) if the
  repair lands, move the static object to sustained medulla types - in FlyVis
  on one clip the DC of Mi1, Mi4, Mi9, Tm3 correlates with the frame at |r|
  0.93-0.97 (Appendix A.1, first block). Bundle a member switch with 31 so the
  corpus is re-simulated once (≈$0.15 re-simulation + ≈$0.27 13B + ≈$0.42
  static flow + renderer).
- **Across all of them - a free A/B already exists.** `/runs/pairs18` on the
  volume holds FlyVis member 000's states over the whole UCF101 corpus (13 GB,
  the "wasted" run `2026-09-20_pairs18_wrong_model`). It is a ready control for
  "does the static branch survive a correct encoder" - reading it is a download
  above 1 GB or a CPU container, so it is gated and priced when proposed.

---

## Gaps

- The review checks use Sintel clips 3 and 10 (checks A.1) and 24-48 random
  clips of one corpus shard (A.2-A.3), one member (000), one seed; they are
  not in `runs.jsonl` and were not re-derived in committed code.
- Check A.2's DC is the baseline-subtracted mean over frames 0-39; the
  corpus's DC is DCT-0 of per-type z-scored maps. Per-type affine differences
  may shift the pooled correlation slightly.
- The multiplicative-residue arm was scored in state space only; whether 13B
  renders it near its ceiling is untested.
- FlyVis's T4 in the checks runs on FlyVis's own connectome; how a *repaired*
  model zero's T4 would behave is inferred, not measured.
- The effect of fine-tuning (3') or activity distillation on direction
  selectivity or on ISS-0015 is unmeasured; the distillation cost is a guess.
- 13B's sharpness ceiling (71/100) was not decomposed into under-training,
  sampler steps and architecture.
- Not read in full: `reports/2026-09-20_step18_5_width_and_the_representation_floor.md`
  (1,037 lines), the prior-side reports 17-26 beyond what touches the
  encoder or the renderer, and `research_notes/` except through the reports
  that cite them. No web search was run; the two §5 references are cited from
  knowledge, not re-checked.

---

## Appendix A - review checks (in-memory, nothing written)

All run from `D:\ML\Fly_Brain` with `.venv\Scripts\python.exe -` (stdin),
using `flydream.generate.invert.load_network`, `clip_from_sintel`, `simulate`
and `flydream.decode.pairs.type_index`. Timings on the owner's machine.

**A.1 (≈30 s).** Sintel clips 3 and 10, 40 frames + 5 margin, from the 1 s grey
steady state, both `"malecns"` and `"flow/0000/000"`. Per type: cells placed
on the 721 lattice by (u, v); `r_fluct` = correlation of per-cell
mean-removed activity over frames 5-39 between the two models; `r_space` = the
same on frame 20. Roughness as `tools/cmp_layers_flyvis.py`. First block
(clip 3, frame 20; roughness, corr with frame): model zero Mi9 0.56 / -0.70,
Mi4 0.58 / +0.74, Mi1 0.29 / +0.95, Tm3 0.21 / +0.93, T4a 0.40 / +0.61, T5a 0.46
/ -0.58; FlyVis Mi9 0.39 / -0.86, Mi4 0.34 / +0.89, Mi1 0.32 / +0.77, Tm3 0.17 /
+0.89, T4a 0.33 / +0.40, T5a 0.18 / +0.25. Still frame 20 held, DC vs frame:
FlyVis Mi1 +0.97, Mi4 +0.96, Mi9 -0.94, Tm3 +0.93, T4a +0.44, T4b -0.05.
Full per-type table in §1.

**A.2 (≈130 s).** 48 clips drawn with `default_rng(0)` from
`data/corpus18/videos_000.npz`; conditions: moving clip, frame 0 held, frame
20 held, grey. T4a+T4b over frames 0-39, grey response subtracted.
Separability = σ1² / Σσ² of the (40 x 1442) still response. Multiplicative
residue: `h_type(t)` by least squares on clips 0-31, tested on 32-47.
Output:

```
malecns: still rank1 0.964 [0.956..0.986] | mult-residue rel.err 0.036 r 0.972; additive-mean-residue rel.err 0.102; DC-repeated rel.err 0.093 | corr(DCmov,DCstill0) 0.697±0.224 corr(DCmov,DCstill20) 0.822±0.135 control other clip -0.001
flow/00: still rank1 0.631 [0.518..0.829] | mult-residue rel.err 0.441 r 0.735; additive-mean-residue rel.err 0.502; DC-repeated rel.err 0.498 | corr(DCmov,DCstill0) 0.597±0.249 corr(DCmov,DCstill20) 0.737±0.170 control other clip 0.039
```

**A.3 (≈45 s).** 24 frames (`default_rng(1)`, frame 20 of each clip) held for
100 frames; DC of frames 0-39 against the mean of frames 95-99 (fixed point);
T4a map against the frame. Output:

```
malecns: corr(DC onset window, fixed point @2s) 0.999±0.000; |fixed point|/|DC| 1.04; T4a corr with frame: DC +0.70, fixed point +0.69
flow/00: corr(DC onset window, fixed point @2s) 0.960±0.017; |fixed point|/|DC| 0.97; T4a corr with frame: DC +0.29, fixed point +0.04
```

Sim speed: ≈0.4 s per 45-frame clip per model at batch 48 on 32 threads
(A.2: 290 clip-simulations plus two model loads in 131 s).
