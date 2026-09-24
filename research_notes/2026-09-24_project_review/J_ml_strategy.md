# J - ML strategy retrospective (reviewer J, 2026-09-24)

Scope: the whole arc of "What Does a Fly Dream Of?" against its idea - what
was done, in what order, which steps paid, which were detours, what a leaner
path would have looked like, and whether the resumed plan converges. Read-only
review: nothing was run on Modal, nothing in the repository was changed. Two
numbers below were recomputed locally from files already on disk (marked
"recomputed here"); everything else cites a report, run id, DECISIONS/ROADMAP
entry or commit.

Timeline convention: wall-clock pace is taken from commit timestamps
(`git log --reverse --format='%ad %s' --date=format:'%m-%d %H:%M'`); report
file names and ROADMAP dates sometimes sit a day later than the commit that
measured them (e.g. 13B commit 09-19 09:28, ROADMAP "13B (2026-09-20)").
Money is the Modal spend written to `reports/runs.jsonl` (151 records, total
**$12.62**), summed by run id; it is a lower bound (see Gaps).

---

## 1. The arc

| Phase | Commit time | Steps | Question | Outcome | Modal $ (runs.jsonl) | Wall-clock |
|---|---|---|---|---|---|---|
| **Reading, part a** | 09-18 (pre 17:19) - 09-18 22:57 | 1 data, 2 model zero, 3 training priced, 4 decoder ladder, first audit | Can MaleCNS run FlyVis dynamics, and what can be read from each layer? | Export + transplant work; the transplant bug found and fixed, DSI 0.020 -> 0.152 (runs `2026-09-18_step2_zero_R_v5`, `_v9_wk50m500oc`; ISS-0003). Step-4 stage curve turned out to be a decoder artefact (ISS-0004). Step 3 priced training ($18 -> ~$12 per member) and a 1.17x kernel benchmark; training never run (ROADMAP "3' ... paused"). Ended by the owner's course change (DECISIONS 2026-09-18 night, "мне кажется ты тратишь мои время и деньги"). | **0.90** (all of it step-3 smokes and the benchmark) | ~0.5 day + pre-commit work |
| **Reading, part b** | 09-18 23:00 - 09-19 10:41 | 8-9 inversion + window, 10' batching, 11 dreams-lite, 12 manipulated states, 13A, 13B, 14 | Can a state be turned back into video; what happens on states no clip caused? | Inversion from every stage at r >= 0.92 on MaleCNS (commit 09-19 00:05); batched ladder 948 -> 176 s (`2026-09-19_generate_malecns_s3_40f5_batch20`); model zero decays to rest without input (step 11, `2026-09-19_dream_malecns_neuron_noise` contrast 0.067 vs 0.329); **13B, the core asset: held-out r 0.966, $0.42** (`reports/2026-09-20_step13b_generative_decoder.md:74`); arbitrary states unreachable, round trip 1.85-2.33 vs 0.029 (step 14); the ignored T4a stripe scores 0.095, better than obeyed conditions (`reports/2026-09-20_step14_controllable_generator.md:49,93`). | **2.45** (8-12 $0.42; 13A $1.61; 13B $0.42; 14 $0) | ~12 h |
| **Dream/source design** | 09-19 10:01 - 10:19; 17 at 09-19 22:09 - 09-20 00:30 | 16 (designed, deferred), 17.0-17.3b | Can the model generate its own states? If not, can states be drawn from a learned prior? | 16: model zero has no intrinsic dynamics; 25 % of the ladder's input and 91 % of T4/T5 output are types not in the model (`reports/2026-09-20_step16_dream_source_design.md` §2); deferred as "not fully formed" (ROADMAP Not started). Goal set by the owner: "я хочу получать новые видео" (DECISIONS 2026-09-20). 17.0 free baseline: unconditional 13B already gives new video, texture not scenes (nn r 0.53 vs 0.52, `2026-09-20_baseline17`). 17.1 flow over states: round trip 1.19 (next to the shuffled 1.77); 17.1b DCT-16 Sintel-only: 0.142; 17.3b exact flow inversion. | **0.37** | ~3 h |
| **Corpus and priors** | 09-20 00:59 - 09-21 15:01 (11 h break 09-20 15:21 - 09-21 02:07) | 18.0-18.24c (~30 sub-steps), research x3, 19, 20-21, 22.0-22.8, 23 | Make a drawn state give a scene | Corpus of 15,514 clips (12,411 from UCF101 + 3,103 procedural, `2026-09-20_corpus18_build`) with a 60x eye renderer at r 1.000000 (18.0). Levers pushed the round trip to and below its own floor (0.0166 vs 0.0209, 18.13) while videos stayed texture; gate retired (`reports/2026-09-20_the_seed_problem.md` §5). Seed-geometry line and VAE ($2.69) negative; PCA first stage (19) and object shrinking (22) negative ("сжать удалось вчетверо, на задачу потока это не повлияло никак", `reports/2026-09-21_step22_a_smaller_object.md` §0). Metric audit found ISS-0010/0011. **Hex-local flow (23, $0.51) moved every number**; fresh draw sharpness 11 -> 22 of 100 (ceiling 66), still not a scene. | **8.48** (18: $7.58; 19-23: $0.90) | ~27 h active |
| **Static branch** | 09-21 15:11 - 17:29; reports 09-22 08:39 - 09:15 | 23.1-23.2, 26, 29a, 29, 29.1 | Is there a still picture in this state space, and can a flow draw one? | A held still frame renders at 0.953 vs 0.916 for a clip (`2026-09-21_prior23_static`); 13B cannot render a time-constant state (best crutch 0.351 vs 0.941, `2026-09-21_prior23_resfix`); a least-squares map DC -> mean frame reads r 0.941 (`2026-09-21_prior23_pic`) but blurs; static flow over 721x2 matches real states on flat fraction (27.1 vs 27.1) and coherence (0.928 vs 0.949) through that blurring map (`2026-09-21_prior29_static_ab`); 26 showed kurtosis can be bought by shrinking the output. The owner cannot judge the flow through the blurred renderer (ROADMAP Done 29). | **0.42** | ~2.5 h + reports |
| **Pause and publication** | 09-23 21:36 - 09-24 09:56 | order 29.3 -> 30 -> 29.2 (+29.4); P1-P5 | Tell the story; go public | Two articles (EN/RU), README, figures, number checks that found ISS-0013/0014 and corrected published numbers; ISS-0015 (OFF pathway broken) found while drawing a public figure (commit 09-24 06:39); repository public (ROADMAP P5). | 0 | ~12 h |
| **Total** | 09-18 - 09-24 (280 commits; 102 touch no code; ROADMAP edited 93 times) | | | | **12.62** (reading $3.35, part 2 $9.27 - matches the article's "about $9.3", `docs/articles/part2_en.md:25`) | ~4.5 active days of research + 1 of publication |

Subagent research spend that is recorded: 09-18 landscape ~630k tokens
(DECISIONS 2026-09-18, delegation entry); step-3 speed scout 122k
(`reports/2026-09-18_step3_training_options.md` §5a); 09-20 video-generation
research 2.03M tokens (`reports/2026-09-20_research_video_generation_and_training.md:237-244`);
09-21 VAE research 858k (`reports/2026-09-21_research_how_vaes_are_trained.md:7`);
09-21 path research ~826k (`reports/2026-09-21_research_path_to_a_video_generator.md:9`).
About 4.5M subagent tokens in all, more than half of it on 09-20/21.

### What paid, what was a detour (by value per dollar and per hour)

| Paid | Evidence |
|---|---|
| Transplant fix (free) | DSI 0.020 -> 0.109 -> 0.152 (ISS-0003; runs `_v5`, `_v7_total_cap3`, `_v9_wk50m500oc`) |
| Inversion ladder on a T4 + batching ($0.22) | r >= 0.92 at 10 stages; 948 -> 176 s |
| 13B ($0.42, ~1 h of commits 08:16 - 09:28 on 09-19) | held-out r 0.966; the component every later line uses |
| Corpus build with own eye renderer (~$0.36 incl. $0.17 wasted/failed) | 15,514 clips, r 1.000000 to FlyVis, 60x faster (`docs/articles/part2_en.md` §5) |
| Free local diagnostics | 17.0 baseline; Option A "13B is not the bottleneck" (`2026-09-20_prior18_13b_diagnostic`); 18.21 K ceiling closed a run before paying; 21b hexcov 0.876 vs 0.267; 23.2 still frame; 29a LS map |
| Hex-local flow (23, $0.51) | the only architectural lever that moved every measured number (ROADMAP Done 23) |
| Static flow (29, $0.42) | first draw not separable from a real state on two structural judges |

| Detour | Cost | Why a detour |
|---|---|---|
| Step 3 training pricing, packing, batch sweep, kernel benchmark | $0.90, ~5 h | training never run; the owner named it (DECISIONS 2026-09-18 night) |
| Ensemble decodability map, lag sweeps, decode on Modal | ~4 h, parked | paper-shaped rigour, later parked (ROADMAP Not started, "Paper-grade rigour") |
| 13A amortised inversion | $1.61 | superseded by 13B two hours later; its comparison to 13B was on different scales (ISS-0014) |
| 18.4-18.16 levers optimised against the round trip | $4.31 (18.4 $1.90, 18.5-18.6 $0.73, 18.11/18.13 $1.06, class line $0.62) | the gate was already known not to read content (step 14 stripe; 18.3 "texture passes the circle perfectly", `reports/2026-09-20_options_after_the_floor.md`); width was the one real finding |
| Seed-geometry and VAE line (18.17-18.24c) | $2.69, ~half a day | ended as "the geometry of the preimage and the quality of a sample are two different diseases" (`docs/articles/part2_en.md` §6); the VAE premise was withdrawn the next day by research (DECISIONS 2026-09-21) |
| PCA first stage and shrinking the object (19-22) | ~$0.39, ~10 h | negative; the conclusion "a linear first stage destroys the geometry" led to 23 |
| 26 draw-time corrections | $0, ~1 h | negative; showed the kurtosis gate was gameable |

---

## 2. How the goal moved, and where the project followed its own principle

### 2.1 The moves

| Date | From -> to | Forced by evidence or chosen | Evidence |
|---|---|---|---|
| 09-18 (plan) | concept: layer-wise information map, then dreams from internally generated activity after a central brain | - | `what_does_a_fly_dream_of.md` Stages 1-5; DECISIONS 2026-09-18 "The order is data, model zero, training, decodability map, extensions, central brain, dreams" (preliminary); initial plan put dreams 6-10 months out (`reports/Коннектом мухи и план проекта.md` §7) |
| 09-18 night | paper-shaped map -> "a working generator, not a paper" | **chosen**, triggered by cost and time, not by a result | DECISIONS 2026-09-18 night: the day "went into price calibration ... none of it moving the generator" |
| 09-19/20 | dreams (model's own activity) -> a learned prior over stimulus-driven states | **mostly forced**: the model has no intrinsic dynamics and no central brain | step 11 (neuron noise gives a faint ripple); step-16 design §2 (25 % / 91 % of connectivity missing); DECISIONS 2026-09-18 ("a feed-forward optic-lobe model with its input removed decays to rest"); initial report §1 (optic lobes process normally in sleep, gating is central). The step-16 plan was deferred, not refuted |
| 09-20 | "new video" via a state prior rather than via 13B's own unconditional branch | **chosen**; 17.0 had already shown unconditional 13B makes new (textured) video | DECISIONS 2026-09-20: the prior "earns its $0.22 by making the state space itself samplable, interpolable and editable" - that claim is item 25, now "Waiting" in ROADMAP |
| 09-20 -> 09-21 | flow over states -> VAE -> PCA + flow -> hex-local flow | **forced by measurements** each time (VAE 0.687 vs PCA 0.890, 18.24c; hexcov 0.876 vs 0.267, 21b), but the VAE step itself was an ML misreading corrected by research the owner demanded ("в чём проблема сделать VAE, это разве научно новая задача — нет!", `reports/2026-09-21_research_how_vaes_are_trained.md:3`) | DECISIONS 2026-09-20 VAE, 2026-09-21 PCA |
| 09-21 | video -> one still picture | **chosen** by the owner ("движение меня пока не парят", step-29 report §0), **supported** by evidence: 23 still not a scene; research §7.3 that noise2state is "unconditional video generation 64x64" on UCF101, a compute-heavy corner (MCVD FVD 1143; VDM 128 TPU-v4 x 700k steps) | `reports/2026-09-21_research_path_to_a_video_generator.md` §7.3 |
| 09-23 | train the next flow -> first build a judge (29.3 conversion check, 30 renderer, 29.4 metric) | **forced**: the owner cannot judge draws through the blurring LS map | ROADMAP "Current approved step", Done 29 |

Net: the project moved from a neuroscience question (what survives in each
layer; what does the model do on its own) to an ML generation problem
(unconditional generation of 721-hexal pictures in a re-encoded space). The
first move was a choice of deliverable; the second was forced by the model
having no dynamics of its own; the narrowing to one picture was a choice that
the evidence supports.

### 2.2 The principle - followed and not

AGENTS "Primary principle": "Choose the smallest experiment that shows the
thing working ... do not build an ensemble, a sweep or a validation suite
unless the step's question cannot be answered without it", and "A result
comes with one control, not five."

**Followed:**
- 17.0 measured the free unconditional baseline before any training
  (DECISIONS 2026-09-20; `2026-09-20_baseline17`).
- Option A settled "is 13B the bottleneck?" for $0 before retraining 13B
  (`2026-09-20_prior18_13b_diagnostic`, r 0.979).
- 18.21 closed the planned K = 8 run before it was paid for (commit 09-20
  12:56).
- 29a asked "is the picture in the state?" with a six-second least-squares
  fit before any training (`2026-09-21_prior23_pic`).
- Controls were almost always single and from the same run (e.g. N(0, I)
  through the same renderer in 29). The "one control" half of the principle
  held well.

**Not followed** - the excess was in arms and sub-steps, not in controls:
- **18.x**: about thirty sub-steps in ~15 active hours, including a seven-arm
  sweep (18.4), the sweep replotted in scaling-law coordinates (18.4f, commit
  09-20 06:41), an edge statistic re-run across all eleven arms (18.10), and a
  four-seed confirmation of the best arm (18.13, commit 09-20 10:02) - all
  scored mainly on a round trip that was already known not to measure scenes.
  The class-conditioning line took four sub-steps and an issue (ISS-0008) to
  conclude "the label carries nothing" usable.
- **22.0-22.5**: six measurements (cut space, cut directions, PCA ladder,
  channels, learned residual, block autoencoder) for one question ("is the
  object too big?"); outcome: "nothing moved". The question was the owner's
  reformulation (step-22 report §0).
- **22.7-22.8, the metric audit**: this is the closest thing to "five
  controls" - a floor for every judge. It was ordered by the owner's stop
  (`reports/2026-09-21_the_draw_distribution.md` §0) and paid once (ISS-0010:
  the draws were scored against the held-out split, the overshoot read 44 %
  where it is 20 %). But the gate it installed - per-coordinate kurtosis -
  was gameable within hours (26) and vacuous on the static target (29: 3.17 vs
  a Gaussian 2.98). Mixed.
- **Step 3** (price calibration) predates the principle and is the reason it
  was written.
- **Research sizing**: AGENTS "How to work" says a research request is "small
  (one or two scouts)"; the 09-20 request used four agents plus five
  sub-agents (2.03M tokens) and concluded that the learning rate was the one
  unjustified deviation - refuted the same day by 18.4b (lr 1e-4 is 6.2x
  worse). The two 09-21 requests (six scouts each) were the useful ones, and
  both were prompted by the owner's questions, not scheduled before a build.

---

## 3. Ordering mistakes an ML practitioner would name

### 3.1 Built late, but gates everything

1. **A scene judge.** The goal "a fresh draw gives a new, meaningful video"
   was set 09-20; a distribution metric on pretrained features (29.4) was
   added by the owner on 09-23. In between, the project judged generators by
   the round trip, then by structural proxies (flat fraction, coherence,
   kurtosis, radius, nearest), each with a failure mode (article part 2 §13:
   "summary statistics disagreed with the picture six times").
   *Known at the time:* the step-14 stripe (round trip 0.095 on a condition
   the generator ignored, 09-19 10:33) already showed the round trip does not
   read content; 17.0 (09-19 22:30) was judged "texture, not scenes" by eye.
   FID/KID and precision/recall are the standard tools for unconditional
   generators. One real obstacle: DECISIONS 2026-09-18 ("Semantic metrics
   (CLIP) mean nothing for a fly") was written for decoding and has not been
   revisited for generation; 29.4 quietly reverses it and should say so.
2. **A renderer that can show a scene for the static state (30).** Priced
   on 09-21 at $0.15-0.50 (commit 09-21 16:22) before the static flow was
   trained (17:29), and deferred by the owner ("отрисовщик не", ROADMAP 30).
   The flow was then judged through a least-squares map that blurs by
   construction, which is why nothing about it can be concluded. Training the
   judge after the generator is the inverse of the usual order.
3. **Looking at the substrate.** ISS-0015 (L3 a dot lattice, Tm9/T5a
   striped, Tm1 wrong sign) was found on 09-24 by drawing layers for a public
   figure; the check is a 40-second local script (`tools/cmp_layers_flyvis.py`).
   *Known at the time (09-18):* ISS-0005 said "every result read through Am,
   T1 or the Tm2/L2 feedback is suspect", listed Am->L3 at 14 vs 1.6 synapses
   per cell, and left open a decision "whether the Am pathways borrow FlyVis's
   own filters ... or stay absent"; T5 DSI was near zero on two of three
   members. The weak T5 was attributed to the missing CT1 (ISS-0002), which
   was a plausible reason. The picture rule existed (DECISIONS 2026-09-18),
   but pictures were of outputs (reconstructions), never of the model's own
   activity - and inversion through the model recovers the input whatever the
   intermediate maps look like, so the inversion ladder could not catch it.
   **A lead for 31, from reading the export (not tested):** in
   `data/ol/filters_R_w5wk50m500oc.json` every lamina and medulla type is
   `stride [1, 1]` except **Am: `stride [5, 4]`, 47 cells, density 0.045**,
   and Am (glutamate) projects to L3 over 19 offsets (total weight 29.4) and to
   R1-R6 and T1. A sparse periodic inhibitory source into L3 is exactly what
   would draw "a regular sublattice of isolated hyperpolarised cells". That
   would make ISS-0015 a consequence of the ISS-0005(a) decision left open on
   day 1 (Am/Lai is 47-49 fragments in MaleCNS, a third-party gap).
4. **Verifying the dynamic-to-static conversion (29.3).** The static flow was
   trained on the DC of moving-clip states on the strength of r 0.922 between
   a clip's DC and its held first frame's state - a number in "no artefact and
   no run record" (step-29 report §9), because the owner asked for a cheap
   conversion ("нам надо научиться превращать динамические слепки в статику —
   дёшево", §0). A full corpus brain pass costs **$0.15** (605 s on a T4,
   `2026-09-20_pairs18m_maps18`), and 23.2 had shown held frames give the
   true static state. Saving $0.15 created the verification debt that is now
   the first gate. Same class of false economy: every static picture shown to
   the owner went through a local PCA-2048 reconstruction (89.5 % variance)
   though the true states sit on the volume (ROADMAP 30).
5. **The prior-art check for the generative line.** The facts that framed
   the problem - noise2state is unconditional UCF101 64x64 generation, a
   corner where published systems used 128 TPU-v4 chips; the state is 13 %
   bigger than the clip (92,288 vs 81,920), so a prior over states is not a
   compression of the video task; conditioning is the only route with large
   measured gains - arrived on 09-21 (`reports/2026-09-21_research_path_to_a_video_generator.md`
   §7.1, §7.3), after ~$7.6 of step 18. AGENTS asks for this check "before a
   large step is built".
6. **The missing easy baseline.** No one trained the same flow on the
   pictures themselves (721 hexals of the corpus frames) at the same size and
   budget. That single run separates "the state space is hard" from "unconditional
   generation at N ~ 13.5k, 721 grey pixels, 3.7M parameters is hard", and it
   is the ceiling the static branch should be read against. For the static
   branch it is especially apt: the static state is a near-linear re-encoding
   of the mean frame (LS r 0.941), so the state flow and a pixel flow are
   close to the same problem in two coordinate systems.

### 3.2 Built early, not needed

- Step-3 training infrastructure (`deploy/modal/train_app.py` smoke, packed,
  batch, benchmark, the Codex optimisation bench): $0.90 and half a day;
  training is paused (ROADMAP Not started, 3').
- The ensemble decodability map, lag sweeps, `map_local.py`, `ensemble.py`,
  Modal decode app: parked rigour (ROADMAP Not started).
- 13A ($1.61): superseded by 13B within hours; its "equal to 13B" claim was
  on different scales (ISS-0014).
- Diagnostic sprawl: `flydream/generate/` holds 47 modules, 10,459 lines,
  most named after a sub-step (`assigned18`, `walk18`, `reach18`, `why18`,
  `trunc18`, ...); the live path needs about five (`invert`, `gen13b`,
  `pairs13`/`stimuli`, `prior17`, `hexflow23`). Three different `nearest`
  functions (ISS-0011) and two round-trip normalisations (ISS-0014) are the
  predictable cost.
- Records overhead: 102 of 280 commits touch no code; ROADMAP edited 93
  times in seven days and cut from 66 KB to 10 KB on 09-21 (commit 05:10).
  The records did pay at publication (number checks caught ISS-0013/0014 and
  retracted claims), so this is overhead, not waste.

### 3.3 A leaner path (hindsight sketch; an estimate, not a measurement)

1. Day 1: data, model zero, transplant fix, **layer pictures against FlyVis**
   (would have surfaced ISS-0015 and forced the Am decision of ISS-0005a),
   inversion ladder. Skip step-3 pricing and the ensemble map.
2. Day 2: 13B; step 11 (the evidence that the model has no dynamics of its
   own); step 14 (reachability). Skip 13A or keep it as a one-hour baseline.
3. Day 3: goal set -> one or two scouts on small-data unconditional
   generation -> **the judge first** (KID and precision/recall on pretrained
   features of upsampled hex rasters, nearest neighbour, each with a floor)
   and **a pixel-space generator at the same scale** -> corpus (18.0) ->
   free hexcov -> a hex-local flow directly on the lattice state (the 23
   design, which the 09-21 research route E and the free 21b measurement
   pointed to).
4. Day 4: statics on true held-frame states (one $0.15 brain pass), a
   generative renderer checked on real states, then the static flow.

Kept runs at their logged prices: reading ~$1.5-2 (inversion $0.22, step 11
$0.15, 13B $0.42, pairs), corpus ~$0.2-0.4, 23 $0.51, static flow $0.42,
renderer ~$0.25 (ROADMAP 30 estimate), pixel baseline ~$0.4 (by analogy with
29), held-frame pass $0.15 -> roughly **$3.5-4.5 and 3-4 days** against
$12.62 and ~4.5 days. It ends where the project stands now plus a judge -
not at a solved generator; nothing in hindsight shows scenes were reachable
faster.

### 3.4 Who drove which detour (for fairness)

Agent-driven: step 3 pricing (paper-shaped contract), the round-trip-gated
sweeps of 18.4-18.13, the VAE argument ("a KL term is precisely a term on
that quantity", DECISIONS 2026-09-20), "T4/T5 are motion detectors, so no
static picture" (step-29 report §2, "Довод был мой"), 23.1's invalid test,
ISS-0012's display normalisation. Owner-driven: fixed noise-state coupling
18.20 ("проблема в сидах, ты их сделал неправильно"; the agent recorded in
advance that it would not buy quality, seed-problem report §4), the step-22
reformulation, the cheap conversion for 29, deferring the renderer. Owner's
corrections that paid: the course change of 09-18, the acceptance criterion
by eye (09-20), the demand for VAE prior art (09-21), the metric stop (22.7),
ISS-0012, rejecting 23.1, and the 09-23 order that puts the judge first.

---

## 4. Against the idea

### 4.1 Concept vs what exists

| Concept element (`what_does_a_fly_dream_of.md`) | Status |
|---|---|
| Stage 1-2: stimulus -> connectome-constrained dynamics; record layers | Done on the right optic lobe of MaleCNS with FlyVis parameters transplanted (steps 1-2); OFF pathway defective (ISS-0015); no central brain |
| Stage 3-4: reconstruct from each layer; map where the image is lost | Inversion recovers the input from every stage at r >= 0.92, i.e. in this model no layer up to T4/T5 loses the image; the decoder stage curve was an artefact (ISS-0004); the "where does it stop being an image" question needs layers beyond T4/T5 (LPi, VS/HS, VPNs), which do not exist in the model |
| Stage 5: remove input, decode internally generated activity | Out of reach with this model: it decays to rest (step 11); no recurrence past T4/T5, no central brain (step-16 design §2); the owner's revised plan names the main risk as arbitrary tau/bias/gain producing oscillations that 13B renders prettily (`docs/ideas/step16_dream_source_revised.md` §7) |
| "The visual stimulus most compatible with an internally generated state" | Replaced by "the picture most compatible with a state drawn from a learned prior over stimulus-driven states" - AGENTS: "A state sampled from a learned prior is not the brain's own activity" |
| Owner's notes: neural state as the generative interface, later fed by deeper states (`docs/ideas/brain_state_prior_new_video_generation.md` §§18-20; `full_project_architecture_brain_to_video.md`) | The interface exists for video (a clip's own seed returns the clip at the 0.884 ceiling, step 23); a fresh draw is not a scene. The same note concedes the ML point: "Если целью было бы просто сделать хороший video generator, использование MaleCNS не даёт очевидного преимущества" (§14) |

### 4.2 Reachable vs structurally out of reach

**Reachable with the current approach**
- A fresh static draw with scene-like statistics, rendered sharply: the
  static flow already matches real states on two structural judges through a
  blurring map; a generative renderer (30) is a known technique at a priced
  $0.15-0.50.
- Editing and interpolation in state space as a product (ROADMAP 25): the
  machinery exists (`seed19.py --controls`), and since 23 interpolation is
  no longer a double exposure (rotation 72 vs 19 degrees). Only a fair
  novelty test is missing.
- Brain-compatible new video in the weak sense ("generated without a source
  clip"): already true since 17.0 / 14.2.

**Uncertain**
- "Meaningful" scenes from an unconditional draw. The target is a 721-hexal
  grey picture (~27x27), averaged over a 0.8 s window; even real states are
  hard to recognise through the current map (owner, step-29 report §8), and
  the figure of real held-out pictures had to be "chosen by eye as
  recognisable" (`docs/articles/part2_en.md`, Fig. 9 caption). N = 13,555
  training clips sits at the low edge of the published memorisation ->
  generalisation transitions (K ~ 8k-16k, research §1). Expect layout-level
  plausibility; recognisable objects are not supported by any measurement in
  the project.

**Structurally out of reach with this approach**
- Spontaneous or "dream" states: needs new types with unknown parameters and
  a central brain (ROADMAP Not started 6, 16).
- Content beyond the fly eye's resolution, colour (ROADMAP C, no parameters)
  and the window.
- A connectome-specific contribution to the static generator: with the
  state a near-linear re-encoding of the picture (r 0.941), the MaleCNS
  wiring acts as a fixed change of coordinates; the lattice geometry the
  hex-local flow exploits is the eye's, shared with FlyVis.

### 4.3 Shortest credible path from here to "a fresh draw gives a new, meaningful scene"

1. **Fix the static data once** (~$0.15): one brain pass over the corpus
   with each clip's frame held still gives true static states, which makes
   29.3 a check of an alternative rather than a gate (ROADMAP 29.3 already
   names this fallback). If 31's cause-finding (local, free) says T4 inputs
   change, do the repair first so this pass is done once.
2. **Pixel-space control** (~$0.4): the same `hexflow23` flow on the
   721-hexal pictures themselves. It is the ceiling of unconditional
   "scene-ness" at this N and size.
3. **Generative renderer (30)** (~$0.25): checked on real held-out states
   first, with its own control - a no-flow N(0, I) state and a shuffled state
   through it - so scene-ness supplied by the renderer's prior (as 13B's
   unconditional branch supplied texture in 17.0) is not credited to the
   state flow.
4. **Scene metric (29.4)**: KID and precision/recall on pretrained features
   (KID rather than FID at 64-256 draws), nearest-neighbour novelty, all
   beside real-state renders, the pixel-space control and the no-flow control.
5. **Then one lever, chosen by what 2-4 show**: if the pixel flow makes
   scenes and the state flow does not, the problem is the state space (29.2
   slices, 28 capacity); if neither does, it is data/scale, and the routes
   with evidence are conditioning (24; research route F: the only route with
   large measured gains, cost = memorisation to be measured) or a larger
   corpus (27), or accept the weaker claim with an external scene source
   (route G).

Rough total: ~$1-1.5 on Modal and two to three days of work to a verdict
that the owner can read by eye and by number.

---

## 5. Does the resumed order converge?

Order on resume (ROADMAP "Current approved step"): **29.3 -> 30 -> 29.4 ->
29.2**; item **31** is "recorded ... not yet ordered against the static
branch" (ROADMAP 31); the brief places it last.

**What it converges on.** A judge the owner can use (renderer + metric) and
a verified static target. That is the right next thing. It does **not** by
itself converge on a better generator: the only generator lever in the order
is 29.2 (fourth); capacity (28), corpus (27) and conditioning (24) come after
it. Nothing in the order would reveal early that unconditional generation at
this scale plateaus at "scene-like statistics" - the pixel-space control would.

**Risks**
1. **31 last invalidates 29.x.** Every state comes from model zero. The
   static object is T4a+T4b - the ON pathway, which ISS-0015 calls clean - but
   T4a receives Mi9 (glutamate, total 19.5) and Mi9 receives L3 (total 52.6)
   in the export, so the L3 defect has a path into T4. Recomputed here from
   the saved cache `data/figures/act_s3_malecns_flyvis.npz` (Sintel clip 3,
   member 000, the `cmp_layers_flyvis.py` metric at frames 10/20/30): T4a
   roughness 0.38/0.40/0.39 on model zero vs 0.34/0.33/0.33 on FlyVis;
   T5a 0.43/0.46/0.46 vs 0.24/0.18/0.18; Tm9 0.54/0.58/0.58 vs 0.31/0.38/0.37.
   The same metric barely separates L3 (0.46 vs 0.43 at frame 20) although
   the picture shows the dot lattice, so it is a weak detector: T4 is mildly
   affected by this number and not proven clean. If a repair changes T4
   statistics, 29.3's check, 30's renderer data, 29.4's reference numbers and
   29.2's flow are rebuilt on re-simulated states. In money that is small
   (brain pass $0.15, 13B retrain $0.22-0.42 if the video line is kept,
   static flow $0.42, renderer ~$0.25: about $1-1.5); in work about a day,
   provided every 29.x step is one committed command.
2. **31 may not be a "fix".** If the cause is Am (a 47-49-fragment
   reconstruction gap, ISS-0005a), the options are to remove Am or borrow
   FlyVis's filters as a named exception - a change to what "on MaleCNS"
   means, and an owner decision.
3. **30 is the critical path and has no code** and no chosen objective
   (ROADMAP 30). A strong generative renderer can hallucinate plausible
   detail from any state; without the renderer-only control (4.3 step 3) a
   good-looking draw proves nothing about the flow.
4. **29.2's "eightfold corpus" overstates the gain.** Eight slices of one
   0.8 s window are correlated samples of the same scene; the effective N
   grows much less than 8x. Slices of a moving clip are also not held-frame
   states, so they need their own 29.3-style check (ROADMAP 29.3 covers
   "slice k").
5. **29.4 needs a pretrained network on 721 grey hexals** - adaptation is
   part of the item (ROADMAP 29.4), and small draw counts make FID unstable.

**Alternative orders and cuts (options with cost, not decisions)**

| Option | What changes | Cost | Buys |
|---|---|---|---|
| A. As planned: 29.3 -> 30 -> 29.4 -> 29.2 -> 31 | - | 29.3 free; 30 $0.15-0.50 + code; 29.2 ~$0.4-0.5; 31 unknown + ~$1-1.5 redo | fastest route to a judgeable picture; risk of one rebuild |
| B. 31a first (cause only, local, free), then decide | if the cause touches T4 inputs, repair before building 30's data | a few hours locally; the Am lead above is a one-edit test (drop or replace Am -> *, rerun `cmp_layers_flyvis.py`) | builds the static corpus once, on the final brain |
| C. Replace 29.3 with true static states | one held-frame brain pass over the corpus; train 30 and the flow on it | ~$0.15 | removes the conversion question instead of measuring it |
| D. Add the pixel-space control before or with 30 | one `hexflow23` run on 721-hexal pictures | ~$0.4 | the ceiling every 29.4 number is read against |
| E. Cut 29.4 to KID + precision/recall + nearest | skip FID | $0 | a metric that is stable at 64-256 draws |
| F. Put 25 (state space as interface) beside 30 | a fair novelty test for interpolation/editing | local, $0 | the claim 17 was accepted on, deliverable with today's video flow |
| G. Decouple 31 from the generator | keep the generator's claims "about this encoder" (ROADMAP 31 already says so); do 31 for the biology and the public comparison | - | no rebuild on the generator line |

---

## Top lessons (ranked)

1. **Build the judge before the generator.** A scene metric with a floor and
   a renderer that can show a scene came last (29.4 on 09-23, 30 still
   unbuilt), after ~$9.3 of generator training judged by proxies that
   disagreed with the picture six times (article part 2 §13).
2. **Run the easy baseline at the same scale.** The state is bigger than the
   clip (research 09-21 §7.1); a flow on the pictures themselves was never
   trained, so no result on the state space can be read as "hard because of
   the state space" or "hard because of the task".
3. **Scout the prior art before the first paid arm, not after thirty.** The
   unconditional-UCF101 cost, the VAE-latent practice and exposure bias
   reached the project on 09-21, after ~$7.6 of step 18, and only because the
   owner asked.
4. **Look at the substrate before building on it, and take the open
   decisions it raises.** A 40-second layer picture found ISS-0015 on 09-24;
   ISS-0005 had flagged Am -> L3 on 09-18 and left the Am decision open; the
   export's only strided type is Am.
5. **Retire an uninformative metric at once.** The round trip was shown not
   to read content on 09-19 (step 14) and "texture passes it perfectly" on
   09-20 (18.3), yet ~$4.3 of sweeps pushed it to and below its floor.
6. **Bound a diagnostic line by the decision it changes.** 18.x (~30
   sub-steps), 22.0-22.5 (six measurements, "nothing moved") and the
   seed-geometry line ($2.69, "two different diseases") answered questions
   that did not change the next action.
7. **No false economy on data plumbing.** The static branch rests on numbers
   from a throwaway script, on DC-of-moving-clip states chosen to save a $0.15
   brain pass, and on pictures seen through a local PCA reconstruction;
   together they made the one result of the static branch unjudgeable.

---

## Gaps

- Modal spend is the sum of `cost` fields in `reports/runs.jsonl` ($12.62).
  Runs without a record (e.g. the 09-18 decode pilot on Modal, commit 22:18;
  local CPU time) are not in it; the figure is a lower bound. Subagent tokens
  are known only for five research requests (~4.5M); the 09-18 audit, the
  article number checks and the delegated step 26 are not costed.
- Pace uses commit timestamps; work before the first commit on 09-18 and any
  uncommitted work are invisible. Report and ROADMAP dates can differ from
  commit dates by a day.
- The layer-roughness numbers are recomputed from one saved cache (one
  Sintel clip, member 000, three frames); nothing was re-simulated. The Am
  hypothesis for ISS-0015 comes from reading the export JSON and is untested;
  the L3 -> Mi9 -> T4a path is read from the same file's edge totals, not
  from a simulation.
- Not verified: how many distinct scenes the 15,514 clips contain (UCF101
  clips come from shared source videos; the effective N may be smaller than
  12,411 source files suggest).
- The "leaner path" and the "shortest path" costs are estimates from logged
  prices of comparable runs (29 for the pixel control, pairs18m_maps18 for a
  brain pass, the ROADMAP 30 estimate for the renderer), not measurements.
- Long step-18 reports (18.4, 18.5 at 1,037 lines) and the research notes
  were skimmed, not read in full; figures were not viewed.
- Attribution of detours to the owner or the agent relies on quotes in
  reports and DECISIONS; the conversations themselves were not available.
