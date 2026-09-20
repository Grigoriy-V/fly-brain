# Roadmap

**Updated:** 2026-09-21.

This file owns **direction, order and authorization** — nothing else. No
results, no numbers, no reasoning: those live in `reports/`, durable choices in
`DECISIONS.md`, defects in `ISSUES.md`. Rules: `AGENTS.md`. Delivery:
`docs/ARTEFACTS.md`. System: `docs/PROJECT_MAP.md`. Operations:
`docs/OPERATIONS_MAP.md`.

---

## Where the project stands

The chain `video → frozen brain → T4/T5 state → generator → video → frozen
brain` works end to end and is measured. What is missing is a **generative
source of states**: the project can render any valid state and cannot yet
invent one.

Current track, decided by the human 2026-09-20: **new video without a source
clip**, through generated brain states.

```text
ε ~ N(0, I) → noise2state → state → 13B → video → frozen brain → round trip
```

| part | what it does | state |
|---|---|---|
| frozen brain | video → T4/T5 state | not trained by us, works |
| 13B | state → video | trained, frozen — not a bottleneck |
| first stage | state ↔ latent | linear PCA-2048, measured, not a bottleneck |
| **noise2state** | **ε → latent** | **the live problem** |

---

## Current task list

Nothing below is started without the human's word. Each step is priced when it
is proposed, against what that step is for.

1. **Grow the state corpus.** The corpus holds about 1.25 variants per source
   file where the build supports six; more states is the cheaper of the two
   levers on noise2state. The first stage is refitted on whatever the corpus
   becomes.
2. **Retrain the flow at the same capacity on the larger corpus.** This is what
   separates "too little data" from "too small a network", before capacity is
   bought.
3. **Then capacity for the flow**, sized by what 1 and 2 show.
4. **Re-run the seed test** (`flydream/generate/seed19.py`) after each: a drawn
   ε through the whole chain, judged as a clip against the raw corpus video,
   beside the distance to the nearest training video.

**The acceptance criterion is the human's and overrides every metric**
(2026-09-20, `reports/2026-09-20_the_seed_problem.md` § 5): a result counts
only as a **clip against the raw corpus video** — never by the gate, the
sparseness or r. Blur is accepted for a first version if scenes appear and a
fresh draw gives a new, meaningful video.

**Open questions this track carries**

- Whether a flow over the latent reaches scene states from a fresh draw at a
  budget this project is willing to spend.
- Whether the first stage stays linear once noise2state works.
- Whether conditioning is needed to get structure, or unconditional sampling
  suffices.

---

## Current stage: 19, a linear first stage and a flow over its latent

Report: `reports/2026-09-21_step19_linear_first_stage.md`. Why two stages and
why the latent is not asked to be Gaussian:
`reports/2026-09-21_research_how_vaes_are_trained.md`. Decision:
`DECISIONS.md` 2026-09-21.

- **19.0 PCA-2048 on the card** — done. `flydream/generate/pca19.py`, run
  `2026-09-21_prior19_pca2048`.
- **19.1 first-stage acceptance** — done, passes. `pcaval19.py`, run
  `2026-09-21_prior19_pcaval`, figure `2026-09-21_malecns_pca19`.
- **19.2 the flow over the latent** — done, two runs.
  `train17(latent_tokens=…)`, runs `2026-09-21_prior19_flow2048` and
  `…_flow2048_b256`.
- **19.3 the seed test** — done. `seed19.py`, run `2026-09-21_prior19_seed`,
  figures `2026-09-21_malecns_seed19`, `…_seed19b`.
- **19.4 where the sampler leaves the truth** — done. `tprofile19.py`.
- **19.5 controls without the flow, and interpolation between two seeds** —
  done. `seed19.py --controls`, figure `2026-09-21_malecns_interp19`.

**The stage in one line:** a seed now produces its clip through the whole
chain, a fresh draw still does not produce a scene, and the reason is measured.
Numbers in the report.

**Fallback ladder**, if the task list above does not close it — nothing
started, each priced when proposed: a learned residual on top of the fixed PCA
path (DC-AE's recipe); a smaller latent traded for a fuller space; ex-post
density over the latent (Dai & Wipf, Ghosh et al.); a learned first stage with
a weak KL.

---

## The seed problem — what is closed

`reports/2026-09-20_the_seed_problem.md` holds the problem in the human's
words, the evidence, and the list of what is eliminated and must not be
reopened without new evidence: classes as a condition, guidance, noise
scaling, radius normalisation, the shared preimage direction, K = 32, a fixed
hand-assigned coupling, DCT truncation below K = 16, and a latent forced to
N(0, I) by its own KL.

---

## Done

One line each; the report holds everything.

- **Research, the landscape and the plan** — `reports/Коннектом мухи и план проекта.md`,
  `reports/Обучение коннектомных сетей и ускорение.md`.
- **1, data** — MaleCNS core files, the FlyVis ensemble, the right optic lobe
  export. `reports/2026-09-18_step1_data.md`.
- **2, model zero** — FlyVis dynamics on the MaleCNS export, parameters
  transplanted by type. `reports/2026-09-18_step2_model_zero.md`.
- **3, training priced** — `reports/2026-09-18_step3_training_options.md`,
  `reports/2026-09-18_training_optimization_bench.md`.
- **4, the decoder ladder** — `reports/2026-09-18_step4_decoder_stack.md`.
- **8-9, the generator by encoder inversion and its window** —
  `reports/2026-09-19_step9_window_margin.md`.
- **11, dreams-lite** — inversion of states no clip caused.
  `reports/2026-09-19_step11_dreams_lite.md`.
- **12, manipulated states** — `reports/2026-09-19_step12_manipulated_states.md`.
- **13A, amortised inversion** — `reports/2026-09-19_step13a_amortised_inversion.md`.
- **13B, the conditional flow generator** — the decoder this whole track uses.
  `reports/2026-09-19_step13b_design.md`,
  `reports/2026-09-20_step13b_generative_decoder.md`.
- **14, what can be prompted** — random, edited and composed states; the first
  video with no source clip.
  `reports/2026-09-20_step14_controllable_generator.md`.
- **17, the brain-state prior** — the unconditional baseline, the flow over
  states, the DCT representation, the noise inversion.
  `reports/2026-09-20_step17_state_prior.md`,
  `…_step17_0_unconditional_baseline.md`, `…_step17_1b_scene_dct_prior.md`,
  `…_step17_3b_noise_inversion.md`.
- **18, the corpus of ordinary video and every arm over it** —
  `reports/2026-09-20_step18_corpus_of_ordinary_video.md`,
  `…_step18_3_prior_on_the_corpus.md`, `…_step18_4_throughput_and_learning_rate.md`,
  `…_step18_5_width_and_the_representation_floor.md`, and the seed problem
  report above.
- **Research, how VAEs are actually trained** — why the first stage may be
  linear and why nobody asks a latent to be Gaussian.
  `reports/2026-09-21_research_how_vaes_are_trained.md`.

---

## Deferred

Nothing here is refuted; each waits for the human's word and is priced when it
is taken up.

- **3', fine-tuning MaleCNS from the transplanted weights** (the human,
  2026-09-19: "не сейчас"). Commands:
  `reports/2026-09-18_step3_training_options.md` § 5б.
- **16, the dream source inside the model** — spontaneous drive, the optic-lobe
  loops the export dropped, central-brain drive. Design
  `reports/2026-09-20_step16_dream_source_design.md`; the human's revision
  `docs/ideas/step16_dream_source_revised.md` takes precedence where they
  differ. Not fully formed, not fully discussed (the human, 2026-09-20).
- **10, a dense MaleCNS export** (every type on all 721 columns) and **a
  64×64 raster of the hexals** — cosmetic, local.
- **5, both eyes, every column, the missing biophysics** — for HDRI / 360°
  input, not for clips.
- **6, the central brain and the state knobs** — new export, long training,
  validation.
- **7, dreams, the full protocol** — the rigorous form of item 11.
- **15, HDRI / 360° panoramas as a stimulus source** (the human, 2026-09-19):
  unlimited clips with exact optic flow from rotations.
- **C, colour as an ML task** (the human, 2026-09-19): MaleCNS has the types,
  there are no parameters to transplant; until then colour in an output is a
  display overlay and says so.
- **Deeper than T4/T5 with generic dynamics** — `docs/ideas/malecns_shiu_lif_baseline_idea.md`;
  collides with `DECISIONS.md` 2026-09-18 on the model class, so that decision
  is revisited first.
- **Multi-level conditioning of the generator** — early, deep and central
  states together as the condition, after a deeper model exists.
- **Paper-grade rigour** — ensembles, several splits, per-type nulls, subset
  curves, validation against the 26 physiology studies. Code exists and is
  tested; picked up only if the human asks.
- **The connectome as a computational substrate** — parked 2026-09-18; if
  revisited, a degree-preserving shuffled connectome must do measurably worse.
- **One-liners, recorded only:** the three-connectome comparison; a second
  training objective; the embodied loop in MuJoCo; inversion on real
  recordings; photoreceptor columns from lamina cartridges; the left lobe
  export; the fitted-dynamics parameter source; ablations on model zero
  (ISS-0002); per-target filter normalisation (measured unstable, `"src"`
  stands).

---

## How this file is kept

- **Only approved work.** An unapproved conclusion is a draft: it belongs in
  `reports/`, or in `docs/ideas/` if it is the human's own note.
- **Direction and order, not results.** Every number, comparison and price
  lives in the report this file links to.
- **One line per done item.** How it was reached is not repeated here.
- **Everything deferred sits in one group**, whatever the reason it waits.
- **Price is discussed per step, against what that step is for.** This file
  carries no budget and no cap.
- **Short beats complete.** If this file needs a table of contents, cut it.
