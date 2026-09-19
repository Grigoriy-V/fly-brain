# Flow matching / rectified flow training recipes: what is known to matter at small scale

Scope: linear-interpolant flow matching (rectified flow / stochastic-interpolant family),
velocity prediction, MSE loss — the project's exact setup. Sources are read directly (arXiv
HTML/PDF and the official GitHub repos' code, via `gh api`), not summarized from memory. Every
reference paper trains 100-1000x larger than this project (DiT/SiT: 33M-675M params, 1.2M-14M+
images seen; SD3: up to 8B params, billions of images), on natural images with class or text
conditioning; this project trains 1.38M params on 13,555 examples, unconditional. Nothing below
was measured at this project's scale — treat every number as a prior from a different regime.

## 0. A sign convention to get right before reusing any parameter below

This project: `x_t = (1-t)*eps + t*x1`, so **t=0 is noise, t=1 is data**. This matches Liu et
al.'s rectified flow, Lipman et al.'s OT path, and the official SiT code (`transport/path.py`,
class `ICPlan`: `compute_alpha_t(t) = t` (data coefficient), `compute_sigma_t(t) = 1-t` (noise
coefficient), i.e. `x_t = t*x1 + (1-t)*x0` — identical to this project's formula).
**SD3 (Esser et al. 2024) uses the opposite convention**: `z_t = (1-t)*x0 + t*eps` with x0=data,
eps=noise, i.e. **t=0 is data, t=1 is noise** (confirmed by direct read of arXiv:2403.03206, SD3 §3.1).
Any *asymmetric* SD3 timestep-density parameter (a nonzero logit-normal location, or a signed
mode-sampling scale) must be mirrored (negate the location, or read `t` as `1-t`) before reuse
here. The symmetric case, logit-normal(mean=0), is unaffected by the flip and transfers directly.

## 1. Timestep sampling density

SD3's §3.1/Table 1 ablation (rectified-flow objective, ImageNet+CC12M eval, ranked by
non-dominated sorting across FID/CLIP at multiple step counts;
[arXiv:2403.03206](https://arxiv.org/abs/2403.03206)):

| Variant | Formula | Rank (all) | Rank (5 steps) | Rank (50 steps) |
|---|---|---|---|---|
| rf (uniform) | t ~ U[0,1] | 5.67 | 6.50 | 5.75 |
| **rf/lognorm(0.00, 1.00)** | logit(t) ~ N(0, 1-squared) | **1.54** | 1.25 | 1.50 |
| rf/lognorm(1.00, 0.60) | logit(t) ~ N(1, 0.6-squared) | 2.08 | 3.50 | 2.00 |
| rf/lognorm(0.50, 0.60) | logit(t) ~ N(0.5, 0.6-squared) | 2.71 | 8.50 | 1.00 |
| rf/mode(1.29) | f(u)=1-u-s(cos^2(pi*u/2)-1+u) | 2.75 | 3.25 | 3.00 |
| eps/linear | (noise-prediction baseline) | 2.88 | 4.25 | 2.75 |
| rf/mode(1.75) | s=1.75 | 3.33 | 2.75 | 2.75 |
| rf/cosmap | t=1-1/(tan(pi*u/2)+1) | 4.13 | 3.75 | 4.00 |
| edm(0.00, 0.60) | EDM-style log-normal sigma | 5.63 | 13.25 | 3.25 |

Logit-normal density: `pi(t;m,s) ~ 1/(s*t*(1-t)) * exp(-(logit(t)-m)^2/2s^2)`, `logit(t) =
log(t/(1-t))`. At 25 steps on ImageNet, FID: uniform 49.70, lognorm(0,1) 45.78, mode(1.75)
**44.39** (best FID, not best overall rank); on CC12M, uniform 94.90 vs lognorm(0,1) 89.91
(Table 2). SD3 picks `rf/lognorm(0.00,1.00)` as the production default: "consistently achieves a
good rank" (§5.1.1). Reasoning (§3.1, their convention t=0=data/t=1=noise): "for t=0, the optimal
prediction is the mean of p1, and for t=1 the optimal prediction is the mean of p0" — each
endpoint degenerates to an unconditional marginal mean (trivial), so the useful signal
concentrates mid-interval, which logit-normal oversamples relative to uniform.

The official SiT/DiT training code implements **none** of this: `transport/transport.py`,
`Transport.sample()`, is `t = th.rand((B,)) * (t1-t0) + t0` — plain uniform, regardless of the
`--loss-weight` CLI flag (github.com/willisma/SiT). Logit-normal sampling is a later (SD3-era)
addition on top of the SiT/DiT-style loop, not inherited from it by default — worth knowing if
copying a "DiT-style" reference implementation wholesale. This project already uses logit-normal,
the best-ranked SD3 family member; no ablation at this project's token/model size was found —
extrapolated from image-generation scale.

## 2. Loss weighting over t, and velocity vs. noise vs. score prediction

SD3 (their own §2) shows resampling t from density pi(t) is mathematically equivalent to
uniform-t sampling with an explicit per-sample loss weight `w_t = (t/(1-t)) * pi(t)`: "the loss
changes based on the timestep distribution, not sample-level weighting" — the logit-normal choice
above *is* the weighting mechanism for this family, not a separate knob. SiT's ablation confirms
prediction-target choice matters too ([arXiv:2401.08740](https://arxiv.org/abs/2401.08740),
SiT-B, 400K steps, 250-step Euler ODE, ImageNet 256x256 FID, Table 3-4):

| Interpolant | Prediction | Loss | FID |
|---|---|---|---|
| SBDM-VP (diffusion) | score | unweighted L_s | 43.6 |
| SBDM-VP | score | weighted L_s*lambda | 39.1 |
| SBDM-VP | **velocity** | L_v (MSE) | 39.8 |
| Linear | velocity | L_v (MSE) | **34.8** |
| GVP | velocity | L_v (MSE) | 34.6 |

Unweighted score prediction is clearly worst (43.6); weighted score and plain velocity are close
(39.1 vs 39.8) at the same (diffusion) path — prediction-target choice matters less than path
choice. Switching the *path* from a VP diffusion path to Linear (this project's choice) is the
larger effect (39.8 to 34.8); GVP adds another small increment (34.8 to 34.6) not obviously worth
its extra complexity at this scale. The official code confirms *why* velocity showed no weighted
variant in Table 3: `transport/transport.py`, `Transport.training_losses()` —
`if self.model_type == ModelType.VELOCITY: terms['loss'] = mean_flat((model_output - ut) ** 2)`,
an unconditional unweighted MSE; the `WeightType.{VELOCITY,LIKELIHOOD}` branch is only reachable
for `NOISE`/`SCORE` targets. **For velocity prediction with MSE (this project's setup), the
reference implementation has no t-dependent loss weight — the only lever is the t-sampling
density from the section above**, consistent with SD3's weighting-equivalence math.

## 3. EMA rate vs. schedule length

DiT and SiT both hard-code a fixed per-step decay, independent of batch size or run length:
`update_ema(..., decay=0.9999)` (github.com/facebookresearch/DiT and github.com/willisma/SiT,
both `train.py`). EDM instead derives the per-step decay from an image-count half-life and the
batch size each step (`training/training_loop.py:143-146`): `ema_beta = 0.5 **
(batch_size/ema_halflife_nimg)`, with a 5%-of-images-seen ramp-up early on
(`ema_rampup_ratio=0.05`, `training_loop.py:38`; disabled only for `--transfer` resumes,
`train.py:169`). Converting decay to a half-life fraction (ln(0.5)/ln(decay)) makes the
schedule-length interaction concrete:

| Config | Decay / half-life | Run length | Half-life as % of run |
|---|---|---|---|
| DiT/SiT default (decay 0.9999) | approx 6,931 steps | 400K steps (SiT-B ablation) | 1.7% |
| DiT/SiT default, same decay | approx 6,931 steps | 7M steps (SiT-XL) | 0.1% |
| DiT/SiT default, same decay, hypothetically | approx 6,931 steps | 20K steps (this project's length) | **34.7%** |
| EDM default (`--ema=0.5` Mimg) | 500 kimg | 200,000 kimg (default duration) | 0.25% |
| EDM ImageNet-64 config (`--ema=50`) | 50,000 kimg | 2,500,000 kimg | 2.0% |
| **This project (decay 0.999)** | approx 693 steps | 20K steps | **3.5%** |

Two things follow. First, verbatim-copying the DiT/SiT `decay=0.9999` onto a 20K-step schedule
would give a half-life covering 34.7% of the run — much slower-adapting than either reference
recipe ever actually used (their own ratios are 0.1-1.7%); the project's actual `decay=0.999`
(3.5%) is a reasonable correction, landing close to the EDM ImageNet-64 ratio (2.0%). Second,
there is no fixed cross-run ratio even within one paper's own repo (EDM: 0.25% default vs. 2.0%
for its longest run) — half-life is retuned per run length, not derived from a formula. EDM2
([arXiv:2312.02696](https://arxiv.org/abs/2312.02696), CVPR 2024) makes this explicit: FID vs.
EMA half-life has an optimum that is "quite sharp" and specific to architecture/training
time/guidance, so they introduce **post-hoc EMA**: store a couple of EMA profiles during
training, reconstruct *any* half-life afterward from them instead of committing upfront
(abstract; supporting quote in [NVIDIA's summary](https://developer.nvidia.com/blog/rethinking-how-to-train-diffusion-models/)).
No source gives a rule for picking the ratio a priori — only that it should be swept or
reconstructed, never assumed.

## 4. Learning rate and batch size

| Source | Batch | LR | Schedule |
|---|---|---|---|
| DiT / SiT (`train.py`, both repos) | 256 | 1e-4 | constant, no warmup, AdamW wd=0, beta(0.9,0.999) |
| EDM CIFAR-10 (default `train.py` flags) | 512 | 1e-3 (`--lr` default `10e-4`) | 10,000-kimg linear rampup, then constant, Adam |
| EDM FFHQ/AFHQv2 | 256 | 2e-4 | same rampup shape |
| EDM ImageNet-64 | 4096 | 1e-4 | same rampup shape (`README.md` training-config table) |
| Lipman et al. FM, ImageNet-64 (Table 3, [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)) | 2048 | 1e-4 | Adam beta(0.9,0.999), warmup + polynomial decay, fp16 |
| **This project** | 32 | 3e-4 | 100-step warmup + cosine, AdamW fused, beta(0.9,0.99), wd 0.01, clip 1.0 |

None of these papers ablate batch size and LR jointly — each fixes one operating point per
dataset/budget, and the pairs do not move monotonically (EDM alone: 512 to 1e-3, 256 to 2e-4,
4096 to 1e-4), so there is no scaling curve to read off this literature directly, only the
general Adam theory as a soft prior: McCandlish et al. 2018 and Malladi et al. 2022 give a power
law LR(B) proportional to B^alpha, alpha in [0.5,1] — between "hold LR fixed" and "scale
linearly," already covered in this project's
`research_notes/Обучение коннектомных сетей и ускорение/recipe_levers.md`, unverified for this
architecture family. Applying it as a sanity check: scaling DiT/SiT's 1e-4@256 down to batch 32
(divide by 8) predicts 1.25e-5 (linear) to 3.5e-5 (sqrt-rule); the project's actual 3e-4 is
**8.5-24x above either extrapolation** — a real, flagged deviation, not necessarily a problem
(the model is 1.38M params vs. DiT/SiT's 33M-675M, a different modality: DCT coefficients, not
RGB latents), but exactly what a short paired run (1e-4 vs 3e-4, same batch and steps) would
settle directly, more cheaply than trusting either extrapolation. On warmup: DiT/SiT use zero;
EDM's default 10,000-kimg rampup, at its CIFAR-10 batch of 512, is approximately 19,500 steps —
longer than this project's entire 20K-step run. The project's 100-step (0.5%) warmup sits inside
that implicit range (0% to "exceeds one short run"), not flagged as a concern by comparison.

## 5. Sampler: step count and solver order

SD3 Table 6 ([arXiv:2403.03206](https://arxiv.org/abs/2403.03206)) reports CLIP-score degradation
at low step counts relative to a 50-step baseline, for their rectified-flow objective, by model
depth:

| Depth | 5 steps | 10 steps |
|---|---|---|
| 15 | -4.30% | -0.86% |
| 30 | -3.59% | -0.70% |
| 38 | -2.71% | -0.14% |

"Larger models can be sampled using fewer steps, which we attribute to increased robustness and
better fitting the straight-path objective" (SD3 §5.1.1); Figure 3 states the rectified-flow
parameterization degrades less than `eps/linear` or `v/cos` baselines below 25 steps, same
codebase/eval.

EDM (a diffusion probability-flow ODE, not a linear/rectified interpolant, but the same
Euler/Heun solver family) ties its reduction to *both* solver order and preconditioning jointly,
not order alone — from the repo's three reproducible CLI configs for its Fig. 2a ablation
(github.com/NVlabs/edm README): plain Euler needs **512** steps to match quality; adding
2nd-order Heun plus EDM's own `{t_i}` step spacing cuts that to **128**; adding the full
sigma(t)/s(t) preconditioning reaches the same quality at **18** steps (NFE=35). Shipped
recommendation: 18 Heun steps for CIFAR-10, 40 for FFHQ/AFHQv2, 256 *stochastic* for ImageNet-64.

Rectified flow (Liu et al., [arXiv:2209.03003](https://arxiv.org/abs/2209.03003)) gets its step
reduction a third way — straightening the ODE itself via **reflow**: one round ("2-rectified
flow") measurably lowers the straightness metric
`S(Z) = integral over [0,1] of E[ ||(Z1-Z0) - v(Zt,t)||^2 ] dt` (Fig. 3d), enough that "the flow
becomes nearly straight and hence yield[s] good results even with a single Euler discretization
step"; the un-reflowed model already gives usable results at "a very small number (e.g., >=2)"
Euler steps (Fig. 1 caption) — qualitative, no NFE-vs-FID table was recovered. InstaFlow
([arXiv:2309.06380](https://arxiv.org/abs/2309.06380), Table 1a/4) quantifies the same effect on
Stable-Diffusion-scale text-to-image: reflow *alone*, no distillation, at 1 step is far worse than
the multi-step teacher (FID 68.3 vs. 22.8) — straightening is necessary but not sufficient; only
reflow+distillation recovers usable 1-step quality (FID 31.0). A second reflow round improves this
a little further (31.0 to 29.3) at steep added cost (approximately 110 more GPU-days across two
more training stages, Appendix D) — diminishing returns after the first round.

Synthesis: every concrete step-count win here came from fixing the noise schedule/preconditioning
(EDM) or from literally straightening the path (reflow), not from raising solver order alone —
EDM's own 512-to-128 jump already needs the `{t_i}` schedule change alongside Heun. A
linear/velocity interpolant (this project's choice, matching SiT's default and Lipman et al.'s OT
path) is already straighter than a typical diffusion probability-flow ODE by construction —
Lipman et al. Table 1: OT path FID 14.45 at NFE 138 vs. diffusion path FID 16.88 at NFE 187, same
ImageNet-64 setup — so the EDM-style Euler-to-Heun win is unlikely to transfer at full size onto
an already-linear, un-reflowed path. No source read here ran a matched Euler-vs-Heun-at-equal-NFE
ablation for a plain linear/velocity SiT-style model (see Gaps); at 20 Euler steps on an
already-fairly-straight path, 10 Heun steps (same NFE) is plausible but unverified.

## 6. Pitfalls for *unconditional* flow models on small data

Weakest-evidenced section: nothing found targets exactly this combination (unconditional, ~13.5K
examples, 1.38M params). Three adjacent, lower-confidence points, each flagged:

- SD3's endpoint argument (above) generalizes: blurry, near-mean predictions at the high-noise
  end of the schedule are the *correct* target there, not a training bug — do not diagnose a
  collapsed model from that end of the trajectory alone.
- "Entropy-Controlled Flow Matching" ([arXiv:2602.22265](https://arxiv.org/abs/2602.22265), Feb
  2026 — very recent, uncited elsewhere; one data point, not consensus) *proves existence of*
  "near-optimal collapse counterexamples for unconstrained flow matching" — standard MSE-velocity
  FM can, in principle, concentrate mass and lose modes under adversarial data distributions. An
  existence proof, not a measurement that it happens on real data at any given scale.
- Memorization is the small-dataset risk with an actual empirical measurement, but for *diffusion*
  models, not flow matching: Carlini et al. (USENIX Security 2023) extracted over a thousand
  near-verbatim training images from diffusion models, worse for smaller/more-duplicated data. No
  equivalent study for rectified-flow/velocity models was found, but the mechanism — MSE
  regression toward specific training targets — is shared, so the risk plausibly transfers; an
  unverified extrapolation, not a citation for this model family.
- Scale note in the project's favor, also unverified: at 1.38M parameters against 13,555
  examples, the params:examples ratio is far below every cited paper's regime (10^7-10^9 params
  on 10^6-10^9+ images) — overfitting/memorization risk from raw capacity is a priori lower here,
  but no source measured this specific ratio.

## Gaps

- No source gives timestep-sampling, weighting, or interpolant ablation numbers at this project's
  scale (721 tokens, 128-dim, depth 4, 1.38M params, 13,555 examples) — extrapolated throughout
  from ImageNet/CC12M-scale runs; SD3's per-table numbers specifically were extracted via an
  automated read of the arXiv HTML page, not visually verified against the rendered PDF.
- No NFE-vs-FID *table* (only qualitative statements) was recovered from Liu et al. 2022 for
  un-reflowed rectified flow; its full Table 2/3 results were not directly readable (PDF exceeded
  fetch size limits; the ar5iv HTML mirror also failed to parse for the EDM paper).
- No matched Euler-vs-Heun-at-equal-NFE ablation was found for a plain (non-reflowed)
  linear-interpolant/velocity SiT-style model — the Heun-vs-Euler evidence above is entirely from
  EDM's diffusion-parameterized model; "unlikely to transfer at full size" is this note's
  inference, not a citation.
- No source addresses batch-size/LR scaling *within* the flow-matching family specifically (all
  cited configs are single fixed points); the LR/batch section's linear/sqrt Adam-scaling
  extrapolation is unverified for this architecture and loss.
- EDM2's post-hoc EMA method and "sharp optimum" claim were read from the abstract and an NVIDIA
  summary blog, not the CVPR paper body (the CVF PDF returned HTTP 403 here) — the FID-vs-EMA
  half-life figure was described, not seen directly.
- The small-data-pitfalls section above is the thinnest by design: no paper found ablates
  unconditional generation, small datasets (O(10^4) examples), or this params:examples ratio for
  flow matching specifically; the three points cited are adjacent (diffusion memorization, a very
  recent uncited theory paper, SD3's endpoint argument), not direct measurements of this regime.
- The SiT paper's own hyperparameter appendix ("Appendix E") was referenced by the text but not
  independently read; the DiT/SiT optimizer/EMA/batch defaults used here come from the training
  *code*, which the paper says matches its own settings ("we strictly follow the training
  settings of DiT and did not tune any hyperparameters," SiT §3).
