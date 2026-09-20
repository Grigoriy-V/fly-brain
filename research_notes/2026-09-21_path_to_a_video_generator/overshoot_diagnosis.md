# Known causes and fixes for a flow/diffusion sampler whose samples have the right spread but a systematically too-large norm

Scope: literature evidence as of 2026-09-21 for the signature "inverted-noise trajectory tracks
the ideal exactly; fresh N(0,I) trajectory departs early and ends with too large a radius".
Every claim carries its source URL. Numbers are quoted from the sources, not re-derived.

Reading note on method: several items below were read through a fetch-and-summarise tool. Where
the summary was suspiciously on-the-nose I re-fetched the abstract verbatim and corrected the
record; one such correction is flagged explicitly in the heavy-tail section. Treat any line
without a verbatim-looking quote as a paraphrase of the source.

## Q1. Exposure bias / train–sample mismatch: does the literature document norm drift of exactly this kind, and what corrections does it report?

### Takeaway
Yes — and this is the closest documented match to the measured signature. Ning et al. (ICLR 2024)
state directly that the *sampling* marginal's variance is always larger than the *training*
marginal's variance, and that the network's output L2-norm during sampling is always larger than
during training; their fix is a single scalar down-scaling of the network output (Epsilon Scaling),
training-free, worth 5–53% FID. An inverted (encoded) trajectory is by construction on the
training marginal, so it is exactly the case exposure bias does *not* affect — which is why the
preimage trajectory can track the ideal while the fresh draw does not.

### Cited Findings
- Exposure bias is defined as "the input mismatch between training and sampling", with
  prediction error at each sampling step as the root cause — [Elucidating the Exposure Bias in Diffusion Models, Ning et al., ICLR 2024](https://arxiv.org/abs/2308.15321)
- The paper states the sampling distribution's variance is **always larger** than the variance of
  the training distribution q(x_t|x_0), by a magnitude that depends on the prediction error e_t —
  [ar5iv full text](https://ar5iv.labs.arxiv.org/html/2308.15321)
- Their Figure 2 observation: "the L2-norm of ε_θ^s is always larger than that of ε_θ^t" — i.e. the
  network, fed out-of-distribution inputs during sampling, emits larger-magnitude predictions than
  it ever did in training — [ar5iv full text](https://ar5iv.labs.arxiv.org/html/2308.15321)
- The variance error of multi-step sampling grows towards the end of sampling: "the closer to t=1
  (the end of sampling), the larger the variance error of multi-step sampling" (their time
  convention; t=1 is the end of sampling, i.e. near the data) — [ar5iv full text](https://ar5iv.labs.arxiv.org/html/2308.15321)
- Fix — **Epsilon Scaling**: replace ε_θ(x_t,t) by ε_θ(x_t,t)/λ_t, "explicitly moves the sampling
  trajectory closer to the vector field learned in the training phase by scaling down the network
  output". Derived form λ(t)=kt+b; the practical recommendation is the *constant* λ(t)=b, because
  "the slope k is approaching 0 as the sampling step T′ increases"; b is found by a parameter
  search of "6 to 10 trials" — [ar5iv full text](https://ar5iv.labs.arxiv.org/html/2308.15321)
- Reported FID before → after with Epsilon Scaling (ADM-ES): CIFAR-10 100 steps 3.37 → 2.17
  (−35.6%); CIFAR-10 50 steps 4.43 → 2.49 (−43.8%); LSUN 64×64 100 steps 3.59 → 2.91 (−18.9%);
  FFHQ 128×128 100 steps 14.52 → 6.77 (−53.4%); ImageNet 64×64 2.71 → 2.39 (−11.8%);
  ImageNet 128×128 3.55 → 3.37 (−5.1%) — [ar5iv full text](https://ar5iv.labs.arxiv.org/html/2308.15321).
  The abstract confirms the headline 2.17 FID on CIFAR-10 at 100 unconditional steps —
  [abstract](https://arxiv.org/abs/2308.15321)
- The method is reported to transfer across frameworks: "Experiments on various diffusion
  frameworks (ADM, DDIM, EDM, LDM, DiT, PFGM++) verify the effectiveness" — [abstract](https://arxiv.org/abs/2308.15321);
  reference implementation [github.com/forever208/ADM-ES](https://github.com/forever208/ADM-ES)
- Epsilon Scaling is *not* applied uniformly: the paper reports that ε_θ predictions near t=0 are
  "very bad, with the loss larger than other timesteps by several orders of magnitude", so the
  correction is withheld there — [ar5iv full text](https://ar5iv.labs.arxiv.org/html/2308.15321)
- Earlier, training-side fix by the same first author — **DDPM-IP / input perturbation**: perturb
  the ground-truth samples during training to simulate inference-time prediction errors; reported
  to give "a significant improvement in sample quality while reducing both the training and
  inference times, without affecting recall and precision" — [Input Perturbation Reduces Exposure Bias in Diffusion Models, Ning et al., ICML 2023 (PMLR v202)](https://proceedings.mlr.press/v202/ning23a/ning23a.pdf); [arXiv](https://arxiv.org/abs/2301.11706)
- **Time-Shift Sampler (TS-DPM)**, a sampler-side fix that works off the *variance* of the current
  state: "By adjusting the next time step during sampling according to the approximated variance of
  the current generated samples, one can effectively alleviate exposure bias"; the shifted timestep
  is chosen inside a window [t−w/2, t+w/2] — [Alleviating Exposure Bias in Diffusion Models through Sampling with Shifted Time Steps, Li et al., ICLR 2024](https://arxiv.org/abs/2305.15583)
- TS-DPM's variance criterion (their Theorem 3.1): σ_{t_s} ≈ σ_{t−1} − ‖e‖²/(d(d−1)), where σ_{t−1}
  is the variance of the predicted x_{t−1}, e the network prediction error, d the input dimension;
  shifts occur in **both** directions (to smaller or larger t within the window) — [HTML v5](https://arxiv.org/html/2305.15583v5)
- TS-DPM FID before → after, 10 steps: CIFAR-10 F-PNDM 6.99 → 3.88 (−44.5%), CIFAR-10 DDIM
  18.71 → 12.21 (−34.7%); CelebA DDIM 17.18 → 10.61 (−38.2%), CelebA F-PNDM 9.23 → 6.96 (−24.6%) —
  [HTML v5](https://arxiv.org/html/2305.15583v5); code [TS-DPM](https://github.com/mingxiao-li/ts-dpm), [TS-DPM-ADM](https://github.com/tingyu215/TS-DPM-ADM)
- **Signal-leak bias** (a different train/sample mismatch at the *start* of sampling): during
  training images "are not corrupted up to complete noise but always contain a signal leak"; when
  that leak's distribution differs from the noise distribution, "sampling the initial latent from
  only noise creates a bias, because the model expects to find a signal leak in the initial latent".
  In Stable Diffusion the visible consequence is generated images of medium brightness. Fix: model
  the signal-leak distribution (in spatial-frequency and pixel domains) and include a signal leak in
  the initial latent — no retraining — [Exploiting the Signal-Leak Bias in Diffusion Models, Everaert et al., WACV 2024](https://openaccess.thecvf.com/content/WACV2024/papers/Everaert_Exploiting_the_Signal-Leak_Bias_in_Diffusion_Models_WACV_2024_paper.pdf); [arXiv HTML](https://arxiv.org/html/2309.15842v2); [code](https://github.com/IVRL/signal-leak-bias)
- Flow-matching-specific treatment of exposure bias exists but is unstable as a citation:
  **ReflexFlow** (Anti-Drift Rectification under training-time scheduled sampling + Frequency
  Compensation) reports "a 35.65% reduction in FID on CelebA-64" and diagnoses two root causes:
  "the model lacks generalization to biased inputs during training" and "insufficient low-frequency
  content captured during early denoising, leading to accumulated bias". **Caveat: submitted
  2025-12-04 and withdrawn by the authors 2026-02-06 for substantial revisions** — treat as a
  pointer, not evidence — [arXiv:2512.04904](https://arxiv.org/abs/2512.04904)
- Related, unverified-by-me pointers found in search listings only (titles/URLs, not read):
  [Frequency Regulation for Exposure Bias Mitigation in Diffusion Models](https://arxiv.org/pdf/2507.10072),
  [Mitigating Exposure Bias in Discriminator Guided Diffusion Models](https://arxiv.org/pdf/2311.11164)

### Inferences
- The measured asymmetry (preimage trajectory exact, fresh draw departs at t≈0.1) is what exposure
  bias predicts structurally: an inverted preimage is, by construction of the inversion, a point the
  learned field maps back along the training marginal, so no input mismatch arises; a fresh N(0,I)
  draw is only *distributionally* correct, and any per-step velocity error pushes the state off the
  marginal, after which the network is evaluated off-distribution and (per Ning et al.) returns
  larger-magnitude outputs — a positive feedback on radius.
- Epsilon Scaling is the cheapest experiment that discriminates: it is one scalar, training-free,
  and its published effect is precisely "shrink the effective output magnitude so the trajectory
  returns to the training vector field". In a velocity-parameterised flow the analogue is scaling
  the predicted velocity v_θ by 1/λ (or equivalently shrinking the step), fitted so that the
  measured sd(t) of a fresh-draw batch matches √((1−t)²+t²). This is an inference about the
  translation to flow matching, not something the source states.

### Gaps
- Ning et al. work in the DDPM/EDM ε-parameterisation; I found no paper that measures the
  *radius* of final samples (as opposed to FID) before and after Epsilon Scaling, so the size of the
  radius correction one should expect is unsourced.
- No source I found reports the exposure-bias norm drift for a *rectified-flow / logit-normal-t*
  model specifically; the only flow-matching-native treatment (ReflexFlow) is withdrawn.
- I found no source that measures exposure bias by comparing an inverted preimage trajectory with a
  fresh-draw trajectory in the way the project measured it. The asymmetry itself appears to be
  undocumented as a published diagnostic.

## Q2. Timestep-distribution mismatch: logit-normal training t vs uniform sampling grid — is the velocity field under-trained near t→0 and t→1, and does that show up as a norm error?

### Takeaway
The logit-normal density vanishes at both endpoints by construction, and SD3's own sweep shows
logit-normal(0,1) beating uniform by a wide margin on rank — so the endpoint starvation is a
deliberate trade, not an oversight. There is secondary evidence that the per-timestep loss profile
is U-shaped and that endpoint starvation leaves those regimes unresolved, but I found **no source
that ties logit-normal timestep sampling to a systematic norm/radius error in the samples.**

### Cited Findings
- SD3 defines the logit-normal timestep density π_ln(t;m,s) = 1/(s√(2π)) · 1/(t(1−t)) ·
  exp(−(logit(t)−m)²/(2s²)), with m biasing towards data (negative m) or noise (positive m) —
  [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis, Esser et al. 2024](https://arxiv.org/html/2403.03206v1)
- SD3 swept 30 combinations, m ∈ [−1,1], s ∈ [0.2,2.2]. Winner **rf/lognorm(0.00, 1.00)** with
  average rank 1.54, against uniform rectified flow at rank 5.67 — [Esser et al. 2024](https://arxiv.org/html/2403.03206v1)
- The logit-normal density "always vanishes at the endpoints 0 and 1"; SD3 introduced "Mode Sampling
  with Heavy Tails" — a distribution with strictly positive density on [0,1] — specifically to test
  whether excluding the endpoints hurt — [Esser et al. 2024](https://arxiv.org/html/2403.03206v1)
- SD3's resolution timestep shift: t_m = (√(m/n)·t_n) / (1 + (√(m/n)−1)·t_n), where n and m are pixel
  counts at source and target resolution; motivation is that higher resolutions have more pixels and
  therefore need more noise to destroy the signal. Human-preference studies selected a shift value
  of **3.0** for 1024×1024 — [Esser et al. 2024](https://arxiv.org/html/2403.03206v1)
- Esser et al. do **not** discuss endpoint under-training or terminal-SNR behaviour explicitly —
  [Esser et al. 2024](https://arxiv.org/html/2403.03206v1) (checked; absence noted)
- Secondary (search-surfaced, lower-confidence) statement of the endpoint problem: the practice of
  middle-biased t "reduces exposure to the clean endpoint, precisely where converting an
  x-prediction into velocity space amplifies prediction error", and per-timestep training loss shows
  "a U-shaped difficulty profile with persistent errors near the boundary regimes, implying that
  under-sampling the endpoints leaves fine details unresolved"; middle-biased sampling is said to
  accelerate early convergence but give "worse asymptotic fidelity than uniform sampling" —
  [Prediction–Loss Alignment for Sampler-Robust Flow Matching Training (Binary Flow Matching), arXiv:2602.10420](https://arxiv.org/html/2602.10420)
- Curriculum over the t distribution (two-phase: middle-biased then broadened) is an active line —
  [Curriculum Sampling: A Two-Phase Curriculum for Efficient Training of Flow Matching, arXiv:2603.12517](https://awesomepapers.io/generative-models/papers/2603.12517) (listing only; not read in full)

### Inferences
- The SD3 shift formula is a *time reparameterisation of the sampling grid* fitted to a data-scale
  change, and it is the only published, cheap knob that changes where the 20 or 100 Euler steps land
  without retraining. A shift fitted to make the fresh-draw sd(t) follow √((1−t)²+t²) is the same
  kind of move as TS-DPM's variance-matched time shift (Q1), which *is* published as an exposure-bias
  fix. Both point at the same one-parameter experiment.
- Since the project's departure is at t ≈ 0.1 — the low-t end, where logit-normal(0,1) puts little
  mass and where Ning et al. independently report the worst network loss — an under-trained early
  field is consistent with the observation. But the sources support only "errors are largest near the
  endpoints", not "endpoint error produces a radius overshoot".

### Gaps
- No source quantifies the velocity error as a function of t for a logit-normal-trained model on a
  non-image latent, so there is no published baseline to compare the measured departure at t ≈ 0.1
  against.
- No source studies the interaction of logit-normal training t with a *uniform* Euler grid at
  sampling (SD3 samples on a shifted grid); the mismatch the project has is not itself a studied
  configuration as far as I found.

## Q3. Small-N effects: the learned score at small t is the score of an empirical mixture of N Gaussians — what happens to the flow map from a fresh draw vs from an inverted sample?

### Takeaway
Biroli–Mézard et al. give exact times for the two transitions of the backward process and an
explicit curse-of-dimensionality criterion α = log n / d; with D = 2,048 and N = 13,555, α ≈ 0.0046,
far below the O(1) they say is needed, which puts the project's regime deep in the
"collapse/memorisation" side of their analysis. But **none of the sources I read claim that a
memorising model overshoots the radius**, and Biroli et al. explicitly do not analyse what the
empirical score does to sample variance. So this line explains "a draw is not a scene" better than
it explains "the radius is 59% high".

### Cited Findings
- Three dynamical regimes in the backward process; "the generative dynamics starting from pure noise
  ... encounters first a 'speciation' transition where the gross structure of data is unraveled ...
  followed at later time by a 'collapse' transition where the trajectories of the dynamics become
  attracted to one of the training points" — [Dynamical Regimes of Diffusion Models, Biroli, Bonnaire, de Bortoli, Mézard, Nature Communications 15:9957 (2024)](https://www.nature.com/articles/s41467-024-54281-3); [arXiv:2402.18491](https://arxiv.org/abs/2402.18491)
- Speciation time: defined by Λe^(−2t_S) = 1 with Λ the principal eigenvalue of the data covariance,
  i.e. t_S = ½ log Λ; for a high-dimensional Gaussian mixture with |m|² = d μ̃², t_S ≈ ½ log d —
  [arXiv HTML](https://arxiv.org/html/2402.18491v1)
- Collapse time for Gaussian mixtures: t_C = ½ log[1 + σ²/(n^(2/d) − 1)]; the controlling ratio is
  **α = log(n)/d**, and "one needs α ∼ O(1)" for a meaningful collapse time, i.e. n ≈ e^(αd) samples —
  this is their statement of the curse of dimensionality — [arXiv HTML](https://arxiv.org/html/2402.18491v1)
- Below the collapse time the empirical distribution "decomposes in separated lumps around the points
  of the training set, and a given trajectory is committed to the attractor of the original data
  point, that is reached at t=0" — i.e. memorisation of the training set — [arXiv HTML](https://arxiv.org/html/2402.18491v1)
- Their framework assumes the "exact empirical score" / optimally trained score; they state "Moving
  beyond the exact empirical score hypothesis opens up multiple avenues for further research", and
  they do **not** analyse whether empirical-score dynamics reproduce the data variance or norm —
  [arXiv HTML](https://arxiv.org/html/2402.18491v1) (absence noted)
- Memorisation→generalisation transition is a function of N: for CelebA, training sets up to
  N = 3200 give samples identical to training examples, while genuinely new samples emerge at
  N ≥ 6400; for large N two independently trained networks "converge to the same score function and
  thus sample from the same model density, generating nearly identical samples"; generalisation is
  attributed to a geometry-adaptive harmonic basis — [Generalization in diffusion models arises from geometry-adaptive harmonic representations, Kadkhodaie et al., ICLR 2024](https://arxiv.org/pdf/2310.02557)
- The number of training samples needed for the memorisation→generalisation transition is reported to
  scale linearly with the intrinsic dimension of the dataset — search-surfaced claim attributed to
  follow-up work; I did not read the primary source, see Gaps —
  [Memorisation, convergence and generalisation in generative models, arXiv:2605.21402](https://arxiv.org/html/2605.21402) (listing only)
- Finite-sample flow matching, closest published analysis of the empirical velocity field: the
  empirical velocity is "a time-dependent weighted average of directions pointing toward individual
  training data points"; the raw empirical minimiser "is generally not a gradient field, even when
  the individual conditional velocity fields are gradients" (a finite-sample obstruction to
  Benamou–Brenier optimality); and for affine conditional flows with positive terminal scale,
  averaging conditional terminal laws over the empirical target measure "gives exactly a kernel
  density estimator" — [On the Hidden Biases of Flow Matching Samplers, Lim et al., arXiv:2512.16768](https://arxiv.org/html/2512.16768)
- Crucially for this project's signature, the same paper's "energetic bias": **"The kinetic energy
  distribution of generated samples is primarily governed by the choice of the source distribution
  rather than the target data"**, with "Gaussian sources produce light energy tails, while
  polynomially tailed sources yield corresponding polynomial bounds"; Figure 1 shows empirical
  survival curves of integrated kinetic energy on two moons, eight Gaussian clusters and
  checkerboard for Gaussian vs Student-t sources with ν ∈ {2,5,10}, and Figure 3 matches the
  predicted −ν/2 tail exponent in an affine ODE setting — [arXiv:2512.16768](https://arxiv.org/html/2512.16768)
- That paper does **not** discuss Euler vs higher-order discretisation ("These approximations may
  introduce additional biases", left to future work) and does **not** discuss inverted vs fresh-draw
  initialisation — [arXiv:2512.16768](https://arxiv.org/html/2512.16768) (absences noted)
- Other small-t / low-noise analyses found but not read in full:
  [Diffusion models under low-noise regime, arXiv:2506.07841](https://arxiv.org/pdf/2506.07841);
  [On the Closed-Form of Flow Matching: Generalization Does Not Arise from Target Stochasticity, arXiv:2506.03719](https://arxiv.org/html/2506.03719v1);
  [Generalization Dynamics of Linear Diffusion Models, arXiv:2505.24769](https://arxiv.org/pdf/2505.24769)

### Inferences
- With D = 2,048 and N = 13,555, α = log N / D ≈ 9.51/2048 ≈ 0.0046 ≪ O(1). On Biroli et al.'s own
  criterion this is the regime where the collapse transition happens early in the backward process,
  i.e. an optimally-trained empirical score would be expected to commit trajectories to training
  points. The observed behaviour is the opposite of that (the draw is *not* a training point and its
  radius is too large), which suggests the model is **not** at the exact-empirical-score optimum —
  consistent with a velocity-field error story (Q1/Q2) rather than a memorisation story. This is my
  inference; the sources do not make it.
- "Preimage is Gaussian, fresh draw overshoots" does **not** match the memorising-model picture as
  the sources describe it: a memorising model would put the draw *on* a training point, hence at the
  training radius (41.1), not at 53.1. Inversion returning a clean Gaussian preimage (radius 44.4,
  kurtosis 3.02) is evidence the map is well-behaved on the data's own preimages, which is
  orthogonal to memorisation.

### Gaps
- No source I found analyses the flow map's behaviour on inverted preimages vs fresh draws at finite
  N; this appears to be an unstudied comparison.
- The "linear in intrinsic dimension" scaling of the memorisation transition was only search-surfaced;
  I did not verify it against the primary text.
- Biroli et al. explicitly leave the non-exact-score case open, so there is no published statement
  about the radius of samples produced by a *trained* (not exact) empirical score at small α.

## Q4. Heavy-tailed / mixture-of-scales data: do Gaussian-base flows overshoot scale, and is a heavier-tailed base the fix?

### Takeaway
The heavy-tail literature is clear that a Gaussian base plus a Lipschitz network cannot produce
power-law tails and that Student-t bases fix tail estimation; and "Hidden Biases" says the *kinetic
energy tail* of generated samples is set by the source, not the data. But I found **no source that
reports a Gaussian-base flow producing a systematically too-large radius on heavy-tailed data** —
the documented failure is in the tails (rare/extreme events), and the direction is not stated as
overshoot. One search summary asserted radial overshoot; the verbatim abstract does not, and I have
marked that as a tool artefact, not evidence.

### Cited Findings
- "Standard generative models struggle with heavy-tailed data: Lipschitz architectures cannot produce
  power-law tails from Gaussian noise, and interpolating between heavy-tailed data and Gaussians is
  ill-posed." Fix proposed: a coordinate-wise soft-log transform φ(x) = sign(x)·log(1+|x|) before
  training, exponentiate after generation, with a Hill diagnostic deciding per coordinate whether to
  transform, "leaving light-tailed margins untouched". On a 144-configuration multivariate benchmark
  (3 copulas, d up to 100, 4 tail indices) Log-FM "dominates specialized baselines on W₁, CVaR₉₉, and
  extreme-quantile metrics, and is the only method with zero severe divergences across 2,880 runs" —
  verbatim abstract, [Tail Annealing for Heavy-Tailed Flow Matching, Jean Pachebat, arXiv:2605.20068](https://arxiv.org/abs/2605.20068) (submitted 2026-05-19, revised 2026-06-21)
- **Correction / epistemic flag:** a first fetch of the same PDF returned a summary claiming the paper
  identifies Gaussian-base flow matching "overshoot[ing] in the radial direction—generating samples
  with excessively large norms". The verbatim abstract (above) contains no such claim. Treat the
  radial-overshoot phrasing as a summariser artefact and **do not cite it**.
- Student-t diffusion/flow: the multivariate Student-t is used as the base noise distribution with its
  degrees of freedom giving "controllability over tail estimation"; because it has polynomially
  decaying density it can model heavy tails, whereas "traditional diffusion and flow-matching models
  with standard Gaussian priors fail to capture heavy-tailed behavior". Instantiations **t-EDM** and
  **t-Flow** "outperform standard diffusion models in heavy-tail estimation on high-resolution weather
  datasets", with tail behaviour controlled by a single scalar hyperparameter — [Heavy-Tailed Diffusion Models, Pandey et al., ICLR 2025, arXiv:2410.14171](https://arxiv.org/abs/2410.14171); [NSF PAR copy](https://par.nsf.gov/servlets/purl/10640550)
- The source distribution, not the data, sets the kinetic-energy tail of generated samples in
  finite-sample flow matching (quoted in Q3) — [On the Hidden Biases of Flow Matching Samplers, arXiv:2512.16768](https://arxiv.org/html/2512.16768)
- Related listings not read in full:
  [Denoising Lévy Probabilistic Models / α-stable diffusion](https://arxiv.org/pdf/2502.09306) (found
  only as a citation inside a Langevin-analysis paper — see Gaps);
  [Lévy-Flow Models, arXiv:2604.00195](https://arxiv.org/pdf/2604.00195);
  [Go With the Flow: Fast Diffusion for Gaussian Mixture Models, arXiv:2412.09059](https://arxiv.org/pdf/2412.09059)

### Inferences
- The project's latent (kurtosis 15, radius 41.1 ± 19.0 where √D = 45.3, an explicit mixture of
  scales) is exactly the data class these papers target, and the Log-FM result suggests the cheapest
  version of this fix is a *data-side* monotone squashing of the heavy coordinates rather than a new
  base distribution or a retrained architecture — it is a preprocessing change, invertible, and the
  Hill diagnostic decides per coordinate. This is an inference about applicability; the paper's
  benchmark is synthetic copulas up to d = 100, not d = 2,048.
- "Hidden Biases" gives the only mechanism I found that directly links the *base* to a *scale*
  property of the output. But it is about the upper tail of kinetic energy, not the mean radius, so
  it does not by itself explain a uniform 59% shift.

### Gaps
- I did not find a paper measuring the *mean radius* of samples from a Gaussian-base flow trained on
  heavy-tailed, whitened data against the data's own radius — the direct analogue of the project's
  measurement. This is the central gap of this section.
- I could not verify Shariatian et al. "Denoising Lévy Probabilistic Models" from a primary source
  within the call budget; it appeared only as a reference inside another paper.
- No source states whether whitening/PCA before a Gaussian-base flow makes the tail mismatch better
  or worse.

## Q5. Sampler-side fixes with reported gains

### Takeaway
Two sampler-side fixes are published specifically against this failure mode and are one-parameter
and training-free: Epsilon Scaling (constant down-scaling of the network output, Q1) and the
Time-Shift Sampler (variance-matched timestep, Q1). Beyond those, the documented levers are
higher-order solvers, stochasticity as an error-contracting mechanism, and keeping the initial noise
on its Gaussian shell — the last of which is documented as a *constraint to avoid radial drift*
during noise optimisation, which is the closest published statement to a "radius control" trick.

### Cited Findings
- High-dimensional isotropic Gaussian noise concentrates in a thin spherical shell, ‖z‖ = σ√d ± O(1);
  "for isotropic Gaussian priors, the probability density depends only on the noise norm, and
  high-dimensional samples concentrate in a thin annulus around their typical radius"; to preserve
  prior likelihood, noise "can move along the fixed-radius sphere passing through the initial noise",
  and methods "constrain the optimized noise to remain on the sphere determined by its initial draw,
  allowing the noise to move along a fixed-radius Gaussian shell rather than toward a
  different-norm region" — [Manifold-Constrained Noise Optimization for Diverse Diffusion Sampling, arXiv:2607.23937](https://arxiv.org/html/2607.23937)
- Trajectory-scale facts from the same line of work: "the magnitude of the predicted noise is
  approximately distributed around √d, and the total length of the sampling trajectory is
  approximately σ_T√d" — [arXiv:2607.23937](https://arxiv.org/html/2607.23937); see also
  [On the Trajectory Regularity of ODE-based Diffusion Sampling, arXiv:2405.11326](https://arxiv.org/pdf/2405.11326)
  and [Geometric Regularity in Deterministic Sampling Dynamics of Diffusion-based Generative Models, arXiv:2506.10177](https://arxiv.org/pdf/2506.10177) (listings; not read in full)
- Heun / second-order correction: "Heun's method introduces an additional step for the correction of
  first-order sampling, updating current states with an averaged gradient term ... achieving higher
  quality while allowing for fewer sampling steps" — search-surfaced summary of the EDM sampler
  family; reference implementation of EDM stochastic Heun with churn parameters at
  [NVIDIA PhysicsNeMo, edm_stochastic_heun](https://docs.nvidia.com/physicsnemo/latest/_modules/physicsnemo/diffusion/samplers/edm_stochastic_heun.html)
- Stochasticity as error correction: "stochastic transitions contract errors but slow convergence,
  while near-deterministic transitions accelerate sampling but are prone to error accumulation. An
  effective sampler should combine both" — [Adaptive Stochastic Coefficients for Accelerating Diffusion Sampling, arXiv:2510.23285](https://arxiv.org/pdf/2510.23285);
  related [On the Error-Correcting Effects of Stochasticity in Discrete Diffusion, arXiv:2605.26582](https://arxiv.org/pdf/2605.26582) (listing only)
- TS-DPM (Q1) is a pure sampler-side fix: it changes only which timestep the next step is taken at,
  based on the measured variance of the current state, with "minimal additional computations", and
  reports 24–44% FID reductions at 10 steps — [arXiv:2305.15583](https://arxiv.org/abs/2305.15583); [HTML v5](https://arxiv.org/html/2305.15583v5)
- Epsilon Scaling (Q1) is a pure sampler-side fix: one constant b, found in 6–10 trials, no retraining,
  5–53% FID reduction across five datasets — [ar5iv](https://ar5iv.labs.arxiv.org/html/2308.15321)
- Magnitude control on the *training* side: EDM2 observes that "explicit normalizations in the residual
  paths try to keep magnitudes under control, but nothing prevents them from growing in the main path",
  and that in their baseline config "the magnitudes of both activations and weights grow without bound
  over training"; their fix redesigns every operation (convolution, activation, concatenation,
  summation) to preserve magnitudes in expectation, and constrains weight rows to unit norm, removing
  the need for explicit weight decay — [Analyzing and Improving the Training Dynamics of Diffusion Models, Karras et al., CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Karras_Analyzing_and_Improving_the_Training_Dynamics_of_Diffusion_Models_CVPR_2024_paper.pdf)

### Inferences
- The published ranking by evidence strength for a one-shot, training-free trial is: (1) Epsilon
  Scaling / velocity down-scaling — strongest evidence and directly about output magnitude;
  (2) variance-matched time shift (TS-DPM) or an SD3-style shift of the sampling grid — strong
  evidence, same one-parameter cost, and uses the sd(t) curve the project already measures as its
  fitting target; (3) more steps / Heun — cheap but addresses discretisation error, which the
  project's 20-vs-100-step measurement can already rule in or out; (4) drawing ε on the shell √D
  instead of from N(0,I) — this is documented as *preserving* the prior's typical radius, i.e. it
  changes nothing about the *mean* radius of an N(0,I) draw in D = 2,048 (both are ≈ 45.3 with
  different spread), so on the sources it should shrink the *spread* of output radii, not the mean.
  Since the project's measured spread already matches (15.0 vs 15.2), the shell trick is predicted by
  the sources to be the least useful of the four here. All four rankings are my inference.
- Dynamic thresholding and "norm rescaling" of the *output* would move the reported radius by
  construction; I found no source that validates that as a fix for this mechanism rather than a
  cosmetic clamp, so it should be treated as a measurement artefact risk, not a fix.

### Gaps
- I found no paper reporting the effect of step count (20 vs 100) on the *norm* of samples,
  only on FID; so the project's own two-step-count measurement is the only evidence available on that.
- No source I found evaluates guidance-interval tricks in an unconditional setting, which is what this
  project has; guidance-interval papers assume classifier-free guidance.
- Dynamic thresholding (Imagen) was not retrieved in this pass; its published justification is
  saturation under high guidance weight, not norm drift from a fresh draw.

## Which of these matches the measured signature

### Takeaway
Only one line in the literature states the measured signature in the same terms: exposure bias as
formalised by Ning et al. — sampling-marginal variance always exceeding the training marginal, and
the network's output norm always larger during sampling than during training. The heavy-tail line
explains the *data* the project has, and the small-N line explains why a draw need not be a scene,
but neither is documented as producing a too-large radius.

### Cited Findings
- Match: "the sampling distribution variance is always larger than the variance of the training
  distribution q(x_t|x_0)" and "the L2-norm of ε_θ^s is always larger than that of ε_θ^t" —
  [ar5iv, Ning et al. ICLR 2024](https://ar5iv.labs.arxiv.org/html/2308.15321). This is the same
  direction (too large), the same object (the marginal's scale), and it is caused by starting from
  noise and accumulating per-step error, which is exactly the leg of the project's measurement that
  fails while the inverted leg succeeds.
- Match, mechanism only: the source distribution, not the data, governs the kinetic-energy tail of
  generated samples in finite-sample flow matching — [arXiv:2512.16768](https://arxiv.org/html/2512.16768).
  Supports "the base choice can set a scale property of the output"; does not claim the mean radius is
  too large.
- Partial match, location of the error: network loss near t=0 is "larger than other timesteps by
  several orders of magnitude" — [ar5iv](https://ar5iv.labs.arxiv.org/html/2308.15321) — and the
  logit-normal density "always vanishes at the endpoints 0 and 1" —
  [Esser et al. 2024](https://arxiv.org/html/2403.03206v1). Both are consistent with the measured
  departure at t ≈ 0.1; neither states that the consequence is a norm error.
- Non-match: a memorising model commits trajectories to training points reached at t=0 —
  [arXiv:2402.18491](https://arxiv.org/html/2402.18491v1) — which would place a draw at the training
  radius, not 29% above it. The measured preimage statistics (radius 44.4 ± 0.6, kurtosis 3.02,
  round-trip 0.1%) say the map is well-conditioned on data preimages, which the memorisation
  literature does not connect to overshoot either way.
- Non-match as cited: no source states that Gaussian-base flow matching on heavy-tailed data yields a
  systematically too-large sample radius; the one summary that said so was a tool artefact, corrected
  above against the verbatim abstract — [arXiv:2605.20068](https://arxiv.org/abs/2605.20068).

### Inferences
- Within what the sources support, the signature is best described as exposure bias / train–sample
  input mismatch, plausibly concentrated where the logit-normal t distribution and the reported
  per-timestep loss profile both say the field is weakest (small t), and the two published
  training-free corrections for it (a constant output down-scale; a variance-matched time shift) are
  the experiments with the strongest evidence behind them.
- A discriminating measurement the sources imply but do not run: compute the norm of the predicted
  velocity along a fresh-draw trajectory and along a preimage trajectory at matched t. Ning et al.'s
  Figure 2 is the ε-space version of exactly this plot; if the fresh-draw velocity norm is uniformly
  larger, the diagnosis is theirs and the constant-λ fix applies directly.

### Gaps
- Because no source measures radius before/after any of these fixes, the literature cannot predict
  how much of the 59% the corrections would remove. Ranking above is by strength of evidence for the
  *mechanism*, not by expected effect size on radius.
- The project's specific configuration — logit-normal-trained velocity field, uniform Euler grid,
  whitened PCA latent of non-image data at D = 2,048 with kurtosis 15 and N ≈ 1.4×10⁴ — is not
  covered as a studied setting by any source found in this pass.
