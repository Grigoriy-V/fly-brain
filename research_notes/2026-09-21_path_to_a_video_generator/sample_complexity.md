# Sample complexity of diffusion / flow models: the memorisation→generalisation transition, and small-data recipes

Scope: evidence only, as of 2026-09-21. Every number carries its source URL. Numbers that
could not be verified from a primary source are listed under Gaps, not asserted.

Notation used below: `N` = number of training samples, `D`/`d` = dimension of the data vector
the generative model sees (pixels, or latent dimensions), `M` = model capacity.

---

## Q1. What sample counts N (vs. D and model size) mark the memorisation→generalisation transition in published work?

### Takeaway

The published transitions are reported at N in the range 10^3–10^5 for image data of
D ≈ 10^3–10^4 dimensions, with no single universal N: every paper that looks carefully says
the threshold moves with architecture, image size, data distribution and model capacity. The
one theory that gives a scaling law (Biroli et al., Nature Comms 2024) says the sample count
needed to avoid collapse onto training points grows *exponentially* in d, controlled by
α = log(n)/d, and that α must be O(1) — a condition that no practical image or latent dataset
satisfies, which is why the empirical thresholds are architecture-dependent rather than
dimension-determined.

### Cited Findings

**Kadkhodaie, Guth, Simoncelli, Mallat, ICLR 2024 — "Generalization in diffusion models arises
from geometry-adaptive harmonic representations"**

- Training set sizes swept N = 10^0, 10^1, 10^2, 10^3, 10^4, 10^5 on CelebA downsampled to
  80×80 (D = 6,400 grayscale pixels); architecture a UNet with 7.6 M parameters (3
  encoder/decoder blocks) — [ar5iv 2310.02557](https://ar5iv.labs.arxiv.org/html/2310.02557)
- At N = 10^5, "empirical test and train error are matched for all noise levels"; the paper's
  own summary is that roughly 10^5 images suffices for the generalisation regime — [ar5iv
  2310.02557](https://ar5iv.labs.arxiv.org/html/2310.02557)
- Secondary architecture/scale points: BF-CNN (~700 k parameters) on CelebA-HQ at 40×40 with N
  up to 10^4, and LSUN bedrooms at 32×32 with N = 20,000 — [ar5iv
  2310.02557](https://ar5iv.labs.arxiv.org/html/2310.02557)
- Explicit disclaimer on universality: "The minimum size of the training set, N, for which the
  model transitions from memorization to generalization indeed depends on the architecture,
  image size and data distribution." Their Figure 15 is described as preliminary evidence that
  larger resolutions need more data, but not proportionally — [ar5iv
  2310.02557](https://ar5iv.labs.arxiv.org/html/2310.02557)
- Paper of record / code: [arXiv abs](https://arxiv.org/abs/2310.02557),
  [OpenReview](https://openreview.net/forum?id=ANvmVS2Yr0),
  [GitHub](https://github.com/LabForComputationalVision/memorization_generalization_in_diffusion_models)

**Biroli, Bonnaire, de Bortoli, Mézard, Nature Communications 15:9957 (2024) — "Dynamical
regimes of diffusion models"**

- Three regimes in the reverse dynamics: a *speciation* transition (broad class structure
  emerges), then a *collapse* transition (the trajectory is captured by one specific training
  point = memorisation) — [arXiv HTML 2402.18491](https://arxiv.org/html/2402.18491v1),
  [Nature Comms](https://www.nature.com/articles/s41467-024-54281-3)
- Collapse time for a Gaussian mixture: t_C = ½ log(1 + σ²/(n^{2/d} − 1)) — [arXiv HTML
  2402.18491](https://arxiv.org/html/2402.18491v1)
- The controlling dimensionless parameter is α = log(n)/d, and "one needs α ~ O(1) in order to
  have t_C ~ O(1)" — i.e. "an exponential number of data in d to avoid the collapse" — [arXiv
  HTML 2402.18491](https://arxiv.org/html/2402.18491v1)
- Collapse is located by an excess-entropy condition s(t_C) = s^sep(t_C) — [arXiv HTML
  2402.18491](https://arxiv.org/html/2402.18491v1)
- Speciation time from the data covariance: Λ e^{−2 t_S} = 1, i.e. t_S = ½ log Λ with Λ the
  leading covariance eigenvalue — [arXiv HTML 2402.18491](https://arxiv.org/html/2402.18491v1)
- Empirical (n, d) pairs used: MNIST (10,000; 1,024), CIFAR-10 (3,000; 3,072), ImageNet16
  (2,000; 768), ImageNet32 (2,000; 3,072), LSUN (40,000; 12,288) — [arXiv HTML
  2402.18491](https://arxiv.org/html/2402.18491v1); code at
  [GitHub](https://github.com/tbonnair/Dynamical-Regimes-of-Diffusion-Models)

**Zhang, Yu et al., ICML 2024 — "The Emergence of Reproducibility and Consistency in Diffusion
Models"** (arXiv title: "…and Generalizability")

- CIFAR-10 subsets swept from 2^6 = 64 to 2^15 = 32,768 samples; architectures UNet-64,
  UNet-128, UNet-256 (embedding dimension) — [arXiv HTML
  2310.05264](https://arxiv.org/html/2310.05264v3)
- Two regimes: a memorisation regime "when the model has much larger capacity than the size of
  training data", and a generalisation regime when trained on a large dataset "without full
  capacity to memorize the whole dataset"; the phase transition is shown graphically (their
  Fig. 2) and **no numerical cut-off N is given** — [arXiv HTML
  2310.05264](https://arxiv.org/html/2310.05264v3),
  [PMLR](https://proceedings.mlr.press/v235/zhang24cn.html)
- Metrics: RP score = probability that a pair of samples from two independently trained models
  exceeds SSCD similarity 0.6; MAE score = probability that pixel-space MAE < 15.0 on [0,255];
  GL score measures dissimilarity from training data — [arXiv HTML
  2310.05264](https://arxiv.org/html/2310.05264v3)

**"Memorization to Generalization: Emergence of Diffusion Models from Associative Memory"
(2025)**

- Critical points (A = onset of memorisation, B = onset of generalisation) measured per
  architecture on CIFAR-10: U-Net(64) at K = 500 and K = 8,000; U-Net(96) at K = 2,000 and
  K = 16,000; U-Net(128) at K = 2,000 and K = 16,000. On LSUN-Church: U-Net(64) A = 1,000,
  B = 8,000; U-Net(128) A = 4,000, B = 16,000 — [arXiv HTML
  2505.21777](https://arxiv.org/html/2505.21777v3)
- Direction of the capacity effect: "with a smaller number of parameters, the
  memorization-generalization transition happens at an earlier stage (or smaller K)"; larger
  models push the transition to larger K — [arXiv HTML
  2505.21777](https://arxiv.org/html/2505.21777v3)
- Spurious/emergent attractor states, absent from the training set, appear at the boundary
  between the two regimes — [arXiv HTML 2505.21777](https://arxiv.org/html/2505.21777v3)

**Buchanan, Pai, Ma, de Bortoli (2025) — "On the Edge of Memorization in Diffusion Models"**

- The threshold is set by the ratio of model capacity M to sample count N, not by N alone: a
  crossover M* with, in their experiments, M* ≈ (4/5) N — capacity above ~80 % of the sample
  count starts memorising — [arXiv HTML 2508.17689](https://arxiv.org/html/2508.17689v1)
- Training-loss gap scaling: ℒ_N(x̄_pmem,M) − ℒ_N(x̄*) = Θ((1 − M/N)·d·σ*²) — [arXiv HTML
  2508.17689](https://arxiv.org/html/2508.17689v1)
- Their controlled setting is an *analytic Gaussian-mixture denoiser* (not a network): d = 50,
  K = 12 modes, N = 200; plus a low-rank FashionMNIST-template image model showing the same
  transition qualitatively — [arXiv HTML 2508.17689](https://arxiv.org/html/2508.17689v1)

**Yoon, Choi, Kwon, Ryu, ICML workshop 2023 — "Diffusion Probabilistic Models Generalize when
They Fail to Memorize"**

- The stated thesis is a "memorization–generalization dichotomy": generalisation and
  memorisation are mutually exclusive, and diffusion models generalise precisely when they fail
  to fully memorise the training set — [OpenReview
  PDF](https://openreview.net/pdf?id=shciCbSk9h),
  [OpenReview forum](https://openreview.net/forum?id=shciCbSk9h),
  [ICML listing](https://icml.cc/virtual/2023/28053)

### Inferences

- The three independent empirical studies that sweep N on natural images place the *onset of
  generalisation* somewhere between ~8 × 10^3 and 10^5 samples for D ≈ 10^3–6 × 10^3, with the
  spread explained by capacity: small denoisers (700 k–7.6 M parameters, UNet-64) transition at
  the low end (K ≈ 8,000 in the associative-memory paper), large ones at the high end.
- Because the reported control variable is the capacity-to-samples ratio (M* ≈ 0.8 N in the
  analytic model; "transition at larger K for larger models" empirically), N alone is not a
  sufficient statistic for where a given run sits — model size must be quoted with it.
- Biroli et al.'s α = log(n)/d is a useful order-of-magnitude yardstick rather than a
  prescription: for every dataset in their own table α ≪ 1 (e.g. CIFAR-10: log 3000 / 3072 ≈
  0.0026), yet diffusion models trained on those datasets are useful. The theory's own reading
  is that real data lie on far lower-dimensional manifolds than the ambient d, so the effective
  d in the collapse condition is not the nominal one.

### Gaps

- Gu et al. 2023 "On Memorization in Diffusion Models" was not opened in this pass; its
  "effective model memorization" quantity and its N thresholds are unverified here.
- Yoon et al.'s actual N values, dimensions and datasets were not extracted (only the
  abstract-level thesis); the workshop PDF should be read for numbers before citing any.
- Zhang et al. give no numeric transition N, only a figure; anyone needing a number must read
  their Fig. 2 off the axis.
- No source found that states a transition N for a *flow-matching / SiT* model specifically, as
  opposed to DDPM/EDM-style diffusion.

---

## Q2. How does the transition depend on inductive bias — UNet (locality, equivariance) vs. MLP or token transformer over an unstructured vector?

### Takeaway

The strongest evidence on record says the memorisation→generalisation transition is a property
of the *architecture's* inductive bias, not of the data alone: a convolutional UNet transitions
as N grows, a diffusion transformer largely does not, and destroying the spatial structure of
the data (pixel shuffling) destroys the benefit even at N = 10^5. No source was found that
measures the transition for a token transformer or MLP over a PCA/whitened vector latent, so
the claim "such a model needs far more data" is *suggested* by this evidence but not directly
measured anywhere I could verify.

### Cited Findings

- Kadkhodaie et al.: trained denoisers are "inductively biased towards … geometry-adaptive
  harmonic bases even when trained on image classes for which the harmonic basis is
  suboptimal", and two UNets trained on non-overlapping subsets converge to nearly the same
  score function once N is large enough — [arXiv abs
  2310.02557](https://arxiv.org/abs/2310.02557)
- Kadkhodaie et al., shuffled-pixel control: N = 10^5 CelebA images with a fixed pixel
  permutation applied gives "substantially worse performance than unshuffled faces" — i.e. the
  same N and the same D, minus spatial structure, loses the gain — [ar5iv
  2310.02557](https://ar5iv.labs.arxiv.org/html/2310.02557)
- Kadkhodaie et al., low-dimensional-manifold controls (disk images, sine-wave images, a
  single-face manifold) at N = 10^5, 80×80: networks show a "suboptimal PSNR slope that is less
  than 1.0" — [ar5iv 2310.02557](https://ar5iv.labs.arxiv.org/html/2310.02557)
- "On Inductive Biases That Enable Generalization of Diffusion Transformers" (2024): with a
  UNet, Jacobian eigenvectors show memorisation patterns at N = 10 and geometry-adaptive
  harmonic bases at N = 10^5; for a DiT, "the DiT's eigenvectors exhibit neither the
  memorization effect at N=10 nor harmonic bases at N=10^5", they are "random sparse patterns
  regardless of the training dataset size", and the eigenvalue distribution changes far less
  between N = 10 and N = 10^5 than the UNet's — [arXiv HTML
  2410.21273](https://arxiv.org/html/2410.21273v1),
  [project page](https://dit-generalization.github.io/)
- Same paper, N swept at 10, 10^3, 10^4, 10^5; 32×32 resolution (stated as dimensionally
  equivalent to 512×512 for latent DiTs with 2×2 patches); 400 k steps, batch 64 — [arXiv HTML
  2410.21273](https://arxiv.org/html/2410.21273v1)
- Same paper, locality restored by hand: restricting attention to local windows in early layers
  cut the PSNR generalisation gap by up to 28.53 % at N = 10^4 on CelebA (DiT-S/1) and improved
  FID from 16.10 to 11.68 at the same N — [arXiv HTML
  2410.21273](https://arxiv.org/html/2410.21273v1)
- Spectral-bias theory: spectral bias persists in deep MLP-based UNets, while convolutional
  UNets show rapid near-simultaneous emergence of many modes — weight sharing "merely
  multiplies learning rates", whereas local convolution "introduces a qualitatively different
  bias" — [arXiv HTML 2503.03206](https://arxiv.org/html/2503.03206v2)
- MLP capacity control: when MLP capacity matched dataset size the model strictly memorised
  training samples; restricting capacity avoided memorisation and produced novel outputs —
  [arXiv HTML 2605.16415 "Diffusion Models, Denoiser Architecture and
  Creativity"](https://arxiv.org/html/2605.16415v3)
- Same source's counterweight: both global and local architectures can exhibit genuine
  generative novelty *and* training-set memorisation under equivalent settings — architecture
  alone does not decide the outcome — [arXiv HTML
  2605.16415](https://arxiv.org/html/2605.16415v3)
- Minimum-norm shallow-network theory of when diffusion memorises: [arXiv
  2506.19031](https://arxiv.org/abs/2506.19031)

### Inferences

- The DiT result is the closest published analogue to a token transformer over an unstructured
  vector: it says the *mechanism* the UNet uses to generalise (a geometry-adaptive harmonic
  basis discovered from spatial structure) is simply not what a transformer learns, and that
  adding locality back recovers a measurable part of the gap. A latent whose coordinates carry
  no spatial adjacency has no locality to restore.
- The shuffled-pixel control is the cleanest evidence that N and D alone do not determine which
  regime a model is in: same N = 10^5, same D, structure removed, performance falls.
- Taken together these support a weaker, defensible statement: a transformer/MLP over a
  structureless vector cannot rely on the locality bias that produced the published N ≈ 10^4–10^5
  transitions, so those numbers are *lower bounds* rather than transferable estimates. They do
  not license a numerical multiplier.

### Gaps

- No paper found that sweeps N for a diffusion/flow model over a PCA or whitened vector latent
  and reports where the transition sits. This is the single most load-bearing gap for the
  project's question.
- "Bigger Isn't Always Memorizing: Early Stopping Overparameterized Diffusion Models"
  ([arXiv 2505.16959](https://arxiv.org/html/2505.16959v1)) appeared in searches and appears
  directly relevant to the capacity/early-stopping axis but was not read; its numbers are
  unverified here.
- Whether the MLP-vs-UNet spectral-bias result implies a *quantitative* data-requirement ratio
  is not addressed by any source found.

---

## Q3. Small-data diffusion recipes and the gains they report at N ≈ 10^3–10^4

### Takeaway

The published small-data recipes for diffusion are patch-wise training, non-leaking
augmentation, and pretrain-then-finetune. The best-documented number in the N ≈ 5 × 10^3 range
is Patch Diffusion's: FID improvements of roughly 0.1–1.5 points over an EDM baseline on
three ~5,000-image datasets, at ≥2× less training compute. All of these recipes exploit *image*
structure (crops, flips, patch coordinates), which is what makes them hard to port to a vector
latent.

### Cited Findings

- Patch Diffusion (Wang et al., NeurIPS 2023) targets datasets "as few as 5,000 images to train
  from scratch" and reports ≥2× faster training at comparable or better quality — [arXiv
  abs 2304.12526](https://arxiv.org/abs/2304.12526),
  [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/e4667dd0a5a54b74019b72b677ed8ec1-Paper-Conference.pdf)
- Patch Diffusion small-data results at 64×64, ~5,000 images each, versus EDM-DDPM++ baseline:
  AFHQv2-Cat FID 3.11 vs 4.60; AFHQv2-Dog 4.80 vs 4.94; AFHQv2-Wild 1.93 vs 2.59; all trained
  for 75 M images on 16 V100s — [arXiv HTML 2304.12526](https://arxiv.org/html/2304.12526)
- Mechanism claim: the patch-coordinate conditioning turns even one image into thousands of
  patch samples, which is how data efficiency is bought — [arXiv HTML
  2304.12526](https://arxiv.org/html/2304.12526)
- Patch Diffusion's small-data comparison is against the EDM backbone, **not** against ADA-style
  augmentation baselines — [arXiv HTML 2304.12526](https://arxiv.org/html/2304.12526)
- Non-leaking augmentation: the requirement is a wide augmentation set that prevents
  overfitting while provably not leaking into samples, with the leak-preventing conditions
  analysed; and an EDM ablation "without non-leaking augmentation … exhibits clear signs of
  overfitting", relieved by more dropout or weight decay — [Score Augmentation for Diffusion
  Models, arXiv 2508.07926](https://arxiv.org/html/2508.07926v1) (secondary description of
  Karras et al. 2022)
- Precursor for the augmentation idea and its limited-data setting: ADA / "Training Generative
  Adversarial Networks with Limited Data" — [NeurIPS 2020
  PDF](https://papers.nips.cc/paper/2020/file/8d30aa96e72440759f74bd2306c1fa3d-Paper.pdf)
- EDM2 (Karras et al., CVPR 2024) is the reference for the training-dynamics fixes
  (magnitude-preserving layers, EMA sweeps) that later small-data work builds on — [CVPR 2024
  PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Karras_Analyzing_and_Improving_the_Training_Dynamics_of_Diffusion_Models_CVPR_2024_paper.pdf)

### Inferences

- At N ≈ 5,000 with an image-structured model, a from-scratch diffusion model is already
  producing usable samples (FID 1.9–4.8 at 64×64) — small N alone is not disqualifying when the
  architecture matches the data's structure.
- Every gain reported in this literature comes from *manufacturing more effective samples out of
  structure* (patches, crops, flips). A vector latent with no spatial axis offers no equivalent
  free augmentation, so none of these numbers transfer as-is.

### Gaps

- Karras et al. 2022 (EDM) was not fetched directly; the exact non-leaking-augmentation
  probability schedule and its FID delta on CIFAR-10/FFHQ are quoted here only through a
  secondary source and should be re-verified from the EDM paper itself before use.
- Few-shot diffusion (DDPM fine-tuning on 10–100 images, e.g. DreamBooth-class and
  few-shot-generation work) was not searched in this pass; no numbers to report.
- No source found that reports a data-efficiency recipe for diffusion over a *non-image* vector
  at N ≈ 10^4.

---

## Q4. Tabular diffusion as the nearest practical analogue: N ≈ 10^4–10^5 rows, tens–hundreds of dims, MLP denoisers

### Takeaway

Tabular diffusion is the one mature body of work that trains diffusion with MLP denoisers on
unstructured vectors at N ≈ 10^3–10^5, and it does report novelty measurements — distance to
closest record (DCR). TabDDPM's DCR is several times larger than SMOTE's interpolation
baseline, which is direct evidence that an MLP diffusion model at these N does not reduce to
copying nearest training rows. But the dimensions involved are tens, not thousands, so the
analogy is strong on architecture and weak on dimension.

### Cited Findings

- TabDDPM (Kotelnikov et al., ICML 2023) evaluates 15 real-world datasets with train sizes from
  856 rows (Insurance) to 157,638 rows (Facebook Comments); 2–50 numerical features and 0–8
  categorical features — [ar5iv 2209.15421](https://ar5iv.labs.arxiv.org/html/2209.15421),
  [arXiv abs](https://arxiv.org/abs/2209.15421),
  [PMLR PDF](https://proceedings.mlr.press/v202/kotelnikov23a/kotelnikov23a.pdf)
- TabDDPM's denoiser is a plain MLP: blocks of `Dropout(ReLU(Linear(x)))`, 2–8 layers, width
  128–1024 (hyperparameters), 128-d sinusoidal time embedding, input projection fixed at 128 —
  [ar5iv 2209.15421](https://ar5iv.labs.arxiv.org/html/2209.15421)
- Novelty: mean DCR on Adult is 0.295 for TabDDPM vs 0.082 for SMOTE (higher = further from the
  nearest real record), and TabDDPM beats SMOTE on DCR across most datasets — [ar5iv
  2209.15421](https://ar5iv.labs.arxiv.org/html/2209.15421)
- Black-box privacy attack success: TabDDPM near chance (~0.5), SMOTE frequently above 0.9 —
  [ar5iv 2209.15421](https://ar5iv.labs.arxiv.org/html/2209.15421)
- DCR definition in use across this literature: Euclidean distance from a synthetic record to
  its nearest real record; low DCR means synthetic samples are essentially copies, higher DCR
  means genuinely new records — [arXiv HTML 2510.16037, "Membership Inference over
  Diffusion-models-based Synthetic Tabular Data"](https://arxiv.org/html/2510.16037)
- Contradiction worth flagging: one comparison reports TabDDPM's mean DCR better than SMOTE but
  *worse* than TVAE and CTABGAN+ — so "diffusion gives the most novel rows" is not a settled
  claim — [arXiv HTML 2510.16037](https://arxiv.org/html/2510.16037)
- Membership-inference contrast between the two leading tabular diffusion models: TabDDPM more
  vulnerable, TabSyn resilient — [arXiv HTML 2510.16037](https://arxiv.org/html/2510.16037)
- Adjacent tabular diffusion baselines for completeness: [TabDiff, arXiv
  2410.20626](https://arxiv.org/pdf/2410.20626), [TabMT, arXiv
  2312.06089](https://arxiv.org/pdf/2312.06089),
  [survey/TKDD](https://dl.acm.org/doi/10.1145/3742435)

### Inferences

- Architecturally the project's setting is closer to tabular diffusion (MLP/transformer over an
  unstructured vector, no locality) than to image diffusion — and tabular diffusion works at
  N ≈ 10^3–10^5. That is an existence proof for "vector diffusion at N ≈ 10^4", at 1–2 orders of
  magnitude lower dimension.
- DCR-style nearest-training-neighbour distance is the field's own accepted novelty statistic,
  reported *beside* a near-duplicate baseline (SMOTE). That is the same shape of evidence the
  project's contract asks for (a number beside one control).

### Gaps

- TabSyn's dataset sizes, feature counts, latent token dimension and DCR numbers could not be
  extracted (the arXiv abstract page carried no tables); [arXiv
  2310.09656](https://arxiv.org/abs/2310.09656) needs a full-text read.
- No tabular paper found that plots quality or novelty *as a function of N* — the dependence on
  sample count in this literature is not characterised, only the per-dataset outcome.
- CoDi was not examined.

---

## Q5. Diffusion / flow over PCA or whitened latents of a few thousand dims with ~10^4 samples

### Takeaway

I found no published work matching this configuration. The closest documented analogue — a
diffusion prior over a semantic embedding vector — is DALL·E 2's diffusion prior over 
CLIP image embeddings, which was trained on 250 M image/caption pairs, four orders of magnitude
more data than the project has. The absence of a matching reference is itself the finding.

### Cited Findings

- DALL·E 2's prior is a decoder-only Transformer that denoises a CLIP *image embedding* vector,
  conditioned on the encoded text, the CLIP text embedding and a timestep embedding, with a
  final placeholder token whose output predicts the denoised embedding — architecturally a
  token transformer over an unstructured vector, i.e. the same class as the project's
  model — [AssemblyAI, "How DALL-E 2 Actually Works"](https://www.assemblyai.com/blog/how-dall-e-2-actually-works)
- Data volume: CLIP was trained on 400 M image/caption pairs; the prior, decoder and upsampler
  were trained on the 250 M-pair DALL·E training set — [AssemblyAI, "How DALL-E 2 Actually
  Works"](https://www.assemblyai.com/blog/how-dall-e-2-actually-works)
- The authors compared autoregressive and diffusion priors and chose the diffusion prior as
  more compute-efficient — [AssemblyAI, "How DALL-E 2 Actually
  Works"](https://www.assemblyai.com/blog/how-dall-e-2-actually-works)
- Reference implementation with the prior's architecture in code —
  [lucidrains/DALLE2-pytorch](https://github.com/lucidrains/DALLE2-pytorch)

### Inferences

- The only well-documented "diffusion over a semantic vector" system was trained at 250 M
  samples. That does not prove 10^4 is insufficient, but it means the project has no reference
  run at its own scale to compare against, and no published prior on what N such a model needs.
- The tabular-diffusion results in Q4 are the only evidence that vector diffusion works at all
  at N ≈ 10^4, and they sit at d ≈ 10–100 rather than d ≈ 10^3.

### Gaps

- DALL·E 2 numbers above come from a secondary explainer, not the Ramesh et al. paper; they
  should be re-checked against arXiv 2204.06125 before being used in a report.
- DiffAE (Diffusion Autoencoders), latent diffusion over semantic latents, and any "vector
  diffusion" line were not located with numbers in this pass.
- No source found that characterises the effect of *whitening* a latent (which removes the
  covariance spectrum) on the speciation/collapse picture — this matters, because Biroli et
  al.'s speciation time t_S = ½ log Λ depends on the leading covariance eigenvalue Λ, and
  whitening sets Λ = 1 by construction. Whether that is benign or removes the mechanism that
  organises the early reverse dynamics is, as far as I can verify, unaddressed in the
  literature.

---

## What this implies for N = 13,555, D = 2,048

Strictly what the sources support, with no project plan attached:

1. **N = 13,555 at D = 2,048 is inside the range where published image-diffusion work places
   the transition, but at its low end.** The empirical onsets of generalisation reported are
   K ≈ 8,000 (UNet-64) to K ≈ 16,000 (UNet-128/96) on CIFAR-10 and LSUN-Church ([arXiv
   2505.21777](https://arxiv.org/html/2505.21777v3)), 2^15 = 32,768 as the top of the swept
   range in [arXiv 2310.05264](https://arxiv.org/html/2310.05264v3), and ≈10^5 for matched
   train/test error in [ar5iv 2310.02557](https://ar5iv.labs.arxiv.org/html/2310.02557). N =
   13,555 sits between the smallest model's onset and the larger models' onset.
2. **Those numbers were all measured with convolutional UNets on spatially structured data, and
   the sources say explicitly that the threshold depends on architecture, image size and data
   distribution** ([ar5iv 2310.02557](https://ar5iv.labs.arxiv.org/html/2310.02557)).
   Transferring them to a token transformer over a whitened PCA vector is not supported by any
   source.
3. **The architecture-side evidence argues the project's setting is the harder one.** A DiT does
   not show the UNet's N-driven transition at all — "neither the memorization effect at N=10 nor
   harmonic bases at N=10^5", random sparse eigenvectors at every N ([arXiv
   2410.21273](https://arxiv.org/html/2410.21273v1)) — and reinstating locality recovers a
   measurable fraction of the gap (PSNR gap −28.53 %, FID 16.10 → 11.68 at N = 10^4). A whitened
   PCA latent has no locality to reinstate.
4. **Destroying structure while holding N and D fixed measurably hurts:** shuffled-pixel CelebA
   at N = 10^5 performs "substantially worse" than unshuffled ([ar5iv
   2310.02557](https://ar5iv.labs.arxiv.org/html/2310.02557)). A whitened PCA latent is, from
   the network's point of view, closer to the shuffled condition than to the natural one.
5. **The capacity ratio, not N, is the reported control variable.** M* ≈ (4/5)N in the analytic
   model ([arXiv HTML 2508.17689](https://arxiv.org/html/2508.17689v1)) and "larger models
   transition at larger K" empirically ([arXiv 2505.21777](https://arxiv.org/html/2505.21777v3)).
   At N = 13,555, any statement about which regime the run is in requires the parameter count
   alongside N.
6. **The Biroli et al. yardstick puts the project firmly in the small-α regime**, α = log N / D =
   log(13,555)/2,048 ≈ 0.0046, versus the α ~ O(1) the theory says is needed to avoid collapse at
   order-one times ([arXiv HTML 2402.18491](https://arxiv.org/html/2402.18491v1)). Every dataset
   in that paper's own table is also α ≪ 1, so this is not a verdict — but it does mean the
   theory offers the project no comfort, and the theory's escape route is intrinsic
   dimensionality far below the ambient D, which for this latent is an unmeasured quantity.
7. **Existence proof at comparable N, lower D:** TabDDPM-class MLP diffusion over unstructured
   vectors trains successfully from 856 to 157,638 rows and produces samples measurably far from
   nearest training rows (DCR 0.295 vs SMOTE's 0.082 on Adult) ([ar5iv
   2209.15421](https://ar5iv.labs.arxiv.org/html/2209.15421)). Vector diffusion at N ≈ 10^4 is
   not without precedent; at D ≈ 2,048 it is.
8. **The one documented transformer-over-semantic-vector prior used 250 M training samples**
   ([AssemblyAI](https://www.assemblyai.com/blog/how-dall-e-2-actually-works)). There is no
   published reference run at N ≈ 10^4 and D ≈ 10^3 for this model class to compare against.
9. **The nearest-training-neighbour distance the project already uses is the field's own novelty
   statistic**, reported beside a near-duplicate baseline (DCR vs SMOTE, [ar5iv
   2209.15421](https://ar5iv.labs.arxiv.org/html/2209.15421)), and used in the
   memorisation-transition literature as the GL/RP score family ([arXiv
   2310.05264](https://arxiv.org/html/2310.05264v3)). Nothing in the sources supports reading a
   failure to decode a fresh draw as evidence about *either* regime; the published diagnostics
   for regime membership are train/test denoising-error matching ([ar5iv
   2310.02557](https://ar5iv.labs.arxiv.org/html/2310.02557)) and cross-seed reproducibility
   between two models trained on disjoint halves ([arXiv
   2310.05264](https://arxiv.org/html/2310.05264v3)) — neither of which is the same measurement
   as "does a fresh ε decode".

Not supported by any source found: any specific multiplier for how much more data a
structureless-latent transformer needs than a UNet; any claim that 13,555 is definitively
inside or outside the generalisation regime for this architecture; any claim that the observed
radius inflation of fresh draws is or is not a memorisation symptom.
