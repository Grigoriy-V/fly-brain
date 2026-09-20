# Non-neural and semi-parametric samplers for a distribution known only through ~15k samples in a ~2,000-d whitened PCA latent

Scope: evidence only, state of knowledge as of 2026-09-21. Every claim carries its source URL.
Numbers marked *(via fetched full text)* were read out of the paper's HTML/PDF rendering during this
session; where only an abstract or a secondary page was reachable, that is said explicitly.

---

## Q1. Gaussian mixtures, KDE, and interpolation-as-generation at N ≈ 10^4, D ≈ 10^3

### Takeaway
KDE is asymptotically hopeless at D ≈ 10^3 (optimal MISE rate O(n^(-4/(d+4))), i.e. effectively no
convergence for any realistic n), but *low-rank / small-component mixtures* fitted **post hoc** to a
frozen autoencoder latent are a documented, cheap fix for exactly the failure the project observes
(fresh N(0,I) draws decoding to nothing): a 10-component GMM cut FID from 23.92 to 9.81 on MNIST and
from 48.20 to 44.68 on CelebA versus N(0,I) in the same latent. Convex interpolation between
neighbours is a real generator in latent spaces, but the linear midpoint's norm collapse is a known
artefact and spherical interpolation is the documented remedy.

### Cited Findings
- Optimal MISE for multivariate KDE converges at rate **O(n^(-4/(d+4)))**, with bandwidth
  H = O(n^(-2/(d+4))); the (d+4) denominator is the curse of dimensionality in closed form —
  [Multivariate kernel density estimation, Wikipedia](https://en.wikipedia.org/wiki/Multivariate_kernel_density_estimation)
- Multivariate rules of thumb both scale as n^(-1/(d+4)) per dimension: Silverman
  √H_ii = (4/(d+2))^(1/(d+4)) · n^(-1/(d+4)) · σ_i, Scott √H_ii = n^(-1/(d+4)) · σ_i, both with zero
  off-diagonal terms —
  [Multivariate kernel density estimation, Wikipedia](https://en.wikipedia.org/wiki/Multivariate_kernel_density_estimation).
  At d = 2048 the exponent −1/(d+4) ≈ −0.00049, i.e. the rule-of-thumb bandwidth is ≈ σ_i regardless
  of n — the estimate is the prior, not the data.
- "The greater the data dimension, the greater is the sample size required to obtain efficient
  estimates" — the curse of dimensionality is stated explicitly for kernel estimation of
  multidimensional densities —
  [OpenTURNS kernel smoothing documentation](https://openturns.github.io/openturns/latest/theory/data_analysis/kernel_smoothing.html)
- **Ex-post density estimation on a frozen latent (the closest published analogue to the project's
  situation).** Ghosh et al., *From Variational to Deterministic Autoencoders* (RAE) train a
  deterministic autoencoder, then fit a density on the learned latent: a full-covariance multivariate
  Gaussian and a **10-component GMM**. Latent dimensionalities: MNIST 16, CelebA 64, CIFAR-10 128.
  FID, N(0,I) → GMM: MNIST **23.92 → 9.81**; CIFAR-10 **83.87 → 76.28**; CelebA **48.20 → 44.68**
  *(via fetched full text)* — [ar5iv:1903.12436](https://ar5iv.labs.arxiv.org/html/1903.12436),
  [arXiv:1903.12436](https://arxiv.org/abs/1903.12436)
- The RAE paper's stated diagnosis: "the learned aggregated posterior distribution q_φ(z) rarely
  matches the assumed latent prior", so sampling the fixed isotropic prior produces
  out-of-distribution latents the decoder never saw *(via fetched full text)* —
  [ar5iv:1903.12436](https://ar5iv.labs.arxiv.org/html/1903.12436)
- **Norm collapse under linear interpolation, quantified.** In a 100-dimensional Gaussian latent,
  random vectors have length ≈ 10, but the linear midpoint of two of them has magnitude ≈ 7 — "over 4
  standard deviations away from the expected length". Spherical linear interpolation (slerp),
  slerp(z1,z2,μ) = [sin((1−μ)θ)/sin θ] z1 + [sin(μθ)/sin θ] z2 with θ = arccos(z1ᵀz2), preserves the
  norm and keeps interpolants in the region of maximal prior density —
  [Tom White, *Sampling Generative Networks*, arXiv:1609.04468](https://arxiv.org/pdf/1609.04468)
- Latent/manifold mixup: "interpolation in the latent or embedding space is equivalent to
  interpolating along a manifold in the input space", and latent-space convex mixing
  ẑ = γ ẑ1 + (1−γ) ẑ2 is used directly as an augmentation operator in latent diffusion pipelines —
  [Embedding Space Interpolation Beyond Mini-Batch, arXiv:2311.05538](https://arxiv.org/pdf/2311.05538);
  [Diffusion-Augmented Coreset Expansion, arXiv:2412.04668](https://arxiv.org/pdf/2412.04668)
- Geometry-aware variants normalise encoder outputs to a unit hypersphere and synthesise by
  **spherical** interpolation between cluster-level neighbours rather than linear mixing —
  [Semantic spherical mixup, Knowledge and Information Systems](https://link.springer.com/article/10.1007/s10115-026-02719-z)
- **Counter-evidence on neighbour interpolation in high dimensions.** Blagus & Lusa tested SMOTE
  (synthesis by convex combination with k nearest neighbours) on high-dimensional class-imbalanced
  data: SMOTE "does not attenuate the bias towards classification in the majority class for most
  classifiers when data are high-dimensional" and is less effective than random undersampling; only
  Euclidean k-NN benefited substantially, and only when variable selection was done **before** SMOTE —
  [Blagus & Lusa, BMC Bioinformatics 14:106 (2013)](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-14-106)
- A fully non-parametric generator can beat a trained one in a restricted regime: patch
  nearest-neighbour synthesis unconditionally generates diverse images from a single image with
  "visual quality exceeding single-image GANs by a large margin" and runtime reduced "from hours to
  seconds" — [Granot et al., *Drop the GAN*, CVPR 2022, arXiv:2103.15545](https://arxiv.org/abs/2103.15545)

### Inferences
- The project's own numbers reproduce the White norm-collapse arithmetic: a linear midpoint of two
  independent vectors of radius ≈41 would sit near 41/√2 ≈ 29, while the observed *spherical* midpoint
  stays at 34.6 against 34.2 for data. That the spherical midpoint decodes to a structured video and a
  fresh N(0,I) draw does not is the same pattern the RAE paper reports as prior/aggregate-posterior
  mismatch — not a failure of the decoder.
- Reported kurtosis 15 and radius 41.1 ± 19.0 (a mixture of scales) is precisely what a small-K
  mixture of full- or low-rank-covariance Gaussians is parameterised to represent; the RAE result is
  evidence that K ≈ 10 suffices to beat N(0,I) in a frozen latent, though at 16–128 dimensions, not
  2,048.
- A full-covariance component in 2,048 dimensions needs ~2.1M free parameters per component against
  13,555 samples, so any mixture here has to be low-rank/diagonal-plus-factor (mixture-of-PPCA-style)
  or shrunk; no source found gives a fitted full-covariance GMM at D ≈ 10^3 with N ≈ 10^4.
- KDE with a rule-of-thumb bandwidth at D = 2048 is arithmetically degenerate (bandwidth ≈ σ);
  the only defensible KDE-like object at this D is a *local* one — a kernel restricted to the k nearest
  neighbours, which is operationally the same object as neighbour interpolation.

### Gaps
- Silverman's often-quoted table of required sample sizes per dimension (Density Estimation for
  Statistics and Data Analysis, 1986, Table 4.2) could **not** be verified: the reachable online
  excerpt of the book covers §2.4 on kernel estimators only and contains no such table
  ([excerpt](https://ned.ipac.caltech.edu/level5/March02/Silverman/Silver2_4.html)). Do not quote its
  numbers from memory.
- No source found that fits a GMM or KDE in a latent of dimension ≈10^3 with N ≈ 10^4 and reports
  sample quality; the RAE evidence is at 16–128 d.
- No source found that measures whether decoded latent-mixup samples are *coherent scenes* (as
  opposed to useful augmentations for a classifier). The augmentation literature scores downstream
  accuracy, not sample coherence.

---

## Q2. ICA / Gaussianisation flows / copulas — marginal-wise density models after PCA

### Takeaway
Gaussianisation-style models (marginal 1-D transforms alternated with rotations) are the branch of
density estimation with published *small-sample* advantages, and they are cheap: Gaussianization Flows
beat FFJORD and NAF on 500–4,500-sample subsets, and sliced-iterative flows (GIS/SIG) are strongest
below a few thousand samples and fit in seconds. Their documented ceiling is exactly the project's
dimensionality: RBIG has "been limited to medium dimensionality data (on the order of a thousand
dimensions)".

### Cited Findings
- Iterative Gaussianization is a fixed-point procedure that transforms any continuous random vector
  into a Gaussian one; Gaussianization flows are universal approximators for continuous distributions
  under regularity conditions, with both efficient likelihood and efficient inversion for sampling —
  [Meng, Song, Song, Ermon, *Gaussianization Flows*, AISTATS 2020, PMLR 108:4336–4345](https://proceedings.mlr.press/v108/meng20b.html)
- Gaussianization Flows architecture and small-sample result: 10–150 stacked Gaussianization blocks
  (alternating trainable kernel and rotation layers), "50 to 100 anchor points work well in practice"
  for the trainable-KDE marginal layer, Householder-reflection rotations (D reflections for tabular
  data; 4×4 patch rotations for images). Reported test NLL: Power −0.57, Gas −10.13, Hepmass 17.59,
  Miniboone 10.32, BSDS300 −152.82 nats, MNIST 1.29 bpd. On **small subsets of 500–4,500 samples**
  drawn from the 1.3M-sample datasets, GF "significantly outperforms FFJORD and NAF in all settings"
  *(via fetched full text)* — [ar5iv:2003.01941](https://ar5iv.labs.arxiv.org/html/2003.01941),
  [arXiv:2003.01941](https://arxiv.org/abs/2003.01941)
- Sliced Iterative Normalizing Flows (SINF): iterative optimal transport of 1-D slices, axes chosen to
  maximise the Wasserstein PDF difference per iteration, "which enables the algorithm to scale well to
  high dimensions"; two variants, GIS (data→latent, density estimation) and SIG (latent→data,
  sampling) — [Dai & Seljak, ICML 2021, PMLR 139](https://proceedings.mlr.press/v139/dai21a.html),
  [arXiv:2007.00674](https://arxiv.org/abs/2007.00674)
- SINF numbers: training-set sizes swept at 100 / 1,000 / 10,000 / 100,000 on UCI data (6D–63D);
  images MNIST and Fashion-MNIST 784D, plus CIFAR-10 and CelebA 64×64. SIG FID: **MNIST 4.5,
  Fashion-MNIST 13.7, CIFAR-10 66.5, CelebA 37.3**. At 100 training points GIS fits in **0.53 s**
  (POWER, 6D) and **7.4 s** (BSDS300, 63D) against 10–4,500 s for MAF / FFJORD / RQ-NSF; the paper
  states GIS "achieves highest performance on small training sets" and beats density-trained
  normalizing flows there *(via fetched full text)* —
  [ar5iv:2007.00674](https://ar5iv.labs.arxiv.org/html/2007.00674)
- Secondary summary of the same paper: SINF "achieve[s] better performance for small training data
  (below a few thousand particles) and is considerably faster than alternatives" —
  [Semantic Scholar entry](https://www.semanticscholar.org/paper/Sliced-Iterative-Normalizing-Flows-Dai-Seljak/2cced468bf2d8a3f18e3bc1dbce5770113c3ee76)
- RBIG (Rotation-Based Iterative Gaussianization): marginal Gaussianization of every dimension
  followed by an orthonormal rotation, repeated; PCA or ICA both work as the rotation, and
  "RBIG achieves successful convergence regardless of the choice of orthonormal rotations", ICA needing
  fewer but costlier iterations. It yields entropy, total correlation, mutual information and KL as
  by-products — [Laparra, Camps-Valls, Malo, *Iterative Gaussianization: From ICA to Random
  Rotations*, IEEE TNN 22(4)](https://dl.acm.org/doi/abs/10.1109/TNN.2011.2106511),
  [arXiv:1602.00229](https://arxiv.org/pdf/1602.00229)
- RBIG's stated limit: "RBIG has been limited to medium dimensionality data (on the order of a
  thousand dimensions), and in images its application has been restricted to small image patches or
  isolated pixels, because rotation in RBIG is based on PCA or ICA which are difficult to learn and
  scale" — [Orthonormal Convolutions for the Rotation Based Iterative Gaussianization](https://www.researchgate.net/publication/361181393_Orthonormal_Convolutions_for_the_Rotation_Based_Iterative_Gaussianization)
- Gaussianization flows are positioned as capturing multimodal targets "without compromising the
  efficiency of sample generation", and match or beat Real NVP, Glow and FFJORD on tabular data —
  [Gaussianization Flows, PMLR](https://proceedings.mlr.press/v108/meng20b.html)

### Inferences
- A PCA-whitened latent is already the "rotation" half of one RBIG/GF iteration, so fitting
  non-Gaussian marginals on the 2,048 whitened coordinates and sampling them independently is the
  cheapest member of this family and is exactly one Gaussianization block; the family's own evidence
  says one block is generally not enough (10–150 blocks used), i.e. an independent-marginals sampler
  will capture the kurtosis/scale mixture but not the dependence structure that whitening leaves
  behind.
- The small-sample evidence in this family (500–4,500 samples beating deep flows) is at
  dimensionalities ≤ 784, and RBIG's own literature names ~1,000 dimensions as the practical ceiling;
  2,048 is at or just past that boundary, so the family is plausible but not demonstrated at the
  project's D.

### Gaps
- No copula-specific source with numbers at D ≈ 10^3 was found. The copula material that surfaced is
  tabular and low-dimensional (e.g. [TVineSynth, arXiv:2503.15972](https://arxiv.org/pdf/2503.15972)),
  and vine-copula parameter counts grow as O(D²) pairs; I found no evidence either way for D = 2048.
- No source found reporting Gaussianization-flow or SINF performance at D ≈ 2,000 with N ≈ 13,000.
- What these models "cannot generate" is not addressed quantitatively anywhere I found: no paper in
  this family reports a novelty/nearest-neighbour measurement on its samples.

---

## Q3. Retrieval-augmented generation (RDM, kNN-Diffusion, Re-Imagen, ReDi)

### Takeaway
Retrieval-conditioning demonstrably buys parameter count and training-data requirements, not novelty
guarantees: RDM reaches FID 12.21 on ImageNet 256² with 400M parameters against ADM's 26.21 at 554M,
and kNN-Diffusion trains a 400M text-to-image model — "a tenth of baseline models such as CogView,
DALL-E and GLIDE" — with **no paired text-image data**. Both retrieve k ≈ 4–10 CLIP neighbours. Neither
reports a quantitative novelty metric; RDM shows neighbours and nearest training images qualitatively.

### Cited Findings
- RDM design: a small diffusion/autoregressive model plus an external image database; for each training
  instance a set of nearest neighbours is retrieved and the generator is conditioned on them, so
  "the retrieval approach provides the local content, the model focuses on learning the composition of
  scenes" — [Blattmann et al., *Semi-Parametric Neural Image Synthesis*, arXiv:2204.11824](https://arxiv.org/pdf/2204.11824),
  [project page](https://ommer-lab.com/publications/semi-parametric-neural-image-synthesis/)
- RDM numbers *(via fetched full text)*: RDM **400M** parameters vs ADM **554M** and IC-GAN **191M**;
  retrieval database OpenImages, ~**20M** examples via cropping; alternative databases MS-COCO and
  WikiArt (138k images); retrieval encoder **CLIP ViT-B/32**, 512-d embeddings; ScaNN search ≈ **0.95 ms**
  for 20 neighbours out of 20M; storage **2 GB per 1M examples**; optimal **k = 4** for unconditional
  generation, **k = 8** for text-to-image generalisation. ImageNet 256²: RDM-OI **FID 12.21 / IS 77.93**
  vs ADM **26.21 / 39.70** and IC-GAN **18.17 / 59.00** —
  [ar5iv:2204.11824](https://ar5iv.labs.arxiv.org/html/2204.11824)
- RDM's database is disjoint from the train set (D_train ∩ X = ∅), and samples are shown beside both
  the retrieved neighbours and the nearest training images **in CLIP feature space** (their Fig. 5) —
  a qualitative, not numerical, novelty check *(via fetched full text)* —
  [ar5iv:2204.11824](https://ar5iv.labs.arxiv.org/html/2204.11824)
- RDM domain transfer: "simply swapping the database for one with different contents transfers a
  trained model post-hoc to a novel domain", giving class-conditional synthesis, zero-shot stylisation
  and text-to-image it was not trained on; but "a database whose examples are from a different domain
  than those of the train set leads to degraded sample quality"
  *(via fetched abstract and full text)* — [arXiv:2204.11824](https://arxiv.org/abs/2204.11824),
  [ar5iv:2204.11824](https://ar5iv.labs.arxiv.org/html/2204.11824)
- Acknowledged cost: "an inherent tradeoff between database size and model performance, as storing and
  searching indices for databases of up to billions of images can become quite costly" —
  [paperswithcode summary of RDM](https://paperswithcode.com/paper/retrieval-augmented-diffusion-models)
- kNN-Diffusion: CLIP image encoder maps images into a shared embedding space, kNN retrieval supplies
  the k most similar image embeddings as conditioning, which "extend[s] the distribution of
  conditioning embeddings" and bridges the text/image gap, enabling a text-to-image model trained
  without text — [Sheynin et al., ICLR 2023, arXiv:2204.02849](https://arxiv.org/abs/2204.02849)
- kNN-Diffusion numbers *(via fetched full text)*: **400M** parameters for both the discrete and
  continuous backbones, "a tenth of baseline models such as CogView, DALL-E, and GLIDE"; retrieval
  indices of **69M** image embeddings (modified PMD) and **400M** images (stickers); optimal **k = 10**
  with an ablation over k ∈ {1, 5, 10, 20, 100, 1000}; FID **12.5** MS-COCO, **42.9** CUB, **35.6**
  LN-COCO, **40.8** stickers (discrete); CLIP 512-d embeddings with cosine similarity; human evaluation
  favourable against LAFITE and FuseDream — [ar5iv:2204.02849](https://ar5iv.labs.arxiv.org/html/2204.02849)
- ReDi is retrieval of **trajectories**, not content: "efficient learning-free diffusion inference via
  trajectory retrieval" — it retrieves a similar partial diffusion trajectory from a precomputed
  knowledge base and skips intermediate steps, i.e. an inference-time speedup rather than a
  data-requirement reduction — [ReDi, arXiv:2302.02285](https://arxiv.org/pdf/2302.02285)
- The retrieval-conditioning idea has since been pushed toward removing the external memory
  altogether (prototype-based conditioning), which is evidence the memory cost is considered the
  method's main drawback —
  [Prototype-Guided Diffusion, arXiv:2508.09922](https://arxiv.org/pdf/2508.09922);
  [RISSOLE, arXiv:2408.17095](https://arxiv.org/pdf/2408.17095)

### Inferences
- In both RDM and kNN-Diffusion the retrieved neighbours are *conditioning*, and the generator is
  still a trained diffusion model — the saving is in parameters and in paired supervision, not in
  "no model to train". Neither paper supports a claim that retrieval conditioning works at N ≈ 10^4.
- The RDM k = 4 / kNN-Diffusion k = 10 optima are the only published guidance on how many neighbours
  carry useful content; both were selected by ablation, so k is a tunable, not a constant.
- The one structural transfer RDM demonstrates — swap the database, keep the model — is the closest
  published mechanism for "generate something the training set did not contain", and it degrades when
  the swapped database leaves the training domain.

### Gaps
- **Re-Imagen** (Chen et al., retrieval-augmented text-to-image, ICLR 2023) did not surface in the
  searches run and its numbers are not verified here. Treat it as uncovered.
- No retrieval-augmented paper found reports a nearest-neighbour *distance* distribution for its
  samples; novelty is argued by showing neighbours side by side.
- No source found applies retrieval conditioning in a non-image latent (e.g. a neural-state latent) or
  at N ≈ 10^4 database size.

---

## Q4. Generation by editing a real training sample (partial noising / Boomerang / SDEdit)

### Takeaway
Boomerang is the cleanest published statement of the mechanism: noise a real sample to a partial
timestep t_Boom, then run the reverse process from there with any pretrained diffusion model, and the
single knob t_Boom/T trades fidelity against novelty — "as t_Boom approaches T, the content ... strays
further away from the starting image". The published operating range is roughly 25–80 % of T depending
on the goal, and it is measured with LPIPS, FID, downstream accuracy and identity-recognition rates.

### Cited Findings
- Boomerang mechanism: "adding noise to an input image, moving it closer to the latent space, and then
  mapping it back to the image manifold through a partial reverse diffusion process", generating images
  "similar, but nonidentical, to the original input", with proximity controlled by the amount of noise;
  works "with any pretrained diffusion model, such as Stable Diffusion, without necessitating any
  adjustments to the reverse diffusion process" —
  [Luzi et al., arXiv:2210.12100](https://arxiv.org/abs/2210.12100), published in TMLR 2024
  ([ML Anthology record](https://mlanthology.org/tmlr/2024/luzi2024tmlr-boomerang/))
- Boomerang operating points *(via fetched full text)*: anonymisation tested at t_Boom/T = **20 %,
  50 %, 70 %, 80 %**; data augmentation at ≈ **25–40 %** (FastDPM 40 %, Patched Diffusion 30 %,
  DLSM 25 %); perceptual enhancement at t_Boom = **50–150** out of T = 250 —
  [ar5iv:2210.12100](https://ar5iv.labs.arxiv.org/html/2210.12100)
- Boomerang fidelity metrics *(via fetched full text)*: LPIPS **0.338** at t_Boom = 100 vs Deep Image
  Prior 0.353 and linear interpolation 0.449; FID **5.10** vs Deep Image Prior 7.14 —
  [ar5iv:2210.12100](https://ar5iv.labs.arxiv.org/html/2210.12100)
- Boomerang novelty measured as identity change *(via fetched full text)*: at t_Boom/T = 80 %,
  ~**100 %** of anonymised faces are declared different people by face-recognition networks (embedding
  distance) — [ar5iv:2210.12100](https://ar5iv.labs.arxiv.org/html/2210.12100)
- Boomerang downstream gains *(via fetched full text)*: CIFAR-100 62.7 % → **63.6 %**;
  ImageNet-200 66.6 % → **70.5 %**; ImageNet 63.3 % → **64.4 %**. Training only on Boomerang-edited
  data vs purely synthetic data: ImageNet **57.8 %** (Boomerang) vs 39.8 % (StyleGAN-XL);
  CIFAR-100 **55.6 %** vs 26.9 % (DLSM) — [ar5iv:2210.12100](https://ar5iv.labs.arxiv.org/html/2210.12100)
- The fidelity/novelty trade-off is stated as monotone in the one knob, with "variance dramatically
  increasing at higher t_Boom values" *(via fetched full text)* —
  [ar5iv:2210.12100](https://ar5iv.labs.arxiv.org/html/2210.12100)
- Boomerang's own comparison places **linear interpolation** as the weakest of the three local-sampling
  baselines it measures (LPIPS 0.449 vs 0.338) *(via fetched full text)* —
  [ar5iv:2210.12100](https://ar5iv.labs.arxiv.org/html/2210.12100)

### Inferences
- Boomerang needs a *pretrained* diffusion/score model to run the partial reverse process; it removes
  the need to train a generator only if one already exists. In a latent with no working score model,
  the analogous object with no model at all is "perturb a real latent and decode", for which Boomerang
  provides the measurement protocol (LPIPS/FID against the source, plus an identity/recognition test
  for novelty) but not the generator.
- Boomerang's numbers make the augmentation regime (25–40 % of T) and the "genuinely different sample"
  regime (80 %) distinguishable: the project's novelty threshold determines which knob setting is even
  relevant, and Boomerang shows the two are far apart.

### Gaps
- SDEdit's own numbers (Meng et al. 2021) were not fetched in this session; only Boomerang's framing of
  partial noising is verified here.
- No source found that quantifies Boomerang-style editing in a *whitened PCA* latent, or that relates
  the noise level to a latent-radius statistic of the kind the project measures.

---

## Q5. Novelty measurement standards

### Takeaway
There is no single standard, but there is a clear division of labour: **improved precision/recall**
(Kynkäänniemi et al. 2019) for fidelity vs coverage, **authenticity** (Alaa et al. 2022) as the
explicit third axis measuring how much a model copies training data, and **nearest-neighbour retrieval
in a strong feature space with a similarity threshold** (Somepalli et al. 2023, threshold ≥ 0.5 on
SSCD/DINO features) as the operational test for "is this sample new?". Somepalli's dataset-size sweep
is the most directly usable result: copying is severe at 300–3,000 training images and drops sharply
with more data.

### Cited Findings
- Improved precision/recall: the real and generated manifolds are approximated by a union of
  hyperspheres, each centred on a sample with radius set by its **k-th nearest neighbour**; improved
  precision = fraction of synthetic points inside the real support, improved recall = fraction of real
  points inside the synthetic support. It gives single numbers rather than a curve, improving on
  Sajjadi et al. (2018) — [Kynkäänniemi et al., NeurIPS 2019, arXiv:1904.06991](https://arxiv.org/abs/1904.06991),
  [official code](https://github.com/kynkaat/improved-precision-and-recall-metric)
- Alaa et al. introduce a **three-dimensional** metric — α-precision, β-recall, **authenticity** —
  estimated by sample-level binary classification; α-precision and β-recall score high only when the
  *typical* regions of the real and synthetic supports overlap, and diagnose mode invention, mode drop
  and density shift separately. Authenticity is an "additional, independent dimension to the
  fidelity-diversity trade-off that quantifies the extent to which a model copies training data" —
  [Alaa, van Breugel, Saveliev, van der Schaar, ICML 2022, PMLR 162](https://proceedings.mlr.press/v162/alaa22a.html),
  [arXiv:2102.08921](https://arxiv.org/abs/2102.08921),
  [PDF](https://proceedings.mlr.press/v162/alaa22a/alaa22a.pdf)
- Replication detection in practice *(via fetched full text)*: 10 feature extractors compared, best
  being **DINO ViT-B/16** with a split-product metric, **Swin-B** and **SSCD ResNet-50**; similarity =
  inner product of feature vectors, plus a "split-product" variant that chunks the feature vector,
  takes per-chunk inner products and returns the max; Stable Diffusion analysis used a threshold of
  **≥ 0.5**, hit by ≈ **1.88 %** of randomly sampled generations —
  [ar5iv:2212.03860](https://ar5iv.labs.arxiv.org/html/2212.03860),
  [Somepalli et al., CVPR 2023, arXiv:2212.03860](https://arxiv.org/abs/2212.03860)
- Training-set-size effect on copying *(via fetched full text)*: Celeb-A at **300** images — blatant
  full copying; at **3,000** — frequent but not ubiquitous, with the similarity histogram's "mass
  shift[ing] drastically to left"; full Celeb-A — minimal detectable replication, generated/training
  similarity distributions "highly overlapping"; Oxford Flowers at **1,083** images — clear copying;
  ImageNet LDM across 100 classes — "no significant copying" —
  [ar5iv:2212.03860](https://ar5iv.labs.arxiv.org/html/2212.03860)
- The field's own metrics are contested: a 2026 position paper argues "all current generative fidelity
  and diversity metrics are flawed" — [arXiv:2505.22450](https://arxiv.org/pdf/2505.22450) — and an
  extension paper unifies the precision/recall family —
  [Unifying and extending Precision Recall metrics, arXiv:2405.01611](https://arxiv.org/pdf/2405.01611)
- Precision/recall metrics are in routine use as the standard pair in follow-up work (e.g. iterative
  retraining stability analyses) — [arXiv:2310.00429](https://arxiv.org/pdf/2310.00429)

### Inferences
- For a binary "is this sample new?" the retrieval protocol (strong feature extractor → similarity to
  the nearest training sample → threshold → report the fraction above it) is the one with published
  thresholds and a published false-positive discussion; precision/recall answer a different question
  (support overlap), and authenticity answers "does the model copy" at the model level.
- Somepalli's sweep implies a 13,555-sample corpus sits in the regime where copying is *measurable*
  and dataset-size-sensitive (between the 3,000-image "frequent copying" and full-Celeb-A "minimal"
  points), so a replication measurement is not optional for a corpus of this size — though their
  models were trained diffusion models, not interpolators.
- Every one of these metrics needs the feature space fixed in advance; Somepalli's comparison of 10
  extractors is evidence that the choice of feature space, not the threshold, is where the measurement
  is most fragile.

### Gaps
- "Memorisation ratio" as a named standard metric did not resolve to a single canonical definition in
  these searches; what exists is the thresholded-similarity fraction above, plus authenticity.
  Carlini et al.'s extraction-attack line of work was not fetched here.
- No source found that validates any of these metrics in a latent space of a frozen non-image encoder
  (i.e. where "feature space" is the latent itself); all published protocols use an image feature
  extractor trained for copy detection or self-supervision.

---

## What this implies for the project's latent
Strictly within what the sources above support:

1. The observed failure (fresh N(0,I) draw decodes to nothing; spherical midpoint of two real preimages
   decodes to structure) matches two documented facts: prior/aggregate-posterior mismatch in a frozen
   latent ([RAE](https://ar5iv.labs.arxiv.org/html/1903.12436)) and the norm collapse of interior
   interpolants versus the norm-preserving slerp ([White](https://arxiv.org/pdf/1609.04468)). It is
   evidence about the *prior*, not about the decoder.
2. A post-hoc low-component mixture is the cheapest published remedy with measured gains in a frozen
   latent (10-component GMM, FID 23.92 → 9.81 on MNIST), but the published evidence is at 16–128
   latent dimensions, not 2,048 ([RAE](https://ar5iv.labs.arxiv.org/html/1903.12436)).
3. Plain KDE is not a candidate at D = 2,048: the rule-of-thumb bandwidth exponent −1/(d+4) makes the
   bandwidth ≈ σ_i irrespective of n, and MISE converges as n^(−4/(d+4))
   ([Wikipedia: multivariate KDE](https://en.wikipedia.org/wiki/Multivariate_kernel_density_estimation)).
4. Interpolation-as-generation has both support (latent/manifold mixup, spherical variants, Drop-the-GAN)
   and a high-dimensional warning (SMOTE loses its benefit in high dimensions unless variable selection
   precedes it, [Blagus & Lusa](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-14-106)).
   The PCA step the project already applies is the "variable selection before interpolation" that paper
   found to be the precondition.
5. Gaussianisation-family models (GF, SINF/GIS, RBIG) are the branch with published small-sample wins
   (500–4,500 samples beating deep flows; GIS best below a few thousand) and second-scale fitting
   times, and RBIG's own literature names ~1,000 dimensions as its practical ceiling — so 2,048 is at
   the edge of the demonstrated envelope, not inside it.
6. Boomerang supplies the measurement protocol for "start from a real sample and move it": one knob,
   with 25–40 % of T reported as the augmentation regime and 80 % as the regime where identity is
   essentially always changed ([ar5iv:2210.12100](https://ar5iv.labs.arxiv.org/html/2210.12100)).
7. For the project's own novelty claim, the published protocol with a stated threshold is thresholded
   nearest-neighbour similarity in a fixed feature space (≥ 0.5, ~1.88 % of samples for Stable
   Diffusion), and a corpus of 13,555 sits inside the size range where Somepalli et al. measured
   copying to be size-sensitive ([ar5iv:2212.03860](https://ar5iv.labs.arxiv.org/html/2212.03860)).
   Precision/recall ([arXiv:1904.06991](https://arxiv.org/abs/1904.06991)) and authenticity
   ([PMLR 162](https://proceedings.mlr.press/v162/alaa22a.html)) answer support-overlap and
   model-level copying respectively, and are complements, not substitutes.

## Gaps (overall)
- Silverman's sample-size-vs-dimension table unverified (see Q1 Gaps); no substitute table found.
- Nothing found that fits *any* of these samplers at D ≈ 2,000 with N ≈ 1.3×10^4 and reports either
  sample quality or novelty. Every quantitative result above is at lower D, higher N, or both.
- Re-Imagen and SDEdit not fetched; ReDi verified only as a trajectory-retrieval speedup.
- Copula models at D ≈ 10^3: no evidence found either way.
- Several numbers above were read out of ar5iv HTML renderings via a summarising fetch rather than the
  publisher PDF (the PMLR and CVPR PDFs were unreadable or 403 in this session). Numbers from
  [ar5iv:2204.11824](https://ar5iv.labs.arxiv.org/html/2204.11824),
  [ar5iv:2204.02849](https://ar5iv.labs.arxiv.org/html/2204.02849),
  [ar5iv:2210.12100](https://ar5iv.labs.arxiv.org/html/2210.12100),
  [ar5iv:2007.00674](https://ar5iv.labs.arxiv.org/html/2007.00674),
  [ar5iv:2003.01941](https://ar5iv.labs.arxiv.org/html/2003.01941),
  [ar5iv:1903.12436](https://ar5iv.labs.arxiv.org/html/1903.12436) and
  [ar5iv:2212.03860](https://ar5iv.labs.arxiv.org/html/2212.03860) should be re-checked against the
  camera-ready PDF before any of them is quoted in a report.
