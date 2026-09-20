# How autoencoders/VAEs are built for neural population activity, vs. linear methods

Research request date: 2026-09-21. Scope: literature on latent-variable models of
recorded/simulated neural population activity (not image VAEs), for context on a
setting of 8 cell types x 721 columns x 16 DCT coefficients per sample, ~13,500
training samples, goal = a low-dimensional latent that can be *sampled* to produce
new valid population states. This note reports what the literature does; it does not
recommend a design for the project (out of scope per the request).

## 1. LFADS and successors

**LFADS (Pandarinath et al. 2018, Nature Methods)** — sequential autoencoder for
single-trial spiking data.
- Architecture: bidirectional-GRU *encoder* reads the binned spike train and
  produces (a) a per-trial "initial condition" distribution and (b) optionally a
  time-varying "inferred input" distribution; a GRU *generator* (decoder) evolves
  from the sampled initial condition to produce, at each time step, low-dimensional
  "factors" that are linearly read out to log firing rates.
- Likelihood: **Poisson**, on binned spike counts (this is the standard choice for
  spiking data, not Gaussian).
- Latents: factor dimensionality is typically small (single digits to ~tens; the
  paper's own text does not give one universal number, it is tuned per dataset) —
  a KL term regularizes the initial-condition and inferred-input posteriors toward
  Gaussian priors.
- LFADS is used on datasets of hundreds to low thousands of trials and 50-200+
  simultaneously recorded units (e.g. the NDT/NLB comparison datasets below,
  2,300-2,900 trials, 130-200 neurons).
- Generation: LFADS's generator, once trained, is driven by *inferred* (encoded)
  initial conditions/inputs from real trials — it denoises/reconstructs observed
  trials rather than being used as a from-scratch sampler in the original paper.
  Sources: [Nature Methods paper](https://www.nature.com/articles/s41592-018-0109-9),
  [biorxiv preprint](https://www.biorxiv.org/content/10.1101/152884.full.pdf),
  [lfads-torch reimplementation](https://arxiv.org/pdf/2309.01230).

**AutoLFADS (Keshtkaran et al. 2022, Nature Methods)** — LFADS plus **population-based
training (PBT)**: a population of LFADS models is trained in parallel; every few
epochs the worst performers are replaced by perturbed copies of the best performers'
hyperparameters (KL weight, learning rate, dropout, etc.). This removes the manual
per-dataset hyperparameter search that plain LFADS needs and was shown to beat plain
LFADS on macaque motor cortex, somatosensory cortex, and DMFC datasets, all without
using behavioral/task labels. On the Neural Latents Benchmark it is the strongest
non-transformer entry (see table below).
Source: [Nature Methods](https://www.nature.com/articles/s41592-022-01675-0),
[biorxiv](https://www.biorxiv.org/content/10.1101/2021.01.13.426570v2).
Gap: could not retrieve the exact PBT population size / GPU-hours from the sources
that loaded (429s on repeated fetch) — only that it is described as needing a
compute cluster ("large-scale... framework", "scalable computational
infrastructure": [PMC10374446](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10374446/)).

**Neural Data Transformer (NDT, Ye & Pandarinath 2021)** — replaces the RNN with a
BERT-style Transformer *encoder only* (no separate decoder, no recurrence).
- Architecture: 6 Transformer layers (1-2 attention heads, notably fewer heads than
  NLP transformers), trained with **masked modeling** (~20% of spike bins zeroed,
  model predicts the masked bins), Poisson likelihood on the output rates.
- No stochastic bottleneck / no KL term as used here — NDT is **not a generative,
  sampling model**; it only infers firing rates from observed spikes and cannot
  generate new activity from a prior. It is an autoencoder, not a VAE.
- Dataset: 2,296 trials, 202 neurons, 10 ms bins, 700 ms windows (macaque reaching);
  also synthetic Lorenz/chaotic-RNN datasets of 1,300-1,560 trials, 29-50 channels.
- Training: ~9.4 GPU-hours for the 6-layer model (vs. ~45 min for a single LFADS fit,
  though LFADS needs many such fits for hyperparameter search).
- Quality: R² 0.934 (Lorenz, synthetic) vs LFADS 0.921; R² 0.846 (chaotic RNN) vs
  LFADS 0.869 (LFADS wins here); R² 0.918 (motor cortex kinematic decoding) vs 0.915.
  So NDT is roughly on par with, not dramatically better than, LFADS on
  reconstruction/decoding — its main advantage is 6.7x faster inference (3.9 ms),
  useful for real-time BCI, not sample quality.
  Source: [arxiv 2108.01210 via ar5iv](https://ar5iv.labs.arxiv.org/html/2108.01210).

**STNDT (Le & Shlizerman, NeurIPS 2022)** — extends NDT with a **spatial** attention
block (neuron-to-neuron attention) alongside the temporal attention block, plus a
contrastive loss on top of masked modeling, because plain NDT "neglects the rich
covariation between individual neurons." Reported to reach state-of-the-art on
4 neural datasets (same NLB-style benchmarks). Could not retrieve exact latent
dimensionality or the numeric results table (PDF fetch failed with a parsing error
and OpenReview required interactive verification) — treat the "SOTA" claim as
reported by the abstract, not independently confirmed here.
Sources: [arxiv abstract](https://arxiv.org/abs/2206.04727),
[NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/file/72163d1c3c1726f1c29157d06e9e93c1-Paper-Conference.pdf).

**CEBRA (Schneider, Lee & Mathis, Nature 2023)** — not an autoencoder at all: a
**contrastive, self-supervised embedding** method (InfoNCE-style loss) that maps
neural population activity (optionally jointly with behavior/time labels) onto a
low-dimensional hypersphere, with a theoretical identifiability guarantee. It
produces a deterministic (or near-deterministic) embedding, not a generative latent
you sample from a prior — it is built for **consistent, decodable representations
across sessions/subjects/modalities**, not for generating new population states.
Validated on calcium imaging and electrophysiology, across species and tasks.
Detailed architecture/latent-dimensionality numbers were not retrievable through the
available web tools (Nature page required login; arxiv abstract page lacked
methods detail) — this is a **gap**.
Sources: [cebra.ai](https://cebra.ai/), [Nature paper](https://www.nature.com/articles/s41586-023-06031-6)
(paywalled/login-gated in this session), [arxiv abstract](https://arxiv.org/abs/2204.00673).

**pi-VAE (Zhou & Wei, NeurIPS 2020)** — "Poisson identifiable VAE." Unlike LFADS/NDT,
this one is explicitly framed as a *generative* latent-variable model with a
**learned, label-conditioned prior** (implemented via a normalizing-flow-like
network mapping task/behavioral labels to a Gaussian prior over the latent),
combined with a Poisson observation likelihood, giving both identifiability
(the latent is recoverable up to simple transformations, unlike a standard VAE)
and interpretability. Applied to rat hippocampus and macaque motor cortex spike
data; reported to fit better and reveal novel structure vs. baselines.
Could not retrieve exact latent dimensionality, dataset sizes, or — critically —
whether the paper explicitly validates **samples drawn from the prior** as
realistic population states (vs. only validating the *posterior* / reconstruction);
the PDF fetch failed twice (size limit, then binary parsing failure). This is
flagged as an open gap requiring a follow-up read of
[the NeurIPS paper](https://proceedings.neurips.cc/paper/2020/file/510f2318f324cf07fce24c3a4b89c771-Paper.pdf)
or the [GitHub repo](https://github.com/zhd96/pi-vae) if this line is pursued further.

**Latent circuit models (Langdon & Engel, Nature Neuroscience 2025)** — a different
family: not an autoencoder, but a **constrained low-rank RNN** fit directly to
(usually PCA-reduced) neural or trained-RNN activity, where task variables interact
through an interpretable low-dimensional recurrent connectivity matrix. Used to
reverse-engineer computation (e.g., a context-gating/suppression mechanism in
decision-making), not to generate novel population states for their own sake.
Relevant here only as a contrast: it optimizes for **mechanistic interpretability**
of a fixed low-dim circuit, not for a sampleable generative latent.
Source: [Nature Neuroscience](https://www.nature.com/articles/s41593-025-01869-7),
[GitHub](https://github.com/engellab/latentcircuit).

## 2. Do any of these models actually *generate* new population states?

Across the LFADS family and NDT/STNDT, the answer is essentially **no in practice**:
all of them are trained and evaluated as **encoders of real trials** — the
generator/decoder is always driven by a code inferred from an actual observed spike
train (LFADS) or by unmasking real (partially masked) spikes (NDT/STNDT). Their
benchmark metric, **co-smoothing** (Neural Latents Benchmark, see below), is itself
defined as predicting *held-out neurons on real trials*, never "is a state sampled
from the prior a plausible population state." None of the sources found report
sampling z ~ p(z) from an untethered prior and checking the result against held-out
real data as a generative check.

**pi-VAE** is the one architecture in this set explicitly built around a structured,
sample-able prior (conditioned on task labels) rather than an unconstrained
N(0, I) — but whether its paper demonstrates and validates *unconditional* sampling
from that prior (as opposed to using the prior only to regularize/interpret the
posterior) could not be confirmed here (see gap above).

This is a **structural difference from image VAEs**, where sampling z ~ N(0, I) and
decoding is the standard qualitative check (e.g., grids of generated digits/faces).
In the neural-population literature reviewed here, model quality is judged almost
exclusively by (a) reconstruction/held-out-neuron prediction (bits/spike, R²) and
(b) whether the recovered latent trajectory correlates with behavior — not by the
quality of unconditional samples. This matches the project's own framing (a
generated state's validity should be checked by a control comparing it to a known-
unreachable state, and by round-tripping it through a frozen encoder) rather than
by literature precedent for "does it look right," since that precedent barely exists
for this data type.

## 3. Nonlinear autoencoders vs. PCA / factor analysis / GPFA

**GPFA (Yu, Cunningham et al. 2009, NeurIPS)** is the standard linear-Gaussian
comparison point: it couples dimensionality reduction with a Gaussian-process
smoothness prior over time, so it *jointly* smooths and reduces dimensionality,
unlike the two-stage "smooth-then-PCA/FA" pipeline. It was shown to characterize
population activity better than the two-stage approach using a leave-one-neuron-out
predictive metric — but this is a comparison against **naive two-stage PCA**, not
against a trained nonlinear autoencoder.
Sources: [NeurIPS 2008 proceedings](https://proceedings.neurips.cc/paper/2008/hash/ad972f10e0800b49d76fed33a21f6698-Abstract.html),
[Cunningham & Yu 2014 review, "Dimensionality reduction for large-scale neural
recordings"](https://pmc.ncbi.nlm.nih.gov/articles/PMC4433019/).

**Direct, quantitative autoencoder-vs-linear numbers on neural population data**
were not found as a single head-to-head paper in this search; the closest
quantitative evidence is the **Neural Latents Benchmark (NLB '21)** table (below),
which puts GPFA (linear+smooth) beside NDT/AutoLFADS (nonlinear) on the same
co-smoothing metric on the same real datasets:

| Model | MC_Maze (182 units, 2,295/574 train/test trials) | MC_RTT (130 units, 1,080/271) | Area2_Bump (65 units, 364/98) | DMFC_RSG (54 units, 1,006/283) |
|---|---|---|---|---|
| trial-averaged "Smoothing" baseline | 0.211 | 0.147 | 0.154 | 0.120 |
| GPFA (linear + GP smoothness) | 0.187 | 0.155 | 0.168 | 0.118 |
| SLDS (linear dynamics) | 0.219 | 0.165 | 0.187 | 0.120 |
| NDT (nonlinear, Transformer) | 0.329 | 0.160 | 0.267 | 0.162 |
| AutoLFADS (nonlinear, RNN) | 0.346 | 0.192 | 0.259 | 0.181 |

(units: bits/spike, co-smoothing = held-out-neuron prediction on real test trials;
NLB '21, Pei et al., [arxiv 2109.04463](https://ar5iv.labs.arxiv.org/html/2109.04463),
[NeurIPS Datasets & Benchmarks proceedings](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/979d472a84804b9f647bc185a877a8b5-Paper-round2.pdf)).

Reading of this table: the nonlinear models (NDT, AutoLFADS) beat GPFA/SLDS by a
wide margin on MC_Maze and Area2_Bump (roughly +50-75%), but the margin nearly
vanishes or even *reverses direction of expectation* on MC_RTT (GPFA 0.155 vs NDT
0.160 — barely different; SLDS 0.165 actually beats NDT) and is present but smaller
on DMFC_RSG. MC_RTT is a **continuous, unstructured reaching task** with less
stereotyped trial structure than the instructed-delay MC_Maze task — i.e., the
nonlinear advantage tracked the amount of exploitable **structure/regularity** in
the task, not dataset size (all four datasets are of the same order, hundreds to a
few thousand trials, tens to ~200 neurons). This is consistent with the general
finding elsewhere in this search (PCA/linear methods fail specifically when
structure is curved/nonlinear or depends on higher-order statistics; they are not
categorically worse when the underlying dynamics really are closer to linear).
No case was found in this search where a nonlinear method was reported to
underperform PCA/GPFA by a *large* margin, only cases of a near-tie.

## 4. Spatially structured / topographic latents

Little direct precedent was found for population activity specifically. What
exists:
- **Convolutional VAEs on 2-D spatial neural fields**: found mainly in adjacent
  domains — e.g. a 5-layer conv encoder/decoder VAE used as a model of **visual
  cortex** representations, but note this is a VAE trained on **natural images**
  (ImageNet, >2M images, 128x128x3, 1,024 latents) whose learned image-latent space
  is then linearly mapped to/from fMRI voxels — it is not a VAE trained directly on
  neural population activity, and it does not validate samples from its prior as
  neural states, only as images. This is an important disanalogy to flag: most
  "VAE + visual cortex" literature uses the VAE as an image model and the brain data
  only as a linear readout target, not as the VAE's own training distribution.
  Source: [VAE for fMRI visual cortex, PMC6592726](https://pmc.ncbi.nlm.nih.gov/articles/PMC6592726/),
  [biorxiv](https://www.biorxiv.org/content/10.1101/214247v2.full).
- **Topographic VAE (Keller & Welling, NeurIPS 2021)**: organizes the *latent
  units themselves* on an n-dimensional lattice/grid so that nearby latents are
  more correlated than distant ones, and shows this yields equivariant "capsule"-
  like features (e.g. digit class/width/style separated spatially in the latent
  grid) on MNIST/sequences. This is a mechanism for imposing topographic structure
  *on the latent space*, distinct from having a spatial *input* field (columns) —
  potentially relevant as a technique but was demonstrated on image/toy data, not
  neural population recordings.
  Source: [arxiv 2109.01394](https://arxiv.org/abs/2109.01394),
  [NeurIPS proceedings](https://proceedings.neurips.cc/paper/2021/file/f03704cb51f02f80b09bffba15751691-Paper.pdf).
- **Hexagonal-lattice CNNs**: an established sub-literature (HexCNN, HexaConv,
  Hexnet) showing that native hexagonal convolution avoids the interpolation
  artifacts of square-grid resampling and enables genuine 6-fold rotation
  equivariance, but the search surfaced no work combining this with a VAE trained
  on neural population activity — only image classification/generation
  applications. Gap: no direct hex-lattice-VAE-on-neural-data precedent found.
  Sources: [HexaConv, ICLR 2018](https://indico.cern.ch/event/712901/attachments/1614275/2564782/go),
  [Hexnet hexagonal deep learning](https://arxiv.org/pdf/2101.00337).
- No paper was found using a **per-location (per-column) latent** vs. a single
  **global latent** comparison for topographic/retinotopic neural data specifically;
  this is a genuine gap in what this search could surface — the closest analog
  (Topographic VAE) structures the latent grid for correlation, not for a
  per-location generative factor with a shared global code.

## 5. VAEs on visual-system population activity (fly / mouse / primate), used for generation

- No published VAE trained directly on **Drosophila visual-system population
  activity** for generation was found. The one Drosophila-relevant connectome+VAE
  hit is unrelated to activity generation: a framework combining FlyWire subgraph
  extraction with a generative model to get **interpretable latent variables of
  circuit structure** (connectivity, not activity)
  ([arxiv 2505.13011](https://arxiv.org/html/2505.13011v1)) — a different object
  (structural graph embedding) from a VAE over activity states. The MM-GPVAE
  (multi-modal Gaussian-Process VAE) was applied to fly **whole-brain calcium
  imaging plus limb tracking**, jointly modeling neural and behavioral time series,
  but this is a general fly-brain, not specifically visual-system, dataset, and its
  role is representation/decoding of recorded trials rather than sampling new
  unconditioned brain states.
  Source: [arxiv 2310.03111](https://arxiv.org/html/2310.03111).
- For **mouse V1**: found synthetic-response *generation pipelines* used as data
  augmentation for training encoders (predicting responses to novel images from a
  trained encoder + real neuron coordinates — reported >30% reconstruction
  improvement from this augmentation), a diffusion-based (not VAE) decoder for V1
  neural-to-image reconstruction that uses a VAE only inside a Stable-Diffusion-style
  image pipeline, and a Transformer-based VAE for modeling **individual-neuron
  nonstationary dynamics and inter-area coupling** validated on synthetic GLM/EIF
  data plus real recordings — none of these are a VAE whose purpose is to sample new
  V1 population states from a learned prior; they use VAE/generative machinery for
  encoding, decoding-to-images, or coupling-estimation instead.
  Sources: [Learnable Diffusion Framework for Mouse V1, Advanced Science 2026](https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.202520220),
  [Transformer-VAE for inter-area interactions, arxiv 2506.02263](https://arxiv.org/html/2506.02263v1).
- No primate-V1 population-activity VAE used specifically for generation (as
  opposed to decoding/encoding analysis) was found in this pass.

**Overall for this question**: the search did not surface a single clear precedent
of "VAE trained on real or simulated visual-system population activity, then
sampled from its prior to generate new population states, with the generated
states validated as plausible." This is the closest thing to a genuine open gap
for the project's exact use case — the closest available precedents are (a) pi-VAE's
structured, label-conditioned prior (general neural data, not visual-system, and
generative-sampling validation unconfirmed — see gap above) and (b) LFADS/NDT-style
reconstruction-only autoencoders that are not built to be sampled from at all.

## 6. Training-set-size guidance

No neuroscience-specific rule of thumb for VAE latent size vs. sample count was
found. General ML guidance surfaced:
- A commonly cited (non-neuroscience-specific) heuristic: roughly 10-20 training
  samples per model parameter for confidence that a model learned real structure
  rather than memorizing, with more conservative versions suggesting parameters
  ≤ samples/50 for deep nets; these numbers come from general ML discussion, not a
  peer-reviewed neuroscience source, and should be treated as folklore-level
  guidance, not a derived bound.
  Source: [general ML rules of thumb](https://towardsdatascience.com/machine-learning-rules-of-thumb-b50232b4b2f8/),
  a "sample sizes" explainer ([theneuralbase.com](https://theneuralbase.com/model-card/learn/intermediate/sample-sizes/)).
- One VAE-focused source ("How to train your VAE") reports that for typical VAE
  tasks, most quality gains happen going from 10² to 10⁴ training samples, with
  diminishing returns above 10⁴, and that **latent-space dimensionality matters
  less than training-set size** for final quality — i.e. once in the 10⁴ range,
  spending effort on picking exactly the right latent size is lower-leverage than
  making sure there is enough data.
  Source: [arxiv 2309.13160](https://arxiv.org/html/2309.13160v3).
- Cross-referencing against the actual neural-population papers above: NDT/LFADS/
  AutoLFADS/NLB all train successfully (and are considered standard/adequate) on
  datasets of **hundreds to ~2,900 trials** (MC_Maze: 2,295 train trials, 182
  units), i.e. an order of magnitude *below* ~13,500 samples, using models with
  6 Transformer layers or GRU encoder/decoder/generator stacks with tens-to-~100
  units of hidden state and single-digit-to-tens of latent factors. This is the
  most concrete anchor found: the field's own standard benchmark datasets are
  smaller than the project's ~13,500-sample setting, using comparably modest
  model sizes (not deep/wide networks), and still separate the nonlinear methods
  from linear baselines on the harder (more-structured) tasks.

## Gaps

- **CEBRA**: exact latent dimensionality, encoder architecture, and quantitative
  comparison numbers vs. PCA/UMAP/pi-VAE could not be retrieved (Nature page
  login-gated, arxiv abstract page lacked methods section in this tool's rendering).
- **pi-VAE**: could not confirm from the primary source (PDF fetch failed twice)
  whether the paper explicitly validates *unconditional* samples drawn from the
  learned prior as realistic neural states, vs. only validating reconstructions/
  posteriors — this is the single most relevant unresolved question for the
  project's "sample new valid states" goal and would be worth a direct follow-up
  read of https://github.com/zhd96/pi-vae or the NeurIPS PDF if this line matters.
- **STNDT**: could not retrieve the numeric results table or exact latent
  dimensionality (PDF parse failure, OpenReview bot-check) — only the qualitative
  "SOTA" claim from the abstract is reported here.
- **AutoLFADS**: PBT population size / GPU-hours / wall-clock training budget could
  not be retrieved (repeated 429s from Nature/biorxiv/PubMed in this session).
- **Per-location vs. global latent for topographic/retinotopic data**: no direct
  comparison found in the neural-population literature; Topographic VAE and
  hexagonal-CNN literature are the closest analogs but neither is applied to
  population activity, and neither directly answers "one latent per spatial
  location vs. one shared latent."
  - **No visual-system (fly/mouse/primate) VAE explicitly built and validated for
  *generating* new population states was found** — every fly/mouse hit that uses
  VAE-like machinery does so for encoding, decoding-to-image, augmentation, or
  coupling estimation, not for unconditional generation of new activity states.
  This gap should be treated as informative in itself: it suggests the project's
  goal (sample new valid states from a connectome-constrained model) has little
  direct precedent to imitate and correspondingly little precedent for *how to
  judge* a generated state's validity beyond what AGENTS.md already specifies
  (round-trip through the frozen encoder, distance-to-nearest-training-state,
  time-shuffle / unreachable-state controls).
- Direct head-to-head "nonlinear AE vs PCA/FA on neural population data, same
  metric, many latent sizes" papers (beyond the NLB table, which only reports one
  operating point per model, not a sweep) were not found in this pass; a sweep over
  latent dimensionality vs. reconstruction quality for GPFA vs. an autoencoder on
  a common dataset would need a further targeted search if wanted.
