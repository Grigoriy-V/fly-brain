# Spatial inductive bias in diffusion models: locality, equivariance, small-N compute, and non-rectangular grids

Knowledge state: 2026-09-21. Every claim carries its source URL. Numbers not found are
listed under Gaps, never invented.

## Q1. Kamb & Ganguli 2024/2025 (ELS machine), Niedoba et al. 2024, Kadkhodaie et al. 2024 — what does the inductive bias actually buy?

### Takeaway
Three independent lines of work converge on the same mechanism: a diffusion denoiser that is
**local** (its output at a site depends only on a neighbourhood) and **translation-equivariant**
(the same denoising rule at every site) cannot fit the optimal score of a finite training set,
and that failure is exactly what produces novel images — a locally consistent mosaic of training
patches. Kamb & Ganguli reproduce trained convolutional diffusion models' outputs analytically
with no training at all (median r² 0.94–0.96), and the same theory only partially explains
self-attention UNets (r² ≈ 0.77), i.e. non-locality moves the model away from the mechanism.
One 2025 paper disputes the *origin* of that locality, attributing it to image data statistics
rather than to the convolutional architecture — a distinction that matters directly for a
non-image state space.

### Cited Findings

**Kamb & Ganguli, "An analytic theory of creativity in convolutional diffusion models"
(arXiv:2412.20292, ICML 2025)**
- The framing: "Score-matching diffusion models can generate highly original images that lie far
  from their training data, yet optimal score-matching theory suggests that these models should
  only be able to produce memorized training examples" — i.e. the **exact/optimal score of a
  finite dataset is a memorizing score**; deviation from it is a prerequisite for novelty —
  [arXiv:2412.20292](https://arxiv.org/abs/2412.20292)
- They identify "two simple inductive biases — locality and equivariance — that induce a form of
  combinatorial creativity by preventing optimal score-matching; and result in fully analytic,
  completely mechanistically interpretable, local score (LS) and equivariant local score (ELS)
  machines" — [arXiv:2412.20292](https://arxiv.org/abs/2412.20292)
- Mechanism: "a locally consistent patch mosaic mechanism of creativity, in which diffusion
  models create exponentially many novel images by mixing and matching different local training
  set patches at different scales and image locations" —
  [arXiv:2412.20292](https://arxiv.org/abs/2412.20292)
- The ELS machine is "not a trained diffusion model, but rather a set of equations which can
  analytically predict the composition of denoised images based solely on the mechanics of
  locality and equivariance" — [ICML 2025 poster/slides summary](https://icml.cc/virtual/2025/poster/44336)
- Agreement with *trained* convolution-only diffusion models, median r²: **0.95 CIFAR-10, 0.94
  FashionMNIST, 0.94 MNIST, 0.96 CelebA** — [arXiv:2412.20292](https://arxiv.org/abs/2412.20292)
- On non-local architectures: the theory "partially predicts the outputs of pre-trained
  self-attention enabled UNets (median r² ∼ 0.77 on CIFAR10), revealing an intriguing role for
  attention in carving out semantic coherence from local patch mosaics" —
  [arXiv:2412.20292](https://arxiv.org/abs/2412.20292)
- Presented at ICML 2025 (Vancouver); Stanford's own write-up describes it as "an analytic,
  interpretable and predictive theory of creativity in convolutional diffusion models" —
  [Stanford EE](https://ee.stanford.edu/surya-gangulis-research-uncovers-how-ai-creates)

**Niedoba et al., "Towards a Mechanistic Explanation of Diffusion Model Generalization"
(arXiv:2411.19339, ICML 2025)**
- They propose "a training-free mechanism explaining diffusion model generalization by comparing
  pre-trained diffusion models to theoretically optimal empirical counterparts", identifying a
  **shared local inductive bias across network architectures**, and hypothesise that "network
  denoisers generalize through localized denoising operations" that approximate the training
  objective over much of the distribution — [arXiv:2411.19339](https://arxiv.org/abs/2411.19339)
- Their construction, **Patch Set Posterior Composites (PSPC)**, aggregates *local empirical*
  denoisers and "exhibits consistent visual similarity to neural network outputs with lower mean
  squared error than previously proposed methods" —
  [arXiv:2411.19339](https://arxiv.org/abs/2411.19339)
- Their earlier finding: diffusion models deviate from the exact empirical denoiser only at
  **intermediate timesteps**, and errors in that band are "primarily responsible for diffusion
  generalization" — [arXiv:2411.19339](https://arxiv.org/abs/2411.19339);
  code: [plai-group/pspc](https://github.com/plai-group/pspc)

**Kadkhodaie, Guth, Simoncelli, Mallat, "Generalization in diffusion models arises from
geometry-adaptive harmonic representations" (arXiv:2310.02557, ICLR 2024 — oral)**
- Denoisers trained on photographic images "perform a shrinkage operation in an orthonormal basis
  consisting of harmonic functions adapted to the geometry of features in the underlying image"
  (geometry-adaptive harmonic bases, GAHB) —
  [arXiv:2310.02557](https://arxiv.org/abs/2310.02557)
- Training-set-size ladder actually run: **N = 1, 10, 100, 1,000, 10,000, 100,000** images —
  [ar5iv rendering of 2310.02557](https://ar5iv.labs.arxiv.org/html/2310.02557)
- Resolution: main experiments at **80×80** (CelebA, LSUN bedroom), with additional 40×40, 32×32,
  160×160 — [ar5iv rendering](https://ar5iv.labs.arxiv.org/html/2310.02557)
- Architectures: a **UNet with ~7.6 M parameters** (3 encoder/decoder blocks + mid block, bias-free,
  for 80×80), and a **BF-CNN with 21 conv layers, 64 channels, 3×3 kernels, ~700 k parameters** —
  [ar5iv rendering](https://ar5iv.labs.arxiv.org/html/2310.02557)
- Convergence criterion: two networks trained on **non-overlapping** subsets "learn nearly the same
  score function, and thus the same density, when the number of training images is large enough";
  the ar5iv extraction puts "nearly identical samples", with train/test PSNR essentially equal, at
  **N = 10^5** — [arXiv:2310.02557](https://arxiv.org/abs/2310.02557);
  [ar5iv rendering](https://ar5iv.labs.arxiv.org/html/2310.02557).
  A secondary summary of the same paper states generalization is reached "with large but realizable
  training sets, roughly **10,000 images or more**" —
  [PDF at NYU CNS](https://www.cns.nyu.edu/pub/lcv/kadkhodaie24a.pdf). **These two figures differ by
  10×**; treat 10^4 as the onset of strong generalization and 10^5 as the point of near-identical
  score functions, and flag the discrepancy.
- The bias is not merely data-fitting: GAHBs "arise not only when the network is trained on
  photographic images, but also when it is trained on image classes supported on low-dimensional
  manifolds for which the harmonic basis is suboptimal" — i.e. the network imposes the basis —
  [arXiv:2310.02557](https://arxiv.org/abs/2310.02557)
- Caveat on the locality reading of this paper: for 80×80 inputs the UNet's receptive field is
  ≈ **92×92**, i.e. larger than the image, and the paper does not itself attribute generalization to
  a receptive-field constraint — [ar5iv rendering](https://ar5iv.labs.arxiv.org/html/2310.02557)

**Contradicting the architectural reading — "Locality in Image Diffusion Models Emerges from Data
Statistics" (arXiv:2509.09672, 2025)**
- Central claim, explicitly against prior work: "locality in deep diffusion models emerges as a
  statistical property of the image dataset, *not* due to the inductive bias of convolutional
  neural networks" — [arXiv:2509.09672](https://arxiv.org/html/2509.09672v1)
- Evidence 1 — cross-architecture: both UNets and Diffusion Transformers (which have no explicit
  locality constraint) learn similar sensitivity patterns —
  [arXiv:2509.09672](https://arxiv.org/html/2509.09672v1)
- Evidence 2 — causal manipulation: by editing pixel correlations in CIFAR-10 to embed a "W"
  pattern, trained models adopted that **non-local** sensitivity field; "any desired pattern can be
  induced in the sensitivity of a trained neural network by embedding the pattern into the data
  covariance" — [arXiv:2509.09672](https://arxiv.org/html/2509.09672v1)
- Datasets: CIFAR-10, CelebA-HQ, MNIST, FashionMNIST, AFHQv2 —
  [arXiv:2509.09672](https://arxiv.org/html/2509.09672v1)

### Inferences
- Kamb & Ganguli and Niedoba et al. are the same claim from two directions: a hand-built *local*
  composite of training patches reproduces what a trained convolutional denoiser does, without
  training. Novelty is therefore attributable to a **restricted hypothesis class**, not to anything
  learned about semantics.
- The r² 0.95 → 0.77 drop when self-attention is present is the only direct architectural
  comparison in this literature: it says a non-local model is *less* well described by the
  patch-mosaic mechanism, not that it generalises worse. Kamb & Ganguli read attention as adding
  semantic coherence on top of the mosaic. This is not evidence that a fully non-local model
  composes novel scenes.
- 2509.09672 reframes the design question: the payoff is not "convolution" per se but **a
  representation whose data covariance is local**. If the state's covariance between hex columns
  falls off with hex distance, a local architecture matches the data; if it does not, imposing
  locality is a mismatch and (by that paper's logic) a global model would have learned the true
  non-local sensitivity anyway.

### Gaps
- Kamb & Ganguli's **training set sizes N and image resolutions** are not stated in any source I
  could read (abstract, ICML poster page, alphaXiv and Moonlight summaries); the full PDF exceeded
  the fetch size limit and the ICML slide deck did not decode to text. Presumably the standard full
  datasets (CIFAR-10 50k, MNIST/FashionMNIST 60k), but **unverified** — do not cite a number.
- The fitted **receptive-field / patch sizes** in the LS/ELS machine, and the method used to
  estimate a trained model's effective receptive field, were not recoverable.
- The **explicit combinatorial count** behind "exponentially many novel images" (a formula or a
  number) is not quantified in any extract I obtained; alphaXiv's summary confirms it is absent
  from the abstract-level text.
- Whether Kamb & Ganguli run their theory at **small N** (10^3–10^4) at all is unknown; their
  claims are about pre-trained models on standard datasets.
- No source found that tests the ELS mechanism on **non-image, non-rectangular** data.

## Q2. Conv/UNet vs MLP or DiT-style token models at small N (≤10^4–10^5)

### Takeaway
The one paper that addresses this head-on (Wang et al., "On Inductive Biases That Enable
Generalization of Diffusion Transformers", arXiv:2410.21273) finds that DiT generalization tracks
the **locality of its attention maps**, and that *injecting local attention windows* improves both
generalization and sample quality specifically **when less training data is available**. Patch
Diffusion (NeurIPS 2023) is the complementary result: making training patch-local ≥2× speeds up
training and lets models train from scratch on as few as **5,000 images**. Against that, some
sources report DiT adapting *better* than CNN backbones on small/less-diverse datasets — the
evidence is not unanimous, and much of the "transformers need more data" framing in secondary
sources is unsourced folklore.

### Cited Findings
- DiT denoisers "cannot be explained through geometry-adaptive harmonic bases" (unlike UNet
  denoisers); instead the authors find that "**locality of attention maps are closely associated
  with generalization**" — [arXiv:2410.21273](https://arxiv.org/pdf/2410.21273)
- Their intervention: inject **local attention windows** into DiT to strengthen the locality bias;
  "both the placement and the effective attention size of these local attention windows are crucial
  factors" — [arXiv:2410.21273](https://arxiv.org/pdf/2410.21273)
- Result, on CelebA, ImageNet and LSUN: strengthening locality "improved both generalization and
  generation quality **when less training data is available**" —
  [arXiv:2410.21273](https://arxiv.org/pdf/2410.21273)
- **Patch Diffusion** (Wang et al., NeurIPS 2023): patch-level training with coordinate channels
  encoding patch location and randomized patch sizes gives "≥ 2× faster training" while matching or
  beating full-image quality; FID **1.77 on CelebA-64×64** and **1.93 on AFHQv2-Wild-64×64**; and it
  "improves diffusion model training on limited data", demonstrated "with as few as **5,000 images**
  to train from scratch" — [arXiv:2304.12526](https://arxiv.org/abs/2304.12526);
  [NeurIPS 2023 proceedings PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/e4667dd0a5a54b74019b72b677ed8ec1-Paper-Conference.pdf)
- Counter-evidence in the other direction: a survey-style secondary source reports that
  "transformer backbone models adapt well to different datasets efficiently, whereas CNN backbones
  fail to produce meaningful samples on certain smaller datasets, with transformer backbones
  producing high quality results with better FID scores for less diverse cases" — this claim traces
  to DiffScaler — [DiffScaler, arXiv:2404.09976](https://arxiv.org/pdf/2404.09976)
- On Gaussian-process synthetic data, "both DiT and UNet-based diffusion models capture decay
  patterns in temporal correlation, however DiT exhibits much better learning results, matching the
  ground truth" — [arXiv:2407.16134](https://arxiv.org/pdf/2407.16134)
- Text-to-image scaling study: "increasing transformer blocks is more parameter-efficient for
  improving text-image alignment than increasing channel numbers" —
  [Li et al., CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_On_the_Scalability_of_Diffusion-based_Text-to-Image_Generation_CVPR_2024_paper.pdf)
- From-scratch DDPM on tiny datasets: DDPMs "need enough training samples (e.g. **1,000 images**) to
  synthesize diverse results and avoid replicating the training samples"; overfitting "becomes
  serious when the number of training data becomes smaller", so FID is measured periodically and the
  best checkpoint chosen — [DomainStudio, arXiv:2306.14153](https://arxiv.org/pdf/2306.14153)
- A 2025 flow-based DiT explicitly targeted at the limited-data regime exists (text-to-image on
  limited data with a flow-based diffusion transformer) — [oboro, arXiv:2511.08168](https://arxiv.org/pdf/2511.08168)

### Inferences
- The *actionable* result across 2410.21273 and Patch Diffusion is not "UNet beats DiT" but
  "locality beats globality at small N, in either architecture". A DiT with windowed/neighbourhood
  attention is the cheapest way to get the bias without leaving the token formulation — and it is
  the same change GenCast makes on a sphere (Q4).
- A token model over a **PCA latent** has no notion of neighbourhood at all, so neither the
  windowed-attention fix nor the patch-mosaic mechanism is available to it in any form. This is a
  structural statement, not an empirical one.
- The DiffScaler/GP-data counter-evidence concerns fine-tuning and synthetic correlated data
  respectively, not from-scratch unconditional generation at N≈10^4, so it does not directly
  contradict 2410.21273.

### Gaps
- I found **no** controlled study of MLP-vs-conv diffusion at fixed N on images, and none comparing
  DiT vs UNet with matched parameters and matched N ≈ 10^4 with FID reported for both. The
  "transformers need more data" claim appears in secondary/tutorial sources
  ([apxml](https://apxml.com/courses/advanced-diffusion-architectures/chapter-3-transformer-diffusion-models/unet-vs-transformer-diffusion),
  [ICLR 2026 blogpost](https://iclr-blogposts.github.io/2026/blog/2026/diffusion-architecture-evolution/))
  without a primary citation I could verify.
- 2410.21273's actual numbers (FID by training-set size, window sizes in tokens) were not
  recoverable from the abstract-level fetch; the PDF should be read for them before any number is
  quoted.
- U-ViT was not reached within the call budget; no data-efficiency numbers for it here.

## Q3. Compute actually spent on small 64×64 unconditional diffusion / flow models

### Takeaway
The reference points are brutal by T4 standards: EDM's CIFAR-10 (50k, 32×32) run is ~2 days on
**8× V100** and its 64×64 runs ~4 days on **8× V100**, i.e. roughly **380 and 770 V100-GPU-hours**.
A 1,000-image from-scratch DDPM is ~30 hours on **8× A6000** (≈240 A6000-GPU-hours) for 60k
iterations. Patch Diffusion's ≥2× training speedup and its 5,000-image from-scratch result are the
cheapest published route to the same class of output.

### Cited Findings
- EDM official training recipes: **CIFAR-10 32×32 — 8× V100, ~2 days, batch 512**, DDPM++/NCSN++
  backbone; **FFHQ 64×64 — 8× V100, ~4 days, batch 256**, `--cres=1,2,2,2 --lr=2e-4 --dropout=0.05
  --augment=0.15`; **AFHQv2 64×64 — 8× V100, ~4 days, batch 256**, `--dropout=0.25 --augment=0.15` —
  [NVlabs/edm README](https://github.com/NVlabs/edm/blob/main/README.md)
- EDM's headline CIFAR-10 result: **FID 1.79** class-conditional (1.97 unconditional per the paper);
  development used V100 and A100 GPUs — [NVlabs/edm](https://github.com/NVlabs/edm)
- 1,000-image regime: "For datasets containing 1000 images, DDPMs are typically trained for **60K
  iterations (about 30 hours on ×8 NVIDIA RTX A6000 GPUs)**"; best FID reached at 30K/60K/50K
  iterations for Babies / Sunglasses / LSUN-Church respectively —
  [DomainStudio, arXiv:2306.14153](https://arxiv.org/pdf/2306.14153)
- Patch Diffusion: **≥2× faster training**, FID 1.77 CelebA-64, 1.93 AFHQv2-Wild-64, trainable from
  scratch on **5,000 images** — [arXiv:2304.12526](https://arxiv.org/abs/2304.12526)
- Kadkhodaie's models in the N = 10^4 regime are small: **~7.6 M-parameter UNet at 80×80**, and a
  **~700 k-parameter 21-layer BF-CNN** — [ar5iv rendering of
  2310.02557](https://ar5iv.labs.arxiv.org/html/2310.02557)
- Convergence-speed lever on the *latent side*: VA-VAE + LightningDiT reach ImageNet-256 FID 2.11 in
  **64 epochs**, "an over **21× convergence speedup** compared to the original DiT" —
  [arXiv:2501.01423](https://arxiv.org/abs/2501.01423)

### Inferences
- Order-of-magnitude arithmetic (mine, from the EDM table): 8 GPUs × 48 h ≈ **384 V100-hours** for
  CIFAR-10-scale 32×32; 8 × 96 h ≈ **768 V100-hours** for 64×64. A T4 is materially slower than a
  V100 in fp32/fp16 throughput, so a single-T4 reproduction of a full EDM recipe is in the
  weeks-to-months range — not a step this project can run as specified.
- What *is* in range on one T4: the Kadkhodaie-scale model (10^5–10^7 parameters, N ≈ 10^4), which is
  1–2 orders of magnitude smaller than an EDM config, plus Patch-Diffusion-style patch training for
  the ≥2× factor. The published evidence is that this scale *does* reach strong generalization at
  N ≈ 10^4 (with the 10^4-vs-10^5 caveat in Q1).
- No source gives a T4 number for any of these, so any T4-hour estimate is an extrapolation and must
  be labelled as one.

### Gaps
- **No published FID for FFHQ-64 or AFHQv2-64 was found in the EDM README** (the table gives GPUs and
  days, not FID); Papers-with-Code tables were not fetched within the call budget.
- No source found that states **GPU-hours for a T4** on any small 64×64 diffusion run, and no
  V100→T4 conversion factor from a primary source.
- Kadkhodaie et al. give no training time / GPU / epoch numbers in anything I could extract — the
  6.6 MB PDF's content streams did not decode.
- No paper found that reports GPU-hours **as a function of N** for the same architecture, which is
  what a cost projection for N = 13,555 would actually need.

## Q4. Diffusion on hex / graph / spherical grids, and how to give a hex-column image a conv or attention bias

### Takeaway
I found **no image diffusion model trained natively on a hexagonal lattice**. The mature pattern for
non-rectangular domains is the one GenCast uses: keep the diffusion objective unchanged, and put the
locality bias in the *network* as a **graph over the fixed mesh with local (k-hop neighbourhood)
attention**. For a hex raster specifically, HexaConv (ICLR 2018) is the standard trick: store hex
data in a squeezed square layout and reuse ordinary optimized convolutions, which additionally
upgrades the symmetry group from 4-fold to 6-fold.

### Cited Findings
- **GenCast** (diffusion for medium-range weather) "maps input from the native latitude-longitude
  grid to an internal learned representation defined on a **6-times-refined icosahedral mesh**",
  combined with "a graph transformer architecture in which **each node attends to its k-hop
  neighbourhood** on the mesh", trained with "a standard diffusion model denoising objective" on 40
  years of ERA5 — [arXiv:2312.15796](https://arxiv.org/html/2312.15796v2)
- **Appa** (latent diffusion for global data assimilation) encodes lat-lon input "through
  increasingly coarser icosahedral meshes, from a 7-times refined icosahedron down to a 3-times
  refined icosahedron for the latent representation", with "graph pooling attention layers and graph
  local self-attention blocks that preserve the spherical topology" —
  [arXiv:2504.18720](https://arxiv.org/pdf/2504.18720)
- **HexaConv** (Hoogeboom et al., ICLR 2018): "efficient implementations of planar convolution and
  group convolution over hexagonal lattices can be achieved by **re-using existing highly optimized
  convolution routines**"; "square tiling provides 4-fold rotational symmetry, hexagonal tiling has
  **6-fold** rotational symmetry"; hexagonal convolution "provides better accuracy than planar
  convolution with square filters given a fixed parameter budget", and the increased symmetry
  "increases the effectiveness of group convolutions by allowing more parameter sharing" —
  [arXiv:1803.02108](https://arxiv.org/pdf/1803.02108);
  [ar5iv](https://ar5iv.labs.arxiv.org/html/1803.02108); code:
  [ehoogeboom/hexaconv](https://github.com/ehoogeboom/hexaconv)
- Rasterise-to-square vs native: hex convolution "can be implemented by rearranging hexagonally
  sampled data and hexagonal kernels, where the original input is squeezed into a square layout and
  sub-kernels are convolved with specific patches according to the hexagonal layout"
  ([arXiv:1803.02108](https://arxiv.org/pdf/1803.02108)), whereas **HexCNN** takes hexagon-shaped
  input and does native hexagonal forward/backward passes, "eliminating memory and computation
  overhead from padding operations" —
  [HexCNN, ICDM 2020](https://www.ruizhang.info/publications/ICDM2020%20HexCNN-A%20Framework%20for%20Native%20Hexagonal.pdf)
- Hex deep learning has been applied to generation before diffusion: "Biologically Inspired
  Hexagonal Deep Learning for Hexagonal Image Generation" —
  [ResearchGate record](https://www.researchgate.net/publication/347627930_Biologically_Inspired_Hexagonal_Deep_Learning_for_Hexagonal_Image_Generation)
- Diffusion on other fixed non-rectangular domains: a **cortical-surface** diffusion generative
  model over neurodevelopmental trajectories —
  [arXiv:2508.03706](https://arxiv.org/pdf/2508.03706); graph diffusion methods surveyed in
  [IJCAI 2023 survey](https://www.ijcai.org/proceedings/2023/0751.pdf)
- Hexahedral-mesh diffusion work exists (DDPM-Polycube,
  [arXiv:2503.13541](https://arxiv.org/pdf/2503.13541); PolycubeNet,
  [arXiv:2605.20274](https://arxiv.org/pdf/2605.20274)) but this is *hexahedral mesh geometry*, not
  a hexagonal image lattice — do not cite it for hex-image diffusion.

### Inferences
- Two defensible routes for 721 hex columns, both supported above: (a) **rasterise** the hex grid
  into the squeezed-square layout of HexaConv and run an ordinary small conv UNet — cheapest, reuses
  every optimized kernel, and it is exactly the trick HexaConv was written for; (b) treat the 721
  columns as **nodes with a fixed 6-neighbour adjacency** and use local graph attention, the GenCast
  pattern, which is the published precedent for a diffusion objective on a non-rectangular fixed
  mesh.
- The Q2 finding (windowed attention rescues DiT generalization at small N) and the GenCast design
  (k-hop neighbourhood attention) are the same architectural move. The minimal change to the
  project's existing flow-matching transformer is therefore to give its tokens hex positions and
  restrict attention to a hex neighbourhood — not necessarily to switch to a UNet.
- 721 columns ≈ a 27×27 raster; that is between MNIST (28×28) and CIFAR-10 (32×32) in spatial
  extent, the exact regime the Q1 papers measured.

### Gaps
- **No hexagonal-lattice image diffusion paper found.** Searches returned hex *convolution* work and
  hexahedral *mesh* diffusion, nothing in between. This looks like a genuine gap in the literature
  rather than a search failure, but one round of searching cannot establish that.
- No source compares **rasterise-to-square vs native hex/graph conv inside a diffusion model**, so
  there is no evidence on which of routes (a) and (b) generates better samples.
- No source measured the cost of hex-to-square rasterisation artifacts (the axial shear) on
  generative quality.

## Q5. Many channels per site vs spatial extent

### Takeaway
The consistent finding in the latent-diffusion literature is that **raising per-site channel/feature
dimension improves reconstruction but makes generation harder**: it "requires substantially larger
diffusion models and more training iterations to achieve comparable generation performance", and at
least one video VAE reports FVD getting *worse* when latent channels were raised from a smaller count
to 16 despite reconstruction improving. Nobody in the sources studied 128 channels per site as an
image-like generative target.

### Cited Findings
- The optimization dilemma, stated directly: "while increasing the per-token feature dimension in
  visual tokenizers improves reconstruction quality, it requires substantially larger diffusion
  models and more training iterations to achieve comparable generation performance"; the authors
  attribute it to "the inherent difficulty in learning unconstrained high-dimensional latent spaces"
  — [Yao et al., CVPR 2025, arXiv:2501.01423](https://arxiv.org/abs/2501.01423)
- Their fix is a **constraint on the latent space**, not more capacity: aligning the tokenizer's
  latent space with a pre-trained vision foundation model (VA-VAE), which "reduces the need for
  massive parameter counts in generative models of high-dimensional tokenizers"; with
  LightningDiT this gives ImageNet-256 **FID 1.35**, and **FID 2.11 in 64 epochs (>21× convergence
  speedup over original DiT)** — [arXiv:2501.01423](https://arxiv.org/abs/2501.01423);
  [CVPR 2025 PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Yao_Reconstruction_vs._Generation_Taming_Optimization_Dilemma_in_Latent_Diffusion_Models_CVPR_2025_paper.pdf)
- Concrete regression when channels were raised: "increasing the latent channels of LeanVAE to 16,
  while significantly improving video reconstruction, does not yield corresponding gains in
  generation performance. The FVD scores actually increased by **45.56 and 10.88**" —
  [LeanVAE, arXiv:2503.14325](https://arxiv.org/pdf/2503.14325)
- The reconstruction side of the same trade: "reconstruction performance improves significantly as
  the number of latent channels increases. However, larger latent channels may increase convergence
  difficulty in training diffusion models" — [LeanVAE, arXiv:2503.14325](https://arxiv.org/pdf/2503.14325)
- Channel counts in practice: SD/SDXL VAEs use **4 channels** at 8× downsampling (48× overall
  compression), FLUX uses **16 channels** at ~12× compression, preserving more detail —
  [LCUDiff, arXiv:2602.04406](https://arxiv.org/pdf/2602.04406) (secondary, but consistent with the
  primary sources above)
- A 4-channel latent "introduces a strong information bottleneck for restoration tasks such as
  super-resolution... sufficient for generative image synthesis" but limiting for fine texture —
  [TOC-SR, arXiv:2605.02767](https://arxiv.org/pdf/2605.02767)
- Parameter-efficiency direction in a scaling study: "increasing transformer blocks is more
  parameter-efficient for improving text-image alignment than increasing channel numbers" —
  [Li et al., CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_On_the_Scalability_of_Diffusion-based_Text-to-Image_Generation_CVPR_2024_paper.pdf)

### Inferences
- The 128-channels-per-column state is far outside the 4–16 channel range where all published
  evidence sits, and on the wrong side of the trade-off these papers describe. The one published
  remedy that does not require more compute is **constraining the per-site latent** (VA-VAE's move),
  i.e. compressing or whitening the 8×16 channel stack per column while *keeping* the 721-column
  spatial layout — a per-site channel reduction rather than the project's current global PCA over all
  92,288 dimensions, which destroys the layout too.
- The evidence is about learned VAE latents whose dimensions are unconstrained; the fly state's
  channels are DCT-in-time × cell type, which are interpretable and likely far from isotropic. None
  of the cited papers covers that case, so the transfer is a hypothesis.

### Gaps
- **No paper found that studies diffusion with ≳32 channels per spatial site as an image-like
  target**, nor any "channel-wise DiT" or "video-as-channels at 64×64" study. Searches surfaced only
  VAE-latent channel counts.
- No source quantifies how generative difficulty scales with channels at *fixed* spatial extent
  (e.g. FID vs C for C = 4, 8, 16, 32, 64) — the LeanVAE datapoint is a single pair.
- No evidence on whether time-frequency channels (DCT coefficients) behave like VAE channels.

## What this predicts for a spatial prior over 721 hex columns with N = 13,555

Strictly within the sources above.

1. **The present PCA-token flow has no access to the published mechanism of novelty.** Kamb &
   Ganguli's account requires locality *and* equivariance over a spatial domain to prevent optimal
   score-matching ([arXiv:2412.20292](https://arxiv.org/abs/2412.20292)); Niedoba et al.'s account
   requires local patch denoisers ([arXiv:2411.19339](https://arxiv.org/abs/2411.19339)). A 16-token
   whitened-PCA latent has no neighbourhood structure, so neither mechanism is definable on it. This
   is a structural argument, not a measured comparison: no source tested a PCA-latent diffusion
   model against a spatial one.
2. **N = 13,555 sits exactly at the published onset of generalization, not comfortably past it.**
   Kadkhodaie et al. ran N = 1…10^5 at 80×80 and report strong generalization "roughly 10,000 images
   or more" in one summary and near-identical scores at N = 10^5 in the paper text
   ([NYU PDF](https://www.cns.nyu.edu/pub/lcv/kadkhodaie24a.pdf);
   [ar5iv](https://ar5iv.labs.arxiv.org/html/2310.02557)). At 13,555 a spatial model is plausibly
   inside the generalizing regime and a memorization check is mandatory, not optional.
3. **Model size should be Kadkhodaie-scale, not EDM-scale.** The published N ≈ 10^4 experiments used
   a ~7.6 M-parameter UNet at 80×80 and a ~700 k-parameter BF-CNN
   ([ar5iv](https://ar5iv.labs.arxiv.org/html/2310.02557)). EDM's 64×64 configs cost ~8× V100 × 4
   days ([NVlabs/edm README](https://github.com/NVlabs/edm/blob/main/README.md)) and are not a
   feasible target on one T4. 721 columns ≈ a 27×27 raster, i.e. MNIST-to-CIFAR spatial extent — the
   regime those small models were measured in.
4. **If the token formulation is kept, restrict attention to a hex neighbourhood.** That is precisely
   the intervention shown to improve generalization *and* quality when training data is scarce
   ([arXiv:2410.21273](https://arxiv.org/pdf/2410.21273)), and the same design GenCast uses to run a
   diffusion objective on a non-rectangular fixed mesh with k-hop node attention
   ([arXiv:2312.15796](https://arxiv.org/html/2312.15796v2)). The published earlier attempt with 721
   columns as *globally attending* tokens is not a test of the spatial hypothesis.
5. **A hex raster is the cheap implementation.** HexaConv's squeezed-square layout reuses standard
   convolution kernels and raises the symmetry group from 4- to 6-fold
   ([arXiv:1803.02108](https://arxiv.org/pdf/1803.02108)), which is the correct equivariance group
   for the ommatidial lattice. Note that equivariance over a 721-column disc is only approximate:
   the retina has a boundary, and the ELS theory assumes translation equivariance.
6. **128 channels per column is the main untested risk.** Every cited datapoint says raising per-site
   dimension makes generation harder even as reconstruction improves
   ([arXiv:2501.01423](https://arxiv.org/abs/2501.01423);
   [arXiv:2503.14325](https://arxiv.org/pdf/2503.14325)), with published practice at 4–16 channels.
   The supported move is a *per-column* channel compression that preserves the hex layout, in place
   of a global PCA over all 92,288 dimensions.
7. **One measurement decides whether locality is the right bias at all.** Per
   [arXiv:2509.09672](https://arxiv.org/html/2509.09672v1), a trained denoiser's sensitivity field
   follows the **data covariance**. So: compute the covariance of the state across column pairs as a
   function of hex distance. If it decays with distance, a local architecture matches the data and
   items 4–5 are justified; if it does not, imposing locality is a mismatch and that paper predicts
   a global model would have learned the true non-local field anyway. This is a cheap CPU
   measurement on the existing 13,555 states, and no source substitutes for it.
