# How autoencoders/VAEs are trained to reconstruct well — the engineering

Research request 2026-09-21. Scope: training recipe engineering (steps, batch,
lr, loss composition, architecture, capacity vs PCA), not the theory. Setting
to make it applicable to: a 3.89M-parameter transformer autoencoder, width
192, depth 4, 4 heads, 721 tokens/sample, AdamW lr 1e-3 cosine, batch 32,
8,000–20,000 steps on 13,555 training samples (~19–47 epochs), fp16 autocast,
EMA. This project (Fly_Brain) is not itself researched here — findings are
from named external systems only.

Every number below is cited to the source it came from. "Measured ablation"
vs "common practice / unverified secondary account" is marked per item.

---

## 1. How long do working autoencoders actually train — in samples seen

| System | Steps | Batch | Epochs | Dataset size | Samples seen | Source |
|---|---|---|---|---|---|---|
| SD `kl-f8` VAE, **ft-EMA** stage (fine-tune of the original OpenImages-trained autoencoder) | 313,198 steps | 192 (16×A100, 12/GPU) | — | LAION-Aesthetics + LAION-Humans, 1:1 (size undisclosed) | ≈ **60.1M samples** | [stabilityai/sd-vae-ft-mse-original README](https://huggingface.co/stabilityai/sd-vae-ft-mse-original) |
| SD `kl-f8` VAE, **ft-MSE** stage (continued from ft-EMA) | +280,000 steps | 192 | — | same | ≈ **53.8M more samples** (≈114M cumulative over both fine-tune stages) | same |
| SD `kl-f8` VAE, **original** base config (`configs/first_stage_models/kl-f8/config.yaml`) | not stated in config; `base_learning_rate: 4.5e-6`, `batch_size: 4` (per-GPU, LDM convention: effective lr = `base_lr × n_gpu × batch_size × accumulate_grad_batches`) | 4/GPU | — | OpenImages, 384px crops → 256 | — | [CompVis/latent-diffusion config.yaml](https://raw.githubusercontent.com/CompVis/latent-diffusion/main/models/first_stage_models/kl-f8/config.yaml) |
| VQGAN (Taming Transformers), first-stage tokenizer, ImageNet config (`configs/imagenet_vqgan.yaml`) | `disc_start: 250001` (discriminator turns on after 250K steps, implying total run is well beyond that) | 12/GPU | — | ImageNet (~1.28M train images) | ≥ 3M+ (unspecified total) | [CompVis/taming-transformers config.yaml](https://raw.githubusercontent.com/CompVis/taming-transformers/master/configs/imagenet_vqgan.yaml) |
| VQGAN, largest class-conditional ImageNet **transformer** (second stage, not the autoencoder) | 2.4M steps, batch 16 × 8 grad-accum = effective 128 | 128 (effective) | — | ImageNet | ≈ **307M samples**; took 45.8 days on 1×A100 | [Esser et al. 2021, "Taming Transformers", Appendix B / Table 8](https://ar5iv.labs.arxiv.org/html/2012.09841) — **note: this row is the autoregressive transformer stage, not the VQGAN tokenizer itself; the paper does not publish first-stage (tokenizer) step/epoch counts in the text, only in the released configs above** |
| MAGVIT-2, **image tokenizer** (Stage I) | — | 256 | 10 epochs | OpenImages, 8M images | **80M samples** | [Yu et al. 2024, "Language Model Beats Diffusion", arXiv:2310.05737](https://arxiv.org/html/2310.05737v2) |
| MAGVIT-2, **video tokenizer** (Stage II) | — | 128 | 1 epoch | 9.3M clips sampled from Panda-70M | **9.3M clips** (17 frames, 128×128, stride 1) | same |
| MAGVIT-2, tokenizer trained for the paper's headline Kinetics-600 numbers | — | 256 | **190 epochs** on Kinetics-600 | Kinetics-600 (~390K train clips) | ≈ **74M clips** | same |
| aMUSEd (Open-MUSE reproduction), 256px **masked transformer** (not the VQGAN tokenizer — Open-MUSE reused a pretrained MaskGIT-VQGAN, `openMUSE/maskgit-vqgan-imagenet-f16-256`, and never shipped its own tokenizer training script) | 1,000,000 steps | 2,048 (2×8×A100, 128/GPU) | — | — | ≈ **2.05B samples** | [huggingface/open-muse README](https://raw.githubusercontent.com/huggingface/open-muse/main/README.md) |
| l-DeTok / "Latent Denoising Makes Good Tokenizers" (ViT-B/ViT-B tokenizer, 171.7M params) — **measured ablation study of tokenizer training itself, useful as a controlled small-scale reference** | — | 1,024 global | ablations: 50 epochs (no GAN); final: **200 epochs**, GAN enabled from epoch 100 | ImageNet-256 (~1.28M images) | ablation run ≈ **64M images**; final run ≈ **256M images** | [Chen et al., "Latent Denoising Makes Good Tokenizers", arXiv:2507.15856](https://ar5iv.labs.arxiv.org/html/2507.15856) |
| "Sample what you can't compress" (SWYCC) tokenizer, MaskGIT-style conv encoder + diffusion-refiner decoder | 10⁶ steps | 256 | ≈200 epochs | ImageNet-256 | ≈ **256M samples** | [arXiv:2409.02529](https://ar5iv.labs.arxiv.org/html/2409.02529) |
| REPA-E end-to-end VAE tuning (SiT-XL + VAE jointly) — **measured ablation of how fast a VAE adapts** | 400K steps (80 "epochs" in their notation) | 256 | 80 | ImageNet-256 | ≈ **102M samples**, already beating a VAE-frozen baseline run 10× longer (800 epochs / 5.90 FID vs their 4.07 FID) | [Leng et al., "REPA-E", arXiv:2504.10483](https://arxiv.org/html/2504.10483v1) |

**Reading across these**: every named system that trains an autoencoder/tokenizer
component from scratch runs it for tens to low-hundreds of millions of samples
seen (80M–300M+ is the common range; SD's VAE fine-tune alone is ~114M samples
on top of an already-converged OpenImages base). The project setting in this
request (8,000–20,000 steps × batch 32 = **256K–640K samples seen**, on a
13,555-sample dataset repeated 19–47 times) is one to three **orders of
magnitude** below every named reference's sample budget — even below the
*fine-tuning-only* budgets (SD's ft-EMA stage alone is ~100× more samples
seen). None of the references train an autoencoder-class model on a dataset
this small for this few total gradient updates; the closest analogue (REPA-E,
102M samples, already the fastest-converging case found) is still ~200–400×
more samples seen. This is a straightforward "samples seen" comparison, not a
claim about what should happen when the input/architecture differ from these
systems (all are conv/ViT image or video tokenizers, not this project's
domain) — treat it as scale context, not as a verdict.

---

## 2. Batch size and LR schedules — standard practice and known sensitivity

**Common practice (not necessarily ablated) across the named systems:**

- SD VAE fine-tune: `base_learning_rate = 4.5e-6` at effective batch 192 (16 GPUs × 12/GPU) — [HF README](https://huggingface.co/stabilityai/sd-vae-ft-mse-original), [CompVis config](https://raw.githubusercontent.com/CompVis/latent-diffusion/main/models/first_stage_models/kl-f8/config.yaml).
- Taming-Transformers VQGAN (ImageNet config): `base_learning_rate = 4.5e-6`, batch 12/GPU — [config.yaml](https://raw.githubusercontent.com/CompVis/taming-transformers/master/configs/imagenet_vqgan.yaml). LDM/taming-transformers' `train.py` convention multiplies this base LR by `n_gpu × batch_size × accumulate_grad_batches` to get the actual optimizer LR — so the realized LR at batch 192 works out to a low-4-digit ×10⁻⁴ regime, not 4.5e-6 itself. (Common practice / documented convention, not a separate ablation.)
- MAGVIT-2 tokenizer: peak LR **1×10⁻⁴**, linear warmup then cosine decay, Adam with β₁=0, β₂=0.99, batch 256, **EMA decay 0.999** — [arXiv:2310.05737](https://arxiv.org/html/2310.05737v2).
- l-DeTok: peak LR **4×10⁻⁴** at batch 1024, linearly scaled from a 1×10⁻⁴-at-batch-256 base rate (i.e. linear LR-batch scaling used explicitly), warmup for 25% of training then cosine decay, AdamW β=(0.9, 0.95) — [arXiv:2507.15856](https://ar5iv.labs.arxiv.org/html/2507.15856).
- SWYCC tokenizer: LR warmed up over 10⁴ steps from 0 to 1×10⁻⁴, then cosine-decayed to 0, batch 256, global grad-norm clip 1 — [arXiv:2409.02529](https://ar5iv.labs.arxiv.org/html/2409.02529).
- REPA-E: constant LR **1×10⁻⁴**, AdamW, batch 256 (no schedule at all, and it still converges within 400K steps) — [arXiv:2504.10483](https://arxiv.org/html/2504.10483v1).

**Pattern**: every named system's *effective* (post batch-scaling) autoencoder
learning rate clusters in the **1×10⁻⁴ to 4×10⁻⁴** range at batch sizes of
192–1024, with warmup + cosine (or constant) decay — not 1e-3. This project's
setting (lr 1e-3, batch 32, cosine) is roughly 3–10× higher than the common
autoencoder-training LR band found here, even accounting for the smaller batch
size (linear LR-batch scaling from l-DeTok's own stated rule — 1e-4 at batch
256 — would put batch-32 at ≈1.25e-5, not 1e-3).

**Dedicated LR/batch ablation for autoencoder reconstruction quality**: not
found. No named source runs a controlled sweep of learning rate or batch size
against final reconstruction quality for an image/video autoencoder — this is
a **gap** (see below). The generic ML-grid-search hits from this search
(batch 32/64/128/256, lr 1e-2..1e-4 on small anomaly-detection or synthetic
autoencoders) are not from named, citable production systems and are excluded
per the "no generic deep-learning advice" instruction.

---

## 3. Architectural details that are load-bearing (measured ablations)

**MAGVIT-2, Table 5 (ImageNet 128×128), the only clean incremental architecture
ablation found with reconstruction-quality numbers** — [arXiv:2310.05737](https://arxiv.org/html/2310.05737v2):

| Cumulative modification | FID↓ | LPIPS↓ |
|---|---|---|
| MAGVIT (baseline) | 2.65 | 0.1292 |
| + Lookup-Free Quantization (LFQ) | 2.48 | 0.1182 |
| + Large vocabulary | 1.34 | 0.0821 |
| + improved up/downsampler | 1.21 | 0.0790 |
| + deeper model | 1.20 | 0.0686 |
| + adaptive (group) normalization | 1.15 | 0.0685 |

This is a **measured ablation**: depth ("deeper model") and the up/downsampler
design each bought a comparable or larger LPIPS gain than the normalization
change; adaptive GroupNorm was the last, smallest increment, not the biggest.
Architecture choices named as load-bearing here: strided-conv downsampling,
depth-to-space upsampling, adaptive group norm, and — separately, the biggest
single jump in the table — vocabulary/codebook size.

**SWYCC** (measured, but reported qualitatively / by CMMD deltas, not a
by-component sweep table) — [arXiv:2409.02529](https://ar5iv.labs.arxiv.org/html/2409.02529):
their encoder uses **GroupNorm** (not LayerNorm) with GeLU, standard ResNet
blocks; their decoder is explicitly **split into a deterministic
initial-reconstruction head and a heavier stochastic diffusion-refiner** — an
explicit decoder-heavier-than-encoder asymmetry, which they describe as
"critical for training dynamics." They report perceptual loss (VGG-feature L2)
was "particularly important to be competitive" — see loss table in §4.

**EMA**: MAGVIT-2 uses EMA decay 0.999 as standard practice (not separately
ablated in the fetched material) — [arXiv:2310.05737](https://arxiv.org/html/2310.05737v2).
REPA-E also applies EMA to the generative/VAE-joint model for "stable
optimization" (stated as practice, not ablated) — [arXiv:2504.10483](https://arxiv.org/html/2504.10483v1).
The SD VAE ships two public checkpoints from the *same* fine-tune data,
differing only in loss weighting (ft-EMA vs ft-MSE), and both are described as
using EMA weights for the released checkpoint — [HF README](https://huggingface.co/stabilityai/sd-vae-ft-mse-original).
No source in this search runs an EMA-on vs EMA-off ablation with numbers; this
is common practice, not a measured ablation, for the autoencoder case
specifically (EMA's benefit is measured extensively for diffusion/GAN
generators, not for AE/VAE reconstruction — see Gaps).

**Codebook/bottleneck dimension** (practitioner account, not a paper ablation):
in `lucidrains/vector-quantize-pytorch` issue #69, a user reports that
increasing the codebook dimension from 128×256 to 256×256 visibly reduced
blurriness of VQ reconstructions — [GitHub issue #69](https://github.com/lucidrains/vector-quantize-pytorch/issues/69).
Issue #35 in the same repo reports an L2-distance codebook failing to converge
where a cosine-similarity codebook did — [GitHub issue #35](https://github.com/lucidrains/vector-quantize-pytorch/issues/35).
`lucidrains/magvit2-pytorch` issue #4 reports LFQ (lookup-free quantization)
training instability that was resolved for that user by switching to classical
VQ — [GitHub issue #4](https://github.com/lucidrains/magvit2-pytorch/issues/4).
These are single-user anecdotes on an unofficial reimplementation, not
controlled ablations — treated as practitioner accounts only.

---

## 4. Loss composition beyond MSE

**MSE-only autoencoders are blurry — stated directly in the VQGAN paper**:
"the textures produced by the VQVAE [L2/MSE-trained] are blurry, whereas those
of the VQGAN are crisp and realistic" — [Esser et al. 2021, §C](https://ar5iv.labs.arxiv.org/html/2012.09841).
VQGAN's reconstruction FID beat VQVAE-2's (~7.94 vs ~10 on ImageNet, per the
paper's own comparison table) by adding perceptual + adversarial loss on top
of the same reconstruction target — same source. Loss = `L_VQ(E,G,Z) +
λ·L_GAN`, with λ computed adaptively from the ratio of reconstruction-loss to
GAN-loss gradients at the last decoder layer, and **λ = 0 during an initial
warm-up phase** — "we found that longer warm-ups generally lead to better
reconstructions" (measured/empirical statement in the paper) — same source.

**SWYCC gives a rare quantitative loss-ablation table** (CMMD, lower is
better) — [arXiv:2409.02529](https://ar5iv.labs.arxiv.org/html/2409.02529):

| Loss config | CMMD↓ |
|---|---|
| diffusion loss alone | 0.43 |
| + MSE on the deterministic decoder head | 0.32 (−26%) |
| + perceptual loss (λ_percep=0.1, λ_mse=1) | 0.15 (−65% vs diffusion-alone) |

Their own framing: perceptual loss was "particularly important to be
competitive" with GAN-based methods, and they note auxiliary losses mainly
*accelerate convergence* rather than changing the theoretical final optimum —
i.e., MSE + perceptual got most of the practical benefit that adversarial loss
usually supplies, without a discriminator.

**Cosmos-adjacent / l-DeTok loss weights** (stated recipe, not an ablation of
the weights themselves): `λ_KL=1e-6`, `λ_perceptual=1.0`, `λ_GAN=0.1`, on top
of MSE — [arXiv:2507.15856](https://ar5iv.labs.arxiv.org/html/2507.15856). The
SD VAE fine-tune stages differ *only* in this loss recipe: ft-EMA uses **L1 +
LPIPS**, ft-MSE (continuing from ft-EMA) switches to **MSE + 0.1·LPIPS** — same
underlying data and step-neighbourhood, different reconstruction-loss term —
[HF README](https://huggingface.co/stabilityai/sd-vae-ft-mse-original). KL
weight in the original LDM `kl-f8` config is **1e-6** — [config.yaml](https://raw.githubusercontent.com/CompVis/latent-diffusion/main/models/first_stage_models/kl-f8/config.yaml)
— i.e. the KL term is kept small relative to reconstruction across every
KL-regularized autoencoder found here (1e-6, matching l-DeTok's 1e-6).

**When is plain MSE said to be "enough"**: not found as an explicit claim from
any named system in this search — every named working image/video tokenizer
in this list uses at least MSE/L1 + perceptual, and the higher-fidelity ones
add adversarial loss on top. This is worth flagging as a **gap**: no source
found here argues MSE alone is sufficient for pixel-level reconstruction at
production quality; the uniform practice across every cited system is to add
perceptual loss at minimum.

---

## 5. When an autoencoder underperforms PCA at the same latent dimension

**Direct measured example found** (a practitioner reproduction, not a peer-reviewed
paper, but it is a controlled head-to-head with numbers) — [666mhy666/image-representation-learning](https://github.com/666mhy666/image-representation-learning):
with 64 latent dims, 10 training epochs, 8,000 train / 1,000 val / 1,000 test
images, held-out pixel MSE was **PCA 0.00750, AE 0.01677, VAE 0.03600** — PCA
best, plain AE 2.2× worse, VAE 4.8× worse. The repo's own write-up does not
propose or test a fix; it only notes the metric (pixel MSE) doesn't capture
perceptual quality. Treated as a **documented instance of the failure mode**,
not as a validated diagnosis-and-fix case study — no confirmed "AE was fixed
and then beat PCA" thread was found in this search (see Gaps).

**Formal treatment of the capacity question** — Recovery of Linear Components
(RLC) paper — [arXiv:2012.07543](https://ar5iv.labs.arxiv.org/html/2012.07543):
under a matched-parameter-budget comparison (`θ_SAE = θ_RLC`, i.e. giving a
standard symmetric autoencoder (SAE) the same weight count as an asymmetric
linear-encoder/nonlinear-decoder design (RLC)), the paper reports the
**standard symmetric autoencoder can only match plain linear PCA**, while the
asymmetric RLC design is "substantially superior" at the *same* parameter
count. Their stated reading: **architecture (specifically, how the parameter
budget is split between encoder and decoder, and whether the encoder is
forced through an unnecessary nonlinearity) matters more than raw parameter
count** for beating PCA at a fixed bottleneck width. This is the closest
finding in this search to a rule connecting "an AE needs architecture X, not
just more capacity, to beat PCA."

**What practitioners are reported to check first, synthesized from the
sources above (not a single citable checklist — flagged as inference)**:
whether the bottleneck forces an unwarranted nonlinearity where a linear map
would suffice (RLC paper); whether loss is pixel-MSE-only, which is known to
be blurry/low-signal relative to perceptual loss (VQGAN, SWYCC); and — from
the "samples seen" comparison in §1 — whether the run is simply far short of
the sample budget every named working system uses (tens to hundreds of
millions of samples seen), since none of the sources here report a converged,
production-quality autoencoder trained for under ~50M samples.

---

## 6. Capacity rules of thumb: parameters vs latent dim vs input dim

**No explicit formula found** relating parameter count, latent dimension, and
input dimension to "how many parameters are needed to beat a k-dim linear
projection." The RLC paper (§5 above) is the closest — it shows *at matched
parameter count*, architecture choice (asymmetric linear-encode/nonlinear-decode
vs symmetric AE) is what separates beating PCA from merely matching it; it
does not give a minimum-parameters-per-latent-dim number. This is a **gap**
(below).

---

## Gaps

- **No dedicated LR/batch-size ablation for autoencoder reconstruction
  quality** was found from any named, citable system. Every source states its
  own recipe (§2) but none publishes a sweep showing sensitivity. The
  "effective LR clusters at 1e-4–4e-4" observation in §2 is a pattern across
  recipes, not a measured sensitivity curve.
- **No formal parameter-count-vs-latent-dim rule of thumb** ("k latent dims
  needs at least N parameters to beat PCA") was found in any source. The RLC
  paper is suggestive (architecture > raw count at matched budget) but doesn't
  give a numeric guideline.
- **No confirmed "AE lost to PCA, root-caused, and fixed" narrative thread**
  (GitHub issue, forum post, or paper) was found — only the one controlled
  PCA/AE/VAE numeric comparison (§5), which does not include a fix.
- **VQGAN's own first-stage (tokenizer) training length in steps/epochs is
  not published in the paper text** — only in the released `config.yaml`
  (batch size, LR, disc-start step). The 2.4M-step/45.8-day figure in the
  paper's Appendix B is for the second-stage transformer, not the VQGAN
  autoencoder itself; this note's table (§1) flags that row accordingly so it
  isn't mistaken for a tokenizer-training number.
- **EMA's effect on autoencoder/VAE reconstruction specifically** (as opposed
  to diffusion/GAN generator training, where it's well studied) is not
  measured in any source found here — every mention is "we use EMA" as stated
  practice, never an EMA-on-vs-off reconstruction-quality comparison.
- **No source explicitly argues plain MSE is sufficient** for production
  image/video autoencoder reconstruction; this may reflect a genuine practice
  gap (nobody ships MSE-only) or may just reflect what this search's queries
  surfaced — worth a targeted follow-up search on "ablation LPIPS vs MSE only"
  specifically in the SD-VAE / VQGAN GitHub issue trackers if this question
  becomes load-bearing.
- **Open-MUSE / aMUSEd tokenizer's own training recipe was not found** — the
  repository reuses a pretrained MaskGIT-VQGAN and never shipped its own
  tokenizer training script (confirmed from the repo README itself); the
  1M-step/batch-2048 figure attributed to aMUSEd in this note is for the
  masked *transformer* stage, not a tokenizer, and is kept in the table only
  as context, clearly labeled.
- **NVIDIA Cosmos Tokenizer's exact training hyperparameters (batch size, LR,
  step count) were not recovered** — an earlier search result mis-attributed
  ViT-B/ViT-S, 200-epoch, λ_KL/λ_percep/λ_GAN numbers to "Cosmos" when they in
  fact belong to a different paper (l-DeTok, arXiv:2507.15856, corrected in
  §1/§2/§4 above). The real Cosmos Tokenizer report (arXiv:2501.03575) was
  reached but its appendix hyperparameters were not extractable through the
  tools available in this session; flagged rather than guessed.
