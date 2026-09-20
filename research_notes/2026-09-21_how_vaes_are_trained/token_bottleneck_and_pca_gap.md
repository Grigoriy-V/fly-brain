# Token-wise bottlenecks in transformer autoencoders, and the AE-loses-to-PCA gap

Research notes, 2026-09-21. Literature only — nothing here is a measurement of
this project's model. Every claim carries a source URL and the latent shape it
refers to. Sections are marked **[measured]** (a number from an ablation or
results table in the source) or **[opinion/theory]** (an argument, a proof, or
a design statement without a controlled number).

The setting these notes are aimed at (stated by the requester, not verified
here): 721 tokens x 128 features in (= 92,288 numbers), encoder 4 transformer
blocks of width 192, bottleneck 3 numbers *per token* (721 x 3 = 2,163),
mirror-image transformer decoder, 3.89 M parameters total; a global linear PCA
with 2,048 components reconstructs substantially better.

---

## 0. Two arithmetic facts worth fixing first

- **Compression ratio.** 92,288 -> 2,163 is 42.7x overall, and 128 -> 3 = 42.7x
  per token. This ratio is *not* unusual. The Stable Diffusion KL-f8 VAE
  compresses an 8x8 RGB patch (192 numbers) into 4 latent channels per
  position, 8*8*3/4 = **48x**, latent 4x32x32 for a 256x256 image.
  (https://arxiv.org/pdf/2510.04961 — SSDD, which states the f8c4 ratio
  explicitly; original model: https://arxiv.org/abs/2112.10752)
  FlatDINO compresses DINOv2 patch features 256x768 (196,608) into 32x128
  (4,096) = **48x**. (https://arxiv.org/abs/2602.04873)
  So a 42.7x per-token squeeze is squarely inside the regime that is known to
  work — *at other parameter budgets*.

- **Parameter budget.** The comparison against PCA-2048 is not a fair fight in
  parameters. A rank-2048 basis over 92,288 dimensions holds ~189 M free
  numbers (the requester's own figure), fitted in closed form to the global
  optimum of the squared-error objective. The autoencoder has 3.89 M
  parameters fitted by SGD. That is a ~48x smaller model being asked to beat
  the exact optimum of a larger one. **[opinion]** This is the single largest
  difference between the two sides of the comparison and should be ruled in or
  out before anything about "bottleneck shape" is concluded.

  For scale, every transformer autoencoder in this literature that achieves
  good reconstruction at comparable compression is 20 M – 400 M parameters:
  TiTok-S/B/L = 22 M / 86 M / 307 M (https://arxiv.org/abs/2406.07550);
  FlatDINO = ViT-B encoder + ViT-L decoder, ~390 M
  (https://arxiv.org/abs/2602.04873); SD KL-f8 VAE = ~81 M, 42 M encoder +
  39 M decoder (https://gist.github.com/madebyollin/ff6aeadf27b2edbc51d05d5f97a595d9).

---

## 1. Is "a fixed, uniform latent width per token" standard?

**Yes — it is the dominant design in production tokenisers.** It is what every
convolutional VAE does (a fixed channel count at every spatial position) and
what most ViT tokenisers do.

Concrete per-position widths actually shipped:

| System | Latent shape | Per-token/position width | Source |
|---|---|---|---|
| SD / LDM KL-f8 (SD1.x, SD2) | 4 x 32 x 32 for 256px | **4 channels per 8x8 patch** | https://arxiv.org/abs/2112.10752 |
| SDXL / SD3-class VAEs | 16 channels at f8 | **16** | https://arxiv.org/abs/2403.03206 |
| ViT-VQGAN "factorized codes" | 32x32 tokens | **32-d or 8-d** code vector (projected down from 768-d) | https://arxiv.org/abs/2110.04627 |
| MAGVIT-v2 (LFQ) | 5 x 16 x 16 latent for 17x128x128 video | **~18 binary dims** per token (vocabulary 2^18) | https://arxiv.org/abs/2310.05737 |
| TiTok | 32 / 64 / 128 1D tokens | **16 channels**, codebook 4,096 | https://arxiv.org/abs/2406.07550 |
| DC-AE | f32c32, f64c128, f128c512 | 32 / 128 / 512 | https://arxiv.org/abs/2410.10733 |
| Perceiver IO multimodal AE | 784 (or 392, 196) latents | **512 channels** | https://arxiv.org/abs/2107.14795 |

**[measured]** ViT-VQGAN found that *narrowing* the per-token code helps:
projecting the encoder's 768-d output down to a 32-d or 8-d vector for codebook
lookup "consistently achieves better reconstruction quality" and greatly
improves codebook usage, versus a 256-d code. ImageNet 256x256 FID 4.17 / IS
175.1 vs vanilla VQGAN 17.04 / 70.6.
(https://arxiv.org/abs/2110.04627, https://research.google/blog/vector-quantized-image-modeling-with-improved-vqgan/)
So narrowness per token is not inherently the enemy.

**[measured]** FlatDINO ablates exactly the tokens-vs-channels trade at fixed
total latent size (2,048 dims), compressing 256x768 DINOv2 features, rFID
against DINOv2 patch features:

| Tokens | Width | Total | rFID |
|---|---|---|---|
| 16 | 128 | 2048 | 1.19 |
| 32 | 64 | 2048 | 0.97 |
| 64 | 32 | 2048 | **0.96** |

Their conclusion, quoted: "for a given total latent dimensionality, allocating
capacity to more tokens rather than larger per-token features yields better
reconstruction." (https://arxiv.org/abs/2602.04873 — *Laminating Representation
Autoencoders for Efficient Diffusion*, Calvo-González & Fleuret, Feb 2026)

This is the single most directly relevant measured ablation found, and it
points *against* "many narrow tokens is the wrong shape" — but note their
narrowest tested width is 32, ten times wider than 3, and their model is ~390 M
parameters.

### Alternatives that are actually used, with shapes

1. **Cross-attention read-out into k latent slots (Perceiver family).** A
   learned latent array of N x D attends over the inputs; outputs are produced
   by cross-attention from per-output queries, so *every output position sees
   the whole latent array*.
   **[measured]** Perceiver IO latent arrays: language 256 x 768; optical flow
   256 x 322 (3D-Fourier variant 256 x 451); StarCraft II 32 or 64 x 128;
   ImageNet D = 1024. Original Perceiver used 512 x 1024 for ImageNet
   (https://arxiv.org/abs/2103.03206).
   **[measured]** Kinetics multimodal autoencoding: 50,657 input elements,
   803,297 output elements, latent 784 x 512 = 401,408 (88x compression) gives
   video PSNR 24.37 / audio PSNR 26.97; 392 and 196 latents give 176x and 352x
   and degrade markedly. (https://arxiv.org/abs/2107.14795)

2. **Inducing points / seed vectors (Set Transformer).** ISAB_m: m trainable
   inducing points I in R^{m x d}; H = MAB(I, X) in R^{m x d}, then
   out = MAB(X, H) in R^{n x d}; cost O(nm). PMA_k pools a set into k vectors
   via k learnable seeds. **[measured]** m = 16 in their MoG clustering,
   CIFAR-100 clustering and point-cloud experiments; k = 1 for most tasks,
   k = 4 for amortized clustering; their ablation shows accuracy rising with m
   until ISAB_n matches full self-attention.
   (https://arxiv.org/abs/1810.00825)

3. **Learned token merging into a fixed M.** PatchMerger: a linear map scores
   each incoming token against M output slots, softmax-normalises, and each
   output token is the weighted sum — output count independent of input count.
   (https://arxiv.org/abs/2202.12015). ToMe merges r similar tokens per layer
   by bipartite matching, no training needed
   (https://arxiv.org/abs/2210.09461).

4. **A global CLS-style code.** Present in the literature mainly as the k=1
   special case of PMA (Set Transformer) or a 1-token TiTok/FlexTok.
   **[measured]** FlexTok reconstructs from as few as **1** token, but only
   with a rectified-flow decoder and nested dropout; TiTok at 16 tokens is
   rFID 13.0, i.e. visibly degraded. (https://arxiv.org/abs/2502.13967,
   https://arxiv.org/abs/2406.07550)

5. **Variable / adaptive allocation.** FlexTok: 1–256 ordered 1D tokens, nested
   dropout + flow decoder, stable down to one token. ElasticTok: nested dropout
   between a min and max token count, but **the minimum had to be set to 128 or
   256 because of instabilities below that**. One-D-Piece: 1–256 with tail-token
   drop. ALIT: recurrent allocation with a stopping rule.
   (https://arxiv.org/abs/2502.13967, https://arxiv.org/abs/2410.08368,
   https://arxiv.org/abs/2501.10064, https://arxiv.org/abs/2411.02393)
   **[measured, from FlexTok's comparison]** other methods fail below a
   threshold — ElasticTok below 128 tokens, ALIT below 32 — while FlexTok holds
   at 1–32.

6. **Hierarchy.** Not covered well by the sources gathered here; see Gaps.

---

## 2. How many latent tokens x how many channels, for what quality

**[measured] TiTok-L (307 M), ImageNet 256x256, 1D discrete tokens, 16 channels
per token, codebook 4,096** (https://arxiv.org/abs/2406.07550):

| Tokens | rFID |
|---|---|
| 16 | 13.0 |
| 32 | 6.6 |
| 64 | 4.0 |
| 128 | 3.0 |
| 256 | 2.5 |

**[measured] rFID vs model size at 64 tokens:** TiTok-S 5.9, TiTok-B 5.9,
TiTok-L 4.0. Capacity matters, but not monotonically at every size — S and B
tie, L jumps. The paper's own summary: "scaling up the tokenizer model size
significantly improves performance ... especially when number of tokens is
limited (e.g. 32 or 64)".

**[measured] The training recipe dominated the architecture.** The same TiTok
architecture at 32 tokens gets **rFID 2.21 with the two-stage recipe (warm-up
against MaskGIT-VQGAN "proxy codes", then decoder fine-tuning to pixels) versus
5.15 with plain single-stage training** — a 2.3x difference in rFID from the
*target*, not the bottleneck. (https://arxiv.org/abs/2406.07550)
This is the strongest single piece of evidence that a bad number at a narrow
bottleneck is not automatically evidence that the bottleneck is wrong.

**[measured] Perceiver IO Kinetics autoencoding:** 88x -> video PSNR 24.37;
176x and 352x degrade. (https://arxiv.org/abs/2107.14795)

**[measured] LDM downsampling factor:** f=4 and f=8 work best; f=16 is
noticeably worse and f=32 "limits the overall sample quality".
(https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf)

**[measured] Video tokenisers.** MAGVIT-v2: temporal 4x, spatial 8x, latent
5x16x16 for a 17x128x128 clip. NVIDIA Cosmos Tokenizer: spatial 8x or 16x and
temporal 4x or 8x, up to 2048x total, reported +4 dB PSNR over prior
tokenisers on DAVIS. (https://arxiv.org/abs/2310.05737,
https://research.nvidia.com/labs/cosmos-lab/cosmos-tokenizer/)

---

## 3. The failure mode: a learned autoencoder losing to a linear map

### 3a. The strongest documented case: DC-AE

**[measured]** *Deep Compression Autoencoder* (https://arxiv.org/abs/2410.10733)
isolates exactly this. They hold **total latent size fixed** and raise the
spatial compression ratio, converting latents with space-to-channel, and add
encoder/decoder stages so capacity *increases*. Quoting the paper:

> "Even with the same total latent size and stronger learning capacity, we
> still observe degraded reconstruction accuracy when the spatial compression
> ratio increases."

and their diagnosis:

> "the accuracy gap comes from the model learning process: while we have good
> local optimums in the parameter space, the optimization difficulty hinders
> high spatial-compression autoencoders from reaching such local optimums."

Their fix — **Residual Autoencoding** — is to make the network learn a
*residual on top of a parameter-free linear rearrangement*: the downsample
block applies space-to-channel (H x W x C -> H/2 x W/2 x 4C), splits into two
groups and averages them to the output channel count, and the network learns
the delta from that; the decoder mirrors it with channel-to-space + duplication.

**[measured]** Same latent shape, different architecture, ImageNet 512x512:
f64c128 rFID **16.84 (SD-VAE) vs 0.22 (DC-AE)**; f128c512 rFID **100.74 vs
0.23**; FFHQ 1024x1024 f64c128 **6.62 vs 0.23**. (Table 2 of that paper.)

**[opinion, but well grounded]** This is the closest published analogue of
"my learned autoencoder loses to a linear projection of the same rank", and
the published answer is: *optimisation, not capacity or bottleneck shape* —
and the remedy is to hand the network a linear shortcut that already achieves
the trivial solution, so gradient descent only has to learn the correction.

### 3b. Theory: a linear autoencoder *is* PCA at its optimum

- Bourlard & Kamp (1988) and Baldi & Hornik (1989): a linear autoencoder under
  squared loss has no spurious local minima; its global optimum spans the
  top-k principal subspace. So the rank-k linear solution is inside the
  hypothesis class of any autoencoder that can represent linear maps.
- Plaut (2018), *From Principal Subspaces to Principal Components with Linear
  Autoencoders* (https://arxiv.org/abs/1804.10253): the LAE recovers the
  subspace, not the ordered individual components; an SVD of the decoder
  recovers them.
- Kunin et al. (2019), *Loss Landscapes of Regularized Linear Autoencoders*
  (https://arxiv.org/abs/1901.08168): L2 regularisation changes the landscape
  so the actual principal directions become the minima.
- **[measured/theory] Bao, Lucas, Sachdeva & Grosse, NeurIPS 2020,
  "Regularized linear autoencoders recover the principal components,
  eventually" (https://arxiv.org/abs/2007.06731):** with non-uniform L2 or
  deterministic nested dropout, gradient descent does converge to ordered,
  axis-aligned PCs — but **convergence is slow because of ill-conditioning that
  gets worse as the latent dimension grows**, and they propose a Rotation
  Augmented Gradient update that fixes it empirically.
  **This is directly on point for a latent of 2,163:** even in the *purely
  linear* case, reaching the PCA solution by gradient descent at that latent
  dimension is expected to be slow.

**[opinion]** Consequence for diagnosis: because the linear solution is in the
hypothesis class (modulo the architecture actually being able to express it —
see §5), "AE < PCA at equal rank" is by default evidence of an optimisation or
capacity shortfall, not evidence about the data. Nonlinearity can only *add* to
the achievable minimum; it cannot make the linear optimum unreachable in
principle.

### 3c. The honest counterweight: reconstruction loss is not representation quality

**[theory, ICML 2026]** Conde Mendes, Bardone, Koller, Medina Moreira, Erba,
Troiani, Zdeborová, *A solvable high-dimensional model where nonlinear
autoencoders learn structure invisible to PCA while test loss misaligns with
generalization* (https://arxiv.org/abs/2602.10680). They construct a model with
two latent factors — one visible in the covariance, one only in higher moments.
PCA and linear AEs recover only the first; nonlinear AEs recover both. The
paper's headline: there is a regime where **the nonlinear AE has better
representation quality and *higher* reconstruction loss**, and "self-supervised
test loss is poorly aligned with representation quality".

**[opinion]** So "PCA reconstructs better" does not by itself prove the learned
latent is worse *for the downstream use*. It does prove the learned latent is
worse *at reconstruction*, which is the thing being claimed if reconstruction
is the objective.

### 3d. Weaker / more generic evidence

- Empirical comparison of AEs vs PCA on MNIST, Fashion-MNIST, CIFAR-10
  (https://arxiv.org/abs/2103.04874): k-NN accuracy comparable on PCA and AE
  projections "provided a big enough dimension"; PCA two orders of magnitude
  faster. **[measured but weak]** — classification, not reconstruction, and no
  per-dimension reconstruction table.
- scRNA-seq: a deep VAE (Tybalt) "can outperform PCA, ZIFA, UMAP and t-SNE"
  *when tuned*, and performs "remarkably poor" on the same data untuned
  (https://pmc.ncbi.nlm.nih.gov/articles/PMC6417816/). **[measured]** —
  a documented case of hyperparameters alone deciding whether a deep model
  beats PCA.

---

## 4. Diagnostics: which of the three causes is it?

**[opinion — a procedure synthesised from the mechanisms above, not a cited
protocol]** Ordered cheapest-first; each test isolates one hypothesis.

1. **Is the code even being used? (bottleneck / collapse)** Take the 721x3
   latent matrix over a batch, stack to (N*721) x 3 or N x 2163, and take its
   SVD. If the effective rank is far below 2,163, the model is not using the
   bottleneck it has — that is collapse, not a width problem. Related
   mechanisms: posterior collapse in VAEs; rank / dimensional collapse
   (https://arxiv.org/abs/2110.09348).

2. **Is the *decoder* or the *encoder* at fault?** Freeze the trained encoder,
   discard the decoder, and fit a **closed-form least-squares linear decoder**
   from the 2,163 latents back to the 92,288 outputs. If that linear decoder
   beats the trained transformer decoder, the encoder's code is fine and the
   decoder is the bottleneck. If it is still far below PCA-2048, the encoder is
   throwing information away.

3. **Is it the bottleneck shape or everything else?** Replace the bottleneck
   with the identity (width 128 per token, no compression) and retrain briefly.
   If the model still cannot reconstruct well, the transformer stack — not the
   bottleneck — is the limit. This is the cheapest single decisive test.

4. **Optimisation vs capacity.** Compare *training* reconstruction error
   against *held-out*. Train error already above PCA's train error = pure
   underfitting (optimisation or capacity); train error good and val error bad =
   overfitting/generalisation, a different problem. Also: can the model overfit
   a single batch of, say, 8 samples to near-zero error? If not, it is an
   optimisation/expressivity failure, full stop.

5. **Capacity specifically.** Scale width and depth (e.g. 192 -> 384 -> 768,
   4 -> 8 -> 12 blocks) holding everything else fixed and plot error vs
   parameters. If the curve is still falling steeply at the current size, the
   answer is "not enough parameters". TiTok's S/B/L points (5.9 / 5.9 / 4.0 at
   64 tokens) show what such a curve looks like.

6. **Training length / schedule.** Bao et al. predict slow convergence at large
   latent dimension even in the linear case, so plot the loss against a log
   time axis and check it is actually flat, not merely slow.

7. **Does a linear shortcut close the gap?** DC-AE's answer: add a
   parameter-free (or PCA-initialised) linear path from input to output and let
   the network learn only the residual. If that immediately closes most of the
   gap to PCA, the diagnosis is optimisation difficulty, exactly as DC-AE
   concluded. **[opinion]** This is both a diagnostic and, if it works, the fix.

8. **Input normalisation and loss.** PCA is computed on mean-centred data and
   is scale-aware; an autoencoder trained on unnormalised or
   heterogeneously-scaled features optimises a different effective objective.
   Check that the AE is trained on the same centring/scaling as the PCA and
   compare in the *same* units, in the same code path.

---

## 5. Why a 4-block, width-192 transformer over 721 tokens may not be able to
## express a global rank-2048 projection

**[theory, cited]** Three separate results bear on this.

1. **Low-rank bottleneck in multi-head attention.** Bhojanapalli, Yun, Rawat,
   Reddi, Kumar, ICML 2020 (https://arxiv.org/abs/2002.07028): scaling head
   size as d/h creates a low-rank bottleneck on the attention matrix; they
   argue the head size should be set to **the input sequence length**,
   independent of the number of heads, to obtain "provably more expressive"
   attention layers, and show smaller embedding dimensions then train better.
   For n = 721 with model width 192, head size is at most 192 (one head) and
   typically 48 (four heads) — far below 721. The attention matrix is
   provably rank-limited here.

2. **Rank collapse with depth.** Dong, Cordonnier, Loukas, ICML 2021
   (https://arxiv.org/abs/2103.03404): pure self-attention converges doubly
   exponentially with depth to a rank-1 matrix ("token uniformity"); skip
   connections and MLPs are what prevent it. Relevant as a check that the
   residual/MLP path is genuinely present and not being swamped.

3. **Softmax disperses as the sequence grows.** Veličković, Perivolaropoulos,
   Barbero, Pascanu, *Softmax is not Enough (for Sharp Size Generalisation)*
   (https://arxiv.org/abs/2410.01104): with bounded logits, the maximum weight
   any one key can receive decays as O(1/n) and normalised entropy tends to 1;
   "any learned circuitry must disperse as the number of items grows".
   Sparse alternatives (alpha-entmax) bound entropy at O(log s) instead
   (https://arxiv.org/abs/2506.16640, ICLR 2026). At n = 721 this is a real
   constraint on how sharply one position can read a specific other position.

**[opinion]** Put together: a global PCA projection applies a *different* dense,
signed 128-dimensional functional at every one of 721 positions. Softmax
attention's position-mixing is a set of non-negative scalars a_ij applied to a
*shared* value projection V x_j — position-specific content transforms have to
be synthesised through depth and through positional embeddings modulating V. A
4-block, width-192 stack with head size ~48 over 721 positions is a thin channel
for that. This argument is mine, assembled from the three cited results; I found
no paper that states it in this form (see Gaps).

### Practical designs for a spatially structured field from a per-position latent

**[measured / design, cited]**

- **Give every output position the whole latent, not just its own slot.**
  Perceiver IO's decoder is cross-attention from a per-output query into the
  latent array: "each output point depends only on its query and the latent
  array", which also allows subsampling outputs during training (they trained
  on 512 audio + 512 pixel samples per example out of 803,297 outputs).
  (https://arxiv.org/abs/2107.14795)
- **Use a convolutional decoder when the field is locally structured.** The
  SD/LDM decoder is a residual conv stack (~39–49 M parameters) and is what
  makes per-position 4 channels work at 48x compression.
  (https://arxiv.org/abs/2112.10752)
- **Give the decoder a linear/identity shortcut.** DC-AE's Residual
  Autoencoding (§3a).
- **Choose the decoder's *form* to suit a narrow code.** TiTok needed proxy
  codes; FlexTok needed a rectified-flow decoder to reconstruct from 1–32
  tokens where deterministic decoders fail; SSDD uses a single-step diffusion
  decoder (https://arxiv.org/abs/2510.04961).
- **Narrow the *lookup* code, not the feature width.** ViT-VQGAN projects
  768-d features to an 8–32-d code for quantisation and projects back up — the
  decoder still works in 768-d. (https://arxiv.org/abs/2110.04627)
  **[opinion]** The analogue here: 3 numbers can be a *transport* width without
  the encoder's and decoder's working width being 3 — which it already is not,
  at 192 — but the project-down/project-up pair around the bottleneck is worth
  checking for being a single linear layer with no nonlinearity budget.

---

## 6. The "3 numbers per token" regime specifically

- **No source found that tests a per-token continuous width of 3.** The
  narrowest widths located in the literature: 4 channels/position (SD KL-f8,
  and it works — with 81 M parameters, LPIPS + adversarial losses, and a conv
  decoder); 8-d factorized codes in ViT-VQGAN (works, 768-d working width);
  ~18 binary dims/token in MAGVIT-v2 LFQ; 16 channels/token in TiTok; 32 in
  FlatDINO's best configuration.
- **[measured]** FlatDINO's ablation says that at *fixed total* latent size,
  narrower-and-more-numerous beats wider-and-fewer (64x32 rFID 0.96 vs 16x128
  rFID 1.19) — down to width 32.
- **[measured]** The failures at extreme narrowness in the 1D-tokeniser
  literature are *instability*, and they are fixed by the training recipe, not
  by widening: ElasticTok could not train below 128 tokens; TiTok at 32 tokens
  goes 5.15 -> 2.21 rFID by changing the target to proxy codes; FlexTok reaches
  1 token with nested dropout plus a flow decoder.

**[opinion]** What people do instead when a very narrow per-token code does not
work, in rough order of how often it appears in these papers:
1. change the *target/recipe* (proxy codes, two-stage, nested dropout,
   distillation) before touching the shape;
2. change the *decoder class* (flow/diffusion decoder, conv decoder) rather
   than the latent;
3. add a linear residual shortcut (DC-AE);
4. re-shape the latent to fewer, wider tokens read out by cross-attention
   (Perceiver / TiTok / PMA) — note this also *reduces* the total latent size,
   so it is a different trade;
5. raise the parameter count.

---

## 7. What I would take away for the stated setting

**[opinion — a reading of the evidence, not a measurement]**

The evidence does **not** support "a uniform 3-per-token bottleneck is a
categorically wrong shape". FlatDINO's ablation points the other way at fixed
total latent size, and 42.7x per-position compression is the normal regime.
The two hypotheses the literature makes most plausible are, in order:

1. **Parameter budget.** 3.89 M against a 189 M-number PCA basis, where every
   comparable system is 20–400 M. TiTok's S/B/L curve shows this literature
   living an order of magnitude higher.
2. **Optimisation difficulty of the compressive map**, exactly as DC-AE
   diagnosed and as Bao et al. predict for a latent dimension of 2,163 even in
   the linear case — with a strong, cheap, published remedy (a linear/PCA
   residual shortcut) that doubles as a diagnostic.

A third, specific to this architecture rather than to the bottleneck: the
attention stack (4 blocks, width 192, head size ~48, 721 positions) is thin by
the standards of Bhojanapalli et al.'s expressivity argument and of softmax
dispersion at n = 721. Test 3 in §4 (identity bottleneck) separates this from
the bottleneck question in one run.

---

## Gaps

- **No source found that ablates a per-token continuous latent width of 3, or
  anything below 4.** The 3-wide regime is outside what this literature
  reports. The FlatDINO trend (narrower is better at fixed total) is measured
  only down to 32.
- **No source found that directly reports "transformer autoencoder loses to PCA
  at equal latent rank" as a headline result.** DC-AE is the closest analogue
  (learned AE loses to a *reshaped linear* baseline at equal latent size) and
  I am extrapolating from it. The generic AE-vs-PCA sources found
  (arXiv 2103.04874, the scRNA-seq literature, several blog posts) measure
  classification or clustering, not reconstruction at matched rank.
- **The §5 argument that softmax attention cannot cheaply synthesise
  position-specific dense linear functionals is assembled by me from three
  papers.** No source states it in that form. It should be treated as a
  hypothesis to test (via the identity-bottleneck run), not a cited fact.
- **Parameter counts for TiTok's encoder vs decoder split, and for FlatDINO
  overall, were not confirmed** — only the total model sizes (22/86/307 M) and
  the ViT-B/ViT-L designations.
- **The LDM appendix Table 8 numbers (per-f rFID/PSNR) were not retrieved
  first-hand**; only the paper's qualitative conclusion (f=4, f=8 best; f=16
  worse; f=32 limits quality) and secondary summaries.
- **Hierarchical latent designs** (multi-scale / Matryoshka-style token
  hierarchies) were not researched; only the flat and the variable-length
  families were covered.
- **Positional-embedding capacity specifically** (how many distinguishable
  positions a d-dimensional learned embedding supports, and whether 192 is
  enough for 721 positions) — no source located. Open question.
- **SD-VAE parameter split (42 M / 39 M) comes from a well-known gist, not a
  peer-reviewed source.**
- Two arXiv ids cited here have 26xx numbers (2602.04873 FlatDINO, 2602.10680,
  2603.x, 2606.x in passing) — these are 2026 preprints; FlatDINO's ablation in
  particular is a 2026 preprint and has not been independently replicated.
