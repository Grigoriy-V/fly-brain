# How modern generative systems actually get a drawable latent

Research note, 2026-09-21. Scope: the two-stage recipe (weakly-regularised
autoencoder, then a separate generative model over its latent), what the
first stage is actually trained with, and what it is *not* asked to
guarantee. Every configuration number below carries its URL. Claims are
tagged **[paper]**, **[repo config]**, **[repo code]** or **[docs]**.

---

## 0. The short answer

The industry does not build a VAE whose latent is standard-normal enough to
sample from. It builds an autoencoder whose KL term is weighted at about
`1e-6` — so weak that it does not shape the distribution, only stops the
latent scale from running away — and then trains a **second** generative
model (diffusion, flow matching, or an autoregressive prior) over that
latent. The "scale factor" every latent-diffusion system carries (0.18215,
0.13025, 1.5305, 0.3611) is the direct, published admission that the
aggregate posterior is *not* N(0, I): it is a per-checkpoint constant whose
only job is to divide out an empirically measured standard deviation that
is 5.5x, 7.7x, 0.65x or 2.8x off unit.

No production text-to-image or text-to-video system samples `z ~ N(0, I)`
and decodes it. The Gaussian draw happens at the *input of the second-stage
model*, never at the input of the decoder.

---

## 1. Latent Diffusion Models (Rombach et al., CVPR 2022)

Paper: <https://arxiv.org/abs/2112.10752> (arXiv v2 PDF used for appendix
text below).

### 1.1 What the first stage is trained with — the paper's own words

**[paper]** Appendix G, "Details on Autoencoder Models", quoted verbatim
from the arXiv v2 PDF:

> We train all our autoencoder models in an adversarial manner following
> [23], such that a patch-based discriminator D_psi is optimized to
> differentiate original images from reconstructions D(E(x)). **To avoid
> arbitrarily scaled latent spaces, we regularize the latent z to be zero
> centered and obtain small variance by introducing an regularizing loss
> term L_reg.** We investigate two different regularization methods: (i) a
> low-weighted Kullback-Leibler-term between q_E(z|x) = N(z; E_mu,
> E_sigma^2) and a standard normal distribution N(z; 0, 1) as in a standard
> variational autoencoder [46, 69], and, (ii) regularizing the latent space
> with a vector quantization layer by learning a codebook of |Z| different
> exemplars [96]. **To obtain high-fidelity reconstructions we only use a
> very small regularization for both scenarios, i.e. we either weight the
> KL term by a factor ~10^-6 or choose a high codebook dimensionality |Z|.**

The full objective (their Eq. 25):

```
L_Autoencoder = min_{E,D} max_psi ( L_rec(x, D(E(x)))
                                    - L_adv(D(E(x))) + log D_psi(x)
                                    + L_reg(x; E, D) )
```

**[paper]** Section 3.1 adds the loss composition: the reconstruction term
is "a combination of a perceptual loss [106] (= LPIPS, Zhang et al.) and a
patch-based [33] adversarial objective [20, 23, 103]". And the framing
sentence for the regularisation is:

> In order to avoid arbitrarily high-variance latent spaces, we experiment
> with two different kinds of regularizations. The first variant, *KL-reg.*,
> imposes a **slight** KL-penalty towards a standard normal on the learned
> latent, similar to a VAE, whereas *VQ-reg.* uses a vector quantization
> layer within the decoder.

Read the stated purpose carefully. It is **"avoid arbitrarily scaled /
high-variance latent spaces"** and **"zero centered, small variance"**. It
is *not* "make the aggregate posterior match N(0, I)". The paper never
claims the latent is standard normal and never relies on it being so.

### 1.2 The actual loss code and its normalisation

This matters more than the headline weight, because `kl_weight = 1e-6` means
nothing without knowing how the KL is summed.

**[repo code]**
<https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/losses/contperceptual.py>
(`LPIPSWithDiscriminator`):

```
rec_loss      = |inputs - reconstructions|                       # L1, elementwise
rec_loss      = rec_loss + perceptual_weight * LPIPS(inputs, recon)
nll_loss      = rec_loss / exp(logvar) + logvar                  # learned scalar logvar
weighted_nll  = sum(weights * nll_loss) / nll_loss.shape[0]
kl_loss       = posteriors.kl()
kl_loss       = sum(kl_loss) / kl_loss.shape[0]                  # mean over batch only
d_weight      = ||grad(nll_loss)|| / (||grad(g_loss)|| + 1e-4)   # adaptive, clamped [0, 1e4]
loss          = weighted_nll + kl_weight * kl_loss
                             + d_weight * disc_factor * g_loss
```

**[repo code]**
<https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/distributions/distributions.py>
(`DiagonalGaussianDistribution.kl`):

```
0.5 * sum(mean^2 + var - 1.0 - logvar, dim=[1, 2, 3])   # sum over C, H, W; per-sample
logvar clamped to [-30.0, 20.0]
```

So the KL is **summed over every latent dimension** and only averaged over
the batch, while the reconstruction/NLL term is a **mean over pixels**.
Arithmetic for the SD f=8, 256x256 setting: the latent is 4 x 32 x 32 =
**4096 dims** per image; the image is 3 x 256 x 256 = 196 608 elements.
An L1+LPIPS nll of order 0.1-1.0 is being balanced against
`1e-6 x (per-dim KL) x 4096` ≈ `4e-3 x (per-dim KL in nats)`. For the KL
to contribute even 0.1 to the loss, the model would have to be carrying
~25 nats per latent dimension. In practice the KL term is a soft leash on
the scale of `mu` and on `logvar` collapsing to -inf, nothing more. This is
a deterministic autoencoder with a scale penalty, wearing VAE clothes.

Corroborating empirical note **[third-party, informal]**: Ollin Boer Bohan's
notes on the SD VAE observe that "SD VAE KL noise has very little effect",
that the posterior "variances are really small", and that "taking the
encoder mean instead of sampling causes no difference in most cases" —
<https://gist.github.com/madebyollin/ff6aeadf27b2edbc51d05d5f97a595d9>.
Consistent with this, `diffusers`' `AutoencoderKL.forward` defaults to
`sample_posterior: bool = False`, i.e. the *mode*, not a draw
(<https://huggingface.co/docs/diffusers/api/models/autoencoderkl>).

### 1.3 The scale factor — what it is and what it proves

**[paper]** Appendix G, immediately after the loss, quoted verbatim:

> **DM Training in Latent Space** Note that for training diffusion models on
> the learned latent space, we again distinguish two cases when learning
> p(z) or p(z|y) (Sec. 4.3): (i) For a KL-regularized latent space, we
> sample z = E_mu(x) + E_sigma(x) * eps =: E(x), where eps ~ N(0, 1). When
> rescaling the latent, we estimate the component-wise variance
>
>     sigma_hat^2 = (1 / bchw) * sum_{b,c,h,w} (z_{b,c,h,w} - mu_hat)^2
>
> from the **first batch in the data**, where mu_hat = (1/bchw) *
> sum_{b,c,h,w} z_{b,c,h,w}. The output of E is scaled such that the
> rescaled latent has unit standard deviation, i.e.
> **z <- z / sigma_hat = E(x) / sigma_hat**. (ii) For a VQ-regularized
> latent space, we extract z before the quantization layer and absorb the
> quantization operation into the decoder.

**[paper]** Appendix D.1 says why it is needed:

> the signal-to-noise ratio induced by the variance of the latent space
> (i.e. Var(z)/sigma_t^2) significantly affects the results for
> convolutional sampling. For example, when training a LDM directly in the
> latent space of a KL-regularized model (see Tab. 8), **this ratio is very
> high**, such that the model allocates a lot of semantic detail early on in
> the reverse denoising process. In contrast, when rescaling the latent
> space by the component-wise standard deviation of the latents as described
> in Sec. G, the SNR is decreased. [...] **Note that the VQ-regularized
> space has a variance close to 1, such that it does not have to be
> rescaled.**

So:

- The scale factor is **one scalar**, `1 / sigma_hat`, estimated from a
  **single batch** of training latents, applied globally.
- It exists because the KL-regularised latent's variance is **far from 1**.
  The paper says the SNR is "very high", i.e. Var(z) >> 1.
- The diffusion model's noise schedule assumes data of roughly unit
  variance. The scale factor is a shim between the autoencoder's arbitrary
  scale and the diffusion schedule. It is *not* a claim of Gaussianity — a
  single global scalar cannot make a distribution Gaussian; it can only fix
  its second moment, and only in aggregate.

**[docs]** The `diffusers` documentation states this literally:

> `scaling_factor` (float, optional, defaults to 0.18215): The
> component-wise standard deviation of the trained latent space computed
> using the first batch of the training set. This is used to scale the
> latent space to have unit variance when training the diffusion model. The
> latents are scaled with the formula `z = z * scaling_factor` before being
> passed to the diffusion model. When decoding, the latents are scaled back
> to the original scale with the formula: `z = 1 / scaling_factor * z`. For
> more details, refer to sections 4.3.2 and D.1 of the paper.

<https://huggingface.co/docs/diffusers/api/models/autoencoderkl>

**What 0.18215 says about the actual distribution.** `1 / 0.18215 = 5.49`.
The raw SD 1.x latent has standard deviation ~5.5, not 1. If the KL had
succeeded in matching N(0, I), the scale factor would be 1.0 and nobody
would have needed to measure it.

### 1.4 The autoencoder zoo (Table 8), and what it shows about the trade-off

**[paper]** Table 8, LDM arXiv v2, autoencoders trained on OpenImages,
evaluated on ImageNet-val. `f` = downsampling factor, `c` = latent channels,
`|Z|` = codebook size (VQ) or "KL":

| f | \|Z\| | c | R-FID ↓ | PSNR ↑ | SSIM ↑ |
|---|-------|---|---------|--------|--------|
| 32 | 16384 | 16 | 31.83 | 17.45 | 0.41 |
| 16 | 16384 | 8 | 5.15 | 20.83 | 0.54 |
| 8 | 16384 | 4 | 1.14 | 23.07 | 0.65 |
| 8 | 256 | 4 | 1.49 | 22.35 | 0.62 |
| 4 | 8192 | 3 | 0.58 | 27.43 | 0.82 |
| 4 | 256 | 3 | 0.47 | 26.43 | 0.80 |
| 2 | 2048 | 2 | 0.16 | 30.85 | 0.91 |
| 32 | KL | 64 | 2.04 | 22.27 | 0.61 |
| 32 | KL | 16 | 7.30 | 20.38 | 0.53 |
| 16 | KL | 16 | 0.87 | 24.08 | 0.68 |
| 16 | KL | 8 | 2.63 | 21.94 | 0.59 |
| 8 | KL | 4 | 0.90 | 24.19 | 0.69 |
| 4 | KL | 3 | 0.27 | 27.53 | 0.82 |
| 2 | KL | 2 | 0.086 | 32.47 | 0.93 |

Note the pairs at fixed `f`: `f=32, c=64` gives R-FID 2.04 while `f=32,
c=16` gives 7.30; `f=16, c=16` gives 0.87 while `f=16, c=8` gives 2.63.
**Latent capacity, not regularisation strength, is what buys reconstruction.**

---

## 2. The real config values across the industry

All of these are **[repo config]** unless noted. The last column is
`1 / scaling_factor`, i.e. the measured standard deviation of the raw
latent — the number that would be 1.0 if the latent were N(0, I).

| System | latent ch. | f | scaling_factor | shift_factor | implied raw sigma |
|---|---|---|---|---|---|
| LDM `autoencoder_kl_32x32x4` (`kl_weight: 0.000001`, `disc_weight: 0.5`, `disc_start: 50001`, `embed_dim: 4`, `z_channels: 4`, `ch_mult [1,2,4,4]`) | 4 | 8 | — | — | — |
| SD 1.x (`v1-inference.yaml`, `scale_factor: 0.18215`) | 4 | 8 | 0.18215 | — | **5.49** |
| SDXL VAE (`stabilityai/sdxl-vae/config.json`) | 4 | 8 | 0.13025 | — | **7.68** |
| SD3 / SD3.5 (`SD3LatentFormat`) | 16 | 8 | 1.5305 | 0.0609 | **0.65** (mean 0.0609) |
| FLUX.1 (`AutoEncoderParams`, `z_channels: 16`) | 16 | 8 | 0.3611 | 0.1159 | **2.77** (mean 0.1159) |
| Playground v2.5 VAE (per-channel) | 4 | 8 | 0.5 | `latents_mean` | see below |

URLs:

- LDM AE config:
  <https://github.com/CompVis/latent-diffusion/blob/main/configs/autoencoder/autoencoder_kl_32x32x4.yaml>
  — `kl_weight: 0.000001`, `disc_weight: 0.5`, `disc_start: 50001`,
  `embed_dim: 4`, `z_channels: 4`, `double_z: True`, `ch: 128`,
  `ch_mult: [1,2,4,4]`, `num_res_blocks: 2`, `base_learning_rate: 4.5e-6`.
- SD 1.x: <https://github.com/CompVis/stable-diffusion/blob/main/configs/stable-diffusion/v1-inference.yaml>
  — `scale_factor: 0.18215`, `embed_dim: 4`, `z_channels: 4`,
  `ch_mult: [1,2,4,4]`.
- SDXL VAE: <https://huggingface.co/stabilityai/sdxl-vae/blob/main/config.json>
  — `"latent_channels": 4`, `"scaling_factor": 0.13025`,
  `"block_out_channels": [128,256,512,512]`, `"sample_size": 1024`.
- SD3: <https://github.com/Stability-AI/sd3-ref/blob/master/sd3_impls.py>
  — `class SD3LatentFormat`, docstring "Latents are slightly shifted from
  center - this class must be called after VAE Decode to correct for the
  shift", `scale_factor = 1.5305`, `shift_factor = 0.0609`,
  `process_in(latent) = (latent - shift_factor) * scale_factor`,
  `process_out(latent) = (latent / scale_factor) + shift_factor`.
- FLUX: <https://github.com/black-forest-labs/flux/blob/main/src/flux/util.py>
  — `AutoEncoderParams(..., z_channels=16, ch_mult=[1,2,4,4],
  scale_factor=0.3611, shift_factor=0.1159)`, identical across every FLUX
  variant (dev, schnell, canny, depth, redux, fill, kontext, krea).
- Playground v2.5 VAE:
  <https://huggingface.co/playgroundai/playground-v2.5-1024px-aesthetic/blob/main/vae/config.json>
  — `"latents_mean": [-1.6574, 1.886, -1.383, 2.5155]`,
  `"latents_std": [8.4927, 5.9022, 6.5498, 5.2299]`,
  `"scaling_factor": 0.5`, `"latent_channels": 4`.

**Playground v2.5 is the clearest single piece of evidence in this note.**
It is an SDXL-family VAE whose latent statistics were measured per channel
rather than globally. The four channel means are −1.66, +1.89, −1.38,
+2.52 — nowhere near zero. The four channel standard deviations are 8.49,
5.90, 6.55, 5.23 — five to eight times unit, and *unequal to each other*.
That is the aggregate posterior of a production latent-diffusion VAE, in
public, in numbers. It is not N(0, I) and it is not even isotropic. What
the industry does about it is not to fix the VAE; it is to store the mean
and std in the config and normalise with them.

**[docs]** `diffusers` has first-class support for exactly this:
`AutoencoderKL(..., scaling_factor=0.18215, shift_factor=None,
latents_mean=None, latents_std=None, ...)` — an optional per-channel mean
and std applied automatically in `encode`/`decode` when the checkpoint
carries them.

### 2.1 SD3 on latent channels

**[paper]** SD3 (Esser et al., 2024, <https://arxiv.org/abs/2403.03206>),
Table 3, autoencoder reconstruction as a function of latent channel count:

| channels | FID ↓ | Perceptual Sim. ↓ | SSIM ↑ | PSNR ↑ |
|---|---|---|---|---|
| 4 | 2.41 | 0.85 | 0.75 | 25.12 |
| 8 | 1.56 | 0.68 | 0.79 | 26.40 |
| 16 | 1.06 | 0.45 | 0.86 | 28.62 |

They state that increasing latent channels `d` "significantly boosts
reconstruction performance", and chose `d = 16` because the improvement
carries through to generation for large models. The trend across the field
is the same: SD 1.x/SDXL at 4 channels, SD3 and FLUX at 16, and 2025-era
work going much higher still (see §5). **The direction of travel is toward
*more* latent capacity and *weaker* distributional constraint**, in exactly
the opposite direction from "make the latent standard normal".

---

## 3. The two-stage VAE literature: the same diagnosis, stated as theory

### 3.1 Dai & Wipf, "Diagnosing and Enhancing VAE Models" (ICLR 2019)

<https://arxiv.org/abs/1903.05789>

**The problem identified.** Even a VAE trained to optimality has an
**aggregate posterior that does not match the prior**. Their words: the
distribution of latent samples from the encoder, averaged over the training
data, "will have lingering latent structure that is errantly incongruous
with the original isotropic Gaussian prior." This is precisely the failure
mode of ancestral sampling: draw `z ~ N(0, I)`, push through the decoder,
get something outside the region the decoder was ever trained on.

Supporting analysis:

- **The gamma (decoder variance) argument.** Theory says the optimal
  decoder variance gamma -> 0, but practitioners fix gamma ≈ 1. Fixing it
  artificially degrades both reconstruction and the shape of the latent.
- **The manifold argument.** When data lies on an r-dimensional manifold in
  d-dimensional ambient space with r < d, an optimal VAE can drive the
  objective to −inf while perfectly reconstructing the manifold using only
  r latent dimensions — and the *excess* dimensions get blocked by the
  decoder. The consequence is that the useful latent region is a
  lower-dimensional structure inside the latent space, not a filled
  Gaussian ball.

**What the second stage does.** Train VAE-1 on `x`. Encode the training set
to get `{z}`. Train a **second, independent VAE-2 treating those `z` as
data**, learning `q(u|z)` and `p(z|u)`. To generate: draw `u ~ N(0, I)`,
decode to `z` via VAE-2, decode to `x` via VAE-1's decoder. The point is
that in the second stage the manifold dimension equals the ambient
dimension (the latents fill their own space), which is the regime where a
VAE *can* match its prior.

**Headline results.** CelebA 32x32, neutral architecture: plain VAE with
learned gamma FID 60.5, WAE-GAN 42, **2-stage VAE 44.4**. Under the exact
WAE architecture and protocol: plain VAE 63, WAE-GAN 42, **2-stage VAE 34**.
Claimed as the first VAE pipeline competitive with GANs under fair
comparison.

The structural lesson is the one that matters here: **the fix for a
non-Gaussian aggregate posterior is a second model over the latent, not a
stronger KL on the first model.**

### 3.2 Ghosh et al., "From Variational to Deterministic Autoencoders" (RAE, ICLR 2020)

<https://arxiv.org/abs/1903.12436>

Abstract, verbatim in the relevant part:

> We observe that sampling a stochastic encoder in a Gaussian VAE can be
> interpreted as simply injecting noise into the input of a deterministic
> decoder. We investigate how substituting this kind of stochasticity, with
> other explicit and implicit regularization schemes, can lead to an equally
> smooth and meaningful latent space **without forcing it to conform to an
> arbitrarily chosen prior**. To retrieve a generative mechanism to sample
> new data, we introduce an **ex-post density estimation** step that **can be
> readily applied also to existing VAEs, improving their sample quality**.

- **Regularisers used instead of the KL**: L2 weight decay on the decoder,
  gradient penalty on the decoder, spectral normalisation, and implicit
  regularisation (batch norm, dropout). Plus an L2 penalty on `z` itself to
  keep the latent bounded — the same "don't let the scale run away" job that
  LDM's 1e-6 KL does.
- **Ex-post density estimation**: after training, fit a density to the
  latents. They compare an isotropic Gaussian, a full-covariance Gaussian,
  and a **10-component GMM**. The GMM wins, which is itself a measurement
  that the latent is not a single isotropic Gaussian.
- The step improves plain VAEs and WAEs too, **without retraining them** —
  direct evidence that even a properly trained VAE's aggregate posterior is
  far enough from N(0, I) that replacing the prior at sampling time is worth
  real FID.

**The phrase "ex post density estimation" is the honest name for what
latent diffusion does.** LDM's second stage is an ex-post density estimator
over the autoencoder's latent; it is just a diffusion model instead of a
10-component GMM.

### 3.3 VQ-VAE-2 (Razavi et al., NeurIPS 2019) — the same shape, discrete

<https://arxiv.org/abs/1906.00446>

- **Stage 1**: train a hierarchical VQ-VAE (ImageNet 256: top latent 32x32,
  bottom 64x64; codebook 512 entries of dim 64; FFHQ-1024 adds a third level
  at 128x128).
- **Stage 2**: **freeze** the encoder/decoder, then fit PixelCNN
  autoregressive priors over the discrete codes.
- Sampling means sampling from the autoregressive prior, then decoding.

There is no pretence anywhere that the code distribution is simple. The
prior is *learned*, after the fact, and that is where all the distributional
modelling lives.

---

## 4. Alternatives to a Gaussian latent, and what each trades away

| Approach | Latent | Second stage | What you give up |
|---|---|---|---|
| **Strong-KL VAE, sample z~N(0,I)** | continuous, ~Gaussian | none | reconstruction quality; blurry, low-detail decodes. Essentially unused at scale. |
| **Weak-KL AE + diffusion/flow** (LDM, SD, SDXL, SD3, FLUX) | continuous, arbitrary scale/shape | diffusion or flow matching | the ability to sample in one shot; you must train and run a second model. |
| **VQ-VAE + autoregressive prior** (VQ-VAE-2, VQGAN+Transformer) | discrete codes | PixelCNN / transformer | sampling is sequential (slow); codebook collapse / low utilisation; discrete gradients need straight-through. |
| **FSQ** | discrete, implicit codebook | same as VQ | a few dimensions only (d < 10), so per-token capacity is low; you lose the learned codebook's adaptivity. |
| **Representation AE (frozen DINOv2 + trained decoder)** | continuous, very high-dim, not Gaussian at all | diffusion transformer | no compression in the channel sense at all (768-1024 channels). |

**FSQ — Finite Scalar Quantization** (<https://arxiv.org/abs/2309.15505>).
Projects the encoder output to `d` dimensions (typically < 10), bounds each,
and rounds each to `L` levels. The codebook is the implicit product set, of
size `L^d`; nothing is learned about it. It **removes** the commitment loss,
the codebook loss, EMA codebook updates, and codebook splitting/reseeding.
Reported codebook utilisation is ~100% across configurations, where VQ drops
below 50% for codebooks above 2^11 and keeps falling. Example configuration:
for a 2^12 codebook, levels `[7,5,5,5,5]` (= 4375). On MaskGIT (ImageNet
256) FSQ is within 0.5-3% of VQ; on UViM it is competitive and sometimes
better. **The relevance here: FSQ's lesson is that the constraint on the
latent should be as cheap and as structural as possible, and that removing
learned distributional machinery from the first stage costs almost nothing.**

---

## 5. What the autoencoder actually has to guarantee (2025-era evidence)

If a flow or diffusion model is going to be fitted over the latent anyway,
Gaussianity is not on the list. What *is* on the list, according to the
recent literature that measured it:

**Smoothness / low latent complexity.** EQ-VAE (<https://arxiv.org/abs/2502.09509>)
observes that SD-VAE latents are **not equivariant** to spatial transforms:
an image and its scaled version encode to latents that are *not* related by
the corresponding transform, so the diffusion model has to learn a nonlinear
relation that need not exist. The fix is a reconstruction-side
regularisation — match `L_rec(tau(x), D(tau(E(x))))` — applied with 50%
probability during a short fine-tune. Result: **x7 speedup on DiT-XL/2**
(1.5M vs 7M iterations to gFID 8.8), SiT-XL/2 gFID 17.2 -> 16.1 at 400K
iters, **x4 speedup on REPA** (1M vs 4M iters to gFID 5.9). Five epochs of
SD-VAE fine-tuning, no change to Gaussianity.

**Spectral content.** "Improving the Diffusability of Autoencoders"
(Skorokhodov et al., ICML 2025, <https://arxiv.org/abs/2502.14831>),
abstract verbatim in part:

> we perform a spectral analysis of modern autoencoders and identify
> **inordinate high-frequency components in their latent spaces, which are
> especially pronounced in the autoencoders with a large bottleneck channel
> size**. We hypothesize that this high-frequency component interferes with
> the coarse-to-fine nature of the diffusion synthesis process and hinders
> the generation quality. To mitigate the issue, we propose scale
> equivariance [...] It requires minimal code changes and only up to 20K
> autoencoder fine-tuning steps, yet significantly improves generation
> quality, **reducing FID by 19%** for image generation on ImageNet-1K 256²
> and **FVD by at least 44%** for video generation on Kinetics-700 17x256².

Again: the property that predicted generation quality was the *frequency
structure* of the latent, not its marginal distribution.

**And dimensionality is not the constraint either.** `AutoencoderRAE` in
`diffusers` (<https://huggingface.co/docs/diffusers/api/models/autoencoder_rae>),
from "Diffusion Transformers with Representation Autoencoders" (Zheng, Ma,
Tong, Xie, NYU VisionX, <https://arxiv.org/abs/2510.11690>), pairs a
**frozen** DINOv2 / SigLIP2 / MAE encoder with a trainable ViT decoder. The
latent for the DINOv2-base variant is **768 x 16 x 16**. There is no KL, no
VQ, no prior of any kind on the first stage — the encoder is not even
trained. The docs describe it plainly:

> In the two-stage RAE training recipe, the autoencoder is trained in stage
> 1 (reconstruction), and then a diffusion model is trained on the resulting
> latent space in stage 2 (generation).

and:

> Some pretrained checkpoints include per-channel `latents_mean` and
> `latents_std` statistics for normalizing the latent space. When present,
> `encode` and `decode` automatically apply the normalization and
> denormalization, respectively.

A frozen self-supervised feature space, 768 channels wide, is about as far
from N(0, I) as a latent gets — and a diffusion transformer is trained over
it successfully. **The generative model does not care what shape the latent
is, provided it can be modelled.**

### The working list

What the first stage must deliver, ranked by the evidence above:

1. **Reconstruction fidelity.** It is a hard ceiling on the whole system:
   nothing the second stage generates can be better than what the decoder
   can render from a real latent. This is why every production
   configuration spends its regularisation budget at `~1e-6`.
2. **Bounded, stable scale.** Not unit variance — just not drifting, not
   exploding, measurable once and dividable out. This is the entire job of
   LDM's KL term and of RAE's L2-on-z.
3. **Decoder robustness to small perturbations.** The second-stage model
   will never land exactly on a training latent. The decoder must degrade
   gracefully off-manifold. (This is what the VAE's noise injection buys
   incidentally, and what RAE buys deliberately with explicit decoder
   regularisation; `AutoencoderRAE` even carries a `noise_tau` parameter
   for "adds noise to latents during training".)
4. **Smooth / low-complexity / spectrally sane structure**, so that the
   second-stage model's inductive bias (coarse-to-fine denoising,
   locality, equivariance) matches the latent. This is what EQ-VAE and the
   diffusability paper measured, and it is worth 4-7x in training compute.
5. **Statistics you can record.** Per-channel mean and std, stored in the
   config, applied at the boundary.

Gaussianity is **not** on the list. Low dimensionality is **not** on the
list either (SD3 went 4 -> 16, RAE went to 768).

### What actually breaks if the latent is not close to Gaussian

Almost nothing, provided you do the two things the field does:

- **Normalise.** If the raw latent has std 5.5 and you feed it to a
  diffusion schedule calibrated for unit variance, the SNR at every
  timestep is wrong and the model "allocates a lot of semantic detail early
  on in the reverse denoising process" (LDM Appendix D.1). Divide by the
  measured std (and subtract the measured mean, as SD3 and FLUX do) and the
  problem goes away. Per-channel is strictly better than global; Playground
  v2.5's per-channel stds range 5.2-8.5, so a single global scalar is a
  compromise.
- **Fit a second model.** Not a closed-form prior.

What *does* break:

- **Direct ancestral sampling** `z ~ N(0, I) -> D(z)`. This is the thing
  that stops working, and it is the thing nobody at scale does.
- **The SNR schedule**, if you skip normalisation — see above.
- **Training efficiency of the second stage**, if the latent is spectrally
  or geometrically nasty (high-frequency junk, non-equivariance). You pay
  in iterations, not in achievability: EQ-VAE's baselines still converge,
  they just take 4-7x longer.
- **Heavy tails / outliers.** A latent with rare extreme values makes the
  second stage's job harder at the tails. This is a real but second-order
  concern, and the practical handle is the same weak scale penalty plus
  normalisation, not a stronger KL.

---

## 6. Does anyone sample `z ~ N(0, I)` through a VAE decoder at scale?

**No, and the configs prove it rather than merely suggesting it.**

Evidence, strongest first:

1. **Every latent-diffusion checkpoint ships a scale factor.** The number
   exists only because the latent's variance was measured and found to be
   wrong. SD 1.x implies sigma ≈ 5.49, SDXL ≈ 7.68, FLUX ≈ 2.77, SD3 ≈ 0.65.
   If the aggregate posterior were N(0, I), all four would be 1.0.
2. **SD3 and FLUX also ship a *shift* factor** (0.0609 and 0.1159). The
   latent is not even zero-mean. The SD3 reference code says so in the class
   docstring: "Latents are slightly shifted from center".
3. **Playground v2.5 publishes the per-channel statistics**: means
   −1.66/+1.89/−1.38/+2.52, stds 8.49/5.90/6.55/5.23. Anisotropic and
   off-centre by a wide margin.
4. **The KL weight is 1e-6 against a KL summed over 4096 dimensions** while
   reconstruction is a per-pixel mean. There is no mechanism by which this
   term could shape the distribution.
5. **The LDM authors state the purpose is "avoid arbitrarily scaled latent
   spaces" / "zero centered and small variance"**, not prior matching, and
   they name the weight as "very small" deliberately, "to obtain
   high-fidelity reconstructions". They knew the trade-off and chose
   reconstruction.
6. **Every one of these systems trains a second generative model over the
   latent**, which would be redundant work if step 1 had produced a
   drawable prior.
7. **Dai & Wipf and Ghosh et al. reached the same conclusion from the
   theory side**, five and six years before FLUX, and both concluded that
   the answer is a second model / an ex-post density over the latent, not a
   stronger first-stage prior.

The only remaining systems where `z ~ N(0, I) -> decoder` is the sampling
path are small-scale research VAEs and the "VAE baseline" rows in papers
that exist to be beaten. Dai & Wipf's own numbers are the calibration: a
plain VAE at CelebA 32x32 scores FID 60-63; adding the second stage takes
it to 34-44. That is the price of insisting on a drawable single-stage
latent, at 32x32, in 2019.

---

## Gaps

Things I did **not** confirm, or confirmed only weakly:

- **`perceptual_weight` in the LDM autoencoder configs.** The loss code
  multiplies LPIPS by `self.perceptual_weight`, but I did not read the
  default value in `LPIPSWithDiscriminator.__init__`, and
  `autoencoder_kl_32x32x4.yaml` does not override it. The default is
  believed to be 1.0 but I have **not verified it**. Similarly
  `disc_factor` and `disc_num_layers` defaults were not read.
- **Whether the SD 1.x / SDXL VAEs were trained with exactly
  `kl_weight = 1e-6`.** The `1e-6` is confirmed for the CompVis
  `latent-diffusion` autoencoder config and matches the LDM paper's
  "~10^-6". Stability never published the training config for the SD 1.x or
  SDXL VAEs, nor for the `sd-vae-ft-mse` / `sd-vae-ft-ema` fine-tunes. The
  SDXL paper and the SD3 paper do not state a KL weight. **Treat "SDXL used
  1e-6" as an inference from lineage, not a documented fact.**
- **SD3 / FLUX first-stage loss composition.** Neither publishes an
  autoencoder training config. The SD3 paper reports only the channel
  ablation (Table 3). I could not fetch the SD3 or FLUX `vae/config.json`
  from Hugging Face — both returned HTTP 401 to an unauthenticated fetch —
  so the SD3 and FLUX numbers above come from the vendors' own reference
  repositories (`Stability-AI/sd3-ref`, `black-forest-labs/flux`) rather
  than from the model cards.
- **SD3 latent normalisation in the paper.** The shift/scale is confirmed in
  Stability's reference code, **not** in the SD3 paper text; my fetch of the
  paper found no discussion of latent normalisation.
- **The claim that `latents_mean`/`latents_std` are used by many models.**
  Confirmed concretely for Playground v2.5 and `AutoencoderRAE`, and
  confirmed as a supported `diffusers` config field. I did **not** enumerate
  which other checkpoints carry them.
- **Exact RAE (Ghosh et al.) FID table values.** I confirmed the method, the
  regulariser list, and that the GMM variant wins, but did not extract the
  numeric table. The qualitative claim ("competitive or better than VAE/WAE,
  GMM best") is from the paper's own summary, not from a table I read.
- **Direct empirical demonstration of decoding `N(0, I)` through the SD
  VAE.** I found only blog-level assertions, which I have not relied on.
  The argument in §6 is made from configs and paper text instead, which is
  stronger. If a demonstration is wanted, it is a five-line experiment.
- **"Essentially nobody does this at scale" is an argument from absence**
  for closed systems (Midjourney, Imagen, Veo, Sora), whose autoencoder
  configurations are not public. The claim is solid for every system whose
  configuration I could read.
- **Ollin Boer Bohan's note that the SD VAE's posterior noise is
  negligible** is an informal third-party gist, quoted as corroboration
  only. Not a primary source.
