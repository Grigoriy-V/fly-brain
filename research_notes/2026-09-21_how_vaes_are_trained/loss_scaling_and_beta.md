# How the reconstruction/KL balance is actually parameterised in working VAEs

Research request: given a loss written as `loss = rec + beta*kl`, with `rec` a
**mean** squared error over all `D = 92,288` data dimensions and `kl` a **sum**
over `K = 2,163` latent dimensions (averaged over the batch), what beta values
are sane, and how does this compare to what working systems actually do.

Every number below is tagged with the loss normalisation it belongs to. A
beta without its normalisation is not comparable across systems — this is the
central, repeated finding.

## 0. The one piece of algebra everything else reduces to

For any two scalar objectives that are positive linear combinations of the
same two quantities — `a*MSE + b*KL` and `c*MSE + d*KL` — the location of a
stationary point depends only on the **ratio** `a/b` (equivalently `c/d`), not
on the absolute scale (multiplying a loss by a positive constant does not
move its minimum). So converting between two (normalisation, beta) pairs that
are meant to express the same reconstruction/rate trade-off is a ratio-match:

```
a1 * MSE  +  b1 * KL      (convention 1: coefficients a1, b1)
a2 * MSE  +  b2 * KL      (convention 2: coefficients a2, b2)

same trade-off  <=>  a1/b1 = a2/b2  <=>  b2 = b1 * (a2/a1)
```

Two conventions differ only in how `MSE` and `KL` are *reduced* before the
loss combines them:
- `MSE`: **mean** over all `D` elements (`rec = (1/D) * sum_d (x_d - xhat_d)^2`) vs **sum** over `D` (`= D * mean`) vs the paper convention `(D/2) * mean` (drops the `1/2` from the Gaussian log-likelihood into the coefficient explicitly).
- `KL`: **sum** over the `K` latent dims (correct, matches `D_KL(q(z|x)||p(z))` for a diagonal Gaussian) vs an (incorrect, but common in framework defaults) **mean** over the `K` latent dims, which is `K` times smaller.

The project's own loss (`rec` = plain mean over `D`, `kl` = plain sum over
`K`, both then batch-averaged) is a normalisation choice with `a=1`, `b=1`
(coefficients in front of `MSE` and `KL` respectively) — this is the same
convention diffusers/CompVis's `AutoencoderKL` and 1Konny's beta-VAE
reproduction use for the KL side, but a lighter-weight rec side (they sum
pixels; the project means them). This one fact — sum vs mean on the rec side
— is exactly the `D`-sized rescaling that must be tracked below.

## 1. Standard conventions, and the conversion table

| Convention | rec term | kl term | Effective "textbook" beta (rec-coeff : kl-coeff = D/2σ² : 1) |
|---|---|---|---|
| **Textbook ELBO**, unit-variance Gaussian decoder | `(1/2) * sum_d (x-xhat)^2` = `(D/2) * mean` | `sum_k KL_k` | β = 1 (this *is* the ELBO, no free parameter) |
| **σ-VAE / calibrated-decoder papers** (Rybkin et al. 2021, eq. 6–7) | `(D/2σ²) * mean` | `sum_k KL_k` | any σ; β-VAE is this with σ fixed instead of learned |
| **This project's code** | `1 * mean` (coefficient 1, not `D/2`) | `sum_k KL_k` | need `beta = 2σ²/D` to match a real ELBO at decoder-variance σ² — **derivation below** |
| **Framework-default "mean everywhere"** (a bug the σ-VAE paper explicitly names) | `mean` | `mean over K` (not sum) | need `beta = 2σ²·K/D` to match — very different number again |

**Derivation for the project's exact loss** (`loss = mean_MSE + beta*sum_KL`,
`D=92,288`, decoder variance σ² assumed constant/isotropic, i.e. a Gaussian
likelihood `N(x; x̂, σ²I)`):

The true negative-ELBO (in nats, per sample) for that likelihood is
```
NLL + KL  =  D·ln(σ) + (D/2σ²)·MSE  +  sum_k KL_k          [[constant D·ln(σ) drops out of the (θ,φ) gradient]]
          =  (D/2σ²)·MSE + sum_k KL_k
```
Ratio-matching this to `1·MSE + beta·sum_k KL_k` (§0):
```
(D/2σ²) / 1  =  1 / beta   =>   beta = 2σ² / D
```
So for `D = 92,288`:
- **σ = 1** (the loss you get from plain, un-normalised MSE, i.e. what most
  code implicitly assumes when it just writes `rec = mse_loss`): the
  ELBO-consistent beta is **`beta = 2/92288 ≈ 2.17e-5`**.
- More generally: `beta = 2σ²/92288`, i.e. every order of magnitude change
  in beta corresponds to a factor-`√10` change in the implied decoder
  standard deviation σ (in z-scored units, since `x` is z-scored).

This lands the "principled, ELBO-consistent" beta for this exact code
almost exactly between the two smallest values tried (`1e-5` and `1e-4`) —
`1e-4` corresponds to an implied `σ² = 1e-4 * 92288/2 ≈ 4.6`, i.e. a decoder
that is claiming *more* noise than the unit-variance data itself (a very
loose likelihood, heavily favouring rate suppression); `1e-5` corresponds to
`σ² ≈ 0.46`, still looser than unit variance; `beta=0` is "ignore the prior
entirely," a plain autoencoder. None of the three tried values is close to
the `beta≈1` a naive "just use beta=1 like the original VAE paper" instinct
would suggest — and that instinct is exactly the trap the σ-VAE paper and
CompVis's own code (§3) warn about: `beta=1` is only ELBO-correct if `rec`
is *summed* like `kl` is, not meaned. This D-scale gap (β should carry a
`~1/D` factor whenever `rec` is a mean and `kl` is a sum) is the single most
common source of an accidentally-wrong (loss normalisation, beta) pair.
[Derived here from the ELBO; not itself a number from any paper — cross-checked
against §2's σ-VAE algebra below, which uses a different rec-convention and
is kept separate rather than force-reconciled.]

## 2. Gaussian likelihood → beta: the σ-VAE line (Rybkin, Daniilidis, Levine, ICML 2021)

Source: [arXiv:2006.13202](https://arxiv.org/abs/2006.13202) (verified against
the ar5iv rendering of the actual equations, not a summary).

Their convention: `MSE(x̂,x)` is a **mean** over the `D` data dims (same as
this project's `rec`). Their key equations (their numbering):

```
(2)  -ln p(x|z) = (1/2)||x̂-x||^2 + D·ln(√2π)  = (D/2)·MSE(x̂,x) + c     [σ=1, i.e. plain MSE loss ⇔ assuming unit decoder variance]
(5)  -ln p(x|z) = D·ln σ + (D/2σ²)·MSE(x̂,x) + c                         [general σ]
(6)  L_θ,φ,σ    = D·ln σ + (D/2σ²)·MSE(x̂,x) + D_KL(q(z|x)||p(z))         [σ-VAE objective, KL coefficient = 1, i.e. a genuine ELBO]
(7)  L^β        = (D/2)·MSE(x̂,x) + β·D_KL(q(z|x)||p(z))                 [β-VAE, "unit variance" convention]
```
Their stated conclusion, quoted verbatim: *"The β-VAE objective is then
equivalent to a σ-VAE with a constant variance σ² = β/2 (for a particular
learning rate setting)."* A direct ratio-match of their own eq. 6 and eq. 7
(§0's method) gives `β = σ²` instead of `σ² = β/2`; I could not resolve this
factor-of-2 gap from the paper text alone (flagged in Gaps) — it does not
change the order of magnitude of anything below.

Their **"optimal σ-VAE"** computes σ analytically per batch rather than
tuning β by hand: `σ*² = MSE(x, x̂)` (their eq. 8, the maximum-likelihood
variance estimate — literally the mean squared reconstruction error of the
batch). Their public code
([orybkin/sigma-vae-pytorch](https://github.com/orybkin/sigma-vae-pytorch))
implements this as `log_sigma = ((x-xhat)**2).mean([0,1,2,3]).sqrt().log()`,
soft-clipped to `log_sigma >= -6`.

Their explicit critique of hand-tuned β (quoted): *"By tuning the β term,
practitioners are able to tune the variance of the decoder, manually
producing a more calibrated decoder"* — i.e. a hand-picked β is not really a
weighting hyperparameter at all, it is a **stand-in for an unstated,
un-fitted decoder noise level σ**, chosen by trial and error instead of by
maximum likelihood. Concrete numbers from their Table 1 (SVHN, `D=32×32×3=3072`,
their MSE-mean convention): the learned optimal-σ-VAE converges to an
**implicit β ≈ 0.006**; manual β-sweep over `{0.001 … 10}` (their convention)
found the best FID at **β = 0.01** — i.e. hand-tuning independently lands
within 2× of what the analytic-σ calculation gives, which is their empirical
argument for replacing the β sweep with a per-batch analytic σ estimate.

A second and equally important point from the same paper, quoted directly
because it is exactly the project's situation:

> *"For the correct evidence lower bound computation, it is necessary to add
> [sum] the values of the MSE loss and the KL divergence across the
> dimensions. We observe that common implementations of these losses...use
> averaging instead, which will lead to poor results if the number of image
> dimensions is significantly different from the number of the latent
> dimensions... While this can be conveniently ignored in the β-VAE regime,
> where the balance term is tuned manually anyway, for the σ-VAE it is
> essential to compute the objective value correctly."*

Related, smaller follow-up: **Beta-Sigma VAE**
([arXiv:2409.09361](https://arxiv.org/pdf/2409.09361)) explicitly separates β
(rate control) from σ (calibration), arguing the two get conflated in plain
σ-VAE — relevant if this project ever wants a beta *and* a calibrated σ
simultaneously rather than only one knob; not verified in depth here (see
Gaps).

## 3. What real, working systems use — with their normalisation

| System | rec normalisation | kl normalisation | beta / kl_weight | Source |
|---|---|---|---|---|
| **Original β-VAE** (Higgins et al., ICLR 2017) | Bernoulli NLL, **summed** over pixels (binary dSprites/chairs/CelebA) | **summed** over latent dims | β = 4 (3D Chairs), β = 10 (CelebA), Burgess capacity variant for dSprites — order 1–250 across the paper's sweeps | [Higgins et al. 2017](https://www.cs.toronto.edu/~bonner/courses/2022s/csc2547/papers/generative/disentangled-representations/beta-vae,-higgins,-iclr2017.pdf); reproduction confirming the sum-then-batch-mean convention: [1Konny/Beta-VAE solver.py](https://github.com/1Konny/Beta-VAE/blob/master/solver.py) (`F.binary_cross_entropy_with_logits(..., size_average=False).div(batch_size)`; `klds.sum(1).mean(0)`) |
| **Stable Diffusion / LDM `AutoencoderKL`** (`kl-f8`, `kl-f4`, `kl-f8x8x64` etc.) | **L1**, elementwise, combined with a **learned global logvar** (`nll = rec_loss/exp(logvar) + logvar`, i.e. the model learns its own effective σ per training run, not fixed) — NOT plain mean MSE | `torch.sum(..., dim=[1,2,3])` then `/batch_size` → **summed over spatial+channel latent dims**, batch-averaged (the correct convention, matches §0/§2 warning) | **kl_weight = 1e-6** in every published autoencoder config (`kl-f4`, `kl-f8`, `kl-f8x8x64`, `kl-f16`, `kl-f32`) | configs: [autoencoder_kl_32x32x4.yaml](https://github.com/CompVis/latent-diffusion/blob/main/configs/autoencoder/autoencoder_kl_32x32x4.yaml), [kl-f8/config.yaml](https://github.com/CompVis/latent-diffusion/blob/main/models/first_stage_models/kl-f8/config.yaml); loss code: [contperceptual.py](https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/losses/contperceptual.py); distribution/KL code: [distributions.py](https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/distributions/distributions.py); paper statement (quoted): *"we...weight the KL term by a factor ~ 10⁻⁶"* — [Rombach et al. 2022, arXiv:2112.10752](https://arxiv.org/pdf/2112.10752) |
| **NVAE** (Vahdat & Kautz, NeurIPS 2020) | discretized-logistic-mixture NLL, summed over pixels, **per hierarchical group** | KL **summed** over each group's latent dims, with an explicit **"KL balancing"** coefficient per group (not a single scalar β — a spectral-regularization + per-group-temperature scheme instead) | no single β; regularises via `--weight_decay_norm` spectral term and per-group KL balancing coefficients, reports final quality in **bits/dim** (2.91 bits/dim on CIFAR-10) rather than a tuned β | [Vahdat & Kautz, arXiv:2007.03898](https://arxiv.org/abs/2007.03898); code: [NVlabs/NVAE](https://github.com/NVlabs/NVAE) |
| **VDVAE** (Child, ICLR 2021) | discretized-logistic-mixture NLL, summed, hierarchical (multiple resolutions) | **"free bits"**-style hinge per group: `max(λ, KL_group)` rather than a linear β·KL — caps the *minimum* KL any group can contribute rather than reweighting the average | not confirmed with a concrete number in this pass (see Gaps) | general "free bits" formulation, e.g. as summarised via [apxml VAE course notes](https://apxml.com/courses/vae-representation-learning/chapter-2-vaes-mathematical-deep-dive/vae-training-difficulties); not independently verified against Child's own repo in this pass |
| **Stable Audio 2.0 VAE** (`Oobleck`, Stability AI) | multi-resolution STFT loss (`mrstft: 1.0`) + disabled L1 (`l1: 0.0`), NOT MSE | KL loss weight applied directly (reduction not independently re-verified, presumed sum-then-mean per the `AutoencoderOobleck`/stable-audio-tools pattern) | **kl weight = 1e-4** | [stable_audio_2_0_vae.json](https://github.com/Stability-AI/stable-audio-tools/blob/main/stable_audio_tools/configs/model_configs/autoencoders/stable_audio_2_0_vae.json); model card: [AutoencoderOobleck, HF diffusers docs](https://huggingface.co/docs/diffusers/api/models/autoencoder_oobleck) |
| **VoxCPM causal audio VAE** | Mel-spectrogram + adversarial + KL (exact rec normalisation not confirmed) | not confirmed | practitioner-reported **kl weight = 5e-5** (from a user's question, not a maintainer-confirmed spec) | [OpenBMB/VoxCPM issue #145](https://github.com/OpenBMB/VoxCPM/issues/145) — **weak source, see Gaps** |

Reading the table as a whole: every working system's beta is small
(`1e-4`–`1e-6`) **only when paired with a summed-KL, non-mean-MSE rec term**
(SD, audio VAEs); the moment the rec term is *also* summed at comparable
scale to a properly weighted decoder (β-VAE's `(D/2)·MSE`, or a discretized
mixture NLL as in NVAE/VDVAE), the natural coefficient on KL climbs back
toward order 1–100. There is no such thing as a "correct beta" independent of
what `D` and the rec reduction are — every number in this table is only
interpretable next to its normalisation, which is exactly what the research
question asked to confirm.

## 4. Rate-distortion view (Alemi, Poole, Fischer, Dillon, Saurous, Murphy — "Fixing a Broken ELBO", ICML 2018)

Source: [arXiv:1711.00464](https://arxiv.org/abs/1711.00464), read via ar5iv
(equations verified directly, not summarised).

Definitions (their eq. 2–5, nats):
```
H ≡ -∫ p*(x) log p*(x) dx                         data entropy (fixed, unknown, a ceiling)
D ≡ -E_{p*(x)} E_{e(z|x)} [ log d(x|z) ]            distortion = reconstruction NLL (this project's "rec" term, in nats)
R ≡ E_{p*(x)} E_{e(z|x)} [ log (e(z|x)/m(z)) ]      rate = the KL term, in nats
H - D  ≤  I(X;Z)  ≤  R                              sandwich bound on mutual information
```
The ELBO = `-(D+R)`. Setting `min D + β·R` and taking `β=1` recovers the
plain ELBO; **any β traces one point on the rate-distortion frontier** —
literally `β = ∂D/∂R` at that point (a Legendre dual), not a "prior strength"
in any deeper sense. Sweeping β traces out the R-D curve; a
`(loss-normalisation, beta)` pair is "sane" exactly when it lands somewhere
on the feasible, monotonic part of that curve for the given architecture,
not any particular numeric value.

Concrete numbers they report, all on **static/binary MNIST (D=784 pixel
dims)**, useful as an order-of-magnitude anchor even though this project's
`D` is two orders of magnitude larger:
- Best achieved ELBO ≈ `-80.2` nats at `R≈0` (`H^ = 80.2` nats is their
  estimated upper bound on the data entropy of binarized MNIST).
- `β = 1.10`: `R = 0.0004` nats, `D = 80.6` nats — **posterior collapse** /
  "autodecoder": the rate is near zero, decoder ignores z, reconstructions
  stop depending on the input.
- `β = 0.1`: `R = 156` nats, `D = 4.8` nats — near-pixel-perfect
  reconstruction ("autoencoder" extreme), but they report the *prior samples*
  from this model are poor quality (the latent no longer looks like the
  prior it's regularized toward, because so little weight is put on KL).
  This is the generic failure mode of low-β: good reconstructions, bad
  samples/generation from the prior.
- A KL-annealed autoregressive-decoder run reached `R = 0.77` nats,
  `D = 89.60` nats, ELBO = `-90.37` — near-collapse even with annealing,
  illustrating that a powerful decoder can defeat almost any β if the
  architecture lets it ignore z.

**What rate to expect for a 42× compression of 92,288 → 2,163 dims**: this
paper gives no image-VAE number at that compression ratio, and no other
source found in this pass reports a *published* nats/sample figure for
SD's `kl-f8` (compression ≈48×, comparable order to this project's 42×) — its
`config.yaml` gives the *weight* (`kl_weight=1e-6`) but not the resulting
rate. The honest answer from this literature is structural, not a target
number: **rate is not something to target by picking a D-scaled beta in
advance — it is measured after training (§5) and beta is adjusted to move
it**, with the MNIST anchor above (`R` spanning 4 orders of magnitude,
`0.0004` to `156` nats, for `β` spanning barely one decade, `0.1` to `1.1`)
serving as a warning about how *rate-sensitive* this trade-off is to small
beta changes near collapse. Given this project's `K=2,163` latent dims
each contributing up to `~0.5·ln(2π)+...` nats before saturating, a rough
ceiling is `R ≲ few × K` nats if every dimension is meaningfully active —
i.e. thousands of nats, not tens — but whether the architecture can actually
use that much rate usefully is an empirical question, not a formula (§5).

## 5. Diagnosing "the beta is wrong" vs "the architecture is wrong"

Three complementary, cheap diagnostics recur across this literature (none
requires retraining if per-dimension KL and per-sample rate/distortion are
logged during training):

1. **Active units** (Burda, Grosse, Salakhutdinov, "Importance Weighted
   Autoencoders", 2016, and used as a standard posterior-collapse diagnostic
   since): for each latent dimension `z_k`, compute
   `A_k = Var_x( E_{q(z|x)}[z_k] )` — i.e. how much the *posterior mean* of
   that dimension actually moves across different inputs `x`, not how much
   the sampling noise moves it. A dimension is "active" if `A_k > 0.01`
   (the threshold used across the posterior-collapse literature, e.g.
   summarised at
   [emergentmind: Posterior Collapse in VAEs](https://www.emergentmind.com/topics/posterior-collapse)).
   If active-unit count << K (2,163 here), most of the latent budget is
   dead regardless of what the aggregate KL number says — a **KL-vs-beta**
   symptom (raise beta too little rate is used; but also an architecture
   symptom if active units stay low even as beta→0, meaning the encoder
   itself can't produce information-carrying posteriors).

2. **Per-dimension KL histogram**: plot `KL_k = 0.5(mu_k^2+exp(logvar_k)-1-logvar_k)`
   for all `k` (already computed per the project's own kl formula, just
   don't sum it away before logging). A healthy, well-used latent shows a
   spread of KL values across dimensions with a discernible mass of
   near-zero (unused) and a long tail of clearly-nonzero (carrying real
   information) dimensions — not literally uniform, and not a single spike
   near zero with all mass on one or two dims (bottleneck / degenerate
   posterior) or literally all-equal at some small constant (KL is not being
   allowed to differentiate importance — usually a sign beta is too large
   relative to the model's actual capacity to use the rate, or optimization
   hasn't run long enough). This is directly actionable in this project's
   existing per-dim `0.5*(mu^2+exp(logvar)-1-logvar)` computation.

3. **Rate-distortion sweep** (the Alemi et al. framework, §4): train (or
   fine-tune from a checkpoint) at several beta values spanning a couple of
   orders of magnitude around the current guess (e.g.
   `{1e-6, 1e-5, 1e-4, 1e-3}` given this project already has `1e-4,1e-5,0`
   measured), plot the resulting `(R, D)` pairs. A **sane point** sits on a
   monotonic, smoothly-varying curve — `D` decreasing as `R` increases, with
   no cliff. A **broken point** (this project's own prior finding, per the
   run log: "both ends of the KL dial fail") shows either (a) `R≈0`
   regardless of beta — architecture-side collapse, the decoder/encoder pair
   can't or won't carry information at any beta tried, matching Alemi et
   al.'s "powerful decoder ignores z" failure mode — or (b) `D` barely
   moving even as `R` grows very large — the *opposite* pathology, meaning
   distortion is dominated by something beta cannot fix (decoder capacity,
   not rate), which is exactly the encoder-decoder-bottleneck conclusion the
   project's own 2026-09-21 measurement (`083d161`) already reached. The
   sweep is what turns "we tried 3 betas and none worked" into "here is
   which side of the R-D curve is flat, and why beta can't be the fix,"
   per the σ-VAE and Alemi-et-al. framework above.

A **Multi-Rate VAE** ([arXiv:2212.03905](https://arxiv.org/pdf/2212.03905))
exists specifically to get the whole R-D curve from a single training run
(conditioning the network on beta) rather than a full sweep of separate
trainings — noted as a cheaper way to run diagnostic 3 if repeated sweeps
become expensive, not independently verified in depth here (see Gaps).

## Gaps

- **The σ-VAE paper's own factor-of-2** (`σ²=β/2` in text vs `β=σ²` from a
  direct ratio-match of their own eq. 6 and eq. 7, §2): read the actual
  equations via ar5iv rather than a summary, and still could not reconcile
  the stated conclusion with the displayed algebra. Does not change any
  order-of-magnitude conclusion above, but the exact constant is
  unconfirmed — if it matters, re-derive from Rybkin's supplementary
  material or the official PDF (not just ar5iv's HTML rendering) rather than
  trusting either value here uncritically.
- **No published nats/sample rate number found for SD's `kl-f8`/`kl-f4`
  autoencoders** at their ~48× compression, despite their `kl_weight` being
  public. Only the weight is documented, not the resulting measured rate —
  this project would have to measure its own `R` after training rather than
  relying on a published anchor at a comparable compression ratio.
- **VDVAE's free-bits λ value(s)** not obtained — the "free bits" mechanism
  itself is confirmed generically (cap per-group KL below a hinge threshold
  rather than a linear β), but I did not pull the actual λ Child used per
  resolution level from the paper or repo in this pass.
- **VoxCPM's kl_weight=5e-5** comes from a still-open GitHub issue (a user's
  question quoting a value found in the repo, not a maintainer's confirmed
  answer, and not independently checked against the repo's actual training
  config in this pass) — weak source, flagged as such in §3's table, treat
  as an unverified data point only.
- **Stable Audio Oobleck's KL reduction** (sum vs mean over the latent
  sequence before applying `kl=1e-4`) was not independently confirmed
  against the loss source code — only the config weight was fetched, not
  the loss-computation code path (unlike SD's `contperceptual.py`, which was
  fetched and quoted directly).
- **Beta-Sigma VAE** (arXiv:2409.09361) was found and cited by title/abstract
  only — its actual decoupling formula between β and σ was not extracted;
  worth a follow-up read if the project wants both a rate-control knob and a
  calibrated decoder simultaneously.
- **NVAE's per-group KL-balancing coefficients** (the actual numeric
  schedule, not just "a spectral regularization term exists") were not
  extracted from the paper — only the high-level mechanism and the
  bits/dim headline number were confirmed.
- No primary source was fetched for a **video** VAE's KL weight specifically
  (CogVideoX's compression ratios were found, but its loss config/kl_weight
  was not; the search surfaced only architecture-level facts). If a
  video-VAE anchor is needed later, CogVideoX's or Open-Sora's training
  configs on GitHub are the next thing to pull.
