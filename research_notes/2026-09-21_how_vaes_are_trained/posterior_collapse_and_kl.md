# Posterior collapse and the KL-control toolkit — research notes

Scope: general VAE literature on diagnosing and controlling posterior collapse /
KL weighting, to inform decisions about a per-token latent VAE (setting given:
721 tokens × 3 dims = 2,163 dims, Gaussian posterior, linear KL warm-up over
30% of training, free bits available but off; observed: at KL weight 1e-4,
sigma≈0.866 / mu sd≈0.415 with aggregate latent ≈ N(0,I); at 1e-5, sigma≈0.323
/ mu sd≈0.821 with aggregate latent off the sqrt(D) sphere). This file does not
make a recommendation for the project — it reports what the literature
measures and separates established results from folklore, per the request.

---

## 1. Diagnostics: active units, KL/dim, rate, mutual information

**Active Units (AU).** Defined by Burda, Grosse & Salakhutdinov, *Importance
Weighted Autoencoders* (2016, arXiv:1509.00519). For each latent dimension
*i*, compute the covariance, across the data distribution, of the posterior
mean: `A_i = Cov_x( E_{q(z|x)}[z_i] )`. A unit counts as "active" if `A_i >
0.01`; the AU statistic is the count of such units out of the total latent
dimensionality. It is a cheap proxy for "does the encoder's mean move at all
when the input changes," and is reported alongside log-likelihood in most
posterior-collapse papers (e.g. it is the headline metric used to show that
plain VAEs plateau in active units as depth increases while IWAE's active-unit
count keeps growing with more importance samples, Burda et al. Table 1).
It says nothing about whether the *decoder* uses the active dimension, only
whether the *encoder* is not fully collapsed to a constant.
[Burda et al. 2016](https://www.researchgate.net/publication/281487457_Importance_Weighted_Autoencoders)

**KL per dimension** is the direct decomposition of the closed-form Gaussian
KL term:
`KL(q(z|x)‖N(0,1)) = 1/2 Σ_j (σ_j² + μ_j² − 1 − log σ_j²)` — summed or
reported per-dimension `j`. This is the quantity free bits, delta-VAE and
target-KL regularization all operate on directly (see below). A dimension
whose per-dim KL sits at machine-epsilon for (almost) every input is
collapsed; the AU threshold (0.01) and a near-zero per-dim KL threshold
measure closely related but not identical things — AU is about *variation
across inputs*, per-dim KL is about *average information transmitted*. A unit
can have nonzero average KL while still being nearly input-independent if
`μ` and `σ` drift together without tracking `x` (this is a case free bits does
not catch — see §2). Formula reproduced from Seetharaman & Kumar, *Taming
Audio VAEs via Target-KL Regularization* (Adobe, 2026, arXiv:2605.17085, eq.
2), and from the closed-form derivation used throughout the field since
Kingma & Welling (2013).

**Rate (nats) and the rate–distortion framing.** Hoffman & Johnson, *ELBO
surgery: yet another way to carve up the variational evidence lower bound*
(NeurIPS Workshop 2016) decompose the aggregate KL term exactly:
`E_x[KL(q(z|x)‖p(z))] = I_q(x;z) + KL(q(z)‖p(z))`, i.e. **index-code mutual
information** (how much the code tells you about which datapoint it came
from) plus a **marginal KL** (how far the *aggregate* posterior `q(z) =
E_x[q(z|x)]` is from the prior). This is the diagnostic that most directly
answers "is the KL term doing anything useful": a run can drive the KL term
to a moderate value while `I_q(x;z) ≈ 0` and all of the KL sits in the
marginal-mismatch term, or vice versa. Alemi, Poole, Fischer, Dillon, Saurous
& Murphy, *Fixing a Broken ELBO* (ICML 2018, arXiv:1711.00464) formalize this
further: they derive variational upper and lower bounds on `I(x;z)` and use
them to trace out a **rate–distortion curve** — for a fixed ELBO value there
is a whole family of models trading rate (≈KL, the compression cost) against
distortion (≈reconstruction error), and the same ELBO number can correspond
to a model that ignores `z` entirely (rate≈0, distortion=data entropy) or one
that copies `x` into `z` (rate=high, distortion≈0). Practically: report both
the average KL/nats *and* an estimate of `I_q(x;z)` (or at minimum the AU
count, which is a cheap proxy for `I_q(x;z) > 0`), because the raw KL number
alone conflates "informative code" with "aggregate posterior has simply
drifted away from the prior without carrying information," which is exactly
the ambiguity the project's own control (time-shuffle) should be built to
distinguish.
[Hoffman & Johnson 2016](http://www.cs.columbia.edu/~blei/fogm/2025F/readings/HoffmanJohnson2016.pdf),
[Alemi et al. 2018](https://arxiv.org/abs/1711.00464)

**Mutual information estimates** in practice are usually one of: (a) the
Hoffman–Johnson decomposition applied with a Monte-Carlo estimate of `H(z)`
(expensive, needs the aggregate posterior density); (b) the Alemi et al.
variational upper/lower bounds (cheap, only need samples + the prior
density); (c) a downstream proxy — linear classification accuracy from `z`
to a label, used by Razavi et al. (delta-VAE, below) precisely because exact
MI is hard to estimate reliably in high dimensions. No universal "safe MI
value" exists in the literature; MI is compared relative to the rate axis of
the model's own rate–distortion curve, not against a fixed threshold.

---

## 2. Free bits and its variants

**Free bits (Kingma, Salimans, Jozefowicz, Chen, Sutskever & Welling,
*Improving Variational Inference with Inverse Autoregressive Flow*, NeurIPS
2016, arXiv:1606.04934, Appendix C.8).** Verified from the primary source.
The latent dimensions are partitioned into `K` groups (a feature map, or an
individual dimension if ungrouped), and the objective becomes:

```
L̃_λ = E_x[ E_q[log p(x|z)] ] − Σ_{j=1..K} max( λ, E_x[ KL(q(z_j|x) ‖ p(z_j)) ] )
```

i.e. a hinge: once a group's average KL exceeds `λ` nats, gradients stop
pushing it down further, removing the incentive to prune information once the
"free" budget is used. **Values tested: λ ∈ {0, 0.125, 0.25, 0.5, 1, 2, 4,
8} nats**; **λ ∈ {0.125, 0.25, 0.5, 1, 2}** all gave "more than 0.1 nats
improvement in bits/pixel" on CIFAR-10 relative to λ=0 (i.e. plain ELBO),
so the method is reported as fairly insensitive to the exact value inside
that range. Figure 7 of the paper (stack plots of nats/layer vs. training
epoch for λ ∈ {0, 0.125, 0.5, 2}) shows that λ=0 lets most of the 24
stochastic layers collapse to ~0 nats and stay there, while λ>0 keeps most
layers active throughout training — this is the direct empirical
demonstration that free bits fixes the "encoder gradients have low
signal-to-noise ratio near `q(z|x)≈p(z)`" failure mode, which the paper
attributes to Bowman et al. (2015) and Sønderby et al. (2016). The paper's
own best CIFAR-10 result (3.11 bits/dim, ResNet VAE + IAF) used this
objective, not annealing.
[Kingma et al. 2016 full text, fetched](https://arxiv.org/pdf/1606.04934)

**Known failure mode of plain (per-dimension or per-group) free bits.**
Because the hinge is applied per group and is non-smooth, individual groups
can still be driven to exactly `λ` and then the *gradient signal that would
otherwise push the group above λ* is masked group-by-group rather than
holistically — Razavi et al. (delta-VAE, next) report that "optimising
models with the free-bits loss was challenging and sensitive to
hyperparameter values" and that in their CIFAR-10 comparison free bits
produced points that were *not* on the rate–distortion frontier the way
delta-VAE or beta-VAE (well-tuned) were (Fig. 4a, delta-VAE paper). Seetharaman
& Kumar (audio VAE, 2026) independently report the same qualitative finding
("directly regressing the KL to a target value worked better in practice"
than free bits) and give a smooth alternative:

```
KL_target = B·log(2) / S        (bitrate B, frame rate S)
L_target-KL = (KL − KL_target)²
```
optimized with weight λ∈{1,2,10} on the squared-error term (not a Lagrange
multiplier — a fixed-weight regression toward a target rate). This is a
useful frame for the current setting: it treats the *target rate itself* as
the tunable, and converts "what KL weight do I use" into "what KL/bitrate do
I want," which is measurable and can be swept.
[Seetharaman & Kumar full text, fetched](https://arxiv.org/pdf/2605.17085)

**Delta-VAE (Razavi, van den Oord, Poole & Vinyals, *Preventing Posterior
Collapse with δ-VAEs*, ICLR 2019, arXiv:1901.03416).** Instead of a loss-side
hinge, constrains the *variational family itself* so that
`min_{θ,φ} KL(q_φ(z|x)‖p_θ(z)) ≥ δ` by construction — the minimum achievable
KL, over all choices of the posterior parameters, is bounded below `δ`
("committed rate"), so gradient descent cannot get to `KL=0` no matter what it
does. Two constructions given:
- **Sequential / AR(1) prior + independent posterior**: posterior per
  timestep `N(μ_t,σ_t)`, prior an AR(1) process `z_t=αz_{t-1}+ε_t` with
  `σ_ε=√(1−α²)`. The mismatch between an independent posterior and a
  temporally-correlated prior gives a *provable* lower bound
  `KL ≥ 1/2 Σ_k[(n−2)ln(1+α_k²) − ln(1−α_k²)]` (n = sequence length, k
  indexes latent dims per step) — δ is set by solving this for α.
- **Independent δ-VAE** (Gaussian posterior vs. standard Gaussian prior,
  no temporal structure): `μ_q² ≥ 2δ + 1 + ln(σ_q²) − σ_q²`, solved
  numerically for the feasible `(μ_q,σ_q)` range, i.e. the encoder's own
  output is reparameterized to be structurally incapable of reaching `q=p`.
  Table 3 of the paper: independent δ-VAE with δ=0.08 gets test ELBO 3.08
  (KL 0.08) and 66% linear-probe accuracy on CIFAR-10, vs. the *temporal*
  AR(1) construction at the same target rate getting a better ELBO (3.02)
  at similar accuracy (65%) — i.e. the temporal correlation structure of
  the prior mattered more than the raw target-rate number for
  density-modeling quality, though not much for representation quality.

Results (CIFAR-10, PixelSNAIL decoder): δ-VAE + AR(1) prior reached 2.85
bits/dim at committed rate 0.02 bits/dim (~61 bits/image); with a *learned
auxiliary prior* over the aggregate posterior (an LSTM matching `q(z)`,
trained without touching encoder/decoder — see §4), 2.83 bits/dim at 0.01
bits/dim rate, essentially matching the pure-autoregressive baseline (2.83)
while still using the latent. Compared head-to-head against β-VAE and
free-bits at matched rates (Fig. 4a/b), **δ-VAE dominates the
rate–distortion frontier on held-out data** and gets better downstream
linear-classification accuracy at the same rate; β-VAE and free-bits needed
"considerable effort" to converge or avoid collapse, and with linear KL
annealing the authors "were not able to train a model with significant usage
of latent variables" over a wide range of end-of-schedule steps — the KL
collapsed as soon as the weight approached 1.0. This is the strongest
head-to-head comparison found in the search: same decoder, same dataset,
annealing < free bits < delta-VAE in stability, with delta-VAE the only one
that did not require hyperparameter search per run.
[Razavi et al. 2019 full text, fetched](https://arxiv.org/pdf/1901.03416)

**GECO (Rezende & Viola, *Taming VAEs*, arXiv:1810.00597, presented at the
2018 NeurIPS Bayesian Deep Learning workshop as "Generalized ELBO with
Constrained Optimization").** Rephrases training as a *constrained*
optimization: minimize the rate term subject to reconstruction distortion
staying under a tolerance `κ` (rather than the reverse), solved via a
Lagrange multiplier `λ` on the reconstruction constraint that is updated
online (typically an exponential moving average of the constraint
violation) so `λ` self-tunes instead of being fixed by hand. Razavi et al.
note as a caveat that the resulting `λ` "does not necessarily approach one,"
so the optimized objective may not remain a valid ELBO lower bound during
some of training — a real trade-off against its main selling point (no
manual annealing schedule to tune). *Gap: could not verify GECO's exact
Lagrangian update equation or its reported numbers from primary text — the
fetched PDF did not parse; the summary above is reconstructed from
Razavi et al.'s description of GECO and from secondary-source search
results, not from Rezende & Viola directly. Treat the mechanism description
as reliable (cross-confirmed by two independent papers) but the *numbers* as
unverified.*

**Practical summary of what's established vs. folklore for §2:**
- Established (primary-source, numeric): free bits values that helped on
  CIFAR-10/IAF (λ∈[0.125,2]); free bits' known brittleness relative to
  delta-VAE and target-KL regression, reported independently by two
  different groups eight years apart (Razavi et al. 2019, Seetharaman &
  Kumar 2026); delta-VAE's formulas and its CIFAR-10/LM1B numbers.
- Folklore / practitioner consensus, not a controlled study: "GECO's
  self-tuning λ is easier to operate than fixed free bits in production
  pipelines" — repeated in secondary sources but not demonstrated
  head-to-head against free bits or delta-VAE in the primary GECO material
  reachable here.

---

## 3. KL annealing: monotonic vs. cyclical

**Monotonic annealing** originates with Bowman, Vilnis, Vinyals, Dai,
Jozefowicz & Bengio, *Generating Sentences from a Continuous Space* (CoNLL
2016, arXiv:1511.06349): anneal the KL weight linearly from 0 to 1 over
roughly the first half of training (paired with word/input dropout on the
decoder as a second, independent lever). Their own characterization of what
happens during the ramp: "The KL spikes early in training while the model
can encode information in z cheaply, then drops substantially once it
begins paying the full KL divergence penalty, and finally slowly rises
again before converging" — i.e. even the paper that introduced the trick
observed the KL non-monotonically dip mid-schedule, which is the seed
observation behind cyclical annealing (below). Multiple later papers (Kingma
et al. 2016 Appendix C.8; Razavi et al. 2019 §4) report that this schedule
is **sensitive to the exact end-of-warm-up step** and can still collapse
regardless of the chosen schedule length, especially with a powerful
decoder.

**Cyclical annealing (Fu, Li, Liu, Gao, Celikyilmaz & Carin, *Cyclical
Annealing Schedule: A Simple Approach to Mitigating KL Vanishing*, NAACL
2019, arXiv:1903.10145).** Formula (verified via full-text fetch):

```
β_t = f(τ) if τ ≤ R, else 1;   τ = mod(t−1, ⌈T/M⌉) / (T/M)
```
with `M` = number of cycles (default 4) and `R` = the fraction of each cycle
spent ramping β from 0 to 1 (default 0.5, i.e. ramp for half the cycle, then
hold β=1 for the other half) — so a schedule with the *same total training
length* as a monotonic one is instead split into `M` repeated warm-ups. Their
argument: each new cycle restarts from a decoder that has already learned to
use a partially-informative `z` from the previous cycle (a warm restart),
rather than the very weak decoder faced by the very first ramp.
Reported numbers (language modeling, Penn Treebank, monotonic 10-epoch ramp
over a 40-epoch run vs. 4-cycle schedule of the same total length):
monotonic KL 0.858 nats / perplexity 103.41 vs. cyclical KL 1.457 nats /
perplexity 101.30 — i.e. cyclical held onto roughly 70% more KL and got
better perplexity in the same compute budget. On dialogue response
generation (Switchboard) the gap was much larger: monotonic KL 0.265 /
reconstruction perplexity 36.16 vs. cyclical KL 4.104 / perplexity 29.77.
The paper reports no active-units table; its evidence is KL-nats and
downstream task metrics (perplexity, and qualitative latent-space
clustering).
[Fu et al. 2019 full text, fetched](https://arxiv.org/pdf/1903.10145)

**Warm-up length in practice:** no single number recurs as a rule; 30–50%
of total training appears repeatedly across the papers surveyed here
(Bowman et al.: ~50%; the project's own 30% is within that band), but every
paper that reports a schedule also reports sensitivity to the exact
end-point, which is the underlying reason free bits / delta-VAE / target-KL
were developed as schedule-free alternatives.

---

## 4. Aggregate-posterior / prior mismatch ("posterior holes")

**Terminology and mechanism.** The "posterior holes" name is due to Rezende
& Viola, *Taming VAEs* (2018) — regions of the prior `p(z)` with high
density that the aggregate posterior `q(z) = E_x[q(z|x)]` essentially never
visits, so the decoder was never trained on samples from there; drawing
`z~p(z)` at generation time can land in a hole and decode badly even though
reconstruction (encode-then-decode, always inside the aggregate posterior's
support) looks fine. Hoffman & Johnson's ELBO-surgery decomposition (§1)
gives the exact quantity to watch: the **marginal KL term** `KL(q(z)‖p(z))`
in `KL_total = I_q(x;z) + KL(q(z)‖p(z))`. A model can have `q(z) ≈ N(0,I)`
in *aggregate* first- and second-moment terms (mean 0, unit variance
averaged over the dataset) while still having a badly-shaped `q(z)` locally
— aggregate moment-matching is a much weaker condition than distributional
match, and moment-matching is exactly what a closed-form Gaussian KL term
checks per-dimension. This is the most directly relevant point to the
project's own diagnostic (aggregate latent ≈ N(0,I) at KL weight 1e-4): matching
the first two moments does not rule out a prior hole, because the KL term
being small only forces the *marginal* Gaussian KL to be small, not that
`q(z)` looks like an isotropic Gaussian in higher-order structure (e.g. it
could be a thin, curved, or multimodal manifold whose per-dimension mean
and variance still average out to 0 and 1). This distinction — moment match
vs. distributional match — is established (it follows directly from the
definition of the KL term for a factorized Gaussian) but the paper's own
diagnosis of *which* case applies at 1e-4 was not something this search
could adjudicate from outside literature; it requires either (a) sampling
`z~N(0,I)` and decoding, compared against decoding real encoded `z`, or (b)
a held-out nearest-neighbor / density check of `q(z)` against `N(0,I)`
beyond first two moments.

**Standard fixes, as described in primary/secondary sources found:**

- **Learned auxiliary prior matching the aggregate posterior** (used by
  Razavi et al., delta-VAE, §2): train a separate autoregressive model
  (a single-layer LSTM with conditional-Gaussian outputs, in their case) on
  samples from `q(z|x)` *after* the encoder/decoder are otherwise fixed, and
  sample from that auxiliary prior instead of the fixed `N(0,1)` prior at
  generation time. In their CIFAR-10 ablation this closed a visible gap: at
  a high committed rate, samples drawn from the fixed AR(1) prior were
  "too smooth compared to natural images" because of the prior/aggregate-
  posterior gap, while the auxiliary-prior samples did not show this
  artifact (Fig. 10 caption, delta-VAE paper). It also reduced the *rate*
  needed to hit the same NLL by "more than 50% (30 bits per image)" on
  CIFAR-10, i.e. much of the raw KL/rate was going toward correcting for the
  fixed prior's mismatch rather than toward useful compression. This
  approach does not touch the ELBO training of the encoder/decoder at all —
  it is a **post-hoc, ex-post density estimation** step exactly in the sense
  the prompt named.
- **Two-stage VAE (Dai & Wipf, *Diagnosing and Enhancing VAE Models*, ICLR
  2019)**: train a *second* VAE whose input/output is the first VAE's latent
  code, so the second stage learns to sample directly from (an approximation
  of) the true aggregate posterior `q(z)` rather than from the fixed prior,
  bypassing the mismatch by construction rather than correcting for it.
  *Gap: could not verify this paper's numeric results or its diagnostic
  statistic from primary text — both PDF fetches for arXiv:1903.05789
  returned unparseable binary content; the description above is from
  secondary sources (search-engine summaries) only, moderate confidence on
  the mechanism, no verified numbers.*
- **VampPrior (Tomczak & Welling, AISTATS 2018, arXiv:1705.07120)**:
  replace the fixed prior with a mixture of the posteriors evaluated at `K`
  learned "pseudo-input" points, `p(z) = 1/K Σ_k q(z|u_k)`, where `u_k` are
  themselves trainable parameters (initialized to look like data). Because
  each mixture component is literally a posterior the encoder can produce,
  the prior is constructed to lie inside the encoder's own reachable set —
  a mismatch-by-construction fix rather than a post-hoc one. This is one of
  the papers Razavi et al. cite as motivating their own auxiliary-prior
  fallback. *Gap: only the mechanism was verified via secondary sources;
  the paper's own reconstruction/FID numbers were not fetched from primary
  text.*
- **Flow prior / learned prior more generally**: replacing `N(0,I)` with a
  normalizing-flow density (IAF *in the prior direction*, or other flows)
  is the same fix as VampPrior in spirit — give the prior enough capacity
  to match `q(z)` rather than the reverse. Not independently verified with
  numbers here beyond the IAF paper's own use of flows on the *posterior*
  side (§2/Appendix D of Kingma et al. 2016 notes that an IAF-flexible
  posterior with a factorized prior is mathematically equivalent to a
  factorized posterior with an autoregressive prior — the two fixes,
  flexible-posterior and flexible-prior, are formally interchangeable in
  their construction, which is a useful established fact when deciding
  which side of the model to spend flexibility budget on).

---

## 5. Per-token / per-position latent width

**No paper found treats "2–4 channels per token/position" as a named
design pattern with its own literature**; the evidence is indirect, from two
adjacent lines of work — image/video tokenizer channel-count ablations, and
one directly relevant audio-VAE paper that uses a per-frame (not quite
per-token, but the closest analogue found) latent.

**Image tokenizer channel counts, verified numbers:**
- SD1.x/SDXL VAE: 4 channels at 8× spatial downsampling per side (48×
  compression counting all 3 color channels → 4 latent channels).
- SD3 (Esser et al., *Scaling Rectified Flow Transformers for
  High-Resolution Image Synthesis*, arXiv:2403.03206, Table 3, **verified
  via full-text fetch**): ablated d=4, 8, 16 latent channels at fixed spatial
  compression. Reconstruction FID: **d=4 → 2.41, d=8 → 1.56, d=16 → 1.06**
  — reconstruction quality improves monotonically and substantially with
  channel count in their range, and "the d=16 autoencoder exhibits better
  scaling performance" as the downstream diffusion model is scaled up. SD3
  shipped with 16 channels.
- FLUX.1/FLUX.2: 32-channel VAE (only 6× overall compression) — informal
  source (a practitioner's technical-notes gist on the SD-family VAEs,
  madebyollin), not a paper; treat as directionally reliable but not
  peer-reviewed evidence.
- The same gist's independent, practically important observation: **"SD VAE
  KL noise has very little effect… variances are really small… taking the
  encoder mean instead of sampling causes no difference in most cases."**
  This is the *opposite* regime from the project's own measurement (sigma
  dominating mu): in the widely-deployed SD-family image VAEs, the trained
  posterior std is reported as near-zero almost everywhere, i.e. those
  VAEs are (in practice) close to deterministic autoencoders wearing a thin
  KL regularizer, not models that need the noise at decode time. *This is a
  single informal source, not a controlled study — flagged as
  practitioner folklore, but it is a load-bearing enough claim (and
  reproducible by any reader who checks a public SD checkpoint) that it is
  included with that caveat rather than omitted.*

**Video tokenizer channel counts:** Cosmos Tokenizer (NVIDIA) ships 16-channel
continuous variants (`C=16`); CogVideoX's continuous VAE is also described
in secondary sources as using "a larger number of latent channels" to
preserve information relative to 4-channel image VAEs, but exact reported
numbers/ablation tables for CogVideoX, Wan2.1 or OmniTokenizer were not
independently verified from primary text in this pass — flagged as a gap
below.

**Per-frame audio VAE, closest verified analogue to "per-token, few
channels" (Seetharaman & Kumar 2026, primary text verified):** their DAC-VAE
produces a 40 Hz frame rate (i.e. one latent frame per ~25 ms of audio,
structurally similar to "one token per unit of input") and is trained across
a bitrate sweep from ~1.8 to ~74 kbps by varying the *target KL* (hence the
*effective information per frame*, not a fixed channel count — the paper's
latent width `D=128` per frame is fixed across the sweep, only the target
rate changes). Downstream text-to-audio generation quality (Table 2, KAD/FAD
scores) was **non-monotonic in bitrate**: their target-KL≈200 (11.56 kbps)
model beat both lower-bitrate (over-regularized) and higher-bitrate
(under-regularized) models on text-audio similarity (70.67) and KAD (1.70,
best of the sweep), with both the KL=132.63 (7.65 kbps) and KL=1284.21
(74.10 kbps) endpoints scoring worse. For text-to-speech (Table 3) the
pattern was similar but not identical — lower-bitrate VAEs mostly gave
better WER and speaker similarity, with one exception the authors could not
fully explain (a high-bitrate model with low WER but "less natural, more
monotonous" speech on listening). Their qualitative conclusion: reconstruction
quality and generative-model-friendliness trade off, and the trade-off has
an interior optimum found only by sweeping, not by an a priori channel-count
rule.

**Conclusion for this question:** the literature's consistent finding is
that reconstruction quality increases monotonically (sometimes with
diminishing but still positive returns) with channel/dimension count per
token/frame — nobody found a paper reporting reconstruction quality
*peaking* at a small per-token width and then declining as width increases
further. What trades off against width is exclusively *downstream
generative-model quality* (harder to model a higher-dimensional, less
compressed latent; the Adobe audio paper explicitly found a downstream-task
optimum away from both extremes of their bitrate sweep). There is no
evidence located that 2–4 dims/token is a "known-good" width in some
architecture-independent sense; every paper found treats width as a
per-application sweep variable, not a fixed target.

---

## 6. Decoding the posterior mean vs. a sample

Two directly relevant, mutually opposite data points were found, plus the
general "blurriness" explanation:

- **SD-family image VAEs (informal source, madebyollin gist):** posterior
  std is reported as extremely small essentially everywhere after training,
  so decoding the mean vs. a sample is reported to make "no difference in
  most cases" — the model has, in effect, learned to make the KL penalty
  nearly free by collapsing `σ→0` almost everywhere rather than by using the
  noise. This is the well-known "the decoder never needed the noise, so the
  encoder minimized the KL cost by shrinking σ" failure mode, though the
  gist does not use "collapse" terminology and it is *not* full posterior
  collapse (μ is still informative — the model is not ignoring z, it is
  ignoring the *stochasticity* of z).
- **General VAE blur explanation** (multiple secondary sources, consistent
  with textbook VAE theory, not attributable to one paper): a Gaussian
  decoder trained under expected reconstruction loss over `z~q(z|x)` is
  implicitly asked to be good *in expectation* over the posterior's spread,
  which for a genuinely broad (non-collapsed) posterior pulls the decoder
  toward an average-case, smoothed-out output — i.e. the well-known
  "VAEs are blurry" phenomenon is usually attributed to the reconstruction
  loss (Gaussian/L2, which is optimized by the conditional mean) rather
  than specifically to sampling noise at decode time, though the two
  interact: a decoder that has to reconstruct well under many different
  noise draws for the same `x` is pushed toward outputs that are safe
  under that spread.
- No paper was found that frames "**decoding the mean is much worse than
  decoding a sample**" (the reverse of the SD-VAE finding, and the
  direction implied by the project's own high-sigma setting) as a named,
  independently-studied phenomenon. The nearest primary-literature contact
  point is the general rate–distortion framing (§1, Alemi et al.): if a
  model sits at a point on the rate–distortion curve with high rate
  (σ still carrying real, input-independent-looking variance that the
  decoder has learned to read as a *signal*, not noise — because the
  decoder was never trained against `z=μ`), then `z=μ` is out-of-distribution
  for that decoder relative to what it saw in training (which was always
  `z~q(z|x)`, noise included), and a decoder that has come to rely on
  matching the *training-time sampling distribution* exactly could
  plausibly degrade when fed the (unseen at training time) deterministic
  mean. This is a plausible mechanism consistent with rate–distortion
  theory and with how VAEs are trained (reconstruction loss is always
  evaluated on a *sample*, never on the mean, unless the practitioner adds
  an explicit mean-decoding term), but it was not found stated or measured
  as its own named result in any source reached in this search.

---

## Established vs. folklore — summary

**Established (primary-source, numeric, reproducible from the cited
paper):**
- AU statistic definition and threshold (Burda et al. 2016).
- ELBO surgery's exact MI/marginal-KL decomposition (Hoffman & Johnson
  2016) and the rate–distortion framing (Alemi et al. 2018).
- Free bits' exact formula and the λ range that helped on CIFAR-10/IAF
  (Kingma et al. 2016).
- delta-VAE's exact formulas, its CIFAR-10/ImageNet/LM1B numbers, and its
  documented head-to-head instability comparison against free bits and
  β-VAE with a fixed schedule (Razavi et al. 2019).
- Cyclical annealing's exact schedule formula and its PTB/Switchboard
  numbers (Fu et al. 2019).
- Target-KL regularization's formula, and its bitrate sweep + non-monotonic
  downstream-quality finding on audio (Seetharaman & Kumar 2026).
- SD3's channel-count reconstruction-FID ablation table (Esser et al.
  2024).
- "Posterior holes" terminology and the auxiliary-prior fix's measured
  effect on CIFAR-10 sample quality and required rate (Razavi et al. 2019,
  citing Rezende & Viola 2018 for the term itself).

**Folklore / informal / unverified-from-primary-text (flagged inline
above, repeated here for visibility):**
- GECO's exact Lagrangian update rule and its own reported numbers
  (mechanism cross-confirmed by two papers citing it; numbers not
  independently verified here).
- Two-stage VAE's (Dai & Wipf) diagnostic statistic and numeric results
  (secondary sources only).
- VampPrior's own reconstruction numbers (mechanism verified, numbers not).
- "SD-family VAEs have near-zero posterior std / mean-decoding ≈
  sample-decoding" (one informal but detailed and checkable practitioner
  source, not a peer-reviewed measurement).
- CogVideoX / Wan2.1 / OmniTokenizer exact channel counts and reconstruction
  ablations (secondary-source claims only, not independently verified).
- "Decoding the mean is much worse than decoding a sample" as a named,
  independently-studied phenomenon: **not found in the literature reached
  by this search** — the rate–distortion argument above is a plausible
  inference, not a reported/measured result.

---

## Gaps

1. **GECO primary text (Rezende & Viola, *Taming VAEs*, arXiv:1810.00597)**
   did not parse from the fetched PDF (binary/compressed stream issue on
   this fetch tool) — its Lagrangian update equation and numeric results are
   reported here only via Razavi et al.'s (2019) description and secondary
   search summaries. If GECO specifically becomes relevant, re-fetch and
   read primary text directly.
2. **Two-stage VAE (Dai & Wipf, ICLR 2019, arXiv:1903.05789)** — same
   fetch failure. Its diagnostic statistic for the prior/aggregate-posterior
   gap and its numeric FID results are unverified beyond a one-paragraph
   secondary-source mechanism description.
3. **VampPrior (Tomczak & Welling 2018)** — mechanism verified via
   secondary sources and via its citation in Razavi et al., but its own
   MNIST/OMNIGLOT/Caltech numbers were not fetched from primary text.
4. **Video tokenizer channel-count ablations** (CogVideoX, Wan2.1,
   OmniTokenizer, Cosmos) — only Cosmos's C=16 configuration was confirmed
   with any specificity; no controlled ablation table (analogous to SD3's
   Table 3) was located for a video tokenizer varying channel count with
   reconstruction and downstream-FID held out. This is the weakest-evidenced
   part of §5 and the part most directly on-point for a per-token,
   video-adjacent design question.
5. **"Decoding the mean is much worse than a sample" as a named phenomenon**
   — not located as an independently-reported/measured result anywhere in
   this search; the write-up above states the best available *inference*
   from rate–distortion theory, not a citation to a paper that measured it
   directly. A targeted follow-up search phrase to try next time: "posterior
   collapse decoder relies on injected noise" / "deterministic decoding
   degrades VAE" / "z=mu ablation reconstruction."
6. **Mutual information estimators in practice** — the three approaches
   named in §1 (Hoffman–Johnson exact decomposition, Alemi et al.
   variational bounds, downstream linear-probe accuracy as in delta-VAE) were
   identified, but no paper's *specific numeric MI estimate* (in nats, on a
   concrete model) was pulled into these notes; only the AU count and raw
   KL/dim numbers are quoted with actual figures throughout.
7. **All numbers in this file trace to the searches and fetches run in this
   single research pass** (2026-09-21); no cross-check against a second,
   independent reading of the same primary sources was performed, and no
   agent-of-record other than the one that ran this search reviewed the
   fetched PDFs before extraction.
