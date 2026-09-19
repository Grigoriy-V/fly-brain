# Video generative models: temporal representation and conditioning — what transfers to a compressed, non-pixel signal

Researched 2026-09-20. Five questions about video generative models — temporal
representation, safe temporal compression, class conditioning, capacity vs. data at
small scale, evaluation of unconditional samples beyond FID — read for what transfers
to this project's own setting: an unconditional flow-matching transformer
(`SiTStates`/`SiTBlock`, `flydream/generate/prior17.py` and `gen13b.py`: a pre-norm
transformer with adaLN-Zero, architecturally a small DiT/SiT clone) over a
(16, 8, 721) tensor — 16 of 40 DCT coefficients on the time axis, 8 T4/T5 cell types,
721 spatial columns — trained unconditionally on ~13,500 trajectories, ~1.4M
parameters at width 128 / depth 4 (ROADMAP 17.1/18.3, DECISIONS 2026-09-20).

**Method note.** Five parallel research agents were launched, one per question.
Before four returned, scope was corrected mid-session to "write from what is already
held, no further fan-out"; those four (Q1, Q2, Q4, Q5) were stopped and are not used
below. The class-conditioning agent (§3) had already completed and is used in full,
its central numbers cross-checked against two direct re-fetches of the same primary
sources. §1, §2, §4 and §5 rest on this author's own direct source-checking, narrower
than a dedicated pass — flagged per claim, and listed again in Gaps.

**Marking convention.** A linked claim was read directly this session (quote or table
value taken from the fetched page). **[†unverified]** marks this author's own
training-knowledge recollection of a published fact, not re-confirmed against the
primary source this session: treat the paper/venue as probably right, the exact digit
as needing a check before it drives a decision.

## 1. How video generators represent time

**Our own architecture is already in this family.** `SiTBlock`
(`flydream/generate/gen13b.py:158`) is a pre-LN transformer block with full
self-attention and an explicit `# adaLN-Zero` modulation — the same block DiT and SiT
use, tokenizing spatial columns instead of image patches. DiT/SiT findings below are
an architectural match; only the token content differs.

**Pixel video, two camps.** *3D/causal-VAE tokenizers* compress space **and time**
together before a transformer sees the clip: CogVideoX's own abstract states "a 3D
Variational Autoencoder (VAE) to compress videos along both spatial and temporal
dimensions" — [Yang et al., "CogVideoX," ICLR 2025, arXiv:2408.06072](https://arxiv.org/abs/2408.06072) (title/authors/venue confirmed by direct fetch; exact ratio not extracted from the abstract this session — commonly cited elsewhere as ~4× temporal × 8×8 spatial, **[†unverified]**). MAGVIT-v2 and W.A.L.T. are usually described as following the same causal-3D-CNN recipe — [Yu et al., ICLR 2024, arXiv:2310.05737](https://arxiv.org/abs/2310.05737); [Gupta et al., ECCV 2024, arXiv:2312.06662](https://arxiv.org/abs/2312.06662) — **neither fetched this session; citation and digits both [†unverified]**. *Frame-wise latents with no temporal VAE compression*: Stable Video Diffusion is usually described as encoding each frame independently (SD's 8×8-spatial image VAE) and putting all temporal modeling into added attention/conv layers — [Blattmann et al., arXiv:2311.15127](https://arxiv.org/abs/2311.15127) — **also not fetched this session; [†unverified]**, though the qualitative architecture split (VAE compresses space only; time is handled by attention on top) is a well-known, higher-confidence fact than any specific digit.

**Sora** (OpenAI): compresses a clip "both temporally and spatially" into a latent,
then cuts that latent into "spacetime patches" fed to a transformer — confirmed via
several secondary summaries of the technical report (a direct fetch of openai.com
returned HTTP 403 this session). **No compression factor, patch count, parameter
count, or dataset size is published anywhere in the report** — [OpenAI, "Video generation models as world simulators," 2024](https://openai.com/index/video-generation-models-as-world-simulators/); a reverse-engineering survey exists but no numbers were extracted from it this session — [Liu et al., arXiv:2402.17177](https://arxiv.org/abs/2402.17177). Even the highest-profile example is a source for the *idea* (compress, then patchify), not for numbers.

**Fixed (non-learned) temporal bases are not a pixel-video habit — they are standard
for short, smooth, low-dimensional trajectories, which is closer to our actual
regime:**
- **Human motion prediction** represents a joint trajectory in DCT space instead of
  frame space, one coefficient per temporal frequency across the whole sequence, fed
  to a graph convolutional network — [Mao, Liu, Salzmann, Li, "Learning Trajectory Dependencies for Human Motion Prediction," ICCV 2019 (oral), arXiv:1908.05436](https://arxiv.org/abs/1908.05436) (title/authors/venue confirmed by direct fetch). Became a standard representation in that field; a precise "K coefficients → error" ablation table was not located this session (see Gaps).
- **Robot action trajectories**: "Robot action trajectories are highly smooth, with
  most energy concentrated in a few low-frequency discrete cosine transform modes" —
  confirmed by direct fetch of the full text, with measured numbers: first 2 DCT
  modes = 98.5% of energy (>80% on every task tested), k=1 alone = 93.2%; on a
  synthetic benchmark, 6 of 64 modes = 99% — [Zhang et al., "Hyper-DP3: Frequency-Aware Right-Sizing of 3D Diffusion Policies for Visuomotor Control," arXiv:2605.01581 (2026)](https://arxiv.org/abs/2605.01581). Capacity result in §4.
- **Real brain signal, spectral transform, flow matching** — the closest domain match
  found: fMRI BOLD time series (116 ROIs × 232 steps, MDD; 100 ROIs × 200 steps,
  ABIDE) are wavelet-decomposed (5-level DWT) into a 2D coefficient map, then
  block-DCT'd (block size 4), and a **flow-matching** U-ViT generates in that spectral
  space before inverse-DCT/DWT recovers the signal — [Tew et al., "Functional MRI Time Series Generation via Wavelet-Based Image Transform and Spectral Flow Matching for Brain Disorder Identification," ICLR 2026, arXiv:2605.30387](https://arxiv.org/abs/2605.30387) (confirmed by direct fetch of abstract + HTML). Differs from our approach in an important way: its DCT step is JPEG-style (compressing a wavelet map), not a hard top-K-of-N truncation of raw time as ours is — no compression-factor number is published for it (checked, not found), and no parameter count is published either (checked, not found).

**No head-to-head fixed-vs-learned temporal-basis comparison was found**, in pixel
video or in the three adjacent domains above; each domain picked one approach and
reported results for it alone. Treat our own K=8/12/16 sweep (below, §2) as one of
the only such comparisons that exists anywhere, fixed-only though it is.

**Transfer flags.** 3D-VAE tokenizer numbers assume pixel redundancy (RGB, spatial
smoothness exploited by conv kernels) and do not obviously bound a 721-node graph
signal. LPIPS-family perceptual metrics anywhere in this literature are trained on
natural images and are not meaningful for T4/T5 activity. The DCT-trajectory
precedents (Mao et al., Hyper-DP3, DSFM) are the load-bearing transfer evidence for
this question, not the pixel-video tokenizers.

## 2. How much temporal compression is safe — factor vs. quality

The literature-wide table the task asked for could not be built at full rigor this
session (the assigned agent was stopped before returning); what follows is what could
be checked directly, plus this project's own already-measured numbers.

| Source | Domain | Compression | Metric | Value |
|---|---|---|---|---|
| This project, `reports/2026-09-20_step17_1b_scene_dct_prior.md` | our T4/T5 states, DCT | K=8 of 40 | round trip | 0.58 |
| ″ | ″ | K=12 of 40 | ″ | 0.20 |
| ″ | ″ | **K=16 of 40 (99.3% energy, in use)** | ″ | **0.045** |
| ″ | ″ | K=40 (no compression) | ″ | 0.019 (floor) |
| This project, `reports/2026-09-20_step18_3_prior_on_the_corpus.md` | corpus states, DCT-16 | K=16 of 40 (99.32% energy) | round trip, generated samples | 0.095 (floor alone 0.021) |
| Hyper-DP3, arXiv:2605.01581 | robot action trajectories | 2 of ~20-64 modes | fraction of energy | 98.5% (task-avg; >80% worst task) |
| ″ | synthetic "lowfreq" benchmark | 6 of 64 modes | fraction of energy | 99% |
| CogVideoX, arXiv:2408.06072 | pixel video, 3D causal VAE | not extracted this session | — | **[†unverified]**, commonly cited ~4× temporal |
| MAGVIT-v2 / W.A.L.T. | pixel video, causal 3D-CNN tokenizer | not extracted this session | rFID | **[†unverified]** |
| SVD, arXiv:2311.15127 | pixel video, per-frame VAE | 1× temporal (no compression) | — | architecture fact, digits unverified |

**Reading this table.** Our own rows are the only controlled factor-vs-quality sweep
on one fixed downstream metric: K=16 is not marginal (8× better than K=8, only 2.4×
worse than no truncation) but is not free either — the DCT-16 "floor," re-measured on
the corpus as 0.021, is named in ROADMAP 18.3 as a live suspect for the residual gap
between a sample and a real trajectory. Hyper-DP3's numbers are energy-concentration,
not a reconstruction-quality curve — robot actions, not our signal — but point the
same direction, and our own table already shows energy kept understates the cost of
truncation on a downstream metric (16/40 keeps 99.3% of energy, still costs 2.4× the
round trip of no truncation at all). The pixel-tokenizer rows could not be filled with
checked numbers this session — see Gaps.

**Transfer flag.** PSNR/rFID/FVD measure something different in kind from a round
trip through a frozen brain model; a number from one is not a threshold for the
other, only a rough sense of how far tokenizer designers have pushed this lever.

## 3. Class/label conditioning — gains, and behaviour at ~100 examples/class

Covered by the one completed research agent; central numbers were additionally
cross-checked by this author against the DiT and SiT arXiv pages directly.

**Conditioning alone, matched architecture/compute — the cleanest ablation found:**
[Dhariwal & Nichol, "Diffusion Models Beat GANs on Image Synthesis," NeurIPS 2021, arXiv:2105.05233](https://arxiv.org/abs/2105.05233), Table 4, same ADM U-Net, same 2M-iteration budget, ImageNet 256×256, **no guidance on either side**: unconditional FID 26.21 vs. class-conditional FID **10.94** — a **2.4× improvement from the label alone**, all else fixed.

**DiT / SiT contain no unconditional-vs-conditional ablation at all** (confirmed by
full-text search of both papers) — their "conditioning helps" story is really a
guidance-scale story, CFG on vs. off inside an already class-conditional network:
[Peebles & Xie, "Scalable Diffusion Models with Transformers," ICCV 2023, arXiv:2212.09748](https://arxiv.org/abs/2212.09748), Table 2, DiT-XL/2 FID 9.62 (no CFG) → 2.27 (cfg=1.5); model-size ablation (Table 4, 400K steps, no CFG, all class-conditional): DiT-S/2 68.40, DiT-B/2 43.47, DiT-L/2 23.33, DiT-XL/2 19.47 — DiT-S/2 is the smallest published DiT config (~33M parameters per the paper's own architecture table, **[†unverified]** — param count not independently re-confirmed this session), still ~24× our 1.4M. [Ma et al., "SiT," ECCV 2024, arXiv:2401.08740](https://arxiv.org/abs/2401.08740): SiT-XL FID 2.06 vs. DiT-XL 2.27 at matched size/Gflops (cfg=1.5) — again purely conditional.

**More classes, not fewer, helped more** (same architecture, EDM, conditioning the
only variable) — [Adaloglou et al., "Rethinking cluster-conditioned diffusion models for label-free image synthesis," WACV 2025, arXiv:2403.00570](https://arxiv.org/abs/2403.00570), Table 1: CIFAR-10 (10 classes, ~5,000 img/class) FID 2.07→1.81 (13% relative gain); CIFAR-100 (100 classes, ~500 img/class) FID 3.41→2.21 (35% relative gain). Granularity has a ceiling, though: past an optimal number of classes/clusters for the dataset size, quality degrades and samples drift out-of-distribution (their Figure 3).

**At ~100 examples/class specifically:**

| Study | Model | Regime | Result |
|---|---|---|---|
| [Shahbazi et al., ICLR 2022, arXiv:2201.06578](https://arxiv.org/abs/2201.06578) | **GAN** (StyleGAN2/BigGAN) | 20 classes × 100 img/class = 2,000 total | Naive conditioning **worse**, e.g. FID 23 (uncond) vs. 100 (cond) on one benchmark; crossover to "conditioning helps" only above ~5,000 total images |
| [Giannone et al., "Few-Shot Diffusion Models," arXiv:2205.15463](https://arxiv.org/abs/2205.15463) | Diffusion, meta-learned | 5 examples/class at test time, ample data at meta-train time | Conditional beats unconditional widely (e.g. FID 35 vs. 63) — not a from-scratch small-data result |
| [You et al., "DPT," NeurIPS 2023, arXiv:2302.10586](https://arxiv.org/abs/2302.10586) | Diffusion, 585M params | 1-5 *labeled*/class atop a full 1.28M-image unlabeled pool | FID 3.08→2.50 as labels/class rise 1→5 — semi-supervised, not small-data |

**Only Shahbazi et al. matches our per-class count (100/class)**, and found
conditioning actively harmful there — but the mechanism is a GAN discriminator
exploiting per-class structure to overfit faster, a failure mode with no counterpart
in a regression-style flow-matching loss (nothing to collapse against). No
diffusion/flow-matching study was found at anywhere near our (model size, total data,
class count) triple simultaneously.

**When conditioning hurts, beyond the GAN case**: over-fine granularity relative to
dataset size (Adaloglou et al., above); label noise — direction confirmed, exact
numbers not table-verified this session — [Na et al., "Label-Noise Robust Diffusion Models," ICLR 2024, arXiv:2402.17517](https://arxiv.org/abs/2402.17517). No study was found on *weakly-related-but-not-wrong* labels, this project's actual candidate case (a UCF101 action-class label, or a procedural-stimulus `kind`, only loosely coupled to the T4/T5 trajectory it produces) — no direct precedent located.

**Our own label inventory** (checked locally, not literature): `label_of()` in
`flydream/data/video_corpus.py` already stores a UCF101 action-class label per clip
(~101 classes over 12,411 passing clips, ~123/class) but it is unused —
`prior17.py`'s `SiTStates` takes **no condition** by design (its own docstring: "the
state as the data and **no condition**"); procedural stimuli carry a coarser `kind`
label (`{expand, contract, rotate}` / `{white, pink}`) with far fewer categories.

## 4. Capacity vs. data at small scale; width vs. depth

**DiT's own headline finding is that Gflops, not raw parameter count, is the scaling
variable that tracks FID** — confirmed directly from the abstract: "We analyze the
scalability of our Diffusion Transformers (DiTs) through the lens of forward pass
complexity as measured by Gflops" ([arXiv:2212.09748](https://arxiv.org/abs/2212.09748)). Its smallest published config, DiT-S/2, is the smallest anything in the DiT/SiT family publishes and is still ~24× our 1.4M parameters (§3); no DiT/SiT-family curve reaches near our scale, checked or otherwise.

**The strongest small-scale evidence found is out-of-domain but structurally close.**
Hyper-DP3 shrinks a diffusion-policy transformer to **2.52M parameters** (vs. 255.8M
for the DP3/Flow-Policy/MP1 baselines it compares against — **101× smaller**) by
sizing the model to a DCT-truncated trajectory rather than the raw one, and matches or
**beats** those 100×-larger baselines: RoboTwin2.0 (50 tasks) success rate 63.2% vs.
55.2% (DP3) / 41.1% (Flow Policy) / 55.2% (MP1); Adroit/MetaWorld (10 tasks) 78.4% vs.
73.0% / 72.6% / 76.7% — trained on **10-50 demonstrations per task** — [Zhang et al., arXiv:2605.01581](https://arxiv.org/abs/2605.01581) (all four numbers confirmed by direct fetch of the full text). Scale baseline: [Ze et al., "3D Diffusion Policy," RSS 2024, arXiv:2403.03954](https://arxiv.org/abs/2403.03954) itself trains from 10-40 demonstrations/task. The domain differs sharply (robot joint trajectories and task success rate, not neural population activity and sample-distribution fidelity), but the structural claim — a compact temporal representation does not by itself need a large model, and a small model built for it can beat a large model built for the raw one — is the most directly relevant capacity evidence located this session.

**Width vs. depth**: no controlled ablation isolating this at fixed parameter count
was found or verified this session for DiT/SiT specifically. What is confirmed
instead is the adjacent claim that DiT's chosen scaling axis is training-compute
Gflops (a joint function of width, depth and token count), not depth or width read
separately. The general transformer-scaling claim that "width matters more than depth
once a minimum depth is met" is **[†unverified this session]** — not checked against
a primary source here and not attached to a specific citation; treat as absent
evidence, not a supported claim, until checked.

**Our own numbers, restated for this question** (ROADMAP 18.3, DECISIONS 2026-09-20):
1.4M parameters (width 128, depth 4), ~13,500 training trajectories (~92,000 values
each after DCT-16 compression), 20,000 steps at batch 32 (≈47 epochs); validation loss
still falling at the end (0.5165→0.5064 over the last 5,500 steps, decelerating) —
consistent with either an unfinished run or a capacity ceiling, and the project's own
report says explicitly it cannot tell which from the curve shape alone. Width 192-256
(≈$0.4-0.9/run) is named as the next thing to try, not yet run. **This author's own
inference, not a citation:** Hyper-DP3's result argues against "our small model is
obviously undersized" as a default assumption once a signal is already compressed to
its compact basis — but Hyper-DP3 sizes its model to a trajectory an order of
magnitude shorter, for a pass/fail task metric rather than a distributional
sample-quality target, so this is a directional prior, not a substitute for running
the width-192-256 check the project already has queued.

## 5. Evaluating unconditional samples beyond FID

**Precision/recall and density/coverage.** [Kynkäänniemi, Karras, Laine, Lehtinen, Aila, "Improved Precision and Recall Metric for Assessing Generative Models," NeurIPS 2019, arXiv:1904.06991](https://arxiv.org/abs/1904.06991) (title/authors/venue confirmed by direct fetch) builds "explicit, non-parametric representations of the manifolds of real and generated data" — a k-NN hypersphere membership test in a feature space — and reports precision and recall separately rather than one blended number (the same Precision/Recall columns used in DiT's own tables, §3). [Naeem, Oh, Uh, Choi, Yoo, "Reliable Fidelity and Diversity Metrics for Generative Models," ICML 2020, arXiv:2002.09797](https://arxiv.org/abs/2002.09797) (author list corrected here after a direct fetch — not the authors this note initially assumed) proposes Density and Coverage because precision/recall variants "fail to detect the match between two identical distributions, are not robust against outliers, and the evaluation hyperparameters are selected arbitrarily." **Neither paper's stance on non-image feature spaces was retrieved this session** — both are conventionally run on Inception-v3 features, and whether either has been applied to a non-image, no-pretrained-extractor setting like ours was not confirmed (Gap).

**What the nearest non-image domain reports instead of FID.** DSFM (fMRI generation,
§1) reports **downstream classification accuracy** on real vs. synthetic-augmented
data (MDD 70.84±5.89%, ABIDE 71.54±1.87%) and functional-connectivity topology
preservation (edges/node-strength correlation 0.99±0.00), not FID or precision/recall
— [arXiv:2605.30387](https://arxiv.org/abs/2605.30387). EEG-signal diffusion-generation papers similarly evaluate with per-channel Pearson correlation and MSE against real signals, plus a downstream classifier trained on synthetic vs. real data — **[†weakly sourced: arXiv ids surfaced by search, not individually fetched or table-checked this session]**, e.g. arXiv:2510.17832, arXiv:2401.16878. Neither domain uses an Inception-style pretrained extractor; both fall back to signal-level correlation plus a downstream task — weak secondary evidence that a correlation-based check is the norm for neural/physiological time series, not a shortcut around a "real" metric that exists for this data type.

**Memorization / near-duplicate checks.** Standard references for pixel diffusion are
usually given as [Somepalli et al., "Diffusion Art or Digital Forgery?," CVPR 2023, arXiv:2212.03860](https://arxiv.org/abs/2212.03860) and [Carlini et al., "Extracting Training Data from Diffusion Models," USENIX Security 2023, arXiv:2301.13188](https://arxiv.org/abs/2301.13188) — **[†both unverified this session: titles/venues/ids from training knowledge, not re-confirmed by direct fetch]**. Both search an image-feature space (e.g. CLIP/SSCD embeddings) for near-duplicates above a similarity threshold; no specific percentile or threshold number is asserted here without a check.

**Our own method, for comparison** (not literature, restated for context): nearest-
training-trajectory correlation in raw signal space (no learned feature extractor),
plus a coverage proxy — inverting the flow to the noise a state came from and
comparing its norm to the ‖ε‖²/D ≈ 1.0±0.014 expected of a standard Gaussian (real
states land at 1.07-1.29; an off-manifold/shuffled state at ≈4) —
`reports/2026-09-20_step17_3b_noise_inversion.md`. This is not a published metric
under either name above; it sits closer in spirit to density/coverage (Naeem et al.)
than to precision/recall, since it asks "does this point sit where mass is expected,"
but it was not designed against either paper and has not been checked for the failure
modes (outlier sensitivity, arbitrary hyperparameters) Naeem et al. raise against
precision/recall — an open question, not answered by this note.

**FID at small sample counts** (tens to low hundreds, our actual regime, vs. FID's
usual 10k-50k): no source was checked this session. Candidates known by reputation —
Chong & Forsyth on small-N FID bias, Parmar et al. on resizing/implementation
subtleties, Google's CMMD — are **not cited here**, since none was fetched or searched
this session; listed only in Gaps as work still to do, not as claims.

## Gaps

- **Four of five assigned research agents were stopped before returning** (Q1
  pixel-tokenizer landscape, Q2 compression-factor table, Q4 capacity/width-depth, Q5
  evaluation-beyond-FID), per a scope correction mid-session. §1, §2, §4, §5 above rest
  on this author's own narrower, direct source-checking around that correction, not a
  dedicated research pass; only §3 reflects a completed dedicated pass.
- **No verified compression-factor numbers for Open-Sora, Open-Sora-Plan, LTX-Video,
  NVIDIA Cosmos Tokenizer, MAGVIT-v2, or W.A.L.T.** — the task's "ideal table" (factor
  → reconstruction metric) is filled only with this project's own states and with
  Hyper-DP3's energy-concentration numbers (not a reconstruction metric). CogVideoX's
  commonly-cited "~4×8×8," and SVD's "no temporal VAE compression," are both flagged
  †unverified-this-session despite moderate-to-high confidence.
- **No published fixed-vs-learned temporal-basis head-to-head was found** anywhere
  checked (pixel video or the three adjacent domains) — likely a genuine gap, not a
  search failure, but only three adjacent domains were checked.
- **One motion-prediction "K coefficients → error" figure surfaced by search** (first
  5 DCT coefficients: 93% of position energy vs. 37% of velocity energy) could not be
  pinned to a specific paper and table and is **omitted** above rather than cited
  unsourced.
- **No DiT/SiT-family model near our scale (1.4M params) exists in the checked
  literature** — DiT-S/2 (~33M, itself unverified) is the smallest found; no scaling
  curve below ~10M parameters was located. SD3's scaling-law plot, EDM2's capacity
  study, and any small-DiT paper were not checked this session.
- **No controlled width-vs-depth ablation at fixed parameter count** was found or
  verified for DiT/SiT/ViT-style models; the transformer-scaling folklore that width
  matters more once a minimum depth is met is not cited here because it was not
  checked against a primary source.
- **DSFM (arXiv:2605.30387) has no published parameter count and no published hard
  truncation ratio** — the closest domain match found is incomplete on exactly the two
  numbers (size, ratio) most wanted here.
- **No study was found on weakly/loosely-related (as opposed to absent, random, or
  corrupted) conditioning labels** — this project's actual candidate case has no
  direct precedent in anything checked.
- **No diffusion/flow-matching analogue of Shahbazi et al.'s GAN "conditioning
  collapses below ~5,000 images" result** — whether a non-adversarial objective shows
  the same pattern, and at what data size, is untested as far as this search reached.
- **Kynkäänniemi et al. and Naeem et al.'s applicability to non-image feature spaces**
  was not retrieved from either paper's own text this session; both are conventionally
  run on Inception-v3 features.
- **Somepalli et al. and Carlini et al.** are cited from training knowledge only,
  not re-confirmed by direct fetch this session; no threshold/percentile number from
  either is asserted here.
- **FID-at-small-sample-count literature** (Chong & Forsyth, Parmar et al., CMMD) was
  not checked at all this session — named only as work still to do.
- **EEG-generation correlation/MSE evaluation practice** rests on search-result
  summaries of two papers, neither individually fetched or table-checked — treated as
  weak, directional evidence only.
