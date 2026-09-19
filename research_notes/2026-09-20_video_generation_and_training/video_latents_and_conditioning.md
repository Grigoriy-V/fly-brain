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

**Method note.** Five parallel research agents were launched, one per question. Scope
was corrected mid-session to stop launching further agents and write from what was
already held; a stop request was then sent to each still-running agent, in the spirit
of that correction, but the harness gave this session no permission to actually
terminate them. Four completed on their own regardless and are used in full below —
the compression-factor table (§2), class conditioning (§3), capacity vs. data (§4),
evaluation beyond FID (§5) — each cross-checked by this author against at least one
primary source directly. Only the temporal-representation agent (§1) never delivered
a result; whether it completed without a notification, errored, or is still running
is unknown. §1 rests on this author's own narrower, direct source-checking instead of
a dedicated pass.

**Marking convention.** A linked claim was read directly this session (quote or table
value taken from the fetched page). **[†unverified]** marks this author's own
training-knowledge recollection of a published fact, not re-confirmed against a
primary source this session.

## 1. How video generators represent time

**Our own architecture is already in this family.** `SiTBlock`
(`flydream/generate/gen13b.py:158`) is a pre-LN transformer block with full
self-attention and an explicit `# adaLN-Zero` modulation — the same block DiT and SiT
use, tokenizing spatial columns instead of image patches. DiT/SiT findings (§3-4) are
an architectural match; only the token content differs.

**Pixel video, two camps, now resolved with checked numbers (via §2's agent).**
*3D/causal-VAE tokenizers* compress space **and time** together: CogVideoX ships a
4× temporal / 8×8 spatial 3D VAE, with an 8×-temporal variant also tested —
[Yang et al., "CogVideoX," ICLR 2025, arXiv:2408.06072](https://arxiv.org/abs/2408.06072), Table 1. Open-Sora/Open-Sora-Plan settled on the same 4× temporal / 8×8 spatial recipe from v1.2 onward, and Open-Sora-Plan v1.5.0 retrained natively at 8× — [hpcaitech/Open-Sora technical reports](https://github.com/hpcaitech/Open-Sora); [PKU-YuanGroup/Open-Sora-Plan reports](https://github.com/PKU-YuanGroup/Open-Sora-Plan). MAGVIT-v2 (causal 3D CNN + lookup-free quantization) and W.A.L.T. (causal 3D CNN, shared image/video tokenizer) both use 4× temporal — [Yu et al., ICLR 2024, arXiv:2310.05737](https://arxiv.org/abs/2310.05737); [Gupta et al., ECCV 2024, arXiv:2312.06662](https://arxiv.org/abs/2312.06662). NVIDIA's Cosmos Tokenizer publishes a family named directly by ratio (CV4x8x8, CV8x8x8, CV8x16x16, continuous and discrete) — [arXiv:2501.03575](https://arxiv.org/abs/2501.03575). *Frame-wise latents, no temporal VAE compression at all*: Stable Video Diffusion's VAE is confirmed to compress space only (1× temporal) — the unmodified per-frame SD 2.1 image encoder, one latent frame per pixel frame, temporal structure handled entirely by ~656M added attention/conv parameters in the denoising U-Net; the SVD paper itself never states this in prose (no VAE reconstruction table exists in it at all) — confirmed instead by full-text search plus the released config (`Stability-AI/generative-models`, `svd.yaml`) — [Blattmann et al., arXiv:2311.15127](https://arxiv.org/abs/2311.15127). Full numbers for all of the above: §2.

**Sora** (OpenAI): compresses a clip "both temporally and spatially" into a latent,
then cuts that latent into "spacetime patches" fed to a transformer — confirmed via
several secondary summaries of the technical report (a direct fetch of openai.com
returned HTTP 403 this session). **No compression factor, patch count, parameter
count, or dataset size is published anywhere in the report** — [OpenAI, "Video generation models as world simulators," 2024](https://openai.com/index/video-generation-models-as-world-simulators/); a reverse-engineering survey exists but no numbers were extracted from it this session — [Liu et al., arXiv:2402.17177](https://arxiv.org/abs/2402.17177). Even the highest-profile example is a source for the *idea* (compress, then patchify), not for numbers.

**Fixed (non-learned) temporal bases are not a pixel-video habit — they are standard
for short, smooth, low-dimensional trajectories, closer to our actual regime:**
- **Human motion prediction** represents a joint trajectory in DCT space instead of
  frame space, fed to a graph convolutional network — [Mao, Liu, Salzmann, Li, "Learning Trajectory Dependencies for Human Motion Prediction," ICCV 2019 (oral), arXiv:1908.05436](https://arxiv.org/abs/1908.05436) (confirmed by direct fetch). Became a standard representation in that field; a precise "K coefficients → error" ablation table was not located this session.
- **Robot action trajectories**: "highly smooth, with most energy concentrated in a
  few low-frequency discrete cosine transform modes" — confirmed by direct fetch of
  the full text: first 2 DCT modes = 98.5% of energy (>80% on every task tested),
  k=1 alone = 93.2% — [Zhang et al., "Hyper-DP3," arXiv:2605.01581 (2026)](https://arxiv.org/abs/2605.01581). Capacity result in §4.
- **Real brain signal, spectral transform, flow matching** — the closest domain match
  found: fMRI BOLD (116 ROIs × 232 steps, MDD; 100 ROIs × 200 steps, ABIDE) wavelet-
  decomposed (5-level DWT), then block-DCT'd, and a **flow-matching** U-ViT generates
  in that spectral space — [Tew et al., "Functional MRI Time Series Generation via Wavelet-Based Image Transform and Spectral Flow Matching...," ICLR 2026, arXiv:2605.30387](https://arxiv.org/abs/2605.30387) (confirmed by direct fetch). Its DCT step is JPEG-style (compressing a wavelet map), not a hard top-K-of-N truncation of raw time as ours is; no compression-factor or parameter count is published for it (checked, not found).

**No head-to-head fixed-vs-learned temporal-basis comparison was found**, in pixel
video or in the three adjacent domains above. Treat our own K=8/12/16 sweep (§2) as
one of the only such comparisons that exists anywhere, fixed-only though it is.

**Transfer flags.** 3D-VAE tokenizer numbers assume pixel redundancy (RGB, spatial
smoothness exploited by conv kernels) and do not obviously bound a 721-node graph
signal. LPIPS-family perceptual metrics are trained on natural images and are not
meaningful for T4/T5 activity. The DCT-trajectory precedents (Mao et al., Hyper-DP3,
DSFM) are the load-bearing transfer evidence for this question, not pixel tokenizers.

## 2. How much temporal compression is safe — factor vs. quality

Built from a completed research agent (primary-source fetches, cross-checked 2-3×
where flagged) plus this project's own measured numbers. Condensed from a ~35-row
survey to the most load-bearing rows; full detail in the agent's own transcript.

| Source | Domain | Spatial | Temporal | Metric | Value |
|---|---|---|---|---|---|
| project, `reports/…step17_1b_scene_dct_prior.md` | our T4/T5 states, DCT | — | K=8 of 40 | round trip | 0.58 |
| ″ | ″ | — | K=12 of 40 | ″ | 0.20 |
| ″ | ″ | — | **K=16 of 40 (in use)** | ″ | **0.045** |
| ″ | ″ | — | K=40 (none) | ″ | 0.019 (floor) |
| project, `reports/…step18_3_prior_on_the_corpus.md` | corpus states, DCT-16 | — | K=16 of 40 | round trip, samples | 0.095 (floor 0.021) |
| CogVideoX, arXiv:2408.06072 Table 1, no-compression baseline | pixel video, 3D VAE | 8×8 | 1× | Flickering[px]/PSNR[px] | 93.2 / 28.4 |
| CogVideoX, shipped variant | ″ | 8×8 | 4× | ″ | 86.3 / 28.7 |
| Cosmos CV4x8x8, arXiv:2501.03575 Table 5 | pixel video, continuous | 8×8 | 4× | PSNR/rFVD (DAVIS) | 32.80 / 15.93 |
| Cosmos CV8x8x8, same table, **temporal-only change** | ″ | 8×8 | 8× | PSNR/rFVD | 30.61 / 30.16 |
| W.A.L.T., arXiv:2312.06662 Table 3(f), c=4 | pixel video, causal 3D | 8×/side | 4× | rFVD | 37.7 |
| W.A.L.T., c=8 (default) | ″ | 8×/side | 4× | rFVD | 17.1 |
| W.A.L.T., c=32 | ″ | 8×/side | 4× | rFVD | 3.5 |
| Open-Sora, native, re-eval in arXiv:2502.11897 Table 2 | pixel video, 3D VAE | 8×8 | 4× (native) | PSNR/SSIM | 31.22 / 0.897 |
| Open-Sora, same VAE, **pooled post-hoc, not retrained** | ″ | 8×8 | 16× | PSNR/SSIM | 27.30 / 0.739 |
| Open-Sora-Plan v1.5.0, **natively retrained** | pixel video, 3D VAE | 8×8 | 8× | PSNR/LPIPS/rFVD | 36.91 / 0.0205 / 52.53 |
| SVD, arXiv:2311.15127 (config-confirmed) | pixel video, per-frame VAE | 8×8 | **1× (none)** | — | no reconstruction table published |
| Hyper-DP3, arXiv:2605.01581 | robot action trajectories, DCT | — | 2 of ~20-64 modes | fraction of energy | 98.5% (task-avg) |

**Reading this table.** Our own rows are the only controlled factor-vs-quality sweep
on one fixed downstream metric; K=16 is not marginal (8× better than K=8) but is not
free either — the 0.045→0.019 gap (re-measured on the corpus as a 0.021 floor) is
named in ROADMAP 18.3 as a live suspect for the residual gap. In pixel video, Cosmos's
CV4x8x8→CV8x8x8 pair is the cleanest **temporal-factor-alone** comparison found (one
more temporal halving costs ~2.2 dB PSNR and roughly doubles rFVD, spatial and channel
count both held fixed); CogVideoX's own table moves channel count and temporal factor
together, so it does not isolate the same thing. **Retraining matters at least as much
as the raw factor**: naive post-hoc pooling of Open-Sora's VAE to 16× with no
retraining costs ~4 dB PSNR (31.22→27.30), but Open-Sora-Plan's v1.5.0, *natively
retrained* at 8×, reaches 36.91 PSNR — beating other tokenizers' own **4×** numbers in
the same table (Wan2.1 35.77, CogVideoX 35.72-36.38 depending on who measured it, see
Gaps). A compression factor alone does not predict quality; training recipe and data
do too. SVD sidesteps the whole tradeoff by not compressing time at all.

**Transfer flag.** PSNR/rFVD measure something different in kind from a round trip
through a frozen brain model; a number from one is not a threshold for the other,
only a rough sense of how far tokenizer designers have pushed this lever — and, per
the reliability note above, not even reliably comparable paper-to-paper on its own
terms.

## 3. Class/label conditioning — gains, and behaviour at ~100 examples/class

Covered by a completed research agent; central numbers cross-checked by this author
against the DiT and SiT arXiv pages directly.

**Conditioning alone, matched architecture/compute — the cleanest ablation found:**
[Dhariwal & Nichol, "Diffusion Models Beat GANs," NeurIPS 2021, arXiv:2105.05233](https://arxiv.org/abs/2105.05233), Table 4, same ADM U-Net, same 2M-iteration budget, ImageNet 256×256, **no guidance on either side**: unconditional FID 26.21 vs. class-conditional FID **10.94** — a **2.4× improvement from the label alone**, all else fixed.

**DiT / SiT contain no unconditional-vs-conditional ablation at all** (confirmed by
full-text search of both) — their "conditioning helps" story is really a
guidance-scale story, CFG on vs. off inside an already class-conditional network:
[Peebles & Xie, "Scalable Diffusion Models with Transformers," ICCV 2023, arXiv:2212.09748](https://arxiv.org/abs/2212.09748), Table 2, DiT-XL/2 FID 9.62 (no CFG) → 2.27 (cfg=1.5); model-size ablation (Table 4, 400K steps, no CFG, all class-conditional): DiT-S/2 68.40, DiT-B/2 43.47, DiT-L/2 23.33, DiT-XL/2 19.47 — DiT-S/2 is the smallest published DiT config (33M parameters, confirmed from the paper's own Tables 1/4), still ~24× our 1.4M. [Ma et al., "SiT," ECCV 2024, arXiv:2401.08740](https://arxiv.org/abs/2401.08740): SiT-XL FID 2.06 vs. DiT-XL 2.27 at matched size/Gflops — again purely conditional.

**More classes, not fewer, helped more** (same architecture, EDM, conditioning the
only variable) — [Adaloglou et al., "Rethinking cluster-conditioned diffusion models...," WACV 2025, arXiv:2403.00570](https://arxiv.org/abs/2403.00570), Table 1: CIFAR-10 (10 classes, ~5,000 img/class) FID 2.07→1.81; CIFAR-100 (100 classes, ~500 img/class) FID 3.41→2.21 — a larger relative gain with more, thinner classes. Granularity has a ceiling, though: past an optimal class count for the dataset size, quality degrades and samples drift out-of-distribution (their Figure 3).

**At ~100 examples/class specifically:**

| Study | Model | Regime | Result |
|---|---|---|---|
| [Shahbazi et al., ICLR 2022, arXiv:2201.06578](https://arxiv.org/abs/2201.06578) | **GAN** | 20 classes × 100 img/class = 2,000 total | Naive conditioning **worse**, e.g. FID 23 (uncond) vs. 100 (cond); crossover only above ~5,000 total images |
| [Giannone et al., "Few-Shot Diffusion Models," arXiv:2205.15463](https://arxiv.org/abs/2205.15463) | Diffusion, meta-learned | 5/class at test time, ample data at meta-train time | Conditional beats unconditional widely — not from-scratch small-data |
| [You et al., "DPT," NeurIPS 2023, arXiv:2302.10586](https://arxiv.org/abs/2302.10586) | Diffusion, 585M params | 1-5 *labeled*/class atop 1.28M unlabeled | FID 3.08→2.50 as labels/class rise 1→5 — semi-supervised, not small-data |

**Only Shahbazi et al. matches our per-class count (100/class)**, and found
conditioning actively harmful — but the mechanism is a GAN discriminator exploiting
per-class structure to overfit faster, a failure mode with no counterpart in a
regression-style flow-matching loss. No diffusion/flow study was found at anywhere
near our (model size, total data, class count) triple simultaneously.

**When conditioning hurts, beyond the GAN case**: over-fine granularity relative to
dataset size (Adaloglou et al., above); label noise — direction confirmed, numbers not
table-verified — [Na et al., ICLR 2024, arXiv:2402.17517](https://arxiv.org/abs/2402.17517). No study was found on *weakly-related-but-not-wrong* labels, our actual candidate case (a UCF101 action-class or stimulus-`kind` label only loosely coupled to the T4/T5 trajectory it produces).

**Our own label inventory** (checked locally, not literature): `label_of()` in
`flydream/data/video_corpus.py` already stores a UCF101 action-class label per clip
(~101 classes over 12,411 passing clips, ~123/class) but it is unused — `prior17.py`'s
`SiTStates` takes **no condition** by design (its own docstring: "the state as the
data and **no condition**"); procedural stimuli carry a coarser `kind` label
(`{expand, contract, rotate}` / `{white, pink}`) with far fewer categories.

## 4. Capacity vs. data at small scale; width vs. depth

Covered by a completed research agent (direct PDF extraction, not search snippets);
its own capacity-vs-data judgment for our case is explicitly its inference, marked as
such below, not a literature consensus.

**No DiT/SiT/U-ViT paper trains below ~33M parameters.** DiT-S 33M FID 68.40, DiT-B
130M FID 43.47, DiT-L 458M FID 23.33, DiT-XL 675M FID 19.47 (ImageNet 256×256, 400K
steps, no CFG) — [Peebles & Xie, arXiv:2212.09748](https://arxiv.org/abs/2212.09748), Tables 1/4 (DiT-S confirmed exactly 33M). SiT matches every DiT config's params/Gflops and beats it at every size (e.g. SiT-S 57.6 vs. DiT-S 68.4) — [Ma et al., arXiv:2401.08740](https://arxiv.org/abs/2401.08740). DiT's own headline finding, quoted: **"parameter counts do not uniquely determine the quality of a DiT model"** — Gflops (width, depth and token count jointly) is what tracks FID, monotonic with no saturation across all 12 tested configs. Neither paper runs a width-vs-depth ablation at fixed parameter count. U-ViT's smallest config is 44M — [Bao et al., CVPR 2023, arXiv:2209.12152](https://arxiv.org/abs/2209.12152). SD3's rectified-flow-transformer scaling study (depth 15-38, width fixed at 64×depth) saw **no saturation even at 8B parameters** — [Esser et al., ICML 2024, arXiv:2403.03206](https://arxiv.org/abs/2403.03206) — so it bounds nothing about where a ceiling appears, only that none of these papers found a model too small.

**Width vs. depth is genuinely mixed, not a settled "width wins."** For U-Net (not
transformer) diffusion models on CIFAR-10, memorization capacity scales
**monotonically with width** but **non-monotonically with depth** — "scaling model
width emerges as a more viable approach for increasing the memorization ratio
[capacity]" — [Gu et al., "On Memorization in Diffusion Models," TMLR 2025, arXiv:2310.02664](https://arxiv.org/abs/2310.02664), the closest primary-source support found for "width matters," but U-Net, not DiT. Against this: [Alabdulmohsin et al., NeurIPS 2023, arXiv:2305.13035](https://arxiv.org/abs/2305.13035) fit compute-optimal ViT exponents where **width should grow slowest** of three axes (depth 0.45, width 0.22, MLP 0.6); [Tay et al., ICLR 2022, arXiv:2109.10686](https://arxiv.org/abs/2109.10686) recommend a "DeepNarrow" strategy, prioritizing depth. One rare **clean fixed-width, depth-only ablation** exists: U-ViT-S (13 layers, 44M, FID 5.95) vs. U-ViT-S-Deep (17 layers, 58M, FID **5.48**) — depth alone, width fixed, improved quality (Bao et al., above). [Levine et al., NeurIPS 2020, arXiv:2006.12467](https://arxiv.org/abs/2006.12467) offer a mechanistic bridge: below a width-dependent transition, a network is "too deep relative to its size" and should widen; their fitted formula (depth range 6-48 only; ours is 4, outside it) extrapolated to depth 4 gives a transition width of **≈193 — close to our candidate 192**. This is the agent's own out-of-range extrapolation on a different architecture/modality (causal language transformers); treat the closeness as a mildly encouraging coincidence, not a validated target.

**Params-per-example, cross-domain.** Our ratio (1.4M / 13,500 ≈ **104 params/example**)
sits almost exactly between DiT-B's ratio on full ImageNet (≈101) and EDM2-XS's
(≈98) — [Karras et al. (EDM2), CVPR 2024, arXiv:2312.02696](https://arxiv.org/abs/2312.02696) — neither flagged as capacity-starved or overfitting in its own paper, and DiT's sweep kept improving FID past 5× our ratio at a comparable epoch count (400K steps/batch 256 ≈ 80 ImageNet epochs vs. our 20K steps/batch 32 ≈ 47). A cross-domain numerical analogy, not a proof — natural images are far more diverse per example than a structured connectome tensor likely is.

**Reading our own validation curve.** EDM2 and [Favero, Sclocchi, Wyart, arXiv:2505.16959](https://arxiv.org/abs/2505.16959) both diagnose overfitting/memorization onset the same way: **validation loss turns upward** while training loss keeps falling — not merely decelerates. Our validation loss (0.5165→0.5064, still falling, decelerating) does **not** show that signature — consistent with, but not proof of, "not yet capacity-limited." Favero et al. additionally find memorization onset scales roughly **linearly with training-set size**, and that a ≈500M-parameter model on 16,384 CelebA images (≈30,500 params/example, 300× our ratio) still generalized well (FID 5.4, 0% copy rate) before its own onset — a data point against "our ratio must already be too rich."

**Out-of-domain but structurally close**: Hyper-DP3 shrinks a diffusion-policy
transformer to **2.52M parameters** (vs. 255.8M for its DP3/Flow-Policy/MP1
baselines — **101× smaller**) by sizing the model to a DCT-truncated trajectory
rather than the raw one, and matches or **beats** those 100×-larger baselines on 10-50
demonstrations/task — [Zhang et al., arXiv:2605.01581](https://arxiv.org/abs/2605.01581) (§1). Robot trajectories and task success rate, not neural activity and distributional fidelity, but the structural claim — a compact temporal representation does not by itself need a large model — again points the same direction as the three lines above.

**Our own numbers, restated** (ROADMAP 18.3, DECISIONS 2026-09-20): 1.4M parameters
(width 128, depth 4), ~13,500 trajectories, 20,000 steps at batch 32. Width 192-256
(≈$0.4-0.9/run) is named as the next thing to try, not yet run. Five independent,
cross-domain, non-definitive lines above (DiT's own ratio sweep, params/example, the
validation-curve signature, the extrapolated depth-to-width transition, and
Hyper-DP3) lean the same direction — mildly toward "not yet oversized, growing is a
reasonable low-risk experiment" — but none studies this architecture, modality or
data scale directly, and none substitutes for running the width-192-256 check already
queued.

## 5. Evaluating unconditional samples beyond FID

Covered by a completed research agent, built from direct primary-source fetches
(arXiv HTML/PDF), flagged inline where it instead relied on a search summary.

**Precision/recall/density/coverage — three papers, one family:**

| Paper | Feature space | Core test | Non-image validation |
|---|---|---|---|
| [Sajjadi et al., NeurIPS 2018, arXiv:1806.00035](https://arxiv.org/abs/1806.00035) | Inception Pool3 (images); 4096-d BiLSTM (**text**) | k-means (k=20), PRD curve | demonstrated on text |
| [Kynkäänniemi et al., NeurIPS 2019, arXiv:1904.06991](https://arxiv.org/abs/1904.06991) | VGG-16 fc2 | k-NN hypersphere manifold membership, k=3 | none |
| [Naeem et al., ICML 2020, arXiv:2002.09797](https://arxiv.org/abs/2002.09797) | VGG-16 fc2, **also explicit random-init VGG-16** | Density/Coverage, real-only neighborhoods, k=5 | **explicitly tested on MNIST and audio spectrograms with a random, untrained network**, arguing the guarantees are "distribution-type and dimensionality agnostic" |

Naeem et al.'s random-network result is the single most load-bearing fact here: when
no pretrained domain feature extractor exists (true for us — no "Inception for T4/T5
activity"), a **fixed random projection is an explicitly validated stand-in**, not an
ad hoc compromise.

**Directly on point:** [Stein et al., NeurIPS 2023, arXiv:2306.04675](https://arxiv.org/abs/2306.04675) states that for class-conditional ImageNet with only **~100 samples per class**, k-NN precision/recall estimates become unreliable — our own "tens to low hundreds of generated samples per run" regime is named in the literature as too small for the standard form of this metric. The same paper shows Inception-V3-based FID/precision-recall correlate poorly with human judgment and recommends DINOv2 features instead.

**Memorization checks.** [Somepalli et al., CVPR 2023, arXiv:2212.03860](https://arxiv.org/abs/2212.03860): flagged Stable-Diffusion near-duplicates at similarity > 0.5 (chosen post hoc, not validated against a held-out criterion); replication rate scales **inversely with training-set size** (frequent at 300 training images, undetectable at ImageNet scale) — a caution given our own ~13,500 examples. Their follow-up, [NeurIPS 2023, arXiv:2305.20086](https://arxiv.org/abs/2305.20086): replication **"often does not happen for unconditional models," common mainly in the text-conditional case** — favorable to our unconditional design, not a guarantee. [Carlini et al., USENIX Security 2023, arXiv:2301.13188](https://arxiv.org/abs/2301.13188) used an **adaptive, neighbor-relative threshold** on CIFAR-10 (flagged only if abnormally close *relative to the spread of its own local neighborhood* — distance normalized by 0.5 × mean distance to 50 neighbors) — a concrete, transferable pattern for a data-driven threshold instead of an arbitrary cutoff on raw correlation. [van den Burg & Williams, NeurIPS 2021, arXiv:2106.03216](https://arxiv.org/abs/2106.03216): **nearest-neighbor-distance tests specifically fail to detect real instance-level memorization** — a "far from its nearest neighbor" result is evidence against near-duplication, not proof against memorization in a technical sense.

**FID's small-sample bias, and one transferable number.** [Chong & Forsyth, CVPR 2020, arXiv:1911.07023](https://arxiv.org/abs/1911.07023): FID at finite N is biased by a generator-specific amount, enough to reverse a model ranking at different N. [Jayasumana et al. (CMMD), CVPR 2024, arXiv:2401.09603](https://arxiv.org/abs/2401.09603): FID needs **>20,000 images** to be reliably estimated. The sharpest illustration of why comes from molecular generation — [Preuer et al. (FCD), arXiv:1803.09518](https://arxiv.org/abs/1803.09518): **FCD = 76.46±5.03 at N=5, falling to ≈0.02±0.00 only by N≈5,000** — any Fréchet/Gaussian-fitting metric needs a full covariance estimate and is unusable at our sample counts. For video: [Luo et al. (JEDi), arXiv:2410.05203](https://arxiv.org/abs/2410.05203): FVD needs **4,350-4,700 samples** to converge within 5%; their replacement needs as few as 700.

**How non-image, no-pretrained-extractor domains actually evaluate** (most
transferable section): molecular generation reports validity, uniqueness,
**novelty** (fraction absent from training, exact match), nearest-neighbor similarity
to the reference set, and **internal diversity** (mean pairwise dissimilarity within
the generated set, catching mode collapse) as a deliberate pair — [Polykovskiy et al. (MOSES), arXiv:1811.12823](https://arxiv.org/abs/1811.12823); [Brown et al. (GuacaMol), arXiv:1811.09621](https://arxiv.org/abs/1811.09621). Protein structure generation built a Fréchet-style metric on **ESM3 features projected to 32 PCA dimensions specifically to keep the covariance matrix well-conditioned at low N**, and validated that **rankings by it are preserved with as few as 50 generated structures** — [arXiv:2505.08041](https://arxiv.org/abs/2505.08041), the closest precedent found for our own sample budget. Graph generation uses **no learned feature extractor at all** — MMD between hand-designed statistics (degree distribution, clustering coefficient, orbit counts) — [You et al. (GraphRNN), ICML 2018, arXiv:1802.08773](https://arxiv.org/abs/1802.08773). Time-series generation trains small post-hoc classifiers instead: a discriminative real-vs-synthetic score and a train-synthetic-test-real predictive score — [Yoon et al. (TimeGAN), NeurIPS 2019](https://papers.neurips.cc/paper/2019/file/c9efe5f26cd17ba6216bbe2a7d26d490-Paper.pdf) — or swap Inception for a self-supervised time-series encoder (TS2Vec) inside the same Fréchet formula — [Jeha et al., arXiv:2108.00981](https://arxiv.org/abs/2108.00981).

**The single most actionable transferable method found**: [Mukherjee & Chang, arXiv:2504.08446](https://arxiv.org/abs/2504.08446) (extended 2026, [arXiv:2601.18156](https://arxiv.org/abs/2601.18156)) run an unbiased two-sample **MMD² with a Gaussian RBF kernel, bandwidth by the median heuristic** (fully data-adaptive), significance by **permutation test** — validated to detect a real difference (p<0.01) with **as few as 5-10 samples per group**. Needs no pretrained feature extractor (runs on any fixed embedding, including a random projection per Naeem et al.), and gives a p-value rather than an ad hoc number. It detects *difference*, not *sameness*: rejecting H₀ for generated-vs-shuffled supports novelty; failing to reject H₀ for generated-vs-held-out-real is indistinguishability, not proof of equivalence.

**Against our own method** (nearest-training-trajectory correlation in raw signal
space, plus a noise-norm coverage proxy from inverting the flow —
`reports/2026-09-20_step17_3b_noise_inversion.md`): structurally the same first move
as the molecular SNN metric, Somepalli's similarity search and Carlini's threshold,
but without a principled, data-driven threshold — Carlini's neighborhood-relative
form is directly adoptable. The noise-norm coverage proxy has no match in anything
surveyed; it is a legitimate, flow-matching-specific substitute for density/coverage
exploiting our generator's own invertibility, not a standard method, and should be
described as such. Two concrete, budget-matched additions the literature suggests:
run existing checks in a **reduced-dimensionality feature space** (Protein FID's
PCA-32 trick, stable at N=50) rather than the raw ~92,000-dim tensor, and add the
**MMD permutation test** against a shuffled/noise control — no new machinery, and
validated at our own sample counts.

## Gaps

- **One of five assigned agents did not return.** §1's agent never delivered a result
  (a stop request found no matching task; completed-without-notification, errored, or
  still running is unknown). §1 rests on this author's own narrower source-checking;
  §2, §3, §4, §5 reflect completed dedicated passes, each spot-checked further by
  this author against at least one primary source.
- **§2's two "cleanest" ablations both have caveats**: CogVideoX's Table 1 varies
  temporal factor and channel count together (not isolated); Cosmos's CV4x8x8-vs-
  CV8x8x8 is the one source that isolates temporal factor alone. LTX-Video publishes
  **zero** reconstruction metrics anywhere (confirmed absent). MAGVIT-v2 never
  ablates its own temporal factor (fixed at 4× throughout). No source in §2 computes
  a reconstruction metric on anything but RGB pixels or an ImageNet/Kinetics-
  pretrained network — no existing precedent for how these metrics behave on an
  8-channel, 721-node signal. The same CogVideoX VAE scores PSNR 28.7 / 29.3-32.1 /
  35.7-35.8 / 36.4 across four different papers measuring it — absolute values are
  not comparable cross-paper without a matched protocol.
- **No published fixed-vs-learned temporal-basis head-to-head was found** (§1), in
  pixel video or the three adjacent trajectory domains checked — each picked one
  approach and reported it alone.
- **One motion-prediction "K coefficients → error" figure surfaced by search** (5
  coefficients: 93% position energy vs. 37% velocity energy) could not be pinned to a
  specific paper/table and is **omitted** above rather than cited unsourced.
- **No DiT/SiT/U-ViT paper trains below ~33M parameters or near our 1.4M scale**
  (§4); the only sub-10M transformer figure found at all is ViT-Ti (5M, cited
  secondhand), a classification model, not diffusion. **No paper gives a
  Chinchilla-equivalent compute- or data-optimal parameter:example ratio for
  diffusion/flow-matching models** — searched directly, treated as a genuine
  open gap, not a search failure. **No paper runs a width-vs-depth ablation at
  strictly fixed parameter count with both trained equally from scratch**: DiT/SiT
  scale width and depth together; TinyFusion's depth-pruned small models are
  distilled from a larger teacher, not trained from scratch; Gu et al.'s
  width/depth memorization study is U-Net, not transformer. Levine et al.'s
  depth-to-width transition formula is fit on depth 6-48 only; applying it at our
  depth 4 is extrapolation outside the tested range, on a different architecture
  and modality (causal language transformers, not vision/flow-matching) — its
  numeric closeness to our candidate width (192) is flagged, not relied on.
- **DSFM (arXiv:2605.30387) has no published parameter count and no published hard
  truncation ratio** — the closest domain match to our own pipeline is incomplete on
  exactly the two numbers most wanted.
- **No study was found on weakly-related-but-not-wrong conditioning labels** (§3) —
  our actual candidate case has no direct precedent.
- **No paper evaluates a domain matching ours closely** (§5): molecular (graph, no
  time), motion (temporal, skeleton not lattice), sensor/EEG (temporal, multichannel,
  not graph-structured), graph generation (structural, no time) each match only part
  of our structure — a spatio-temporal signal on a fixed non-Euclidean graph has no
  single precedent.
- **The "FLD works with as few as 200 samples" figure could not be confirmed**
  against the FLD primary source (only a reduced-reference-set figure, no stated
  numeric floor) — do not cite it.
- **Kynkäänniemi et al.'s stance on non-image feature spaces** was not retrieved from
  the paper's own text; unlike Naeem et al., it has no non-image discussion at all.
- **No formal equivalence test** (e.g. TOST-style) for "generated is statistically
  indistinguishable from real" was found — every novelty test surveyed, MMD
  permutation test included, detects difference, not sameness.
- **EEG-generation evaluation practice (§5)** rests on search-snippet-level
  aggregation, not one deeply-read primary paper.
- **This session's WebSearch budget was exhausted partway through**; each agent's own
  transcript names secondary leads it did not chase down.
