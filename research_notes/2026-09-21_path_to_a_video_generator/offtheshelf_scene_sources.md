# Off-the-shelf scene sources: open pretrained video / image+motion generators on a T4 or CPU

Scope: what an external generator could supply for the fly-brain pipeline — either as an
unbounded corpus of scenes to train a state prior on, or as the scene source itself
(generated clip -> frozen brain -> state -> decoder -> video). The pipeline needs only
**20 frames, 64x64, grayscale, 20 ms/frame**, which is far below every model's native
output; all the models below would be generated at their smallest supported resolution
and then downsampled, so quality is almost irrelevant and *cost per clip* is the whole
question.

Knowledge state: 2026-09-21. Every number carries its source. Timings below are
**almost never measured on a T4** — see the Gaps sections, this is the single largest
weakness of these notes.

---

## Q1. Open text-to-video / unconditional video models that fit a T4 (16 GB) or L4 (24 GB) in fp16

### Takeaway
Three families plausibly fit a 16 GB T4 with room to spare and carry permissive licences:
**CogVideoX-2B (Apache 2.0, from ~4 GB with offload)**, **Wan 2.1 T2V-1.3B (Apache 2.0,
8.19 GB)** and **LTX-Video 2B distilled (OpenRail-M, distilled = "15x faster")**. The older
ModelScope 1.7B / ZeroScope line is cheap and small but **non-commercial** (CC-BY-NC-4.0 /
CC-BY-NC-ND), and Stable Video Diffusion is under Stability's own community licence. The
practical hazard for this project is not VRAM but **architecture**: the T4 is Turing (sm75)
with no bf16 tensor cores, and the 2025-2026 models ship bf16 weights.

### Cited Findings

**CogVideoX-2B**
- Single-GPU inference "starting from 4GB" VRAM in fp16 with diffusers (using
  `enable_model_cpu_offload()` / `enable_sequential_cpu_offload()`); INT8 "starting from
  3.6GB"; 10 GB multi-GPU without offload — [THUDM/CogVideoX-2b model card](https://huggingface.co/THUDM/CogVideoX-2b)
- Speed at 50 inference steps: "Single A100: ~90 seconds", "Single H100: ~45 seconds" — [THUDM/CogVideoX-2b](https://huggingface.co/THUDM/CogVideoX-2b)
- Output is fixed: **720x480 only**, 6 seconds, 8 fps (i.e. 49 frames) — [THUDM/CogVideoX-2b](https://huggingface.co/THUDM/CogVideoX-2b)
- Licence: **Apache 2.0** for the 2B model — [THUDM/CogVideoX-2b](https://huggingface.co/THUDM/CogVideoX-2b)

**Wan 2.1 T2V-1.3B**
- "T2V-1.3B ... requires only 8.19 GB VRAM, making it compatible with almost all
  consumer-grade GPUs" — [Wan-AI/Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B)
- "~4 minutes to generate a 5-second 480P video" on an RTX 4090 without quantisation;
  further reductions with `offload_model` + `t5_cpu` — [Wan-AI/Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B)
- 480P supported and recommended for the 1.3B; 720P "possible but less stable due to
  limited training at this resolution" — [Wan-AI/Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B)
- Licence: **Apache 2.0**, "We claim no rights over the your generate contents" — [Wan-AI/Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B)
- A community aggregator estimates ~1m 13s for **25 frames at 768x512, fp16, 30 steps** on
  an RTX 4090; ~4m 35s on an RTX 3060 12 GB; ~6m 55s on an RTX 4060 8 GB — [WillItRunAI: Wan Video 2.1 1.3B](https://willitrunai.com/video-models/wan-video-2-1-1-3b).
  **Caveat: that page carries no T4 or L4 row and states "All estimates are approximations
  based on mathematical models and public specifications", i.e. these are modelled, not
  measured.**

**LTX-Video (2B and 13B distilled)**
- Variants: `ltxv-2b-0.9.8-distilled` ("smaller model, slight quality reduction"),
  `ltxv-13b-0.9.8-distilled` ("faster inference, lower VRAM"), plus fp8 variants of both — [Lightricks/LTX-Video GitHub](https://github.com/Lightricks/LTX-Video)
- Distilled models are "15x faster inference than non-distilled"; 13B distilled on an H100
  makes HD video in ~10 s — [Lightricks/LTX-Video GitHub](https://github.com/Lightricks/LTX-Video)
- Resolution must be **divisible by 32**, frame count **divisible by 8 plus 1** (9, 17, 25,
  257); works best under 720x1280 — [Lightricks/LTX-Video GitHub](https://github.com/Lightricks/LTX-Video).
  This matters: a 20-frame target maps to 17 or 25 generated frames, and 64x64 is not a
  legal size (must be a multiple of 32 — 64x64 *is* legal, 96x96 also; the smallest legal
  sizes are well below the model's trained range).
- fp8 kernels are "available for Ada architecture GPUs and later" — [Lightricks/LTX-Video GitHub](https://github.com/Lightricks/LTX-Video) — i.e. **not usable on a T4 (Turing) and, for fp8, arguably not on an L4? (L4 is Ada, so fp8 is available on the L4 but not the T4).**
- Licence: **OpenRail-M**, "permits commercial use with specified restrictions and
  obligations" — [Lightricks/LTX-Video GitHub](https://github.com/Lightricks/LTX-Video)
- A 2026 community report puts LTX Video 2B at ~1m 28s for a **25-frame** clip on an RTX
  4090 24 GB — search result surfaced from [WillItRunAI LTX Video 2B](https://willitrunai.com/video-models/ltx-video-2-3) (aggregator; same estimated-not-measured caveat)

**ModelScope T2V 1.7B (damo/ali-vilab)**
- 1.7B parameters; with attention + VAE slicing under Torch 2.0 it can "generate videos up
  to 25 seconds on less than 16GB of GPU VRAM" — [ali-vilab/text-to-video-ms-1.7b](https://huggingface.co/damo-vilab/text-to-video-ms-1.7b)
- Licence: **CC-BY-NC-ND** (non-commercial, no derivatives) — [ali-vilab/text-to-video-ms-1.7b](https://huggingface.co/damo-vilab/text-to-video-ms-1.7b)

**ZeroScope v2 576w**
- 576x320, 24 frames per sequence; "uses 7.9gb of vram when rendering 30 frames at
  576x320" — [cerspense/zeroscope_v2_576w](https://huggingface.co/cerspense/zeroscope_v2_576w)
- Licence: **CC-BY-NC-4.0** (non-commercial) — [cerspense/zeroscope_v2_576w](https://huggingface.co/cerspense/zeroscope_v2_576w)

**Stable Video Diffusion (img2vid)**
- 14 frames at 576x1024; on an **A100 80 GB** "SVD takes ~100s for generation, and SVD-XT
  takes ~180s"; the card notes optimisations exist for lower-VRAM cards — [stabilityai/stable-video-diffusion-img2vid](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid)
- Licence: "stable-video-diffusion-community"; commercial use routed to
  https://stability.ai/license — i.e. **not a plain permissive licence** — [stabilityai/stable-video-diffusion-img2vid](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid)

**AnimateDiff (SD1.5 + motion module)**
- "takes only ~12GB VRAM to inference and can run on a single RTX3090" — [guoyww/AnimateDiff issue #92](https://github.com/guoyww/AnimateDiff/issues/92)
- A 2026 community post reports AnimateDiff at 512x512 running "3-5 minutes on an RTX
  5090" — [Easton Dev, ComfyUI low-VRAM optimisation](https://eastondev.com/blog/en/posts/ai/20260721-comfyui-low-vram-acceleration/) (blog, single author, no methodology; treat as weak)

**Open-Sora**
- "Code, training pipeline, and weights are all published under **Apache 2.0**" — [hpcaitech/Open-Sora](https://github.com/hpcaitech/Open-Sora)
- Reported inference: **30 s for a 2-second 240p video on an RTX 3090**, 60 s for 4 s — surfaced via [Open-Sora upgrade announcement / HPC-AI blog](https://company.hpc-ai.com/blog/open-soras-comprehensive-upgrade-unveiled-embracing-16-second-video-generation-and-720p-resolution-in-open-source)
- Open-Sora-Plan v1.3.0 "supports 93x480p within 24G VRAM" — [LanguageBind/Open-Sora-Plan-v1.3.0](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.3.0)
- **240p is the lowest natively-supported resolution found anywhere in this survey, and the
  30 s / 2 s figure is the best throughput-per-clip number with a named consumer GPU.**

**Wan 2.2 TI2V-5B (newer, larger)**
- "can generate a 5-second 720P video in under 9 minutes on a single consumer-grade GPU";
  the reference command "can run on a GPU with at least **24GB VRAM** (e.g. RTX 4090)" — [Wan-Video/Wan2.2 GitHub](https://github.com/Wan-Video/Wan2.2), [Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)
- Aggregator estimate: ~4m 23s for 25 frames at 768x512, fp16, 30 steps on an RTX 4090 — [WillItRunAI Wan2.2 TI2V 5B](https://willitrunai.com/video-models/wan-video-2-2-ti2v-5b) (estimated)
- **24 GB minimum puts it on the L4, not the T4.**

**The Turing/bf16 hazard (applies to the T4 specifically)**
- Turing GPUs (T4, RTX 20-series) have no native bf16 tensor-core support — that arrived
  with Ampere; on Turing bf16 falls back to a slower path while fp16 gets full tensor-core
  acceleration — synthesis of search results including [unsloth issue #4970 (dtype mismatch BF16 vs FP16 on T4)](https://github.com/unslothai/unsloth/issues/4970)
- A concrete 2026 instance: users asking Lightricks for an **fp16 (not just bf16)**
  transformer file so older cards can run at full precision without an int8/nvfp4 quality
  trade-off — [Lightricks/LTX-2.5 discussion #13](https://huggingface.co/Lightricks/LTX-2.5/discussions/13)
- Newer models can produce **black images / NaN latents under fp16** (a reported failure
  mode for a 2026 image model) — [Tongyi-MAI/Z-Image issue #14](https://github.com/Tongyi-MAI/Z-Image/issues/14)

### Inferences
- The licence axis cleanly separates the field: **Apache 2.0 (CogVideoX-2B, Wan 2.1/2.2,
  Open-Sora)** vs **non-commercial (ModelScope 1.7B, ZeroScope, SD/SDXL-Turbo)** vs
  **bespoke (SVD community, LTX OpenRail-M)**. For this project, which is not commercial but
  may publish weights/outputs, the Apache 2.0 group is the only one that raises no
  redistribution question about generated clips.
- None of these models can be asked for a 64x64, 20-frame, 20 ms clip natively: CogVideoX-2B
  is locked to 720x480x49, ZeroScope to 576x320x24, SVD to 576x1024x14. The realistic
  workflow is *generate at the model's smallest legal setting, then crop/downsample/
  resample time to 20 frames*, which means **the cost per usable clip is the model's full
  cost per clip — there is no saving from the tiny target size** except where resolution is
  free (LTX and Wan accept arbitrary sizes divisible by 32; Open-Sora accepts 240p).
- Because the target is 64x64 grayscale, the *cheapest* model that produces any coherent
  motion wins; LTX-2B-distilled at a minimal legal resolution and 17-25 frames, or
  Open-Sora at 240p, are the two whose published knobs go lowest.
- The T4's lack of bf16 is a real schedule risk for the 2025-2026 models (LTX 2.x, Wan 2.2);
  CogVideoX-2B and Wan 2.1-1.3B both document fp16 paths, so they are the T4-safe choices.
  An **L4 (Ada)** removes the bf16 and fp8 problems entirely and is already allowed by the
  project's compute rule when memory-bound.

### Gaps
- **No source found with a measured T4 or L4 timing for any of these models.** Every
  per-clip number above is for a 4090, 3090, A100, H100 or 5090, or is an aggregator's
  model-based estimate. A T4 number for this project would have to be measured, not cited.
- No source found giving timings at genuinely low resolutions (128x128 or below) for LTX or
  Wan — the cost at the project's scale is unknown from the literature.
- HunyuanVideo and Mochi were not investigated in detail (both are ~13B+ class and appear
  only in 24 GB+ guides); Latte's own licence and VRAM were not confirmed — the only Latte
  mention found was that Open-Sora "no longer support[s] 2+1D models" as of v1.2.0 — [hpcaitech/Open-Sora](https://github.com/hpcaitech/Open-Sora).
- "Pusa", LTX-2, and other 2026 small models were not verified against primary sources;
  search surfaced mostly SEO/aggregator pages (spheron.network, willitrunai.com,
  localaimaster.com, ltxworkflow.com, runaihome.com, wan27.org) whose numbers I could not
  trace to a model card or repo. I have deliberately not reported their figures as facts.
- Whether any of these models runs usefully on a **32-core CPU inside a 10-minute budget**
  is unsupported by any source I found. No CPU-inference benchmark for a video diffusion
  model turned up.

---

## Q2. Image generator + synthetic motion as the cheaper route

### Takeaway
A still image plus a scripted camera motion is orders of magnitude cheaper than video
diffusion — a one-step image model produces a 512x512 frame in tens of milliseconds, and
the motion is then pure warping — but the two best-known fast image models (SD-Turbo,
SDXL-Turbo) are **non-commercial research licences**, and depth-based parallax
(3D Ken Burns) adds a depth network and inpainting per image.

### Cited Findings
- SD-Turbo "generates approximately 38ms per 512x512 image"; SDXL-Turbo generates a
  512x512 image in **207 ms on an A100 with a single denoising step** — [Stability AI: Introducing SDXL Turbo](https://stability.ai/news/stability-ai-sdxl-turbo), [stabilityai/sd-turbo](https://huggingface.co/stabilityai/sd-turbo)
- Both are "a fast generative text-to-image model that can synthesize photorealistic images
  from a text prompt in a **single network evaluation**", trained by Adversarial Diffusion
  Distillation (ADD), sampling in 1-4 steps — [stabilityai/sd-turbo](https://huggingface.co/stabilityai/sd-turbo)
- Licence: **STABILITY AI NON-COMMERCIAL RESEARCH COMMUNITY LICENSE AGREEMENT**; "SDXL Turbo
  is only available under a non-commercial research license" — [stabilityai/sd-turbo](https://huggingface.co/stabilityai/sd-turbo), [Stability AI SDXL Turbo announcement](https://stability.ai/news/stability-ai-sdxl-turbo)
- 3D Ken Burns from a single image: the framework "leverages a depth prediction pipeline",
  "maps the input image to a point cloud and synthesizes the resulting video frames by
  rendering the point cloud from the corresponding camera positions", with automatic and
  interactive (user-controlled camera) modes — [Niklaus et al., 3D Ken Burns Effect from a Single Image, arXiv:1909.05483](https://arxiv.org/abs/1909.05483), [ACM TOG](https://dl.acm.org/doi/10.1145/3355089.3356528)
- Reference PyTorch implementation exists and "can animate a still image with a virtual
  camera scan and zoom subject to motion parallax" — [sniklaus/3d-ken-burns](https://github.com/sniklaus/3d-ken-burns); an extended
  implementation with training code for the depth and inpainting networks — [pierlj/ken-burns-effect](https://github.com/pierlj/ken-burns-effect)
- Pretrained low-resolution unconditional diffusion exists at exactly the project's scale:
  the EDM ImageNet 64x64 model's recommended stochastic sampler uses 256 steps = **511 NFE**;
  Skip-Tuning reaches a large FID improvement on pretrained EDM ImageNet-64 with **19 NFE** — surfaced via [Restart Sampling, arXiv:2306.14878](https://arxiv.org/pdf/2306.14878) and the Skip-Tuning result cited in [Diffusion Sampling Correction, arXiv:2411.06503](https://arxiv.org/pdf/2411.06503)

### Inferences
- At 38 ms per 512x512 frame, generating 1,000 stills is under a minute of GPU time; the
  motion step (affine pan/zoom/rotate warps, or optical-flow warping) is trivially cheap and
  runs on the 32-core CPU box well inside its 10-minute budget. This route is therefore
  **two to four orders of magnitude cheaper per clip** than any video diffusion model above.
- The trade is scientific, not computational: affine-warped stills contain only global
  camera motion — exactly the class of motion the fly's T4/T5 and LPLC neurons are tuned to,
  but with **no object motion, no looming, no independently moving figures**. A state prior
  trained on such a corpus would be a prior over a narrow motion manifold. Depth-based
  parallax (3D Ken Burns) adds motion parallax and hence some local flow, at the cost of a
  depth net + inpainting per still.
- A pretrained **unconditional** ImageNet-64 diffusion model is the closest thing in the
  literature to a native-resolution source for this project: 64x64 output, no text encoder,
  no prompt distribution to design, and Apache/permissive research code lineage. Its cost is
  NFE-bound (511 NFE recommended, ~19 with post-hoc tuning), and it produces stills, so it
  still needs the warp step.
- Licence-wise the clean version of this route is **not** SD-Turbo but a permissively
  licensed small image model or the ImageNet-64 diffusion checkpoints.

### Gaps
- I did not find a licence statement for the EDM / ADM ImageNet-64 checkpoints themselves
  (NVIDIA EDM code is commonly CC-BY-NC for weights; **unverified**, do not assume).
- No measured wall-clock number for EDM ImageNet-64 sampling on a T4 or on CPU was found —
  only NFE counts, which are not wall-clock.
- No source found that quantifies the cost of the 3D Ken Burns pipeline per image (it
  predates the fast-inference era and the papers report quality, not throughput).
- DPT-depth-plus-warp as a named cheap alternative to 3D Ken Burns was not confirmed with a
  source in this pass.

---

## Q3. Precedent for using a generative model as a data source for a downstream model

### Takeaway
There is strong, well-cited precedent that synthetic images from text-to-image models train
competitive *discriminative* models — in places beating real data at equal or larger scale —
and an equally well-cited body of work showing that training *generative* models on their
own kind of output degrades the distribution's tails. The second is the direct caveat for
training a state prior on generator outputs.

### Cited Findings
- StableRep: "With solely synthetic images, the representations learned by StableRep surpass
  the performance of representations learned by SimCLR and CLIP using the same set of text
  prompts and corresponding real images, on large scale datasets" — [StableRep, arXiv:2306.00984](https://arxiv.org/abs/2306.00984)
- "StableRep trained with 20M synthetic images achieves better accuracy than CLIP trained
  with 50M real images" (with language supervision) — [StableRep, arXiv:2306.00984](https://arxiv.org/abs/2306.00984); also [MIT News coverage](https://news.mit.edu/2023/synthetic-imagery-sets-new-bar-ai-training-efficiency-1120)
- StableRep's stated mechanism: "greater control in sampling is achieved via the guidance
  scale and text prompts, and generative models have the potential to generalize beyond
  their training data and provide a richer synthetic training set" — [StableRep, arXiv:2306.00984](https://arxiv.org/abs/2306.00984)
- StableRep also finds that only "when the generative model is configured with proper
  classifier-free guidance scale" does synthetic training match or beat real — i.e. the
  sampling configuration is a first-order variable, not a detail — [StableRep, arXiv:2306.00984](https://arxiv.org/abs/2306.00984)
- "Is synthetic data from generative models ready for image recognition?": synthetic data
  "proved suitable and effective for model pre-training, delivering superior transfer
  learning performance and even outperforming ImageNet pre-training, especially in
  unsupervised model pre-training and with ViT-based backbones"; and works
  "collaboratively with real data" when initialised from ImageNet weights — [arXiv:2210.07574](https://arxiv.org/abs/2210.07574)
- The Curse of Recursion: "use of model-generated content in training causes irreversible
  defects in the resulting models, where tails of the original content distribution
  disappear"; model collapse is demonstrated for **VAEs, Gaussian Mixture Models and LLMs** — [Shumailov et al., arXiv:2305.17493](https://arxiv.org/abs/2305.17493)
- The collapse is not inevitable if data is *accumulated* rather than *replaced*: "Is Model
  Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic
  Data" — [arXiv:2404.01413](https://arxiv.org/abs/2404.01413)

### Inferences
- The positive precedent is specifically for **discriminative / representation learning** on
  synthetic images. That is a weaker match to this project's step than it first appears: the
  project wants to train an *unconditional generative prior* (over states) on data produced
  by another generative model, which is exactly the recursive configuration the collapse
  literature warns about — with the mitigation that the two generative models live in
  **different spaces** (natural video vs 92,288-dim brain state) and the mapping between
  them (the frozen connectome network) is fixed and not learned from the synthetic data.
- The collapse result is about a model trained on *its own* outputs across generations. Here
  there is one generation and no feedback loop, so the applicable risk is narrower and
  well-named: **the state prior inherits the video generator's distribution, including its
  missing tails**, and no amount of sampling from the prior recovers scenes the video
  generator cannot make. The arXiv:2404.01413 mitigation maps directly onto the project:
  mix the UCF101-derived real states (N = 13,555) with generated-clip states rather than
  replacing them.
- StableRep's guidance-scale finding predicts that the *diversity knob* of whatever video
  model is used (guidance scale, prompt distribution, seed variety) will matter more to the
  prior's quality than the model's visual fidelity — which is fortunate, since at 64x64
  grayscale fidelity is discarded anyway.

### Gaps
- **I found no paper that trains an unconditional generative prior directly on another
  generator's outputs and measures the result.** This is the precedent the project would
  most want, and I could not source it. Everything cited above is either
  synthetic-data-for-discriminative-training or self-recursive collapse.
- No source found quantifying how synthetic-corpus breadth translates into a prior's
  coverage of a downstream latent space.

---

## Q4. Precedent in neuroscience projects: a pretrained generator as the prior, the brain only conditioning it

### Takeaway
The dominant fMRI reconstruction papers of 2023-2024 all follow exactly the architecture
this project is considering: a **frozen, pretrained Stable Diffusion** supplies all the
image/video prior, and the brain data only supplies conditioning. Their standard phrasing is
"reconstruction from brain activity", and the dream-decoding lineage (Horikawa & Kamitani)
is careful to claim decoded *contents* against verbal reports, not a rendered dream.

### Cited Findings
- Takagi & Nishimoto (CVPR 2023) "uses a latent diffusion model termed Stable Diffusion to
  reconstruct images from human brain activity obtained via fMRI, reducing computational cost
  while preserving high generative performance" — no generator training from scratch — [CVPR 2023 poster page](https://cvpr.thecvf.com/virtual/2023/poster/21159), [project site](https://sites.google.com/view/stablediffusion-with-brain/), [code](https://github.com/yu-takagi/StableDiffusionReconstruction)
- The follow-up improves reconstruction "using latent diffusion models via multiple decoded
  inputs" — the brain supplies several conditioning streams into the same frozen generator — [arXiv:2306.11536](https://arxiv.org/abs/2306.11536)
- MinD-Video / "Cinematic Mindscapes": the method uses "co-training with an augmented Stable
  Diffusion model that incorporates network temporal inflation", where "the augmented Stable
  Diffusion is trained with videos and then tuned with the fMRI encoder"; it "guides the
  diffusion model conditioned on visual fMRI features" — [Chen et al., arXiv:2305.11675](https://arxiv.org/abs/2305.11675), [NeurIPS 2023 proceedings PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/4e5e0daf4b05d8bfc6377f33fd53a8f4-Paper-Conference.pdf)
- MindEye: "Reconstructing the Mind's Eye: fMRI-to-Image with Contrastive Learning and
  **Diffusion Priors**" — the prior is a named, separate, pretrained component — [NeurIPS 2023](https://neurips.cc/virtual/2023/poster/70292), [OpenReview](https://openreview.net/forum?id=rwrblCYb2A)
- NeuroClips (NeurIPS 2024) continues the fMRI-to-video line — [NeurIPS 2024 proceedings PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/5c594bf6223b67109441c9e0c97542ed-Paper-Conference.pdf)
- Horikawa, Tamaki, Miyawaki & Kamitani (Science 2013) decoded sleep-onset imagery by
  "predict[ing] the contents of visual imagery during the sleep-onset period by discovering
  links between human fMRI patterns and **verbal reports** with the assistance of lexical and
  image databases"; they conclude "the visual content of dreams is represented by the same
  neural substrate as observed during awake perception" — a claim about representation, not a
  rendered dream — [Science 340:639-642](https://www.science.org/doi/10.1126/science.1234330), [PDF](https://www.cse.iitk.ac.in/users/se367/14/Readings/papers/horikawa-tamaki-kamitani-13_neural-decoding-of-dreams.pdf), [code](https://github.com/KamitaniLab/HumanDreamDecoding)
- Horikawa & Kamitani (2017) extend this to "Hierarchical Neural Representation of Dreamed
  Objects Revealed by Brain Decoding with Deep Neural Network Features" — again object
  representation, decoded against reports — [PMC5281549](https://pmc.ncbi.nlm.nih.gov/articles/PMC5281549/)
- The field's own credibility debate is live: a 2025 reanalysis paper is titled "Advancing
  credibility and transparency in brain-to-image reconstruction research: Reanalysis of
  Koide-Majima, Nishimoto, and Majima (Neural Networks, 2024)" — [arXiv:2511.07960](https://arxiv.org/pdf/2511.07960)
- A recurring stated limitation of the whole line is "the necessity for subject-specific
  models", since individual differences hinder a universal decoder — surfaced in
  [StableMind, arXiv:2605.02586](https://arxiv.org/pdf/2605.02586)

### Inferences
- The precedent is unambiguous that **leaning on an external pretrained generator for the
  prior is the field-standard move**, not a compromise — Takagi & Nishimoto, MinD-Video and
  MindEye all do it, and none claims to have learned the image prior from brain data.
- But the precedent is for the *opposite direction of dependence* from option (b) in this
  project. In those papers the brain signal is the **conditioning** and the generator is the
  prior; the claim "reconstruction from brain activity" is earned because the brain
  determines which sample is produced. In option (b) here the generated clip *precedes* the
  brain state and determines it, so the brain contributes nothing to what the video shows —
  the project's own proposed wording ("video from a state produced by another generator") is
  strictly more honest than the field's, and the field's phrasing would not be available.
- Option (a) — generator as corpus source for the prior — maps far better onto the
  precedent, and is closer to what MinD-Video does when it pre-trains the video-augmented
  Stable Diffusion on videos before touching fMRI.
- The 2013 dream paper is the right citation for the project's bounded language: it decodes
  *contents* against verbal reports and never renders a dream image, which is precisely the
  distinction the project's contract already makes about the word "dream".

### Gaps
- I did not find a published critique specific to Takagi & Nishimoto 2023 (e.g. on how much
  of the reconstruction is Stable Diffusion's prior rather than brain information); the
  closest is the general credibility reanalysis at arXiv:2511.07960, which targets a
  different paper. The "how much is the prior doing?" question is exactly the project's
  concern and I could not source a direct answer.
- No source found in the fMRI literature that generates from a **sampled/unconditional**
  state rather than a measured one — which is the project's actual hard step.

---

## Q5. Cost per 1,000 clips at 64x64-equivalent on a T4/L4

### Takeaway
No source gives a T4 or L4 timing, so every figure here is a **transfer from a named
different GPU and must be treated as an order-of-magnitude bound, not a price**. On the
published numbers, the cheapest video-diffusion option is roughly **$5-$25 per 1,000 clips**
on a T4-class card if the per-clip time lands between 30 s and 150 s; the image+warp route is
**under $1 per 1,000 clips**.

### Cited Findings (the throughput numbers the arithmetic rests on)
- Open-Sora: 30 s per 2-second 240p clip on an **RTX 3090** — [HPC-AI Open-Sora blog](https://company.hpc-ai.com/blog/open-soras-comprehensive-upgrade-unveiled-embracing-16-second-video-generation-and-720p-resolution-in-open-source)
- CogVideoX-2B: ~90 s per clip (49 frames, 720x480, 50 steps) on an **A100** — [THUDM/CogVideoX-2b](https://huggingface.co/THUDM/CogVideoX-2b)
- Wan 2.1-1.3B: ~240 s per 5-second 480P clip on an **RTX 4090** — [Wan-AI/Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B)
- LTX-Video 2B: ~88 s per 25-frame clip on an **RTX 4090 24 GB**, per aggregator estimate — [WillItRunAI](https://willitrunai.com/video-models/ltx-video-2-3)
- SD-Turbo: ~38 ms per 512x512 image, single network evaluation — [stabilityai/sd-turbo](https://huggingface.co/stabilityai/sd-turbo)
- T4 price in this project's own operating context: ≈ $0.59/h (project brief; Modal pricing,
  not an external citation)

### Arithmetic (transparent, from the above; every input is a non-T4 GPU)
Per-1,000-clip GPU cost at $0.59/h = $0.000164 per GPU-second.

| Option | Per-clip time (source GPU) | 1,000 clips at that rate | If the T4 is 3x slower | If 6x slower |
|---|---|---|---|---|
| SD-Turbo still + CPU warp | 0.038 s (A100-class) | $0.006 | $0.02 | $0.04 |
| Open-Sora 240p, 2 s | 30 s (RTX 3090) | $4.92 | $14.8 | $29.5 |
| CogVideoX-2B | 90 s (A100) | $14.8 | $44.3 | $88.6 |
| LTX-2B distilled, 25 frames | 88 s (RTX 4090, estimated) | $14.4 | $43.3 | $86.6 |
| Wan 2.1-1.3B, 5 s 480P | 240 s (RTX 4090) | $39.4 | $118 | $236 |

### Inferences
- The scaling factor is the whole uncertainty. A T4 is roughly an order of magnitude below a
  4090 or A100 in fp16 throughput on paper; the 3x and 6x columns are a deliberately wide
  bracket rather than a claim. **A single measured T4 smoke run on one model would collapse
  this table into a real price** and is far cheaper than more reading.
- For a corpus on the scale of the existing one (N = 13,555 states), even the pessimistic end
  of the cheap options is modest: Open-Sora at the 6x column implies roughly $400 for 13,555
  clips; CogVideoX-2B implies roughly $1,200. The image+warp route implies well under $1.
- If clips are needed by the tens of thousands (the "unbounded corpus" framing), only the
  image+warp route and the 240p-class video model are in a comfortable range; the rest turn
  the corpus into a four-figure line item.
- The project's "use the GPU to the full" rule bites here: all the per-clip numbers above are
  batch-of-one. Since 64x64x20 is tiny, many clips per forward batch should fit in 16 GB even
  for the 2B models, and the effective per-clip cost could fall substantially below the table
  — but no source quantifies batched video-diffusion throughput, so this is an expectation,
  not a number.

### Gaps
- **No measured T4/L4 per-clip time for any model** — stated again because it undermines
  every row of the table.
- No source gives per-clip cost at low resolution; all timings are at the models' native
  (much larger) resolutions, so the table may substantially overstate the cost of the
  LTX/Wan/Open-Sora rows if they scale down well — and no source establishes that they do.
- No source found for batched (multi-clip-per-forward) throughput on any of these models.
- Modal's own T4 hourly price is taken from the project brief, not from an external citation.

---

## Cheapest viable options for 64x64 grayscale 20-frame clips

Strictly within what the sources above support:

1. **A one-step image model plus scripted camera motion.** Cheapest by two-to-four orders of
   magnitude (38 ms per 512x512 still, [sd-turbo](https://huggingface.co/stabilityai/sd-turbo)),
   the motion step is CPU-only and fits the 10-minute local budget, and 3D Ken Burns gives a
   published route to real motion parallax if global warps are too narrow
   ([arXiv:1909.05483](https://arxiv.org/abs/1909.05483), [sniklaus/3d-ken-burns](https://github.com/sniklaus/3d-ken-burns)).
   Its limitation is stated in the sources by omission: warped stills contain camera motion
   only. **Licence caveat: SD-Turbo/SDXL-Turbo are non-commercial research licences**
   ([Stability AI](https://stability.ai/news/stability-ai-sdxl-turbo)); a permissively
   licensed small image model, or a pretrained unconditional ImageNet-64 diffusion model
   (native 64x64, 511 NFE recommended / ~19 NFE with Skip-Tuning), would be needed for a
   clean one — and that checkpoint's licence is an open gap.
2. **Open-Sora at 240p.** The only model in the survey with a published sub-minute per-clip
   time on a consumer card (30 s for a 2-second 240p clip on an RTX 3090,
   [HPC-AI](https://company.hpc-ai.com/blog/open-soras-comprehensive-upgrade-unveiled-embracing-16-second-video-generation-and-720p-resolution-in-open-source))
   and it is **Apache 2.0 for code, pipeline and weights**
   ([hpcaitech/Open-Sora](https://github.com/hpcaitech/Open-Sora)). 240p is also the lowest
   natively supported resolution found anywhere here, which matters for a 64x64 target.
3. **CogVideoX-2B.** Apache 2.0, documented to run "starting from 4GB" in fp16 with offload
   (so comfortably T4-safe, with an explicit fp16 path and no bf16 dependency), ~90 s per
   clip on an A100 ([THUDM/CogVideoX-2b](https://huggingface.co/THUDM/CogVideoX-2b)). Its
   drawback is the locked 720x480x49 output — every clip is paid for at full size and then
   thrown away down to 64x64.

**Wan 2.1-1.3B** (Apache 2.0, 8.19 GB, fp16) is the safe fallback if Open-Sora or CogVideoX
prove unusable, at roughly 3x their per-clip cost on the published numbers. **LTX-2B
distilled** is attractive on speed ("15x faster" distilled) and accepts arbitrary sizes
divisible by 32 and 17/25-frame counts that match the target almost exactly, but it is
OpenRail-M rather than Apache, and its fp8 acceleration is Ada-only — i.e. an L4 job, not a
T4 one ([Lightricks/LTX-Video](https://github.com/Lightricks/LTX-Video)).

Excluded on licence: ModelScope T2V 1.7B (CC-BY-NC-ND), ZeroScope v2 (CC-BY-NC-4.0),
SD/SDXL-Turbo for anything beyond research (Stability non-commercial), Stable Video Diffusion
(bespoke community licence). Excluded on memory: Wan 2.2 TI2V-5B and above need 24 GB, so L4
only.

**The one number that would change all of this** is a measured T4 seconds-per-clip for one
model at a near-64x64 setting. No source provides it.
