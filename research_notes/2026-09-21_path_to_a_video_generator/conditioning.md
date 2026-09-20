# Conditioning as a data- and compute-efficiency lever for small generative models (state of knowledge 2026-09-21)

Scope: what published work measures about how a label, a still image / first frame, a retrieved
neighbour, or a frozen pretrained embedding changes the data and compute a diffusion / flow model
needs to produce coherent new samples — with emphasis on small-scale (64x64) video generation and
video prediction.

## Q1. Class-conditional vs unconditional at small N: what is actually measured?

### Takeaway
Conditioning on a label reliably buys quality at a fixed data and compute budget, but the measured
gap is modest when the label is coarse relative to the data (CIFAR-10: FID 2.72 uncond -> 2.24 cond)
and large when the conditioning partitions the data finely (ImageNet 64x64: 6.44 -> 3.08, a 2.1x FID
reduction). The strongest published effects come not from ground-truth class labels but from
*fine-grained* conditioning signals — many clusters, or a self-supervised representation vector —
which can cut unconditional FID by 50-82%. The cost of this lever is memorisation: conditioning
shrinks the effective sample set per condition, and on a 50k dataset uninformative per-sample
conditioning drove replication of training data from 0% to >65%.

### Cited findings
- EDM (Karras et al. 2022) CIFAR-10: FID 1.79 class-conditional vs 1.97 unconditional, same design,
  35 NFE — i.e. the label is worth ~9% relative FID at N = 50,000 — [EDM repo / NVlabs](https://github.com/nvlabs/edm)
- "Why Are Conditional Generative Models Better Than Unconditional Ones?" (Bao et al., arXiv 2212.00362)
  measures matched uncond / cond / self-conditioned (k-means on MoCo features) diffusion:
  - CIFAR-10: uncond FID 2.72, class-conditional 2.24, self-conditioned with K=10 clusters 2.23
  - ImageNet 64x64: uncond 6.44, class-conditional 3.08, self-conditioned K=1000 3.94
  - CelebA 64x64: uncond 2.14, self-conditioned K=10 1.91; LSUN Bedroom 64x64: uncond 2.69, K=100 2.25
  — [ar5iv 2212.00362](https://ar5iv.labs.arxiv.org/html/2212.00362), [PDF](https://arxiv.org/pdf/2212.00362)
- The same paper's stated mechanism: a conditional model "produces samples from a mixture of several
  unconditional models" sharing one backbone, and conditioning "partitions the data into simpler
  groups according to the semantics of data"; its sufficient condition for superiority says the gain
  grows as the *conditional* distribution gets simpler — [ar5iv 2212.00362](https://ar5iv.labs.arxiv.org/html/2212.00362)
- Self-Guided Diffusion Models (Hu et al., arXiv 2210.06462): replacing labels with self-annotation
  (clustering of self-supervised features) improves FID by 25-50% over unguided; ImageNet32 FID 7.3
  with self-labeled guidance vs 14.3 unguided; ImageNet64 12.1 vs 36.1 (a 3x FID reduction). FID
  improves monotonically as cluster count rises 1 -> 5,000; at K=1,000 self-labels match ground-truth
  labels, at K=5,000 they beat them (16.4 vs 17.9). MSN / DINO ViT features were the best feature
  extractors, beating label-supervised backbones — [Self-Guided Diffusion Models (ar5iv)](https://ar5iv.labs.arxiv.org/html/2210.06462), [PDF](https://arxiv.org/pdf/2210.06462)
- Rethinking cluster-conditioned diffusion (Adaloglou et al., WACV 2025 / arXiv 2403.00570): with the
  optimal cluster count, label-free cluster conditioning reaches FID 1.67 on CIFAR-10 and 2.17 on
  CIFAR-100 and gives "a strong increase in training sample efficiency"; optimal granularity was 100
  clusters for CIFAR-10, 200 for CIFAR-100, 400 for FFHQ-64 — i.e. ~10x more clusters than there are
  real classes; they also report no significant association between clustering *accuracy* and
  cluster-conditional FID — [arXiv 2403.00570](https://arxiv.org/pdf/2403.00570), [WACV 2025 PDF](https://openaccess.thecvf.com/content/WACV2025/papers/Adaloglou_Rethinking_Cluster-Conditioned_Diffusion_Models_for_Label-Free_Image_Synthesis_WACV_2025_paper.pdf), [alphaXiv summary](https://www.alphaxiv.org/abs/2403.00570)
- Representation-Conditioned Generation (RCG, Li et al., NeurIPS 2024): conditioning generation on a
  *generated* self-supervised representation vector rather than a label takes unconditional ImageNet
  256x256 FID from a prior best of 5.91 to 2.15 (-64%), and reduces unconditional FID for LDM-8, ADM,
  DiT-XL/2, MAGE-B and MAGE-L by 71%, 76%, 82%, 54% and 51% respectively, putting label-free
  generation "in the same tier" as class-conditional models — [arXiv 2312.03701](https://arxiv.org/pdf/2312.03701), [NeurIPS 2024 PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/e304d374c85e385eb217ed4a025b6b63-Paper-Conference.pdf), [code](https://github.com/LTH14/rcg)
- Memorisation cost of conditioning (Gu et al., "On Memorization in Diffusion Models", arXiv 2310.02664):
  memorisation occurs mainly on smaller datasets; larger data delays it, larger models accelerate it;
  and conditioning training data on *uninformative random* labels made >65% of generated samples exact
  replicas of training images on full 50k CIFAR-10, against 0% without that conditioning — the given
  interpretation is that conditioning restricts the effective sample set contributing to the density
  estimate, increasing local sparsity — [arXiv 2310.02664](https://arxiv.org/pdf/2310.02664), [OpenReview](https://openreview.net/forum?id=D3DBqvSDbj)
- Classifier-free guidance itself can trigger memorisation: CFG pushes the trajectory into an
  "attraction basin" around a memorised sample; delaying CFG until a transition point avoids it, and
  the effect is documented specifically for fine-tuning on small datasets and for duplicated data —
  [Jain et al., CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Jain_Classifier-Free_Guidance_Inside_the_Attraction_Basin_May_Cause_Memorization_CVPR_2025_paper.pdf), [arXiv 2411.16738](https://arxiv.org/abs/2411.16738)
- Data-efficiency at ~5k images is achievable with training-scheme changes rather than conditioning:
  Patch Diffusion reports >=2x faster training and better quality on datasets "as few as 5,000
  images" — [arXiv 2304.12526](https://arxiv.org/abs/2304.12526), [NeurIPS 2023 PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/e4667dd0a5a54b74019b72b677ed8ec1-Paper-Conference.pdf)

### Inferences
- The size of the conditioning gain scales with how much the condition simplifies the conditional
  distribution, not with the existence of a label. A 91-class label over 13.5k samples leaves ~150
  samples per class of a still very diverse conditional distribution; the papers that get 2-3x FID
  gains condition on 1,000-5,000 groups or on a continuous representation vector. That is consistent
  with the project's earlier null result from class conditioning (report 18.12/18.16) and predicts
  that a *finer* or *continuous* condition is where the leverage is.
- Fine conditioning and memorisation are the same mechanism seen from two sides (fewer effective
  samples per condition). At N = 13,555 any strong conditioning must be paired with a
  nearest-training-neighbour distance check, which the project already specifies as its novelty
  measurement.

### Gaps
- No paper found that sweeps N (dataset size) x number of classes on a fixed architecture and reports
  FID, so the "how much data does the label save" question has no direct curve; the sample-efficiency
  claim in Adaloglou et al. is stated qualitatively in every summary I could reach, and the exact
  factor (iterations or samples to a target FID) could not be extracted — the arXiv and CVF PDFs
  exceeded the fetch size limit / returned HTTP 403.
- No evidence found on class-conditional vs unconditional *flow-matching* specifically at N ~ 10^4;
  all matched comparisons above are DDPM/EDM-family diffusion.

## Q2. Video prediction from a frame vs generation from nothing, at 64x64

### Takeaway
On identical data and architecture the prediction task (given past frames) is far easier than
unconditional generation, and the published numbers are not close: MCVD's own UCF-101 64x64
unconditional FVD is 1143.0 while the same family reaches FVD 87.9-95.6 on BAIR and 144-277 on KTH
when conditioned on 1-10 past frames; the MCVD authors state outright that generation remains
substantially more challenging than prediction. The compute for these small 64x64 results is modest
by 2026 standards: 40-193 GPU-hours on a single V100/A100 per dataset, 28-740M parameters.

### Cited findings
- MCVD (Voleti et al., NeurIPS 2022): 64x64 for SMMNIST, KTH, BAIR and UCF-101 (128x128 for
  Cityscapes); parameter counts 27.9M (SMMNIST concat) up to 739.4M (UCF spatin); 2-10 conditioning
  past frames; models predict 4-5 frames per block, extended autoregressively —
  [ar5iv 2205.09853](https://ar5iv.labs.arxiv.org/html/2205.09853), [NeurIPS PDF](https://papers.neurips.cc/paper_files/paper/2022/file/944618542d80a63bbec16dfbd2bd689a-Paper-Conference.pdf)
- MCVD results: BAIR 64x64 FVD 87.9-125.8 across variants (concat past-mask: FVD 95.6, PSNR 18.8,
  SSIM 0.832, predicting 15 frames from 1 conditioning frame); KTH 64x64 FVD 276.7 for 40-frame
  prediction; Cityscapes 128x128 FVD 141.31 vs prior best 418 — [ar5iv 2205.09853](https://ar5iv.labs.arxiv.org/html/2205.09853)
- MCVD UCF-101 *unconditional* generation: FVD 1143.0 (spatin) vs DIGAN 655.0; the authors
  acknowledge generation remains substantially more challenging than prediction across datasets —
  [ar5iv 2205.09853](https://ar5iv.labs.arxiv.org/html/2205.09853)
- MCVD training cost: 39.7 GPU-hours (SMMNIST) to 192.83 GPU-hours (Cityscapes), batch size 64, on
  Tesla V100 or A100; per-dataset: KTH concat 65.7 V100-h, KTH spatin 45.8 A100-h, BAIR concat 78.2
  V100-h, BAIR spatin 50.0 A100-h; overall "1-12 days using <= 4 GPUs" —
  [ar5iv 2205.09853](https://ar5iv.labs.arxiv.org/html/2205.09853), [arXiv abs](https://arxiv.org/abs/2205.09853)
- Video Diffusion Models (Ho et al. 2022), 16x64x64: UCF-101 unconditional Inception Score 57 (+-0.62)
  and FVD 295 (+-3), against TGAN-v2 IS 26.60 / FVD 3431 and DVD-GAN; BAIR Robot Pushing prediction
  (1 conditioning frame -> 15 frames) FVD 66.92 with reconstruction-guided Langevin sampling (256
  steps), beating the prior best 86.9 — [NeurIPS 2022 PDF](https://proceedings.neurips.cc/paper_files/paper/2022/file/39235c56aef13fb05a6adc95eb9d8d66-Paper-Conference.pdf), [ar5iv 2204.03458](https://ar5iv.labs.arxiv.org/html/2204.03458)
- VDM compute: the large 16x64x64 text-conditioned model used 128 TPU-v4 chips for 700,000 steps
  (the 9x128x128 SR model 800,000 steps at comparable resources) — [ar5iv 2204.03458](https://ar5iv.labs.arxiv.org/html/2204.03458)
- VDM's own framing: models trained *unconditionally* can be turned into predictors at sampling time
  via reconstruction guidance and still reach state of the art — i.e. the conditioning does not have
  to be baked into training — [ar5iv 2204.03458](https://ar5iv.labs.arxiv.org/html/2204.03458)
- RaMViD (Höppe et al., "Diffusion Models for Video Prediction and Infilling", TMLR 2022): randomising
  which frames are the conditioning mask lets one architecture do prediction, infilling and
  unconditional generation, since the conditional and unconditional objectives are trained together;
  reported FVD 84 on BAIR — [arXiv 2206.07696](https://arxiv.org/pdf/2206.07696), [OpenReview](https://openreview.net/forum?id=lf0lr4AYM6)
- Benchmark sizes for context: BAIR ~44,000 robot-pushing videos at 64x64 (1 frame -> 15);
  UCF-101 13,320 videos / 101 classes rescaled to 64x64 — [ar5iv 2204.03458](https://ar5iv.labs.arxiv.org/html/2204.03458)

### Inferences
- The FVD scales are not directly comparable across papers (different clip lengths, samples and I3D
  protocols), but within MCVD — one codebase, one dataset family — unconditional UCF-101 FVD (1143)
  sits an order of magnitude above conditioned BAIR/KTH prediction FVD (88-277). Whatever part of
  that is task difficulty rather than protocol, the direction is unambiguous and is stated by the
  authors.
- UCF-101 at 13,320 clips is precisely the regime where every paper that tried unconditional
  generation got poor absolute numbers (MCVD 1143, TGAN-v2 3431) and only very large models trained
  with joint image+video data (VDM, 128 TPU-v4) got FVD 295. That is direct evidence that
  unconditional generation on this corpus size is not a small-compute task, while prediction from a
  given frame is (tens of GPU-hours on one card).
- RaMViD and VDM together show the practical route: train one model with a randomised
  conditioning mask (or train unconditionally and condition at sampling time by guidance), so the same
  weights serve conditioned and unconditioned sampling and the conditional objective acts as a
  learning signal that reduces the effective entropy of each training example.

### Gaps
- I could not retrieve a single paper that reports, for one model and one dataset, both the
  unconditional FVD and the first-frame-conditioned FVD in the same table — the cleanest available
  contrast is MCVD's UCF-101 generation vs BAIR/KTH prediction, which changes dataset as well as task.
- SVG-LP, Seer and the 2023-2026 "small video diffusion" works were not reached within the tool
  budget; their data/GPU-hour figures are unverified here.

## Q3. Factorising a clip into a still and motion; is motion low-entropy given the still?

### Takeaway
Every large image-to-video system is built on the assumption that the still carries most of the
information and motion is a comparatively cheap residual: SVD stages a frozen text-to-image model ->
video pretraining -> short high-quality finetune; AnimateDiff learns a motion module with the image
model completely frozen. The strongest small-data evidence is SinFusion, which learns the dynamics of
a *single* video from "mostly 2-3 dozen" frames and then generates diverse new videos of that scene
and extrapolates it in time. What is missing is a quantitative entropy or bits-per-frame comparison
between the still and the motion parts.

### Cited findings
- Stable Video Diffusion (Blattmann et al. 2023) identifies three stages — text-to-image pretraining
  (Stable Diffusion 2.1 as the initialisation), video pretraining, and high-quality video finetuning
  — and reports that training on *curated subsets* consistently beat training on larger uncurated
  data; total SVD training ~200,000 A100-80GB hours, mostly on 48x8 A100s —
  [arXiv 2311.15127](https://arxiv.org/pdf/2311.15127), [model card](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt), [overview](https://www.alphaxiv.org/overview/2311.15127v1)
- AnimateDiff (Guo et al., ICLR 2024): a plug-in motion module (temporal transformer inserted after
  ResNet/attention blocks) is trained once on WebVid-10M with the base text-to-image parameters
  frozen, plus a first-stage LoRA domain adapter to absorb video-data artefacts (watermarks); the
  learned motion prior then transfers to arbitrary personalised image models without per-model tuning —
  [ar5iv 2307.04725](https://ar5iv.labs.arxiv.org/html/2307.04725), [repo](https://github.com/guoyww/AnimateDiff), [motion adapter card](https://huggingface.co/guoyww/animatediff-motion-adapter-v1-5)
- SinFusion (Nikankin et al., ICML 2023): a diffusion model trained on a *single* image or video;
  in the video case it learns appearance and dynamics from "very few frames (mostly 2-3 dozens, but
  already apparent for fewer frames)", then generates diverse new videos of the same dynamic scene,
  extrapolates short videos far forward and backward in time, and does temporal upsampling; limits are
  small camera motion and non-rigid multi-part objects — [arXiv HTML 2211.11743](https://arxiv.org/html/2211.11743), [ICML PDF](https://proceedings.mlr.press/v202/nikankin23a/nikankin23a.pdf)
- MoCoGAN (Tulyakov et al., CVPR 2018) is the explicit content/motion split: a per-frame latent is
  split into a fixed content part and a stochastic motion process, learned with separate image and
  video discriminators; UCF-101 (13,220 videos, 101 classes) frames scaled to 85x64 —
  [CVPR 2018 PDF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Tulyakov_MoCoGAN_Decomposing_Motion_CVPR_2018_paper.pdf)
- VideoFusion (Luo et al., CVPR 2023) decomposes the per-frame diffusion noise into a shared base
  noise plus a residual, explicitly separating content from motion in the noise —
  [arXiv 2303.08320](https://arxiv.org/pdf/2303.08320)

### Inferences
- The industry stage design (image model first, motion second, image weights frozen) is itself the
  strongest available evidence that the motion factor is learnable with far less capacity and data
  than the appearance factor — but it is architectural evidence, not a measured entropy split.
- SinFusion is the concrete data point for "motion from very few samples once the content is given":
  a few dozen frames of one scene suffice to sample new, diverse dynamics. It does not show transfer
  of motion across scenes from few samples.

### Gaps
- No source found that measures the information content (bits, explained variance, or FVD at fixed
  budget) of the still versus the motion components of a clip, which is the number that would justify
  a "generate the still, then the motion" split quantitatively.
- AnimateDiff's motion-module GPU-hours and parameter count were not confirmed from the paper within
  the tool budget (the model cards give adapter files, not training cost).

## Q4. Retrieval / neighbour conditioning as a data-efficiency lever

### Takeaway
Retrieval conditioning is the best-documented "buy quality with data instead of parameters" lever:
RDM reaches competitive ImageNet 256 numbers with 400M trainable parameters against ADM's 554M and an
LDM baseline ~1.3x larger, using a 20M-image OpenImages database, k=4 neighbours at training and
15-20 at inference, with ~1 ms retrieval and 2 GB storage per 1M examples — and swapping the database
post-hoc moves the model to a new domain (zero-shot stylisation on WikiArt's 138k images) with no
retraining. kNN-Diffusion goes further and trains text-to-image with *no* paired text-image data at
all by conditioning on CLIP embeddings of retrieved neighbours.

### Cited findings
- RDM / Semi-Parametric Neural Image Synthesis (Blattmann, Rombach, Oktay, Müller, Ommer, NeurIPS 2022):
  RDM-OI 400M trainable parameters vs LDM baseline ~550M (1.3x more), ADM 554M, IC-GAN 191M;
  ImageNet 256x256 unconditional-style FID 12.21-24.50 and IS 45.29-70.64 depending on sampling
  strategy, against ADM FID 26.21 (train) / 32.50 (val) and ADM-G with classifier guidance 12.00;
  precision 0.57-0.72, recall 0.51-0.66 — [ar5iv 2204.11824](https://ar5iv.labs.arxiv.org/html/2204.11824), [NeurIPS page](https://proceedings.neurips.cc/paper_files/paper/2022/hash/62868cc2fc1eb5cdf321d05b4b88510c-Abstract-Conference.html)
- RDM retrieval setup: 20M OpenImages database, k=4 neighbours during training, 15-20 at inference,
  0.95 ms to retrieve 20 nearest neighbours, 2 GB storage per 1M examples; the paper's framing is that
  retrieval supplies *local content* while the model learns *scene composition* —
  [ar5iv 2204.11824](https://ar5iv.labs.arxiv.org/html/2204.11824), [abs](https://arxiv.org/abs/2204.11824), [code](https://github.com/CompVis/retrieval-augmented-diffusion-models)
- RDM post-hoc transfer: exchanging the OpenImages database for WikiArt (138k images) gives zero-shot
  text-guided stylisation without retraining; retrieved neighbours are kept disjoint from generated
  samples, and the paper's Figure 5 compares generations against training-set nearest neighbours to
  argue the outputs are new, unseen samples — [ar5iv 2204.11824](https://ar5iv.labs.arxiv.org/html/2204.11824)
- kNN-Diffusion (Sheynin et al., ICLR 2023): trains a small, efficient text-to-image diffusion model
  using only pretrained multimodal (CLIP) embeddings and kNN retrieval, with *no* explicit text-image
  dataset; training conditions on image embeddings and uses kNN to widen the conditioning distribution
  so the model generalises to text embeddings at sampling time; swapping the retrieval database at
  inference yields out-of-distribution generation —
  [arXiv 2204.02849](https://arxiv.org/pdf/2204.02849), [OpenReview](https://openreview.net/forum?id=x5mtJD2ovc)
- ReDi (Zhang et al., ICML 2023) is a different, inference-side use of retrieval: it retrieves
  *trajectories* from a precomputed knowledge base to skip intermediate diffusion steps
  learning-free — relevant to sampling cost, not to data efficiency — [arXiv 2302.02285](https://arxiv.org/pdf/2302.02285)

### Inferences
- The RDM/kNN-Diffusion result that matters for a 13.5k-sample corpus is not the FID but the
  mechanism: conditioning each training example on its own neighbours converts "learn the whole
  distribution" into "learn the local deformation from neighbours to target", which is a
  lower-entropy problem and the reason a smaller network suffices. The training corpus can double as
  the retrieval database.
- The novelty question is handled in those papers exactly the way this project already specifies it:
  show the generated sample beside its nearest training neighbour. No quantitative
  novelty-vs-neighbour metric is reported, so the check remains qualitative in the source literature.

### Gaps
- Neither RDM nor kNN-Diffusion reports a data-size ablation (FID vs N of training samples with and
  without retrieval), so the "gain at small N" is inferred from the parameter-count reduction and the
  stated motivation, not measured.
- No retrieval-conditioned *video* generator was found in this search.

## Q5. Conditioning on a frozen pretrained embedding at tiny scale

### Takeaway
The best-evidenced version of this idea is not CLIP text but self-supervised *image* representations:
RCG shows that generating a representation vector first and then conditioning on it closes most of
the unconditional-vs-conditional gap (up to -82% FID for DiT-XL/2), and Self-Guided Diffusion shows
DINO/MSN features used as the conditioning source beat ground-truth labels. unCLIP established the
two-stage prior -> embedding -> decoder pattern with a frozen CLIP, but at very large scale.

### Cited findings
- RCG (arXiv 2312.03701, NeurIPS 2024): the pipeline is a frozen self-supervised encoder (Moco v3), a
  small "representation diffusion model" that generates a representation vector unconditionally, and
  an image generator conditioned on that vector; ImageNet 256x256 FID 2.15 vs prior unconditional best
  5.91; per-backbone unconditional FID reductions of 71% (LDM-8), 76% (ADM), 82% (DiT-XL/2), 54%
  (MAGE-B), 51% (MAGE-L); the paper claims this is reached at lower training cost than current
  generative models — [arXiv 2312.03701](https://arxiv.org/pdf/2312.03701), [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/e304d374c85e385eb217ed4a025b6b63-Paper-Conference.pdf), [v2 HTML](https://arxiv.org/html/2312.03701v2)
- Self-Guided Diffusion (arXiv 2210.06462): MSN- and DINO-pretrained ViT features give the best
  FID/IS trade-off as the self-annotation source and "even improve over label-supervised backbones";
  ImageNet64 FID 12.1 with self-labeled guidance vs 36.1 unguided —
  [ar5iv 2210.06462](https://ar5iv.labs.arxiv.org/html/2210.06462)
- unCLIP / DALL-E 2 (Ramesh et al., arXiv 2204.06125): CLIP is frozen while the prior and decoder are
  trained; the diffusion prior beats the autoregressive prior, human raters preferred the full unCLIP
  stack for photorealism and caption similarity, and unCLIP reaches zero-shot MS-COCO FID 10.39 —
  [arXiv 2204.06125](https://arxiv.org/pdf/2204.06125), [ar5iv](https://ar5iv.labs.arxiv.org/html/2204.06125)

### Inferences
- RCG is the closest published analogue of what this project needs: the hard part of unconditional
  sampling is delegated to a *small* model operating in a low-dimensional semantic space, and the
  high-dimensional generator only ever runs conditioned. If the project's 16x8x721 state is split into
  "a compact semantic code" (e.g. the k=0 DCT still, or a PCA/embedding of it) and "the rest given the
  code", the structure matches RCG's two-stage design, and RCG's measured gains are for exactly the
  no-labels-available situation.
- Self-Guided Diffusion's monotone FID improvement with cluster count up to 5,000 plus Adaloglou's
  optimum at ~10x the true class count both say the same thing: a 91-way label is too coarse to be the
  useful conditioning signal on a corpus like UCF101.

### Gaps
- I found no paper that trains a CLIP/DINO-conditioned generator on ~10^4 samples and reports whether
  the samples are coherent and novel; RCG and Self-Guided Diffusion are ImageNet-scale (10^6),
  unCLIP is web-scale. The small-N behaviour of embedding conditioning is unverified.
- The DALL-E 2 ablation numbers surfaced by search ("FIDs of 9.16, 7.99 and 16.55" for a
  text-embedding-conditioned small decoder vs a small unCLIP stack vs a third variant) could not be
  attributed to the specific variants with confidence from the snippet, so they are not reported as a
  finding above.
- No evidence found on flow matching (as opposed to diffusion) with representation conditioning.

## What this implies for a state prior conditioned on a still or a class (strictly within the sources)

- A 91-way class label over 13,555 samples is the weakest form of this lever, and the sources predict
  a small effect: matched comparisons put the class-label gain at 2.72 -> 2.24 FID on CIFAR-10
  ([2212.00362](https://ar5iv.labs.arxiv.org/html/2212.00362)), and both cluster-conditioning papers
  find the useful granularity is ~10x-50x the number of real classes
  ([2403.00570](https://arxiv.org/pdf/2403.00570); [2210.06462](https://ar5iv.labs.arxiv.org/html/2210.06462)).
  The project's null result from class conditioning is what this literature would expect.
- The documented large effects come from conditioning on something continuous and fine-grained: a
  self-supervised representation (RCG: -51% to -82% unconditional FID,
  [2312.03701](https://arxiv.org/pdf/2312.03701)) or thousands of clusters (ImageNet64 36.1 -> 12.1,
  [2210.06462](https://ar5iv.labs.arxiv.org/html/2210.06462)). A still-image-like code (the k=0 DCT
  coefficients, or a low-dimensional embedding of them) is a continuous, fine-grained condition of
  exactly this kind; RCG's design — generate the code with a small model, then generate the full
  object conditioned on it — is the published template for doing that without labels.
- Video specifically: the conditioned task is the one small compute can reach. 64x64 conditional
  prediction on small corpora costs 40-193 GPU-hours on one V100/A100 with 28-740M parameters
  ([ar5iv 2205.09853](https://ar5iv.labs.arxiv.org/html/2205.09853)), while unconditional UCF-101
  generation at the same resolution gave FVD 1143 for MCVD and required 128 TPU-v4 chips for 700k
  steps in VDM to reach FVD 295 ([ar5iv 2204.03458](https://ar5iv.labs.arxiv.org/html/2204.03458)).
  MCVD's authors state generation is substantially harder than prediction.
- Training one model that handles both cases is documented practice, not a compromise: RaMViD
  randomises the conditioning mask so conditional and unconditional objectives train together
  ([2206.07696](https://arxiv.org/pdf/2206.07696)), and VDM conditions an unconditionally trained model
  at sampling time by reconstruction guidance ([2204.03458](https://ar5iv.labs.arxiv.org/html/2204.03458)).
  Either keeps the "no source clip" claim available while training on the easier objective.
- Retrieval conditioning is a way to use the same 13,555 states twice — as training data and as the
  database. RDM got competitive ImageNet numbers at 400M parameters vs ADM's 554M with k=4 neighbours
  at training ([ar5iv 2204.11824](https://ar5iv.labs.arxiv.org/html/2204.11824)); its neighbour-vs-sample
  figure is the same novelty check this project already requires.
- The price of every strong conditioning signal is memorisation, and it is measured: uninformative
  per-sample conditioning drove replication from 0% to >65% on 50k CIFAR-10
  ([2310.02664](https://arxiv.org/pdf/2310.02664)), memorisation concentrates on small datasets, and CFG
  itself can steer samples onto memorised ones, notably when fine-tuning on small data
  ([CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Jain_Classifier-Free_Guidance_Inside_the_Attraction_Basin_May_Cause_Memorization_CVPR_2025_paper.pdf)).
  Any conditioned state prior at N ~ 10^4 therefore needs the nearest-training-state distance reported
  beside the sample, and guidance scale treated as a memorisation knob.
- Nothing in these sources supports a claim that conditioning makes a coherent generator possible at
  N ~ 10^4 in this specific setting; the evidence supports only that the conditioned objective is
  cheaper and scores better than the unconditional one at matched budget, and that fine/continuous
  conditions are where the measured gains are.
