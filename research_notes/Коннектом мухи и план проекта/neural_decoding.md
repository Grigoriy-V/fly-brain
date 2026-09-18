# Neural decoding / brain-to-image reconstruction: methods that transfer to a simulated connectome-constrained fly network

Scope note: research conducted 2026-09-18 via web search + page fetches. Several full-text pages were blocked (ScienceDirect, Springer, bioRxiv rate-limit), so some entries rely on abstracts/search snippets; those are flagged. Anything not sourced is in Gaps.

---

## KQ1. State of the art in image/video reconstruction from neural data (2022–2026): fMRI, invasive animal recordings, retina, insects

### Takeaway
The field has converged on a two-part recipe: (a) a cheap mapping from neural activity to an intermediate latent (pixels, VAE/VDVAE latents, CLIP embeddings, or a DNN feature vector), and (b) a strong pretrained generative prior (Stable Diffusion / SDXL / video diffusion / denoiser) to turn that latent into a plausible image or video. For animal recordings at single-neuron resolution (mouse V1, macaque V1/V4/IT, primate RGCs) linear or small CNN/MLP decoders remain competitive, and the most recent mouse-V1 *movie* result (Bauer, Margrie & Clopath, eLife 2026) was obtained not with a trained decoder at all but by gradient-descent inversion of a differentiable encoding model. No insect (Drosophila) stimulus-reconstruction study was found.

### Cited Findings

**fMRI (human)**
- Takagi & Nishimoto (CVPR 2023) map fMRI to Stable Diffusion's VAE latent (early visual cortex) and CLIP text embeddings (ventral cortex) with *linear regression* only, then run the LDM; on NSD their reported numbers are CLIP similarity 0.642, SSIM 0.300, PCC 0.175 — [CVPR poster](https://cvpr.thecvf.com/virtual/2023/poster/21159); [GitHub](https://github.com/yu-takagi/StableDiffusionReconstruction); numbers quoted from a comparison table in [Towards Interpretable Visual Decoding (arXiv 2509.23566)](https://arxiv.org/pdf/2509.23566)
- Brain-Diffuser (Ozcelik & VanRullen, Sci. Rep. 2023) is a two-stage framework: stage 1 reconstructs low-level layout from fMRI via a VDVAE (very deep VAE) latent; stage 2 refines with Versatile Diffusion conditioned on predicted CLIP-vision and CLIP-text features — [Scientific Reports](https://www.nature.com/articles/s41598-023-42891-8)
- MindEye (Scotti et al., NeurIPS 2023): fMRI → CLIP image space with contrastive learning + a diffusion prior; retrieval and reconstruction submodules — [arXiv 2305.18274](https://arxiv.org/pdf/2305.18274)
- MindEye2 (Scotti et al., ICML 2024): linear (ridge) functional alignment of every subject into a shared latent space, shared non-linear MLP backbone to CLIP image space, diffusion prior, and SDXL fine-tuned to accept CLIP latents ("unCLIP"); pretrained on 7 NSD subjects then fine-tuned on 1 hour of a new subject's data; human raters picked MindEye2 reconstructions as closer to ground truth 97.82% of the time; metrics: PixCorr, SSIM, AlexNet(2/5), Inception, CLIP, EffNet-B, SwAV, plus image retrieval — [arXiv 2403.11207](https://arxiv.org/abs/2403.11207); [GitHub MedARC-AI/MindEyeV2](https://github.com/MedARC-AI/MindEyeV2); [Stability AI summary](https://stability.ai/news/mindeye2-fmri-to-image-with-1-hour-of-data)
- MinD-Vis (Chen et al., CVPR 2023) is the standard fMRI baseline that was also ported to mouse data (see Sensorium-Viz below) — [Sensorium-Viz PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13248838/)
- Video: NeuroClips (NeurIPS 2024) uses a "semantics reconstructor" (keyframes) plus a "perception reconstructor" (low-level flow) to condition a pretrained text-to-video diffusion model; reconstructs up to 6 s at 8 FPS; reports 128% SSIM and 81% spatiotemporal-metric improvement over MindVideo; metrics are SSIM, PixCorr, 2-way/50-way identification, and video-level classification — [arXiv 2410.19452](https://arxiv.org/abs/2410.19452); [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/5c594bf6223b67109441c9e0c97542ed-Paper-Conference.pdf)
- Video metrics convention (from MindVideo/NeuroClips line): frame-based metrics (SSIM, PSNR, PixCorr, N-way top-K classification over 1000 ImageNet classes) vs video-based metrics (N-way classification with a VideoMAE classifier over 400 Kinetics classes, emphasizing temporal smoothness) — [NeuroClips arXiv](https://arxiv.org/pdf/2410.19452); [SemVideo arXiv 2602.21819](https://arxiv.org/pdf/2602.21819)
- A 2025 survey of fMRI-to-image reconstruction organizes methods into fMRI signal encoding, feature mapping, and image generator stages, and lists data scarcity, cross-subject variability, and low semantic consistency as the main open problems — [arXiv 2502.16861](https://arxiv.org/abs/2502.16861)

**Mouse V1 (two-photon / Neuropixels)**
- Bauer, Margrie & Clopath, "Movie reconstruction from mouse visual cortex activity" (eLife 2026): ~8,000 GCaMP6s neurons per mouse (Sensorium 2023 data, 10 mice), 10-s natural movies at 30 Hz, 50 clips reconstructed from 5 mice. Method = *inversion of the SOTA dynamic encoding model DwiseNeuro*: start from a blank video and optimize it by gradient descent until predicted responses match recorded responses. Spatio-temporal pixel correlation r = 0.569, mean-frame spatial correlation r = 0.512, ~2.4x better than prior static-image reconstruction (r = 0.24). Averaging 7 independently trained encoders improved results 28%; dropping 50% of neurons cost ~10% correlation, dropping 75% cost ~25%; high spatial (<1 px) and temporal (>30 Hz) frequencies not recoverable; ~60 min per 10-s video on an RTX 4070 — [eLife 105081](https://elifesciences.org/articles/105081)
- DwiseNeuro (1st place, Sensorium 2023 dynamic competition): core of factorized 3D conv blocks with residual connections, positional encoding, SiLU, then 3 FC "cortex" layers and per-mouse 1-D conv readout with Softplus; single-trial correlation 0.291, trial-averaged 0.542 on >78,000 neurons / 10 mice / >2 h of video per neuron — [GitHub lRomul/sensorium](https://github.com/lRomul/sensorium); [arXiv 2305.19654](https://arxiv.org/abs/2305.19654); [Retrospective, PubMed](https://pubmed.ncbi.nlm.nih.gov/39040641/)
- Sensorium-Viz (Deng, Schwendeman & Guan, Advanced Science 2026): conditional Diffusion Transformer (28 DiT blocks, adaLN conditioning) for mouse V1 static images; neurons embedded onto a 32x32 grid by inverse-distance weighting; 5 mice with 7,334–8,372 neurons, ~6,000 trials each, 144x256 grayscale ImageNet stimuli; PixCorr 0.4581 vs MinD-Vis 0.4140, SSIM 0.4235 vs 0.3862; 11.36x less training time than MinD-Vis; trained 80k steps, batch 32, on NVIDIA L40S; *no CLIP*; synthetic augmentation with 82,784 COCO images passed through a pretrained encoding model. Limitations: needs 10-repeat trial averaging, single-trial performance degrades substantially, ~20–40k image-response pairs needed without augmentation — [PMC13248838](https://pmc.ncbi.nlm.nih.gov/articles/PMC13248838/); [Wiley](https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.202520220)
- Yoshida & Ohki (Nat. Commun. 2020): natural images linearly decodable from a small number of highly responsive V1 neurons; adding the remaining neurons *degraded* reconstruction — [Nature Communications](https://www.nature.com/articles/s41467-020-14645-x)
- Earlier mouse V1 work decoded ~100 L2/3 neurons with an optimal linear estimator and an ANN, concluding readout is approximately linear — [bioRxiv 300392](https://www.biorxiv.org/content/10.1101/300392v1.full)
- Allen Neuropixels (Chen et al., PLOS Comput Biol 2024): MLP (input → 16,384 → 8,092 → 4,096) + conv encoder/decoder producing 256x256 frames; MSE loss, Adam 1e-3, batch 16, 400 epochs; SSIM/PSNR metrics; 13 areas, 355–2,443 cells per area — [PLOS CB](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012297)
- Frontiers 2023 (Allen Neuropixels): natural-movie frames reconstructed from *optimal linear classifiers*; visual cortex > thalamus/midbrain > hippocampus (~chance); accuracy higher when train/test behavioural state matches — [Frontiers Comput Neurosci](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2023.1269019/full)

**Macaque (multi-unit)**
- Brain2GAN (Dado et al., PLOS Comput Biol 2024): MUA from macaque visual cortex (faces, natural images); feature-disentangled StyleGAN w-latents explain more variance than z-latents or CLIP latents along the ventral stream; multivariate decoding of w-latents gives "state-of-the-art spatiotemporal reconstructions" — [PLOS CB](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012058); [bioRxiv](https://www.biorxiv.org/content/10.1101/2023.04.26.537962v2)
- Inverse Receptive Field Attention (Le et al., 2025): end-to-end decoder for macaque V1/V4/IT spikes that learns an "inverse receptive field" per pixel via attention, weighting neurons across visual field and feature space; no pretrained generative model needed — [arXiv 2501.03051](https://arxiv.org/abs/2501.03051)

**Retina**
- Brackbill et al. (eLife 2020): hundreds of macaque RGCs (ON/OFF parasol + midget), least-squares linear reconstruction S = RW; rho = 0.76 ± 0.12 over 2,250 images/15 recordings; log nonlinearity Δρ = −0.0017, pairwise interaction terms Δρ = +0.0093 (i.e., nonlinear additions negligible); movie reconstruction filters ~85% space-time separable and spatial parts correlated ρ = 0.87 with static filters — [eLife 58516](https://elifesciences.org/articles/58516)
- Kim et al. (Neural Computation 2021): nonlinear decoding of natural images from large-scale primate RGC recordings — [MIT Press](https://direct.mit.edu/neco/article/33/7/1719/100579/Nonlinear-Decoding-of-Natural-Images-From-Large)
- Wu, Brackbill, Sher, Litke, Simoncelli & Chichilnisky (NeurIPS 2022): MAP reconstruction combining a GLM encoder (with spike-history and neighbour-coupling terms) and a deep denoiser as implicit image prior; matches/exceeds SOTA perceptual similarity with far fewer parameters; degrading either the encoder or the prior hurts substantially — [NYU LCV abstract](https://www.cns.nyu.edu/~lcv/pubs/makeAbs.php?loc=Wu22); [PDF](https://www.cns.nyu.edu/pub/lcv/wu22-reprint.pdf)
- Zhang, Jia, Yu et al. (Neural Networks 2020) spike-image decoder (SID): deep network decoding natural scenes from RGC spikes; [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0893608020300435) (full text blocked; only abstract-level claim available). A 2025 follow-up decodes natural scenes "via learnable representations of neural spiking sequences" — [PubMed 40700800](https://pubmed.ncbi.nlm.nih.gov/40700800/) (full text blocked)
- Mouse RGC CNN reconstruction (bioRxiv 2022) exists — [bioRxiv 2022.06.10.482188](https://www.biorxiv.org/content/10.1101/2022.06.10.482188.full.pdf) (fetch rate-limited; details not verified)
- "Brain-inspired decoder" for natural image reconstruction from RGC and V1 spikes uses receptive-field priors — [PMC10185745](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10185745/)

**Insects**
- FlyVis (Lappalainen et al., Nature 2024): connectome-constrained deep mechanistic network of the fly optic lobe; parameters fit by training on motion detection; predicts ON/OFF split and direction selectivity neuron by neuron; released in PyTorch — [Nature](https://www.nature.com/articles/s41586-024-07939-3); [GitHub TuragaLab/flyvis](https://github.com/TuragaLab/flyvis); [docs](https://turagalab.github.io/flyvis/)
- A 2026 Neuroinformatics paper ("Range-Limited Scale Stability of Dynamic Direction Representations in a Connectome-Constrained Fly Visual Model") analyses direction representations inside FlyVis — [Springer](https://link.springer.com/article/10.1007/s12021-026-09811-3) (paywalled; not read)
- CCN 2024: data-driven DNN models of Drosophila optic-lobe calcium imaging, inputs are 30x30 pixel retina images — [CCN PDF](https://2024.ccneuro.org/pdf/554_Paper_authored_Data-driven-deep-neural-network-models-of-visual-processing-in-Drosophila.pdf) (encoding, not decoding; PDF not parseable here)

### Inferences
- For single-neuron-resolution data with thousands of units, the performance gap between linear and nonlinear decoders is small (retina: Δρ < 0.01; mouse V1: adding neurons can hurt a linear decoder), so a linear/ridge decoder is the correct first baseline for a simulated fly network.
- The mouse-V1 movie result that best matches the project's setting (thousands of neurons, temporal data, existing differentiable encoder) is Bauer et al. 2026, which is exactly encoder inversion.
- No insect stimulus-reconstruction paper was found; the fly project would be, as far as this search shows, first in that niche.

### Gaps
- Exact MindEye2 numeric table (PixCorr/SSIM etc.) not extracted (abstract only).
- MindVideo's own numbers not retrieved (only NeuroClips' relative improvement).
- Zhang et al. 2020 / 2025 RGC decoder details (cell counts, metrics) not accessible.
- No Drosophila or other insect image/video reconstruction study found with any query ("Drosophila visual stimulus decoding reconstruction", "optic lobe calcium imaging population decoder").

---

## KQ2. Methods (linear, CNN/U-Net, diffusion+CLIP, encoder inversion) and their metrics

### Takeaway
Four method families are in use: (1) linear/ridge decoders to pixels or latents; (2) MLP+CNN/U-Net decoders trained end-to-end with MSE; (3) latent-space decoders (CLIP / VAE / StyleGAN-w) followed by a frozen diffusion or GAN generator; (4) analysis-by-synthesis, i.e. gradient descent through a differentiable encoder, optionally with a learned image prior (MAP). Standard metrics are low-level (PixCorr, SSIM, PSNR), mid/high-level feature similarity (AlexNet-2/5, Inception, CLIP, EffNet-B, SwAV, LPIPS), and identification accuracy (2-way, N-way top-K, retrieval).

### Cited Findings
- Linear decoders: retina least-squares S = RW with rho 0.76 — [Brackbill eLife](https://elifesciences.org/articles/58516); Allen Neuropixels optimal linear classifiers reconstruct movie frames — [Frontiers 2023](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2023.1269019/full); Takagi & Nishimoto use only linear regression to LDM latents — [GitHub](https://github.com/yu-takagi/StableDiffusionReconstruction)
- Inverted encoding models (IEMs, cognitive-neuroscience lineage): fit a linear encoder (voxel = weighted sum of channels), invert the weight matrix to decode; criticisms: IEM reconstructions reflect population-level model responses, not the stimulus or single-unit tuning, and their apparent width conflates selectivity with SNR — [eNeuro 2018](https://www.eneuro.org/content/5/3/ENEURO.0098-18.2018); [eNeuro 2019](https://www.eneuro.org/content/6/2/ENEURO.0363-18.2019); enhanced IEM adds trial-by-trial goodness-of-fit — [bioRxiv](https://www.biorxiv.org/content/10.1101/2021.05.22.445245v3.full)
- MLP + conv encoder/decoder (U-Net-like) for Neuropixels movie frames, MSE loss, 400 epochs — [PLOS CB 2024](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012297)
- Diffusion Transformer conditioned on spatially embedded neural responses, no CLIP, 11x cheaper than MinD-Vis — [Sensorium-Viz](https://pmc.ncbi.nlm.nih.gov/articles/PMC13248838/)
- CLIP-aligned diffusion priors: MindEye/MindEye2 (contrastive + diffusion prior + unCLIP SDXL) — [arXiv 2403.11207](https://arxiv.org/abs/2403.11207); Brain-Diffuser (VDVAE + Versatile Diffusion with CLIP-vision/text) — [Sci Rep](https://www.nature.com/articles/s41598-023-42891-8); MindDiffuser (semantic + structural control) — [arXiv 2308.04249](https://arxiv.org/pdf/2308.04249)
- Feature-disentangled GAN latents (StyleGAN w) decoded linearly from macaque MUA — [Brain2GAN](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012058)
- Encoder inversion / analysis-by-synthesis: Bauer et al. optimize the input video of DwiseNeuro by backprop to match recorded activity — [eLife 105081](https://elifesciences.org/articles/105081); Wu et al. MAP with GLM encoder + denoiser prior — [NeurIPS 2022](https://www.cns.nyu.edu/~lcv/pubs/makeAbs.php?loc=Wu22); Shen et al. "Deep image reconstruction" optimizes pixels so DNN features match features decoded from fMRI — [PMC6347330](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6347330/); Koide-Majima et al. 2024 add Bayesian estimation + Langevin dynamics on top of the Shen framework, reporting 90.7% identification for imagery — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0893608023006470); [GitHub](https://github.com/nkmjm/mental_img_recon); an independent reanalysis raises concerns about selective reporting, inflated metrics, and whether the Bayesian sampling component helps — [arXiv 2511.07960](https://arxiv.org/html/2511.07960)
- Activation-maximization lineage (finding stimuli that maximize an encoder's predicted response rather than matching a recorded pattern): Inception loops / MEIs for mouse V1 (Walker et al., Nat Neurosci 2019) — [Nature Neuroscience](https://www.nature.com/articles/s41593-019-0517-x); XDREAM (Ponce et al. 2019) evolves images with a GAN to drive real neurons — [Semantic Scholar summary](https://www.semanticscholar.org/paper/Inception-loops-discover-what-excites-neurons-most-Walker-Sinz/309736b42bf16deba902d5d004fe97eedbcdefda); BOLDreams (2025) applies feature visualization ("dreaming") to fMRI encoding models — [arXiv 2501.14854](https://arxiv.org/abs/2501.14854); "Training-free stimulus encoding for retinal implants via sparse projected gradient descent" applies projected gradient descent through an encoder with stimulus bounds — [arXiv 2602.10906](https://arxiv.org/html/2602.10906)
- Metrics: MindEye2 uses PixCorr, SSIM, AlexNet(2), AlexNet(5), Inception, CLIP, EffNet-B, SwAV, retrieval — [arXiv 2403.11207](https://arxiv.org/abs/2403.11207); LPIPS compares deep features across layers and correlates better with human perception than pixel metrics — [NeuroClips arXiv](https://arxiv.org/pdf/2410.19452); mouse/animal papers mostly report pixel correlation and SSIM/PSNR — [eLife 105081](https://elifesciences.org/articles/105081), [PLOS CB 2024](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012297); Sensorium-Viz also reports "pairwise correlation similarity" (2-way identification by correlation) 91.91% — [PMC13248838](https://pmc.ncbi.nlm.nih.gov/articles/PMC13248838/)

### Inferences
- The metric split (low-level PixCorr/SSIM vs high-level CLIP/Inception/identification) exists because diffusion decoders produce semantically right but pixel-wrong images; for a fly model, where semantics are absent, pixel correlation, SSIM, and 2-way/N-way identification by correlation are the relevant metrics, plus spatiotemporal correlation for video.
- Encoder inversion needs no decoder training data but pays per-sample optimization cost (Bauer: ~1 GPU-hour per 10-s clip); trained decoders amortize this.

### Gaps
- No paper found that directly benchmarks trained decoders vs encoder inversion on the *same* animal dataset with the same metrics (Bauer et al. compare to prior static-image results only).

---

## KQ3. Layer-wise decodability across processing stages and information-theoretic measures

### Takeaway
Across CNNs and brains, pixel-level reconstructability decays monotonically with processing depth (Dosovitskiy & Brox for CNN layers; Chen et al. 2024 for mouse areas with R > 0.9 correlation to anatomical hierarchy score), while decoded content shifts from low-level to categorical. Information is usually quantified indirectly (reconstruction correlation/SSIM, identification accuracy, decoded-feature correlation, Fisher-information-based discrimination thresholds) rather than with mutual information directly.

### Cited Findings
- Dosovitskiy & Brox (CVPR 2016): up-convolutional network inverting HOG/SIFT and AlexNet layers; reconstructions become increasingly blurry with depth, and blurriness serves as an indication of the information preserved; colour and rough object position survive even from top layers — [arXiv 1506.02753](https://arxiv.org/abs/1506.02753)
- Shen et al. 2019: mid-level DNN layers preserve enough information to reconstruct perceptually similar images; hierarchical processing does not discard all pixel detail — [PMC6347330](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6347330/)
- Horikawa & Kamitani "decodability" framework: decode each DNN unit's activation from fMRI; decodability varies strongly across units but is consistent across subjects — [Scientific Data 2019](https://www.nature.com/articles/sdata201912)
- Brain Hierarchy (BH) score (Nonaka et al., iScience 2021): for each DNN unit find the "top ROI" (V1, V2, V3, V4, higher VC) that decodes it best; BH score = Spearman correlation between layer index and top ROI; across 29 DNNs, BH score is *negatively* correlated with ImageNet accuracy — [PMC8426272](https://pmc.ncbi.nlm.nih.gov/articles/PMC8426272/)
- Chen et al. (PLOS Comput Biol 2024), Allen Neuropixels, 13 areas: movie-frame reconstruction fidelity (SSIM/PSNR) highest in VISp, "considerable" in LGN/LP/APN, "notably inferior" in hippocampus; decoding quality correlates with anatomical hierarchy score (R > 0.9) and with receptive-field size; performance saturates at ~500 cells for cortex/thalamus vs ~1,000 for hippocampus; cross-area decoder transfer is at shuffled-baseline level, implying area-specific codes — [PLOS CB](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012297)
- Frontiers 2023 (same dataset, linear decoders): visual cortex > thalamus/midbrain > hippocampus (near chance) for movie-frame classification — [Frontiers](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2023.1269019/full)
- Retina cell-type contributions (Brackbill): ON+OFF ρ = 0.76 vs ON alone 0.64 / OFF alone 0.67; parasol+midget 0.81 vs parasol 0.77 / midget 0.73; ON and OFF carry largely independent, complementary information — [eLife 58516](https://elifesciences.org/articles/58516)
- Fisher-information-style measure: Stringer et al. (Cell 2021) decode orientation from up to 50,000 mouse V1 neurons with discrimination thresholds of 0.35° (V1) / 0.37° (higher areas), ~100x finer than behaviour, implying perceptual limits are set by downstream decoders not sensory noise — [Cell](https://www.cell.com/cell/fulltext/S0092-8674(21)00373-1); [PubMed](https://pubmed.ncbi.nlm.nih.gov/33857423/)
- A 2025 preprint proposes "hierarchical neural information gradients" for adaptive decoding in mouse visual tasks — [arXiv 2510.09451](https://arxiv.org/pdf/2510.09451) (not read in full)
- In the dream-decoding work, the decoded-feature correlation profile across DNN layers was used as the per-layer information measure — [PMC5281549](https://pmc.ncbi.nlm.nih.gov/articles/PMC5281549/)

### Inferences
- A direct analogue for the fly model: train identical decoders on each FlyVis cell type / layer (photoreceptors R1-6 → L1-L5 → Mi/Tm → T4/T5) and plot pixel-corr / SSIM / identification vs depth; compare ON vs OFF pathway contributions as Brackbill did.
- The "saturation at ~500 cells" observation in mouse suggests decodability curves vs neuron count are a standard and cheap analysis.

### Gaps
- No study found that reports mutual information (bits) between stimulus and successive layer activations for natural images; measures are all decoder-based proxies.
- No layer-wise decodability study found for any insect visual system.

---

## KQ4. Decoding from spontaneous / internally generated activity (replay, dreams, DeepDream, "what does the network dream")

### Takeaway
The established protocol is to train a decoder on stimulus-evoked activity and apply it to spontaneous activity (sleep-onset fMRI, hippocampal replay), validated against an independent report or behaviour; authors consistently describe the output as decoded *features/categories consistent with perception*, not as the percept itself. "Dreaming" of artificial networks (DeepDream, feature visualization, BOLDreams) is a different operation: input optimization to maximize unit activation, with no spontaneous activity involved. No project was found that runs a generative decoder on spontaneous activity of an artificial/connectome-constrained network.

### Cited Findings
- Horikawa & Kamitani (Science 2013; Front Comput Neurosci 2017): decoders trained on stimulus-evoked fMRI, applied to NREM stage 1–2 sleep-onset activity before awakening reports; dreamed object categories identified above chance; decoded DNN features correlate with the dreamed category's features mostly at mid-to-high layers, with correlations lower than for perception or imagery. Authors' caveats: analyses are category-level and give no evidence about low-level features in lower ROIs; results show feature recruitment "in a manner similar to perception" but do not prove dreams are replays; layer profile differs slightly from perception, possibly reflecting dream-generation mechanisms — [PMC5281549](https://pmc.ncbi.nlm.nih.gov/articles/PMC5281549/); [Frontiers](https://www.frontiersin.org/articles/10.3389/fncom.2017.00004/full)
- Mental imagery reconstruction (Koide-Majima 2024) targets volitional imagery, not spontaneous activity, and its validity is disputed — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0893608023006470); [reanalysis arXiv 2511.07960](https://arxiv.org/html/2511.07960)
- "Decoding in the dark" (V1, 2012): maximum-likelihood decoding of spontaneous activity in a blank period predicted the *preceding* stimulus orientation at ~25% accuracy, i.e., spontaneous activity carries stimulus-history information — [PMC3403617](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3403617/)
- Stringer et al. (Science 2019): in mouse V1 the behaviour-driven subspace of spontaneous activity is nearly orthogonal to the stimulus-driven subspace (one shared dimension); 32 natural images were decoded separately from stimulus-only, behaviour-only, and spontaneous-only subspaces — [PubMed](https://pubmed.ncbi.nlm.nih.gov/31000656/); [bioRxiv](https://www.biorxiv.org/content/10.1101/306019v1); a 2024 primate study reports similar orthogonalization across the areal hierarchy — [bioRxiv 2024.07.01.601463](https://www.biorxiv.org/content/10.1101/2024.07.01.601463v1.full)
- Hippocampal replay: a 2024 PNAS Nexus study argues replay sequences are governed by spontaneous brain-wide dynamics (internal circuit dynamics) rather than by the experience per se — [PNAS Nexus](https://academic.oup.com/pnasnexus/article/3/4/pgae078/7609348); replay of movie-related hippocampal sequences resembles place-cell sequence replay — [bioRxiv 2022.09.05.506667](https://www.biorxiv.org/content/10.1101/2022.09.05.506667v1.full); Science 2025 "The time course and organization of hippocampal replay" — [Science](https://www.science.org/doi/10.1126/science.ads4760) (not read)
- Human memory reinstatement decoding (eNeuro 2024) maps spatiotemporal trajectories of object-memory reconstruction during recall, asking whether retrieval is a reversed copy of perception or involves format transformations — [eNeuro](https://www.eneuro.org/content/11/9/ENEURO.0091-24.2024)
- DeepDream (Google, 2015) performs activation maximization / feature visualization by gradient ascent on the input; Olah, Mordvintsev & Schubert, "Feature Visualization", Distill 2017 formalized it — [GitHub DeepDream write-up](https://github.com/stephenjarrell19/DeepDream); "Diverse feature visualizations reveal invariances in early layers of deep neural networks" — [arXiv 1807.10589](https://arxiv.org/pdf/1807.10589)
- BOLDreams (2025): "dreaming" = feature visualization via input optimization on feature-weighted receptive-field fMRI encoding models (NSD); used with integrated-gradient attention maps to show biologically plausible features drive predicted signals; it is an in-silico probe of the encoder, not decoding of measured spontaneous activity — [arXiv 2501.14854](https://arxiv.org/abs/2501.14854)
- MEIs (Walker et al. 2019): images synthesized by gradient ascent on an encoder drove real V1 neurons better than controls when presented back in vivo, providing a closed-loop validation that synthesized stimuli are meaningful — [Nature Neuroscience](https://www.nature.com/articles/s41593-019-0517-x)

### Inferences
- The closest methodological template for "what does the fly network dream" is Horikawa & Kamitani's protocol: train the decoder (or calibrate the inversion) on stimulus-evoked simulated activity, then apply it to activity generated without input (noise-driven or recurrent), and validate with a controlled ground truth (e.g., stimulus-history in the "decoding in the dark" sense) rather than claiming a percept.
- Stringer's orthogonality result implies that a large part of spontaneous activity may lie outside the stimulus subspace; decoders should report the fraction of spontaneous variance that projects onto the stimulus subspace before showing decoded images.

### Gaps
- No published "generative decoder applied to spontaneous activity of an artificial or connectome-constrained network" project was found with queries combining spontaneous/dream/generative/replay terms.
- Sleep-decoding work more recent than 2017 (e.g. REM decoding with diffusion decoders) was not found in this search.

---

## KQ5. Which decoder architectures suit a few thousand to a few hundred thousand simulated neurons with temporal (video) data, and which are cheapest to train?

### Takeaway
Evidence from single-neuron-resolution recordings (retina, mouse V1, Neuropixels) says: start with ridge/linear regression to pixels (retina rho 0.76; nonlinear gains < 0.01), then an MLP+small conv decoder (Chen 2024: 4 FC layers + 4-conv encoder/6-conv decoder, 400 epochs) or an encoder-inversion loop; diffusion decoders (Sensorium-Viz) add a few points of PixCorr/SSIM but need ~20–40k image–response pairs (or synthetic augmentation from an encoding model) and L40S-class GPUs. For video, the only single-neuron-resolution SOTA is encoder inversion (Bauer 2026), costing ~1 GPU-hour per 10-s clip on an RTX 4070 with no decoder training.

### Cited Findings
- Linear decoding sufficient for retina and mouse V1 (see KQ1): [Brackbill](https://elifesciences.org/articles/58516); [Yoshida & Ohki](https://www.nature.com/articles/s41467-020-14645-x); [bioRxiv 300392](https://www.biorxiv.org/content/10.1101/300392v1.full)
- Adding more neurons to a linear decoder degraded reconstruction in mouse V1 (overfitting / noise); best results from a small set of highly responsive neurons — [Nature Communications 2020](https://www.nature.com/articles/s41467-020-14645-x)
- Decoding quality saturates at ~500 cells (cortex/thalamus) in Neuropixels movie decoding — [PLOS CB 2024](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012297)
- Bauer et al.: dropping 50% of ~8,000 neurons cost ~10% correlation; ensembling 7 encoders gave +28%; cost ~60 min per 10-s clip on RTX 4070 — [eLife 105081](https://elifesciences.org/articles/105081)
- Sensorium-Viz: DiT decoder, 11.36x less training than MinD-Vis, but 80k steps at batch 32 on L40S GPUs and 20–40k pairs needed; single-trial decoding fails without 10x averaging — [PMC13248838](https://pmc.ncbi.nlm.nih.gov/articles/PMC13248838/)
- Synthetic-data trick: pass 82,784 COCO images through a pretrained encoding model to generate extra (image, response) pairs for decoder training — [PMC13248838](https://pmc.ncbi.nlm.nih.gov/articles/PMC13248838/)
- MindEye2 shows that ridge alignment into a shared space + shared nonlinear head lets a decoder trained on other "subjects" transfer with 1 h of new data — [arXiv 2403.11207](https://arxiv.org/abs/2403.11207)
- Wu et al.: MAP decoding with GLM encoder + denoiser prior uses "substantially fewer parameters" than prior methods while matching perceptual quality — [NeurIPS 2022 abstract](https://www.cns.nyu.edu/~lcv/pubs/makeAbs.php?loc=Wu22)
- Video diffusion decoders for fMRI (NeuroClips) rely on pretrained text-to-video models and keyframe semantics — [arXiv 2410.19452](https://arxiv.org/abs/2410.19452)

### Inferences
- In a simulated network there is no trial noise, unlimited stimuli, and no need for repeat averaging, which removes the main limitations reported for Sensorium-Viz and Bauer et al.; the synthetic-augmentation trick is native (the simulator *is* the encoder).
- Cost ranking (cheapest first): ridge regression → MLP/conv decoder → encoder inversion (no training, per-sample cost) → diffusion/DiT decoder. For 10^5 simulated neurons, a spatial embedding of neurons onto a hexagonal/retinotopic grid (analogous to Sensorium-Viz's 32x32 IDW grid) plus 3D convolutions is the natural way to keep parameter counts low.

### Gaps
- No paper found that reports decoder training cost vs population size systematically beyond the Bauer and Chen ablations.
- No U-Net-specific decoder paper for single-neuron data was found (the conv encoder/decoder in Chen 2024 is the closest).

---

## KQ6. Is gradient-descent inversion of a differentiable encoder (FlyVis) a viable alternative to a trained decoder? Documented pros/cons

### Takeaway
Yes: the current best mouse-V1 movie reconstruction (Bauer et al. 2026) is exactly this, and the retina MAP work (Wu et al. 2022) and the Shen/Koide-Majima fMRI line use the same principle. Documented pros: no decoder training data needed, uses all the encoder's knowledge, easily combined with a prior. Documented cons: per-sample optimization cost, ill-posedness (needs ensembling / priors / bounds), failure at frequencies the encoder is insensitive to, and reconstructions inherit encoder errors from unmodelled factors.

### Cited Findings
- Bauer et al.: "iteratively optimized an initially blank input video to the SOTA DNEM until the predicted activity ... matched the ground truth recorded neuronal activity"; ~60 min per 10-s video on RTX 4070; ensembling 7 encoders +28%; fails at spatial frequencies < 1 px and temporal > 30 Hz; encoder predictions deviate from real activity for some stimuli "likely due to unmeasured top-down factors" — [eLife 105081](https://elifesciences.org/articles/105081)
- Wu et al. (NeurIPS 2022): MAP inversion of a GLM retinal encoder with a denoiser prior; "rudimentary encoding models" and simpler priors significantly reduce quality, i.e., the approach is only as good as encoder + prior — [NYU abstract](https://www.cns.nyu.edu/~lcv/pubs/makeAbs.php?loc=Wu22)
- Retinal-implant stimulus design uses projected gradient descent through an encoder with sparsity and stimulus bounds, "training-free" — [arXiv 2602.10906](https://arxiv.org/html/2602.10906)
- Inverted-encoding-model critiques: reconstructions reflect the model's assumed channels, are ill-posed for single-unit inference, and their shape depends on SNR — [eNeuro 2018](https://www.eneuro.org/content/5/3/ENEURO.0098-18.2018); [eNeuro 2019](https://www.eneuro.org/content/6/2/ENEURO.0363-18.2019)
- Shen et al. (2019) inversion through a DNN's feature space (pixels optimized so DNN features match decoded features) produced recognizable natural images and letters — [PMC6347330](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6347330/); Koide-Majima's Langevin/Bayesian extension is disputed by a reanalysis for selective reporting and inflated metrics — [arXiv 2511.07960](https://arxiv.org/html/2511.07960)
- Closed-loop validation exists for encoder-synthesized stimuli (MEIs drive real neurons better than controls) — [Walker et al. 2019](https://www.nature.com/articles/s41593-019-0517-x)
- FlyVis is a PyTorch model with released weights — [GitHub](https://github.com/TuragaLab/flyvis); fit on motion-detection tasks and validated neuron by neuron on ON/OFF and direction selectivity — [Nature 2024](https://www.nature.com/articles/s41586-024-07939-3)

### Inferences
- With FlyVis the "unmeasured top-down factors" problem disappears when activity is simulated by the same model (inversion becomes an exact-model inverse problem); the remaining issues are ill-posedness (many stimuli produce the same activity, especially in later layers) and compute per clip. A learned prior (denoiser / diffusion) and ensembling over the FlyVis parameter ensemble (the released model ships multiple trained instances) are the literature-supported remedies.
- Inversion output should be reported with a "compatibility" score (encoder response correlation between recorded and re-predicted activity), which Bauer et al. and Wu et al. both use implicitly as the optimization objective.

### Gaps
- No paper directly compares inversion vs trained decoder on identical data with wall-clock and accuracy; the trade-off above is assembled from separate studies.
- No published inversion of FlyVis specifically.

---

## KQ7. How do papers handle the caveat that images decoded from spontaneous activity are "most compatible stimulus," not percepts?

### Takeaway
The literature handles this in three ways: (1) wording — decoded outputs are described as features/categories "recruited in a manner similar to perception," never as what was seen/dreamed; (2) validation against an external ground truth (dream reports, preceding stimulus, behavioural replay, closed-loop presentation); (3) subspace analysis showing how much spontaneous activity lies in the stimulus-driven subspace at all.

### Cited Findings
- Horikawa & Kamitani explicitly state they cannot speak to low-level features in lower ROIs, that positive correlations show feature recruitment "in a manner similar to perception," and that dreams may additionally involve memory and abstract knowledge — [PMC5281549](https://pmc.ncbi.nlm.nih.gov/articles/PMC5281549/)
- "Decoding in the dark" frames spontaneous-activity decoding as recovering information about *preceding stimuli* (25% accuracy), i.e., a testable ground truth — [PMC3403617](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3403617/)
- Stringer et al. quantify that spontaneous/behavioural activity is nearly orthogonal to the stimulus subspace with one shared dimension, and decode images separately from each subspace — [PubMed 31000656](https://pubmed.ncbi.nlm.nih.gov/31000656/)
- IEM critiques: a reconstruction is "an arbitrary model response, not the stimulus" and reflects population-level assumptions — [eNeuro 2019](https://www.eneuro.org/content/6/2/ENEURO.0363-18.2019)
- BOLDreams labels its outputs as feature visualizations of an *in-silico* encoding model — [arXiv 2501.14854](https://arxiv.org/abs/2501.14854)
- Hippocampal replay work treats decoded sequences as products of internal circuit dynamics and validates against experienced trajectories/movies — [PNAS Nexus 2024](https://academic.oup.com/pnasnexus/article/3/4/pgae078/7609348)
- Closed-loop validation: MEIs presented back in vivo drove target neurons more than controls — [Walker 2019](https://www.nature.com/articles/s41593-019-0517-x)
- Sensorium-Viz notes single-trial reconstructions are unreliable without averaging, illustrating that decoders return the most likely image under noise, not the trial's percept — [PMC13248838](https://pmc.ncbi.nlm.nih.gov/articles/PMC13248838/)

### Inferences
- A defensible framing for the fly project: call outputs "stimulus most compatible with the network state under the encoder," report the compatibility score and the projection of spontaneous activity onto the stimulus subspace, and where possible use a ground truth (e.g., inject a stimulus, silence input, and test whether decoded images track stimulus history).

### Gaps
- No formal, widely adopted statistic for "percept vs most-compatible-stimulus" was found; the field relies on wording and external validation.
