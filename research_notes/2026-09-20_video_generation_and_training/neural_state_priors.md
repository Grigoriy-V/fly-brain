# Generative priors over neural population activity: methods and evaluation practice

Scope: literature review conducted 2026-09-20 via web search and paper fetches, for item 17 (brain-state prior, `ROADMAP.md`). Our setting: a flow-matching model trained over 0.8 
s trajectories of 8 motion-selective cell types (T4a-d, T5a-d) on 721 columns; a sample from noise is decoded to video by a separate model (13B) and re-read through the frozen 
connectome-constrained brain ("round trip"). Below, **rate** means a continuous, deterministic firing-rate-like state per cell/type/step, as ours is; every method found instead 
assumes **spikes** (point process / binned counts) or **calcium** (fluorescence), flagged per entry. Some PDF fetches returned corrupted binary content this session; where that 
happened the note used the HTML/abstract mirror instead, flagged inline, and low-confidence claims were dropped rather than reported (see Gaps).

## 1. Generative priors over population activity

Two families, one important distinction: most of what is called a "generative model of neural activity" in this literature is actually an *inference* model — a sequential VAE 
trained to denoise or reconstruct real single-trial recordings, whose latent code can be interpolated but whose unconditional prior is rarely sampled on its own (the LFADS line). 
A smaller, newer set of GAN/diffusion/flow models supports true unconditional sampling from noise, which is the operation our flow-matching prior performs.

- **LFADS line — inference, not free sampling (spikes).** LFADS itself (Sussillo, Jozefowicz, Abbott & Pandarinath, preprint 2016, 
[arXiv:1608.06315](https://arxiv.org/abs/1608.06315); published as Pandarinath et al., *Nature Methods* 2018) is a sequential VAE: an encoder RNN maps observed spike counts to a 
per-trial initial condition, a generator RNN produces low-dimensional factors, and a Poisson observation model reads out de-noised firing rates from real spikes — it is not 
normally run as an unconditional sampler. AutoLFADS (Keshtkaran et al.) automates its hyperparameter search with population-based training and coordinated dropout. Later members 
relax or extend the VAE: pi-VAE ([arXiv:2011.04798](https://arxiv.org/abs/2011.04798), Zhou & Wei, NeurIPS 2020) restores identifiability by conditioning the latent on a 
behavioural label but drops LFADS's temporal dynamics, treating each time bin as an independent Poisson draw; TNDM (Hurwitz et al., NeurIPS 2021, 
[PDF](https://proceedings.neurips.cc/paper/2021/file/f5cfbc876972bd0d031c8abc37344c28-Paper.pdf)) splits the latent into behaviourally relevant and irrelevant subspaces; the 
Neural Data Transformer (Ye & Pandarinath) replaces the RNNs with a masked-autoencoder Transformer trained by predicting masked spikes. All report tens-to-low-hundreds of neurons 
(single motor/sensory areas) and are scored by how well they reconstruct held-out real trials, not by the realism of freely sampled ones.
- **GAN line — true samplers, small populations (spikes / calcium).** Spike-GAN (Molano-Mazón, Onken, Piasini & Panzeri, ICLR 2018, 
[arXiv:1803.00338](https://arxiv.org/abs/1803.00338)) adapts Wasserstein-GAN to binary spike patterns from tens of salamander retinal ganglion cells, explicitly benchmarked 
against classical maximum-entropy and dichotomised-Gaussian population models, matching first- and second-order statistics and approximating higher-order ones without specifying 
those statistics up front. CalciumGAN (Li, Amvrosiadis, Rochefort & Onken, [arXiv:2009.02707](https://arxiv.org/abs/2009.02707)) is the calcium-trace analogue: a WaveGAN-style 1-D 
convolutional generator/critic trained with the Wasserstein distance, evaluated by KL divergence between real/generated distributions, per-neuron mean firing rate, and pairwise 
Pearson correlation.
- **Diffusion line — true samplers, motor cortex (spikes).** GNOCCHI (McCart, Sedler, Versteeg, Mifsud, Rigotti-Thompson & Pandarinath, 2024, 
[arXiv:2407.21195](https://arxiv.org/abs/2407.21195)) extends InfoDiffusion (a diffusion model with an auxiliary disentangling encoder) to smoothed spiking activity from 
96-electrode macaque M1 recordings during self-paced reaching; it generates novel samples by linear traversal of the learned code and is validated by ridge-decoding hand position 
and held-out target location from generated activity, not by re-encoding through a forward model. LDNS (Kapoor, Schulz, Vetter, Pei, Gao & Macke, NeurIPS 2024, 
[arXiv:2407.08751](https://arxiv.org/abs/2407.08751); [code](https://github.com/mackelab/LDNS)) is the closest architectural analogue to a diffusion/flow prior over population 
trajectories: an S4 (structured-state-space) autoencoder compresses discrete spike trains into continuous latents, and a diffusion model trained over those latents is sampled 
**fully unconditionally from Gaussian noise**, or conditionally on a behavioural covariate (reach angle, reach-velocity trace). Demonstrated on macaque motor-cortex reaching 
(DANDI 000128) and human intracortical BCI data during attempted speech (~100-200 units). To our knowledge this is the nearest published instance of "train a diffusion/flow model 
over population trajectories, then sample brand-new ones from noise" — the same operation as our 13B-fed prior, over discrete spikes rather than continuous type-level rates.
- **Flow line, closer to our own method (rate-like latents, not raw data).** A 2026 survey (Kong, Deng, Dong, Liu, Chen, Wang et al., 
[arXiv:2606.10530](https://arxiv.org/abs/2606.10530)) catalogues Neural-ODE and flow-based dynamics models as a distinct branch: PLNDE and ODIN model deterministic latent 
evolution with a Neural ODE, and LangevinFlow ([arXiv:2507.11531](https://arxiv.org/abs/2507.11531)) is a sequential VAE whose latent evolves by an underdamped Langevin SDE with a 
learned potential, built to bias generation toward the oscillatory, locally-coupled dynamics seen in real neural populations. None of these is flow-matching in the 
SiT/rectified-flow sense we use; they are closer to neural-SDE priors than to a straight-line noise-to-data velocity field, and they still operate on an inferred *latent*, not 
directly on a population rate vector.
- **Rate vs. spikes vs. calcium, and why it matters here.** Everything above except CalciumGAN is fit to spikes; CalciumGAN fits calcium fluorescence. No method found operates on 
a continuous, deterministic rate variable per cell type the way our FlyVis-derived T4/T5 trajectories are. Our setting has no observation-noise or discretisation step between the 
generative target and the "real" data, which the inference-model literature exists partly to handle (Poisson/spike-noise removal). That should make distributional realism easier 
to hit for us and harder to read as "biological realism," since there is no real recording-noise floor to match, only FlyVis's own simulated one.

## 2. How plausibility of generated activity is evaluated

No single agreed "necessary" test exists, but one recurring candidate comes closest: in the LFADS/inference line, the benchmark treated as closest to a gate rather than a nicety 
is co-smoothing / bits-per-spike from the Neural Latents Benchmark (Pei, Ye, Zoltowski, Wu, Chowdhury, Sohn, O'Doherty, Shenoy, Kaufman, Churchland, Jazayeri, Miller, Pillow, 
Park, Dyer & Pandarinath, NeurIPS 2021 Datasets & Benchmarks, [arXiv:2109.04463](https://arxiv.org/abs/2109.04463)): split neurons into held-in and held-out, infer latents from 
held-in neurons only, and score the predicted firing rate of held-out neurons against their real spikes, normalised to bits/spike using each neuron's mean rate. It is a genuine 
generalisation test — a model that only memorises per-neuron statistics fails it — and is this field's closest analogue to a held-out-scene control (Q4). It only applies where 
a real population recording exists to split against, which is not our situation once we are sampling from noise with no paired real trial.

- **Distributional statistics recurring across the GAN/diffusion line.** The same handful recur in Spike-GAN, CalciumGAN, GNOCCHI and LDNS 
([1803.00338](https://arxiv.org/abs/1803.00338), [2009.02707](https://arxiv.org/abs/2009.02707), [2407.21195](https://arxiv.org/abs/2407.21195), 
[2407.08751](https://arxiv.org/abs/2407.08751)): (a) population spike-count histogram (total spikes per bin, across the population), compared by KL divergence; (b) pairwise 
correlation matrix across all unit pairs, compared by RMSE or Pearson r between real and generated matrices; (c) per-neuron firing rate and inter-spike-interval mean/SD, compared 
by RMSE; (d) autocorrelation (LDNS specifically reports recovering the refractory-period dip at short lags, which needed an added spike-history observation model to reproduce); 
(e) power spectral density, per latent dimension. LDNS reports these against an AutoLFADS baseline on macaque reach data — population-count KL 0.0039 vs. 0.0040, 
pairwise-correlation RMSE 0.0025 vs. 0.0026, mean-ISI RMSE 0.037 vs. 0.039 — i.e. the diffusion prior matches but does not clearly beat a well-tuned inference model on pure 
distributional statistics; its stated advantage is unconditional / variable-length / conditional sampling, not better summary statistics.
- **Dimensionality.** The standard single-number summary of population geometry is the participation ratio (Gao, Trautmann, Yu, Santhanam, Ryu, Shenoy & Ganguli, [preprint 
2017](https://ganguli-gang.stanford.edu/pdf/17.theory.measurement.pdf)) — the squared-sum-over-sum-of-squares of covariance eigenvalues, a smooth, scale-aware "number of 
dimensions that matter," used to contrast task-engaged vs. spontaneous dimensionality across species/areas. No paper found here reports a participation-ratio number for 
*generated* activity next to real activity as an explicit realism check, though the survey lists dimensionality and power-spectrum comparison among general evaluation practices 
([2606.10530](https://arxiv.org/abs/2606.10530)).
- **Which statistics are decorative.** The clearest documented warning is domain-general rather than neuroscience-specific: Bischoff, Darcher, Deistler, Gao, Gerken, Gloeckler et 
al., "A Practical Guide to Sample-based Statistical Distances for Evaluating Generative Models in Science" ([arXiv:2403.12636](https://arxiv.org/abs/2403.12636)), show a 
classifier two-sample test (C2ST) can sit near its "indistinguishable" value of 0.5 — a visual pass — even when the two distributions are clearly different, if the classifier 
is weak or samples are few, and can conversely look falsely bad in high dimensions from many small per-dimension mismatches accumulating; in their drift-diffusion-model case study 
of primate reaction times, C2ST called two models equivalent when sliced-Wasserstein and MMD both showed one model's distribution was visibly too broad. Their recommendation is 
never a single distance: sliced-Wasserstein as a cheap first screen, MMD with a median-heuristic kernel as a second, independent check, because no one distance dominates across 
dimensionality and sample size. Applied to our setting, this is a direct warning against trusting one scalar (the round trip's mean) as the entire plausibility claim.
- **Conditioning-variable decode-back is treated as necessary, not decorative, whenever there is something to condition on.** Both diffusion papers make decoding the conditioning 
variable back out of the generated sample their headline validation (GNOCCHI: ridge-decoded hand position and held-out target location; LDNS: linearly decoded reach 
angle/kinematics from conditionally generated spikes) rather than an afterthought beside the distributional statistics — this is effectively a small-scale round trip; see Q3.

## 3. Cycle-consistency / encoder-in-the-loop evaluation

The term's origin is vision, and it is usually a training loss there, not only an eval: "cycle consistency" as a name comes from CycleGAN (Zhu, Park, Isola & Efros, ICCV 2017, 
[project page](https://junyanz.github.io/CycleGAN/)), a loss term pushing F(G(x)) ≈ x for unpaired image translation. Its use as a *post-hoc* evaluation of a generative model, 
separate from training, is much rarer in what was found here.

- **Closest match: decode the conditioning variable back out, not the full state.** Both GNOCCHI and LDNS ([2407.21195](https://arxiv.org/abs/2407.21195), 
[2407.08751](https://arxiv.org/abs/2407.08751)) do a version of round-trip validation on a *low-dimensional conditioning variable*: generate activity conditioned on a 
target/behavioural label, decode that label back out with a small trained (ridge/linear) readout, and check it matches. LDNS calls this a "closed loop assessment." Structurally 
the same idea as our round trip, scaled down from "the whole 8×721 state" to "one behavioural scalar," using a small trained decoder rather than the same frozen forward simulator 
that produced the training data.
- **A very recent, explicit "cycle consistency" check on a brain-activity foundation model — synthetic data only.** Bracher, Intes & Radev (2026, 
[arXiv:2604.23865](https://arxiv.org/abs/2604.23865)) invert a text-to-brain "foundation model" (TRIBEv2) with simulation-based inference to recover latent stimulus properties 
(valence, arousal, etc.) from brain maps, and explicitly build a cycle-consistency baseline: generate stimuli from a target score, use a second model to re-estimate the score from 
the generated stimulus, and correlate recovered against true scores. Their finding is a caution as much as a validation: SBI posterior estimates were better calibrated than the 
cycle-consistency numbers alone, which showed "noticeable off-diagonal cross-correlations" (dimensions leaking into each other) that a single scalar agreement number would hide. 
All experiments are on synthetic brain responses from the same emulator used for training — the authors state this validates the emulator's own information content, not real 
brain recordings. This is the only paper found that is this recent, uses "cycle consistency" as an explicit named method for a brain-activity model, and is candid that the check 
alone was not fully trusted even by its own authors.
- **A full-state round trip through a real forward/encoding model, on real neural data, was not found.** No paper (i) samples a full population-activity vector from a learned 
prior and (ii) passes it through an independently motivated forward/encoding model to check it lands back near itself, on real (not synthetic-only) data. The nearest bodies of 
work sit on either side of that gap. *Stimulus-side closed loop, through a real animal:* Bashivan, Kar & DiCarlo (*Science* 2019, 
[doi:10.1126/science.aav9436](https://doi.org/10.1126/science.aav9436)) and Walker, Sinz, Cobos, Muhammad, Froudarakis, Fahey et al. ("Inception loops," *Nat. Neurosci.* 2019, 
[doi:10.1038/s41593-019-0517-x](https://doi.org/10.1038/s41593-019-0517-x)) invert a trained encoding model to synthesise an image predicted to drive a target neural population, 
physically display it to the real animal, and compare the actually recorded response to the model's prediction — a true closed loop, but through the animal, validating a 
stimulus, not a sampled internal state. *Identifiability caution:* "System identification of neural systems: if we got it right, would we know?" 
([arXiv:2302.06677](https://arxiv.org/abs/2302.06677)) shows standard model-comparison scores (linear regression, CKA, RSA) between two encoding models can look similarly good 
while one is architecturally wrong — a good-looking match number is not proof a forward model is right, a reason not to over-read a single round-trip mean as validating either 
the prior or the frozen brain used to score it. *Simulation-based inference's standard workflow is the same shape, smaller scale:* posterior-predictive checking — draw 
parameters from the fitted posterior, re-simulate, compare simulated to observed summary statistics — is standard SBI practice (Cranmer, Brehmer & Louppe, *PNAS* 2020, 
[doi:10.1073/pnas.1912789117](https://doi.org/10.1073/pnas.1912789117); demonstrated on mechanistic circuit models by Gonçalves, Lueckmann, Deistler, Nonnenmacher, Öcal, 
Bassetto et al., *eLife* 2020, [doi:10.7554/eLife.56261](https://doi.org/10.7554/eLife.56261)), but applied to recovering a handful of biophysical parameters of a small circuit 
from data, not to validating a learned prior over a high-dimensional population state via an independent forward model.
- **Net read for Q3.** Our round trip — sample a full state from the prior, decode to video, re-encode through the same frozen connectome-constrained simulator, compare the two 
states directly in the simulator's own units — has no named precedent doing exactly this. It sits closest to the SBI posterior-predictive check (compare re-simulated to original 
in the same measurement space) crossed with the LDNS/GNOCCHI decode-back logic (validate a sample using a component external to its own training objective), scaled up from one 
scalar or a handful of parameters to the full state.

## 4. Decoding/reconstruction of stimuli from neural activity: metrics and controls

This overlaps our own `research_notes/Коннектом мухи и план проекта/neural_decoding.md` (2026-09-18); only what is new, or specifically about 
controls/spontaneous decoding, is repeated here.

- **The same three controls recur across fMRI, mouse, primate and retina work, regardless of species or modality.** (1) A **held-out test set** of stimuli never used to fit the 
decoder — the minimum bar in every paper surveyed, from Natural Scenes Dataset fMRI work to Chen, Wang, Jozwik, Golomb, Boly, Postle & Van Essen's mouse-hierarchy decoding 
(*PLOS Comput. Biol.* 2024, [doi:10.1371/journal.pcbi.1012297](https://doi.org/10.1371/journal.pcbi.1012297)). (2) A **shuffle control**, most often a time-shuffle or 
label-shuffle applied to the neural-activity/stimulus pairing before decoding, whose resulting accuracy sets the floor the real decoder must clear. (3) **Identification/retrieval 
accuracy** as an alternative to continuous similarity when reconstructions are imperfect — rank the true stimulus among *n* distractors by decoder-predicted similarity (2-way, 
n-way, top-k), used throughout the fMRI line (MindEye2's image retrieval, NeuroClips' n-way video classification) precisely because a correlation number alone conflates "wrong in 
a small way" with "wrong in a way that changes identity."
- **Nearest-neighbour baselines** are a sanity floor distinct from the shuffle floor: a decoder is credited with real information only if it beats simply returning the nearest 
training-set stimulus by some fixed similarity metric, guarding against a decoder that has implicitly memorised training-set stimulus statistics rather than reading the trial's 
activity.
- **Decoding *spontaneous*, not stimulus-locked, activity is the closest published analogue to a state with "no source clip."** Horikawa, Tamaki, Miyawaki & Kamitani ("Neural 
decoding of visual imagery during sleep," *Science* 340:639, 2013, [doi:10.1126/science.1234330](https://doi.org/10.1126/science.1234330)) decode dream content from fMRI recorded 
seconds before waking sleeping subjects, training the decoder only on stimulus-evoked (awake) activity and testing generalisation to internally generated activity with no stimulus 
present — a decoder trained on one regime, evaluated on a state with no source stimulus, structurally close to our use of a decoder (13B) trained on video-caused states and then 
applied to a sampled state with no source clip. The follow-up (Horikawa & Kamitani, "Generic decoding of seen and imagined objects using hierarchical visual features," *Nat. 
Commun.* 8:15037, 2017, [doi:10.1038/ncomms15037](https://doi.org/10.1038/ncomms15037)) extends this to volitional imagery, decoding DNN-layer feature vectors from fMRI and 
identifying the seen/imagined object by correlating the decoded vector against a large candidate pool of features for object categories never used in training — an n-way 
identification test again, not pixel-level reconstruction.
- **Encoder-inversion reconstruction (the family closest to our own decoder 8-12 work) effectively bakes a round-trip-style number into its training objective.** Bauer, Margrie & 
Clopath ("Movie reconstruction from mouse visual cortex activity," *eLife* 2026, [doi:10.7554/eLife.105081](https://doi.org/10.7554/eLife.105081)) optimise a video by gradient 
descent so a frozen encoding model's *predicted* response matches the *recorded* response — the objective directly minimises a round-trip-style gap, so a low prediction error is 
close to guaranteed by construction; their reported spatio-temporal pixel correlation (r=0.569) is instead a property of the reconstructed video against the *original* stimulus 
video, a genuinely independent number. This is the key structural difference from our setting: their round trip is baked into the optimisation (an inversion), ours is a post-hoc 
check on a prior trained by an unrelated objective (flow matching over states) — so a good round trip in our pipeline is evidence about the prior in a way Bauer et al.'s number 
is not evidence about their inversion.

## 5. Fly visual system specifics: T4/T5 population statistics as a sanity check

- **Single-neuron direction tuning is thoroughly mapped; population-level joint statistics (correlation structure, dimensionality) are not.** Maisak, Weir, Serbe, Mendes, Yamada, 
Silies et al. ("A directional tuning map of Drosophila elementary motion detectors," *Nature* 2013, [doi:10.1038/nature12320](https://doi.org/10.1038/nature12320)) used two-photon 
calcium imaging to establish the now-standard picture: T4 responds to moving ON (brightness-increment) edges and T5 to moving OFF edges, each split into four anatomical subtypes 
tuned to one of the four cardinal directions, tiling the visual field retinotopically. Salazar-Gatzimas, Chen, Creamer, Mano, Mandel, Matulis, Pottackal & Clark ("Direct 
measurement of correlation responses in Drosophila elementary motion detectors reveals fast timescale tuning," *Neuron*, 2016-10-05, 
[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0896627316305761)) imaged T4 responses to ON/OFF edges moving at 30°/s with two-photon calcium imaging, showed 
the correlation-response timescale is fast enough to constrain candidate models' temporal filters, and that T4/T5 activity is necessary for and predictive of the fly's own 
behavioural response to the same correlated stimuli. Arenz, Drews, Richter, Ammer & Borst ("The temporal tuning of the Drosophila motion detectors is determined by the dynamics of 
their input elements," *Curr. Biol.* 2017, [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0960982217300866)) show T4/T5 temporal-frequency tuning shifts toward 
higher frequencies under octopamine-receptor activation, and that the shift is fully explained by faster dynamics in upstream input neurons rather than a change in T4/T5 
themselves — i.e. temporal tuning is inherited, not computed locally. *We could not independently verify exact peak temporal frequencies in Hz from primary text this session — 
secondary summaries gave numbers that looked like they conflated temporal frequency with angular velocity; treat any specific Hz figure as unverified until read from the paper 
directly.*
- **FlyVis's own validation is the most relevant population-scale check, and it is already the one the project cites.** Lappalainen, Shiozaki, Ariel, Behnia, Dubs, Egelhaaf et al. 
("Connectome-constrained networks predict neural activity across the fly visual system," *Nature* 2024, 
[doi:10.1038/s41586-024-07939-3](https://doi.org/10.1038/s41586-024-07939-3)) validate model responses cell-type by cell-type against **26 previously published experimental 
studies** (their Supplementary Note 3) — the same number `AGENTS.md` already names as this project's physiology bar. For T4/T5 specifically the model reproduces the ON/OFF 
selectivity split, the four-cardinal-direction subtype map, and a direction-selectivity index that correlates with task-optimisation performance across their DMN ensemble (r=0.60, 
p=2.6×10⁻⁶); validation spans flash, moving-edge (12 directions × 6 speeds), single-ommatidium-flash receptive-field, and naturalistic-video (Sintel) stimulus classes. The 
paper also states that for many cell types the ensemble predicts "strongly clustered" responses across models — a population-level statement — but does not report a quantified 
correlation matrix or dimensionality number for T4/T5 as a population in the main text (deferred to a supplementary data file we could not fetch this session).
- **No population-level (multi-column, multi-type joint) correlation or dimensionality statistic for T4/T5 was found anywhere in this search**, fly or otherwise — every fly 
source above characterises tuning per cell type/subtype, not the joint statistics of the 8-type × 721-column population vector our model actually samples. This is a genuine, 
specific gap relative to the mammalian literature in Q2 (which does report participation ratio and correlation structure for real V1/motor populations), worth naming explicitly 
rather than assuming FlyVis's single-neuron validation transfers to population-level realism.

## Gaps

- No paper found trains a generative prior over a *rate-based, deterministic, connectome-constrained* population state (our exact setting); every generative-prior paper found fits 
spikes or calcium from real recordings, so the noise/discretisation-driven statistics much of that literature reports (ISI, spike-count histograms) may not have a meaningful 
analogue for us, and no discussion of what the equivalent "necessary" statistic is for a rate-based target was found.
- LDNS's reported metric values and "closed loop assessment" description come from a fetched HTML/text extraction of the paper, not a manual read of the PDF (the PDF itself 
repeatedly failed to parse through the fetch tool this session); the OpenReview discussion thread, which might hold reviewer pushback on the realism metrics, could not be 
retrieved (login wall). Treat the numbers as likely-correct but not independently cross-checked against a second source.
- Bischoff et al.'s statistical-distances guide illustrates its neuroscience example with a drift-diffusion behavioural model (reaction times), not population neural activity 
directly; no paper applying sliced-Wasserstein/MMD/C2ST specifically to generated-vs-real neural *population trajectories* was found.
- Could not verify Arenz et al. 2017's exact peak temporal-frequency numbers for T4/T5 from primary text (Q5); could not access FlyVis's Supplementary Data files 5-7, which 
reportedly hold the full 26-study comparison and any population-level statistics — these are exactly the numbers that would show whether FlyVis's own pretrained ensemble has 
realistic T4/T5 population correlation structure, and are worth pulling directly from the project's own copy of the paper rather than re-requesting by search.
- No insect/Drosophila-specific generative-prior-over-population-activity paper was found at all; every candidate is mammalian (macaque, mouse, human) or, for the pure 
spike-statistics GAN line, salamander retina. "A generative prior over connectome-constrained fly population states" appears to have no direct precedent, consistent with our own 
earlier novelty search (`research_notes/prior_art_2026/novelty.md`).
- Simulation-based inference's posterior-predictive checking (Q3) is a large, mature literature in its own right (the `sbi` toolbox; Gonçalves et al. 2020) that was only skimmed 
for its round-trip-shaped logic; a dedicated look at how SBI practitioners report predictive-check numbers (single scalar vs. distributional) could sharpen what "beside a control" 
should mean for a round-trip metric, but was out of scope for the searches budgeted here.
- Direct arXiv PDF fetches frequently returned corrupted binary content through the fetch tool used this session; where this happened the note fell back to the HTML mirror or 
abstract page, noted inline. A couple of low-confidence claims (e.g., a claimed expansion of the GNOCCHI acronym, from a single low-quality aggregator page) were dropped rather 
than reported.

## What we could adopt

1. **A population distributional-distance panel beside the round-trip mean** — population state-vector histogram (KL), pairwise across-type/column correlation matrix (RMSE or 
Pearson r against clip-driven states), and a power-spectrum comparison, the same trio LDNS/CalciumGAN/Spike-GAN use — computed once between generated states and clip-driven 
states from an existing run; cheap, reuses states already sampled.
2. **Participation ratio of generated vs. clip-driven states** (Gao et al. 2017 formula: squared-sum over sum-of-squares of covariance eigenvalues) as one more scalar beside the 
round trip, to check the prior is not collapsing onto a lower-dimensional subset of the real state manifold — a few lines of code on an existing state tensor, no new run needed.
3. **Sliced-Wasserstein or MMD between the generated-state distribution and the clip-driven-state distribution**, per Bischoff et al.'s explicit recommendation, as a second, 
differently-shaped number beside the round-trip mean — guards against the C2ST-style failure mode where a single aggregate metric looks fine while the distributions visibly 
differ.
4. **n-way identification accuracy as a companion to the round trip's continuous number**: for each generated state, rank its source-adjacent clip (or nearest training clip) 
against k distractors by round-trip distance, reusing the existing round-trip computation — the standard fMRI-decoding control for when a continuous similarity number alone 
could hide "right ballpark, wrong identity."
5. **A cheap linear decode-back check on composed/edited states before the video/brain round trip**, in the LDNS/GNOCCHI style: fit a small linear probe from state to a coarse 
label (e.g. dominant direction per region for the item-14.2 region-composed states) and check the generated state decodes to the intended label — isolates whether the *prior's 
output* carries the intended structure, at a fraction of the cost of the full video-generation-plus-brain round trip.
