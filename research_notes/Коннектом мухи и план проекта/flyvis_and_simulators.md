# FlyVis and connectome-constrained fly simulators (state as of 2026-09-18)

Scope: FlyVis (Lappalainen et al., Nature 2024) in technical depth, plus follow-up / related connectome-constrained models of the Drosophila visual system and whole brain (2024–2026). Compiled 2026-09-18. Every claim carries an inline source; missing items are listed under Gaps.

---

## KQ1. FlyVis: architecture, dynamics, parameters, training task, validation

### Takeaway
FlyVis is a "deep mechanistic network" (DMN): a PyTorch recurrent network of 45,669 point neurons from 64 optic-lobe cell types on a 721-column hexagonal lattice, with connectivity and synapse counts fixed by the FIB-25/FIB-19 FIB-SEM connectomes, synapse sign fixed by neurotransmitter/receptor data, and only 734 free parameters (per-cell-type resting potential, time constant, per-cell-type-pair synapse scaling) learned by BPTT on an optic-flow regression task on Sintel. An ensemble of 50 models was trained; predictions were validated against 26 published studies (ON/OFF selectivity for 32 cell types, T4/T5 direction selectivity).

### Cited Findings
- Paper: Lappalainen, Tschopp, Prakhya, McGill, Nern, Shinomiya, Takemura, Gruntman, Macke, Turaga, "Connectome-constrained networks predict neural activity across the fly visual system", Nature 634, 1132–1140, published online 11 Sept 2024; DOI 10.1038/s41586-024-07939-3 — [PMC11525180](https://pmc.ncbi.nlm.nih.gov/articles/PMC11525180/); [Nature](https://www.nature.com/articles/s41586-024-07939-3); preprint bioRxiv 2023.03.11.532232 (March 2023) — [bioRxiv](https://www.biorxiv.org/content/10.1101/2023.03.11.532232v1.full)
- Scale: 64 cell types, 45,669 neurons, 1,513,231 synapses, 721 hexagonal columns (extent 15, i.e. 31 columns across) modelling the central visual field; each cell type repeated ~once per column, with CT1 split into two compartments (medulla M10 and lobula Lo1) — [bioRxiv v1](https://www.biorxiv.org/content/10.1101/2023.03.11.532232v1.full); [flyvis connectome tutorial](https://turagalab.github.io/flyvis/examples/01_flyvision_connectome/)
- Connectome sources: two Janelia FlyEM FIB-SEM volumes, FIB-25 (7 medulla columns, 702 neurons) and FIB-19 (entire optic lobe, 1,099 neurons), assembled "into a coherent local connectome spanning the retina, lamina, medulla, lobula, and lobula plate"; connectome file in the package is `data/connectome/fib25-fib19_v2.2.json` (class `ConnectomeFromAvgFilters`, config `extent=15, n_syn_fill=1`) — [bioRxiv v1](https://www.biorxiv.org/content/10.1101/2023.03.11.532232v1.full); [flyvis connectome tutorial](https://turagalab.github.io/flyvis/examples/01_flyvision_connectome/)
- Cell types in the network: photoreceptors R1–R8; lamina L1–L5, Lawf1, Lawf2; C2, C3; medulla Mi (Mi1–Mi9 etc.), Tm (Tm1–Tm9 etc.), TmY (TmY3–TmY18), T4a–d, T5a–d, CT1, Am — [flyvis connectome tutorial](https://turagalab.github.io/flyvis/examples/01_flyvision_connectome/); [bioRxiv v1](https://www.biorxiv.org/content/10.1101/2023.03.11.532232v1.full)
- Dynamics: point neurons (single electrical compartment), voltage equation dV_i/dt = (V_rest − V_i)/τ + Σ synaptic input, with a threshold-linear (ReLU) nonlinearity modelling "the time-averaged concentration of synaptic release"; synapses are instantaneous graded release — [bioRxiv v1](https://www.biorxiv.org/content/10.1101/2023.03.11.532232v1.full)
- In code: default dynamics class `PPNeuronIGRSynapses`, ReLU activation (configurable), Euler integration with timestep `dt`; synaptic current = `sign × syn_count × syn_strength`; node params `bias` (resting potential, init Normal(0.5, 0.05)) and `time_const` (default 0.05 s, shared per cell type); edge params `sign`, `syn_count` (lognormal), `syn_strength` (default 0.01, shared per source/target type pair) — [flyvis Network reference](https://turagalab.github.io/flyvis/reference/network/)
- Free parameters: 734 total = 65 resting potentials + 65 membrane time constants (initialised at 50 ms) + 604 unitary synapse scaling factors α; fixed from the connectome: 2,355 synapse-count values and 604 synaptic signs — [bioRxiv v1](https://www.biorxiv.org/content/10.1101/2023.03.11.532232v1.full); [PMC11525180](https://pmc.ncbi.nlm.nih.gov/articles/PMC11525180/)
- Sign rule: histaminergic, GABAergic, glutamatergic → hyperpolarising (−1); cholinergic → depolarising (+1); mostly inferred from cell-type-specific transcriptomics, with "a few cases" adjusted from receptor expression (e.g. R8) — [PMC11525180](https://pmc.ncbi.nlm.nih.gov/articles/PMC11525180/); polarity "from neurotransmitter and receptor profiling" — [flyvis connectome tutorial](https://turagalab.github.io/flyvis/examples/01_flyvision_connectome/)
- Task: optic-flow regression from Sintel; decoder is a "two-layer convolutional decoding network" reading instantaneous activity of D = 34 feature maps (cell types) "from medulla and downstream areas" (the "black box": T-shaped and transmedullary cells), 5×5 hexagonal convolutions, 34→8→3→2 channels, batchnorm/softplus/dropout; docs report `DecoderGAVP` with 34 input channels, 8 intermediate features, 3-channel output (2 flow + confidence) and 7,427 free parameters — [PMC11525180](https://pmc.ncbi.nlm.nih.gov/articles/PMC11525180/); [flyvis optic-flow tutorial](https://turagalab.github.io/flyvis/examples/02_flyvision_optic_flow_task/); [connectome tutorial](https://turagalab.github.io/flyvis/examples/01_flyvision_connectome/)
- Training: BPTT with Adam (β1 0.9, β2 0.999), lr decayed 5e-5 → 5e-6 in ten steps, batch size 4, ~250,000 iterations to converge, L2 loss on flow (docs config uses "epe" end-point-error loss and lr 1e-5), activity regularisation (γ=1, δ=0.01, λV=0.1), 0.5 s fade-in before each training sequence; training sequences of 19 frames = 792 ms at dt=1/50 s; Sintel 23 sequences split vertically into 3 → 69 sequences; augmentations: random temporal crops, flips, rotations, contrast/brightness, gaussian noise — [PMC11525180](https://pmc.ncbi.nlm.nih.gov/articles/PMC11525180/); [flyvis optic-flow tutorial](https://turagalab.github.io/flyvis/examples/02_flyvision_optic_flow_task/)
- Ensemble: 50 task-optimised models with identical connectome; top 10 by task performance analysed in detail — [bioRxiv v1](https://www.biorxiv.org/content/10.1101/2023.03.11.532232v1.full)
- Validation: compared against 26 previously reported studies; median ensemble flash-response index correctly predicts ON/OFF contrast selectivity for all 32 cell types with known selectivity; T4 and T5 direction selectivity (4 subtypes each) correctly predicted — [PMC11525180](https://pmc.ncbi.nlm.nih.gov/articles/PMC11525180/); cell types checked include T4, T5, Mi1–Mi9, Tm1–Tm9, TmY3–TmY18, L1–L5, C, Lawf, CT1, Am — [bioRxiv v1](https://www.biorxiv.org/content/10.1101/2023.03.11.532232v1.full)
- Janelia press release (11 Sept 2024) quotes Turaga: "We now have a computational method for turning measurements of the connectome into predictions of neural activity and brain function" — [Janelia news](https://www.janelia.org/news/researchers-combine-the-power-of-ai-and-the-connectome-to-predict-brain-cell-activity)

### Inferences
- Because synapse strength is one scalar per (source type, target type) pair (604 values) multiplied by per-connection synapse counts, FlyVis has no per-synapse learned weights; the model is far more constrained than a typical RNN and "learning" mostly sets gains and time constants.
- The 250k iterations × batch 4 × 19 frames × 45,669 neurons implies a modest single-GPU workload (the authors never report GPU hours; see Gaps).

### Gaps
- Exact GPU model and wall-clock training time per model are not stated in the Methods I could access (PMC full text, bioRxiv v1, docs). The Europe PMC mirror returned 403 and Nature.com redirects to a cookie gate.
- The exact count of synaptic signs known from data vs inferred is not given.

---

## KQ2. How FlyVis converts video to photoreceptor input; which layers can be read out; code structure, licence, GPU, pretrained ensembles

### Takeaway
A grayscale video is box-filtered onto a hexagonal array of 721 "receptors" (`BoxEye(extent=15, kernel_size=13)`: 13×13-pixel mean around each lattice point, 13 px spacing, ~5.8° per column), temporally resampled to the simulation dt (1/50 s for training, 1/100 s in tutorials), and the same luminance drives R1–R8 in each column. All 64 cell types (every neuron) are simulated and any of them can be read out per cell type via `LayerActivity`; pretrained 50-model ensemble `flow/0000` ships via `flyvis download-pretrained`; MIT licence; PyTorch, Python 3.9–3.12.

### Cited Findings
- Rendering in the paper: Sintel frames at 1,024×436 px; each photoreceptor receives "the greyscale mean value in the 13 × 13-pixel region" around its lattice point; 13-px spacing; hexagonal lattice 31 columns across → 721 receptors; 24 Hz frames resampled to 50 Hz (Δt = 20 ms); R1–R8 all receive the same transduced luminance ("no neuronal superposition modelled") — [PMC11525180](https://pmc.ncbi.nlm.nih.gov/articles/PMC11525180/); [bioRxiv v1](https://www.biorxiv.org/content/10.1101/2023.03.11.532232v1.full)
- `BoxEye` class: params `extent=15` ("Radius, in number of receptors, of the hexagonal array") and `kernel_size=13` ("Photon collection radius, in pixels"); lattice coordinates y = d(u + v/2), x = d·v with d = kernel_size; `__call__(sequence)` takes shape (samples, frames, height, width), `ftype` in {'mean','sum','median'}, and with `hex_sample=True` returns (samples, frames, 1, hexals) — [flyvis rendering reference](https://turagalab.github.io/flyvis/reference/rendering/)
- `HexEye` class (precise eye model for synthetic stimuli): `n_ommatidia=721`, `ppo=25` pixels per ommatidium, ommatidium width/height ≈ 0.101 rad (5.8°), dtype float16; `render_bar`, `render_grating`, `render_bar_movie`; outputs (n_frames, n_ommatidia) — [flyvis rendering reference](https://turagalab.github.io/flyvis/reference/rendering/)
- Temporal resampling utility `resample(stimuli, t_stim, dt)` builds `torch.linspace(0, n_offsets-1, int(t_stim/dt))` indices — [flyvis rendering reference](https://turagalab.github.io/flyvis/reference/rendering/)
- Custom-stimulus tutorial (Moving MNIST 64×64): "31 columns across resulting in 721 columns in total, spaced 13 pixels apart"; `receptors = BoxEye(extent=15, kernel_size=13); rendered = receptors(single_frame)  # [1,1,1,721]`; `dt = 1/100` (10 ms) — [flyvis tutorial 07](https://turagalab.github.io/flyvis/examples/07_flyvision_providing_custom_stimuli/)
- Loading & simulating: `network_view = flyvis.NetworkView(flyvis.results_dir / "flow/0000/000"); network = network_view.init_network()`; `stationary_state = network.fade_in_state(1.0, data.dt, movie_input[[0]])`; `responses = network.simulate(movie_input[None], data.dt, initial_state=stationary_state)`; ensemble: `ensemble = EnsembleView(flyvis.results_dir / "flow/0000"); responses = np.array(list(ensemble.simulate(movie_input[None], data.dt, fade_in=True)))` — [flyvis tutorial 07](https://turagalab.github.io/flyvis/examples/07_flyvision_providing_custom_stimuli/)
- Per-cell-type readout: `responses = LayerActivity(responses, network.connectome, keepref=True); responses["T4c"]  # [n_models, n_sequences, n_frames, 721]; responses.central["T4c"]` (central representative cell); `connectome.central_cells_index` maps cell-type names to representative neuron indices; `connectome.unique_cell_types` lists all types — [flyvis tutorial 07](https://turagalab.github.io/flyvis/examples/07_flyvision_providing_custom_stimuli/); [Network reference](https://turagalab.github.io/flyvis/reference/network/)
- `Network.simulate()` input (batch, n_frames, 1, hexals) → output (batch, n_frames, n_neurons) or `LayerActivity`; default `t_pre=1.0` s grey initialisation; `stimulus_response()` yields (stimulus, response) pairs with `t_pre`, `t_fade_in`; `steady_state()` returns equilibrium under grey — [Network reference](https://turagalab.github.io/flyvis/reference/network/)
- Every neuron of every cell type is simulated, so the readable layers are: R1–R8, L1–L5, Lawf1/2, C2/C3, Mi*, Tm*, TmY*, T4a–d, T5a–d, CT1, Am — [connectome tutorial](https://turagalab.github.io/flyvis/examples/01_flyvision_connectome/)
- Repo: github.com/TuragaLab/flyvis — "A connectome-constrained deep mechanistic network (DMN) model of the fruit fly visual system in PyTorch"; MIT licence; directories `flyvis/`, `flyvis_cli/`, `examples/`, `tests/`, `docs/`; contact janne.lappalainen@uni-tuebingen.de — [GitHub](https://github.com/TuragaLab/flyvis)
- Install: `pip install flyvis` (tested Python 3.9–3.12 on Linux) or `git clone … && pip install -e .`; pretrained models via `flyvis download-pretrained`; data root via `FLYVIS_ROOT_DIR` in `.env`; `flyvis init-config` to start training; CLI `train_single` / `train` (ensemble on a cluster) — [flyvis install](https://turagalab.github.io/flyvis/install/); [optic-flow tutorial](https://turagalab.github.io/flyvis/examples/02_flyvision_optic_flow_task/)
- Releases: v1.1.2 (10 Dec 2023, rename flyvision→flyvis), v1.1.3 (7 Mar 2024), v1.2.0 (6 Aug 2024, ships precomputed response-normalisation constants). No later release listed — [GitHub releases](https://github.com/TuragaLab/flyvis/releases); PyPI current 1.2.0 — [PyPI](https://pypi.org/project/flyvis/)
- Docs: 7 tutorials (connectome, optic-flow training, flash responses, moving-edge responses, ensemble clustering, maximally excitatory stimuli, custom stimuli), all runnable in Colab; API reference for Network, NetworkView, EnsembleView, Connectomes, Task, Decoder, Rendering — [flyvis docs](https://turagalab.github.io/flyvis/)
- A community fork issue reports the connectome build spends 5.4 s in `sleep()` and `LayerActivity` construction ~1.8 s (performance nit) — [KedoKudo/flyvis issue #2](https://github.com/KedoKudo/flyvis/issues/2)

### Inferences
- Because R1–R8 all get identical box-filtered luminance, FlyVis has no colour, no UV/R7-R8 spectral pathway, no photoreceptor adaptation and no optics (point-sampling); real ommatidial acceptance angle (~5°) is approximated only by the 13-px box.
- Field of view ≈ 31 columns × 5.8° ≈ 180° across (inference from stated spacing), i.e. roughly one eye's central field.
- GPU is not strictly required for inference (PyTorch CPU works), but the tutorials' ensemble simulations and 250k-iteration training assume a CUDA GPU; docs never state a VRAM figure.

### Gaps
- No official statement of GPU memory or minimum hardware on the install page or README.
- Pretrained ensemble names beyond `flow/0000` (50 models, `000`–`049`) not enumerated in accessible docs.

---

## KQ3. Limitations of FlyVis and what extending it to the central brain / whole CNS would take

### Takeaway
FlyVis is explicitly reductionist: non-spiking point neurons, instantaneous static graded synapses, no gap junctions, no adaptation/plasticity, no neuromodulation, visual-only with an artificial decoder standing in for the central brain. Extending it requires (i) a whole-brain connectome with neurotransmitter predictions (now available: FlyWire, MaleCNS), (ii) a fitting signal other than a task (whole-brain calcium data) since the "connectome alone" is theoretically insufficient, and (iii) a scalable differentiable simulator; community projects already bolt FlyVis onto MaleCNS LIF brains as a sensory front end.

### Cited Findings
- Authors' limitations (verbatim excerpt): "We have taken a reductionist modelling approach, simplifying the modelling of individual neurons and synapses… Our reductionist model cannot, for example, account for the role played in this circuit by electrical synapses, nonlinear chemical synapses and neuromodulation." — [PMC11525180](https://pmc.ncbi.nlm.nih.gov/articles/PMC11525180/)
- Preprint lists: non-spiking only; excludes electrical synapses, complex synapse dynamics, neuromodulation; static non-adaptive synapses; no central-brain circuits; no activity-dependent plasticity — [bioRxiv v1](https://www.biorxiv.org/content/10.1101/2023.03.11.532232v1.full)
- Theory (Beiran & Litwin-Kumar, Nature Neuroscience, Dec 2025; bioRxiv Feb 2024): "connectome datasets alone are generally not sufficient to predict neural activity", but pairing connectivity with recordings of a subset of neurons yields accurate predictions for unrecorded neurons and can prioritise which neurons to record — [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.02.22.581667v1); [Nature Neuroscience](https://www.nature.com/articles/s41593-025-02080-4)
- Macke (co-author) in Janelia release: "There is a big gap between the static snapshot of the connectome and the dynamics of real-life computation in the living brain" — [Janelia news](https://www.janelia.org/news/researchers-combine-the-power-of-ai-and-the-connectome-to-predict-brain-cell-activity)
- Whole-brain data now available: MaleCNS is "the first finished connectome of an entire male Drosophila central nervous system" (brain + optic lobes + VNC) — [Janelia MaleCNS](https://www.janelia.org/project-team/flyem/male-cns-connectome); the full male optic-lobe connectome with cell-type catalogue: Nern et al., "Connectome-driven neural inventory of a complete visual system", Nature 2025 — [Nature](https://www.nature.com/articles/s41586-025-08746-0)
- Differentiable biophysics at scale exists: Jaxley (Deistler et al., Nature Methods 2025) implements implicit-Euler multicompartment simulation in JAX on CPU/GPU/TPU and trained a network of morphologically detailed neurons with 100,000 parameters on a vision task — [Nature Methods](https://www.nature.com/articles/s41592-025-02895-w); [PMC12695658](https://pmc.ncbi.nlm.nih.gov/articles/PMC12695658/)
- Community whole-CNS integrations that use FlyVis as the eye: Lulzx/fly-brain — 165,122-neuron MaleCNS v1.0 LIF brain in WASM/WebGPU, MuJoCo body, with a flyvis optic-lobe runtime at 50 Hz feeding ~62,000 male-CNS optic-lobe neurons "organized by cell type and retinotopic column" (2×721 rays), MIT — [GitHub](https://github.com/Lulzx/fly-brain); nsfm/fly-afterlife — MaleCNS v1.0 (162,517 neurons) LIF with flyvis graded output converted to spikes by `seam_v2.py` and injected into the brain's own T4/T5 cells on the measured compound-eye geometry, with a `distill.py` step that trains the transplant against flyvis predictions; reported bottleneck: "Direction selectivity from graded front-end inadequate", synapse strengths ~1.5× too strong — [GitHub](https://github.com/nsfm/fly-afterlife); flybench Task 32 "incorporates a rendered visual stimulus processed through the pretrained flyvis optic-lobe model before reaching connectome cells… the first connectome task with an actual sensory front end" — [GitHub](https://github.com/brandoncho369/flybench)

### Inferences
- A "FlyVis-style" central-brain model would need: per-neuron (not per-column) instantiation (no columnar tiling in the central brain, so ~130k–165k explicit neurons); FlyWire/MaleCNS neurotransmitter predictions for signs; and a training objective — either behaviour (as in Cowley's knockout training or the RL locomotion controller below) or whole-brain calcium fitting (as in the BrainTrace/chaobrain project).
- The rate→spike "seam" problem reported by fly-afterlife shows that mixing graded FlyVis output with LIF central brains is non-trivial; a unified differentiable rate or surrogate-gradient spiking model avoids it.

### Gaps
- No official Turaga-lab preprint extending FlyVis to the central brain or to MaleCNS was found in searches through Sept 2026 (queries: "Lappalainen Turaga 2025/2026 whole brain", "flyvis 2", "deep mechanistic network 2026"). Absence of evidence, not evidence of absence.

---

## KQ4. Related and follow-up work 2024–2026 (Shiu LIF, Cowley, other simulators, decoders)

### Takeaway
The other pillar is Shiu et al. (Nature, Oct 2024): a Brian2 LIF model of 127,400 FlyWire v630 neurons / 50M synapses with uniform parameters, which predicts sensorimotor outcomes (>90% of tested predictions) but has no sensory front end and no learning. Since then the ecosystem has grown mostly through (a) faster LIF re-implementations (GPU, Apple MLX, WASM, Loihi 2), (b) MaleCNS-based embodied simulations that use FlyVis as the eye, (c) trainable whole-brain models (RL graph controller; BrainTrace online fitting to calcium imaging), and (d) benchmarks (flybench). Cowley et al. maps DNN units 1-to-1 onto LC neurons via knockout training but does not use the connectome. I found no project that trained a decoder to reconstruct stimuli from a simulated fly network.

### Cited Findings
Shiu et al. LIF whole-brain model
- Shiu et al., "A Drosophila computational brain model reveals sensorimotor processing", Nature 634, 210–219, 2 Oct 2024: 127,400 FlyWire v630 neurons, 50M synapses, Brian2 LIF; V_rest −52 mV, threshold −45 mV, reset −52 mV, membrane τ ≈ 11 ms, refractory 2.2 ms, synaptic delay 1.8 ms, α-synapse τ 5 ms, weight 0.275 mV per synapse (free parameter); GABA/glutamate inhibitory, ACh/DA/OA/5-HT excitatory; 1,000 ms trials, 30 repeats, ~5 min per trial per CPU thread — [PMC11446845](https://pmc.ncbi.nlm.nih.gov/articles/PMC11446845/)
- Validation: activating each candidate cell type at 50 Hz, "11 are predicted to activate MN9; notably, 10 of 11 of these cell types actually do elicit rostrum extension"; overall >90% of experimentally tested predictions — [PMC11446845](https://pmc.ncbi.nlm.nih.gov/articles/PMC11446845/)
- Stated limitations: "does not account for gap junctions, non-spiking neurons, internal state or long-range neuropeptides"; assumes zero basal firing, so "inhibitory connections to an inactive neuron have no effect"; neuromodulated circuits "will be poorly modelled" — [PMC11446845](https://pmc.ncbi.nlm.nih.gov/articles/PMC11446845/)
- Code: philshiu/Drosophila_brain_model (MIT; FlyWire v630 default, v783 alternative; activation = Poisson spiking at fixed rate, silencing = zero synapses; conda `environment.yml`; Brian2 C++ codegen; slower on Colab) — [GitHub](https://github.com/philshiu/Drosophila_brain_model); data at Edmond doi:10.17617/3.CZODIW — [PMC11446845](https://pmc.ncbi.nlm.nih.gov/articles/PMC11446845/)
- Re-implementations: Loihi 2 neuromorphic run of 140k neurons / 50M synapses on 12 chips, "orders of magnitude faster than numerical simulations on conventional hardware" (Wang et al., arXiv 2508.16792, 22 Aug 2025) — [arXiv](https://arxiv.org/abs/2508.16792); Apple-Silicon MLX port (FlyWire v630 and MaleCNS v1.0, 0.29 s per biological second on M4 Pro, checked against Brian2) — [GitHub](https://github.com/Kisame76/drosophila-brain-mlx); Brian2/Brian2CUDA/PyTorch/NEST-GPU backends — [GitHub](https://github.com/dhirajpatra/fly-brain); CUDA "FastFly" and others catalogued in awesome-fly — [GitHub](https://github.com/cobanov/awesome-fly)
- flybench (MIT): 31 pre-registered behavioural tasks on FlyWire v783 and MaleCNS; reference LIF with Shiu constants; finds optimal synaptic gain ≈ 0.45 on v783 vs 1.0 previously; spike-frequency-adaptation variant scores 0.68 vs 0.53 on hard tasks; live at fly-bench.com — [GitHub](https://github.com/brandoncho369/flybench)

Trainable / differentiable whole-brain models
- BrainTrace (Wang, Dong, Ji, Xiao, Jiang, Liu, Huan, Wu, "Model-agnostic linear-memory online learning in spiking neural networks", Nature Communications, 19 Jan 2026): repo chaobrain/fitting_drosophila_whole_brain_spiking_model fits a FlyWire v630/783 whole-brain spiking model to calcium-imaging recordings (figshare 10.6084/m9.figshare.13349282) using eligibility traces and LoRA on synaptic weights (scale 0.000825 mV), with RNN encoder/decoder, GPU — [GitHub](https://github.com/chaobrain/fitting_drosophila_whole_brain_spiking_model); an OpenReview PDF "online fitting connectome-constrained drosophila whole-brain…" describes the same paradigm (FlyWire scaffold, synaptic weights and time constants optimised by online gradient descent against resting-state calcium data) — [OpenReview](https://openreview.net/pdf?id=wCBNxp1qWe) (page blocked by bot check; details from search snippet only)
- Jin, Zhu, Zhang, Sui, "Whole-Brain Connectomic Graph Model Enables Whole-Body Locomotion Control in Fruit Fly" (arXiv 2602.17997, 20 Feb 2026, rev. 14 Jun 2026): instantiates the whole-brain connectome as a graph-structured controller trained by deep RL for a biomechanical fly, reporting better sample efficiency than graph/non-graph baselines — [arXiv](https://arxiv.org/abs/2602.17997)
- Xie et al., "Visual Function Profiles via Multi-Path Aggregation…" (arXiv 2512.06934, 7 Dec 2025): steady-state, adjacency-power "multi-path aggregation" over the whole-brain connectome predicts neuron-level ON/OFF and direction-selective responses (Pearson 0.84±0.12 vs 0.33±0.59 baseline) and is used to replace CNN modules with LC population responses in a navigation simulation — [arXiv](https://arxiv.org/abs/2512.06934)
- Beiran & Litwin-Kumar (Nature Neuroscience, Dec 2025) teacher–student theory of connectome-constrained RNNs (see KQ3) — [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.02.22.581667v1)

Cowley et al. 2024
- Cowley, Calhoun, Rangarajan, Ireland, Turner, Pillow, Murthy, "Mapping model units to visual neurons reveals population code for social behaviour", Nature 629, 1100–1108 (May 2024): DNN with 1-to-1 unit-to-neuron mapping learned by "knockout training" matching genetic silencing of 23 LC visual projection neuron types during male courtship; finds a population code; does NOT use the connectome (framed as enabling "future incorporation of wiring diagrams"); code murthylab/one2one-mapping — [Pillow lab](https://pillowlab.princeton.edu/pubs/abs_Cowley2024knockouttraining.html); [Nature](https://www.nature.com/articles/s41586-024-07451-8)

Other simulators / tools
- Jaxley (Nature Methods 2025) — differentiable biophysical simulator in JAX; no fly-connectome application found — [Nature Methods](https://www.nature.com/articles/s41592-025-02895-w)
- flybody (TuragaLab): anatomical MuJoCo fly body with RL examples; FlyGym/NeuroMechFly (EPFL): sensorimotor biomechanical fly; FlyBrainLab; connectome-interpreter (effective connectivity, differentiable whole-brain models) — [awesome-fly](https://github.com/cobanov/awesome-fly)
- NeurIPS 2024 tutorial/workshop "Simulating the brain and body of the fruit fly" — [NeurIPS](https://neurips.cc/virtual/2024/109313)
- MaleCNS LIF with dopamine-wired olfactory conditioning and habituation (TheMrRaGe/flybrain: MaleCNS v1.0, 162,517 neurons, 6.1M connections, Shiu constants + Tsodyks–Markram depression; code co-written with an AI assistant; unverified items flagged "CHECK") — [GitHub](https://github.com/TheMrRaGe/flybrain)
- State of Brain Emulation Report 2025 (Zanichelli, Schons, Freeman, Shiu, Arkhipov; arXiv 2510.15745, 17 Oct 2025, rev. 5 Nov 2025) surveys the field — [arXiv](https://arxiv.org/abs/2510.15745) (only the abstract page was retrievable)

### Inferences
- The field has bifurcated: the Turaga line (graded, task-trained, columnar optic lobe) and the Shiu line (spiking, untrained, whole brain); 2026 hobbyist/embodied projects splice the two, and the trainable whole-brain work (BrainTrace, RL graph controller) comes from groups outside Janelia.
- Many 2026 GitHub projects (fly-afterlife, TheMrRaGe/flybrain, flybench) are single-author, partly AI-assisted efforts without peer review; treat their numbers as provisional.

### Gaps
- No project found that trains a decoder to reconstruct the visual stimulus from simulated fly-network activity (searched "decoder reconstruct visual stimulus flyvis"); the only decoders are FlyVis's optic-flow head and BrainTrace's calcium encoder/decoder.
- bioRxiv 2026.02.02.700492 "Connectome analysis reveals brainwide visual processing in Drosophila" could not be fetched (HTTP 429 ×3); unknown whether it includes a simulation.
- State of Brain Emulation Report content beyond the abstract not retrieved.
- FutureJJ/ommatid (FlyVis + male CNS driving a hexapod robot) returned 404 at fetch time.

---

## KQ5. State of the art in differentiable whole-brain fly models (Sept 2026) and compute cost

### Takeaway
As of September 2026 there is no peer-reviewed differentiable, activity-validated whole-brain fly model comparable to FlyVis; the closest are BrainTrace (Nat. Commun. Jan 2026, whole-brain SNN fitted online to calcium data), the RL-trained connectomic graph controller (arXiv Feb/Jun 2026), and the graded FlyVis optic lobe stitched to MaleCNS LIF brains in community code. Compute reported: FlyVis ~250k Adam iterations at batch 4 per model × 50 models (GPU type unreported); Shiu LIF ~5 CPU-minutes per 1 s of brain time; Loihi 2 and MLX ports run near or faster than real time.

### Cited Findings
- FlyVis training: ~250,000 iterations, batch 4, lr 5e-5→5e-6, 50-model ensemble; no GPU type or hours given — [PMC11525180](https://pmc.ncbi.nlm.nih.gov/articles/PMC11525180/)
- Shiu LIF: ~5 minutes per 1,000 ms trial per CPU thread — [PMC11446845](https://pmc.ncbi.nlm.nih.gov/articles/PMC11446845/)
- Loihi 2: 140k neurons/50M synapses on 12 chips, orders of magnitude faster than conventional simulation, advantage growing with sparsity — [arXiv](https://arxiv.org/abs/2508.16792)
- MLX port: 0.29 s per biological second on an M4 Pro — [GitHub](https://github.com/Kisame76/drosophila-brain-mlx)
- Browser WASM/WebGPU whole-CNS: 165,122 neurons, 104M synapses, ~30 MB viewer / 23 MB arena payload, desktop Chrome/Edge/Firefox — [GitHub](https://github.com/Lulzx/fly-brain)
- BrainTrace: linear-memory online learning for SNNs applied to the FlyWire whole brain with GPU `--devices` option — [GitHub](https://github.com/chaobrain/fitting_drosophila_whole_brain_spiking_model)
- Theory constraint: connectome alone insufficient; recordings needed — [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.02.22.581667v1)

### Inferences
- A practical 2026 pipeline for a stimulus-to-central-brain simulation is: FlyVis pretrained ensemble → per-column rate output → MaleCNS/FlyWire spiking or rate model, as already done by fly-brain/fly-afterlife/flybench; the unresolved engineering issue is rate-to-spike calibration and direction-selectivity transfer.
- For gradient-based fitting of a whole brain, surrogate-gradient SNNs (BrainTrace) or graded rate models (FlyVis-style) are the two viable routes; Jaxley-style biophysics at 140k neurons is not yet demonstrated.

### Gaps
- No source gives GPU-hours for FlyVis; no source gives GPU-hours for BrainTrace or the RL graph controller.
