# MaleCNS connectome (Janelia FlyEM male Drosophila CNS) — practical research notes

State as of 2026-09-18. Every claim carries an inline source; items I could not verify are in "Gaps". Note: Nature and Cell full texts (Nern 2025, Berg 2026 Cell, BANC Nature 2026, "organization of visual pathways" Cell 2026) were paywalled / rate-limited during this session, so their numbers are taken from preprints, project sites, GitHub READMEs and search snippets and are flagged where second-hand.

## Key Question 0 — Release history, papers, scale, annotations (background)

### Takeaway
MaleCNS is the first complete connectome of a male adult Drosophila CNS (brain incl. both optic lobes + VNC with intact neck connective): v0.9 released 2025-10-03/05, v1.0 released 2026-06-08, paper in Cell 2026-09-03 (preprint bioRxiv 2025-10-09, v2 2025-10-30). Scale: 166,691 neurons, ~46 M presynapses, ~312 M PSDs, 25.6 M connections, 11,691 cell types (11,751 in the v1.0 neuPrint explorer), CC-BY 4.0.

### Cited Findings
- Release timeline: v0.9 released October 3, 2025; v1.0 released June 8, 2026; paper published September 3, 2026 (Cell); bioRxiv preprint v2 posted October 30, 2025 — [male-cns.janelia.org](https://male-cns.janelia.org/); [Janelia project page](https://www.janelia.org/project-team/flyem/male-cns-connectome)
- Release notes give the v0.9 date as October 5, 2025 (site landing page says October 3) and describe v1.0 as "minor proofreading changes" plus "refinement of neuron annotations" relative to v0.9 — [Release Notes](https://male-cns.janelia.org/release/)
- Preprint: Berg S. et al., "Sexual dimorphism in the complete connectome of the Drosophila male central nervous system", bioRxiv 2025.10.09.680999 (first author Stuart Berg; corresponding: Gregory Jefferis, Gerald Rubin, Stuart Berg) — [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.10.09.680999v2.full); PMC copy [PMC12636603](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12636603/)
- Journal version: "Sexual dimorphism in the complete Drosophila male central nervous system connectome", Cell 2026 (ScienceDirect PII S0092867426009426; full text returned HTTP 403 in this session) — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0092867426009426)
- Scale (preprint): 166,691 neurons (incl. sensory axons); 46 million presynapses; 312 million postsynaptic densities; connectome graph of 25.6 million connections between 166,391 neurons; 11,691 cell types; 98.9% of 141,780 detected nuclei proofread — [bioRxiv v2 full](https://www.biorxiv.org/content/10.1101/2025.10.09.680999v2.full)
- Imaging: 0.082 mm³ at 8×8×8 nm isotropic (160 teravoxels), 7 enhanced FIB-SEM systems, 13 months of imaging; synapse-detection precision/recall 0.82/0.81; ~44 person-years of proofreading — [bioRxiv v2 full](https://www.biorxiv.org/content/10.1101/2025.10.09.680999v2.full)
- Completeness: presynaptic completion 94% in neuropils, postsynaptic completion 42%; 40.1% of connections have both partners proofread; 97.5% of neurons matched to FAFB/FlyWire, hemibrain or MANC; ~4% of FlyWire cell types revised — [bioRxiv v2 full](https://www.biorxiv.org/content/10.1101/2025.10.09.680999v2.full)
- Dimorphism: of 7,319 cross-matched central-brain cell types, 114 dimorphic, 262 male-specific, 69 female-specific (4.8% of neurons in males, 2.4% in females); 7,205 isomorphic types; fruitless/doublesex expression annotated — [bioRxiv v1 abstract](https://www.biorxiv.org/content/10.1101/2025.10.09.680999v1); [Janelia project page](https://www.janelia.org/project-team/flyem/male-cns-connectome)
- neuPrint explorer stats for male-cns:v1.0: 11,751 cell types, 164,838 neurons, 171,901,440 synapses; catalog generated 2026-09-09; last dataset update 2026-06-08 — [Male CNS Cell Type Explorer](https://reiserlab.github.io/celltype-explorer-drosophila-male-cns/)
- Collaboration: FlyEM (HHMI Janelia), University of Cambridge Dept. of Zoology (Drosophila Connectomics Group), MRC LMB, Google Research; Google did segmentation, Janelia+Cambridge proofreading/labelling — [Janelia news 2025-10-06](https://www.janelia.org/news/researchers-reveal-connectome-of-the-male-fruit-fly-central-nervous-system)
- Annotations available: cell type, side, neurotransmitter predictions (7 classes: ACh, dopamine, GABA, glutamate, histamine, octopamine, serotonin, via 3D-VGG "synister" classifier trained with gunpowder on synapse crops), hemilineage (`itoleeHl` field, Ito/Lee 2013 nomenclature), cross-dataset types (`flywireType`, `mancType`), motor-neuron exit nerves/muscle targets, sensory modalities — [funkelab/synister_malecns](https://github.com/funkelab/synister_malecns); [natverse malecns docs](https://natverse.org/malecns/); [bioRxiv v2 full](https://www.biorxiv.org/content/10.1101/2025.10.09.680999v2.full)
- Licence: CC-BY 4.0 — [Download page](https://male-cns.janelia.org/download/)

### Inferences
- 166,691 vs 164,838 neurons: the explorer likely counts only bodies with a type/"Neuron" status; the 1,853 difference is plausibly untyped fragments/sensory axons. Similarly 171.9 M "synapses" in the explorer vs 46 M pre / 312 M post in the paper suggests the explorer counts connections' synapse-weights with minconf 0.5, not raw PSDs — treat all such totals as threshold-dependent.
- v0.9→v1.0 changed annotations more than segmentation, so analyses on v0.9 bodyIds mostly carry over, but re-fetch annotations.

### Gaps
- Exact Cell 2026 volume/issue/pages and any number changes vs preprint (403 on ScienceDirect).
- Neurotransmitter prediction accuracy for MaleCNS specifically (synister README gives no metric).

## Key Question 1 — Data access: neuPrint, neuprint-python, Neuroglancer, bulk downloads

### Takeaway
Public dataset name is `male-cns:v1.0` on `https://neuprint.janelia.org` (token required, free account); everything is also downloadable as Feather files from `gs://flyem-male-cns/v1.0/` (1.1 GB weights table, 12.7 GB synapse points, 6.8 GB partner table, 13 MB annotations), plus N5/precomputed image volumes, SWC/precomputed skeletons and the full neo4j dump.

### Cited Findings
- neuPrint URL with dataset preset: https://neuprint.janelia.org/?dataset=male-cns%3Av1.0&qt=findneurons; Neuroglancer state: https://neuroglancer-demo.appspot.com/#!gs://flyem-male-cns/v1.0/male-cns-v1.0.json (layers: em-clahe, cns-seg, neuropil outlines, presyn/postsyn); FlyWire v783 and hemibrain v1.2.1 meshes transformed into MaleCNS space are also provided — [Explore page](https://male-cns.janelia.org/explore/)
- Other UIs: Clio (annotation-focused), NeuronBridge LM matches (since 2025-11-07), VVDViewer, Dimorphism Explorer, Cell Type Explorer — [male-cns.janelia.org](https://male-cns.janelia.org/); [Janelia project page](https://www.janelia.org/project-team/flyem/male-cns-connectome)
- male-cns:v1.0 is routed through the public neuprint and DVID servers; no Clio authentication needed — [natverse/malecns GitHub](https://github.com/natverse/malecns)
- Programmatic: `pip install neuprint-python`; R: `devtools::install_github("natverse/neuprintr")`; both need an account at neuprint.janelia.org and an API token — [Download page](https://male-cns.janelia.org/download/)
- neuprint-python: `Client(server, dataset, token)`; `fetch_neurons(NeuronCriteria)`, `fetch_adjacencies(sources, targets, ...)` returns (neurons_df, roi_conn_df); `fetch_traced_adjacencies()` fetches all Traced non-cropped neurons and "takes several minutes" on hemibrain-scale data; for multi-source/target whole-brain work the docs recommend downloading the entire graph and working locally with networkx/igraph — [neuprint-python Query Tutorial](https://connectome-neuprint.github.io/neuprint-python/docs/notebooks/QueryTutorial.html); [neuprint-python connectivity.py](https://github.com/connectome-neuprint/neuprint-python/blob/master/neuprint/queries/connectivity.py)
- R package `malecns` (v0.4.2.9000, experimental, maintainer G. Jefferis; Zenodo record for 0.4.2): thin wrapper over `malevnc`; `mcns_neuprint_meta()`, `mcns_body_annotations()`, `read_mcns_meshes()`, `choose_mcns_dataset()`; uses `NEUPRINT_TOKEN` env var; Clio write access via Google OAuth for collaborators only — [natverse.org/malecns](https://natverse.org/malecns/); [Zenodo 21993186](https://zenodo.org/records/21993186)
- Bulk downloads (Google Cloud Storage, Apache Arrow Feather; read with pyarrow/pandas or R arrow):
  - `body-annotations-male-cns-v1.0-minconf-0.5.feather` 13 MB (types, sides, classes; no NT)
  - `body-neurotransmitters-male-cns-v1.0.feather` 42 MB (per-neuron aggregated NT)
  - `body-stats-male-cns-v1.0-minconf-0.5.feather` 780 MB (synapse counts per segment)
  - `connectome-weights-male-cns-v1.0-minconf-0.5.feather` 1.1 GB (segment→segment weights; full graph)
  - `syn-points-male-cns-v1.0-minconf-0.5.feather` 12.7 GB (pre/post locations, bodyIds, ROIs, 8 nm voxel units)
  - `syn-partners-male-cns-v1.0-minconf-0.5.feather` 6.8 GB (pre–post pairs with confidence + neuropil)
  - `tbar-neurotransmitters-male-cns-v1.0.feather` 2.7 GB (per-presynapse NT probabilities)
  — [Download page](https://male-cns.janelia.org/download/)
- Image volumes: `gs://flyem_cns_z0720_07m_dvidcoords_n5` (N5, 8 nm uint8 raw EM); `gs://flyem-male-cns/em/em-clahe-jpeg` (precomputed CLAHE JPEG); `gs://flyem-male-cns/v1.0/segmentation` (precomputed uint64, 8 nm); nuclei seg at 16 nm; ROIs: `rois/fullbrain-roi-v4` (256 nm, 96 brain compartments) and `rois/malecns-vnc-neuropil-roi-v0` (27 VNC compartments); skeletons as SWC (8 nm) and precomputed (1 nm), mirrored, and in JRC2018U template (µm) under `gs://flyem-male-cns/v1.0/segmentation/skeletons-*`; full neo4j 4.4.16 DB + CSVs in `gs://flyem-male-cns/v1.0/database/` — [Download page](https://male-cns.janelia.org/download/)
- Example of file-based (no neuPrint) loading: the nfly project downloads only annotations (14 MB) + NT (43 MB) + connectome-weights (1.1 GB), ~1.2 GB total — [zhengxuyu/nfly](https://github.com/zhengxuyu/nfly)
- Ground-truth EM crops for ML are packaged in `torch_em.data.datasets.electron_microscopy.malecns` — [torch-em docs](https://constantinpape.github.io/torch-em/torch_em/data/datasets/electron_microscopy/malecns.html)

### Inferences
- "minconf-0.5" in the file names indicates that the released weight tables and neuPrint synapse counts already apply a 0.5 synapse-detection confidence cut; the raw per-partner confidence is only in `syn-partners`.
- For the visual-system subgraph, the 1.1 GB weights file + 13 MB annotations is the practical path; neuPrint `fetch_adjacencies` with ROI criteria works for smaller cuts but the whole optic lobe (~48k neurons/side) will be slow through Cypher.

### Gaps
- Exact schema (column names) of the Feather files and the neuPrint `Neuron` node property list for male-cns (e.g., property names for hemilineage, NT, column) — not visible without a token; verify with `fetch_neurons(NeuronCriteria(bodyId=...))` or `Client.fetch_custom("MATCH (n:Neuron) RETURN keys(n) LIMIT 1")`.

## Key Question 2 — Optic lobe representation: completeness, photoreceptors, columns/hex coordinates

### Takeaway
Both optic lobes are included and proofread; the right optic lobe is the same tissue as the earlier `optic-lobe:v1.0/v1.1` dataset of Nern et al. 2025. MaleCNS has ~880 columns per eye in axial hex coordinates (hex1 1–36, hex2 1–39), but only ~15 of ~282 optic-lobe cell types carry explicit column tags in the release; other neurons' columns must be inferred from partners.

### Cited Findings
- MaleCNS covers "central brain, optic lobes and the ventral nerve cord" with an intact neck connective — [Janelia project page](https://www.janelia.org/project-team/flyem/male-cns-connectome)
- Both optic lobes included; the work "builds on" the previously reported right optic lobe connectome — [bioRxiv v2 full](https://www.biorxiv.org/content/10.1101/2025.10.09.680999v2.full)
- Nern et al., "Connectome-driven neural inventory of a complete visual system", Nature 641:1225–1237 (May 2025); the optic-lobe connectome is available in neuPrint as `optic-lobe:v1.0`; fewer than 100 cell types account for most cells/connections; the Cell Type Explorer supplement shows retinotopy and connectivity per type — [Nature article](https://www.nature.com/articles/s41586-025-08746-0) (paywalled here); [bioRxiv 2024.04.16.589741](https://www.biorxiv.org/content/10.1101/2024.04.16.589741v1); [PubMed 38659887](https://pubmed.ncbi.nlm.nih.gov/38659887/)
- The visual-system Cell Type Explorer is built on `optic-lobe:v1.1` (pages updated 2025-06-24; last DB edit 2024-09-11) and groups types as ONIN, ONCN, VPN, VCN, other — [Visual system Cell Type Explorer](https://reiserlab.github.io/male-drosophila-visual-system-connectome/index.html); code uses navis, neuprint-python, cloud-volume, snakemake, Blender>3.6 — [reiserlab code repo](https://github.com/reiserlab/male-drosophila-visual-system-connectome-code)
- Column tags in MaleCNS v1.0: "the release annotates retinotopic columns for only 15 of the 282 optic-lobe cell types"; right optic lobe = 48,523 neurons in 274 types; a third-party pipeline infers the other 259 types' columns by synapse-weighted median over placed partners, validated by hold-out on the 15 annotated types (99.89% exact column, 100% within one column) — [CAOS_RES_Conectoma PR #3](https://github.com/fsantibanezleal/CAOS_RES_Conectoma/pull/3)
- Hex lattice geometry: "~880 columns per eye in axial coordinates (hex1 1–36, hex2 1–39)"; T4/T5 carry no column tags, so home column was taken from the strongest hex-tagged input (Mi1/Mi4/Mi9 for T4, Tm1/Tm2/Tm9 for T5): 13,577/13,580 placed; 70,165 optic-lobe cells placed via self/input/output; only 83% of T4/T5 fall inside FlyVis's 721-hex lattice (~107 rim columns/eye get no drive) — [fly-afterlife docs/SEAM.md](https://raw.githubusercontent.com/nsfm/fly-afterlife/main/docs/SEAM.md)
- The Male CNS Cell Type Explorer lets you "browse cell types, connectivity, and eyemaps for the full male CNS" — [Explore page](https://male-cns.janelia.org/explore/); page generator is reiserlab/neuView — [neuView](https://github.com/reiserlab/neuView)
- FlyVis R7/R8 are "named differently" in MaleCNS (i.e., photoreceptor types exist under other names); the FlyVis types absent from MaleCNS are Am, Mi3, Mi11, Mi12, Tm28, TmY9 — [fly-afterlife SEAM.md](https://raw.githubusercontent.com/nsfm/fly-afterlife/main/docs/SEAM.md)
- Cross-dataset optic-lobe type matching table (`olmatching.tsv`) linking Nern et al. types ↔ optic-lobe:v1.0 ↔ FlyWire (Matsliah et al. optic-lobe intrinsic types, Schlegel et al. v2.0 types) — [flyconnectome/ol_annotations](https://github.com/flyconnectome/ol_annotations)
- Follow-up analysis of visual pathways on the male dataset: "The organization of visual pathways in the Drosophila brain", bioRxiv 10.64898/2025.12.22.696097 (v2), Cell 2026 (S0092-8674(26)00941-4) — [bioRxiv v2](https://www.biorxiv.org/content/10.64898/2025.12.22.696097v2.full) (429 rate-limited here); [Cell](https://www.cell.com/cell/fulltext/S0092-8674(26)00941-4) (403 here)

### Inferences
- The hex1/hex2 ranges (1–36, 1–39) and "~880 columns" match a full compound eye (~750–800 ommatidia plus lamina/medulla edge columns), implying MaleCNS retinotopic coverage is complete on both sides, unlike BANC (no lamina).
- Photoreceptors: because FlyVis's R7/R8 are reported as present under different names, and lamina/medulla photoreceptor axons are inside the volume, R1–R8 axons are typed; but the retina itself (cell bodies) is outside a CNS-only volume, so photoreceptor "columns" must come from lamina cartridge / medulla column tags rather than ommatidial positions. (Unverified — see gaps.)
- Which 15 types carry columns is not listed in my sources; from the SEAM notes, Mi1/Mi4/Mi9/Tm1/Tm2/Tm9 are among the hex-tagged input types, and L-cells/Mi1 are the usual anchors in Nern et al.

### Gaps
- Exact list of the 15 column-annotated types and the neuPrint property names for hex coordinates (candidates seen in the wild: `hex1`/`hex2`, `assignedOlHex1`/`assignedOlHex2`); a Cypher `keys(n)` query on an Mi1 neuron will settle it.
- Whether R1–R6 vs R7/R8 photoreceptor axons are individually typed in male-cns:v1.0 and whether their column IDs are populated.
- Nern et al. 2025 headline counts (total optic-lobe neurons, 700+ types, number of new types) — could not fetch Nature/PubMed full text; the PR-derived "282 optic-lobe cell types" and "48,523 neurons/right OL" are third-party counts on MaleCNS v1.0, not the Nern paper.
- Left vs right optic lobe: whether the left OL has the same column annotations as the right (which inherited optic-lobe:v1.x work).

## Key Question 3 — How to programmatically fetch the visual-system subgraph with type labels and column coordinates

### Takeaway
Two working routes: (a) neuPrint `male-cns:v1.0` via neuprint-python with `NeuronCriteria(rois=[optic-lobe ROIs])` + `fetch_adjacencies`, or (b) the Feather bulk files (annotations + connectome-weights) filtered by type/ROI locally; column coordinates must then be taken from the annotated subset and propagated to untagged types via partner-weighted inference, as done by fly-afterlife and CAOS_RES_Conectoma.

### Cited Findings
- neuprint-python `fetch_adjacencies` accepts NeuronCriteria for sources/targets and filters by connection strength or ROI; returns neuron properties and per-ROI weights; whole-graph downloads are recommended for many-to-many queries — [Query Tutorial](https://connectome-neuprint.github.io/neuprint-python/docs/notebooks/QueryTutorial.html); [connectivity.py](https://github.com/connectome-neuprint/neuprint-python/blob/master/neuprint/queries/connectivity.py)
- Bulk alternative: `body-annotations-...feather` (13 MB, types/sides) + `connectome-weights-...feather` (1.1 GB, all segment pairs) — [Download page](https://male-cns.janelia.org/download/); nfly demonstrates loading exactly these files without neuPrint — [nfly](https://github.com/zhengxuyu/nfly)
- Column propagation recipe (third-party, validated): assign each untagged neuron the synapse-weighted median column of its tagged partners; hold-out on 15 tagged types recovers 99.89% exact; caveat: only lamina/medulla columnar types were validated, wide-field lobula types unvalidated — [CAOS_RES_Conectoma PR #3](https://github.com/fsantibanezleal/CAOS_RES_Conectoma/pull/3)
- fly-afterlife's optic-lobe subset (`ol_graph.npz`): 64,373 cells of FlyVis-typed classes, 2.49 M raw edges; whole MaleCNS at weight ≥5: 6.1 M edges over 162,517 neurons; queries used fields type, side (L/R), bodyId — [SEAM.md](https://raw.githubusercontent.com/nsfm/fly-afterlife/main/docs/SEAM.md)
- Type-level average-filter export of the right OL: 151,856,684 connection rows processed in 38 s; 6,140 type→type edges, 107,232 filter entries — [CAOS_RES_Conectoma PR #3](https://github.com/fsantibanezleal/CAOS_RES_Conectoma/pull/3)
- Visual projection neuron / pathway analysis for downstream targets: "The organization of visual pathways in the Drosophila brain" (Cell 2026 / bioRxiv 2025.12.22.696097) — [bioRxiv](https://www.biorxiv.org/content/10.64898/2025.12.22.696097v2.full)

### Inferences
- Minimal Python sketch (property names to be verified against the live schema):
  1. `c = Client('neuprint.janelia.org', dataset='male-cns:v1.0', token=...)`
  2. `neurons, _ = fetch_neurons(NeuronCriteria(rois=['ME(R)','LO(R)','LOP(R)','LA(R)','AME(R)'] ...))` (ROI names follow hemibrain/optic-lobe conventions — verify with `fetch_roi_hierarchy`).
  3. Add VPN targets: neurons whose type is in the VPN/VCN classes of the Cell Type Explorer, or any neuron with ≥N synapses from OL neurons via `fetch_adjacencies(sources=OL_ids, targets=None, min_weight=5)`.
  4. Pull columns from the ~15 tagged types, propagate by weighted median of partners.
  Alternatively read `connectome-weights` Feather with pyarrow, join to `body-annotations` on bodyId, filter by side/ROI.

### Gaps
- No official Janelia notebook for "optic lobe + VPN subgraph with columns" was found; the recipes above are community-derived.
- ROI naming in male-cns:v1.0 (e.g., whether "ME(R)" style or new names) unverified.

## Key Question 4 — Mapping to FlyVis / Lappalainen 2024 (64/65 cell types, 721-hex lattice)

### Takeaway
No official mapping is published, but the open-source fly-afterlife project provides a worked seam: 55 of FlyVis's 65 types exist in MaleCNS (R7/R8 under other names; Am, Mi3, Mi11, Mi12, Tm28, TmY9 missing), MaleCNS's ~880-column axial hex grid is aligned to FlyVis's 721-hex lattice (|u|,|v|,|u+v| ≤ 15) covering ~83% of T4/T5; MaleCNS median synapse counts per connection are 1.48× FlyVis's.

### Cited Findings
- FlyVis = official implementation of Lappalainen et al., Nature 2024; model of ~45,000 neurons over 721 columns, built by tiling a small reconstructed region and using two EM datasets + transcriptomics for synapse signs — [TuragaLab/flyvis](https://github.com/TuragaLab/flyvis); [bioRxiv 2023.03.11.532232](https://www.biorxiv.org/content/10.1101/2023.03.11.532232v1)
- Mapping: 55/65 FlyVis types present in MaleCNS; missing Am, Mi3, Mi11, Mi12, Tm28, TmY9; R7/R8 differently named; home column of T4/T5 from strongest hex-tagged input; 13,577/13,580 placed; 83% of T4/T5 inside FlyVis lattice; median synapse count per connection 1.48× FlyVis (IQR 0.95–2.45); weight threshold ≥5 — [fly-afterlife SEAM.md](https://raw.githubusercontent.com/nsfm/fly-afterlife/main/docs/SEAM.md)
- fly-afterlife: "a real optic lobe seamed onto its own T4/T5 cells on the compound eye's true geometry"; FlyVis drives matching MaleCNS neurons "by cell type and eye position", the spiking brain does the rest; eye geometry derived from connectome anatomy (`seam/eye_geom.py`); FlyVis loads in 20 s, 200 frames in 2.4 s, 794 MB VRAM — [nsfm/fly-afterlife](https://github.com/nsfm/fly-afterlife); [SEAM.md](https://raw.githubusercontent.com/nsfm/fly-afterlife/main/docs/SEAM.md)
- Comparison of the MaleCNS-derived type filters vs the "Nature 2024 reference model": 59/65 types matched, 75.8% of comparable connections recovered, 97.1% sign agreement, rank correlation 0.79 — [CAOS_RES_Conectoma PR #3](https://github.com/fsantibanezleal/CAOS_RES_Conectoma/pull/3)
- Type-name bridge between MaleCNS optic lobe (Nern nomenclature) and FlyWire/FAFB names — [flyconnectome/ol_annotations](https://github.com/flyconnectome/ol_annotations)

### Inferences
- Because MaleCNS uses Nern-et-al. nomenclature and FlyVis uses hemibrain/FIB-25-era names, the discrepancies (e.g., Am, Mi3 split, TmY9 renamed) are mostly nomenclature, not absent cells; `olmatching.tsv` plus Nern's supplementary tables should resolve most of the 10 unmatched types.
- A FlyVis-style model built directly on MaleCNS (both eyes, ~880 columns, real per-cell connectivity) is feasible; no peer-reviewed version exists yet (see Q5).

### Gaps
- No Turaga-lab statement about porting FlyVis to MaleCNS was found.
- The two third-party validations disagree slightly on type match counts (55/65 vs 59/65) — different matching rules; neither is peer-reviewed.

## Key Question 5 — Published simulations/models built on MaleCNS

### Takeaway
As of Sept 2026 there are no peer-reviewed simulation papers on MaleCNS found; there is a large hobbyist/open-source ecosystem (LIF whole-CNS simulators on a single consumer GPU, RL agents, games) curated in awesome-fly, plus arXiv analyses (State of Brain Emulation 2025; multi-path visual function profiles) that cite it.

### Cited Findings
- annel0/flybrain: GPU LIF simulator of MaleCNS v1.0 (166,700 neurons), RTX 3060 12 GB, 2.4× realtime single fly, 8.9× realtime batch-4 fp16; validated median firing-rate ratio 1.01 vs Shiu et al. 2024; notes 57% of network never spikes (optic lobe graded), <1% of types have electrophysiology; not peer-reviewed — [annel0/flybrain](https://github.com/annel0/flybrain)
- zhengxuyu/nfly: MaleCNS v1.0 as a ConnectomeRNN for Gymnasium; 8.4 M-edge visual sub-network, 138,743 neurons engaged, Dale's-law signs from NT predictions, learnable per-edge gains; CPU-viable, ~7× GPU speedup; CartPole 500/500 with supervised head, Pong +8 frozen — [nfly](https://github.com/zhengxuyu/nfly)
- nsfm/fly-afterlife: MaleCNS v1.0 spiking brain (162,517 neurons) + FlyVis optic lobe, Numba LIF, GPU-optional — [fly-afterlife](https://github.com/nsfm/fly-afterlife)
- Curated list of ~30 MaleCNS projects (Doomfly/ViZDoom, Fly64, Minecraft mods, CARLA driving, AxonWeave sparse PyTorch substrate, Apple-MPS simulator, etc.) — [cobanov/awesome-fly](https://github.com/cobanov/awesome-fly)
- MaleCNS wired to an F1 car sim with ES-trained interface — [hotocoo/malecns](https://github.com/hotocoo/malecns)
- Community loader stats: 6.29 M connections at ≥5 synapses, 25.9 M at min-weight 1, "125 million total synapses" (their count) — [FlyBrainMCExp / fly-brain-minecraft](https://github.com/blendi-remade/fly-brain-minecraft/blob/main/README.md)
- MaleCNS cited in "State of Brain Emulation Report 2025" (arXiv 2510.15745) and in "Visual Function Profiles via Multi-Path Aggregation" (arXiv 2512.06934) — [arXiv 2510.15745](https://arxiv.org/pdf/2510.15745); [arXiv 2512.06934](https://arxiv.org/pdf/2512.06934)
- Related peer-reviewed comparative work using MaleCNS annotations: descending/ascending neuron comparative connectomics (Nature 2025) and VNC sex-difference alignment (2026) — [Nature s41586-025-08925-z](https://www.nature.com/articles/s41586-025-08925-z); [PMC13307954](https://pmc.ncbi.nlm.nih.gov/articles/PMC13307954/)

### Inferences
- The Shiu et al. 2024 LIF recipe (used for FlyWire) transfers directly; several repos reproduce its firing statistics on MaleCNS, so a MaleCNS LIF baseline is a solved engineering task on a 12 GB GPU.

### Gaps
- No peer-reviewed whole-MaleCNS or MaleCNS-optic-lobe simulation paper located (searches of arXiv/bioRxiv for "MaleCNS simulation" returned only generic SNN papers).

## Key Question 6 — Whole-CNS matrix size and single-GPU feasibility

### Takeaway
The full graph is ~25.6–25.9 M nonzero directed connections among ~166 k neurons (all weights ≥1, conf ≥0.5); ~6.1–6.3 M at weight ≥5. As sparse CSR (int32 indices + float32) that is ~0.3 GB (≥1) or ~75 MB (≥5) — trivial for one GPU; a dense 166k² float32 matrix (~110 GB) is not.

### Cited Findings
- 25.6 M connections between 166,391 neurons (paper) — [bioRxiv v2 full](https://www.biorxiv.org/content/10.1101/2025.10.09.680999v2.full)
- 25.9 M connections at `--min-weight 1`, 6.29 M at ≥5 synapses (community loader) — [fly-brain-minecraft README](https://github.com/blendi-remade/fly-brain-minecraft/blob/main/README.md)
- 6.1 M edges at weight ≥5 across 162,517 neurons — [fly-afterlife SEAM.md](https://raw.githubusercontent.com/nsfm/fly-afterlife/main/docs/SEAM.md)
- nfly reports "10.5 million synaptic connections" in the complete dataset and 8.4 M edges in its visual sub-network (threshold not stated) — [nfly](https://github.com/zhengxuyu/nfly)
- The weights table itself is 1.1 GB Feather — [Download page](https://male-cns.janelia.org/download/)
- Full-CNS LIF runs at 2.4× realtime on an RTX 3060 12 GB — [annel0/flybrain](https://github.com/annel0/flybrain); nfly runs end-to-end on a laptop CPU — [nfly](https://github.com/zhengxuyu/nfly)

### Inferences
- Memory arithmetic: 25.6 M × (4 B col + 4 B val) + 166 k × 4 B row-ptr ≈ 205 MB CSR; COO with int64 ids ≈ 0.5 GB. Even with per-edge learnable gains + Adam states (×4) it stays under 2 GB. Step cost is an SpMV of 25.6 M nnz, i.e., sub-millisecond on a modern GPU.
- nfly's 10.5 M likely reflects a different threshold (e.g., ≥2) or bodyId-typed filtering; treat 25.6 M (≥1) and ~6.2 M (≥5) as the canonical numbers.

### Gaps
- No official statement of nonzero count by threshold from Janelia; numbers above come from paper text + community loaders.

## Key Question 7 — Comparison with FlyWire, hemibrain, optic-lobe dataset, BANC; practical caveats

### Takeaway
MaleCNS is the only complete male dataset and the only complete CNS with both optic lobes including lamina; BANC (female, 2026) has ~188 k neurons but lacks lamina/ocelli; FlyWire (female brain, v783) lacks the VNC; hemibrain is a partial female brain; MANC is a male VNC only. MaleCNS caveats: postsynaptic completion only 42%, synapse P/R 0.82/0.81, released tables are minconf-0.5, and only 15 optic-lobe types have column tags.

### Cited Findings
- Previous fly brain connectomes were all female (hemibrain 2020; FlyWire full female brain 2024); a female CNS connectome was "in the process of being mapped" (Oct 2025) — [Janelia news](https://www.janelia.org/news/researchers-reveal-connectome-of-the-male-fruit-fly-central-nervous-system)
- BANC: female brain + VNC, ~188,000 neurons, 199 M predicted synapses, 4 nm in-plane ssEM, community/citizen-science proofreading, annotated for type, NT, hemilineage, behaviour; lacks the lamina, ocelli and ocellar ganglion (~9,390 cells missing); paper Bates, Phelps, Kim, Yang et al., Nature 2026 "Distributed control circuits across a brain-and-cord connectome"; data at codex.flywire.ai?dataset=banc — [htem/BANC-project](https://github.com/htem/BANC-project); [Nature s41586-026-10735-w](https://www.nature.com/articles/s41586-026-10735-w); [PMC13518251](https://pmc.ncbi.nlm.nih.gov/articles/PMC13518251/); [Codex BANC](https://codex.flywire.ai/?dataset=banc)
- MaleCNS provides parallel type annotations for FAFB/FlyWire, hemibrain and MANC; 97.5% of neurons matched; FlyWire annotations were extended/cross-validated against MaleCNS — [bioRxiv v2 full](https://www.biorxiv.org/content/10.1101/2025.10.09.680999v2.full); [flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations)
- FlyWire v783 and hemibrain v1.2.1 meshes are available transformed into MaleCNS space — [Explore page](https://male-cns.janelia.org/explore/)
- Reiser lab hosts parallel Cell Type Explorers for male CNS and for the female adult fly brain (FlyWire) via neuView — [reiserlab/neuView](https://github.com/reiserlab/neuView)
- Optic-lobe dataset: `optic-lobe:v1.0` (paper) / `v1.1` (explorer, last edit 2024-09-11) is the right OL of the same male brain — [Nature/бioRxiv Nern 2025](https://www.biorxiv.org/content/10.1101/2024.04.16.589741v1); [OL explorer](https://reiserlab.github.io/male-drosophila-visual-system-connectome/index.html)
- Caveats (numbers): presynaptic completion 94%, postsynaptic 42%, both-ends-proofread connections 40.1%, synapse detection P/R 0.82/0.81 — [bioRxiv v2 full](https://www.biorxiv.org/content/10.1101/2025.10.09.680999v2.full)
- Modelling caveats: 57% of neurons never spike in LIF (graded optic lobe), <1% of types have electrophysiology, point-neuron limits, parameter degeneracy — [annel0/flybrain](https://github.com/annel0/flybrain)
- Column-tag caveat: only 15/282 OL types tagged; lobula wide-field types unvalidated — [CAOS_RES_Conectoma PR #3](https://github.com/fsantibanezleal/CAOS_RES_Conectoma/pull/3)
- Cross-platform NT/alias ambiguity example (OA-AL2b1/2) shows naming drift between platforms — [Neuroinformatics 2026](https://link.springer.com/article/10.1007/s12021-026-09783-4)

### Inferences
- For a visual-system model MaleCNS is the best single source (both eyes, lamina, VPNs and their central targets, NT predictions); FlyWire is the best cross-check for female optic lobes; BANC cannot supply retinotopic input.
- 42% postsynaptic completion means many weak edges are missing; using weight ≥5 (the community default, ~6 M edges) is a reasonable noise floor, while ≥1 keeps 25.6 M edges with many spurious/orphan-fragment links.

### Gaps
- Head-to-head neuron/synapse counts for FlyWire v783 and hemibrain v1.2.1 were not re-verified in this session (outside the scope of fetched sources).
- BANC licence and neuPrint availability unverified (Nature page paywalled).
