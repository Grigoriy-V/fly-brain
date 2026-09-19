# Project Map

The conceptual map of the **current implementation** of What Does a Fly Dream
Of?: components, ownership and data flows. Not a roadmap or a history.
`docs/OPERATIONS_MAP.md` has data locations, configuration and commands,
`ROADMAP.md` has current work, `AGENTS.md` and `docs/ARTEFACTS.md` the rules,
`DECISIONS.md` the reasons behind the boundaries below, `reports/` the
evidence.

**State on 2026-09-20:** data, model zero, the decoder ladder, encoder
inversion, the learned 13A decoders, the 13B conditional flow generator and
the item-14 prompt experiments are measured. The three Modal apps support
training, decoding and generation. States that no single clip caused have
already been rendered (item 11's noise, flash and dark states; 14.2's states
composed by region), and 13B can also draw video from z alone with its
unconditional mask; what does not exist is a **learned model of the
distribution of reachable T4/T5 states**, so a valid state cannot be sampled.
The current track (item 17) builds that prior, after measuring the free
unconditional baseline first. MaleCNS fine-tuning is paused; an
internal source of spontaneous states is a deferred draft (item 16). The
current order and approvals live in `ROADMAP.md`.

## System at a glance

```text
 MaleCNS v1.0 (neuPrint / Feather)        FlyVis 1.2.0 (package + pretrained ensemble)
   annotations, NT, weights, column ROIs     connectome JSON, 50 members ordered by loss
              │                                        │
              ▼                                        ▼
 ┌─────────────────────────────┐        ┌──────────────────────────────┐
 │ flydream.data      (built)  │        │ flyvis.NetworkView (package) │
 │ fetch, manifest, type bridge│        │ params by type and type pair │
 │ column ROIs, orientation,   │        │ flash / edge protocols       │
 │ FlyVis-shaped export        │        │ BoxEye rendering, Sintel     │
 └──────────────┬──────────────┘        └──────────────┬───────────────┘
                │ filters_R.json                        │ transplant (rescaled, capped)
                ▼                                       ▼
        ┌──────────────────────────────────────────────────────┐
        │ flydream.model.zero (built): FlyVis DMN on either     │
        │ connectome; flash + edge validation; shuffled control  │
        └───────────┬───────────────────────────┬───────────────┘
                    │ Network.stimulus_response  │ differentiable
                    ▼                           ▼
        ┌──────────────────────────┐   ┌────────────────────────┐
        │ flydream.decode (built)  │   │ flydream.generate      │
        │ pairs → ridge / hexconv  │   │ (built): invert, 13B    │
        │ metrics, hexraster, map, │   │ flow generator, round   │
        │ figures, sweep, ensemble │   │ trip, prompts; T4 app   │
        └─────────┬────────────────┘   └───────────┬────────────┘
                  │                                │
                  └──────────────┬─────────────────┘
                                 ▼
          data/decode/<run>/ (map.csv, pairs.npz, meta.json)
          data/gen13b/ + data/prompts14/ (states, samples, scores)
          reports/figures/ + reports/<date>_<step>.md + reports/runs.jsonl
          deploy/modal/ (training, decoding and generation apps)
```

The generator loop, and the one piece of it that does not exist yet:

```text
video ─→ frozen brain ─→ T4/T5 state ─┬─→ 13B (built) ─→ video ─→ frozen brain
                                      │                              │
noise ─→ state prior (item 17, NOT    │                              ▼
         BUILT) ────────────────────→─┘                     round trip / controls
```

## Components and owners

- **Data** (`flydream/data/`, built 2026-09-18): `sources.py` is the only
  place a URL is written; `fetch.py` downloads and writes
  `data/manifest.json` (url, bytes, sha256, licence, date);
  `fetch_sintel.py` delegates to flyvis's own Sintel downloader and records
  it; `bridge.py` writes `data/bridge/types.csv` (FlyVis node → MaleCNS
  types, kind, counts, note; unmatched rows are the residue);
  `columns_roi.py` fetches every optic-lobe neuron's column ROI synapse
  counts from neuPrint; `optic_lobe.py` builds one side's neurons and edges
  at the weight threshold; `orientation.py` chooses the integer map from
  MaleCNS hex axes to FlyVis (u, v); `export.py` writes
  `data/ol/filters_<side>.json` in the shape `ConnectomeFromAvgFilters`
  reads. Thresholds and the side are `config.toml [data]`. Known defects of
  the export: `ISSUES.md` ISS-0001, ISS-0002, ISS-0005.
- **Model** (`flydream/model/`, model zero built 2026-09-18): `__init__.py`
  points flyvis at `data/flyvis` (`configure_flyvis_root`, before flyvis is
  imported) and patches datamate's h5 writer for Windows; `zero.py` builds a
  FlyVis `Network` on a connectome export (content-addressed copy, so a
  re-export never meets a stale datamate cache), transplants a pretrained
  member's parameters by cell type and type pair with the gain rescaled to
  preserve total input per target cell and capped (`transplant`,
  `total_n_syn`; DECISIONS 2026-09-18), runs the flash and moving-edge
  protocols on a bare Network (`responses`), scores FRI polarity and T4/T5
  DSI/preferred direction against `flyvis.utils.groundtruth_utils`, and
  carries the shuffled-strength control (`--control`) and the pre-fix
  transplant for comparison (`--no-rescale`). Raw response datasets and the
  capped-pair list are written per run under `data/runs/<run>/`. Training
  and the biophysical additions are still to come.
- **Decode** (`flydream/decode/`, rungs one and two built 2026-09-18):
  `pairs.py` simulates a stimulus set through a network and keeps the
  rendered 721-hexal stimulus beside the activity, split by scene;
  `ridge.py` is rung one with the penalty tuned per cell type on a relative
  grid, plus the two pairing-breaking controls (`shuffle_time`,
  `shuffle_samples`) and cell subsets; `hexconv.py` is rung two, flyvis's
  own `DecoderGAVP` architecture on the hexagonal lattice, taking a cell
  type's activity directly (`fit` is for a handful of types locally; the
  full sweep is a Modal job); `metrics.py` scores PixCorr on the 721-vector,
  R², spatiotemporal correlation, SSIM on the raster and identification
  with flat frames excluded; `hexraster.py` turns the lattice into a
  Voronoi raster (125×125 at four pixels per ommatidium) and a neighbour
  index; `map.py` is the driver, one row per cell type per control, model
  and connectome as arguments; `figures.py` draws the stage ladder and a
  clip from a run's cached pairs. Settings in `config.toml [decode]`,
  including the lag windows (consecutive lags only, ISS-0006); `sweep.py`
  joins lag windows, `ensemble.py` joins members × splits (both beyond the
  branch). The ridge's penalty is chosen on held-out scenes; GCV was
  measured and rejected (under-penalises autocorrelated frames).
- **Generate** (`flydream/generate/`, built 2026-09-19, the deliverable):
  `invert.py` — encoder inversion: the video (frames × 721 hexals, one leaf)
  optimised by Adam through the frozen network so that the chosen cells'
  activity matches a target state, with a total-variation prior; the target
  is re-simulated from a clip with the same grey steady state, or (item 11)
  comes from a state no clip caused; the control is a second input clip.
  `run_ladder` does every stage in one model load; `figures.py` draws the
  ladder and the clip; `clip_from_sintel` takes the clip from the dataset
  so no pairs file is needed. Settings in `config.toml [generate]` include
  frames and the end margin.
  `learned.py` holds the shared hex machinery (`ring_index`) and 13A's linear
  and CNN decoders; `gen13b.py` is the conditional flow generator — the SiT
  linear interpolant, the structured type masks, the two backbones
  (`HexResNet`, `SiTColumns`), the optimised training loop with EMA, and
  `sample` with classifier-free guidance; `roundtrip13.py` builds the states
  of items 11-12 and scores the round trip `state → generator → video →
  frozen brain → state′` per type against a reference variance;
  `prompts14.py` evaluates random, edited, region-composed and hand-written
  states, reads the direction the brain finds in a generated video
  (`direction_energy`) and runs the closed generator-brain loop. The item-17
  prior over states will reuse `gen13b`'s interpolant and backbone; it is not
  written yet. See the 13A, 13B and 14 reports linked in `ROADMAP.md`.
- **Train** (`flydream/train/`, built 2026-09-18): `member.py` composes
  flyvis's own solver from its Hydra config on a connectome export
  (`--init` a transplanted state, `--variant stats_relu` the two measured
  source-level changes, actual-iteration timing, evidence and profiler);
  `optimizations.py` binds those changes reversibly (device-side activity
  statistics, ReLU before the gather; 1.17× on a T4, equivalent within
  CUDA noise); `benchmark.py` is the paired benchmark. Training itself is
  paused (ROADMAP 3').
- **Tools** (`tools/`): `run_log.py`, the only writer of
  `reports/runs.jsonl`; `modal_watch.py` (list, wait on, log Modal apps);
  `map_local.py` (the ensemble map locally, parked); figure scripts
  (`fig_batch_throughput.py`, `fig_optimization_bench.py`);
  `verify_train_optimizations.py`, `summarize_training_benchmark.py`.
- **Tests** (`tests/`, 77 offline passed on 2026-09-19):
  `fixtures/mini_connectome.json` (five
  types on a 19-column lattice, one double-inversion ON pathway and one
  strided wide-field type) with `test_mini_network.py` building and
  simulating it; `test_transplant.py` (the rescale invariant and the cap);
  `test_decode.py` (metrics, batched SSIM against skimage, ridge
  scale-invariance and GCV against hold-out, controls, pairs, raster);
  `test_hexconv.py`; `test_sweep.py`; `test_generate.py`;
  `test_train_optimizations.py`;
  `test_data_helpers.py`. No test downloads, calls Modal or needs a
  credential.
- **Deploy** (`deploy/modal/`, built 2026-09-18/19): three Modal Apps on a
  T4 — `train_app.py` (`smoke`, `smoke_packed`, `smoke_batch`, `train`,
  `train_packed`, `benchmark`), `decode_app.py` (simulate a member on the
  GPU, fit maps on CPU workers; parked, CPU workers cost more than local),
  `generate_app.py` (`ladder`: every stage's inversion in one worker, the
  videos returned, nothing large moved). Volumes `flydream-data` (export,
  Sintel, init states, the pretrained ensemble) and `flydream-runs`
  (results, benchmarks, decode and generate roots). `FLYDREAM_ROOT` points
  a worker's project root at the Volume.

## Boundaries

Engineering benchmark (2026-09-18, completed on one T4):
`flydream/train/optimizations.py` binds reversible per-instance adapters for
device-side diagnostic reductions and node-wise ReLU before edge gathering.
`flydream/train/benchmark.py` compares four variants in separate processes,
with two paired repeats and a separate profiler run. `train/member.py` records
actual optimizer-iteration deltas. Default training remains `baseline`.
The paired runs measured `stats_relu` at 1.164× and 1.172× the corresponding
baseline. Loss curves passed the specified tolerance; final decoder states
did not, while the network's free parameters matched within 3e-8 and the
decoder drift stayed within the observed baseline-to-baseline CUDA variation.
This 30-iteration test does not establish equal long-run convergence. Evidence:
`reports/2026-09-18_training_optimization_bench.md`.

- The repository holds code, settings, manifests, reports, figures and
  notes; data, activity tensors, cached pairs and checkpoints live under
  `data/` (ignored) and on Modal Volumes, reproducible from the manifest
  and the run's config.
- A run's identity is `<date>_<experiment>_<tag>`; a changed config is a new
  run, never an overwrite (`data/decode/…_v2`, `data/runs/…_v7_total_cap3`).
- Every map row carries its controls; a report that shows a number without
  them is a defect (`ISSUES.md`). A number read from one ensemble member,
  one split or one lag window is a draft, not a result.
- The unit of evidence on the generator track is the **round trip** of a
  generated video through the frozen brain, reported beside the same number
  for a real clip and for a state known to be unreachable (a shuffled state,
  noise in the types) — with every control in a table rebuilt in that table's
  own code path, since the two existing shuffled-state controls (13B's 0.96,
  item 14's 19.1) do not share a scale. A result that was not caused by a
  source clip is additionally reported with two distances: the sampled state
  to its nearest training state, and the generated video to its nearest
  training video. Without the second, the claim is "generated without a source
  clip", never "a video that exists in no clip" (`AGENTS.md`).
- Third-party mappings enter through `flydream.data` with a check against
  the source and a listed residue, never by copying a table in.
- Subagent output (`research_notes/`) is data the project agent checks
  before it becomes a record; the 2026-09-18 audit is the example.
- Secrets (`NEUPRINT_TOKEN`, Modal, Hugging Face) live in `.env` and reach
  Modal through one secret published from it; nothing reads them but the
  fetch and deploy code.
