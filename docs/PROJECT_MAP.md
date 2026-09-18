# Project Map

The conceptual map of the **current implementation** of What Does a Fly Dream
Of?: components, ownership and data flows. Not a roadmap or a history.
`docs/OPERATIONS_MAP.md` has data locations, configuration and commands,
`ROADMAP.md` has current work, `DECISIONS.md` the reasons behind the
boundaries below, `reports/` the evidence.

**State on 2026-09-18 (evening):** data, model zero and the first two rungs
of the decoder ladder exist and are tested; inversion (rung three), training
on Modal and the central brain are marked "(to come)" until the step that
builds them removes the mark.

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
        │ flydream.decode (built)  │   │ flydream.decode.invert │
        │ pairs → ridge / hexconv  │   │ (to come): gradient    │
        │ metrics, hexraster, map, │   │ descent on the input,  │
        │ figures                  │   │ ensemble prior,        │
        └─────────┬────────────────┘   │ compatibility score    │
                  │                    └───────────┬────────────┘
                  └──────────────┬─────────────────┘
                                 ▼
          data/decode/<run>/ (map.csv, pairs.npz, meta.json)
          reports/figures/ + reports/<date>_<step>.md + reports/runs.jsonl
          deploy/modal/ (to come at roadmap 3)
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
  including the lag windows, members and splits a reported number must
  sweep (DECISIONS 2026-09-18; ISSUES ISS-0004). Inversion (`invert.py`) is
  to come.
- **Tools** (`tools/`): `run_log.py`, the only writer of
  `reports/runs.jsonl`.
- **Tests** (`tests/`, 49 offline): `fixtures/mini_connectome.json` (five
  types on a 19-column lattice, one double-inversion ON pathway and one
  strided wide-field type) with `test_mini_network.py` building and
  simulating it; `test_transplant.py` (the rescale invariant and the cap);
  `test_decode.py` (metrics, ridge scale-invariance, controls, pairs,
  raster); `test_hexconv.py`; `test_data_helpers.py`. No test downloads,
  calls Modal or needs a credential.
- **Deploy** (`deploy/modal/`, to come at roadmap 3): Modal Apps for
  training, ensemble simulation, the hex-conv sweep and inversion batches;
  Volumes for data and checkpoints; the patterns of the owner's harness
  repository.

## Boundaries

- The repository holds code, settings, manifests, reports, figures and
  notes; data, activity tensors, cached pairs and checkpoints live under
  `data/` (ignored) and on Modal Volumes, reproducible from the manifest
  and the run's config.
- A run's identity is `<date>_<experiment>_<tag>`; a changed config is a new
  run, never an overwrite (`data/decode/…_v2`, `data/runs/…_v7_total_cap3`).
- Every map row carries its controls; a report that shows a number without
  them is a defect (`ISSUES.md`). A number read from one ensemble member,
  one split or one lag window is a draft, not a result.
- Third-party mappings enter through `flydream.data` with a check against
  the source and a listed residue, never by copying a table in.
- Subagent output (`research_notes/`) is data the project agent checks
  before it becomes a record; the 2026-09-18 audit is the example.
- Secrets (`NEUPRINT_TOKEN`, Modal, Hugging Face) live in `.env` and reach
  Modal through one secret published from it; nothing reads them but the
  fetch and deploy code.
