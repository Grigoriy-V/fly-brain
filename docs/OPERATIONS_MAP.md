# Operations Map

Configuration, data locations, Modal, and how a run is started and read
back. Not a roadmap. **State on 2026-09-18:** nothing is set up yet; the
sections below name what each step fills in, so that a later reader finds
one place per fact.

## Machine and environment

- Owner's machine: Windows 11, PowerShell; system Python is 3.14, the
  project runs on Python 3.12.4 in `.venv` made by `uv sync --all-groups`
  from `pyproject.toml` (flyvis 1.2.0, neuprint-python, pyarrow, pandas,
  scipy, torch CPU). No local training.
- `.env` (ignored) holds `NEUPRINT_TOKEN`, Modal credentials and anything
  else secret; `env.example` lists the names with no values.
- FlyVis reads its data root from `FLYVIS_ROOT_DIR`; the pretrained
  ensemble was downloaded with it set to `data/flyvis` (`flyvis
  download-pretrained`), and every script that loads the ensemble sets it
  from `config.toml [flyvis] root_dir` before importing flyvis
  (`flydream.model.configure_flyvis_root`). The ensemble's 50 member
  directories `flow/0000/000` … `049` are ordered by validation loss
  (5.1366 → 5.6779, Spearman(id, loss) = 1.000, measured 2026-09-18): `000`
  is the best member, and a run that names one member is reading an
  extreme, not a sample. `config.toml [decode] members` lists ten spread
  over the ranking.
- Windows: datamate 1.0.0 unlinks an h5 file it still holds open;
  `flydream.model.patch_datamate_for_windows` (applied on import) replaces
  its writer. Without it every connectome build and Sintel render fails
  with WinError 32.
- A machine that sleeps stops the clock in a run's `seconds` field but not
  the run: a member's edge protocol recorded 39,225 s on 2026-09-18 across
  an eleven-hour sleep. Wall times in `runs.jsonl` for that day are suspect.
- `config.toml` holds every non-secret setting: `[data]` (`min_weight` 5,
  `side` R, `extent` 15, the weak-pair exception, columnar outputs),
  `[flyvis]` (version, ensemble, root_dir), `[model]` (`rescale_cap`),
  `[decode]` (model, members, splits, lags, lag_windows, metrics settings),
  `[benchmark]` (the training-optimisation benchmark); `[generate]` is added
  at ROADMAP item 9 (frames, margin).

## Commands (step 1)

```powershell
.venv\Scripts\python.exe -m flydream.data.fetch --core     # 1.2 GB, a gate
.venv\Scripts\python.exe -m flydream.data.bridge
.venv\Scripts\python.exe -m flydream.data.columns_roi      # neuPrint, ~90 s
.venv\Scripts\python.exe -m flydream.data.optic_lobe
.venv\Scripts\python.exe -m flydream.data.orientation
.venv\Scripts\python.exe -m flydream.data.export
.venv\Scripts\python.exe -m flydream.model.zero --models 0 1 2 --protocols flash edge --control --run <id>   # step 2, ~8 min CPU without --control
.venv\Scripts\python.exe -m flydream.model.zero --models 0 --no-rescale --run <id>          # reproduces the pre-fix transplant (ISS-0003)
.venv\Scripts\python.exe -m pytest -q                                                       # 47 offline tests
.venv\Scripts\python.exe tools\run_log.py --agent claude --run <id> --experiment <x> --metric <m> --value <v>
```

## Commands (step 4, the decodability map)

```powershell
.venv\Scripts\python.exe -m flydream.data.fetch_sintel            # 5.63 GB, a gate
.venv\Scripts\python.exe -m flydream.decode.map --stimuli edges --run <id>      # ~4 min CPU, 65 types
.venv\Scripts\python.exe -m flydream.decode.map --stimuli sintel --n-samples <n> --cache --run <id>
.venv\Scripts\python.exe -m flydream.decode.map --model malecns --stimuli edges --run <id>
.venv\Scripts\python.exe -m flydream.decode.map --stimuli sintel --pairs-from <run with pairs.npz> --lags 0 2 4 --run <id>   # one window of the lag sweep, no re-simulation
.venv\Scripts\python.exe -m flydream.decode.figures --run <run with pairs.npz>            # the stage ladder, PNG + GIF
.venv\Scripts\python.exe -m flydream.data.optic_lobe --min-weight 1 --edges-only         # the weight>=1 edge table the weak-pair exception reads
.venv\Scripts\python.exe -m flydream.data.export                                          # writes filters_<side>_<tag>.json from config.toml [data]
.venv\Scripts\python.exe -m flydream.model.init_state --member 0                          # the transplanted state for Modal --init
.venv\Scripts\python.exe -m flydream.decode.figures --run <run> --lags 0 2 4              # the ladder at an 80 ms window, named by it
.venv\Scripts\python.exe -m flydream.decode.map ... --seed 3                              # one scene split; a reported map sweeps [decode] splits
.venv\Scripts\python.exe -m flydream.decode.ensemble --glob "2026-09-18_decode_sintel_m*_s*_lag_*" --tag <tag>   # median + 10-90% over members x splits
.venv\Scripts\python.exe -m modal run deploy/modal/decode_app.py --dry                    # the map on Modal: plan and price, no worker
.venv\Scripts\python.exe -m modal run deploy/modal/decode_app.py --members 0,5 --seeds 0,1 --windows 0,0_2_4
.venv\Scripts\python.exe -m modal volume get flydream-runs /decode/data/decode/<run> data/decode/<run>   # read a Modal map back
.venv\Scripts\python.exe tools\modal_watch.py [--wait <app-id> | --logs <app-id> | --volume /benchmarks]
```

**Decode on Modal** (`deploy/modal/decode_app.py`, App `flydream-decode`): `simulate`
(T4, one FlyVis member over Sintel, caches `pairs.npz`, 2-3 min measured) and
`map_window` (8 CPU cores, no GPU, one (member, split, window), all types with
both controls). The worker's project root is `flydream-runs:/decode` (`FLYDREAM_ROOT`,
a copy of `config.toml` beside `data/decode/`), so run directories have the same
shape as local ones and `flydream.decode.sweep` / `ensemble` read them unchanged.
The pretrained ensemble lives on `flydream-data:/flyvis/results/flow/0000` (6 MB);
flyvis's Sintel rendering cache under `flydream-data:/flyvis/renderings`.

A background run's log must be written unbuffered (`python -u`, and
`grep --line-buffered` if piped): a buffered pipe shows nothing until it
ends, which on 2026-09-18 looked like three hung runs that were fine.

`--model` takes a flyvis NetworkView name (default `[decode] model`) or
`malecns` for the model-zero export, so the same command produces the map on
either substrate. Output is `data/decode/<run>/`: `map.csv` (one row per cell
type per control), `meta.json` (the settings and the model that produced it),
and `pairs.npz` with `--cache` so the simulation is not repeated.

Two traps worth naming here, both measured (`research_notes/decoder_stack/`):
`Network.stimulus_response` assigns `dt` onto the dataset it is given, so one
dataset instance must not be shared across runs at different `dt`; and the full
augmented Sintel set is about 33 GB of activity, so a run uses `--n-samples`
and `--types`, never the whole set at once.

## Data

| Source | Where it comes from | Where it lives | Licence |
|---|---|---|---|
| MaleCNS v1.0 annotations, NT, weights | `gs://flyem-male-cns/v1.0/` (Feather) or neuPrint `male-cns:v1.0` | `data/malecns/` | CC-BY 4.0 |
| MaleCNS synapse points / partners (only if needed) | same bucket, 12.7 GB + 6.8 GB | `data/malecns/` | CC-BY 4.0 |
| FlyVis connectome and pretrained ensemble | `pip install flyvis==1.2.0`; `flyvis download-pretrained` | `FLYVIS_ROOT_DIR` under `data/flyvis/` | MIT |
| Cell-type bridges | `flyconnectome/ol_annotations` (`olmatching.tsv`), `flyconnectome/2025malecns` | `data/bridge/` with the commit hash | per repo |
| Stimuli: Sintel | `flyvis.datasets.sintel_utils.download_sintel()` → `https://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip`, 5,627,783,629 bytes (5.63 GB), unzips larger; a human gate | `data/flyvis/sintel/` (`flyvis.sintel_dir`) | MPI-Sintel terms, research use |
| Stimuli: synthetic | generated by flyvis (`Flashes`, `MovingEdge`, `MovingBar`, gratings); renderings cached | `data/flyvis/renderings/` | MIT (flyvis) |
| Stimuli: other | named per experiment | `data/stimuli/` | per source |
| Validation data | Pang et al. 2024 (Dryad), Drews et al. 2020 values | `data/validation/` | per source |

`data/manifest.json` records url, bytes, sha256, licence and fetch date for
every file `flydream.data.fetch` downloads (the three MaleCNS core files as of
2026-09-18). Not yet in it: the FlyVis ensemble (fetched by flyvis's own CLI
with its checksums) and `data/bridge_olmatching_raw.tsv` (fetched inline on
2026-09-18; to be moved into `sources.py`). A download above 1 GB is a human
gate (`AGENTS.md`).

## Modal

Account: the owner's second Modal account, profile `grigoriy98smile` (the
human, 2026-09-18); `modal profile current` must print it before any
command below. The `modal` client is in the project `.venv`
(`.venv\Scripts\python.exe -m modal ...`). Patterns follow
`D:/ML/pinocchio-finetune/modal_apps/` and
`D:/ML/local-multimodal-agent/deploy/modal/`.

- **App `flydream-train`** (`deploy/modal/train_app.py`): `smoke` (a few
  iterations on the GPU, the price measurement), `smoke_packed` (N members
  as N processes on one card), `smoke_batch` (one member at several batch
  sizes), `train` (one member, `<ensemble>/<member>`), `train_packed` (N
  members packed on one card). The solver is flyvis's own `MultiTaskSolver`
  composed from its Hydra config with the connectome file, `n_syn_fill=0`
  and the iteration count overridden; `--init <state>` starts from a
  transplanted state (`flydream.model.init_state`), without it from
  flyvis's initialisation. GPU by `FLYDREAM_GPU`: **T4** by default, L4 the
  alternative, nothing above (the human, 2026-09-18). Measured on T4: batch
  4 at 0.434 s/iter (9.2 samples/s), throughput saturating at ~14 samples/s
  from batch 16 (~$12 per member at 10^6 samples); batch 64 and above
  cannot run (fewer than 64 training clips).
- **When Modal at all (DECISIONS 2026-09-18, night):** only for a GPU or for
  a measured ≥4× (better ≥6×) wall-clock gain over the owner's machine (32
  cores, 102 GB RAM); a training proposal costs single dollars. The ridge
  map is CPU-bound and runs locally, several processes at once
  (`tools/map_local.py`); `deploy/modal/decode_app.py` stays for a
  GPU-bound case. This supersedes the same day's earlier "decode on Modal
  without a gate".
- **Volumes:** `flydream-data` (`/ol/<export>.json`, `/flyvis/SintelDataSet`,
  `/init/<state>.pt`), `flydream-runs` (`/results/flow/<ensemble>/<member>/`
  as flyvis's NetworkDir, plus datamate's connectome cache and the Sintel
  rendering). Uploads with `modal volume put`; Sintel is 5.8 GB, once.
- **Secret:** none needed for training (no gated downloads). neuPrint is
  never used on Modal.
- **Ensemble ids:** this project's start at `0100`; the reference's
  pretrained release occupies `0000`.
- Every priced call is a gate: the report names the command, the expected
  duration and the price before it is asked for; `smoke` is the first and
  measures the per-iteration cost the others are priced from.

Requirements the export must meet for the flow task, both now settings of
`config.toml [data]` and part of the export's tag: every input type tiles
every column (the stimulus stacks the eight photoreceptor types), and every
output type has one cell per column (`output_units_columnar_only`; the flow
decoder stacks the output types into one map). A density-strided type stays
in the network and is dropped from the outputs.

## Runs and evidence

Training optimization benchmark (run once on a T4, 2026-09-18 22:00, ~$0.30; result:
`stats_relu` 1.17×, equivalent within CUDA noise — `reports/2026-09-18_training_optimization_bench.md`):

```powershell
.venv\Scripts\python.exe -m flydream.train.benchmark  # print local plan only
# The following starts ONE priced worker and requires separate permission:
$env:FLYDREAM_GPU = 'T4'
$env:PYTHONIOENCODING = 'utf-8'
.venv\Scripts\python.exe -m modal run --profile grigoriy98smile deploy/modal/train_app.py::benchmark
```

`config.toml [benchmark]`: four variants, two repeats in reverse order,
30 requested iterations each at batch 16, one additional baseline profile,
plus an unscored cache warm-up. The worker snapshots config/code/export hashes
and uses a new `flydream-runs:/benchmarks/<UTC timestamp and hash>/` directory.
It saves paired initial/final states, loss traces, actual counts, timings,
comparison JSON and the Chrome profiler trace. Failed jobs preserve diagnostics;
no existing experiment is overwritten. Estimates and remaining gate:
`reports/2026-09-18_training_optimization_bench.md`.

- Run id: `<date>_<experiment>_<config hash>`; the config is saved beside the
  outputs on the Volume and copied into the report.
- `reports/runs.jsonl`: one line per measured outcome, written only by
  `tools/run_log.py` (`--agent claude|codex|human`), fields: run id,
  experiment, metric, value, control value, n, cost, date.
- A report per step under `reports/<date>_<step>.md`, with the numbers, the
  controls, the run ids and the commands.

## Commands (item 8, the generator)

```powershell
.venv\Scripts\python.exe -m flydream.generate.invert --run <run with pairs.npz> --sample 3 --types T5a --frames 20 --steps 150   # local CPU, minutes per stage
.venv\Scripts\python.exe -m flydream.generate.invert --run x --from-dataset --sample 3 --types T4a T4b T4c T4d T5a T5b T5c T5d
.venv\Scripts\python.exe -m modal run deploy/modal/generate_app.py --model malecns --sample 3      # every stage on a T4, ~10 min, ~$0.10
.venv\Scripts\python.exe -m flydream.generate.figures --tags <tags from the app's output> --out <name>
```

The T4 app returns the recovered videos in its result (hundreds of KB) and
writes them under `data/generate/<tag>/`; nothing large is downloaded. The
model argument is a flyvis NetworkView name or `malecns[:member]`.

## Checks

- `pytest -q -p no:cacheprovider`: 64 offline tests, no data download, no
  Modal, no credential (~25 s).
- A model's validation (flash and moving-edge protocols against FlyVis's
  targets) is a script, run locally on CPU for one model and on Modal for an
  ensemble.
