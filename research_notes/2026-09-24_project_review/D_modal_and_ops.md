# D — Modal apps and operations tools

Reviewer D, 2026-09-24. Scope: `deploy/modal/{decode_app,generate_app,train_app}.py`,
`tools/{modal_call,modal_watch,run_log,map_local,run_flyvis_ladder,verify_train_optimizations,summarize_training_benchmark,sweep_guidance18,cmp_layers_flyvis}.py`,
`pyproject.toml`, `env.example`, `.gitignore`, `.gitattributes`, `.claude/settings.json`,
checked against `AGENTS.md` ("Compute and money", "Safety and evidence") and
`docs/OPERATIONS_MAP.md`.

Method: read-only. No `modal` command, no network. Files read in full; the Modal
decorators listed via `ast`; `tools/modal_call.coerce` run locally; `reports/runs.jsonl`
parsed strictly; `git grep` / `git log --all -G/-S` for secrets; local (gitignored)
`data/*/…json` summaries read for provenance fields. Line numbers are as of `f7bf74b`.

**Secrets: none found.** No token-shaped string (Modal `ak-/as-`, `hf_`, JWT `eyJ…`,
`NEUPRINT_TOKEN=<value>`, `*_TOKEN/SECRET/KEY = <16+ chars>`) in any tracked file or in
any of the 280 commits of `--all`; `.env` was never tracked; no Modal function takes a
`modal.Secret` or prints/writes environment variables. The only identity leak is D18
(low).

---

## D1 — high — the 18.x entrypoint defaults read the FlyVis "wrong model" states and overwrite the MaleCNS corpus maps in place

- **Where:** `deploy/modal/generate_app.py:1957-1958` (`out18="pairs18"`), `:1982-1988`
  (`--maps18-run` → `maps13b(run=out18, out=f"{gen18}/maps_deep.npz", dct_out=f"{gen18}/maps_dct{k}.npz")`),
  `:470-472` and `:508-510` (`np.savez` with no existence check), `:1971-1975`
  (`--pairs18-run` → `pairs13(out=out18)`), `:216-232` (`pairs13` writes shards in place
  before the manifest); the same `run="pairs18"` default in `train_vae18:624`,
  `pca19:710`, `invert17:979`, `class_probe18:1287`, `bench17:1446`.
- **Statement:** every item-18+ default names `/runs/pairs18`, which the records declare
  unusable (FlyVis states), and the maps builder overwrites `gen18/maps_deep.npz` and
  `gen18/maps_dct16.npz` — the MaleCNS training states of items 18-29 — without a guard
  and without recording which run it was built from.
- **Evidence:** `reports/runs.jsonl` line 72: run `2026-09-20_pairs18_wrong_model`,
  "corpus through the WRONG network — the entrypoint default flow/0000/000 … states
  written to /runs/pairs18 and unusable … kept on the volume at the human's word … the
  correct pass is /runs/pairs18m". Local `data/pairs18/manifest_summary.json` has
  `"model": "flow/0000/000"`, `data/pairs18m/manifest_summary.json` has `"model":
  "malecns"`. `maps13b`'s returned summary (`:482-487`) has no `run`/`model` key —
  `data/gen18/maps.json` confirms (`model`, `run` absent). The only guard in the
  2,130-line file is `dct_maps:1426` ("AGENTS: a dataset is never overwritten").
- **Failure scenario:** `modal run deploy/modal/generate_app.py --maps18-run` (no
  `--out18`) rebuilds gen18 maps from the FlyVis states, overwrites the 9 GB and 2.9 GB
  files every prior of items 18-29 was trained on, and the new `maps.json` does not say
  so; every later `train17/pca19/sample` reads FlyVis states labelled as the MaleCNS
  corpus. `--pairs18-run` with defaults overwrites the kept wrong-model evidence with
  MaleCNS states under the same name; a timeout mid-way leaves the old manifest over a
  mix of old and new shards (Modal volumes commit in the background).
- **Fix:** default `out18="pairs18m"` (or no default); refuse to write `maps13b`/`pairs13`/
  `pairs13_videos` outputs that exist (as `dct_maps` does); record `run` and the pairs
  manifest's `model` in the maps summary and in the `.npz`; write shards to a temp dir
  and rename after the manifest.
- **Confidence:** confirmed by reading (+ the recorded incident); not triggered.

## D2 — high — fixed output names overwrite measured checkpoints and datasets on re-run

- **Where:** `generate_app.py` `train13b:588,607-611` (`/runs/gen13b/<kind>.pt`),
  `train13:244,270,320-322` (`/runs/train13/<cond>_<kind>.pt`), `train17:1046,1212-1229`
  (`<out>/<name>.pt`, default `prior17/state_flow`), `train_vae18:623,668`,
  `pca19:711,813-816`, `roundtrip13:366`, `multi_init13b:1702,1770`,
  `sample13b:1780,1920` (`gen13b/samples.npz` whatever `ckpt`), `sample17:1554,1693`
  (`prior17/samples17.npz` whatever `ckpt`), `class_probe18:1373-1374`,
  `maps13b:413`, `pairs13_videos:163`; local copies in `main` (`:2002`, `:2013`,
  `:2039`, `:2064`).
- **Statement:** no generate function carries a date, config hash or existence check
  in its output path, so re-running a documented command with any changed argument
  silently replaces the measured artefact (AGENTS: a changed configuration gets a new
  identity; overwriting a trained checkpoint is a human gate).
- **Evidence:** `grep -nE "exists\(\)|exist_ok=False|FileExistsError"` in
  `generate_app.py` → only `:309` (resume) and `:1426` (dct_maps). OPERATIONS_MAP:243
  documents `--train13b-run --kind sit --steps13b 20000 …` as a command; it writes
  `/runs/gen13b/sit.pt`, the generator every later item, `sweep_guidance18` and the
  post-1 figures use. `pca19:744-745` guards only the smoke ("иначе она молча затрёт
  настоящий базис") — the authors saw the problem for one case.
- **Failure scenario:** `--train13b-run --kind sit --steps13b 6000` (or a different
  seed) replaces `sit.pt` and `sit_train.json`; `--train13-run --epochs 3` replaces the
  13A decoders and `deep_inversion_roundtrip.npz`, whose `ids` `sample13b:1815` and
  `multi_init13b:1723` read by hard-coded path.
- **Fix:** one helper `out_path(stage, name, cfg)` that appends `<date>_<cfg-hash>` or
  refuses an existing file unless `overwrite=True` is passed explicitly; derive
  sample/probe output names from `ckpt`.
- **Confidence:** confirmed by reading.

## D3 — high — default names that encode a config mislabel the file when one argument is overridden; priors do not record their training data

- **Where:** `resid22:905-907` (`k=1536, code=4, name="resid_ab_k1536_c4"`),
  `blockae22:837-838` (`code=8, lattice="third", name="blockae_ab_third_c8"`),
  `invert17:978` (`prior="…w192…", out="prior18/eps_w192.npz"`), `pca19:711-712`
  (`name="pca2048"` whatever `types`/`k`/`sources`), `train17:1214-1228` (meta has no
  `maps_file`/`run`); OPERATIONS_MAP:222 and `tools/modal_call.py:3`.
- **Statement:** overriding a parameter without also renaming writes a checkpoint whose
  name states the old config and overwrites the measured one; and a prior's checkpoint
  cannot be traced to the maps file it was trained on.
- **Evidence:** `modal_call.py resid22 --kw k=1024` → `prior22/resid_ab_k1536_c4.pt`
  holding a k=1024 model. `pca19 --kw types=T4a,T4b` without `name` → overwrites
  `prior19/pca2048.npz` (the full-state basis; the block basis was saved by hand as
  `pca_ab2048.npz`). The documented example `modal_call.py train17 --kw steps=20000
  name=corpus_dct16 --out data\prior18\train.json` runs with train17's defaults
  `out="prior17"`, `maps_file="gen13b/maps_deep.npz"`, `dct_k=0`: a Sintel-13B,
  non-DCT prior named `corpus_dct16`, its summary written into `data/prior18/`. Local
  `data/prior18/corpus_dct16_train.json`: `maps_file: None, run: None` (only
  `n_train: 13555` hints at the corpus).
- **Failure scenario:** a report cites `resid_ab_k1536_c4` for a k=1024 number; a
  prior is compared across corpora by name only.
- **Fix:** build default names from the arguments (`f"resid_{types}_k{k}_c{code}"`),
  refuse existing names, add `maps_file`, `run`, the pairs `model` and the code hash to
  every `meta`; fix the example to pass `out=prior18 maps_file=gen18/maps_dct16.npz
  run=pairs18m`.
- **Confidence:** confirmed by reading and by the local summary JSON.

## D4 — high (dormant) — `train`/`train_packed` defaults cannot finish inside their timeout, start from scratch, and can busy-wait on a stale cache for 24 h

- **Where:** `deploy/modal/train_app.py:193-203` (`train`: `n_iters=250_000,
  batch_size=4, init=None`, `timeout=24*60*MINUTES`), `:206-212` (`train_packed`:
  8 cores, 32 GB, 24 h), docstring `:38-39`, `:197-200`; datamate
  `.venv/Lib/site-packages/datamate/directory.py` (`while meta.status == "running":
  sleep(0.01)`).
- **Statement:** at the measured 0.434 s/iter (OPERATIONS_MAP:142) the default run is
  250,000 × 0.434 = 108,500 s = 30.1 h > the 24 h Modal maximum, so it bills a full day
  and returns nothing; the default `init=None` is the from-scratch run AGENTS forbids
  ("starts from the transplanted weights, never from scratch").
- **Evidence:** the arithmetic above; `stats_relu` (1.17×) still gives 25.7 h. The
  docstring says "a training run must fit in $0.50". `init` is not part of flyvis's
  NetworkDir config (`flydream/train/member.py:28-40`), so a second `train --run
  0100/000` with the same n_iters/batch/dt but a different `--init` reuses the first
  run's directory (datamate `enforce_config_match=True` reuses on equal config) and
  writes its checkpoints and loss over the old ones. A cache directory left
  `status: running` by a killed container makes the next worker spin until timeout.
- **Failure scenario:** one approved `modal run --detach …::train --run 0100/000` bills
  ≈24 h × (T4 + 1 core + 8 GB) and yields no result record; a packed run bills
  ≈24 h × (T4 + 8 cores + 32 GB).
- **Fix:** require `init` (error when absent unless `from_scratch=True` is passed);
  make `n_iters` required or cap it by `timeout × measured s/iter`; chain 20 h segments
  with `--resume`; include `init` in the run id; timeout ≈ 1.5× the expected duration.
- **Confidence:** confirmed by reading and arithmetic; the reuse path and the stale
  "running" loop are plausible (datamate code read, not exercised). Training is parked,
  so nothing has billed.

## D5 — medium — every Modal record says "T4" whatever card ran, and nothing enforces "nothing above L4"

- **Where:** `generate_app.py:38`, `decode_app.py:42`, `train_app.py:56`
  (`GPU = os.environ.get("FLYDREAM_GPU", "T4")`); image env `generate_app.py:64`,
  `decode_app.py:53`, `train_app.py:66`; every summary's `"gpu": GPU`
  (`generate_app.py:229,354,396,582,613,671,802,891,964,1032,1231,1524,1685,1771,1922`),
  `train_app.py:91,126,186` (`gpu_spec`), `decode_app.py:101`.
- **Statement:** the module is re-imported inside the container, whose image does not
  set `FLYDREAM_GPU`, so `GPU == "T4"` in every returned record even when the function
  was deployed or run with `FLYDREAM_GPU=L4` (or `A100`/`H100`, which are accepted
  unvalidated).
- **Evidence:** the three `.env({...})` calls set only `FLYVIS_ROOT_DIR`,
  `FLYDREAM_ROOT`, `PYTHONUNBUFFERED`. Only `flydream/train/member.py:150` records the
  real `torch.cuda.get_device_name(0)`; generate/decode records do not. For the deployed
  app the card is frozen at `modal deploy` time from whatever the deploying shell had.
- **Failure scenario:** a PowerShell session that set `$env:FLYDREAM_GPU='L4'` for one
  job later runs `modal deploy`; every `modal_call` is L4 for weeks, every summary and
  price says T4.
- **Fix:** `assert GPU in {"T4", "L4"}` at import; record
  `torch.cuda.get_device_name(0)` (and the requested cpu/memory from the decorator, not
  hand-copied constants) in every summary.
- **Confidence:** confirmed by reading.

## D6 — medium — the deployed app runs the code and config of the last deploy, and no record says which

- **Where:** `generate_app.py:61-67` (`add_local_file("config.toml")`,
  `add_local_python_source("flydream")`), `tools/modal_call.py:55-58`,
  OPERATIONS_MAP:221 ("re-run after changing a function").
- **Statement:** `modal_call` executes the `flydream` package and `config.toml` snapshotted
  at the last `modal deploy`; no summary carries a git commit, code hash or config hash
  (only `bench13b:582` records `torch.__version__`), so a number can be attributed to a
  commit whose code did not run.
- **Evidence:** `grep -nE "sha256|hexdigest|git |config_hash"` over `generate_app.py`
  → nothing. The ops map says to redeploy after changing "a function", not after
  changing `flydream/generate/*.py` or `config.toml`, which the functions import.
- **Failure scenario:** a fix to `prior17.train` is committed, `modal_call train17` is
  run without redeploying, the report cites the fixed commit for a run of the old loop.
- **Fix:** bake `GIT_COMMIT`/`CODE_SHA` (hash of `flydream/` + `config.toml`) into the
  image env at deploy; return it in every summary; have `modal_call` compare it with
  the local tree and refuse on mismatch or a dirty tree.
- **Confidence:** confirmed by reading.

## D7 — medium — packed training drains N pipes one after another, so members 2..N stall

- **Where:** `train_app.py:113-118` (`Popen(..., stdout=PIPE, stderr=STDOUT)` × N, then
  `[p.communicate()[0] for p in procs]`).
- **Statement:** while `communicate()` waits on member 1, members 2..N block on write as
  soon as their 64 KiB pipe fills; flyvis logs at INFO (`flyvis/__init__.py:30-34`,
  per epoch and per checkpoint in `solver.py:381,395,426-471`), so a long packed run
  serializes.
- **Evidence:** the code above; the log rate reasoning (≈2 lines per ~15 iterations at
  batch 4) puts the stall at tens of minutes into a run: invisible in the 50-iteration
  `smoke_packed`, decisive in `train_packed`.
- **Failure scenario:** `train_packed --members 4` runs ≈4× longer than measured, hits
  the 24 h timeout (D4), and the packing price derived from `smoke_packed` is wrong.
- **Fix:** send each member's output to a file on the volume (`stdout=open(log,"w")`)
  and `wait()`; or read all pipes with threads/`selectors`.
- **Confidence:** plausible (mechanism certain, output rate estimated, not measured).

## D8 — medium — GPU functions still do CPU work

- **Where:** `decode_app.py:85-101` (`simulate`: flyvis renders Sintel into
  `FLYVIS_ROOT_DIR` on the T4 container on first use, `data_volume.commit()` at `:98`;
  `simulate.map(ms)` at `:149` starts up to 10 such containers at once);
  `generate_app.py:285-303` (`train13`: builds three `ShardSet`s and rescans every
  shard's `states` per condition, on a T4 with `cpu=2`); `:1641-1652` (`sample17`: CPU
  nearest-neighbour over the 4.9 GB bank and the videos on a T4 with `cpu=1`);
  `:1816-1822`, `:1725-1730` (sample13b / multi_init13b re-read `z["index"]` per id and
  a whole shard's `states` per hit).
- **Statement:** AGENTS: "rendering, data assembly and any other CPU step run on a CPU
  container or locally, and the GPU function starts from their finished output";
  these run on billed T4 time with the card idle.
- **Evidence:** the code; `maps13b` (CPU) exists for 13B but 13A's `train13` never got
  its CPU half; the 2026-09-19 incident AGENTS cites is the same pattern as `simulate`.
  Parallel `simulate` containers do not `reload()` the data volume, so each renders the
  same cache and all commit it — duplicated CPU work and last-writer-wins on shared
  files.
- **Failure scenario:** `modal run decode_app.py` on a fresh volume renders Sintel 10×
  in parallel on 10 T4s.
- **Fix:** a CPU `render`/`prepare` function first (as `pairs13_videos`); move
  `_nn_stream` to torch on the card or to a CPU function; index shards once
  (`index` → shard map) instead of per id.
- **Confidence:** confirmed by reading; minutes/dollars not measured.

## D9 — medium — `modal_call --kw` turns a single number into an int, which crashes list parameters after the paid part

- **Where:** `tools/modal_call.py:26-35` (`coerce`), consumers
  `generate_app.py:782` (`ks.split(",")` after `P.fit` at `:781`), `:1493`
  (`batches.split` after part (a) of `bench17`), `:1840` (`guidances.split`).
- **Statement:** a comma-list parameter given one value arrives as `int` and fails with
  `AttributeError` after GPU work has been paid, with nothing saved.
- **Evidence (ran locally):** `coerce('2048') -> 2048`, `coerce('0100') -> 100`,
  `coerce('1_000') -> 1000`, `coerce('nan') -> nan`, `coerce('inf') -> inf`,
  `coerce('128,512') -> '128,512'`.
- **Failure scenario:** `modal_call.py pca19 --kw ks=2048` pays the load and the
  13,555×13,555 Gram + `eigh`, then dies before `np.savez`; `bench17 --kw batches=128`
  loses its measured part (a).
- **Fix:** coerce by the target function's annotations (`inspect.signature` of the
  local module) or only when the key is annotated `int`/`float`; in the functions,
  `str(ks)`.
- **Confidence:** coerce confirmed by running; crash confirmed by reading.

## D10 — medium — `decode_app` still claims it needs no per-action gate, and a bare `modal run` launches the whole sweep

- **Where:** `deploy/modal/decode_app.py:19-21` (docstring), `:130-156` (`main`,
  `dry=False` by default), `:139-143` (price constants).
- **Statement:** the docstring says "The human allowed long decode jobs on Modal without
  a per-action gate", which OPERATIONS_MAP:145-151 says was superseded and AGENTS
  forbids; with no arguments the entrypoint prints a price and immediately starts
  10 T4 simulations + 200 CPU map jobs.
- **Evidence:** `config.toml [decode]`: 10 members × 5 splits × 4 lag windows = 200
  jobs; the script's own formula gives ≈ $61. Its rates ($0.0472/core/h, $0.008/GiB/h)
  disagree with AGENTS' (≈ $0.19/core/h, ≈ $0.024/GB/h); at AGENTS' rates the same
  sweep is ≈ $225.
- **Failure scenario:** an agent reads the module docstring, treats decode as
  pre-approved, and runs the documented `modal run --detach deploy/modal/decode_app.py`.
- **Fix:** delete the stale sentence; make `--dry` the default and require
  `--go`/`--confirm`; one price table in one place (config.toml) with its source.
- **Confidence:** confirmed by reading (which price table is right is not verified).

## D11 — medium — the settings.json backstop never matches the commands the project actually uses

- **Where:** `.claude/settings.json:7-14`.
- **Statement:** `deny: ["Bash(modal run:*)", "Bash(modal deploy:*)"]` matches only a
  command that starts with `modal`, while every documented launch is
  `.venv\Scripts\python.exe -m modal run …` or `python tools/modal_call.py …`
  (OPERATIONS_MAP:74-75, 207, 221-245), so the mechanical backstop for priced workers
  never fires; `Read(./.env)` is denied but `Bash(cat .env)` is not.
- **Evidence:** the file and the ops map lines above.
- **Failure scenario:** a priced call is made without the human's word; nothing
  mechanical stops it.
- **Fix:** add `Bash(*-m modal run*)`, `Bash(*-m modal deploy*)`,
  `Bash(*modal_call.py*)`, `Bash(*modal volume rm*)`, `Bash(*modal volume put*)` and
  `Bash(cat .env*)`-style denies (or `ask` rules).
- **Confidence:** confirmed by reading (matching semantics per Claude Code's prefix
  rules).

## D12 — medium — `sweep_guidance18` writes the guidance-0 arm under the canonical gate-arm name, without seed or size

- **Where:** `tools/sweep_guidance18.py:195-197` (`tag_of`), `:228-238`, `:234`
  (`n_clips=3` hard-coded), `:214-215` (`--out`, `--skip-done`).
- **Statement:** scale 0 is saved as `samples18_<ckpt>.json/.npz`, the name
  `flydream/generate/edges18.py:30-39` reads as the measured gate arm, and no tag carries
  `--seed`, `--samples` or `n_clips`.
- **Evidence:** existing measured arms in `data/prior18/` use exactly that pattern
  (`samples18_corpus_dct16_w192_lr1e3_c.json`, seeds as `_s1/_s2/_s3`); the sweep
  writes `samples18_<ckpt>` for any seed.
- **Failure scenario:** `--ckpt corpus_dct16_w192_lr1e3_c --seed 1` overwrites the
  seed-0 "best by gates" arm; `--skip-done` merges arms made at other seeds/sizes into
  one curve; `edges18_guidance.json` is overwritten for any ckpt.
- **Fix:** tag = `samples18_<ckpt>_g<g>_s<seed>_n<samples>`; refuse to overwrite; check
  `seed/n_samples` of a skipped arm before reusing it.
- **Confidence:** confirmed by reading.

## D13 — medium — `_prepare_root` never refreshes the connectome, and train_app builds from a path-keyed cache

- **Where:** `generate_app.py:70-76` (`if not os.path.exists(dst): shutil.copyfile(src, dst)`),
  `train_app.py:73` (`--connectome /data/ol/<file>` straight into flyvis/datamate).
- **Statement:** a re-uploaded export with the same file name never reaches generate
  workers (they build from the stale copy under `/runs/generate/data/ol/`), and
  train workers are served datamate's cache keyed by the path — the exact staleness
  `flydream/model/zero.py:49-60` (`content_addressed`) exists to prevent locally.
- **Evidence:** the code; `content_addressed`'s docstring: "datamate caches a built
  connectome by its config (the file PATH), not by the file's content, so a re-exported
  file would be served stale". The export name encodes only config settings
  (`flydream/data/export.py:187-205`), not the code that produced it.
- **Failure scenario:** an export fix (e.g. for ISS-0005/0015) is uploaded as
  `filters_R_w5wk50m500oc.json`; every Modal run keeps using model zero's old wiring
  while the reports say the fix is in.
- **Fix:** copy when the sha256 differs (or always), and pass
  `content_addressed(path)` in `train_app` as well; record the connectome hash in every
  summary.
- **Confidence:** confirmed by reading; not triggered.

## D14 — medium — OPERATIONS_MAP does not describe the app that is running

- **Where:** `docs/OPERATIONS_MAP.md:4-10, 152-155, 194-195, 222, 251-258, 266-269`;
  `docs/PROJECT_MAP.md:210-212`.
- **Statement:** the map stops at 13B/14 and misdescribes identity and layout.
- **Evidence:**
  - Ten live functions are absent: `train17`, `sample17`, `train_vae18`, `pca19`,
    `blockae22`, `resid22`, `invert17`, `class_probe18`, `dct_maps`, `bench17`, and the
    `--pairs18-run`/`--maps18-run` flags; ":266-269" still says item 17 "will be
    recorded here once measured".
  - Volume layout omits `flydream-runs:/pairs18`, `/pairs18m`, `/gen18`, `/prior17…/prior22`,
    and `flydream-data:/pairs13/{videos.npz,columns.json,procedural_*.npz}`,
    `/corpus18/videos.npz`; ":253" places the 13A videos under `flydream-runs:/pairs13/`
    but `pairs13_videos` writes `flydream-data:/pairs13/videos.npz`
    (`generate_app.py:183`).
  - ":194-195" "Run id `<date>_<experiment>_<config hash>`; the config is saved beside
    the outputs on the Volume" — not implemented anywhere in `generate_app.py` (D2, D6).
  - ":222" contains a literal TAB: `--out data\prior18<TAB>rain.json` (`\t` of
    `\train.json` was interpreted; introduced in 177f37d).
  - `PROJECT_MAP.md:210-212` says secrets "reach Modal through one secret published
    from it"; no function uses a `modal.Secret` and OPERATIONS_MAP:156 says none is
    needed. `env.example` lists only `NEUPRINT_TOKEN`.
- **Failure scenario:** the next agent copies the example (D3) or looks for the corpus
  maps where the map says they are.
- **Fix:** regenerate the Modal section from the decorators (the `ast` listing in
  "Gaps" below), fix the tab, and state the real run-identity convention.
- **Confidence:** confirmed by reading.

## D15 — medium — the data manifest the repository promises is ignored by `.gitignore`

- **Where:** `.gitignore:16` (`/data/`) and `:15` comment "reproducible from
  data/manifest.json"; AGENTS "the repository holds the code that fetches or makes them
  and a manifest with sizes and hashes"; OPERATIONS_MAP:118-122.
- **Statement:** `data/manifest.json` is excluded by `/data/` and has never been
  committed, so the public repository has no sizes/hashes for any dataset.
- **Evidence:** `git check-ignore -v data/manifest.json` → `.gitignore:16:/data/`;
  `git log --all -- data/manifest.json` → empty; `git ls-files data | wc -l` → 0. Also
  `git ls-files -ci --exclude-standard` → 208 files under `reports/figures/` still
  tracked although `.gitignore:37` ignores the directory (82c431f: "working figures no
  longer committed").
- **Failure scenario:** a reader cannot verify that their download equals the data the
  reports used; working figures keep being pushed when edited.
- **Fix:** `!/data/manifest.json` after `/data/` and commit it (or move it to
  `manifests/`); `git rm --cached -r reports/figures` if the intent stands.
- **Confidence:** confirmed by running git.

## D16 — low/medium — the Modal images do not pin what `uv.lock` pins

- **Where:** `decode_app.py:51-52`, `generate_app.py:63`, `train_app.py:65`;
  `pyproject.toml:6-17`; `uv.lock`.
- **Statement:** images install `flyvis==1.2.0` plus unpinned `hydra-core>=1.3`, `h5py`,
  `pyarrow`, `pandas`, `scipy`, `matplotlib` (decode also `scikit-image`), and torch and
  numpy come transitively at whatever version resolves when the image is built, while
  local runs use torch 2.14.0 / numpy 2.5.3 / scipy 1.18.1 / pandas 3.0.5 from `uv.lock`.
- **Evidence:** today they happen to agree on torch (local summaries record
  `"torch": "2.14.0+cpu"` locally and `"2.14.0+cu130"` from Modal); no summary except
  `bench13b` records any version. `imageio` (used in `flydream/data/video_hex.py:95`,
  `flydream/decode/figures.py:125`) is not declared anywhere.
- **Failure scenario:** any change to an image definition rebuilds it with newer
  torch/numpy; round-trip numbers from Modal and from the local CPU path stop being
  comparable without anyone noticing.
- **Fix:** `uv export --frozen > requirements.lock` and
  `Image.uv_pip_install(requirements=…)` / `uv_sync`; record `torch`, `numpy`, `flyvis`
  versions in every summary.
- **Confidence:** confirmed by reading.

## D17 — low/medium — decode and ladder run names carry no config identity; hard-coded dates

- **Where:** `decode_app.py:65-70, 93-94, 113-114`; `tools/map_local.py:33-38, 73-75,
  96`; `generate_app.py:90, 119, 147` (ladder tag `<date>_{malecns|flyvis}_`);
  `tools/run_flyvis_ladder.py:121`; `tools/verify_train_optimizations.py:33`.
- **Statement:** runs are keyed by date+member+seed+lags only and skipped when present,
  so a same-day re-run after a config change silently returns old results; local
  (`--no-subsets` by default) and Modal (subsets kept) produce different content under
  identical names; the ladder tag drops the FlyVis member (`flow/0000/000` and `/005`
  both become `…_flyvis_`); two tools stamp fixed dates (`2026-09-24_flyvis_`,
  `2026-09-18_optimization_equivalence_`) on any future run.
- **Evidence:** the lines above; `flydream/decode/map.py:199-207` shows `--no-subsets`
  changes `subset_counts` (recorded in `meta.json`, not in the name).
- **Failure scenario:** `run_flyvis_ladder.py` re-run tomorrow overwrites the data
  behind `docs/figures/two_brains` under yesterday's date.
- **Fix:** append a short hash of the effective settings to run names; derive dates
  from the clock; include the member in the ladder tag.
- **Confidence:** confirmed by reading.

## D18 — low — the owner's Modal profile/workspace name is in the public history

- **Where:** commits c2eec0f, d2fb25b, 1bb4070, 65c6f39, 0692737, 6d1f126, 3c4ca69
  (files `deploy/modal/train_app.py`, `docs/OPERATIONS_MAP.md`, `ROADMAP.md`,
  `tools/modal_watch.py`, three 2026-09-18 reports), all on `origin/master`; removed from
  the tree in 82c431f ("no personal Modal profile").
- **Statement:** an account identifier, not a credential; the scrub of 82c431f did not
  reach history.
- **Evidence:** `git log --all -S<profile>` lists the commits above;
  `git branch -r --contains c2eec0f` → `origin/master`. (Name not reproduced here.)
- **Failure scenario:** none directly exploitable without a token; it links the public
  repo to the Modal workspace.
- **Fix:** accept, or rewrite history (a human gate: force-push to a public remote).
- **Confidence:** confirmed by running git.

## D19 — low — `modal_watch` does not work for the deployed app

- **Where:** `tools/modal_watch.py:106-109, 129, 156-169`.
- **Statement:** `--wait` exits only when the state leaves {ephemeral, running,
  deployed}, so on `flydream-generate` (always "deployed") it polls for the full 240 min;
  `modal app logs` on a live app streams, so the 90 s `subprocess.run` timeout raises an
  uncaught `TimeoutExpired` and no log is saved; the prefix filter
  (`prefix[:len(desc)]`) matches any app whose description is empty.
- **Fix:** wait on a function call id instead of an app state; use `Popen` with a read
  deadline and save what arrived; match `desc.startswith(prefix)`.
- **Confidence:** plausible (Modal CLI streaming behaviour not exercised).

## D20 — low — `run_log.py` accepts non-JSON values and labels the log date as the run date

- **Where:** `tools/run_log.py:22-38`.
- **Statement:** `--value nan`/`inf` pass `type=float` and are written as `NaN`/
  `Infinity` (invalid strict JSON); `date` is `date.today()` at logging, not the run's
  date; run-id format and duplicates are not checked.
- **Evidence:** `json.dumps(float('nan'))` → `NaN` (ran). The current file is clean:
  151 records, all parse with NaN rejected, one key set, no duplicate
  (run, experiment, metric). 12 records have a run id dated differently from `date`
  (11 × run `2026-09-20_…` logged `2026-09-19`, 1 × logged `2026-09-21`).
- **Fix:** `math.isfinite` check; rename `date` → `logged` or take the run date from
  the id; warn on an existing (run, metric).
- **Confidence:** confirmed by running (serialisation) and by parsing the file.

## D21 — low — `sample17` ignores the prior's own metadata

- **Where:** `generate_app.py:1576-1604` (pairs13 manifest, `gen13b/maps_deep.npz` bank,
  `R.sample(prior, n, frames=40, k=len(DEEP))`).
- **Statement:** any prior after 17.1 (DCT-16, type subsets, corpus18) is sampled at
  the wrong shape and its novelty is measured against the 13B bank, not its own
  training set.
- **Fix:** take frames/k/dct/types/maps_file from `pmeta`; refuse a prior whose
  `n_train` does not match the bank.
- **Confidence:** confirmed by reading (the later items use the local
  `flydream.generate.samples17` path instead, so this is a trap, not a wrong number).

---

## Checked and fine

- No `keep_warm`/`min_containers`, no import-time `.remote()`/`.spawn()`; importing an
  app module does not contact Modal. No GPU above L4 is named in code (but see D5).
- `pairs13_videos`, `maps13b`, `class_probe18`, `dct_maps`, `map_window` are CPU-only as
  AGENTS requires; `train13b`/`train17`/`pca19` start from finished maps on the volume.
- `train_app.benchmark` + `flydream/train/benchmark.py`: new
  `<UTC time>_<hash>` directory, `exist_ok=False`, code/config/connectome hashes — the
  pattern D2/D6 ask for. `summarize_training_benchmark.py` and
  `verify_train_optimizations.py` refuse to overwrite.
- `reports/runs.jsonl`: 151 valid records, LF, trailing newline.
- `.env` untracked and ignored (`.env`, `.env.*`), `env.example` has no value, no token
  in 280 commits.

## Gaps

- Nothing was run against Modal: background-commit behaviour on timeout (D1), `modal app
  logs` streaming (D19), Modal's handling of argument types in `.remote` (D9) and the
  real rates (D10) are from knowledge of Modal, not verified today.
- D7's stall point depends on flyvis's log volume per epoch, estimated, not measured.
- Whether any overwrite in D1-D3/D12 has already happened cannot be told from the
  repository: volume contents were not listed and the local `data/` copies were read
  only for provenance fields.
- CPU/memory minimality was judged from the code and the ops map's measured numbers
  only; no function's peak RSS was available except where summaries record it.
  `decode_app.simulate` (4 cores, 24 GB for a 2-3 min job) and
  `train_app.smoke_packed/train_packed` (8 cores, 32 GB for four 1-core/8 GB members)
  look above need but were not measured.
- `flydream/*` modules were read only where a Modal function's behaviour depended on
  them (member.py, benchmark.py, invert.GpuSampler, zero.content_addressed,
  prior17.sample/load, decode.map flags).
- Modal function inventory from `ast` (for D14):
  decode: `simulate` T4 4c/24 GB/120 min, `map_window` 8c/16 GB/180 min;
  train: `smoke` T4 4c/16 GB/30, `smoke_packed` T4 8c/32 GB/60, `smoke_batch` T4
  4c/16 GB/60, `train` T4 1c/8 GB/24 h, `train_packed` T4 8c/32 GB/24 h, `benchmark` T4
  4c/16 GB/30; generate: `ladder`/`dreams`/`mix` T4 1c/3 GB/60, `pairs13_videos` 2c/6 GB/120,
  `pairs13` T4 1c/4 GB/120, `train13` T4 2c/12 GB/120, `roundtrip13` T4 1c/4 GB/60,
  `maps13b` 2c/12 GB/60, `bench13b` T4 1c/12 GB/30, `train13b` T4 1c/12 GB/90,
  `train_vae18` T4 1c/12 GB/90, `pca19` T4 1c/12 GB/30, `blockae22` T4 1c/16 GB/45,
  `resid22` T4 1c/16 GB/30, `invert17` T4 1c/16 GB/60, `train17` T4 1c/12 GB/90,
  `class_probe18` 2c/12 GB/30, `dct_maps` 2c/24 GB/45, `bench17` T4 1c/12 GB/30,
  `sample17` T4 1c/8 GB/40, `multi_init13b` T4 1c/6 GB/40, `sample13b` T4 1c/8 GB/40.
