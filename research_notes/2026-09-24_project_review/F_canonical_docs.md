# Review F: canonical documents, their agreement with each other and with the code

Reviewer area: `AGENTS.md`, `CLAUDE.md`, `ROADMAP.md`, `DECISIONS.md`, `ISSUES.md`,
`docs/PROJECT_MAP.md`, `docs/OPERATIONS_MAP.md`, `docs/ARTEFACTS.md`, `config.toml`.
Read-only review, 2026-09-24, repository at `f7bf74b` (clean). No network, no Modal.
Line numbers are for the files as of that commit.

Checks run (all local, read-only):
- `.venv/Scripts/python.exe -m pytest --collect-only -q -p no:cacheprovider` -> `146 tests collected in 5.67s` (collection only, suite not run).
- Every backtick path and every `reports/ docs/ tools/ flydream/ deploy/ tests/` path in the canonical docs checked for existence (all exist; the cited `reports/figures/*` evidence files exist and are tracked).
- Every CLI flag in the `docs/OPERATIONS_MAP.md` command blocks grepped in its module (all exist).
- `ISSUES.md` file:line citations spot-checked (ISS-0010, -0011, -0013, -0014: accurate).
- Every `config.toml` key grepped as a quoted name in `flydream/ tools/ deploy/`.
- `git log -S` to date conflicting rule texts.

---

## HIGH

### F1 - Local compute limit: AGENTS says "hours", DECISIONS and ROADMAP say "ten minutes"
- **Where:** `AGENTS.md:103-105` vs `DECISIONS.md:36`, `DECISIONS.md:549`, `DECISIONS.md:555-557`, and `ROADMAP.md:90-91`.
- **Statement:** The rules file and the decisions/roadmap give incompatible limits on how long local work may run; AGENTS was never updated after the human's 2026-09-21 rule.
- **Evidence:** AGENTS: "The owner's machine ... runs everything that fits in hours". DECISIONS 2026-09-21 (approved): "**no local computation runs longer than about ten minutes** - longer work goes to a GPU or its CPU path is optimised"; catalog row "local jobs stay under ten minutes | standing". ROADMAP: "the owner's machine ... for anything that fits in ten minutes". `git log -S "fits in hours" -- AGENTS.md` -> only `5cedfd5 2026-09-18`; `git log -S "ten minutes"` -> `b31ce52 2026-09-21` (DECISIONS/ROADMAP). A 30-60 min CPU job with no >=4x Modal gain is required-local by AGENTS:106 and forbidden-local by DECISIONS.
- **Fix:** Ask the human which stands; then make AGENTS "Compute and money" state the same limit (and how it interacts with "Modal only for a GPU or a measured >=4x gain").
- **Confidence:** confirmed.

### F2 - What counts as a result: AGENTS "one model, one split, one window" vs DECISIONS, PROJECT_MAP and config.toml "ten members, several splits, a lag sweep"
- **Where:** `AGENTS.md:30-31`, `AGENTS.md:38-40` vs `DECISIONS.md:30` + `DECISIONS.md:280-297`; `docs/PROJECT_MAP.md:193-195`; `config.toml:62-63`, `config.toml:74-76`; `docs/OPERATIONS_MAP.md:72`.
- **Statement:** The contract says one model/split/window is a result and ensembles/per-type nulls are "never gates"; three other canonical sources still say such a number is only a draft.
- **Evidence:** AGENTS: "One model, one split, one window is enough for a result"; "Ensemble spread, subset curves and per-type nulls are optional extras, never gates." DECISIONS catalog row 30 "A decodability number is reported over a lag sweep, several ensemble members and several splits, against a per-type null | standing"; body: "no per-cell-type decodability figure is written into a report unless ... (b) taken from at least ten ensemble members". PROJECT_MAP: "A number read from one ensemble member, one split or one lag window is a draft, not a result." config.toml: "A reported number is read over `members`, never from this one alone"; "A reported number is a sweep over `lag_windows`". The later DECISIONS entry `DECISIONS.md:368-391` parks this rigour but entry 280 carries no Supersedes line.
- **Fix:** Mark DECISIONS 2026-09-18 (lag sweep/members/splits) as superseded by the 2026-09-18 night entry; rewrite PROJECT_MAP:193-195 and the two config.toml comments to the AGENTS rule.
- **Confidence:** confirmed.

### F3 - "Training ... never from scratch" is unscoped and forbids the queued steps
- **Where:** `AGENTS.md:106-107` vs `ROADMAP.md:424-446` (30), `ROADMAP.md:460-473` (29.2), `ROADMAP.md:485-491` (28), `ROADMAP.md:390-393` (31).
- **Statement:** AGENTS states that any training "starts from the transplanted weights, never from scratch", but every generator in the project (13B, prior17-29) was trained from initialisation and the next queued steps train new models (a renderer, a static flow) that have no transplanted weights.
- **Evidence:** AGENTS: "training, when it happens at all, starts from the transplanted weights, never from scratch." ROADMAP 30: "a conditional flow state -> picture ... Priced ... 15 to 50 minutes on one T4"; 31: "re-simulating the corpus states and retraining ... 13B, the state priors". The source of the rule (`DECISIONS.md:376-377`, `389-390`) is about fine-tuning the brain model only ("an optional fine-tune from the transplanted weights").
- **Fix:** Scope the sentence to the brain model (model zero / FlyVis parameters), e.g. "training of the brain model starts from the transplanted weights".
- **Confidence:** confirmed (text); intent of the rule plausible from DECISIONS.

### F4 - Where item 31 sits in the order on resume is ambiguous
- **Where:** `ROADMAP.md:19-23` and `ROADMAP.md:376-377` vs `ROADMAP.md:382-383` and `ROADMAP.md:575`.
- **Statement:** The header gives the resume order 29.3 -> 30 -> 29.4 -> 29.2 without 31, but the Queue - which by the file's own rule "is an order, not a list" - lists 31 first while 31 says it is "not yet ordered".
- **Evidence:** header: "the order is: 29.3 ..., then 30 ..., then 29.4 ..., then 29.2"; queue first entry "31. **Repair model zero's OFF pathway (ISS-0015)** - recorded by the human 2026-09-24 as a work item, not yet ordered against the static branch"; rule "**Queue is an order, not a list.**"; queue preamble "What remains on this branch is ... the renderer - and ... 29.2" (no 29.3/29.4). 31's own text says a fix forces retraining of what 30/29.2 would train, so the order is material.
- **Fix:** Get the human's word on 31's position; place it in the queue at that position (or under "Waiting, not in the order above") and add it to the header sentence.
- **Confidence:** confirmed.

---

## MEDIUM

### F5 - "A push is routine" vs "publishing is a gate" now that the repository is public
- **Where:** `AGENTS.md:137-141` vs `ROADMAP.md:359-367`.
- **Statement:** Since 2026-09-24 master is public, so a routine push publishes content, which AGENTS lists as a human gate; the pre-push check ROADMAP describes is not a rule anywhere.
- **Evidence:** AGENTS: "Human approval is required for ... publishing (a repository, weights, a preprint) ... A commit and a push after a finished step are routine, not a gate." ROADMAP P5: "master pushed (2bc4bfd..de4b182, fast-forward, new commits scanned for tokens first) and the repository made public"; "new commits are kept clean" (`ROADMAP.md:318`).
- **Fix:** Human decision: either keep pushes routine and add the token scan / "public text" check as a rule, or make a push to the public remote a gate.
- **Confidence:** confirmed (text); severity is a judgement.

### F6 - PROJECT_MAP and OPERATIONS_MAP are frozen at 2026-09-20: "the prior is not built", no volumes or commands for 17-29
- **Where:** `docs/PROJECT_MAP.md:10-21`, `:59-66`, `:116-137`, `:70-82`; `docs/OPERATIONS_MAP.md:4-9`, `:252-258`, `:266-269`; `README.md:119`; `DECISIONS.md:118-120`.
- **Statement:** Both canonical maps describe item 17 as the unbuilt current track; the whole prior line (PCA, flows, hex flow, corpus) is absent although the next steps (29.3, 30) read its files.
- **Evidence:** PROJECT_MAP: "**State on 2026-09-20**"; diagram "noise -> state prior (item 17, NOT BUILT)"; "The item-17 prior over states will reuse `gen13b`'s interpolant and backbone; it is not written yet." Code has `flydream/generate/prior17.py`, `pca19.py`, `hexflow23.py` and ~40 other modules; `flydream/data/video_corpus.py`, `video_hex.py` are not in the Data component list. OPERATIONS_MAP: "Item 17 ... its runs and prices are in `ROADMAP.md` and will be recorded here once measured"; volume layout lists only `/pairs13/ /train13/ /roundtrip13/ /gen13b/`, while ROADMAP:42, 61-67, 433-434 and `generate_app.py:536` (`np.load(Path(RUNS) / file)`) use `flydream-runs:/gen18/`, `/prior19/`, `/prior22/`, `/prior23/`, `/prior29/`. The public README says "Commands for every stage are in docs/OPERATIONS_MAP.md".
- **Fix:** Update both maps to the current implementation (state date, components, the prior line, corpus18, volume layout, the `modal_call.py` commands per step).
- **Confidence:** confirmed.

### F7 - Test count and test list stale in both maps
- **Where:** `docs/OPERATIONS_MAP.md:55`, `docs/OPERATIONS_MAP.md:280-281`, `docs/PROJECT_MAP.md:151-161` vs `ROADMAP.md:93`, `README.md:114`.
- **Statement:** The maps say 77 offline tests; collection finds 146, and five test files are missing from PROJECT_MAP's list.
- **Evidence:** `pytest --collect-only -q -p no:cacheprovider | tail -3` -> `146 tests collected in 5.67s`. OPS: "# 77 offline tests"; "77 offline tests passed on 2026-09-19". PROJECT_MAP: "Tests (`tests/`, 77 offline passed on 2026-09-19)" and a list lacking `test_hexflow23.py`, `test_pca19.py`, `test_prior19_latent_flow.py`, `test_resid22.py`, `test_step20_sampler_and_cuts.py`.
- **Fix:** 146 (with date of last full run) and the complete file list, or drop the number from the maps and point to ROADMAP.
- **Confidence:** confirmed (count by collection; pass status not re-run).

### F8 - DECISIONS entries superseded in practice but still "standing" with no Supersedes line
- **Where:** `DECISIONS.md:10-12` (its own rule) vs rows `:21`, `:22`, `:24`, `:31` and entries `:106-120`, `:122-141`, `:167-179`, `:506-547`.
- **Statement:** Several entries were replaced by later ones but are neither shortened nor marked, so a reader of the catalog gets contradictory "standing" choices.
- **Evidence:** (a) "Training, ensembles and inversion batches run on Modal; nothing trains locally | standing" vs the >=4x entry `:357-360` "member simulation stays local too" and AGENTS "Local is the default". (b) "The state prior moves to a VAE with a 2,048-dimensional latent | standing" vs 2026-09-21 `:551-553` "The state VAE's requirement that its own latent be N(0, I) is withdrawn". (c) "The order is data, model zero, training, ... dreams | preliminary" - replaced three times (`:199`, `:250`, `:373`). (d) "Decoding climbs a ladder ... diffusion only after [the three rungs] agree on the shape of the curve | standing", while ISS-0004 says that curve does not exist and the SiT/flow generator is the deliverable. (e) The lag-sweep entry, see F2.
- **Fix:** Add Supersedes lines and set Standing to "superseded by <date entry>" for (a)-(e); shorten as the file's rule says.
- **Confidence:** confirmed.

### F9 - Budget caps still stated as live rules
- **Where:** `docs/OPERATIONS_MAP.md:145-147`; `DECISIONS.md:376-377`, `:389-390`, `:427` vs `AGENTS.md:108-112`, `DECISIONS.md:601-623`.
- **Statement:** The no-budget rule says every earlier cap stays only as history "marked as withdrawn", but a canonical map states a cap as a rule and two DECISIONS entries carry unmarked caps.
- **Evidence:** OPS: "**When Modal at all (DECISIONS 2026-09-18, night):** ... a training proposal costs single dollars." DECISIONS 2026-09-18 night: "an optional fine-tune ... within $0.50"; "a training run within $0.50 and only from the transplant" (no withdrawn note, unlike `:342-343`). DECISIONS 2026-09-20: "the run fits the $0.50 training cap". DECISIONS 2026-09-21: "Earlier cap entries stay in this file as history, marked as withdrawn."
- **Fix:** Delete the clause from OPS:146-147; add the same italic "withdrawn 2026-09-21" note to the 2026-09-18 night and 2026-09-20 entries.
- **Confidence:** confirmed.

### F10 - A DECISIONS entry declares silence as approval
- **Where:** `DECISIONS.md:33`, `DECISIONS.md:405-412` vs `AGENTS.md:203-206`, `DECISIONS.md:7-8`.
- **Statement:** The 2026-09-20 "next stage" entry records an agent design as standing "until they say otherwise", which AGENTS forbids explicitly.
- **Evidence:** DECISIONS: "**Designed by the agent** (a draft the human has not ruled on line by line; it stands until they say otherwise)"; catalog "standing; sub-step design is the agent's". AGENTS: "Writing it into ... `DECISIONS.md` ... does not make it true, and neither does the human reading it without objecting; only an explicit yes does."
- **Fix:** Mark the agent-designed half as a draft/withdrawn (it is historical now - item 17 sub-steps are done) or record the human's explicit approval with its date.
- **Confidence:** confirmed.

### F11 - DECISIONS catalog is missing two entries that AGENTS cites
- **Where:** `DECISIONS.md:14-37` vs entries `DECISIONS.md:340` and `DECISIONS.md:368`; cited by `AGENTS.md:106` ("DECISIONS 2026-09-18, night") and `AGENTS.md:14-15`.
- **Statement:** The catalog has 20 rows for 22 entries; "Modal is used only for a measured >=4x speed-up or for a GPU" and "The deliverable is a working generator, not a paper" are not in it, and the catalog is not in date order despite `:10`.
- **Evidence:** `grep -n "^## 20" DECISIONS.md` -> 22 headings; catalog rows `:18-37` = 20; row `:31` (2026-09-20) precedes row `:32` (2026-09-18).
- **Fix:** Add the two rows (with Standing) and sort by date.
- **Confidence:** confirmed.

### F12 - ISS-0007 and ISS-0008 are fixed and verified by a rerun that is already recorded
- **Where:** `ISSUES.md:37-38`, `:267`, `:311`, `:321` vs `reports/2026-09-20_step18_5_width_and_the_representation_floor.md:769-782`, `:837-838`; `flydream/generate/samples17.py:108-113`.
- **Statement:** ISSUES says ISS-0008 awaits "a rerun of the conditional arm" and ISS-0007 is open, but 18.16 is exactly that rerun on the fixed code for both.
- **Evidence:** ISS-0008: "fixed in code 2026-09-20, **not yet verified by a rerun** ... **A rerun of the conditional arm is what would close this.**" Report 18.16: "То же плечо, что 18.12, на исправленном коде ISS-0008 (инициализация 0,02, таблица без затухания) и ISS-0007 (метка тянется только из 100 обученных классов из 110)... $0,26"; "механизм теперь исправен и доказуемо учится". Code: `pool = pmeta.get("trained_classes") or list(range(n_cls))`.
- **Fix:** Set both to `fixed`, Evidence -> the 18.16 section; move them to Closed, shortened.
- **Confidence:** confirmed.

### F13 - ISS-0004 and ISS-0006 describe work "being redone" that the roadmap stopped
- **Where:** `ISSUES.md:39`, `:359-360` (ISS-0006), `:41`, `:434-435` (ISS-0004) vs `ROADMAP.md:110-111`, `ROADMAP.md:553-555`; `reports/2026-09-18_step4_decoder_stack.md:135`; `AGENTS.md:38-40`.
- **Statement:** Both entries say a recomputation is in progress; no report or run-log record shows it, step 4 is Done, and the rigour it waits for is parked; ISS-0004's closing condition (ten members, five splits) is a gate AGENTS abolished.
- **Evidence:** ISS-0006: "the 80 ms and 160 ms windows of the sweep being recomputed"; ISS-0004: "the figure is withdrawn from the report and the map is being redone"; step-4 report: "ISS-0004 ... остаётся открытой до карты на десяти членах и пяти разбиениях". `grep lag reports/runs.jsonl` -> only lag-0 step-4 records. ROADMAP: "Paper-grade rigour ... measurements stopped 2026-09-18."
- **Fix:** Rewrite the Status lines to what is true (open, no rerun planned; parked with the rigour item), and drop the members/splits closing condition.
- **Confidence:** confirmed.

### F14 - OPERATIONS_MAP commands still use the aliasing lag windows, and one command is corrupted
- **Where:** `docs/OPERATIONS_MAP.md:66`, `:71`, `:75`, `:222` vs `config.toml:78-81`, `ISSUES.md:39`, `docs/PROJECT_MAP.md:112`.
- **Statement:** The documented sweep commands use even-spaced lags that ISS-0006 and config.toml forbid; the only `modal_call.py` example contains a literal tab.
- **Evidence:** OPS:66 `--lags 0 2 4 --run <id>   # one window of the lag sweep`; OPS:71 `--lags 0 2 4   # the ladder at an 80 ms window`; OPS:75 `--windows 0,0_2_4`. config.toml: "Consecutive lags only: ... a [0 2 4] decoder alternates between two regimes frame by frame (ISSUES ISS-0006)". `sed -n 222p docs/OPERATIONS_MAP.md | cat -A` -> `--out data\prior18^Irain.json$` (tools/modal_call.py:3 has `data/prior18/train.json`).
- **Fix:** Use `0 1 2 3 4` / `0_1_2_3_4`; replace the tab line with `--out data\prior18\train.json` (or forward slashes).
- **Confidence:** confirmed.

### F15 - Settings of everything after item 13 are code constants, not config.toml
- **Where:** `AGENTS.md:59-62`, `ROADMAP.md:42-43` vs `config.toml` (no section beyond `[generate.pairs13]`/`[benchmark]`) and code.
- **Statement:** AGENTS names "sampler steps" and clip-length thresholds as config settings, ROADMAP says the corpus settings are in config.toml, but the corpus band, sampler steps, PCA k, DCT K and every prior/flow hyperparameter live as function defaults or module constants.
- **Evidence:** `sample_steps: int = 20` repeated in `deploy/modal/generate_app.py:1554`, `:1779`, `flydream/generate/seed19.py:77`, `prompts14.py:140`, `samples17.py:60`, `baseline17.py:126`, `check18.py:69`, `classcmp18.py:52`, `clip17.py:37`, `noise17.py:77`; `SINTEL_BAND = {"motion": (0.008, 0.20), "contrast": (0.05, 0.45), "cut": (0.0, 12.0)}` at `flydream/data/video_corpus.py:41`; `pca19(k: int = 2048, ...)` `generate_app.py:710`; `train17(steps=20000, batch=32, lr=3e-4, width=128, ...)` `generate_app.py:1045-1052`. No module under `flydream/generate/{prior17,pca19,hexflow23,gen13b}.py` or `flydream/data/video_corpus.py` reads config.toml (grep). AGENTS: "never constants buried in code".
- **Fix:** Either add `[corpus18]`, `[generate.gen13b]`, `[generate.prior]` sections with reference and reason and read them, or narrow the AGENTS rule to what is actually kept in config and correct ROADMAP:43.
- **Confidence:** confirmed.

### F16 - The UCF101 training corpus has no source, manifest entry or licence record
- **Where:** `ROADMAP.md:40-43` vs `docs/OPERATIONS_MAP.md:107-123`, `data/manifest.json`, `flydream/data/sources.py`, `docs/PROJECT_MAP.md:70-71`, `AGENTS.md:187-189`.
- **Statement:** The dataset every current state comes from is not in the Data table, the manifest or `sources.py`, and its licence is recorded nowhere canonical.
- **Evidence:** manifest keys: `flyvis/SintelDataSet` and three `malecns/*.feather` only. `grep -i ucf flydream/data/sources.py docs/*.md` -> nothing; the only mention besides ROADMAP is `README.md:168` "from UCF101 (research use)". PROJECT_MAP: "`sources.py` is the only place a URL is written". AGENTS: "Licences are recorded per dataset". Also still open from OPS:121-122: `data/bridge_olmatching_raw.tsv` "to be moved into `sources.py`" - not done (`grep olmatching flydream/data/sources.py` empty).
- **Fix:** Add a UCF101 row (source URL, location `data/ucf101/`, size, licence/terms) to the OPS Data table and the URL to `sources.py`/manifest.
- **Confidence:** confirmed.

### F17 - ISS-0005: catalog and body disagree on status
- **Where:** `ISSUES.md:40` vs `ISSUES.md:388`, `:426-427`.
- **Statement:** The catalog says mitigated with (b) fixed; the body says open and still waits for v7.
- **Evidence:** catalog "mitigated (b fixed, a third-party) ... v9 DSI 0.152 against v7's 0.109"; body "**Status:** open."; Evidence "(the `capped_pairs` list once v7 lands)" although v7 and v9 exist (ISS-0003, runs.jsonl). ISSUES rule `:15-16`: `fixed` means "verified by a rerun"; `:19` "One defect per entry" - (a) and (b) have separate causes.
- **Fix:** Align the body with the catalog (or split (a) third-party and (b) fixed into two entries); drop "once v7 lands".
- **Confidence:** confirmed.

---

## LOW

### F18 - ISS-0015 cites a retracted number and the wrong roadmap item
- **Where:** `ISSUES.md:65-68`, `:77`, `:30` vs `ROADMAP.md:328-329`, `ROADMAP.md:382`, `docs/articles/part1_en.md:119-123`.
- **Evidence:** ISS-0015: "the weak T5 selectivity (DSI 0.01-0.03, article part 1 section 2)"; ROADMAP: "'T5 0.01-0.03 on every member' was wrong for member 000"; article now: "T5 on members 001 and 002 is barely selective (DSI 0.006-0.020), and on member 000 it is selective but points 49-144° away". Related "roadmap 2" while the repair is queue item 31.
- **Fix:** Quote the article's current numbers; Related -> roadmap 31.
- **Confidence:** confirmed.

### F19 - ARTEFACTS still puts figures in `reports/figures/`, which is now git-ignored; `docs/figures` and `docs/articles` are in no canonical doc
- **Where:** `docs/ARTEFACTS.md:62-68` vs `.gitignore` ("# working figures stay local; public figures live in docs/figures/" / `reports/figures/`), `ROADMAP.md:362`, `ROADMAP.md:320-322`; `AGENTS.md:151-153`.
- **Evidence:** ARTEFACTS: "`reports/figures/<date>_<substrate>_<experiment>[_<variant>].{gif,png}`"; "A committed gif stays small". ROADMAP: "Working figures are no longer committed." `docs/figures/*` names (`layers_activity.gif`, ...) follow no stated scheme. AGENTS forbids delegating edits to `docs/`, which now also holds the public articles and figures (plausibly unintended).
- **Fix:** Add to ARTEFACTS where public figures go (docs/figures, naming, committed) versus working figures (reports/figures, local); list docs/articles and docs/figures in PROJECT_MAP Boundaries; decide whether the delegation ban covers docs/articles.
- **Confidence:** confirmed (the delegation point plausible).

### F20 - `reports/runs.jsonl` was corrected by hand-editing a measured record
- **Where:** commit `87ee67b` vs `docs/OPERATIONS_MAP.md:196-197`, `AGENTS.md:213-214`, `AGENTS.md:178-179`.
- **Evidence:** `git show 87ee67b -- reports/runs.jsonl`: record `2026-09-18_step2_zero_R_v9_wk50m500oc / dsi_malecns_T4_member001` value `0.377` replaced in place by `0.369`. OPS: "written only by `tools/run_log.py`"; `tools/run_log.py` is append-only with no correction path. Disclosed in the commit message and ROADMAP:328, so not silent.
- **Fix:** Give run_log a correction mode (append a record that supersedes one by run+metric) or state in AGENTS how an arithmetic error in the log is corrected.
- **Confidence:** confirmed.

### F21 - Secrets: PROJECT_MAP describes a Modal secret that does not exist
- **Where:** `docs/PROJECT_MAP.md:210-212` vs `docs/OPERATIONS_MAP.md:156-157`, `:18-19`; `deploy/modal/*.py`; `env.example`.
- **Evidence:** PROJECT_MAP: "reach Modal through one secret published from it". OPS: "**Secret:** none needed for training ... neuPrint is never used on Modal." `grep -rn Secret deploy/` -> nothing. OPS: "`env.example` lists the names with no values" - it lists only `NEUPRINT_TOKEN=`.
- **Fix:** Align PROJECT_MAP with OPS (no Modal secret); either list the Modal/HF names in env.example or drop them from the sentence.
- **Confidence:** confirmed.

### F22 - config.toml keys with no reason or never read
- **Where:** `config.toml:39-42`, `:109-113`, `:86`, `:95` vs `AGENTS.md:59-62`.
- **Evidence:** `[flyvis] version` and `ensemble` are read by no code (`grep -rnE '"(version|ensemble)"'` in flydream/tools/deploy -> 0; only `root_dir` at `flydream/model/__init__.py:15`); the ensemble name is hard-coded as `"flow/0000/000"` defaults instead. No comment/reason on `[flyvis]` keys, `[generate] steps, lr, tv, dt, t_pre`, `[decode] seed, batch_size`.
- **Fix:** Read `[flyvis] ensemble` where the ensemble is named, or mark the keys as documentation; add the reference default and reason for the uncommented keys.
- **Confidence:** confirmed.

### F23 - ISSUES entries break ISSUES' own rules (status vocabulary, reproducibility, one defect)
- **Where:** `ISSUES.md:15-16`, `:19`; entries `:36`, `:37`, `:43-44`, `:340`, `:377`, `:174-178`.
- **Evidence:** allowed statuses are `open`, `mitigated`, `fixed`, `won't fix`; used: "fixed in code, rerun pending" (ISS-0008), "open, narrowed" (ISS-0009), "open, third-party input" (ISS-0001/0002). Reproduce points outside the repo: ISS-0007 "`python /tmp/lab.py` pattern", ISS-0006 "the script in the session transcript of 2026-09-18 night". ISS-0011's "seen again" is a different pair of functions (`geometry`), and there is now a third `geometry` at `flydream/generate/geom29.py:47` not listed.
- **Fix:** Normalise statuses (details in the body), give a repo command for each Reproduce, add geom29 to ISS-0011 or split it.
- **Confidence:** confirmed.

### F24 - `.claude/settings.json` backstop does not match the documented command forms
- **Where:** `.claude/settings.json` (deny `Bash(modal run:*)`, `Bash(modal deploy:*)`) vs `CLAUDE.md:6-7`, `docs/OPERATIONS_MAP.md:74-75`, `:207`, `:221-222`, `:237-245`, `ROADMAP.md:91-92`.
- **Evidence:** every documented priced call is `.venv\Scripts\python.exe -m modal run ...` or `tools\modal_call.py <fn>` (which calls `modal.Function.from_name(...).remote`), neither of which begins with `modal run`. CLAUDE.md: "`.claude/settings.json` duplicates some hard rules as a mechanical backstop."
- **Fix:** Add deny/ask patterns for `*python*.exe -m modal run*`, `*-m modal deploy*` and `*modal_call.py*`, or drop the backstop claim.
- **Confidence:** plausible (depends on Claude Code's prefix matching; the mismatch of strings is confirmed).

### F25 - ROADMAP bookkeeping: stale header date, P2/P3 without status, a report defect kept outside ISSUES
- **Where:** `ROADMAP.md:3`, `:345-351`, `:342-344`, `:193-194` vs `:515`; `ISSUES.md:9-12`.
- **Evidence:** "**Updated:** 2026-09-23" while P1/P4/P5 entries are dated 2026-09-24. P2 ("the final is English and longer, with three figures") and P3 carry no done/in-progress state although README.md has the three figures and `docs/figures/` holds 30 files. "Open elsewhere: the step-23 report labels its sharpness ceiling 71 as the full state, while both 66 and 71 trace to block runs" is a false claim in a report, which ISSUES says belongs in ISSUES; ROADMAP itself uses 71 (`:193-194`) and 66 (`:515`) for the ceiling.
- **Fix:** Update the date; state P2/P3 status; move the 66/71 labelling defect to ISSUES and make ROADMAP use one number.
- **Confidence:** confirmed.

---

## Checked and found consistent (no finding)
- ROADMAP's list of done items: every cited report and every cited module/tool exists.
- Modal app names (`flydream-train`, `flydream-decode`, `flydream-generate`), volumes (`flydream-data`, `flydream-runs`), `FLYDREAM_GPU` default `T4`, `tools/modal_call.py --app`, and every `generate_app.py` local-entrypoint flag named in OPERATIONS_MAP exist in `deploy/modal/`.
- ISS-0010, ISS-0011, ISS-0013, ISS-0014 file:line citations match the code at `f7bf74b`; ISS-0012's normalisation is still present (`tools/fig_pic23.py:50`), so "open" is right.
- Subagent/model rules agree across AGENTS:147-161, DECISIONS:322-338 and ARTEFACTS:50-51.
- Nothing canonical depends on `what_does_a_fly_dream_of.md` (only AGENTS:235 names it, as not-to-use).
- CLAUDE.md's list of canonical documents matches AGENTS "Records".
- DECISIONS 2026-09-20 closing "classes as a condition" is backed by 18.15/18.16 on the fixed code, not only by the voided 18.4e/18.12.

## Gaps
- The test suite was only collected, not run; "146 passing" (ROADMAP:93) is not re-verified.
- Modal volume contents were not listed (no network), so volume paths are checked against code only.
- `README.md` and `docs/articles/` were read only where a canonical doc points at them; their own numbers were not checked.
- `reports/` were read only to verify ISSUES/ROADMAP claims (step 4, step 18.16); report-internal consistency is out of scope.
- Permission-rule matching in F24 was not tested against the Claude Code harness.
- Prices in AGENTS:131-132 ($0.19/core-h, $0.024/GB-h, $0.59/h T4) were not checked against Modal's current pricing.
- ISS-0009's per-script fixed/unfixed list and ISS-0001/0002/0003 numbers were not re-derived.
