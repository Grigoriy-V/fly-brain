# G. Public surface and the evidence behind it

Reviewer G, 2026-09-24. Scope: `README.md`, `LICENSE`, `docs/articles/part{1,2}_{en,ru}.md`,
`reports/*.md`, `reports/runs.jsonl`. Read-only; nothing in the repository was edited
except this file. Scratch scripts lived in the session scratchpad only.

## Checks run (and what came back clean)

- Secrets scan of the tracked tree (`git grep -nIE` for JWT/hf_/sk-/ghp_/ak-/as-/AKIA/PEM,
  `token|secret|password|api_key = <long value>`, e-mail addresses, absolute paths
  `D:\`, `C:\Users`, `/d/ML`, `AppData`, scratchpad, Modal profile/workspace names,
  `modal.com/apps`): **no credential in the tree or in history.** `env.example` holds an
  empty `NEUPRINT_TOKEN=`; `.claude/settings.json` is harmless. Only e-mail in the tree
  is a paper author's public address in a research note. The two private items found
  are in G1 and G17. No path bytes in figure files; no profanity (word-bounded scan).
- Links: a resolver over every markdown link/image in README, the four articles and all
  37 reports: **141 local links, all resolve to tracked files**; the one fragment
  (`step9_window_margin.md#item-10-the-ladder-batched-same-day`) resolves. Every
  figure file named in a report exists and is tracked (208 on disk = 208 tracked).
- `runs.jsonl`: 151 lines, all parse, all ten schema fields present in every record, no
  unknown fields, agents valid, no duplicate `(run, metric)` pair. Total cost 12.624 USD
  (README "about $13" holds); records 63-151 sum to 9.274 (part 2 "about $9.3" holds);
  records 1-62 sum to 3.35 (part 1 "a few dollars" holds).
- RU/EN parity: all numbers compared after normalising decimal commas and thin spaces;
  the only residual differences are numerals written as words ("Sixteen" / "16",
  "eightfold" / "в 8 раз", "Part 2"). Claims read section by section: parallel.
- Key numbers traced and confirmed: 60 types (`data/ol/filters_R_w5wk50m500oc.json`
  has 60 nodes), 31,526 units, 721 columns, 1.35 M edges (records 32, 33); DSI v5 0.020,
  v7 0.109, v9 0.152, FlyVis 0.391; v9 direction error 39.2 (step-2 report line 87);
  flash 0.922; member 001 T4 0.369 vs 0.611, 5.5 vs 9.8 deg (record 34); inversion
  0.93-1.00 with control -0.20 to -0.22 (step-9 report table); 13B 0.966 / 0.03 /
  0.031 (record 60); seed table +0.969 / +0.957 / -0.060 (record 106); 73 sigma
  (records 103, 118); kurtosis 4.11 / 5.36 / 8.31 (records 137, 141); static flow
  27.1 / 27.1 / 19.5 and 0.928 / 0.949 / 0.492 (record 150); 146 offline tests
  (`pytest --collect-only`: 146).
- Reproduced locally from saved data (scratchpad script using
  `flydream.decode.pairs/ridge`, same split and lags as `tools/fig_gh_decoding.py`):
  model-zero decoding at lags (0, 1), r / time-shuffled control: R1 1.000 / 0.596,
  L1 1.000 / 0.633, Mi1 0.988 / 0.594, Tm9 0.834 / 0.622, T4a 0.827 / 0.533,
  T5a 0.715 / 0.591. The article's 1.00 / 0.99 / 0.83 / 0.83 / 0.71 and control
  0.53-0.63 are correct (but unlogged, G12). Rank correlation of the current export
  recomputed from `data/ol/filter_comparison_R.csv`: 0.692 on 496 pairs, sign 0.951 on
  493 (article "0.69" correct, unlogged). T5a lag (0, 1) control-corrected:
  0.6397 - 0.1445 = 0.495 (article correct, unlogged). Part 1 section 9 two-seed means
  recomputed from `data/prompts14/summary.json`: white 1.85, swaps/rotations
  2.17-9.53, reversal 1.01 / 5.46, amplify 0.68, compositions 0.099-0.143, direction
  match 2 of 6 - all as the article says.

## Findings

### G1 - high - public Git history still carries the Modal workspace name, a Modal app URL and private local paths
- Where: history of `origin/master` (public since 2026-09-24); removed from HEAD by
  `82c431f` ("Before going public: no personal Modal profile, no paths to other
  repositories ...") but every earlier commit is still reachable. Not a credential.
- Evidence: `git log --all -p -S'<profile>'` returns 11 commits whose diffs contain the
  Modal profile / workspace handle (kind: account handle), e.g. removed lines
  "Account: the owner's second Modal account, profile `<redacted>`" and
  "- Приложение: https://modal.com/apps/<redacted>/main/ap-<redacted>" (kind: Modal
  dashboard URL with workspace and app id). `git log --all -p -G'<other repository names or user path>'`
  shows added lines in `2e0c6fa`, `d2fb25b`, `1380f15`, `d506343`, `df2e315` with
  absolute paths into two other private repositories of the owner
  (kind: names of other private repositories), `C:\Users\<user>\AppData\Local\Temp\claude\D--ML-Fly-Brain\<session-uuid>\scratchpad\...`
  (kind: Windows user path + agent session ids) and `"file": "D:/ML/Fly_Brain/.venv/Lib/site-packages/flyvis/..."`.
  The secret-pattern scan of the full history found no token.
- Fix: the owner decides whether this exposure is acceptable. If not: rewrite history
  (`git filter-repo --replace-text` on those strings) and force-push - a destructive,
  externally mutating action behind the human gate; nothing needs rotating because no
  credential leaked.
- Confidence: confirmed.

### G2 - high - README and part 2 overstate GPU utilisation and understate per-run price
- Where: `README.md:34` ("one T4, 98-99.9 % utilisation per run"), `README.md:106`
  ("98-99.9 % utilisation, $0.05-0.51 per run"), `docs/articles/part2_en.md:511-512`
  ("98-99.9 % T4 utilisation in every training run"; "$0.05-0.51 per run"),
  `part2_ru.md:491-492` (same).
- Evidence (runs.jsonl): training runs at 34 % (line 117, `2026-09-21_prior19_flow2048`),
  41 % (line 119), 94.7 % (line 95), 95.0 % (line 84), 97.5 % (lines 76, 97); non-training
  runs 38-82 % (lines 46, 56, 57, 60, 69, 72, 74, 75). Costs above $0.51: $1.00 (line 52),
  $0.85 (112), $0.82 (111), $0.54 (93), $0.52 (97) - the last four are part-2 runs.
  The part-2 table row itself says "37.4 -> 99.9 % at step 23", contradicting "every".
- Fix: "up to 99.9 % in packed training runs (34-99.9 % across all GPU runs); $0.01-1.00
  per run", or name the runs the range covers.
- Confidence: confirmed.

### G3 - high - part 2 compares the VAE with a PCA number from a different clip set, which the run log itself flags as not comparable
- Where: `docs/articles/part2_en.md:257-259` ("from a clean code the VAE gives 0.687,
  while plain PCA of the same dimension gives 0.890"), `part2_ru.md:247-249`; source
  `reports/2026-09-20_the_seed_problem.md:280, 308, 315` ("0,687 против 0,890 у линейной
  PCA на тех же 2 163 измерениях").
- Evidence: runs.jsonl line 116 (`2026-09-21_prior19_pcaval`): "Control is the learned
  encoder-decoder at the same 2,163-dimensional latent ... measured on the SAME six
  held-out clips with the same code path. The linear stage wins by 0.138 ... Caveat: 18.22
  reported 0.890 for PCA-2048 but on a different clip set with a different ceiling
  (0.977); the like-for-like comparison is the one in this record". Line 110 (0.890)
  is on 2,400-state PCA with ceiling 0.977; the VAE's 0.687 is on six clips with ceiling
  0.952. PCA had 2,048 components, not 2,163. The conclusion (PCA beats the VAE) stands;
  the gap is 0.138 (0.825 vs 0.687; 87 % vs 72 % of ceiling), not 0.203.
- Fix: articles: "plain PCA-2048 gives 0.825 on the same six clips" (and annotate the
  seed-problem report lines 280/308/315 with the later correction).
- Confidence: confirmed.

### G4 - high - part 1 says a composed prompt gives "video that exists in no clip" without a nearest-training-video distance
- Where: `docs/articles/part1_en.md:430-432` ("give plausible video that exists in no clip
  (round trip 0.10-0.14)"), `part1_ru.md:409-411` ("видео, которого нет ни в одном
  клипе"); source wording in `reports/2026-09-20_step14_controllable_generator.md`
  ("no such video existed").
- Evidence: AGENTS.md: "a generated video only against the distance to its nearest
  training video (without that second number the claim is 'generated without a source
  clip', not 'a video that exists in no clip')". Record 61 (`2026-09-20_prompts14`) and
  `data/prompts14/summary.json` hold round trip and direction only - no nearest-video
  number for any prompt.
- Fix: "gives plausible video generated without a source clip", or measure nearest(r)
  against the training bank beside a held-out clip's.
- Confidence: confirmed.

### G5 - high - the type-mask gain is reported against the wrong arm
- Where: `README.md:102` ("'type not given' instead of 'type is zero' ... T4 alone
  0.837 -> 0.904"), `docs/articles/part2_en.md:520` and `part2_ru.md:500` (same pairing).
- Evidence: `reports/2026-09-21_state_restoration.md:216-219`: T4 with honest mask 0.904;
  "T4, T5 обнулён молча" (type is zero under a full mask) **0.889**; "T4, T5 достроен по
  PCA" **0.837**. The mask's own effect is 0.889 -> 0.904 (+0.015), not +0.067. Part 2
  section 7 (`part2_en.md:286-287`) states it correctly ("against 0.837 for completing
  the state by PCA").
- Fix: README and both section-12 rows: "0.889 -> 0.904" (or reword to "honest mask vs
  PCA completion: 0.837 -> 0.904").
- Confidence: confirmed.

### G6 - high - "60x faster, bit-for-bit equal to FlyVis" for the eye renderer overstates both halves
- Where: `README.md:99`; `docs/articles/part2_en.md:162-165` and `:515`;
  `part2_ru.md:157-159` and `:495`.
- Evidence: `reports/2026-09-20_step18_corpus_of_ordinary_video.md:57-64`: "eye render |
  5.46 s (measured under load) / ~2.3 s | **0.09 s**" in a table headed "one thread, idle
  machine"; like-for-like is about 2.3 / 0.09 = 25x (60x mixes a loaded before with an
  idle after). Same report: "the sampler matches `BoxEye` to 1.7e-6" and the Sintel
  check is "r = 1.000000, max |diff| 2.6e-4" - equal to tolerance, not bit-for-bit.
- Fix: "~25x faster (60x against the loaded baseline); equal to FlyVis's BoxEye within
  1.7e-6".
- Confidence: confirmed (the ~2.3 s idle figure is the report's own estimate).

### G7 - medium - README calls 13B "a one-step generator"; it samples in 20 Euler steps
- Where: `README.md:27` ("A one-step generator from brain state to video"); also
  `README.md:12` ("in a single generative pass").
- Evidence: `docs/articles/part1_en.md:370-372`: "give all 40 frames of a video in 20
  Euler steps". To an ML reader "one-step generator" means one network evaluation.
- Fix: "an amortised generator (20 Euler steps, no per-target optimisation)".
- Confidence: confirmed.

### G8 - medium - README generalises the static-flow match to "brain states"
- Where: `README.md:15-16` ("a flow on the hexagonal lattice of the eye learns the
  distribution of brain states well enough that its draws match real states in
  structure"), `README.md:30-31` ("fresh draws match real states in structure").
- Evidence: the match is record 150 only: the 721 x 2 static flow (29), on two judges,
  through a least-squares renderer that blurs, on a target with kurtosis 3.17 vs 2.98
  Gaussian (record 150 note). The video-state flow on the lattice (23) does not match:
  kurtosis 5.36 vs 8.31 +- 1.17 (record 141), sharpness 22 vs ceiling 66 (record 142).
- Fix: "a flow over the static part of the T4 state draws pictures that match real
  states on two structural judges through a blurring renderer".
- Confidence: confirmed.

### G9 - medium - README's static-scene row uses the figure's six-clip numbers, not the run record, and drops the control
- Where: `README.md:52` ("draws match real states on structure (flat field 25.3 % vs
  25.3 %)").
- Evidence: record 150 (`2026-09-21_prior29_static_ab`, 64 draws): 0.271 vs real 0.271,
  control N(0, I) 0.195. 25.3 % comes from `docs/figures/static_draws.png` (six clips,
  "real states: PCA-2048 reconstruction"); the part-2 Fig. 9 caption
  (`part2_en.md:499-502`) explains the difference, README does not. The figure's footer
  attributes its 25.3 % to run `2026-09-21_prior29_static_ab` without that caveat.
  (`static_chain.png` shows 27.1 / 27.1 / 19.5, matching the record.)
- Fix: "flat field 27.1 % vs 27.1 % for real states, 19.5 % for the no-flow control".
- Confidence: confirmed.

### G10 - medium - README decoding row states a result without its control
- Where: `README.md:48` ("retina and lamina r 1.00; motion detectors 0.71-0.83").
- Evidence: the time-shuffled control is 0.53-0.63 for every type (part1_en:194;
  reproduced: T5a 0.715 vs 0.591, T4a 0.827 vs 0.533), i.e. T5a's margin is 0.12. Every
  other row of that table carries its control.
- Fix: append "(time-shuffled control 0.53-0.63)".
- Confidence: confirmed.

### G11 - medium - "a linear readout does not [get it back]" conflicts with the project's own linear decoder on the deep group
- Where: `README.md:80-82`, `README.md:25-26` ("where a linear decoder loses it with
  depth").
- Evidence: part1_en:357-358 and record 52: the 13A **linear** model (shared hex-temporal
  convolution) reaches r 0.927 on T4a-d + T5a-d over 1,996 held-out clips. The per-type
  ridge of step 4 is what weakens with depth. The 13A numbers have no time-shuffle
  control (part1_en:365), so part of 0.927 may be scene-level.
- Fix: "a per-type linear readout weakens with depth".
- Confidence: plausible.

### G12 - medium - numbers in part 1 have no run record, although both articles say every number has one
- Where: claim at `part1_en.md:504-506`, `part2_en.md:14-15, 601-603` (and RU).
  Unlogged numbers: model-zero decoding at lags (0, 1) - `part1_en.md:192-197` and
  `README.md:48` (only `tools/fig_gh_decoding.py` computes them); FlyVis T5a inversion
  0.971 - `part1_en.md:226-227` (only `data/generate/2026-09-24_flyvis_invert_T5a_s3/meta.json`,
  a 2026-09-24 CPU run with no record or report); rank correlation 0.69 -
  `part1_en.md:79` (only `tools/fig_gh_connectome.py`); lag-window 0.057 -> 0.625 and
  0.495 - `part1_en.md:170-176` (`data/decode/2026-09-18_decode_sintel_lag_*`, no
  record); part 1 section 9 two-seed means (record 61 holds seed-0 values: 1.86,
  "0.7-9.6", 0.100-0.150).
- Evidence: reproduced values in "Checks run" above - all numbers are right; the runs
  are local-only and unlogged.
- Fix: log each with `tools/run_log.py` (runs `2026-09-18_decode_sintel_malecns_v9`
  lags 0-1, `2026-09-24_flyvis_invert_*_s3`, `2026-09-18_decode_sintel_lag_*`,
  the step-1 export's current rank correlation).
- Confidence: confirmed.

### G13 - medium - part-1 figure captions cite run ids that are not in runs.jsonl
- Where: `part1_en.md:217, 323` and `part1_ru.md:209, 305` (`2026-09-19_malecns_invert_*_s3`);
  `part1_en.md:324`, `part1_ru.md:306` (`2026-09-19_malecns_dream_eye_noise_*`).
- Evidence: these are local data-directory names (gitignored); the logged ids are
  `2026-09-19_generate_malecns_s3_40f5`, `..._batch20` (lines 45-46) and
  `2026-09-19_dream_malecns_eye_noise` (line 47). The inversion meta files even carry
  tag `2026-09-18_malecns_invert_T5a_s3` (renamed, step-9 report line 33).
- Fix: cite the logged run ids in the captions.
- Confidence: confirmed.

### G14 - medium - the audit's "ceiling 71" mixes objects; only the step-23 report reconciles it with 66
- Where: `reports/runs.jsonl` line 139 ("the renderer's own ceiling is 71 on flat
  fraction, the PCA-1536 stage 23, a fresh draw 11"); `reports/2026-09-21_the_draw_distribution.md:186, 198-201`
  ("потолок не 100, а 71"; "23 против 71 ... ступень отдаёт две трети").
- Evidence: `reports/2026-09-21_step23_hex_local_prior.md:134-137`: 71 is the **full**
  state, 66 the **T4a+T4b block** that the 23 / 11 rows measure; record 142 says 66.
  The audit therefore compares a block number with a full-state ceiling. The articles
  use 66 (`part2_en.md:334, 376`), which is the consistent figure.
- Fix: annotate record 139 and the audit report's section 9 with "71 = full state;
  block ceiling 66".
- Confidence: confirmed.

### G15 - medium - run record 30 gives the current export's edge count as 1.39 M; records 32-33 say 1.35 M
- Where: `reports/runs.jsonl` line 30 (`2026-09-18_step3_timing_cpu`: "filters_R_w5wk50m500oc
  (31,526 nodes, 1.39M edges)"); `reports/2026-09-18_step3_training_options.md:27`
  ("1,39 млн рёбер").
- Evidence: line 32 and line 33 (v9, same export): "1.35M edges"; 1.39 M belongs to v8
  `wk50` without the mass floor (line 31). README and part 1 use 1.35 M.
- Fix: correct the record by an appended correction; fix the step-3 report.
- Confidence: confirmed (which export the CPU timing actually used is not verified).

### G16 - medium - licences: UCF101 and figure imagery are not covered
- Where: `LICENSE` (MIT over "the Software and associated documentation files"),
  `README.md:164-170`, `data/manifest.json`, `docs/OPERATIONS_MAP.md:107-116`.
- Evidence: UCF101 is recorded only in `reports/2026-09-20_step18_corpus_of_ordinary_video.md:25-26`
  ("research use"); it is absent from `data/manifest.json` (keys: Sintel + three MaleCNS
  files) and from the OPERATIONS_MAP data table. `docs/figures/` ships UCF101 frames
  of identifiable people (`static_draws.png` "face", "person", "eye") and Sintel frames
  (`layers_activity.gif`, inversion figures) inside an MIT repository with no carve-out;
  README gives no Sintel attribution (the film is (c) Blender Foundation, CC BY 3.0;
  the MPI-Sintel dataset has its own research terms, recorded in the manifest).
  Cell-type bridges (`flyconnectome/ol_annotations`) are "per repo", licence unrecorded.
  FlyWire is not used directly (`flywireType` is a MaleCNS column), so its absence from
  README is correct. MaleCNS (CC-BY 4.0) and FlyVis (MIT) are named.
- Fix: a "Data and figure licences" note: figures derived from UCF101 / Sintel are not
  MIT and carry their sources' terms; add UCF101 to the manifest and OPERATIONS_MAP;
  Sintel attribution.
- Confidence: confirmed (licence terms of UCF101/Sintel stated from general knowledge,
  not re-fetched - no network).

### G17 - low - a Modal app id remains in the current tree
- Where: `reports/2026-09-18_training_optimization_bench.md:132` (kind: Modal app id
  `ap-...`, no workspace name).
- Fix: drop the id or accept it (not a credential; useless without the workspace).
- Confidence: confirmed.

### G18 - low - run ids misdate runs, and run ids lack the config hash the rules ask for
- Where: `reports/runs.jsonl` lines 57-67 (`2026-09-20_gen13b_*`, `prompts14*`,
  `baseline17`, `prior17*`) with `"date": "2026-09-19"`; line 114.
- Evidence: the records naming 2026-09-20 were committed 2026-09-19 08:58 +0700
  (`5626fb2`), so the id's date is a day late. Only 2 of 121 run ids carry a config hash
  (`..._229624b2a4`, `..._b5023898e9`) against AGENTS.md "a run name with the date and
  the config hash". From line 57 on, `experiment` and `metric` become free-text
  sentences (16 experiment fields over 200 characters), so the log is not
  machine-groupable by metric.
- Fix: note the convention in `tools/run_log.py`; keep metric ids short and move prose
  to `note`.
- Confidence: confirmed.

### G19 - low - runs.jsonl was edited in place, bypassing the append-only writer
- Where: commits `87ee67b` (line 34), `9d449ca` (lines 140-141), `1922536` (a duplicate
  line removed).
- Evidence: `git show --numstat`: 1+/1-, 2+/2-, 0+/1-. The edits carry an in-note mark
  ("This record said 0.377 ... until 2026-09-24"; "said 'fivefold' until 2026-09-22"),
  so they are not silent; `tools/run_log.py` presents itself as "The only writer".
- Fix: append correction records (same run, metric suffixed `_corrected`) or document
  the edit-with-mark policy in `run_log.py`.
- Confidence: confirmed.

### G20 - low - part 1's benchmark equivalence number differs from the run record
- Where: `part1_en.md:294-296`, `part1_ru.md:276-277` ("a difference of up to 4.3e-4
  where two reference runs differ by 3.0e-4").
- Evidence: record 44 value 0.00026 (head, stats_relu vs baseline r0; 1.4e-4 vs r1) against
  3.0e-4 baseline self-difference; 4.3e-4 is `comparisons.json`'s max over all final
  states (bench report line 161). Conservative, not an overclaim, but the article's
  number is not the logged one.
- Fix: "2.6e-4 on the head (up to 4.3e-4 over all final states)".
- Confidence: confirmed.

### G21 - low - the 13A shuffled control is placed beside numbers from a different set
- Where: `part1_en.md:358-361`, `part1_ru.md:340-342` ("round trip on eight held-out
  clips ... A control with shuffled cells gives 2.9-4.5").
- Evidence: the shuffled column exists only in `reports/2026-09-19_step13a_amortised_inversion.md:116-130`
  (run `2026-09-19_roundtrip13`, states of items 11-12), 2.86-4.51 on reachable states,
  with 0.47 (dark) and 6.39 (T5 x 0) outside the quoted range.
- Fix: "(on the item 11-12 states, run 2026-09-19_roundtrip13)".
- Confidence: confirmed.

### G22 - low - README calls the 15,514-clip bank "training clips"
- Where: `README.md:61-62`.
- Evidence: record 142: "nearest of 15,514 is 0.485"; the bank is the whole corpus
  (13,555 train + 713 val + 1,246 test, record 115); record 139: "Against the training
  split alone a real clip reads 0.518". Conservative direction.
- Fix: "its nearest of the 15,514 corpus clips".
- Confidence: confirmed.

### G23 - low - part 2's DCT energy figure is the Sintel number, not the corpus one
- Where: `part2_en.md:516`, `part2_ru.md:496` ("at 99.55 % of the energy").
- Evidence: record 74: corpus `dct16_energy_kept` 0.9932, control (Sintel) 0.9955;
  record 81 "99.32 % for K=16". Every part-2 prior from 18 on is on the corpus.
- Fix: "99.3 % on the corpus (99.55 % on Sintel)".
- Confidence: confirmed.

### G24 - low - `reports/figures/` is both tracked and ignored
- Where: `.gitignore` (comment "working figures stay local; public figures live in
  docs/figures/", followed by the pattern `reports/figures/`).
- Evidence: 208 files tracked, 208 on disk, every report reference resolves - a fresh
  clone is complete today, but any new report figure will silently not be committed and
  the report citing it will break.
- Fix: decide: un-ignore, or state in the reports' convention that new report figures
  are local-only.
- Confidence: confirmed.

### G25 - low - who-said-what and agent process in the public tree
- Where: `reports/runs.jsonl` (e.g. line 100: "The human rejected my semantic explanation
  and said the training was written wrong"), step reports, `AGENTS.md` (verbatim Russian
  quotes of the owner), `deploy/modal/decode_app.py:20` (a verbatim chat quote).
- Evidence: README:138-141 discloses `reports/` and the canonical documents as a lab
  notebook with the owner's instructions; the articles and README are clean (grep for
  human/agent/Claude/Codex/first person returned nothing in the articles).
- Fix: owner's call; the code comment is the one place outside the disclosed notebook.
- Confidence: confirmed.

## Gaps

- Did not rebuild any figure (writing `docs/figures/` was out of scope); figure numbers
  were checked against run records and saved data, and one figure (`static_draws.png`)
  and one chart (`static_chain.png`) were viewed.
- Did not recompute the edge count of `filters_R_w5wk50m500oc` (G15), the 13A/13B
  numbers, or any part-2 training metric from checkpoints; they were traced to records
  and reports only.
- The 60x/25x renderer comparison (G6) rests on the report's own "~2.3 s" idle estimate;
  not re-timed.
- Report-vs-record agreement was checked for the numbers the public texts use and for
  spot samples; not every number in the 37 reports was traced.
- Citations (Berg et al. Cell 2026; Bauer, Margrie & Clopath eLife RP 105081; Chen et al.
  PLOS CB 2024) and dataset licence terms were not verified online (no network).
- Did not check whether the public GitHub repository has other branches, tags or
  releases beyond `origin/master`, or whether GitHub caches the pre-`82c431f` blobs
  in forks.
- Only the 146-test collection was checked, not a test run.
