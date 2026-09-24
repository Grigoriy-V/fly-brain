# E. Figure scripts and public figures: review notes

Reviewer E, 2026-09-24. Scope: `tools/fig_*.py`, `tools/gh_style.py`,
`tools/orphan_figures.py`, `tools/pick_static_examples.py`, the 30 files in
`docs/figures/` and the articles/README that embed them; `reports/figures/`
for missing/large/unreferenced files. Read-only. Every public figure script was
re-run with `--out`/`--out-dir` pointed at
a temporary folder outside the repository and compared with the
committed file.

## Reproduction check (the baseline for the findings)

| public figure(s) | script | regenerated from local `data/` |
|---|---|---|
| decoding_by_type.{gif,png} | fig_gh_decoding.py | byte-identical |
| inversion_by_layer, two_brains, state_to_video, seed_known_and_random (gif+png), connectome_export.png | fig_gh_inversion / two_brains / render / seed / connectome | byte-identical |
| first_priors, flow_inversion, seed_shell (gif+png), levers_and_judge, hex_flow, bought_kurtosis, static_chain | fig_gh_part2.py | byte-identical |
| static_draws.png | fig_gh_static.py | byte-identical (prints r 0.939, flat 25.3 %, as the caption says) |
| dsi_by_version, decoder_window, inversion_chart, speedups, round_trips, closed_loop | fig_gh_charts.py | same content, pixels differ (anti-aliasing; 1.8k-28k pixels in the plot area; side-by-side crops identical by eye) |
| layers_activity.{gif,png} | fig_gh_layers.py | documented command fails; byte-identical only with undocumented `--activity data/figures/act_s3_malecns_flyvis.npz --prefix flow/00_` (E2) |

Every public file has a generating script and every article caption names the
right script. Numbers checked against `reports/runs.jsonl` and the saved data
with no mismatch: the DSI values (v5 0.034/0.026/0.000 = 0.020, v7 0.101/0.150/0.075 = 0.109,
v9 0.200/0.191/0.065 = 0.152, FlyVis 0.391), the inversion r 0.933-1.000 and controls
-0.200..-0.221, eye-noise r (T5a 0.310, Tm5a 0.525), the speed-ups (948/176 s, 7e-4,
9.2/14.0/14.4 samples/s), round trips and loop, the levers (0.142 ... 0.0224, floor 0.0209,
real clip 0.0119 against blank 0.0070), hex_flow (4.11/5.36, 20/6.43 %, 19/72°, 0.809/0.884,
11/22 of 66), fixdraw arms, the static chain (crutches 0.257/0.351/0.137/0.045, ceiling 0.941
recomputed from `data/prior23/resfix23.npz`: 0.941; LS 0.941; flat 27.1/27.1/19.5 %, coherence
0.928/0.949/0.492), connectome (496 of 571, rank 0.692, sign 0.951, 96.9 % of 13,267),
`LINEAR = {R1 1.00, Mi1 0.99, T4a 0.83}` (regenerated 1.000/0.988/0.827), `HELD_OUT_13B = 0.966`
(run log note of `2026-09-20_gen13b_samples`), seed figure 0.485 / 0.52 / 0.884, 25.3 %.

## Findings

### E1 - high - the public figures cannot be regenerated from the public repository

- **Where:** `docs/articles/part1_en.md:491-501`, `docs/articles/part2_en.md:592-598`,
  `README.md:120-124`; `.gitignore:16` (`/data/`); `AGENTS.md:176`.
- **Statement:** every one of the 30 public figures reads derived artefacts under the
  gitignored `data/` (inversion runs, 13B samples, prior draws, `act_s3_malecns_flyvis.npz`,
  a 214 MB `pairs.npz`, 1 GB `corpus18/videos.npz`, trained `.pt` checkpoints) that are not
  published, not listed in any tracked manifest, and not mapped to the step that makes them;
  the articles say the figures "are built locally by the scripts in `tools/` from saved runs".
- **Evidence:** `git ls-files | grep -i manifest` prints nothing; `data/manifest.json` exists
  locally but is under `/data/` and holds only four raw inputs (Sintel and three MaleCNS
  feather files) - none of `act_s3_malecns_flyvis`, `2026-09-24_flyvis_invert`,
  `gen13b/samples`, `prior23/seed23`, `prior29/draw29`, `noise17_local`, `pca_ab2048`,
  `decode_sintel_lag`, `malecns_invert_*_s3` is in it.
- **Failure scenario:** a reader runs `uv run python tools/fig_gh_part2.py` on a clone and
  gets `FileNotFoundError: data/baseline17/baseline17.npz`; there is no pointer to what
  produces it or what it cost (several inputs need a GPU training run on Modal).
- **Fix:** state in the "Reproducing" sections that the scripts need the owner's `data/`, and
  add a tracked table (figure -> input files -> producing command/run id -> size/hash), or
  publish the small derived inputs (most are 1-12 MB npz/json) as a release asset.
- **Confidence:** confirmed by reading (manifest contents printed; scripts' inputs listed).

### E2 - high - README hero figure `layers_activity` cannot be rebuilt by the documented command, and a wrong flag mislabels it silently

- **Where:** `tools/fig_gh_layers.py:39` (`--activity` required, no default), `:42`
  (`--prefix` default `""`), `:43` (label default "the FlyVis reference network, member 000");
  `docs/articles/part1_en.md:496` (`uv run python tools/fig_gh_layers.py  # fig. 3`).
- **Statement:** the documented command exits with an argparse error; the figure is
  reproduced only with `--activity data/figures/act_s3_malecns_flyvis.npz --prefix flow/00_`,
  and the input is written by `tools/cmp_layers_flyvis.py`, which no doc names for this figure.
- **Evidence:** `fig_gh_layers.py: error: the following arguments are required: --activity`;
  with the two flags above the gif and png are byte-identical to `docs/figures/`. The npz holds
  both `malecns_<type>` and `flow/00_<type>` keys; `--prefix malecns_` would draw model zero
  under the hard-coded footer "the FlyVis reference network, member 000".
- **Failure scenario:** someone regenerates the README hero with `--prefix malecns_` (the
  default of the sibling `fig_hook_levels.py`) and publishes MaleCNS activity labelled as FlyVis;
  or cannot regenerate it at all.
- **Fix:** default `--activity` to the cache and `--prefix` to `flow/00_`, derive the footer
  label from the prefix, and put the full command (plus `cmp_layers_flyvis.py`) in the
  reproduce block.
- **Confidence:** confirmed by running.

### E3 - medium - `decoding_by_type` and its caption state the decoder window backwards (t and t-1; the code uses t and t+1)

- **Where:** `tools/fig_gh_decoding.py:34` (`LAGS = (0, 1)`), `:6` ("now and one step back"),
  `:111` (footer "ridge on frames t and t−1"); `docs/articles/part1_en.md:188`,
  `part1_ru.md:181`; `flydream/decode/pairs.py:59-62, 73`.
- **Statement:** `Pairs.frames` adds the lag to the stimulus frame (`a[:, t0 + lag : t1 + lag]`,
  "a positive lag lets the decoder see the response after the stimulus frame"), so lags (0, 1)
  are activity at t and t+1, not t-1.
- **Evidence:** pairs.py docstring and slicing quoted above; Fig. 4 of the same article
  (`decoder_window`) correctly labels the axis "decoder window after the frame".
- **Failure scenario:** a reader takes Fig. 5 as a causal (past-only) readout and compares it to
  Fig. 4; the two figures of one section contradict each other.
- **Fix:** footer "activity at t and t+1 (20 ms after the frame)"; same in both captions and the
  docstring.
- **Confidence:** confirmed by reading.

### E4 - medium - `decoding_by_type` shows reconstructions of a clip the decoder was trained on, beside held-out r, without saying so on the figure

- **Where:** `tools/fig_gh_decoding.py:44` (`--sample 3`), `:10-12` (docstring admits it),
  `:111` (footer "r over the held-out scenes"); caption `part1_en.md:185-190`.
- **Statement:** clip 3 (scene alley_1) is in the training split, so the pictures are in-sample
  predictions while the numbers under them are held-out scores; neither the figure nor the
  article caption says this.
- **Evidence:** `P.split_by_group(pairs.groups, 0.2, 0)` -> `3 in train True, 3 in test False`
  (144 train / 45 test samples).
- **Failure scenario:** the deep-type panels (T4a, T5a) look better than a held-out clip would,
  so the figure overstates what a linear readout recovers - the opposite of the caution in the
  surrounding text.
- **Fix:** pick a held-out clip for the pictures (and the inversion figure beside it can keep
  clip 3), or print "clip A is a training scene; r is over held-out scenes" on the figure.
- **Confidence:** confirmed by running.

### E5 - medium - `flow_inversion` labels a spherical interpolation "a straight line"

- **Where:** `tools/fig_gh_part2.py:133` (row subtitle "a straight line");
  `docs/articles/part2_en.md:143`, `part2_ru.md:139` ("a straight line between the noises").
- **Statement:** the mixes were made with `slerp`; the run's report explicitly rejects the
  straight line.
- **Evidence:** `flydream/generate/noise17.py:149` `R.slerp(eps_a, eps_b, float(al))`;
  `reports/2026-09-20_step17_3b_noise_inversion.md:74` "mixed on the sphere (a straight line
  would shrink the radius by up to 1/√2 and leave the typical set)"; run log note of
  `2026-09-20_prior17b_noise_inversion` "slerp mixes".
- **Failure scenario:** a reader reproduces the double exposure with linear interpolation and
  gets a different (off-shell) result; the public text describes a method that was not run.
- **Fix:** "an arc on the sphere (slerp)" in the figure and both captions.
- **Confidence:** confirmed by reading.

### E6 - medium - the FlyVis half of `two_brains` (and the article's "0.971") has no run record and no report

- **Where:** `tools/fig_gh_two_brains.py:44, 97-98`; `tools/run_flyvis_ladder.py`;
  `docs/articles/part1_en.md:225-228` ("0.971 for FlyVis against 0.933"), `:504` ("Every
  number in the article leads to a step report ... and to a record in runs.jsonl").
- **Statement:** runs `data/generate/2026-09-24_flyvis_invert_<type>_s3` exist locally and the
  figure reproduces, but no line of `reports/runs.jsonl` and no report names them.
- **Evidence:** `grep -rln "2026-09-24_flyvis\|run_flyvis_ladder" --include=*.md --include=*.jsonl`
  -> only the two tools; meta.json values FlyVis R1 0.9987, L1 0.9988, Mi1 0.9985, Tm9 0.9967,
  T4a 0.9879, T5a 0.9712. `run_flyvis_ladder.py` docstring also promises an `s27` variant
  "shown beside decoding" that does not exist on disk.
- **Failure scenario:** the one number that separates the two brains (T5a 0.971 vs 0.933) cannot
  be traced to a run id in the public record the article points to.
- **Fix:** add a run-log record (`tools/run_log.py`) for the FlyVis ladder and a line in the
  step report; drop the stale `s27` sentence.
- **Confidence:** confirmed by reading.

### E7 - medium - README hero alt text presents the FlyVis reference network as "the model"

- **Where:** `README.md:7` ("Activity of eight cell types of the model while the eye watches a
  clip"), under a headline about wiring MaleCNS; `docs/figures/layers_activity.*`.
- **Statement:** the hero shows the FlyVis network (FIB connectome), drawn from FlyVis on
  purpose because model zero's OFF pathway is broken (ISS-0015, commit ebc8d8c); the figure
  footer says so in small type, the article caption says so, the README does not, and the
  pipeline strip highlights "brain model".
- **Failure scenario:** a README reader takes the clean ON/OFF maps as a property of the MaleCNS
  model - exactly the claim ISS-0015 says does not hold for T5/Tm9/L3.
- **Fix:** alt text and a one-line caption "FlyVis reference network (the MaleCNS build is
  compared in part 1, section 2)".
- **Confidence:** confirmed by reading.

### E8 - medium - `reports/figures/` is gitignored but its 208 files (518 MB) are still tracked and on origin

- **Where:** `.gitignore:37` (`reports/figures/`, "working figures stay local"); commit 82c431f
  "working figures no longer committed"; `docs/ARTEFACTS.md:64-67`.
- **Statement:** ignoring does not untrack: HEAD and `origin/master` still carry 208 working
  figures (518 MB, 34 of them over 5 MB, largest 24.6 MB), while every new working figure is
  now silently left out of commits, so new report links will be dead on GitHub.
- **Evidence:** `git ls-files reports/figures | wc -l` -> 208; `git ls-tree -r --name-only
  origin/master reports/figures | wc -l` -> 208; `size-pack: 786.61 MiB`.
- **Failure scenario:** the next report embeds `reports/figures/2026-09-2x_...png`; it renders
  locally and 404s on GitHub; meanwhile the public clone is ~0.8 GB.
- **Fix:** decide one way (human gate: history rewrite): either `git rm --cached -r
  reports/figures` and move cited figures to a release/LFS, or drop the ignore line; update
  ARTEFACTS.md to match.
- **Confidence:** confirmed by running.

### E9 - medium - published post-1 video prints linear-decoder scores under inversion pictures and claims the picture is lost with depth

- **Where:** `tools/fig_hook_post1.py:46-49` (`LINEAR`), `:123-127`, `:131` ("Linear decoder: a
  plain regression from the same layer loses the picture with depth."); `posts/post1_video.png`;
  ROADMAP P4 "post 1 published".
- **Statement:** the pictures in columns 1-3 are inversion outputs (r 0.999/1.000/0.996), but
  the only numbers under them are the ridge decoder's held-out r (1.00/0.99/0.83); the footer's
  claim contradicts the article ("a linear readout does weaken with depth, but that does not
  mean the picture is lost ... We do not claim a monotonic loss", `part1_en.md:199-202`).
- **Evidence:** rendered frame read; values traced to `fig_gh_decoding.py` output.
- **Failure scenario:** a viewer reads "linear decoder: 0.83" as the score of the T4a picture
  shown, and takes away a claim the article disowns.
- **Fix:** put the inversion r under the inversion pictures (or show the ridge output if its
  r is printed), and reword to "weakens with depth".
- **Confidence:** confirmed by reading (the output is untracked in `posts/`).

### E10 - low - `orphan_figures.py` counts a figure's own generating script as a citation, hiding 66 files (99 MB) no text refers to

- **Where:** `tools/orphan_figures.py:20, 31-36, 64, 80-81`.
- **Statement:** the corpus includes every tracked `.py`, so a figure whose only mention is its
  script's default output path is classed "cited ... linked from text that stays public".
- **Evidence:** run to temp: `cited 161 (368 MB), scripted-only 29 (118 MB), orphan 18 (32 MB)`.
  Recomputed with `.md`+`.jsonl` only: 66 of the 161 "cited" (98.7 MB) are named only by code,
  e.g. `2026-09-20_malecns_guidance18.png` (only `tools/fig_guidance18.py`), `probe18`,
  `pipeline18`, `steps18`, `weight18`, `new384`, `trunc18`, `stage22`, `seedclip18`,
  `promptclip18`, `labelclip18`, `2026-09-19_malecns_generator_two_inputs.gif` (12 MB).
- **Failure scenario:** a history clean-up based on `reports/orphan_figures.md` keeps ~100 MB it
  meant to drop.
- **Fix:** exclude `tools/fig_*.py` (or all `.py`) from the "cited" corpus, report "named only by
  its script" as its own group.
- **Confidence:** confirmed by running (with `--out` to temp).

### E11 - low - Fig. 7 caption says the bands are the data's range; the code draws mean ± 1 sd

- **Where:** `tools/fig_gh_part2.py:252-255`; `docs/articles/part2_en.md:407-408`,
  `part2_ru.md:390`.
- **Evidence:** `axhspan(mean - sd, mean + sd)`; `fixdraw23.json` kurtosis 8.31 ± 1.17
  (band 7.15-9.48) while the range is 6.28-10.75 - the two red arms (7.11, 7.17) sit inside
  the range but on/under the band.
- **Fix:** caption "bands: mean ± sd over 40 training subsamples", or draw min-max.
- **Confidence:** confirmed by reading.

### E12 - low - `state_to_video`: the control column is a training clip under a footer "held out of training"

- **Where:** `tools/fig_gh_render.py:47-49, 87, 91`; caption `part1_en.md:377-380`.
- **Evidence:** clip A (Sintel sample 3) is in 13B's training split
  (`data/pairs13/manifest_summary.json` split.train contains 3); `fig_hook_post1.py:17-18`
  says so; the figure prints "true state r = 0.99" there.
- **Fix:** footer "clip A is a training scene" or use a held-out clip for the control.
- **Confidence:** confirmed by reading.

### E13 - low - footers name run ids that are not in the run log

- **Where:** `tools/fig_gh_part2.py:13, 139` ("run 2026-09-20_prior17_noise_inversion"; the
  record is `2026-09-20_prior17b_noise_inversion`), `:118` and caption `part2_en.md:121-122`
  (`2026-09-20_prior17_*` does not match the 17.1b runs `prior17b_*`);
  `fig_gh_charts.py:199-200` and `fig_gh_inversion.py:224` name data folders
  (`2026-09-19_malecns_invert_*_s3`, `..._dream_eye_noise_*`) whose run-log ids are
  `2026-09-19_generate_malecns_s3_40f5` / `2026-09-19_dream_malecns_eye_noise`, and whose
  `meta.json` "tag" says `2026-09-18_malecns_invert_R1_s3`; `decode_sintel_lag_*`
  (Fig. 4) has no run-log record (only the step 4 report).
- **Fix:** print the run-log id (or both) on the footer; correct the 17b id.
- **Confidence:** confirmed by reading.

### E14 - low - numbers printed on public figures that exist only as figure-time computations

- **Where:** `fig_gh_decoding.py:68` (r 1.00/1.00/0.99/0.83/0.83/0.71, controls 0.53-0.63,
  also quoted in `part1_en.md:193-196`), `fig_gh_inversion.py:177` (r to B 0.94-1.00, quoted in
  `part1_en.md:221-222`), `fig_gh_connectome.py:46` (rank 0.69, `part1_en.md:79`),
  `fig_gh_part2.py:135` (r 0.85, `part2_en.md:145`).
- **Statement:** all regenerate exactly, but none has a run-log record, against the articles'
  "every number ... leads to ... a record in runs.jsonl".
- **Fix:** log them once with `tools/run_log.py` (the figure run is free and local).
- **Confidence:** confirmed by running.

### E15 - low - slow-down stated wrongly or not at all on public clips

- **Where:** `fig_gh_inversion.py:223`, `fig_gh_layers.py:213`: `int(round(50 / fps))` prints
  "6× slower" at 8 fps (6.25×, the example in `docs/ARTEFACTS.md:43-44`);
  `fig_gh_two_brains.py:41` fps 6 (8.3×) with no slow-down on the footer; `state_to_video`,
  `seed_known_and_random`, `first_priors`, `flow_inversion`, `seed_shell` state none.
- **Fix:** `f"{50 / fps:.2g}× slower"` on every clip footer.
- **Confidence:** confirmed by reading.

### E16 - low - hard-coded copies of run values, and numbers parsed out of free-text notes

- **Where:** `fig_gh_part2.py:220` (20.0 %), `:222` (0.884), `:223` (66.0), `:153-155`
  (303.8, 0.71, 92,288), `:183` (0.0209), `:269-270` ("data 35.5", "data 8.31"),
  `:309` (the "real held-out state" flat-field bar is `st["value"]`, i.e. the draw's own value);
  `fig_gh_seed.py:191-194` (0.52, 0.88); `fig_gh_two_brains.py:75` (DSI 0.15 vs 0.39);
  `fig_gh_connectome.py:110-111` (96.9 %, 13,267); `fig_gh_charts.py:229-230` (948 s, 176 s,
  7e-4 in the subtitle beside bars read from the log). Regex on notes: `fig_gh_charts.py:207`,
  `fig_gh_part2.py:192-193, 284-293`.
- **Statement:** all match today (checked above); but notes are edited in place (the v9 note
  says it was corrected 2026-09-24, the prior23_accept note 2026-09-22), and line 309 is right
  only because the note says the real state reads exactly the draw's 0.271.
- **Fix:** read each from its record/json; store the real-state flat fraction as its own field.
- **Confidence:** confirmed by reading.

### E17 - low - duplicated helpers have diverged

- **Where:** `tools/fig_hook_levels.py:43-71` re-implements `gh_style.font`/`honeycomb`: seam
  threshold 0.10 vs 0.08, no `pix < 8` guard (gh_style: "below this the seams alias into
  stripes"), no DejaVu fallback (bitmap default font off Windows); `fig_hook_post1.py` imports
  these copies. `corr` is redefined in `fig_gh_{decoding,inversion,part2,render,seed}.py`,
  `fig_hook_post1.py`, `fig_generator_two_inputs.py` (nanmean in one, mean elsewhere).
- **Evidence:** rendering T4a at 7 px with both versions and at 16 px showed the diagonal
  stripes in post 1 are in the data, so no visible defect today.
- **Fix:** import from `gh_style`.
- **Confidence:** confirmed by reading.

### E18 - low - `fig_hook_levels.py` writes into `docs/figures/` by default

- **Where:** `tools/fig_hook_levels.py:3, 88` (`--out docs/figures/hook_levels`, gif+png+mp4).
- **Statement:** no `hook_levels.*` is accepted or referenced; a bare run drops an unreviewed
  public figure and an mp4 into the public folder.
- **Fix:** default to `posts/` like `fig_hook_post1.py`.
- **Confidence:** confirmed by reading.

### E19 - low - public figures depend on Windows fonts

- **Where:** `tools/gh_style.py:34-42`, `fig_gh_charts.py:55`; every subtitle is hand-wrapped
  for Segoe UI metrics at W = 1216.
- **Statement:** off Windows the fonts fall back to DejaVu Sans (wider), so regenerated figures
  change layout and long subtitle lines can run past the right edge; even on this machine the
  six part-1 charts regenerate with pixel differences.
- **Fix:** ship or pin a font file, or check text width against W.
- **Confidence:** plausible (off-Windows not run); pixel differences confirmed by running.

### E20 - low - large files

- `docs/figures/seed_shell.gif` 6.4 MB (the only public file over 5 MB; 40.6 MB for the 30 files).
- 33 tracked working figures over 5 MB, largest
  `2026-09-19_malecns_generator_two_inputs_40f.gif` 24.6 MB, `..._dream_eye_noise.gif` 18.0 MB,
  `..._dream_neuron_noise.gif` 17.1 MB, `..._train13_deep.gif` 15.7 MB (see E8).

### E21 - candidates, not defects: dead scripts and unreferenced public files

- Figure scripts whose output no report, doc, ROADMAP or run-log record names (only the script
  itself): `fig_guidance18`, `fig_probe18`, `fig_pipeline18`, `fig_steps18`, `fig_weight18`,
  `fig_new384`, `fig_trunc18`, `fig_stage22`, `fig_seedclip18`, `fig_promptclip18`,
  `fig_labelclip18`, `fig_new_video18` (only `reports/orphan_figures.md`). `fig_hook_levels`
  (ROADMAP calls it "the first draft", superseded by `fig_hook_post1`).
- `docs/figures/*.png` twins of the gifs (decoding_by_type, first_priors, flow_inversion,
  inversion_by_layer, layers_activity, seed_known_and_random, seed_shell, state_to_video,
  two_brains): referenced by no markdown.
- Markdown -> figure references: none missing (113 link/path references resolved and all
  brace-list names expanded and found; no reference points at an untracked file).

## Gaps

- The ~60 working figure scripts were checked statically only (default input paths exist for
  every fixed file name; f-string names not resolved), for hard-coded title numbers in a sample
  of them (`fig_hexcov21` matched its record), and not re-run; their figures' numbers were not
  traced one by one.
- The DSI chart footer "Moving-edge protocol of FlyVis, 12 directions" and the article table's
  PD errors (39.2° vs 39.1 from the per-member note) were not verified.
- The rendering of the part-1 charts off Windows (E19) was not tried.
- Whether 13B's training split is keyed by the same index as Sintel sample 3 was inferred from
  `manifest_summary.json` and the script docstring, not from the training code.
- `reports/figures/` content was checked for existence and size, not for correctness.
