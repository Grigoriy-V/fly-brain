# Review B: `flydream/generate/`, inversion to step 17

Reviewer: subagent B (Opus), 2026-09-24. Read-only. Nothing was edited except
this file. Scope: `invert.py`, `gen13b.py`, `roundtrip13.py`, `pairs13.py`,
`prompts14.py`, `dreams.py`, `mix.py`, `learned.py`, `stimuli.py`,
`figures.py`, `baseline17.py`, `clip17.py`, `noise17.py`, `prior17.py`,
`samples17.py`, `__init__.py`, and `tests/test_generate.py`. I also read
`deploy/modal/generate_app.py` where it calls these modules
(`pairs13_videos`, `pairs13`, `train13`, `roundtrip13`, `maps13b`, `train13b`,
`sample13b`, `sample17`), because the recorded numbers of 13A/13B come
from there.

Checks were local CPU only. Each one took under 30 s and used
`.venv/Scripts/python.exe`. Scripts are in the session scratchpad, not the
repo. `tests/test_generate.py` passes: 26 passed in 5.2 s with
`-p no:cacheprovider`.

Checked and found correct, so not listed as findings:

- Flow-matching time direction and Euler steps in `gen13b.sample` and
  `prior17.integrate`, including `shift`. An oracle field
  `v = (c - x)/(1 - t)` lands on `c` to 6e-8, with and without guidance.
- CFG uses the all-zero mask, the same as the "none" training mode.
- The logit-normal t.
- EMA copy on save and load.
- `invert_batch` equals the single-task inversion (the test covers this).
- The network is frozen (`load_network` sets `requires_grad_(False)`; the video
  is the only Adam leaf).
- Per-frame PixCorr is taken over hexals (axis −1).
- The DCT matrix is orthonormal.
- `prior17.invert` inverts `integrate`.
- The hex neighbour masks (`clamp(min=0)` × valid) are correct, and
  `structured_noise`'s `nanmean` always has the self column.
- The 13B/prior training reads only frames `[:40]` (`_load_maps`,
  `maps13b` DCT `[:, :tf]`), so neither trained model is touched by B1/B2.

---

## B1 — high — pairs13 Sintel clips get their 5-frame "future" from a rotated copy of the same clip

**File:** `flydream/generate/pairs13.py:80-87` (`sintel_videos`)

**Statement.** The margin is taken from `ds[i+1]` whenever the names match.
In `AugmentedSintel(flip_axes=(0,1), n_rotations=(0..5))`, rows run
`product(chunks, flips, rotations)`, so `i+1` is nearly always the same
temporal chunk rotated by 60° or flipped. It is not the next chunk.

**Evidence.**

```python
if i + 1 < len(df) and df.iloc[i]["name"] == df.iloc[i + 1]["name"]:
    n2 = ds[i + 1]["lum"]  ...  nxt = n2.reshape(...)
v, _ = extend_clip(v, nxt, frames - len(v))
```

Command: `scratchpad/chk_margin.py` rebuilds the same dataset from the local
flyvis cache in 23 s. Output:

```
684   bamboo_2_split_00  temporal 57 flip 0 n_rot 0 ; 685 -> same chunk, n_rot 1
1473  market_6_split_01  temporal 122 flip 1 n_rot 3 ; 1474 -> same chunk, n_rot 4
pairs (i, i+1) with the same name: 2199 of which the same temporal chunk (a rotation/flip, not the future): 2079
684  r(last frame, margin frame used) = 0.19 | true next chunk row [696]: 0.901
1473 r(last frame, margin frame used) = 0.058 | true next chunk row [1485]: 0.628
```

So 2,079 of 2,268 Sintel clips in the 13A/13B pairs have a jump at frame
40. Their stored states in frames 40–44 respond to that jump.

**Failure scenario.** Every score that reads shard states past frame 39
carries the artefact:

- The 45-frame round trips of `train13` (`learned.round_trip_error`) and
  `sample13b` on held-out clips 684 and 1473.
- 13A's `LinearHexTemporal`/CNN, whose output frames 36–39 read state frames
  40–43 through `taps=5`.

Measured with `scratchpad/chk_rt_margin.py` on the saved
`data/train13/deep_*_roundtrip.npz`:

```
inversion 684  saved rt 0.0071 | share of squared error in the 5 margin frames: 0.996 | error vs pairs13 target / vs held-frame target: 188
inversion 1473 saved rt 0.0515 | share ... 0.998 | ... 457
cnn 684        saved rt 0.0141 | share ... 0.603 | ... 2.1
cnn 1473       saved rt 0.0669 | share ... 0.831 | ... 5.1
```

For 13B, `scratchpad/chk_13b_window.py` reproduced the recorded
`samples.json` values exactly and then scored the first 40 frames only.
`test_1473` goes from 0.0800 to 0.0241 and `test_684` from 0.0158 to 0.0089.
Affected records:

- `runs.jsonl` `2026-09-19_train13_deep`: CNN rt 0.029, inversion rt 0.009.
- `2026-09-20_gen13b_samples`: 0.031.
- The "held-out clips, mean of 8" row of
  `reports/2026-09-20_step13b_generative_decoder.md`.

**Fix.** Look the future up by
`(name, flip_ax, n_rot, temporal_split_index + 1)` in `arg_df`, else hold
the last frame. Rebuild `pairs13/videos.npz` and the shards, or score only
`[:frames]` (see B2). 13B and the priors need no retraining.

**Confidence:** confirmed by running.

## B2 — high — the round trip scores the 5 margin frames; the Adam-inversion reference is almost entirely that artefact

**Files:**

- `flydream/generate/roundtrip13.py:116-131` (`round_trip`), called with
  window `(0, T=45)` at `roundtrip13.py:199` and at `prompts14.py:224,250`.
- `flydream/generate/learned.py:233-250` (`round_trip_error`, no window at
  all).
- The inversion is truncated before re-simulation at `roundtrip13.py:180`
  and at `generate_app.py:341`.

**Statement.** The two sides of the comparison see different margins:

- **The generated or decoded video:** it is 40 frames, and the round trip
  extends it by holding the last frame for 5.
- **The target state:** it was simulated with a real continuation (the next
  Sintel chunk for clips A/B, the true 45-frame procedural clip, or B1's
  rotated copy).

The score still includes frames 40–44, so it measures a mismatch the
protocol builds in. The Adam inversion does fit all 45 frames, but its
fitted margin is discarded (`v[:frames]`) before the round trip. Its round
trip is therefore mostly the cost of that discard.

**Evidence.**

```python
vids = np.concatenate([videos, np.repeat(videos[:, -1:], margin, 1)], 1)   # held
...
errs.append(((st[:, :tg.shape[1], sel] - tg[:, :st.shape[1], sel]) ** 2).mean((1, 2)) / var_ref[t])  # window (0, 45)
```

Command: `scratchpad/chk_rt_proc.py` on procedural held-out clips. There the
target margin is genuine, so B1 plays no part:

```
inversion 4770 saved rt 0.0036 | share of squared error in frames 40-44: 0.929
inversion 5019 saved rt 0.0028 | share ... 0.933
inversion 7917 saved rt 0.0006 | share ... 0.980
cnn 4770 saved rt 0.0116 | share ... 0.391 ; cnn 5019 0.135 ; cnn 7917 0.170
```

13B on clip A, from `chk_13b_window.py`: 0.0279 at `(0,45)` against 0.0246
over the first 40 frames. `test_5019`: 0.0600 against 0.0498.

**Failure scenario.**

- The 13A report's "amortisation gap" (CNN 0.029–0.036 against inversion
  0.009, "3×") is understated. On the frames that are actually shown, the
  inversion's error is 13–40× smaller than recorded, while the CNN keeps
  61–97 % of its own.
- The roundtrip13 "inversion" column (0.000–0.028) has the same problem.
- Tables in 13A/13B/14 use `(0, 45)`, while samples17, clip17 and noise17
  use `(0, frames=40)`. So "0.006, the 13B-level number" in the 17.2 report
  and 13B's own 0.031 are on different scales.

**Fix.** Score `window=(0, frames)` everywhere. The margin exists to
constrain the last shown frames, not to be scored. Alternatively, extend
the generated video with the same margin source the target used. Then
re-derive the 13A/13B/14 tables.

**Confidence:** confirmed by running.

## B3 — high — 13B's (and 13A's) "shuffled-state control" round trip is measured against the unshuffled clip-A state; item 14 measures against the shuffled one

**Files:**

- `deploy/modal/generate_app.py:1854-1861, 1886-1887` (`sample13b`).
- `flydream/generate/roundtrip13.py:168-170, 195-198`, the same convention.
- By contrast, `flydream/generate/prompts14.py:207-209, 223-224`.

**Statement.** In `sample13b` the control job is
`add("clip_A", ..., maps=maps_of(st_sh), tag="control_shuffled")`, so
`state_of[key] == "clip_A"`. The round-trip target is clip A's true state.
`states["control_shuffled"] = st_sh` is written but never read.
`roundtrip13.run` likewise appends the unshuffled `tg` for the "shuffled"
video. `prompts14` scores the same kind of video against the shuffled
state. The documented "13B's 0.96 against item 14's 19.1 do not share a
scale" (`docs/PROJECT_MAP.md` Boundaries) is really two different targets.

**Evidence.** Command: `scratchpad/chk_shuffle_target.py`. It uses the saved
`data/gen13b/samples.npz` video `control_shuffled__full__s1__seed0` and the
same permutation (`default_rng(0)`):

```
13B control video vs clip A's state (what sample13b records): 0.948
same video vs the shuffled state it was drawn from (item 14's definition): 19.614
```

**Failure scenario.**

- The 13B report says it is "30× under the shuffled control", with round
  trip defined as "per-type squared error to the target".
- `runs.jsonl` `2026-09-20_gen13b_samples` records `control: 0.957`.
- The 13A roundtrip13 "shuffled" column (2.86–6.39) has the same issue.

In every other row, the round trip is the distance to the state that was
given to the decoder. In the control row it is not. The control therefore
answers a different question ("how far does a scrambled state's video land
from the true state") from the one the table asks.

**Fix.** Score the control against the state it was drawn from
(`state_of` → `"control_shuffled"`, `tgs.append(shuffled_tg)`), or label the
column for what it measures. Keep one definition across 13A/13B/14.

**Confidence:** confirmed by running.

## B4 — medium — the "shuffled state" permutes cells across types, so per-type statistics are not kept

**Files:**

- `flydream/generate/dreams.py:110-116` (`shuffled`, used on multi-type
  stages).
- `roundtrip13.py:168` (`cells_cond` = all 2/8/14 condition types).
- `prompts14.py:207-208`.
- `generate_app.py:1854-1857`.

**Statement.** One permutation is drawn over the concatenated cells of all
types. T4 cells receive T5 values and the reverse. The two descriptions of
the control are therefore wrong for multi-type stages:

- The dreams docstring: "what any state with these statistics gives".
- Item 11: "numbers kept, cell-to-place map destroyed".

**Evidence.** From `chk_shuffle_target.py`:
`fraction of shuffled cells that keep a cell of their own type: 0.128`.
In `data/prompts14/summary.json`, `c_shuffled` `direction_state.per_type`
shows T4a −1.74, T4b −1.74, T4d +1.47 above grey, against −0.22 / −0.10 /
+0.09 for the unshuffled `e_A`. The type means move by more than 1 raw unit,
and 13B's per-type sd is 0.7–0.9.

**Failure scenario.** The control is unreachable for a trivial reason (wrong
type means). Item 14's 19.1 is dominated by those mean shifts, not by the
destroyed spatial layout. The T4+T5 column of item 11 and the deep/all
conditions of 13A compare against a much easier null than stated. By
contrast, samples17 and noise17 permute columns (`[:, :, perm(721)]`), which
keeps per-type statistics, so their "shuffled" rows are not the same control.

**Fix.** Permute within each type, as `samples17` does across columns, or
state the cross-type construction in the reports.

**Confidence:** confirmed by running.

## B5 — medium — the "8 held-out clips" behind r 0.966 are 2 natural clips (with sibling shots in training) and 6 procedural ones

**File:** `deploy/modal/generate_app.py:293`
(`rt_ids = test.index[::len//8][:8]`), reused by `sample13b`. The split
itself is `flydream/generate/pairs13.py:98-111`.

**Statement.** Because the test set is sorted, the stride picks
684 (bamboo_2), 1473 (market_6), 4770/5019/5268 (dots) and
7917/8166/8415 (texture). The held-out scenes have same-location sibling
shots in training: bamboo_1, cave_2, market_2, market_5 and temple_2 are
all in train/val.

**Evidence.** From the local manifest:
`rt_ids [684, 1473, 4770, 5019, 5268, 7917, 8166, 8415]`. The Sintel scenes
in train+val include `bamboo_1`, `cave_2`, `market_2`, `market_5` and
`temple_2`. Per-clip r in `data/gen13b/samples.json`: Sintel 0.975 and
0.969, mean 0.972; procedural mean 0.965; overall 0.9664.

**Failure scenario.** The number itself is not inflated, since the Sintel
clips score higher. But README and `docs/articles` say "r 0.966 on held-out
clips", and that rests on n = 2 natural clips from locations the model has
trained on.

**Fix.** State the composition next to the number, hold out by location
(strip the trailing `_N`), and draw the round-trip clips per source.

**Confidence:** confirmed from the data.

## B6 — medium — noise17 `--corpus` measures video novelty against the wrong training set

**File:** `flydream/generate/noise17.py:175-177`

**Statement.** In corpus mode the clips, the prior and the prior's training
set are all the item-18 corpus. The nearest-training-video search still runs
over the pairs13 bank: Sintel plus procedural, with
`manifest["split"]["train"]`. `samples17.py:82-93` does switch the bank.

**Evidence.**

```python
bank, labels = video_bank(manifest, procedural_file, T, dt, log=log)
train_idx = np.asarray(manifest["split"]["train"])
```

`data/prior18/noise18_local.json` (prior `corpus_dct16.pt`, clips
Basketball / ApplyEyeMakeup) gives:

```
clip_A 0.637 proc edge ; mix_0.5 0.64 proc edge ; B_through_noise 0.512 proc flow ; shuffled_A 0.537 proc flash
```

**Failure scenario.** The slerp mixtures are states with no source clip.
AGENTS requires that they carry the distance to the nearest training video
of the model that produced them. The recorded value comes from a set the
prior never saw. The 18.3 report does not quote it, so no published number
is wrong yet.

**Fix.** Mirror `samples17`'s corpus branch: `bank = cz["videos"]`,
`split = pairs_manifest["split"]`.

**Confidence:** confirmed by running.

## B7 — medium — the 13B report compares round trips on two different normalisers

**Files:** `flydream/generate/learned.py:248-250` normalises by the variance
of the target set. `roundtrip13.py:130` normalises by clip A's variance
(`var_ref`).

**Statement.** The two columns sit side by side on the "held-out clips" row
but are on different scales:

- 13A's "CNN rt" and "inversion rt" (0.029 / 0.009, from `train13`) are
  normalised by the per-type variance of the 8 test targets.
- 13B's value in the same row (0.031, from `sample13b`) is normalised by
  clip A's per-type variance.

**Evidence.** `scratchpad/chk_var_scale.py` gives the ratio of test-set
variance to clip-A variance per type:

```
T4a 0.60, T4b 1.03, T4c 1.02, T4d 0.94, T5a 1.32, T5b 1.35, T5c 1.33, T5d 1.64
```

**Failure scenario.** "Its round trip equals the 13A CNN's (0.031 vs 0.029)"
and "3× the Adam inversion's" compare across scales (on top of B1/B2).

**Fix.** Re-score the saved 13A predictions with `roundtrip13.round_trip`
and clip A's `var_ref`, or retire one of the two functions.

**Confidence:** confirmed by running.

## B8 — medium — 13A's held-out reconstruction r has no null control; the run log records a ceiling as its "control"

**Files:** `deploy/modal/generate_app.py::train13` (only `val_r`, `test_r` and
the round trip are computed) and `reports/runs.jsonl`
`2026-09-19_train13_{deep,early,all}`.

**Statement.** The first AGENTS principle asks for a time-shuffle or
shuffled-state control beside a reconstruction number. `test_r_deep_cnn`
0.96 is recorded with `"control": 0.999`, which is the Adam inversion's r.
That is an upper reference, not a null. roundtrip13's shuffled control
covers only the item 11–12 states, not the 1,996 test clips.

**Fix.** Run the saved decoders on the test maps with a time-permuted (or
within-type column-permuted) state. Record that as the control and the
inversion as the ceiling.

**Confidence:** confirmed by reading.

## B9 — low — procedural clips are seeded with Python's salted `hash()`, so `procedural_800_s0.npz` cannot be regenerated from its seed

**File:** `flydream/generate/pairs13.py:44`

**Evidence.** The call is
`rng = np.random.default_rng([seed, hash(cls) % (2 ** 31), k])`. Two runs of
`python -c "print(hash('dots'), default_rng([0, hash('dots')%2**31, 0]).random())"`
printed:

```
-8999023055987707467 0.9082...
-4165902565539886308 0.4763...
```

`PYTHONHASHSEED` is set only in `flydream/train/benchmark.py`. Under Windows
spawn, each `Pool` worker also gets its own salt.

**Failure scenario.** The "s0" in the file name and the `seed = 0` in
`config.toml [generate.pairs13]` do not reproduce the set. Anyone
regenerating the procedural half gets different clips, so different states
and splits by content.

**Fix.** Use `CLASSES.index(cls)` or `zlib.crc32(cls.encode())` in place of
`hash(cls)`.

**Confidence:** confirmed by running.

## B10 — low — settings that AGENTS requires in `config.toml` are buried in code

**Files:**

- `gen13b.py:34` (`MASK_MODES`), `:229` (`steps=20`), `:248` (EMA 0.999),
  `:276` (AdamW settings).
- `prompts14.py:140`, `samples17.py:60`, `clip17.py:37`, `noise17.py:77`,
  `baseline17.py:126`: `sample_steps: int = 20`.
- `roundtrip13.py:177,184`: the inversion's `lr=0.05, tv=0.02,
  plateau_steps=20, plateau_tol=0.01` are duplicated rather than read from
  `[generate]`.
- `prompts14.py:159`: clips `3` and `10` are hard-coded instead of
  `[generate.mix] sample_a/b`.
- `generate_app.py:1582,1804`: `margin, frames = 5, 40`.

**Statement.** AGENTS names "sampler steps" explicitly as a setting that
must live in `config.toml` with its reason. `config.toml` has no section for
13B or the priors.

**Failure scenario.** A change to `[generate]` or `[generate.mix]` silently
skips these paths. The 13B sampler step count that every table depends on
has no recorded reason.

**Fix.** Add `[generate.gen13b]` (sample_steps, guidance, mask modes, EMA)
and read it from these modules. Read the inversion settings through
`settings()`.

**Confidence:** confirmed by reading.

## B11 — low — `sample17` (Modal) takes its "held-out clips" from 8 flips/rotations of one clip

**File:** `deploy/modal/generate_app.py:1598`

**Statement.** `te = np.where(np.isin(idx_all, z["test"]))[0][:n_clips]`
takes the first 8 sorted test indices. Those are 684–691, all the chunk
`bamboo_2_split_00` at temporal index 57 under different flips and
rotations. `sample17` also calls `R.sample`, not `R.sample_states`, so a DCT
prior would be sampled in the wrong space.

**Failure scenario.** The "real clip" reference row and the diversity
figures are built on one clip. No report quotes this function's output: the
17.x tables come from the local `samples17.py`, which draws across sources.

**Fix.** Use `rng.choice` per source, as `samples17` does, and call
`sample_states`.

**Confidence:** confirmed by reading and from the data.

---

## Gaps

- I did not re-derive the 13A test r (0.927/0.960) with B1 fixed. Only the
  round trips were re-measured. B1's effect on r is likely small, since it
  touches only frames 36–39 of 24 % of the clips, but it is not measured.
- I did not re-score the item-14 or roundtrip13 tables at window `(0, 40)`.
  Only 13B's clip A, 4 test clips, the shuffled control, and 13A's saved
  predictions on 5 test clips were re-scored.
- B5's "sibling shot" claim rests on Sintel naming (same location and
  characters). I did not measure how similar the frames are across siblings.
- `generate_app.py` was read only where it calls my modules. I did not
  review `multi_init13b` (`:1754` has the same `[:, :40]` truncation
  pattern as B2) or `train17`.
- `stimuli.clip` physics (speeds, polarity conventions) and `prompts14`'s
  direction-label mapping (`ROT90`, the grating direction versus FlyVis
  a–d) were not checked against FlyVis ground truth.
- The `AugmentedSintel` identity-augmentation claim (`augment=True` with
  default jitter) was taken from the code comment and not re-verified.
  B1's check used the same dataset settings as pairs13, so B1 is unaffected.
- The tests do not cover `roundtrip13.round_trip`, `learned.round_trip_error`,
  `pairs13.sintel_videos`'s margin, the shuffle helpers' type-preservation,
  or `prompts14`'s state edits. B1–B4 would all have passed the current
  suite.
