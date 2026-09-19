# Step 18.0 / 18.2: a corpus of ordinary video, and what 13B does with it

Date 2026-09-20. Agent: Claude (Fable). Asked for by the human, 2026-09-20:
"собрать большой dataset преимущественно из разнообразных обычных видео,
оставив procedural только небольшой частью", with the order set the same day —
13B is checked against the new distribution before anything is retrained, and
the $0.50 per-run cap is lifted (`DECISIONS.md`). Code
`flydream/data/video_hex.py`, `flydream/data/video_corpus.py`,
`flydream/generate/check18.py`, figure `tools/fig_check18.py`.
**Cost: $0** — everything here is local CPU; the 6.93 GB download is the only
external action. Data `data/corpus18/` (gitignored), clip
`reports/figures/2026-09-20_malecns_check18.{gif,png}` (frames 0/22/44 of every
cell viewed before sending).

## Why a new set

17.1b's prior learned 1,695 states from **19 Sintel scenes** and its samples
sit at a round trip of 0.142 where a real clip reaches 0.006 and the
representation itself allows 0.009. 17.3b added the other half of the picture:
real states invert to a noise radius of 1.07-1.29 where the prior's own draws
sit at 1.000 ± 0.014. Both point at the set, not the method.

## The source

**UCF101**: 13,320 clips, 101 action classes, 320×240 at 25 fps, 6.93 GB,
research use. Chosen for the number of *distinct* scenes per gigabyte, which
is what 17.1b lacked; the low resolution costs nothing, because the eye is 721
hexals (about 31×31).

## The eye, and how it was verified

Every clip goes through flyvis's own chain in flyvis's order — centre crop →
`split` → `BoxEye` (extent 15, kernel 13) → resampling to 1/dt →
`HexRotate`. Two rules are ours, and both are measured, not asserted:

- **The frame is resized to Sintel's 1,024 px across, aspect kept.** T4/T5 are
  speed-tuned, so rendering a 320-px video at its own scale would put objects
  across the eye at a different angular speed from everything 13B knows.
- **Time is resampled by holding frames, not by interpolating.** Measured:
  a Sintel clip at 50 Hz repeats every second frame (21 of 39 consecutive
  pairs are identical) — 24 fps is held, not interpolated. UCF101 at 25 fps
  gives exactly 2× holds, Sintel 2.08×.

Verification: rendering Sintel's own frames through this chain reproduces
flyvis's own dataset item at **r = 1.000000, max |diff| 2.6e-4** (the float16
of the flyvis cache). With linear interpolation instead of holds the same
comparison gives r = 0.998138 — the 0.002 was not rounding, it was the wrong
temporal law.

**Speed.** The build was 90 % eye-rendering, because `BoxEye` convolves the
whole 417×417 frame with a 13×13 box (29 M multiply-adds per frame) and then
reads 721 points out of it — 0.4 % of the pixels. Replaced by running sums
read at the ~31 columns and ~61 rows that carry a receptor, plus cropping
before the resize and taking the views as slices instead of copies:

| per file, one thread, idle machine | before | after |
|---|---|---|
| decode | 0.13 s | 0.13 s |
| eye render | 5.46 s (measured under load) / ~2.3 s | **0.09 s** |
| whole corpus, 20 workers | ~57 min | **9 min** |

The values are unchanged: the sampler matches `BoxEye` to 1.7e-6 and the view
bounds match `rendering.utils.split` exactly
(`tests/test_data_helpers.py::test_video_hex_sampler_matches_flyvis_boxeye`).
A GPU was considered and rejected on measurement: after this change the job is
decode-bound, UCF101 is XviD which a T4's decoder does not take, and a T4 with
16 cores would cost ≈ $0.47 to be slower than 9 free minutes.

## The band, derived

Ordinary video carries three things Sintel does not: hard cuts, tripod shots
and flat frames. The thresholds come from the Sintel clips themselves (189
clips, same code): motion **0.0023-0.0615**, contrast **0.058-0.289** (the 5th
to 95th percentile), cut score **≤ 3.21** (their 99th percentile).

The cut score is `max frame difference / 90th percentile of the non-zero
differences`. Non-zero because holds make every second difference exactly
zero; a high percentile rather than the mean or median because the cut itself
is in the sample and inflates them. Compared on 95 Sintel clips against the
same clips spliced in half: at the threshold that drops 1 % of clean clips it
catches **84 %** of the splices, where max/mean catches 78 % and max/median
58 %. **Soft cuts pass** — that is the honest limit of this filter.

## The corpus

| | |
|---|---|
| ordinary video | **12,411** clips from 13,320 files (93 % passed the band, 0 failed) |
| procedural (13A's stimuli, 9 classes) | **3,103** clips = exactly 20 % of the set |
| total | **15,514** clips of 45 frames, 1.03 GB float16 |
| held out | 10 whole classes (1,246 clips), as 13A held out whole scenes |
| build | 543 s on 20 workers, local CPU, $0 |

One window per file, the most-moving one of the candidates; one hex rotation
per window, cycled, so the T4/T5 directions see the corpus from every heading.
Procedural stimuli are mixed in **at the video level**, so the whole set goes
through the brain in one pass, in one per-type scale, under one split.

## 18.2: the two checks (local CPU, 42 s, $0)

| | 13B round trip | motion | contrast | mean | state lag-1 | one-ring | cross-type |
|---|---|---|---|---|---|---|---|
| corpus (16 clips) | **0.016** (0.008-0.048) | 0.0248 | 0.272 | 0.392 | 0.975 | 0.875 | 0.348 |
| Sintel (8 clips), same code path | **0.011** (0.008-0.026) | 0.0118 | 0.241 | 0.342 | 0.982 | 0.866 | 0.349 |
| control: a clip state with its columns permuted | **1.048** | | | | | | |

- **13B stands.** It renders a state that came from ordinary video at 1.5× its
  error on its own Sintel states, and 65× better than the unreachable control.
  The human's condition for retraining it is not met, so it is not retrained
  (≈ $0.22-0.40 not spent).
- **The states are the same kind of object.** Temporal, spatial and cross-type
  structure agree with Sintel's to within a few hundredths on all three.
- **The corpus moves twice as fast** (0.0248 against 0.0118). That is the
  selection: the most-moving window of each file is kept. It is a property of
  this set, and it will be read again when the prior's samples are scored.

## What this does not tell us

Nothing about the prior yet: 18.3 trains it on this corpus and reads it
against 17.1b's 0.142 and 1.07-1.29 in the same code path. Whether more data
or more capacity moves the gate is exactly what that run answers.
