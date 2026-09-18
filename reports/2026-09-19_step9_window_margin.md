# Step 9: the end of the generator's window is constrained; clip length is a setting

Date 2026-09-19. Agent: Claude (Fable). Code: commits `0838fa5`, `generate_app` config fix.

## Question

The item-8 generator (encoder inversion, `flydream/generate/invert.py`) blurred
the last 2-3 frames of every recovered clip: the activity of a frame depends on
the frames after it (membrane time constants, T4/T5 delays), and the last
frames of a 20-frame window had no future to be constrained by. Does fitting a
margin past the shown window remove the blur, and does the whole Sintel clip
(40 model frames) work as the window?

## What changed

- `config.toml [generate]`: `frames = 40` (the whole clip, 0.8 s of the fly's
  time), `margin = 5`, plus steps/lr/tv/dt/t_pre, all read by
  `flydream.generate.invert` and `deploy/modal/generate_app.py`.
- The optimiser fits `frames + margin`; the saved and scored videos are the
  first `frames`. The margin's video is the next temporal chunk of the same
  scene: in the Sintel set of the map the next sample index is that chunk
  (checked: last frame of clip 3 to first frame of clip 4, r = 0.97); when
  the next sample is another scene the last frame is held (`extend_clip`).
- `tools/fig_generator_two_inputs.py`: the two-input figure of 2026-09-19
  made a script (row A = the clip, row B = the wrong-target control with its
  own input shown).

## Run

`modal run deploy/modal/generate_app.py --model malecns --sample 3` on a T4,
one stage at a time (batch 1), 45-frame window, 150 steps: 948 s of GPU, 1,015 s
wall, ≈ $0.17. Results under `data/generate/2026-09-19_malecns_invert_<stage>_s3/`
(the worker's UTC date 2026-09-18 renamed; the 20-frame runs of 2026-09-18 kept
under `data/generate/_20frames_2026-09-18/`). Figure
`reports/figures/2026-09-19_malecns_generator_two_inputs_40f.{png,gif}`.

## Numbers

Mean PixCorr over the 40 shown frames, control = the same optimisation aimed at
clip 10's state, scored against clip 3:

| stage | r (40 f) | r (20 f, item 8) | control | last-frame r, 20 f → 40 f |
|---|---|---|---|---|
| R1 | 0.999 | 1.00 | −0.20 | |
| L1 | 0.999 | 0.99 | −0.20 | |
| L3 | 0.980 | 0.98 | −0.22 | 0.86 → 0.98 |
| Mi1 | 1.000 | 0.99 | −0.20 | |
| Mi4 | 1.000 | 1.00 | −0.20 | |
| Tm5a | 0.953 | 0.96 | −0.22 | 0.86 → 0.94 |
| Tm9 | 0.998 | 0.98 | −0.20 | |
| T4a | 0.996 | 0.98 | −0.20 | |
| T5a | 0.933 | 0.92 | −0.22 | 0.66 → 0.92 |
| T4+T5 (8 types) | 0.999 | 0.98 | −0.20 | 0.80 → 1.00 |

Per-frame r is flat to the last shown frame in every stage (T5a: 0.92 on frames
34-39; before: 0.94, 0.93, 0.92, 0.82, 0.76, 0.66). The recovered T5a video's
spatial-gradient energy relative to the true clip is 0.26-0.35 across frames
0-39, with no drop at the end. Frames 0, 20 and 39 of the clip were viewed
before sending: the last frame is as sharp as the middle one, labels match the
columns, and the clip ends inside its own scene.

## Where the fit plateaus (for item 10')

Step at which the fit had changed by less than 1 % over 20 steps, of 150:
L3 56, T5a 105, T4+T5 121, Tm5a 131. R1 was flat from step 50. 150 steps are
needed for the deep stages; a plateau stop saves the shallow ones.

## Claim

With a 5-frame fitted margin the end-of-window blur is gone and the whole
40-frame clip inverts as well as the 20-frame one did in the middle. The
recovered video is the stimulus most compatible with the state under this
encoder, not what the fly sees. One clip, one control clip, one model.

## Cost

Modal T4: ≈ $0.17 (one failed import before the GPU started: ≈ $0). Local: CPU
drawing only.

## Item 10': the ladder batched (same day)

All 20 tasks (10 stages × inversion and wrong-target control) optimised in one
pass through one simulation, the loss masked per task to its cells
(`invert_batch`, `task_weights`; Adam is elementwise, so the tasks stay
independent — `tests/test_generate.py` checks two tasks in one pass equal two
passes on the miniature network). Same command, `config.toml [generate]
batch = 0` (all at once).

| | batch 1 | batch 20 |
|---|---|---|
| optimisation, 150 steps × 20 tasks | 948 s | 127 s (0.85 s per step) |
| ladder wall time on the worker | 948 s | 176 s |
| GPU utilisation (nvidia-smi, 2 s samples) | not sampled | 66 % |
| price (T4) | ≈ $0.17 | ≈ $0.05 |
| max |video difference| vs batch 1 | — | 0.0007 (of 1.0) |
| r per stage | | equal to three decimals |

7.5× on the optimisation, 5.4× on the ladder (model load and target
simulation are the rest). Memory was not the limit at batch 20 on a T4; the
card sits at 66 %, so an L4 was not needed. A ladder of one clip now costs
about five cents.

The plateau stop (1 % over 20 steps) did not fire: Mi4's fit at 3·10⁻⁵ of
its start still moved 10 % per 20 steps. The rule now also counts a task as
flat when its change is under 1 % of 10⁻³ of its first fit
(`plateau_floor`); on this run's traces that stops at step 127 of 150 with
every fit within 2.4 % of its final value. Not re-run: the saving is 15 %, and
it applies from the next ladder.
