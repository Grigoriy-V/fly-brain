# Artefacts for the Human

How a result is delivered. The human reads pictures, not logs (2026-09-19:
"дай мне нормальные артефакты, не надо кормить меня мусором вперемешку").
These rules are part of the working contract: `AGENTS.md` points here and
this file is as binding as the rest of it.

## One artefact per message

- **One artefact per message, with its text before it**, in this order: what
  it is, why it exists, what it checks, what it ran on (model, clip, machine,
  price), what it shows. Never a batch of files under one caption, never a
  log line beside a picture.
- A change of course or a finished artefact is stated in three lines, not
  narrated.
- A question gets an answer, not work; work starts on an instruction
  ("я тебя ни о чём не просил, я тебя спросил").

## Layout: the input beside the output

- **When a video caused the state.** The first column is "что видел глаз"
  (the clip), the outputs to its right, one column per stage or method, the
  score under each. A control is a second input row with its own input shown
  ("вход B: клип 10, лес"), never a footer labelled "control".
- **When no video caused the state** — a sampled, edited or hand-written
  state, which is the current track. There is no input clip, so the first
  column is the state itself as a map (one type, e.g. T4a) and the video
  stands beside it; under the video, the round trip and the direction the
  brain read back. The control row is a state of the same shape known to be
  unreachable (a shuffled state, raw noise in the types), with its own map
  shown, so the reader sees what a failure looks like on the same scale.
- **One row, horizontal, full resolution** (the human, 2026-09-20: a
  vertical sheet is unreadable and a downscaled gif is worse). Only the
  comparisons the message is about; a row of everything that was measured is
  not an artefact.
- Mechanics that hold that shape: about 3 inches per cell, hex raster at
  4 px per ommatidium, 55-100 dpi so a gif stays in single-digit megabytes.
  `tools/fig_gen13b_pick.py:row` is the helper; a new figure script reuses it.

## Clips

- **Clips (gif) by default;** a still frame only when the clip cannot show
  the point. A clip states its real duration and its slow-down: 40 frames at
  20 ms are 0.8 s of the fly's time, played at 8 fps that is 6.25× slower.
- A comparison of two brains or two methods is one clip with a label per row
  and the stage names in the header, not two files.
- **Look at every clip before sending it:** view its first, middle and last
  frames and one column end to end; check that the clip ends where the clip
  ends (no jump into the next clip), that brightness does not breathe from
  frame to frame, that labels match the columns. A subagent (Sonnet) may do
  the viewing when the frames are many; the finding is recorded in the
  message. The human found the "boomerang" ending, the L3 flicker and the
  missing control input before the agent did; that is the failure this rule
  exists for.

## One substrate at a time

MaleCNS is the target; FlyVis is run only to validate a method that has not
run anywhere yet, and then once. Doing both on every step doubled the work
of 2026-09-18.

## Where they live, and size

- `reports/figures/<date>_<substrate>_<experiment>[_<variant>].{gif,png}`;
  the report names the script that drew them.
- A committed gif stays small (2026-09-20: 63-72 MB of gifs were committed
  and had to be re-rendered at a lower dpi; the history still carries them).
  Data, activity tensors and checkpoints are never committed (`AGENTS.md`).
