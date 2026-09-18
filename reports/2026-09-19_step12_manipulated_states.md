# Step 12: manipulated states — gain edits, interpolation, hybrids

Date 2026-09-19. Agent: Claude (Fable). Design: the human's note
`docs/ideas/mixed_brain_state_video_generation.md`, agreed in chat. Code:
`flydream/generate/mix.py`, `tools/fig_mix.py`, `deploy/modal/generate_app.py::mix`,
commit `eead1ac`.

## Question

The generator inverts one clip's state. What does it return when the target
state is assembled from several sources or edited by hand: does one type's edit
move the video, is the mix of two states the state of the mixed video, and can
the early types carry one clip while the motion types carry another?

## Method

Clips A = Sintel 3 (the face) and B = Sintel 10 (the forest), the pair of
items 8-9. Targets are the noiseless states the clips cause. A task's loss is
the mean over its types of the type's normalised squared error (cell weight
1 / (k · n_t · var_t), var_t from clip A's state), so T4/T5 at sd ≈ 0.4 cannot
drown L3 at 0.04 (`type_weights`). 15 tasks in one batch, 40 + 5 frames, 150
steps with the plateau stop (fired at 120). Every task is scored by mean
per-frame PixCorr to clip A, to clip B, and to the averaged video ½(A+B).

- **C, gain edits**: clip A's state on the ladder's 14 types (L1, L3, Mi1,
  Mi4, Tm5a, Tm9, T4a-d, T5a-d; 9,624 cells) with one type scaled; × 1 is the
  control.
- **A, interpolation**: T4a's state at α·A + (1−α)·B, read on T4a alone (721
  cells); control: the state the averaged video causes.
- **B, hybrids**: L1, L3 (1,442 cells) from one clip with T4a-d, T5a-d (5,768
  cells) from the other; controls: all from A, all from B, on the same types.

## Run

`modal run deploy/modal/generate_app.py --model malecns --mix-clips 3,10`, T4,
73 s of optimisation, 117 s on the worker, GPU utilisation 72 %, ≈ $0.05. Data
`data/generate/2026-09-19_malecns_mix_<task>/`; clips
`reports/figures/2026-09-19_malecns_mix_{C,A,B}.{gif,png}`, frames 0/20/39
viewed before sending.

## Numbers

| task | r_A | r_B | r_avg | fit at end |
|---|---|---|---|---|
| C × 1 (control) | 1.00 | −0.20 | 0.81 | 0.00006 |
| C T4a × 0.5 | 1.00 | −0.21 | 0.81 | 0.122 |
| C T4a × 2 | 0.99 | −0.18 | 0.82 | 0.492 |
| C T5 × 0 | 0.95 | −0.22 | 0.75 | 0.618 |
| C Mi4 × 1.5 | 1.00 | −0.19 | 0.81 | 0.162 |
| A α = 0 (B) | −0.20 | 1.00 | 0.40 | 0.00004 |
| A α = 0.25 | 0.33 | 0.85 | 0.81 | 0.00004 |
| A α = 0.5 | 0.82 | 0.37 | **0.99** | 0.00005 |
| A α = 0.75 | 0.97 | −0.01 | 0.91 | 0.00007 |
| A α = 1 (A) | 1.00 | −0.20 | 0.81 | 0.00009 |
| A averaged video's state (control) | 0.81 | 0.40 | 1.00 | 0.00003 |
| B all from A (control) | 1.00 | −0.20 | 0.81 | 0.00009 |
| B L1/L3 ← A, T4/T5 ← B | 0.51 | 0.62 | 0.84 | 0.149 |
| B L1/L3 ← B, T4/T5 ← A | 0.88 | 0.15 | 0.91 | 0.152 |
| B all from B (control) | −0.20 | 1.00 | 0.40 | 0.00004 |

## What the pictures show

- **Gain edits do not move the video** (r_A 0.99-1.00 for T4a × 0.5, × 2,
  Mi4 × 1.5) but leave a residual 2,000-8,000× the control's: no input doubles
  T4a while the other 13 types stay put, so the generator returns the nearest
  reachable state, the clip itself. T5 × 0 is the one edit with a visible
  trace (r_A 0.95): a diagonal lattice over the dark regions, the wiring's
  own texture seen in item 11.
- **The mix of states is the state of the mix.** At α = 0.5 the recovered
  video matches the averaged video with r 0.99 and sits between the clips
  (0.82 / 0.37), the same numbers the averaged video's own state gives
  (0.81 / 0.40 / 1.00); the transition over α is monotone; every
  intermediate state is reachable (fit ≈ 5·10⁻⁵). On this pair the map
  video → T4a state is close to linear.
- **Hybrids are a contest, and T4/T5 win.** Early types from A with motion
  types from B gives 0.51 / 0.62; the reverse 0.88 / 0.15. T4/T5 hold 5,768
  cells to L1/L3's 1,442 and, as the ladders showed, their state alone
  carries the frame. Residual 0.15: the combination is unreachable, and the
  video is a compromise with the lattice in the zone of conflict. A hybrid is
  a counterfactual set of constraints, not a perception.

## Not done

The three-source hybrid (the note's D) waits, as planned: B's answer is that
the motion types dominate, so a third source would add nothing until the
readout is balanced (e.g. by weighting types by evidence rather than by cell
count, or by reading T4/T5 on a subset of cells).

## Cost

Modal T4 ≈ $0.05. Local: CPU drawing only.
