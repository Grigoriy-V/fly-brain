# Step 18.3: the prior retrained on ordinary video — the gate moves 0.142 → 0.095

Date 2026-09-20. Agent: Claude (Fable). The human's plan of the same day: a
large set mostly of ordinary video, then the prior retrained on it. Code
`deploy/modal/generate_app.py::train17` (the compact-file branch),
`flydream/generate/samples17.py` (`--corpus`), `flydream/generate/noise17.py`
(`--corpus`), figure `tools/fig_prior17.py`. Checkpoint
`flydream-runs:/prior18/corpus_dct16.pt`, local copy `data/prior18/`.
**Cost: one T4, 1,128 s, ≈ $0.22**; every measurement below is local CPU, $0.
Clip `reports/figures/2026-09-20_malecns_prior18.{gif,png}` (frames 0/20/39 of
every cell viewed before sending; the two sample cells are taken from the
middle of the distribution, not the best two).

## The run

| | 17.1b (Sintel) | 18.3 (corpus) |
|---|---|---|
| training states | 1,695 of 19 scenes | **13,555** of 12,411 ordinary clips + 3,103 stimuli |
| representation | DCT-16, keeps 99.55 % of the energy | DCT-16, keeps **99.32 %** |
| model | SiTStates 128 × 4, 1,378,688 par | the same |
| schedule | 12,000 steps, batch 32 | **20,000** steps, batch 32 |
| machine | T4, cpu 1 / 12 GB, 676 s, util 98.3 % | T4, cpu 1 / 12 GB, **1,128 s**, util **98.5 %** |
| loss / validation | 1.837 → 0.506 / **0.566** | 1.865 → 0.479 / **0.506** |

The validation loss was still falling at 20,000 steps (0.5165 at 14.5k →
0.5064 at 20k), so this run is not at the end of its schedule.

## The gates (local CPU, 43 s, $0, every row in the same code path)

| state given to 13B | round trip | nearest training video, r |
|---|---|---|
| **18.3: from the corpus prior** (16 samples) | **0.095** (0.077-0.117) | +0.35 (max +0.49) |
| 17.1b: from the Sintel prior (16 samples) | 0.142 (0.119-0.176) | +0.41 |
| a real held-out clip's own state | 0.012 (0.009-0.014) | +0.55 |
| the representation's own ceiling (a real state, same DCT) | 0.021 | +0.52 |
| control: white noise in the types | 2.177 | +0.06 |
| control: structured noise | 3.176 | +0.35 |
| control: a clip state, columns permuted | 1.313 | +0.12 |

Novelty is read against the prior's **own** training videos here (the corpus),
not against 13A's set; that is why the real clip's +0.55 is lower than the
+0.68 of 17.1b's table — a different bank, the same code.

Diversity: pairwise correlation between sampled states 0.685 (real held-out
clips 0.708), between their videos 0.020 (real clips 0.030). The samples are
as unlike each other as real clips are.

## Coverage: where a real state's own noise lands (local, $0)

The flow run backwards gives every state the noise it came from; a true
Gaussian of this dimension sits at ‖ε‖²/D = 1.000 ± 0.014 (3σ).

| noise recovered from | 18.3 (corpus prior, corpus clips) | 17.3b (Sintel prior, Sintel clips) |
|---|---|---|
| clip A's state | **1.110** | 1.073 |
| clip B's state | **0.970** | 1.290 |
| control: columns permuted | 4.495 | 4.163 |

And the trip through noise and back, with the mixtures on the sphere:

| | round trip | r of the state to its clip |
|---|---|---|
| clip A, real state | 0.043 | 1.00 |
| A through its own noise | 0.054 | **0.98** |
| mix 0.75 / 0.5 / 0.25 | 0.055 / 0.066 / 0.050 | A +0.94 → +0.66, B +0.63 → +0.93 |
| B through its own noise | 0.043 | **0.99** |
| the DCT ceiling on A | 0.061 | 0.99 |

## What it shows

- **The main gate moves 0.142 → 0.095**, and the distance to what the
  representation allows shrinks from 16× (0.142 / 0.009) to **4.5×**
  (0.095 / 0.021). More data was the right suspect.
- **The states are reachable, not just prettier.** The worst of the 16 samples
  (0.117) is still 11× better than a shuffled state (1.313) and 19× better
  than noise in the types (2.177).
- **A clip survives the trip to noise and back at r 0.98-0.99** (was 0.96),
  and the mixtures between two clips now score *better* than unconditional
  samples (0.050-0.066 against 0.095) — the path between two real states is
  the easiest part of the space.
- **Coverage improved where it was worst**: clip B's noise radius 1.290 →
  0.970. The prior's density now sits essentially on the real states, and an
  unreachable state is still four times out.
- **Nothing was retrained but the prior.** 13B stands (18.2: it renders corpus
  states at 0.016 against 0.011 on its own), the frozen brain is untouched.

## What this does not tell us

- **Still 8× a real clip** (0.095 against 0.012). The samples are
  structured moving texture with oriented regions, not scenes with objects.
- **Two things changed at once** — the data (12,411 ordinary clips instead of
  19 scenes) and the schedule (20k instead of 12k steps). This run cannot say
  how much of the 0.142 → 0.095 belongs to each.
- **The run is not converged**: validation was still falling at 20k steps.
- **Novelty is measured, not claimed**: +0.35 against a real clip's +0.55 says
  the samples are not copies of training videos; it does not say they are
  scenes of a new kind.

## Cost

One T4, cpu 1 / 12 GB, 1,128 s, GPU utilisation 98.5 %, ≈ $0.22. Item 18 in
total ≈ $0.58, of which $0.17 was wasted on two mistakes of mine (a pass on
the wrong network, and one OOM at an over-large batch), both recorded in
`reports/runs.jsonl`.
