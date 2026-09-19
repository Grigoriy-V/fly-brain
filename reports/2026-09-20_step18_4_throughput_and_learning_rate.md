# Step 18.4: both of the research note's predictions tested — one refuted, one resolved against the literature

Date 2026-09-20. Agent: Claude (Fable). The two cheapest items on 18.3's option
list, run back to back. Code `deploy/modal/generate_app.py::bench17` (new) and
`::train17` (unchanged), gates `flydream/generate/samples17.py`, figures
`tools/fig_bench17.py` and `tools/fig_lr18.py`. **Cost: $0.03 + $0.24 = $0.27**,
two T4 containers; every gate below is local CPU, $0. Checkpoint
`flydream-runs:/prior18/corpus_dct16_lr1e4.pt`, local copy in `data/prior18/`.
The control for both is 18.3 itself
(`reports/2026-09-20_step18_3_prior_on_the_corpus.md`).

Both questions came from
`research_notes/2026-09-20_video_generation_and_training/` and were written up
as options in `reports/2026-09-20_research_video_generation_and_training.md`.
Neither was a decision; both are now measurements.

## 18.4a Where the training step goes — the launch-bound reading is wrong

`single_gpu_throughput.md` read our 56.4 ms/step against a generous compute
floor of 15-23 ms and called the job "compatible with launch-bound, not proof
of it; profile." Profiled, in one container, eight variants, 60 timed steps
after 15 warmup steps each:

| variant | batch | ms/step | samples/s |
|---|---|---|---|
| **base — 18.3's own loop** | 32 | **54.72** | 584.8 |
| without the per-step `loss.item()` | 32 | 54.81 | 583.8 |
| `torch._foreach_` EMA instead of the per-parameter loop | 32 | 54.94 | 582.5 |
| both | 32 | 54.83 | 583.6 |
| both | 64 | 106.08 | 603.3 |
| both | 128 | 209.72 | 610.3 |
| both | 256 | 418.13 | 612.2 |
| **`torch.compile(mode="reduce-overhead")`** | 32 | **38.47** | **831.8** |

The stand reproduces the thing it measures: 54.72 ms here against the real
run's 56.4 ms.

**Affine fit across the batch sweep: 2.44 ms fixed + 1.6229 ms per sample.** At
batch 32 the fixed part is **4.5 %** of the step, and an 8× batch buys
**+4.9 %** throughput. The step is paid per sample, not per launch — so the
note's hypothesis is refuted, and with it the plan that followed from it
("batch first, then compile, then L4"). Batch is not a lever on this shape.

**Neither of the two suspects in our own code costs anything measurable.** The
`loss.item()` every step (a host synchronisation) and the per-parameter EMA
loop (≈100 small kernel launches a step) both come back inside noise. That is
consistent with the fit: with only 2.44 ms of fixed cost in the step, there is
no room for them to matter.

**`torch.compile` is the one real win: 1.42×**, one-time warmup 41 s. On a
20,000-step run that is 1,128 s → ≈810 s, **$0.22 → $0.16**. It works by fusing
kernels, not by removing launch overhead — the gain shows up in the per-sample
term, which is where the time is.

*Not claimed:* even compiled, 1.20 ms/sample is still 1.7-2.6× the note's
estimated floor. That floor was computed for dense matmuls; ours is attention
over 721 tokens at width 128 with head dimension 32, which is memory-bound.
Where the remaining factor sits was not measured.

## 18.4b The learning rate — the literature's value is 6.2× worse here

The note flagged our 3e-4 at batch 32 as **8.5-24× above** the extrapolation of
DiT/SiT's 1e-4 @ 256 down to our batch, and called a paired run the way to
settle it. One arm was run: **lr 1e-4, and nothing else changed** — same batch
32, same 20,000 steps, same seed 0, same maps file, same split, same code path,
same container shape. 18.3 is the other arm.

| | 3e-4 (18.3) | 1e-4 (18.4b) |
|---|---|---|
| validation loss at 20,000 | **0.5064** | 0.7328 |
| final training loss | 0.4790 | 0.7301 |
| over the last 5,500 steps | −1.95 % | **−0.73 %** |
| **round trip, 16 samples (median)** | **0.095** (0.077-0.117) | **0.591** (0.411-0.865) |
| video novelty, nearest training video r | +0.35 | +0.16 |
| pairwise correlation between sampled states | 0.685 | 0.472 |
| pairwise correlation between their videos | 0.0197 | 0.0095 |
| T4 seconds / utilisation | 1,128 / 98.5 % | 1,243 / 97.5 % |

The controls are identical in both, as they must be — white noise 2.177,
structured noise 3.176, a clip state with its columns permuted 1.313, the
DCT-16 ceiling 0.021, a real held-out clip 0.012.

**The extrapolation does not transfer.** 1e-4 is worse at every validation
checkpoint, and the gap widens monotonically (+0.023 at 500 steps, +0.226 at
20,000). It is not the same curve running late: over the last 5,500 steps the
1e-4 arm improves by 0.73 % against the 3e-4 arm's 1.95 %, so it is flatter
where it stands, not catching up.

**The diversity numbers read the wrong way round on purpose.** The 1e-4
samples are *less* correlated with each other (0.472 against 0.685, where real
held-out clips sit at 0.708) and *further* from the nearest training video
(+0.16 against +0.35). Both would look like gains read alone. Beside a round
trip of 0.591 they are the opposite: states that are not near anything the
brain can read back. This is exactly why novelty is never reported without the
round trip beside it.

**What this settles.** Our 3e-4 is not an oversight to be corrected toward the
literature — it is the better of the two measured points by 6.2× on the main
gate. The one parameter the research note flagged as unjustified is now
justified by measurement, and the item is closed.

## What this does not tell us

- **Two points are not a sweep.** 3e-4 beats 1e-4; the optimum could be at
  3e-4 or above it. The curve's shape (3e-4 still improving faster at the end)
  is consistent with the optimum being higher, and that is untested.
- **One seed each.** No spread is measured, so a difference of 0.095 vs 0.591
  is safe to read and a small difference would not be.
- **`torch.compile` was not used in either arm**, deliberately: 18.3 is the
  control and fusion changes numerics slightly. The 1.42× is available to the
  next *new* configuration, not retroactively to this comparison.
- **The remaining per-sample cost is not explained**, only bounded (§18.4a).

## Cost

Two T4 containers, cpu 1 / 12 GB each: bench17 120.6 s worker / 139 s wall
≈ $0.03 at 67.6 % utilisation (a figure that includes the model rebuilds
between eight variants and is not a training number); train17 at 1e-4
1,243.2 s ≈ $0.24 at 97.5 %. Gates, figures and analysis local CPU, $0. Item 18
now stands at ≈ $0.85 in total.
