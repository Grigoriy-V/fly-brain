# Issues

Observed defects, one entry per defect. This file is not a plan and
authorizes nothing; `ROADMAP.md` alone orders work. Evidence lives in
`reports/`, the entry links to it.

## Rules

- A defect is observed behaviour, not a suspicion: a wrong number, a broken
  export, a model that fails a validation it passed, a mapping row that
  contradicts its source. A missing capability belongs in the roadmap; a
  false claim of one belongs here.
- New entry: next free number, never reused; put it at the top of **Open**
  and a row at the top of the catalog.
- Status names the defect, not the effort: `open`, `mitigated` (harm reduced,
  defect still there), `fixed` (verified by a rerun), `won't fix`. A closed
  entry stays, shortened, under **Closed**; a defect that comes back moves up
  to **Open** with a "seen again" line and keeps its full body.
- `Cause` stays "unknown" until proven. One defect per entry. `Costs` says
  what it does to a result or a reader; there is no severity word.
- A defect in a third-party input (a dataset table, a community mapping, a
  reference implementation) is recorded here with its source named; it is
  not fixed upstream by this project.
- Fields: Status, Seen, Costs, Reproduce, Cause, Evidence, Related.

## Catalog

| Id | Status | Defect | Related |
|---|---|---|---|
| ISS-0006 | open | a decoder window of even-spaced lags ([0 2 4], [0 2 4 6 8]) aliases Sintel's 24→50 Hz frame hold: taps two steps apart sit in one phase of the two-step hold, the fit averages two regimes and the reconstruction alternates frame by frame (L3 per-frame corr 0.94/0.92/0.94/0.89/0.93/0.86…); consecutive lags [0 1 2 3 4] remove it; the sweep at those two windows is being redone | roadmap 4 |
| ISS-0005 | mitigated (b fixed, a third-party) | the right-lobe export carries about thirty times less total input than FlyVis on lamina pairs through Am (a: MaleCNS has 49 Lai fragments for ~750 Am, a reconstruction gap) and on the feedback pairs Tm2→L2 and Mi4→Tm2 (b: the min_weight cut, fixed by the weak-pair exception; v9 DSI 0.152 against v7's 0.109, all members stable) | roadmap 1, ISS-0003 |
| ISS-0004 | open | the step-4 stage curve (R → L → Mi/Tm → T4/T5) is a decoder artefact: at a 60 ms window instead of lag 0, T5a's control-corrected score goes 0.057 → 0.625 and the order inverts; stage explains R² 0.317 of the map; read from the best of 50 members on one split | roadmap 4 |
| ISS-0003 | mitigated | `transplant` copied FlyVis's `syn_strength` raw, though it is a gain divided by that connectome's own synapse count; on MaleCNS the copy underweighted T5's main drive and overweighted its inhibition, and step 2's DSI 0.020 was at least partly that (member 000 → 0.161 corrected) | roadmap 2, 3; DECISIONS 2026-09-18 |
| ISS-0002 | open, third-party input | the right CT1 (bodyId 10157, "Roughly traced") has 56,517 outgoing rows in the weights table but only 8 to typed right-lobe neurons, so every CT1 pathway FlyVis carries (CT1 → T4/T5, 14–40 synapses per column) is absent from the export | roadmap 1, 2 |
| ISS-0001 | open, third-party input | MaleCNS's column tag on Tm4 disagrees with the column holding most of its synapses in 47% of right-lobe Tm4 cells; every other tagged type agrees ≥99.3% | roadmap 1 |

---

## Open

### ISS-0006 — even-spaced decoder lags alias the stimulus frame hold

- **Status:** open; `config.toml [decode] lag_windows` changed to consecutive
  lags, the 80 ms and 160 ms windows of the sweep being recomputed. Seen
  2026-09-18 (night) by the human in the 80 ms ladder clip
  (`2026-09-18_decode_sintel_flyvis_ladder_lag_0_2_4.gif`): L3 flickers.
- **Seen:** Sintel is 24 fps resampled to the model's 50 Hz, so each source
  frame is held for about two steps and the stimulus changes mostly on even
  steps (mean frame-to-frame change 0.027 on even, 0.013 on odd steps). With
  lags (0, 2, 4) the decoder's taps sit in one phase of that hold; the fit
  averages the two regimes and the prediction alternates: L3 per-frame
  PixCorr on five test clips 0.94 / 0.92 / 0.94 / 0.89 / 0.93 / 0.86 / 0.92
  / 0.85 …, prediction jump even→odd 0.031 against odd→even 0.043. At lag 0
  the curve declines smoothly; at (0, 1, 2, 3, 4) the jumps are 0.0385 /
  0.0378 and the alternation is gone (mean corr 0.896). L3 shows it most
  (an integrating cell whose response to each frame update is what the
  40 ms taps read); Tm5a less.
- **Costs:** the sweep's 80 ms and 160 ms points (report §10, ISS-0004)
  were measured with aliased windows; the stage-level conclusion may hold
  but the per-type numbers at those windows are not the final ones.
- **Reproduce:** the script in the session transcript of 2026-09-18 night;
  `flydream.decode.map --lags 0 2 4` against `--lags 0 1 2 3 4` on
  `2026-09-18_decode_sintel_flyvis/pairs.npz`, per-frame correlation on
  the test clips.
- **Cause:** lags were chosen as a reach in milliseconds without regard to
  the stimulus's own update period; a window must sample every step it
  spans.
- **Related:** roadmap 4; ISS-0004.

### ISS-0005 — the export's lamina and feedback pairs carry ~30× less total input than FlyVis

- **Status:** open. Seen 2026-09-18 while diagnosing why the corrected
  transplant (ISS-0003) sent two of three members to infinity.
- **Seen:** total synapses a target cell receives from a source type,
  FlyVis against `data/ol/filters_R.json`: R1–R6→Am 36 against 1.2, Am→T1
  63 against 3.3, Tm2→L2 7.3 against 0.3, Mi4→Tm2 5.4 against 0.4, Am→L3
  14 against 1.6. Columnar motion pairs agree: Mi1→T4a 64 against 60,
  L2→Tm2 123 against 128, Tm9→T5a 47 against 28. 63 of 395 matched pairs
  differ more than fivefold by this statistic.
- **Costs:** a transplant that preserves total input (ISS-0003) multiplies
  these pairs' gains by up to 34 and the network runs away; a transplant
  that does not leaves them thirty times too weak. Either way model zero's
  lamina is not FlyVis's lamina, and every result read through Am, T1 or
  the Tm2/L2 feedback is suspect.
- **Reproduce:** `zero.total_n_syn` on both networks; the comparison is
  printed by the diagnostic in `reports/2026-09-18_step4_decoder_stack.md`
  §9 (correction).
- **Cause:** two causes, separated 2026-09-18 evening against the raw
  tables (`data/malecns/*weights*.feather`, `data/ol/neurons_R.parquet`).
  (a) **Am is a third-party reconstruction gap, like CT1.** FlyVis's Am is
  one cell per column (721). MaleCNS names it `Lai` (the bridge renames it
  by `olmatching.tsv`), and the right lobe holds **49** Lai bodies, with
  the source table's own note "many OL Lai are fragmented". The 49
  fragments carry every R→Am, Am→T1 and Am→L3 synapse the export sees, so
  those pairs are about fifteenfold short by cell count alone; the
  photoreceptor side adds the rest, because MaleCNS pools R1–R6 as one
  class (`R1-R6`, 887 cells) that the export expands into six types. No
  threshold or column choice recovers cells that are not in the
  reconstruction. (b) **The feedback pairs are a threshold artefact.**
  Tm2→L2 has 1,421 synapses over 670 rows at weight ≥1 and 234 over 42 rows
  at ≥5: `min_weight` 5 removes 84% of the pair; per target cell that is
  2.2 at ≥1 against the export's 0.3 and FlyVis's 7.3. Mi4→Tm2 likewise
  (1,887 → 338). Columnar motion pairs lose about 20% to the same cut
  (Tm9→T5a 41.2 → 32.2 per cell, Mi1→T4a 72.2 → 65.1) and are not the
  problem. Fix for (b): a per-pair floor, keeping every row of a pair
  whose FlyVis total is small; for (a): record as third-party input, and
  decide with the human whether the Am pathways borrow FlyVis's own
  filters (as a named exception, like a CT1 restoration would be) or stay
  absent and are named in every report that reads through the lamina.
- **Evidence:** `data/runs/2026-09-18_step2_zero_R_v6_rescaled/*.json`
  (the `capped_pairs` list once v7 lands), `reports/runs.jsonl`
  (`step2.transplant`).
- **Related:** ISS-0003, ISS-0002 (CT1 is the same class of defect on a
  single cell), roadmap 1 and 5.

### ISS-0004 — the step-4 stage curve is a decoder artefact

- **Status:** open; the figure is withdrawn from the report and the map is
  being redone. Seen 2026-09-18 by the adversarial audit (six auditors and
  a judge, `research_notes/audit_2026-09-18/`), confirmed by the project
  agent on the same `pairs.npz`.
- **Seen:** with the decoder's window changed from lag 0 to lags (0, 2, 4)
  at dt 0.02 s, i.e. 60 ms, and the time-shuffle control refit at the same
  window, the control-corrected PixCorr goes T5a 0.057 → 0.625, T5d 0.111 →
  0.534, TmY15 0.054 → 0.417, T4a 0.141 → 0.345, while L1 0.336 → 0.300 and
  R1 0.579 → 0.358. The four types at the bottom of the curve end above L1.
  Across 11 types, corr(lag-0 score, gain from the window) = −0.909. The
  frame component at lag 0 separates the published sustained class (L3,
  Mi4, Mi9, Tm9: 0.596 ± 0.048) from the transient class (L1, L2, L4, Mi1,
  Tm1–4: 0.390 ± 0.093) with no overlap across three stages: the map read
  temporal-filter identity, not stage. Stage explains R² = 0.317 of the map
  (Chen et al. 2024, the cited benchmark, R > 0.9); 50% of the variance is
  within stage; hop distance from the photoreceptors alone explains 0.367.
- **Costs:** the advertised first figure (ROADMAP item 4, "the curve R → L
  → Mi/Tm → T4/T5") does not exist in that form; the ladder picture sent on
  2026-09-18 is a true picture of what a per-frame linear decoder recovers
  and a false one of what each stage carries.
- **Reproduce:** `config.toml [decode] lags = [0, 2, 4]`, rerun
  `flydream.decode.map --stimuli sintel --cache` on the cached pairs.
- **Cause:** `lags = [0]` was chosen to match FlyVis's own per-frame head
  and never swept; cells with slow membranes carry the frame's luminance
  in their present voltage, transient cells carry it in their recent
  voltage, and a lag-0 decoder can read only the first. Three further
  defects compound it, each verified by the project agent: the map is read
  from `flow/0000/000`, which is the best of the 50 members by validation
  loss (Spearman(id, loss) = 1.000; member ids are a ranking); the
  sample-shuffle "floor" of 0.117 is the PixCorr of the training-mean
  image, the same for every type (SD 0.003), so it is not a per-type null;
  and seed 0 is the minimum of six scene splits for both Mi1 (0.414 against
  a mean of 0.605) and T5a (0.188 against 0.492).
- **Evidence:** `data/decode/2026-09-18_decode_sintel_flyvis/map.csv`,
  `by_stage.csv`; the audit's own refits in `research_notes/audit_2026-09-18/`.
- **Sweep (2026-09-18, four windows on the same pairs, all 65 types,
  `flydream.decode.sweep`, `data/decode/2026-09-18_sintel_lag_sweep/`):**
  median frame component per stage at 0 / 20 / 80 / 160 ms: photoreceptors
  0.579 / 0.404 / 0.358 / 0.349, lamina 0.435 / 0.366 / 0.300 / 0.255,
  medulla Mi 0.498 / 0.509 / 0.392 / 0.249, Tm/TmY 0.438 / 0.314 / 0.326 /
  0.256, T4/T5 0.221 / 0.384 / 0.511 / 0.427. Spearman(score at 0 ms, gain
  to 160 ms) = −0.827 over 65 types: whatever reads best at one window reads
  worst at another. Median raw PixCorr peaks at 80 ms for every stage but
  the lamina, so 80 ms is the window the map is redone at, reported with
  the 0 ms column beside it. Figure `reports/figures/2026-09-18_sintel_lag_sweep.png`.
- **Related:** roadmap 4; `reports/2026-09-18_step4_decoder_stack.md` §9.

### ISS-0003 — the transplant copied a connectome-relative gain across connectomes

- **Status:** mitigated. The basis is corrected and capped, and the v7 run
  (`2026-09-18_step2_zero_R_v7_total_cap3`) is stable on all three members:
  MaleCNS T4/T5 DSI 0.101 / 0.150 / 0.075 (mean 0.109) against the raw
  copy's 0.020 and FlyVis's 0.547 / 0.344 / 0.281 (mean 0.391);
  preferred-direction error 35.7° / 10.2° / 8.2° against FlyVis's 11.5° /
  14.4° / 69.1°; flash polarity 0.933 / 0.933 / 0.967 against 0.938 / 0.875
  / 0.906. The gap to FlyVis went from 20× to 3.6×; what remains is
  attributed, not fixed: the 114 capped pairs (ISS-0005) and the missing
  CT1 (ISS-0002). Seen 2026-09-18 by the audit, verified in flyvis's source
  and by measurement by the project agent.
- **Seen:** `flyvis/network/initialization.py:501` sets `syn_strength =
  scale / <n_syn>` for the pair in that connectome and `dynamics.py:165`
  forms `weight = sign * n_syn * syn_strength`; `zero.transplant` copied
  the value by type-pair key. On the raw copy, by total input per target
  cell, Tm9→T5a–d (T5's main excitatory drive) came out 1.5–1.7× too weak
  and TmY15→T4/T5 (inhibitory) about 2× too strong. Rerun with the gain
  rescaled: member 000's T4/T5 DSI 0.020 → 0.161 (FlyVis 0.547) with
  preferred-direction error 7.0° against FlyVis's own 11.5°; members 001
  and 002 diverged, because the first rescale used the mean per edge, which
  reached 34× on the pairs of ISS-0005.
- **Costs:** step 2's headline (DSI 0.020 against 0.391) was at least partly
  this function and not MaleCNS; it is the number DECISIONS cites for
  putting item 4 before item 3, and the recorded reason is corrected there.
- **Reproduce:** `python -m flydream.model.zero --models 0 --no-rescale`
  against the default.
- **Cause:** known: the parameter's definition was not read before it was
  copied. Fixed basis: `syn_strength * total_src / total_dst` over
  `zero.total_n_syn`, bounded by `config.toml [model] rescale_cap`.
- **Evidence:** `reports/runs.jsonl` (`step2.transplant`),
  `data/runs/2026-09-18_step2_zero_R_v6_rescaled/`, the v7 run.
- **Related:** ISS-0005, ISS-0002; DECISIONS 2026-09-18 (the order of 4
  and 3, corrected); roadmap 2, 3.

### ISS-0002 — the right CT1's partners are not typed optic-lobe neurons

- **Status:** open; a property of MaleCNS v1.0 as released (status "Roughly
  traced"), not of this code. Seen 2026-09-18 on roadmap 2.
- **Seen:** `connectome-weights` rows with `body_pre = 10157` (CT1, right,
  the only CT1 body on that side): 56,517 rows at any weight, of which 8
  reach a typed right-lobe neuron (LT33 284 synapses, OLVC3 13 at weight
  ≥5); no row reaches a T4, T5, Tm9 or Mi1, which in FlyVis receive 7–40
  CT1 synapses per column. The CT1 has a column ROI (31, 24) like any
  columnar cell.
- **Costs:** the export has no CT1 edges, so model zero lacks the CT1
  inhibitory compartments FlyVis has in every column (CT1(M10) onto T4,
  CT1(Lo1) onto T5); the missing input is one candidate for the weak T4/T5
  selectivity seen in the transplant.
- **Reproduce:** the diagnostic in `reports/2026-09-18_step2_model_zero.md`
  §3 (filter `body_pre == 10157` in the weights table and join to
  `data/ol/neurons_R.parquet`).
- **Cause:** unknown. Plausible: the giant CT1 is split into many untyped
  fragments in v1.0 and the typed body holds only part of the arbor; or its
  postsynaptic partners in the medulla/lobula are the unproofread fragments
  (postsynaptic completion 42%). neuPrint `fetch_adjacencies` on 10157
  would show the partner bodies' status.
- **Evidence:** `reports/2026-09-18_step2_model_zero.md`.
- **Related:** roadmap 1 export, roadmap 2; the neuron count 1 per side in
  `data/bridge/types.csv`.

### ISS-0001 — Tm4's column tag is not the column of its synapses

- **Status:** open; a property of the MaleCNS v1.0 annotation, not of this
  code. Seen 2026-09-18 on roadmap 1.
- **Seen:** right optic lobe, 833 Tm4 cells with `assignedOlHex1/2`: the
  column ROI (`ME_R_col_*`) holding most of the cell's synapses equals the
  tag in 52.9% of cells; the partner-median inference agrees with the tag in
  36.3%. For the other 14 tagged types both methods agree with the tag in
  ≥99.3% (C2 99.5%, Tm20 99.4%, Tm9 99.3%, the rest ≥99.8%).
- **Costs:** whichever column is used, Tm4's offsets to its partners are
  shifted by about one column relative to the other types; a filter
  `Tm4 -> X` or `X -> Tm4` carries that shift. Tm4 is an input to T5 and to
  several Tm/TmY types.
- **Reproduce:** `python -m flydream.data.optic_lobe` with
  `data/ol/roi_columns.parquet` present; the "roi exact by type" line.
- **Cause:** unknown. The tag was likely assigned by a different anchor
  (Tm4's lobula terminal, or the column of its main input) than the medulla
  synapse mass; the MaleCNS release notes do not say how the 15 types were
  tagged.
- **Evidence:** `reports/2026-09-18_step1_data.md` §3; `data/ol/neurons_R.parquet`.
- **Related:** roadmap 1; the choice of ROI columns in `DECISIONS.md`
  (none yet; a step-2 question).

---

## Closed

Shortened to what a later reader needs; the linked report has the rest.
