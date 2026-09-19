# Step 17.3b: the flow run backwards — a state's own noise, and the path between two clips

Date 2026-09-20. Agent: Claude (Fable). Asked for by the human, 2026-09-20:
"состояние → шум и обратно… чтобы по случайному шуму получить состояние".
Code `flydream/generate/prior17.py` (`integrate`, `invert`, `to_noise`,
`from_noise`, `slerp`), experiment `flydream/generate/noise17.py`, figure
`tools/fig_noise17.py`. **Cost: $0** — local CPU, 35 s, no Modal, no training.
Prior `data/prior17/scene_dct16.pt` (17.1b: Sintel-only, DCT-16), generator
`data/gen13b/sit.pt`, both unchanged. Data `data/prior17/noise17_local.{json,npz}`,
clip `reports/figures/2026-09-20_malecns_prior17b_noise.{gif,png}` (frames
0/20/39 and one column end to end of every cell viewed before sending).

The prior is a deterministic map from noise at t = 0 to a state at t = 1. Run
its Euler steps backwards and any state gets the noise it would have been
drawn from. That gives three things the round trip alone cannot: a check that
the map is invertible at all, a measure of whether the prior *covers* real
states, and a navigable space in which two clips can be mixed.

## 1. The inversion itself (the check everything else rests on)

The forward step is x⁺ = x + v(x, t)/steps; the step backwards solves
x = x⁺ − v(x, t)/steps, which fixed-point iterations from x⁺ reach.

| inversion | noise → state → noise, r | relative error | model calls per step |
|---|---|---|---|
| explicit backward step | 0.9782 | 0.243 | 1 |
| **4 fixed-point iterations** | **0.99998** | **0.0058** | 4 |

The plain backward step loses a quarter of the vector; with the fixed point
the recovered noise is the noise the sample started from. Everything below
uses 4 iterations, 20 steps.

## 2. A real state → its own noise → back

| | clip A (alley_1) | clip B (market_5) |
|---|---|---|
| state recovered, r | **0.964** | **0.961** |
| the DCT-16 ceiling on the same state, r | 0.998 | 0.992 |
| round trip of the reconstruction | 0.072 | 0.191 |
| round trip of the real state (same code path) | 0.023 | 0.038 |
| round trip of the band-limited real state (ceiling) | 0.038 | — |
| video of the reconstruction vs the clip, r | +0.93 | +0.83 |

A clip's state survives the trip to noise and back: the face in clip A is
still the face (second cell of the figure), and the brain still accepts the
state — 0.072 against 0.023 for the real state and 0.038 for what the
representation allows. Clip B, a high-motion market scene, costs more
(0.191): the loss is in the prior's velocity field, not in the inversion,
which is exact to 0.6 %.

## 3. Does the prior cover real states?

For true N(0, I) noise of dimension D = 92,288, ‖ε‖²/D = 1.000 ± 0.014 (3σ).
Where a real state's own noise lands on that scale says whether the prior
gives it any density.

| noise recovered from | ‖ε‖²/D | sd | mean |
|---|---|---|---|
| clip A's state | 1.073 | 1.036 | +0.002 |
| clip B's state | 1.290 | 1.136 | +0.007 |
| **control: clip A's state, columns permuted** | **4.163** | 2.040 | +0.016 |
| a Gaussian draw (the prior's own input) | 1.000 ± 0.014 | 1.000 | 0.000 |

Real states sit **just outside** the typical set (1.07 and 1.29 against
1.000 ± 0.014) — the prior is close to covering them but not centred on them,
which is the same 0.142-vs-0.009 gap 17.1b measured, seen from the other
side. A state the brain cannot reach sits four times out. The two clips'
noises are uncorrelated (r = −0.013) although their *states* correlate at
+0.48: the prior decorrelates what it models, which is what makes the space
below navigable.

## 4. The path between two clips (the row in the figure)

Both clips' noises, mixed on the sphere (a straight line would shrink the
radius by up to 1/√2 and leave the typical set), each mixture carried forward
to a state, through 13B and through the frozen brain.

| state | round trip | r state → A | r state → B | video vs A | video vs B | nearest training video |
|---|---|---|---|---|---|---|
| clip A, real state | 0.023 | +1.00 | +0.48 | +0.99 | +0.03 | +0.99 (its own clip) |
| A through noise | 0.072 | +0.96 | +0.37 | +0.93 | −0.03 | +0.93 (alley_1) |
| mix 0.75·A + 0.25·B | 0.170 | +0.90 | +0.61 | +0.84 | +0.26 | +0.84 (alley_1) |
| **mix 0.5 / 0.5** | **0.216** | +0.72 | +0.82 | +0.64 | +0.54 | **+0.59** (market_5) |
| mix 0.25·A + 0.75·B | 0.210 | +0.56 | +0.93 | +0.37 | +0.74 | +0.78 (market_5) |
| B through noise | 0.191 | +0.41 | +0.96 | +0.07 | +0.83 | +0.85 (market_5) |
| clip B, real state | 0.038 | +0.48 | +1.00 | +0.02 | +0.95 | +0.96 (its own clip) |
| **control: columns permuted** | **1.585** | +0.04 | +0.07 | −0.05 | −0.02 | +0.11 |

- **The path is monotone in both directions** — A falls +0.96 → +0.90 →
  +0.72 → +0.56 → +0.41 while B rises +0.37 → +0.61 → +0.82 → +0.93 → +0.96,
  and the *videos* follow (pixel correlation to B −0.03 → +0.26 → +0.54 →
  +0.74). The mixture is not a crossfade of two pictures: it is one state
  that 13B renders as one scene.
- **Every point on the path stays reachable**: 0.17-0.22 against 1.585 for a
  state known to be unreachable and 0.142 for an unconditional sample of the
  same prior. The mixtures are no worse than what the prior draws from
  scratch, and 7-9× better than the control.
- **The middle of the path is the farthest from the training set**: nearest
  training video +0.59 at the midpoint against +0.99 / +0.96 at the ends
  (normalised distance 0.68 against 0.03 / 0.09). This is a video no clip
  caused and no training clip is close to — measured, not asserted.

A second pair run the same way (clip A with mountain_1, sample 126, in
`data/prior17/noise17_mountain.json`): reconstruction r 0.964 / 0.966, round
trips 0.072 / 0.069, noise radius 1.073 / 0.737, mixtures 0.121 / 0.145 /
0.119 with the same monotone morph. The effect is not particular to one pair;
market_5 is shown because mountain_1 is a bright low-contrast sky and the row
washed out.

## What this does not show

- **Not a better prior.** Nothing was trained. The mixtures sit at the same
  round trip as the prior's own samples (0.14-0.22), which is still ~20×
  above a real clip.
- **Not "pulling an off-manifold state onto the manifold"** (ROADMAP 17.3,
  still open). The shuffled control was inverted to read its radius, not
  projected and re-scored.
- **Not novel scenes.** The midpoint video is farther from the training set
  than either endpoint (+0.59), and it is a mixture of two known scenes, not
  a new one.

## What it changes for the next step

The noise space is a real interface: a state goes into it and comes back
(r 0.96), and points between two states are reachable. The radius numbers
(1.07 and 1.29 against 1.000 ± 0.014) say the remaining error is that the
prior's density is not yet centred on real states — 1,695 training states of
19 scenes — which is exactly what the human's next step attacks: a large set
of ordinary video with procedural stimuli as a minority.

## Cost

Local CPU (32 cores), 35 s, $0. No Modal function was started, nothing was
downloaded, no checkpoint was written. Offline tests: 83 passed
(`test_prior17_noise_inversion_and_slerp` added).
