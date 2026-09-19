# Idea: Whole-MaleCNS baseline with Shiu-style LIF dynamics

## Goal

Remove the current FlyVis boundary at T4/T5 and allow the project to follow neural activity deeper through the MaleCNS connectome.

Current setup:

```text
MaleCNS wiring
+ FlyVis-compatible rate dynamics
+ transplanted FlyVis parameters
→ current visual model
```

Proposed baseline:

```text
MaleCNS wiring
+ generic Shiu-style LIF dynamics
→ much larger / potentially whole-CNS dynamical model
```

This would let the project simulate cell types and pathways that FlyVis does not contain.

---

## What Shiu-style LIF means

LIF = **Leaky Integrate-and-Fire**.

Each neuron keeps a membrane-potential-like state.

Conceptually:

```text
incoming spikes
→ integrated into membrane potential
→ potential leaks back toward baseline
→ threshold reached
→ neuron emits a spike
→ reset
```

Connections come from the real MaleCNS connectome.

The connectome determines:
- which neurons are connected;
- how many synapses exist between neurons;
- connection direction;
- excitatory / inhibitory sign where available.

The LIF model supplies the missing generic dynamics needed to make that wiring run over time.

---

## Why this helps this project

The current model is limited by FlyVis coverage.

Today the main simulated visual path is approximately:

```text
R → L → Mi/Tm → T4/T5
```

But MaleCNS continues far beyond T4/T5.

With a whole-MaleCNS LIF baseline we could follow activity through:

```text
photoreceptors
→ lamina
→ medulla
→ T4/T5
→ LPi / VS / HS
→ visual projection neurons
→ central brain
→ descending neurons
→ motor pathways
```

The important point is that we no longer need FlyVis parameters for every new cell type before we can simulate it.

---

## What this would enable

### 1. Deeper neural representations

Record state at multiple depths:

```text
T4/T5
VS/HS
visual projection neurons
central brain
descending neurons
```

Then ask at every level:

```text
state_X → decoder / generator → video
```

This would show how much visual information remains decodable deeper in the nervous system.

---

### 2. New conditioning spaces for the generator

Instead of conditioning 13B only on T4/T5, train future generators on deeper states:

```text
VS/HS state → video
projection-neuron state → video
central-brain state → video
```

This could reveal where the representation stops being image-like and becomes more abstract: optic flow, heading, object relevance, behavior, etc.

---

### 3. Whole-brain internal dynamics

A larger MaleCNS model also creates a possible route toward internal-state experiments:

```text
central / recurrent brain dynamics
→ visual-related internal state
→ generator
→ video
```

This is much closer to the long-term “What Does a Fly Dream Of?” direction than the current optic-lobe-only model.

---

## Important limitation

Shiu-style LIF does **not** mean that MaleCNS suddenly becomes a biologically exact brain simulation.

MaleCNS gives the structural wiring.

The LIF model adds a simplified generic rule for how neurons behave.

Therefore:

```text
MaleCNS connectome + LIF
```

should be treated as a **whole-brain dynamical baseline**, not as a literal digital fly brain.

It is useful because it allows signals to propagate through the full connectome.

It does not guarantee that the activity of every neuron matches real physiology.

---

## Suggested development path

### Stage A — one-hop extension

Before attempting the whole CNS, extend beyond T4/T5:

```text
T4/T5 → LPi / VS / HS
```

Check:
- stability;
- motion responses;
- whether signals remain structured;
- whether deeper states remain decodable.

### Stage B — visual projection neurons

Continue:

```text
optic lobe → visual projection neurons
```

Record deeper state and test reconstruction / generation.

### Stage C — central brain

Add the relevant central-brain pathways and inspect how the visual representation changes.

### Stage D — larger / whole MaleCNS

Only after the smaller extensions are stable, scale the same LIF framework toward the full connectome.

---

## Relationship to the current model

The existing FlyVis-derived model should not be discarded.

It can remain the high-quality visual reference:

```text
FlyVis-derived MaleCNS model
= stronger calibrated visual dynamics
```

while the new LIF model becomes:

```text
MaleCNS LIF model
= wider anatomical coverage
```

The two can be compared on the overlapping visual pathway.

This gives a useful tradeoff:

```text
FlyVis-based model:
better calibrated visual physiology
but limited depth

LIF MaleCNS model:
much deeper / potentially whole brain
but rougher dynamics
```

---

## Main idea in one sentence

> Use Shiu-style LIF dynamics as a generic dynamical layer on top of the MaleCNS connectome so the project can move beyond T4/T5 and follow neural signals through deeper visual pathways and eventually the central brain, while keeping the current FlyVis-derived model as the calibrated visual reference.
