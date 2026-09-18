# What Does a Fly Dream Of?
## Project idea / research brief

### Working title
**What Does a Fly Dream Of?**  
Russian: **«Что снится мухе?»**

## Core idea

Use a biologically grounded connectome of the fruit fly nervous system as the central computational substrate of an experimental visual model.

The project has two connected goals:

1. **Visual input:** simulate how external visual information can enter a fly-like nervous system through its visual sensors and propagate through a connectome-constrained neural model.
2. **Visual reconstruction / “dreams”:** train an inverse or generative decoder that reconstructs visual information from internal neural activity. Once the reconstruction pathway works on stimulus-driven activity, explore what the same decoder produces from internally generated or spontaneous model activity.

The phrase **“What Does a Fly Dream Of?”** is a conceptual title rather than a literal claim that the reconstructed images correspond to subjective dreams.

---

## Main research question

> **How much information about the visual world can be reconstructed from different internal states of a connectome-constrained model of the Drosophila nervous system?**

A second, more speculative question follows:

> **What visual content is most compatible with internally generated activity when no external visual stimulus is present?**

---

## Why the project is interesting

A connectome is a structural wiring diagram rather than a complete recording of a living brain. It gives access to neurons, their connectivity, synaptic relationships, cell-type information and related annotations, but does not fully specify all neural dynamics.

This creates an interesting intersection between:

- connectomics;
- computational neuroscience;
- differentiable neural simulation;
- representation analysis;
- generative AI;
- inverse problems.

Instead of using a conventional artificial architecture and loosely calling it “brain-inspired”, the project can constrain the computation using measured biological connectivity.

---

## Proposed progression

### Stage 1 — Visual input

Start from a real or synthetic visual stimulus and transform it into a representation compatible with the fly visual system.

Conceptually:

`visual stimulus → fly visual sensors → connectome-constrained neural dynamics`

The first objective is simply to establish a plausible stimulus-to-neural-state pipeline.

---

### Stage 2 — Record internal representations

Capture neural activity at different stages of visual processing.

Examples:

- photoreceptor-level activity;
- lamina;
- medulla;
- lobula / lobula plate;
- progressively deeper central-brain regions.

This allows the project to ask how visual information changes as it propagates through the nervous system.

---

### Stage 3 — Reconstruct the visual world

Train a decoder that attempts to reconstruct the original visual stimulus from neural activity.

The key experiment is to compare reconstructions from different regions or depths:

`neural state at region X → decoder → reconstructed image/video`

This should not be interpreted as “the image seen by the fly”. A more accurate interpretation is:

> **the visual information that remains decodable from the modeled neural activity at that stage.**

Possible measurements include reconstruction fidelity, motion information, spatial structure, color information and semantic or behavioral features.

---

### Stage 4 — Information loss / transformation map

Compare what can be reconstructed from different parts of the nervous system.

A useful conceptual visualization would be:

`input → retina → lamina → medulla → lobula → central brain`

with a reconstruction produced from each stage.

This could reveal where the representation stops behaving like an image and becomes increasingly specialized for features such as:

- motion;
- direction;
- contrast;
- optic flow;
- object-relevant cues;
- behavioral signals.

---

### Stage 5 — “Dreams”

After the decoder has been validated on stimulus-driven activity, remove or neutralize the external visual input and run the neural model from internally generated activity.

Then decode those internal states:

`internal neural dynamics → trained generative decoder → image/video`

The result can be presented as:

> **the visual stimulus most compatible with a particular internally generated state of the model.**

This is the basis of the **“What Does a Fly Dream Of?”** experiment.

It is important to avoid claiming that such images are literal subjective experiences or biological dreams.

---

## Connectome strategy

The long-term target should be the **MaleCNS** connectome: a reconstruction of the male Drosophila central nervous system that includes the brain and ventral nerve cord and provides cell types, connectivity, synapses, skeletons and related annotations.

However, the project does not need to simulate the entire nervous system from the first prototype.

A sensible approach is:

- use **MaleCNS as the canonical dataset from the beginning**;
- initially activate or extract only the visual pathway / visual subgraph;
- expand progressively into deeper brain regions;
- eventually investigate whole-CNS activity.

This minimizes the conceptual gap between an early visual prototype and the eventual whole-brain experiment.

---

## Reference model: FlyVis

**FlyVis** is an important reference because it demonstrates that a fly visual connectome can be turned into a differentiable dynamical model.

The associated Nature paper used measured connectivity of the Drosophila visual system while learning unknown cellular and synaptic parameters for an optic-flow task.

For this project, FlyVis is useful primarily as:

- a validated precedent;
- a reference for connectome-constrained dynamics;
- a way to test visual stimuli and neural responses;
- a baseline against which a MaleCNS-based implementation can be compared.

The goal does not need to be to permanently build the project around the FlyVis connectome itself.

---

## Scientific caution

Several distinctions should remain explicit throughout the project:

### Connectome ≠ full brain state

A connectome does not contain everything required to exactly reproduce biological neural activity. Missing or uncertain elements may include:

- exact synaptic strength;
- membrane and channel dynamics;
- time constants;
- receptor-specific effects;
- neuromodulation;
- internal physiological state;
- learning-related state;
- detailed compartmental dynamics.

Therefore the project is best described as a **connectome-constrained model**, not a complete digital copy of a living fly brain.

### Reconstruction ≠ subjective perception

If an image can be reconstructed from neural activity, that only demonstrates that information about the stimulus is present and decodable.

It does not demonstrate that the animal experiences that reconstructed image.

### “Dream” is a project metaphor

Images decoded from spontaneous or internally generated model activity should be described as **model-compatible visual reconstructions**, not verified dreams or hallucinations of a biological fly.

---

## Potential research outputs

The project could potentially produce:

- a visual stimulus → fly-connectome simulation pipeline;
- visualizations of activity propagating through the nervous system;
- reconstructions from different visual-processing stages;
- quantitative maps of where different visual information remains decodable;
- comparisons between biological connectivity and artificial neural architectures;
- generative reconstructions from internally generated states;
- interactive visualization of neural activity and corresponding decoded images;
- a research article / technical report;
- an open-source experimental framework.

---

## Possible framing for a paper

### Main title
**What Does a Fly Dream Of?**

### Possible subtitle
**Reconstructing visual information from a connectome-constrained model of the Drosophila nervous system**

Alternative:

**What Does a Fly Dream Of? Generative decoding of internal states in a connectome-constrained fly nervous system**

---

## Key references

### MaleCNS — complete male Drosophila CNS connectome
Project page:  
https://male-cns.janelia.org/

Janelia overview:  
https://www.janelia.org/news/researchers-reveal-connectome-of-the-male-fruit-fly-central-nervous-system

The MaleCNS project provides tools for browsing cell types, connectivity and eye maps, querying the connectome, and downloading annotations, synapses, skeletons and related data.

---

### FlyVis
GitHub:  
https://github.com/TuragaLab/flyvis

FlyVis is a PyTorch implementation of a connectome-constrained deep mechanistic model of the fruit fly visual system.

---

### Lappalainen et al., Nature (2024)
**Connectome-constrained networks predict neural activity across the fly visual system**

Nature article:  
https://www.nature.com/articles/s41586-024-07939-3

DOI:  
https://doi.org/10.1038/s41586-024-07939-3

This is one of the most relevant precedents for the project. It demonstrates a differentiable model whose architecture is constrained by experimentally measured fly visual connectivity and whose unknown neural parameters are optimized for optic-flow estimation.

---

### FlyWire / Codex
Connectome explorer:  
https://codex.flywire.ai/

Useful as a broader reference for interactive exploration and analysis of large Drosophila connectome datasets.

---

## One-sentence project summary

> **Build a connectome-constrained model of the fruit fly visual system, trace how visual information is transformed through the nervous system, reconstruct the world from internal neural states, and finally ask what images are generated when the model is left to its own internal dynamics.**
