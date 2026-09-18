"""Decoders: from model activity back to the stimulus most compatible with it.

The ladder (DECISIONS, 2026-09-18) is ridge, then a convolution on the
hexagonal lattice, then inversion of the model itself; a diffusion or
transformer decoder only after the three agree on the shape of the curve.

- `pairs`: paired rendered stimulus and per-cell-type activity, split by scene
- `ridge`: rung one, with the penalty tuned per cell type
- `metrics`: PixCorr, R2, SSIM on the lattice, identification
- `hexraster`: the hexagonal lattice as a picture and as a neighbourhood
- `map`: the driver that writes one row per cell type per control
"""
