"""Training one member of the MaleCNS model with FlyVis's own solver.

`member.py` is the pure entry point (no Modal import): it composes flyvis's
Hydra config with this project's overrides, builds `MultiTaskSolver` under a
chosen results root, optionally loads a transplanted state, trains, and
returns the timing. `deploy/modal/train_app.py` calls it in one process per
member, or packs several members into one GPU container as subprocesses.
"""
