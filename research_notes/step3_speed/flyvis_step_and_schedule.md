# flyvis 1.2.0: the per-step ops, batch scaling, the schedule, and what can be sped up

Scout: one Sonnet subagent, read-only over `D:/ML/Fly_Brain/.venv/Lib/site-packages/flyvis`, 2026-09-18 evening; 122,125 tokens, 2.2 min. Requested by the project agent after the T4 smoke (0.434 s/iter, ~$18 per member) and the human's "поищи как ускорить обучение и сократить расходы". The text below is the scout's report as returned, lightly trimmed; it is data the project agent checked against the source lines it cites. Nothing here was measured by the scout; the batch-size and memory claims are inferences from shapes, to be measured by `smoke_batch`.

## 1. The step function

`Network.forward` (`network.py:509-546`) loops `for i in range(x.shape[1])` over the ~40 frames calling `_next_state` (`network.py:377-413`), which calls `dynamics.write_state_velocity` (`dynamics.py:185-218`, `PPNeuronIGRSynapses`) and does the Euler update. Per step, for batch B, N = 31,526 nodes, E = 1.35M edges:

1. `index_select` gathers node activity to edges, (B,N) → (B,E) (`tensor_utils.py:23-25`, via `_source_gather`/`_target_gather`, `network.py:303-331`).
2. ReLU on (B,E).
3. `weight * activation`, (E,) broadcast × (B,E). The weight is computed once per forward, not per step.
4. `target_sum` (`network.py:333-351`): a fresh `torch.zeros((B,N))` every step and `scatter_add_` over the E edges — dense, no `torch.sparse` anywhere in the package.
5. Elementwise velocity on (B,N): `1/max(tau, dt) * (-state + bias + input + x_t)`.
6. Euler update creating new `AutoDeref` dicts per step (`network.py:404-410`).
7. `_state_api` (`network.py:415-442`) rebuilds sources/targets with two more `index_select`s and new `AutoDeref` objects per step.
8. One `torch.stack` of the 40 (B,N) outputs after the loop.

No `.item()`/`.cpu()` inside the loop; no in-place ops on autograd views in `_next_state`; shapes are static across steps. What blocks `torch.compile(mode="reduce-overhead")`/CUDA graphs is the `AutoDeref`/`RefTensor` machinery (`tensor_utils.py:11-141`): a `dict` subclass with overridden `__getitem__`/`__setitem__` and a lazily invalidated `_cache`, rebuilt every step — classic Dynamo graph-break material. Making the step graph-capturable means rewriting `_param_api`/`_next_state`/`_state_api`/`target_sum` and `write_state_velocity` to work on plain tensors, and pre-allocating `target_sum`'s buffer: an internal refactor, not a wrapper.

## 2. Batch scaling

Every per-step op has B as a plain leading dimension; there is no per-sample Python loop (`x[:, i]` slices the batch). Since E and N are fixed and a T4 at B = 4 is far from occupied, an iteration at B = 32 is expected to cost well under 8× the B = 4 iteration — inferred from shapes, not measured. Memory is the risk: autograd keeps the per-step (B,E) intermediates for the backward through 40 steps; at B = 32 each such tensor is 1.35M × 32 × 40 × 4 B ≈ 6.9 GB, and `write_state_velocity` creates two or three of them per step, so B = 32 may exceed a T4's 16 GB. B = 16 is the safer upper point; an OOM at 32 is itself a result.

## 3. The schedule

- Validation (`test()`, `solver.py:474-556`) runs inside `checkpoint()` (`solver.py:427-439`), once per `chkpt_every_epoch` = 300 epochs (`scheduler.yaml:21`), at the last epoch, and once at the start.
- A checkpoint saves the network, decoders, optimiser, penalties, `val_loss`, `iteration` (`solver.py:441-458`); `chkpt_index.h5`, `chkpt_iter.h5`, `best_chkpt_index.h5` in the NetworkDir.
- An epoch is one pass over `train_data` (`SubsetRandomSampler`, `drop_last=True`, batch 4, `tasks.py:88-93`); Sintel is cached in RAM at construction (`sintel.py:248, 344-361`), so epochs are short and frequent.
- LR: `HyperParamScheduler` (`solver.py:931-1032`) steps 5e-5 → 5e-6 in 10 discrete levels over `n_iters` (`solver.py:1142-1148`, `scheduler.yaml:1-15`), applied only at epoch boundaries (`solver.py:379-381`).
- Loss to disk: `dir/validation/loss.h5` and `iteration.h5` (`solver.py:543-552`) — a plateau can be read by a script.
- No early stopping (only `iteration >= n_iters` and a NaN `OverflowError` per epoch, `solver.py:258-259, 373-375`). A checkpoint saved before `n_iters` is fully usable: `recover()` (`solver.py:572-638`) recomputes the scheduler at the stored iteration and clamps past the end (`solver.py:1024-1031`). The only consequence of stopping early is that the LR has not reached its floor.

## 4. Built-in speed features

None: no `torch.compile`, CUDA graphs, autocast/half, `torch.backends` settings, `set_num_threads`, or sparse backends anywhere in the package. `DataLoader`s in `tasks.py:88-121` pass no `num_workers` (so 0, synchronous); augmentation (`sintel.py:480`) runs on the CPU between GPU steps with nothing overlapping it.

## 5. Verdict per lever (scout's ranking, gain per unit of risk)

- **(a) Larger batch, fewer iterations, same sample budget.** Config only (`task.batch_size`, `task.n_iters`). Changes nothing about the model; changes the optimisation (the LR schedule is keyed to epochs, which shrink). Must be measured (time, memory) and validated (the 26-study targets), not assumed.
- **(c) Stop at the validation plateau.** A script over `validation/loss.h5`; no solver blocker. The report states the iteration used.
- **(e) `num_workers > 0` on the train loader.** Code-only, small gain, unmeasured.
- **(b) CUDA graphs / `torch.compile`.** Largest theoretical gain, highest risk: an internal refactor of the step that would need a control run for bit-level agreement.
- **(d) Larger dt.** Linear gain, direct scientific cost: `dt` is part of the reference's integration accuracy (`network.py:673-679` warns above 1/50); a setting to derive and justify, never a free speed-up.

## Gaps

No benchmark or profile was run. Sintel's `len(dataset)` (the epoch size) was not resolved. Lappalainen et al.'s stated reason for batch 4 was not checked. Whether `torch.compile` breaks on `AutoDeref` today was not tested.
