# Single-GPU throughput for a small DiT/SiT transformer (T4, Modal)

Scope: the 1.38M-param, 4-layer, width-128 transformer training step described below — not
this project's actual generator config, which should be checked against these levers
separately. Literature/source research only; nothing here was run or profiled on this
project's hardware. Sources are linked inline; PyTorch source citations are to the `main`
branch, fetched 2026-09-20 (line numbers drift with the branch). Setup: batch 32, inputs
`(32,16,8,721)` fp16 resident on GPU, AMP fp16 + `GradScaler`, fused AdamW, grad clip, EMA
every step, `nn.MultiheadAttention(need_weights=False)`, adaLN-Zero, T4 (sm75), Modal
container, cpu 1 / 12 GB. Measured: 20,000 steps in 1,128 s = 56.4 ms/step; `nvidia-smi`
reads 98.5% utilisation.

## TL;DR: 98.5% util is not evidence of compute-bound

`nvidia-smi`'s utilisation is time-occupancy, not FLOP-occupancy: NVIDIA defines it as "the
percent of time over the past sample period during which one or more kernels was executing
on the GPU," sampled roughly once a second ([nvidia-smi docs](https://docs.nvidia.com/deploy/nvidia-smi/index.html)).
It cannot see gaps shorter than its sampling window, and it does not know whether a running
kernel uses 1 SM or all of them. A stream that fires hundreds of tiny, low-occupancy kernels
back-to-back with never-quite-empty gaps reads as ~100% "busy" while doing a small fraction
of peak FLOPs. This is exactly the profile NVIDIA/PyTorch describe for launch-bound small-batch
jobs — see the Mask R-CNN trace in [Accelerating PyTorch with CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/),
"CPU is maxed out at 100% load while GPU is idle most of the time." 98.5% is consistent with
either regime; it does not distinguish them.

A back-of-envelope compute floor (assumptions stated, not a measurement): FLOPs/sample ≈
6·N·T for the linear layers (N=1.38M params, T=721 tokens; the standard transformer heuristic
from the Kaplan et al. scaling-law tradition) plus ≈12·T²·width·layers for the attention-score
matmuls 6ND doesn't count (QKᵀ and AV, x2 for backward, x4 layers, width=128) ≈ 5.97+3.2 = 9.2
GFLOP/sample ≈ 294 GFLOP/step at batch 32. T4 peak FP16 tensor-core throughput is **65 TFLOPS**
([NVIDIA T4 datasheet, PDF](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf)), so a 100%-MFU floor is ≈4.5 ms/step; a generous 20-30% MFU for
width-128 (skinny) matmuls gives ≈15-23 ms/step. Measured is 56.4 ms/step — 2.5-4x that
generous floor, 12x the naive one. Compatible with launch-bound, not proof of it; profile.

## 1. Batch size: 32 -> 128 -> 256 -> 512

Rule of thumb, not measured for this model: at batch 32 a step this small (4 layers, width 128)
is plausibly dominated by fixed per-kernel launch cost (~5-20 microseconds/kernel is the
commonly cited figure) times a kernel count that does **not** grow with batch — so throughput
(samples/s) usually rises steeply from 32->128->256 as the fixed cost amortises over more work
per kernel, then flattens once the step turns compute- or bandwidth-bound. Where that knee sits
depends on the actual kernel count/size, which only a profiler tells you. Optimizer and EMA cost
is O(#parameter tensors) not O(batch), and Python/dispatcher overhead per op is likewise
batch-independent — both shrink as a *fraction* of a larger step, which is why batch size is
the cheapest first lever, before touching `torch.compile`.

One real limiter at large batch here: SDPA's fused backends (flash / memory-efficient) are O(T)
memory, but a silent fallback to the "math" backend materialises the full attention matrix,
O(batch·heads·T²) — at batch 512, 4 heads, T=721, fp16, ≈512·4·721²·2 B ≈ 2.1 GB *per layer*.
Confirm the fused kernel is actually used (§4) before assuming batch 512 is free.

Diagnostics, in increasing order of effort:

```python
# 1. torch.profiler -- Chrome-trace gaps on the CUDA row are the CPU-bound signature
import torch
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

sched = schedule(wait=2, warmup=3, active=10, repeat=1)
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], schedule=sched,
             on_trace_ready=tensorboard_trace_handler("./prof_trace"), record_shapes=True) as prof:
    for step, batch in enumerate(loader):
        train_step(batch)          # forward + backward + opt.step + ema
        prof.step()
        if step >= 14: break
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
prof.export_chrome_trace("trace.json")   # chrome://tracing or ui.perfetto.dev
```
Small, evenly spaced whitespace between kernel blocks on the CUDA row = CPU-bound gaps ([HF "Profiling in PyTorch" guide](https://huggingface.co/blog/torch-profiler); the [PyTorch compile-profiling guide](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_profiling_torch_compile.html) states this is "more prevalent with small batch sizes").

```bash
# 2. nsys -- gaps at true timestamp resolution, with Python call stacks
nsys profile -o t4_step --trace=cuda,nvtx,osrt --pytorch=autograd-shapes-nvtx \
  --python-sampling=true --backtrace=none python train.py --steps 50
# open t4_step.nsys-rep in Nsight Systems; expand "CUDA HW", look for gaps repeating
# once per step (dev-discuss.pytorch.org/t/59)

# 3. cheapest smoke test: force synchronous launches
CUDA_LAUNCH_BLOCKING=1 python train.py --steps 50
# step time inflating 2-10x under blocking = async queue was hiding CPU dispatch behind
# GPU execution (launch-bound); little change = CPU was never ahead (compute-bound already)
```

```python
# 4. minimal batch-size sweep: the empirical answer to "how much do 32->512 gain?"
import time, torch
def bench(step_fn, bs, n_warmup=20, n_iters=100):
    for _ in range(n_warmup): step_fn(bs)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n_iters): step_fn(bs)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n_iters
    return dt, bs / dt          # s/step, samples/s -- plot the second column vs bs
```

## 2. torch.compile

| mode | what it does | compile cost | fits here? |
|---|---|---|---|
| `"default"` | Dynamo+Inductor fusion, modest autotuning | low (~seconds-tens of s) | safe first try |
| `"reduce-overhead"` | adds CUDA Graphs ("cudagraph trees") on top of default | low-moderate + graph warmup iters | targets exactly the launch-bound symptom |
| `"max-autotune"` | benchmarks multiple Triton kernel configs per op at compile time, + CUDA graphs | high (can be minutes; one report showed 0.113 ms to 3.25 ms *on the first call* for a small model — [PyTorch forum](https://discuss.pytorch.org/t/torch-compile-max-autotune-to-prevent-overwriting-clone-the-tensor-outside-of-torch-compile-or-call-torch-compiler-cudagraph-mark-step-begin-before-each-model-invocation/195151)) | worth it only once, since training runs 20k steps |

Definitions per the [torch.compile tutorial](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) and [Hugging Face's compile guide](https://huggingface.co/docs/transformers/en/perf_torch_compile).
Realistic speedups are model-dependent and not published for this exact size/GPU; a commonly
repeated **rule of thumb** (secondary source, A100-class, not verified for T4/1.4M params) is
1.3-1.8x for typical training loops, up to 2-3x with `reduce-overhead`/`max-autotune` when the
job is launch-bound — treat as a plausibility range, not a number to plan around; measure it.
Recompilation triggers on any shape change, so pad/drop the last partial batch — a silent
recompile costs seconds and defeats `reduce-overhead`'s whole premise.

Known breakages relevant to this stack:
- **`nn.MultiheadAttention`**: historically graph-break prone under Dynamo; reported (secondary
  source, a newsletter digest, not independently verified) materially improved PyTorch 2.4->2.6
  — check `TORCH_LOGS=graph_breaks` on the installed version before trusting this.
- **`GradScaler`**: `scaler.step()`/`scaler.update()` contain a host sync (`found_inf.item()`)
  and are normally kept **outside** the compiled region regardless of mode — matters more once
  `reduce-overhead` is in play (§3).
- **EMA**: `torch._foreach_lerp_` is a first-class Inductor op with explicit horizontal-fusion
  support ([`foreach_map` tutorial](https://docs.pytorch.org/tutorials/recipes/foreach_map.html)), so it compiles fine; folding it into the same graphed region as the
  train step is a design choice, not a requirement.

```python
model = torch.compile(model, mode="reduce-overhead")
# keep scaler.step(optimizer) / scaler.update() eager and outside the compiled call,
# regardless of mode.
```

## 3. CUDA graphs, and why `reduce-overhead` is the direct answer to "launch-bound"

CUDA graphs record a fixed sequence of kernels once and replay them with a single
`cudaGraphLaunch`, skipping Python/C++/driver argument setup on every subsequent step — the
exact cost this job is suspected of paying per kernel, every step, 20,000 times. NVIDIA/PyTorch's
own MLPerf v1.0 numbers: Mask R-CNN's graphed backbone went 31 ms->6 ms/iteration (5x) at batch
1, for 1.70x end-to-end at 272 GPUs; BERT got 1.12x at 4096 GPUs, where per-kernel time was
already less overhead-dominated at that batch size ([Accelerating PyTorch with CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)) — "particularly
true for workloads launching many short kernels with small batches" (their DLRM example).
`reduce-overhead` is Inductor's automated version of the same mechanism.

Caveats, quoted from the [PyTorch CUDA semantics notes](https://docs.pytorch.org/docs/2.9/notes/cuda.html):
- Fixed memory addresses and static shapes only: "every replay reads from and writes to the
  same (virtual) memory addresses" and "dynamic shapes are prohibited."
- "Ops that synchronize the CPU with the GPU (e.g. `.item()` calls) are prohibited" during
  capture — this directly hits `GradScaler`: "`GradScaler.step` syncs the CPU with the GPU,
  which is prohibited during capture." Workaround: capture forward+backward (+ foreach EMA,
  +foreach/fused optimizer *if* using `capturable=True`), run `scaler.step()`/`scaler.update()`
  eagerly outside `g.replay()`.
- Warm up "a few eager iterations" on a side stream before capture; dynamic control flow is
  prohibited; multiple RNG generators need `CUDAGraph.register_generator_state`.
- Under `torch.compile(mode="reduce-overhead")`, watch for the documented "accessing tensor
  output of CUDAGraphs that has been overwritten by a subsequent run" trap — call
  `torch.compiler.cudagraph_mark_step_begin()` or clone outputs kept across steps ([issue #148439](https://github.com/pytorch/pytorch/issues/148439)).

```python
# manual capture, adapted from pytorch.org/blog/accelerating-pytorch-with-cuda-graphs
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(static_x, static_cond)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()          # eager -- stays outside capture
torch.cuda.current_stream().wait_stream(s)

g = torch.cuda.CUDAGraph()
opt.zero_grad(set_to_none=True)
with torch.cuda.graph(g):
    with torch.autocast("cuda", dtype=torch.float16):
        static_loss = model(static_x, static_cond)
    static_loss.backward()                                        # fills static .grad in place
    torch._foreach_lerp_(ema_params, model_params, 1 - ema_decay)  # capture-safe, no sync

for x, cond in real_batches:
    static_x.copy_(x); static_cond.copy_(cond)
    g.replay()
    scaler.step(opt); scaler.update()              # the one sync point per step
```

## 4. Does `nn.MultiheadAttention(need_weights=False)` reach SDPA? Which backend on sm75?

Yes, but via the ordinary path, not the "fast path" — the distinction matters for training.
`MultiheadAttention.forward` first checks a `why_not_fast_path` chain
([`torch/nn/modules/activation.py`](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/activation.py), around lines 1327-1454). Two of its conditions are
unconditional blockers for this setup: `elif self.training: why_not_fast_path = "training is
enabled"` and `elif torch.is_autocast_enabled(): why_not_fast_path = "autocast is enabled"` —
plus a third, independent check requiring `not (torch.is_grad_enabled() and any(...requires_grad))`.
**Training with AMP autocast disqualifies the fast path outright**, `need_weights` notwithstanding;
that fast path is for inference and dispatches to the fully-fused native op
`torch._native_multi_head_attention`.

When the fast path is blocked, `forward` calls `F.multi_head_attention_forward`
([`torch/nn/functional.py`](https://github.com/pytorch/pytorch/blob/main/torch/nn/functional.py), line 6677), which branches on `need_weights` independently of
training mode: `need_weights=True` (line 7063) does a manual `baddbmm` then `softmax` then
`bmm`, materialising the full attention matrix; `need_weights=False` (the `else` at line 7100)
reshapes q/k/v and calls, unconditionally, `scaled_dot_product_attention(q, k, v, attn_mask,
dropout_p, is_causal)` at line 7117 — fully autograd- and autocast-aware, no eval/no-grad
requirement. So `need_weights=False` **does** get the fused SDPA kernel during training; what
it skips is only the extra fusion of the in/out projections that the inference fast path adds.
The commonly repeated advice "set `need_weights=False` for best performance" ([docs](https://docs.pytorch.org/docs/2.9/generated/torch.nn.MultiheadAttention.html); [issue #100347](https://github.com/pytorch/pytorch/issues/100347))
is really about avoiding the O(T²) explicit-weights path, with or without training mode.

Which SDPA backend actually runs, from [`aten/src/ATen/native/transformers/cuda/sdp_utils.cpp`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp) (`main`, fetched 2026-09-20):

| backend | hardware gate (source string) | T4 (sm75) | L4 (sm89) |
|---|---|---|---|
| Flash Attention | `check_flash_attention_hardware_support`: "only supports gpu architectures in the range [sm80, sm121]" | **unavailable** | available |
| Memory-Efficient | `check_mem_efficient_hardware_support`: "[sm50, sm121]" | available (this is what SDPA picks) | available |
| Math (fallback) | none (always available) | available, slowest, upcasts fp16 intermediates to fp32 | available |

This matches the prompt's framing exactly: on T4, SDPA's automatic backend selection (force or
inspect it via the [`sdpa_kernel`](https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.scaled_dot_product_attention.html) context manager) lands on memory-efficient attention
(xFormers-derived), not flash; on L4 flash attention becomes eligible and is usually fastest.
The source also flags extra head-dim restrictions on the flash backward kernel for sm86/89/12.x
not fully enumerated here — check current source if L4 flash attention silently falls back.

## 5. Optimizer/EMA overhead

Three PyTorch optimizer-step implementations, increasing kernel-fusion order: for-loop
(single-tensor, one-or-more kernels *per parameter tensor* per math op), `foreach`
(multi-tensor, one kernel per op-kind for the *whole* list), `fused` (the whole update in
one-or-few kernels) — "fused > foreach > for-loop" ([GPU MODE Lecture 6](https://christianjmills.com/posts/cuda-mode-notes/lecture-006/); [`foreach_map` tutorial](https://docs.pytorch.org/tutorials/recipes/foreach_map.html)).
`fused=True` is supported for AdamW on fp16/fp32/fp64/bf16 ([docs](https://docs.pytorch.org/docs/stable/generated/torch.optim.adamw.AdamW.html)), but check rather than trust
the label: fused AdamW has been reported to give worse loss than unfused under fp16/AMP
([#96755](https://github.com/pytorch/pytorch/issues/96755)), and *slower* than foreach in another report ([#121857](https://github.com/pytorch/pytorch/issues/121857)) — benchmark foreach vs fused directly.

EMA: `torch._foreach_lerp_(ema_params, model_params, 1 - decay)` replaces a Python for-loop
`p_ema.mul_(decay).add_(p_model, alpha=1-decay)` with one multi-tensor call. One third-party
measurement (not verified against this model): 1-8.4x faster than the naive loop ([fastxtend EMA docs](https://fastxtend.benjaminwarner.dev/callback.ema.html)).

Rough accounting (not measured — get the real count from `sum(1 for _ in model.parameters())`):
a 4-layer transformer with adaLN-Zero MLPs plausibly has several dozen to ~100 parameter
tensors. Naive for-loop optimizer/EMA at ~5-20 microseconds/kernel ([breakdown, secondary source](https://medium.com/@bhagyarana80/7-pytorch-compile-moves-that-slash-training-time-cb664e666d14))
plausibly costs low-single-digit ms/step for *each* of optimizer-step and EMA-update if
unfused; `foreach`/`fused` collapse each to a handful of kernels (tens of microseconds). Real,
but likely secondary to the model's own per-layer kernel count (§1) and AMP/GradScaler
bookkeeping — worth fixing (nearly free), but don't expect it alone to close 56 ms vs ~20 ms.

## 6. T4 vs L4

| | T4 (sm75, Turing) | L4 (sm89, Ada) | source |
|---|---|---|---|
| FP16 Tensor Core (dense) | 65 TFLOPS | 121 TFLOPS (242 with structured sparsity) | [NVIDIA T4 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf), [L4 datasheet via Lenovo Press](https://lenovopress.lenovo.com/lp1717-thinksystem-nvidia-l4-24gb-pcie-gen4-passive-gpu) |
| FP32 | 8.1 TFLOPS | 30.3 TFLOPS | same |
| Memory bandwidth | **300 GB/s** | **300 GB/s** | same, corroborated independently by [Modal's L4 post](https://modal.com/blog/nvidia-l4-price-article) |
| Memory | 16 GB GDDR6 | 24 GB GDDR6 | same |
| TDP | 70 W | 72 W | same |
| bf16 tensor-core hw | **no** — Turing has no bf16 hardware path | yes (Ampere+) | ["Neither Volta nor Turing has any bf16 hardware"](https://github.com/lablup/mlxcel/issues/1542); BF16 requires `__CUDA_ARCH__>=800` |
| Flash Attention (PyTorch SDPA) | no (sm<80) | yes | §4, `sdp_utils.cpp` |
| Modal price | $0.59/hr | $0.80/hr (ratio 1.36x, matches the prompt) | [T4](https://modal.com/blog/nvidia-t4-price-article), [L4](https://modal.com/blog/nvidia-l4-price-article) |

The load-bearing fact: **identical memory bandwidth**. Every bandwidth-bound elementwise op in
this step — LayerNorm/adaLN modulation, residual adds, dropout, AMP casts, the optimizer and
EMA updates — gets zero benefit from L4; only the tensor-core matmul/attention-score portion
sees up to ~1.86x (dense) from the compute-ratio. If the step is genuinely launch-bound
(§ TL;DR), moving to L4 without first fixing overhead (§§2-3) plausibly buys much less than
1.86x: a faster GPU finishes each tiny kernel *sooner*, making the fixed CPU-side launch cost a
*larger*, not smaller, share of a shorter GPU busy time. No independent (non-vendor) T4-vs-L4
training benchmark for a model this size was found (Gaps); NVIDIA's "2.5X the generative AI
performance of the T4" is inference-workload marketing, not a training number. Practical order:
fix launch-overhead on T4 first (free), remeasure, then decide whether the remaining
compute-bound residual justifies 1.36x the price for L4's ~1.86x dense FP16 compute.

## 7. What not to bother with, and why (tied to this model's size and step time)

- **DDP**: synchronizes gradients *across* GPUs. One GPU has nothing to synchronize with;
  wrapping single-process training in DDP adds reducer/bucketing overhead for zero speedup.
- **ZeRO/FSDP**: shards optimizer state/gradients/parameters across ranks because they don't fit
  one GPU. At 1.38M params, fp32 weights+grad+Adam(m,v)+EMA is tens of MB total — against 16 GB
  (T4) or 24 GB (L4), ~0.1-0.3% of one device. Nothing to shard; the machinery only adds overhead.
- **Gradient checkpointing**: trades ~1.5-2x recompute ([Aman's AI Journal](https://aman.ai/primers/ai/grad-accum-checkpoint/)) for activation memory. This
  step's activations (batch 32, 721 tokens, width 128, 4 layers) are plausibly tens to
  low-hundreds of MB, nowhere near either card's ceiling — checkpointing buys nothing here and
  adds a clean 20-100% recompute tax to an already CPU-suspect step.
- **Sequence parallelism**: splits the token dimension for sequences too long for one device
  (thousands-tens of thousands of tokens). At T=721 the full per-layer attention matrix is
  roughly 1-2 MB/sample — no long-sequence problem to solve, and one device to split across.

All four exist to solve "doesn't fit on, or doesn't fit across, one device." This job's whole
symptom is the opposite: a device with capacity to spare, plausibly spending most of its time
on the CPU side of the kernel-launch boundary. The fix is fewer, larger, longer-running GPU
ops (batching, fusion, graphs), not more devices or more recompute.

## Gaps

- No benchmark was run for this exact model/GPU/batch combination — every "expected"
  batch-scaling and `torch.compile` speedup number here is a rule of thumb or a different
  model's measurement, explicitly flagged inline; profiling this step directly (§1) is the
  only way to get real numbers.
- No independent (non-vendor) T4-vs-L4 *training* throughput comparison was found; only vendor
  TFLOPS specs, Modal's pricing posts, and one NVIDIA inference-marketing ratio.
- The `nn.MultiheadAttention` + `torch.compile` graph-break improvement claimed for PyTorch 2.4
  to 2.6 traces to a secondary newsletter digest, not a primary changelog/PR, and the "6 kernels
  / 6000 launches" AdamW breakdown (§5) is from a secondary blog, not PyTorch source — neither
  independently confirmed, both used only for order-of-magnitude intuition.
- Exact PyTorch/CUDA versions in the Modal container were not given and not assumed; SDPA
  backend gating and the `capturable`/fused-optimizer CUDA-graph story move between minor
  versions — re-check `sdp_utils.cpp` and `torch.optim.adamw` against the container's installed
  version before relying on the line numbers cited above, including Flash Attention's extra
  head-dim restriction on sm86/89/12.x backward kernels (relevant only if moving to L4).
- This model's actual parameter-tensor count, and how much of the 56.4 ms/step is genuinely
  attention vs optimizer vs AMP-cast vs Python dispatch, were not measured — that split is
  exactly what §1's profiler/nsys commands are for.
