

## summary

FlyVis does NOT forbid input gradients anywhere — encoder inversion works on the stock package, no fork needed. `Network.forward` (network.py:509) is a plain differentiable Euler loop over frames, and nothing on the path stimulus→activity is wrapped in `torch.no_grad()` or calls `detach()`. `Network.simulate` (network.py:628) looks like it should block grad but does not: its `with simulation(self)` (utils/nn_utils.py:10) only flips `p.requires_grad=False` on the *parameters* and sets `training=False`; ambient grad mode is untouched, so a `requires_grad` video passed to `simulate()` comes back with a `grad_fn` and backprops correctly (measured |grad| nonzero on every frame, and an analytic-vs-finite-difference directional check agreed to 8e-5 relative). `enable_grad(grad)` (network.py:699) is just a `torch.set_grad_enabled` context manager; `grad=False` is the *default* on `steady_state`, `fade_in_state` and `stimulus_response`, so those four call sites are the only places grad is switched off, plus a hard `torch.no_grad()` in `current_response` (network.py:841) and `@torch.no_grad()` on `Solver.test` (solver.py:473) and `analysis.validation.validate` (validation.py:20) — none of which you need. The one real trap is the `Stimulus` buffer: it is created as a plain non-requires-grad leaf (`torch.zeros`, stimulus.py:161) and `add_input` does an in-place `+=` into it (stimulus.py:206), which is fine and makes the buffer a non-leaf with `IndexPutBackward0` — but `zero()` takes an early-return `buffer.zero_()` path (stimulus.py:155-160) when the shape is unchanged, which *chains* the new graph onto the previous iteration's, so from iteration 2 on you get "Trying to backward through the graph a second time". That is exactly why flyvis's own MEI code passes `retain_graph=True` (optimal_stimuli.py:160, 262); the correct fix is a fresh buffer per step. Cost at 721 columns, batch 1: 6.0 MB of graph per frame (= n_edges × 4 B), i.e. 0.30 GB / 0.5 s per iteration at 50 frames and 1.2-1.4 GB / 3.0 s at 200 frames on a 16-thread CPU. Gradient checkpointing is not provided by the package but is trivial to add (the recurrent carry is a single tensor, `state.nodes.activity`) and I verified a 25-frame-chunk version: graph memory drops below measurement noise and it is gradient-identical to 8e-8 relative.

## code_example

"""Verified end-to-end: encoder inversion against a stock flyvis network.
Run as-is; loss decreased monotonically over 5 Adam steps in my test
(0.791089 -> 0.738695 at 50 frames, 721 columns, dt=1/100)."""
import torch
import flyvis
from flyvis import NetworkView
from flyvis.network.stimulus import Stimulus

dt, n_frames, batch = 1 / 100, 50, 1
net = NetworkView(flyvis.results_dir / "flow/0000/000").init_network()
net.eval()
for p in net.parameters():                  # weights are constants for inversion
    p.requires_grad_(False)                 # (also makes forward's clamp() a no-op)

# ONCE, before the loop: steady_state resizes the shared net.stimulus buffer.
initial_state = net.steady_state(1.0, dt, batch, value=0.5)   # grad=False by default

n_hexals = net.stimulus.n_input_elements    # 721 at extent=15
target = your_target_activity               # (batch, n_frames, net.n_nodes) or a slice of it

# the leaf you optimise: (batch, frames, 1, hexals), NOT the n_nodes buffer
video = torch.full((batch, n_frames, 1, n_hexals), 0.5, requires_grad=True)
opt = torch.optim.Adam([video], lr=1e-2)

stim = Stimulus(net.connectome, batch, n_frames, init_buffer=False)  # 33 ms to build

for step in range(n_steps):
    opt.zero_grad(set_to_none=True)
    # A FRESH buffer every step. Do NOT call stim.zero(): with an unchanged shape it
    # takes the buffer.zero_() path and chains this step's graph onto the last one,
    # so backward() raises "Trying to backward through the graph a second time".
    if hasattr(stim, "buffer"):
        del stim.buffer                     # 0.5 ms; add_input then allocates a new one
    stim.add_input(video)                   # buffer := non-leaf, grad_fn=IndexPutBackward0
    activity = net(stim(), dt, state=initial_state)   # (batch, n_frames, n_nodes)
    loss = (activity - target).pow(2).mean()
    loss.backward()                         # NO retain_graph needed
    opt.step()
    with torch.no_grad():
        video.clamp_(0, 1)                  # projected gradient descent on luminance

# ---------------------------------------------------------------- 200+ frames:
# checkpoint over frame chunks. Verified gradient-identical to the plain loop
# (max abs diff 2.9e-11, 8e-8 relative) with graph memory below measurement noise.
from torch.utils.checkpoint import checkpoint
from flyvis.utils.tensor_utils import AutoDeref

def simulate_checkpointed(net, video, dt, initial_state, chunk=25):
    stim = Stimulus(net.connectome, *video.shape[:2], init_buffer=False)
    stim.add_input(video)
    x, params = stim(), net._param_api()
    def segment(activity, x_seg):           # only ONE carried tensor: nodes.activity
        state = net._state_api(AutoDeref(nodes=AutoDeref(activity=activity),
                                         edges=AutoDeref()))
        ys = []
        for i in range(x_seg.shape[1]):
            state = net._next_state(params, state, x_seg[:, i], dt)
            ys.append(state.nodes.activity)
        return torch.stack(ys, dim=1), state.nodes.activity
    activity, outs = initial_state.nodes.activity, []
    for s in range(0, x.shape[1], chunk):
        ys, activity = checkpoint(segment, activity, x[:, s:s + chunk],
                                  use_reentrant=False)
        outs.append(ys)
    return torch.cat(outs, dim=1)


## api

[
 {
  "name": "Network.forward",
  "file": "D:\\ML\\Fly_Brain\\.venv\\Lib\\site-packages\\flyvis\\network\\network.py:509",
  "signature": "forward(x: Tensor, dt: float, state: AutoDeref = None, as_states: bool = False) -> Tensor | AutoDeref",
  "returns": "Tensor (batch, n_frames, n_nodes) float32, n_nodes=45669 at extent=15; requires_grad=True and grad_fn set whenever x requires grad. With as_states=True, a list of AutoDeref states.",
  "notes": "THE differentiable entry point. No no_grad, no detach. Body is `for i in range(x.shape[1]): state = self._next_state(params, state, x[:, i], dt)` then `torch.stack(..., dim=1)` (lines 535-546), so the graph is exactly n_frames Euler steps deep. Calls self.clamp() first (line 527), which is a no-op once parameters have requires_grad=False. x is the WHOLE-NETWORK buffer of shape (batch, frames, n_nodes), not a video."
 },
 {
  "name": "Network.simulate",
  "file": "D:\\ML\\Fly_Brain\\.venv\\Lib\\site-packages\\flyvis\\network\\network.py:628",
  "signature": "simulate(movie_input: Tensor, dt: float, initial_state='auto', as_states=False, as_layer_activity=False) -> Tensor",
  "returns": "Tensor (batch, n_frames, n_nodes), differentiable w.r.t. movie_input (verified). movie_input must be (batch, frames, 1, hexals).",
  "notes": "Does NOT block input gradients. `with simulation(self)` (line 684) only zeroes parameter requires_grad; the assert at 685-687 is self-fulfilling (the context manager sets the conditions it asserts), so the docstring's 'Raises ValueError if any parameters require grad' never fires. Usable for inversion, with two caveats: (a) it mutates the shared self.stimulus (lines 688-689) so a loop with a fixed initial_state dies on iteration 2 (see gotchas); (b) initial_state='auto' re-runs a 100-frame steady_state every call."
 },
 {
  "name": "Network.enable_grad",
  "file": "D:\\ML\\Fly_Brain\\.venv\\Lib\\site-packages\\flyvis\\network\\network.py:699",
  "signature": "@contextmanager enable_grad(grad: bool = True)",
  "returns": "context manager; yields None",
  "notes": "Nothing flyvis-specific: saves torch.is_grad_enabled(), calls torch.set_grad_enabled(grad), restores on exit (lines 705-710). It is a global switch, not per-module. enable_grad(True) does not 'turn on' input gradients — grad is already on by default; its only real use in the package is the grad=False default of steady_state (583), fade_in_state (625) and stimulus_response (756)."
 },
 {
  "name": "Network._next_state",
  "file": "D:\\ML\\Fly_Brain\\.venv\\Lib\\site-packages\\flyvis\\network\\network.py:377",
  "signature": "_next_state(params, state, x_t: Tensor, dt: float) -> AutoDeref",
  "returns": "AutoDeref with .nodes.activity (batch, n_nodes) plus derived .sources/.targets RefTensors",
  "notes": "One explicit Euler step: `next = state + vel*dt` (lines 404-411), then _state_api. Fully differentiable, no detach. For PPNeuronIGRSynapses the only state variable is nodes.activity (state.edges is EMPTY — verified), which is what makes checkpointing easy."
 },
 {
  "name": "Network._state_api",
  "file": "D:\\ML\\Fly_Brain\\.venv\\Lib\\site-packages\\flyvis\\network\\network.py:415",
  "signature": "_state_api(state) -> AutoDeref",
  "returns": "AutoDeref(nodes, edges, sources, targets)",
  "notes": "Re-derives per-edge source/target views from nodes each step via RefTensor. Needed if you write your own checkpointed loop: call it on a bare AutoDeref(nodes=AutoDeref(activity=...), edges=AutoDeref()) to rebuild a valid state from just the activity tensor. Also the hook point for register_state_hook (network.py:444)."
 },
 {
  "name": "Network.target_sum",
  "file": "D:\\ML\\Fly_Brain\\.venv\\Lib\\site-packages\\flyvis\\network\\network.py:333",
  "signature": "target_sum(x: Tensor) -> Tensor",
  "returns": "Tensor (batch, n_nodes)",
  "notes": "`torch.zeros(...).scatter_add_(-1, idx, x)` — the in-place scatter into a fresh non-requires-grad zeros tensor is differentiable (this is the dominant memory cost: the n_edges-sized x is saved per step)."
 },
 {
  "name": "Network.steady_state",
  "file": "D:\\ML\\Fly_Brain\\.venv\\Lib\\site-packages\\flyvis\\network\\network.py:548",
  "signature": "steady_state(t_pre, dt, batch_size, value=0.5, state=None, grad: bool = False, return_last=True) -> AutoDeref",
  "returns": "AutoDeref state; with grad=False (default) all tensors have requires_grad=False",
  "notes": "grad=False wraps the run in enable_grad(False) (line 583) — correct and harmless for inversion, since the grey-screen initial state does not depend on your video. BUT it calls self.stimulus.zero(batch_size, int(t_pre/dt)) and add_pre_stim (580-581), RESIZING the shared net.stimulus buffer to (batch, 100, 45669) at dt=1/100 — verified clobbering. Call it ONCE before the optimisation loop."
 },
 {
  "name": "Network.stimulus_response",
  "file": "D:\\ML\\Fly_Brain\\.venv\\Lib\\site-packages\\flyvis\\network\\network.py:712",
  "signature": "stimulus_response(stim_dataset, dt, indices=None, t_pre=1.0, t_fade_in=0.0, grad: bool = False, default_stim_key='lum', batch_size=1)",
  "returns": "generator of (stimulus ndarray, response) — response is .detach().cpu().numpy() when grad=False (786-792), the live Tensor when grad=True (793-797)",
  "notes": "grad=True only omits the detach on the OUTPUT. The stimulus itself comes out of a torch DataLoader (747-749), so there is no leaf you own to optimise — this is not the inversion path. Useful only for reading gradients w.r.t. parameters on dataset stimuli."
 },
 {
  "name": "Network.current_response",
  "file": "D:\\ML\\Fly_Brain\\.venv\\Lib\\site-packages\\flyvis\\network\\network.py:801",
  "signature": "current_response(stim_dataset, dt, indices=None, t_pre=1.0, t_fade_in=0, default_stim_key='lum')",
  "returns": "generator of (stim, activity, currents) numpy arrays",
  "notes": "THE ONE method that hard-blocks gradients: `with torch.no_grad():` at line 841, no grad flag. If you need per-edge currents differentiably, call self.dynamics.currents(state, params) yourself on states from forward(..., as_states=True)."
 },
 {
  "name": "flyvis.utils.nn_utils.simulation",
  "file": "D:\\ML\\Fly_Brain\\.venv\\Lib\\site-packages\\flyvis\\utils\\nn_utils.py:10",
  "signature": "@contextmanager simulation(network: nn.Module)",
  "returns": "context manager",
  "notes": "Sets network.training=False and p.requires_grad=False for every named parameter, restoring both on exit (lines 34-45). Critically it does NOT enter no_grad — so input gradients survive it. Its docstring ('temporarily disables gradient computation') is wrong."
 },
 {
  "name": "Stimulus.__init__ / .zero / .add_input / .__call__",
  "file": "D:\\ML\\Fly_Brain\\.venv\\Lib\\site-packages\\flyvis\\network\\stimulus.py:111",
  "signature": "Stimulus(connectome, n_samples=1, n_frames=1, init_buffer=True); zero(n_samples=None, n_frames=None); add_input(x, start=None, stop=None, n_frames_buffer=None, cumulate=False); __call__() -> Tensor",
  "returns": "buffer: Tensor (n_samples, n_frames, n_nodes) float32 on flyvis.device; 9.1 MB at 50 frames / 45669 nodes",
  "notes": "zero() at :142 — line 161 `self.buffer = torch.zeros(...)` makes a NON-requires-grad LEAF; lines 155-160 are an early-return `self.buffer.zero_()` fast path taken when the shape is unchanged, which also forgets to reset self._nonzero (only line 162 does). add_input() at :173 — line 206 `self.buffer[:, slice, self.input_index] += x.to(...)`, advanced-index in-place add; because the buffer is a non-requires-grad leaf this is legal and turns it into a non-leaf with IndexPutBackward0, giving a clean path back to x. __call__() at :249 just returns the buffer, no detach. input_index has shape (8, 721): the video is broadcast to all eight photoreceptor types R1-R8."
 },
 {
  "name": "GenerateOptimalStimuli.artificial_optimal_stimuli / FindOptimalStimuli.regularized_optimal_stimuli",
  "file": "D:\\ML\\Fly_Brain\\.venv\\Lib\\site-packages\\flyvis\\analysis\\optimal_stimuli.py:193",
  "signature": "artificial_optimal_stimuli(cell_type, t_stim=49/200, dt=1/100, lr=1e-2, ..., n_iters=200) -> GeneratedOptimalStimulus",
  "returns": "dataclass with the optimised stimulus, responses and loss history (numpy)",
  "notes": "The package's own working proof that input gradients are supported — MEI/most-exciting-input optimisation, i.e. inversion with a different objective. Read it as the reference recipe: freeze params (lines 190-191), `art_opt_stim.requires_grad = True` (234), Adam over the stimulus (237), stimulus.zero()+add_input inside the loop (247-248), forward (253), backward (262). Note it passes retain_graph=True (160 and 262), which is a workaround for the buffer-chaining bug, not a requirement."
 }
]

## gotchas

[
 "THE blocker, and it is not no_grad: reusing one Stimulus (or net.stimulus) across optimisation steps breaks backward from step 2 on. `Stimulus.zero()` (stimulus.py:155-160) early-returns with `self.buffer.zero_()` when the shape is unchanged; on a buffer that is now a graph node this prepends a ZeroBackward0 and keeps the previous step's graph attached (verified: buffer graph grows by 6 autograd nodes per iteration). Iteration 2 then raises RuntimeError: Trying to backward through the graph a second time. Fix: `del stim.buffer` (0.5 ms) or a new Stimulus (33 ms) before each add_input. `retain_graph=True`, which flyvis itself uses at optimal_stimuli.py:160 and :262, makes it run but holds two full graphs at peak (~2x activation memory) — do not copy that pattern.",
 "The same bug bites `net.simulate()` in a loop: with a fixed `initial_state` it fails on iteration 2 (measured). With `initial_state='auto'` it happens to work only because steady_state resizes the buffer to (batch, 100, n_nodes) each call, forcing a fresh allocation — at the cost of an extra 100-frame simulation per iteration. If you use simulate() in a loop, `del net.stimulus.buffer` first.",
 "`steady_state()` and `fade_in_state()` MUTATE the shared `net.stimulus` buffer (network.py:580-581, 617-624), resizing it to int(t_pre/dt) frames. Verified: a buffer holding your 8-frame stimulus comes back 100 frames long. Compute the initial state once, before the loop, and keep your optimisation buffer in a separate Stimulus instance.",
 "Do NOT set `requires_grad=True` on the buffer itself. `stim.buffer.requires_grad_(True)` then `add_input` fails with `RuntimeError: a leaf Variable that requires grad is being used in an in-place operation` — and add_input's `except RuntimeError` (stimulus.py:209-212) swallows it and re-raises a completely misleading 'input has shape ... but buffer has shape ...' message. Put requires_grad on the video you pass in; the buffer picks up the graph from the in-place add.",
 "`simulate()`'s docstring claims it raises ValueError if any parameter requires grad. It cannot: `with simulation(self)` sets requires_grad=False on all parameters immediately before the assert that checks it (network.py:684-687). Likewise `simulation`'s own docstring says it 'temporarily disables gradient computation' — it does not. Don't trust these docstrings; the code is what I verified.",
 "`forward()` calls `self.clamp()` (network.py:527), which only acts on parameters with requires_grad=True. Freezing the parameters for inversion silently disables both the non-negativity clamp on syn_strength and the symmetry averaging. That is what you want during inversion (weights are fixed), but if you ever co-optimise the video and the weights, call net.clamp() explicitly after each opt.step().",
 "`current_response()` is the one method with an unconditional `torch.no_grad()` (network.py:841) and no grad flag. `Solver.test` (solver.py:473) and `analysis.validation.validate` (validation.py:20) carry `@torch.no_grad()` decorators. None is on the stimulus->activity path, so none of them constrains inversion; just don't route through current_response if you need differentiable per-edge currents.",
 "`stimulus_response(grad=True)` is not an inversion route: the stimulus comes from a torch DataLoader (network.py:747), so there is no leaf tensor you own. grad=True only omits the .detach() on the returned activity.",
 "flyvis sets a global default device at import (`torch.set_default_device(device)`, __init__.py:13-14), so bare `torch.zeros`/`torch.rand` land on flyvis.device. `add_input` does `x.to(self.buffer.device)` — differentiable, so a cross-device video still backprops, but it costs a copy per step.",
 "The gradient reaches all eight photoreceptor types at once: `input_index` has shape (8, 721) (stimulus.py:128-131), so one video channel is broadcast to R1-R8. You cannot optimise per-photoreceptor-type input through add_input; write into the buffer yourself if you need that.",
 "Everything is float32. My finite-difference check needed a directional derivative (random unit direction) to be meaningful — per-pixel central differences at eps=1e-3 fall under float32 resolution and look wrong (one probe gave fd exactly 0.0 against a true 2.4e-5). Don't conclude the gradient is broken from a single-pixel FD probe.",
 "Building a Network on Windows needs the datamate `_write_h5` patch this project already has in D:\\ML\\Fly_Brain\\flydream\\model\\__init__.py (datamate unlinks an h5 file it still holds open -> WinError 32). Import that and call configure_flyvis_root() before importing flyvis, as flydream/model/zero.py:21-23 does."
]

## unknowns

[
 "GPU timings. This box is torch 2.14.0+cpu with no CUDA, so all times below are 16-thread CPU: batch 1, 721 columns, n_nodes=45669, n_edges=1513231. 50 frames: forward 0.26-0.36 s, backward 0.17-0.21 s (~0.5 s/iteration). 200 frames: forward 1.4-2.0 s, backward 1.5-1.6 s (~3.0 s/iteration, so 500 Adam steps ~25 min). The memory numbers are device-independent (they are just saved-tensor bytes) but the wall-clock ratio on a GPU is unmeasured — the per-step work is a 1.5M-element gather/multiply plus a scatter_add, which is bandwidth-bound and should scale well, but I did not verify it.",
 "Peak-memory numbers are process RSS deltas (psutil), not an allocator-level peak: graph memory measured 5.7-6.5 MB per frame per sample, converging on 6.05 MB = n_edges x 4 B, i.e. one n_edges-sized float32 tensor saved per Euler step. 50 frames = +0.30 GB (peak working set 1.02 GB including the 0.7 GB network+connectome); 200 frames = +1.22 to 1.36 GB (peak 2.13 GB). Batch scales linearly (batch 4 x 50 frames = +1.19 GB), so 200 frames x batch 4 ~ 4.8 GB of graph — that is the figure to size a GPU against. A CUDA run should be re-measured with torch.cuda.max_memory_allocated.",
 "Whether truncated BPTT is needed for *optimisation quality* (not memory) at 200+ frames — i.e. whether gradients through a 200-step Euler chain vanish or explode for this network — I did not test. Gradient norms were stable and finite at 8/25/50/100/200 frames (|dL/dvideo| ~1.0e3 for a last-frame objective, no blow-up), but I ran no long-horizon convergence experiment. Bauer/Margrie/Clopath's own horizon choice should decide this, not my measurement.",
 "The package offers NOTHING for checkpointing or truncation: no torch.utils.checkpoint import anywhere, no chunking flag on forward/simulate, no truncation option. The `checkpoint` hits in the grep are all model-weight checkpoints (Solver, NetworkView, analysis/response_norms.py), unrelated. My simulate_checkpointed() above is my own code, not library API, and it uses two private methods (net._param_api, net._state_api) plus the fact that PPNeuronIGRSynapses carries only nodes.activity as state (state.edges is empty — verified). A different NetworkDynamics subclass with edge state would need the carry extended, and private-method signatures could change across flyvis versions (this is 1.2.x as installed).",
 "Whether my chunked version is faster than the plain loop on a GPU. On CPU at 200 frames it was *faster* end to end (0.86 s fwd + 1.66 s bwd = 2.52 s, vs 1.43 + 1.58 = 3.01 s plain) because not saving 1.2 GB of tensors beats recomputing the forward. On a GPU with memory to spare the usual ~30% recompute penalty may reappear. Pick chunk size empirically; 25 and 10 were both correct and both collapsed the graph memory."
]