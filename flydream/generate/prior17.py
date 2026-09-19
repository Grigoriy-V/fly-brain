"""17.1: the prior over reachable T4/T5 states — flow matching in state space.

    modal run deploy/modal/generate_app.py --train17-run          # priced, one T4

13B answers `state → video`; it does not model `p(state)`, so a state can only
be read off a video (ROADMAP 17). This module learns that distribution
directly, in the representation 13B already consumes: the (40, 8, 721) map of
the eight T4/T5 types, z-scored per type over the training split
(`maps_deep.npz`, built by `maps13b`). A sample is therefore fed to 13B
unchanged.

Objective: the same linear interpolant as 13B (`gen13b`), with the state as
the data and **no condition** — x_t = (1−t)·ε + t·x₁, the model predicts
v = x₁ − ε, loss MSE, logit-normal t. Sampling integrates dx/dt = v from t = 0
to 1 with Euler steps; there is nothing to guide, so no classifier-free
guidance.

Backbone `SiTStates`: one token per column (721), its features the 40 frames ×
8 types of that column, learned position, adaLN-Zero by t — 13B's
`SiTBlock` with the state in place of the video. About 1.4 M parameters at
width 128 and depth 4, the size 13B's benchmark priced.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from flydream.generate.gen13b import EMA, SiTBlock, sample_t, t_embedding


class SiTStates(nn.Module):
    """Token = column; features = the column's T frames of K types."""

    def __init__(self, frames: int = 40, k: int = 8, width: int = 128, depth: int = 4, heads: int = 4, n: int = 721,
                 n_classes: int = 0):
        super().__init__()
        self.frames, self.k, self.n, self.n_classes = frames, k, n, n_classes
        self.t_dim = 128
        self.t_mlp = nn.Sequential(nn.Linear(self.t_dim, self.t_dim), nn.SiLU(), nn.Linear(self.t_dim, self.t_dim))
        # DiT's own form of conditioning: the label embedding is added to the
        # timestep embedding and modulates every block through adaLN-Zero. No
        # dropout and no null class — the precedent this follows (Dhariwal &
        # Nichol, Table 4: FID 26.21 -> 10.94) is conditioning alone, without
        # classifier-free guidance on either side.
        self.y_emb = nn.Embedding(n_classes, self.t_dim) if n_classes else None
        self.inp = nn.Linear(frames * k, width)
        self.pos = nn.Parameter(torch.randn(1, n, width) * 0.02)
        self.blocks = nn.ModuleList([SiTBlock(width, heads, self.t_dim) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.out_ada = nn.Linear(self.t_dim, 2 * width)
        self.out = nn.Linear(width, frames * k)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """(B, T, K, n) states and (B,) times -> the velocity, same shape."""
        B, T, K, n = x.shape
        tok = x.permute(0, 3, 1, 2).reshape(B, n, T * K)
        h = self.inp(tok) + self.pos
        te = self.t_mlp(t_embedding(t, self.t_dim))
        if self.y_emb is not None:
            if y is None:
                raise ValueError("this prior is class-conditional; pass y")
            te = te + self.y_emb(y)
        for blk in self.blocks:
            h = blk(h, te)
        s, b = self.out_ada(F.silu(te))[:, None].chunk(2, -1)
        y = self.out(self.out_norm(h) * (1 + s) + b)                              # (B, n, T·K)
        return y.reshape(B, n, T, K).permute(0, 2, 3, 1)


def build(frames: int = 40, k: int = 8, **kw) -> nn.Module:
    return SiTStates(frames=frames, k=k, width=kw.get("width", 128), depth=kw.get("depth", 4),
                     heads=kw.get("heads", 4), n_classes=kw.get("n_classes", 0))


def loss_fn(model: nn.Module, x1: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
    eps = torch.randn_like(x1)
    t = sample_t(x1.shape[0], x1.device)
    xt = (1 - t)[:, None, None, None] * eps + t[:, None, None, None] * x1
    v = model(xt, t) if y is None else model(xt, t, y)
    return F.mse_loss(v, x1 - eps)


def conditioned(model, y: torch.Tensor | None):
    """`integrate` and everything built on it call `f(x, t)`; a conditional
    prior needs its label carried along, so bind it once here rather than
    threading `y` through the sampler, the inversion and the refiner."""
    return model if y is None else (lambda x, t: model(x, t, y))


@torch.no_grad()
def integrate(model: nn.Module, x: torch.Tensor, *, steps: int = 20, t0: float = 0.0, t1: float = 1.0) -> torch.Tensor:
    """Euler integration of dx/dt = v(x, t) from `t0` to `t1` — the sampler's
    own map, written once so that sampling, refining and the inversion below
    all use the same discretisation."""
    n = max(1, int(round(steps * (t1 - t0))))
    for i in range(n):
        t = torch.full((len(x),), t0 + (t1 - t0) * i / n, device=x.device)
        x = x + model(x, t) * (t1 - t0) / n
    return x


@torch.no_grad()
def invert(model: nn.Module, x: torch.Tensor, *, steps: int = 20, fixed_point: int = 3) -> torch.Tensor:
    """The inverse of `integrate` over the whole interval: the noise a point came from.

    The forward step is x_{i+1} = x_i + v(x_i, t_i)/steps. Given x_{i+1}, the
    point before it solves x_i = x_{i+1} − v(x_i, t_i)/steps; the step is small,
    so a few fixed-point iterations starting at x_{i+1} reach it.
    `fixed_point=1` is the plain explicit backward step, and the difference
    between the two shows up in the reconstruction, which is measured."""
    for i in reversed(range(steps)):
        t = torch.full((len(x),), i / steps, device=x.device)
        y = x
        for _ in range(max(1, fixed_point)):
            y = x - model(y, t) / steps
        x = y
    return x


@torch.no_grad()
def sample(model: nn.Module, n: int, *, frames: int = 40, k: int = 8, columns: int = 721, steps: int = 20,
           device=None, generator: torch.Generator | None = None, y: torch.Tensor | None = None) -> torch.Tensor:
    """(n, T, K, 721) states drawn from the prior, in 13B's conditioning units."""
    dev = device or next(model.parameters()).device
    x = torch.randn(n, frames, k, columns, device=dev, generator=generator)
    return integrate(conditioned(model, y), x, steps=steps)


@torch.no_grad()
def validate(model: nn.Module, ema: EMA, states: torch.Tensor, *, batch: int, use_amp: bool,
             labels: torch.Tensor | None = None) -> float:
    """Mean interpolant loss with the EMA weights on a fixed t grid — the same every call."""
    backup = [p.detach().clone() for p in model.parameters()]
    ema.copy_to(model); model.eval()
    g = torch.Generator(device=states.device).manual_seed(0)
    tot, n = 0.0, 0
    for i in range(0, len(states), batch):
        x1 = states[i:i + batch].float()
        eps = torch.randn(x1.shape, device=x1.device, generator=g)
        t = torch.linspace(0.05, 0.95, len(x1), device=x1.device)
        xt = (1 - t)[:, None, None, None] * eps + t[:, None, None, None] * x1
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            v = model(xt, t) if labels is None else model(xt, t, labels[i:i + batch])
        tot += float(F.mse_loss(v.float(), x1 - eps)) * len(x1); n += len(x1)
    for p, b in zip(model.parameters(), backup):
        p.copy_(b)
    model.train()
    return tot / n


def train(model: nn.Module, states: torch.Tensor, *, steps: int, batch: int, lr: float = 3e-4, warmup: int = 100,
          seed: int = 0, amp: bool = True, compile_mode: str = "", log_every: int = 100, log=print,
          val: torch.Tensor | None = None, val_every: int = 500, labels: torch.Tensor | None = None,
          val_labels: torch.Tensor | None = None) -> dict:
    """13B's optimised loop without the condition: data already on the device
    (float16), AMP fp16 with a GradScaler, fused AdamW, warmup + cosine, EMA,
    gradient clipping. `states` (N, T, K, n).

    `compile_mode` compiles the forward pass (18.4a: fusion is the only lever
    on this shape — 1.42 x with `reduce-overhead`, of which at most 4.5 % can
    come from its CUDA graphs, since the affine fit puts only 2.44 ms of the
    54.72 ms step in fixed cost). `validate` swaps the EMA weights into the
    same module between steps, which CUDA graphs do not tolerate, so the
    default here is plain fusion; `reduce-overhead` is for a loop without
    validation."""
    dev = states.device
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model.to(dev).train()
    fused = dev.type == "cuda"
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.01, fused=fused)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, steps)))))
    use_amp = amp and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    fwd = torch.compile(model, **({"mode": compile_mode} if compile_mode != "default" else {})) if compile_mode \
        else model                                                   # the optimiser and the EMA keep the real module
    ema = EMA(model)
    hist, t0 = [], time.time()
    run_loss, run_n = 0.0, 0
    for step in range(steps):
        idx = torch.as_tensor(rng.integers(0, len(states), batch), device=dev)
        x1 = states[idx].float()
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            loss = loss_fn(fwd, x1, None if labels is None else labels[idx])
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        ema.update(model)
        run_loss += loss.item(); run_n += 1
        if (step + 1) % log_every == 0 or step == steps - 1:
            rec = {"step": step + 1, "loss": run_loss / run_n, "lr": sched.get_last_lr()[0], "seconds": round(time.time() - t0, 1)}
            if val is not None and ((step + 1) % val_every == 0 or step == steps - 1):
                rec["val_loss"] = validate(model, ema, val, batch=batch, use_amp=use_amp, labels=val_labels)
            hist.append(rec); run_loss, run_n = 0.0, 0
            log(f"  step {step + 1:5d}  loss {rec['loss']:.4f}" + (f"  val {rec['val_loss']:.4f}" if "val_loss" in rec else "")
                + f"  lr {rec['lr']:.2e}  {rec['seconds']:.0f}s")
    return {"history": hist, "ema": ema, "seconds": round(time.time() - t0, 1)}


def dct_matrix(n: int, k: int | None = None) -> np.ndarray:
    """(k, n) rows of the orthonormal DCT-II — `scipy.fft.dct(norm="ortho")`
    as a matrix, so the transform is one einsum in numpy or torch and the
    inverse is its transpose (17.1b: the state's time axis is compressed)."""
    k = k or n
    j = np.arange(n)[None, :]; i = np.arange(k)[:, None]
    d = np.cos(np.pi * (j + 0.5) * i / n) * np.sqrt(2.0 / n)
    d[0] *= 1 / np.sqrt(2)
    return d.astype(np.float32)


def to_dct(x, d):
    """(..., T, K, n) -> (..., k, K, n): the first `k` temporal coefficients."""
    if isinstance(x, torch.Tensor):
        return torch.einsum("kt,btcn->bkcn", torch.as_tensor(d, device=x.device, dtype=x.dtype), x)
    return np.einsum("kt,btcn->bkcn", d, x)


def from_dct(c, d):
    """The inverse of `to_dct`: the band-limited state in frames."""
    if isinstance(c, torch.Tensor):
        return torch.einsum("kt,bkcn->btcn", torch.as_tensor(d, device=c.device, dtype=c.dtype), c)
    return np.einsum("kt,bkcn->btcn", d, c)


def _dct_parts(meta: dict, device):
    """(k, matrix, coefficient mean, coefficient sd) or (0, …) for a frames prior."""
    k = int(meta.get("dct_k", 0))
    if not k:
        return 0, None, None, None
    cm = torch.as_tensor(np.array(meta["coef_mean"], np.float32), device=device)[None, :, :, None]
    cs = torch.as_tensor(np.array(meta["coef_std"], np.float32), device=device)[None, :, :, None]
    return k, dct_matrix(int(meta["time_frames"]), k), cm, cs


def to_model_space(meta: dict, states, device) -> torch.Tensor:
    """(N, 40, 8, 721) states in 13B's units -> what the prior actually models
    (the same array for a frames prior; z-scored DCT coefficients for 17.1b)."""
    x = torch.as_tensor(np.asarray(states, np.float32), device=device)
    k, dm, cm, cs = _dct_parts(meta, device)
    return (to_dct(x, dm) - cm) / cs if k else x


def from_model_space(meta: dict, x: torch.Tensor) -> np.ndarray:
    """The inverse of `to_model_space`, back to 13B's conditioning units."""
    k, dm, cm, cs = _dct_parts(meta, x.device)
    if k:
        x = from_dct(x * cs + cm, dm)
    return x.cpu().numpy().astype(np.float32)


@torch.no_grad()
def sample_states(model: nn.Module, meta: dict, n: int, *, steps: int = 20, device=None,
                  generator: torch.Generator | None = None, y: torch.Tensor | None = None) -> np.ndarray:
    """(n, 40, 8, 721) states in 13B's conditioning units, from either prior:
    the plain one samples frames directly, the DCT one samples coefficients,
    undoes their per-coefficient scaling and transforms back to frames."""
    dev = device or next(model.parameters()).device
    x = sample(model, n, frames=meta["frames"], k=meta.get("k", 8), steps=steps, device=dev, generator=generator, y=y)
    return from_model_space(meta, x)


@torch.no_grad()
def to_noise(model: nn.Module, meta: dict, states, *, steps: int = 20, fixed_point: int = 3, device=None) -> np.ndarray:
    """A state -> the noise the prior would have drawn it from (ODE inversion).

    The prior is a deterministic map from N(0, I) to states; run backwards it
    gives every real state its own noise, which is what makes the noise space
    navigable — two states can be mixed there, and how far a state sits from
    the typical radius √D says whether the prior covers it at all."""
    dev = device or next(model.parameters()).device
    x = to_model_space(meta, states, dev)
    return invert(model, x, steps=steps, fixed_point=fixed_point).cpu().numpy().astype(np.float32)


@torch.no_grad()
def from_noise(model: nn.Module, meta: dict, noise, *, steps: int = 20, device=None) -> np.ndarray:
    """The forward direction of `to_noise`: given noise, the state it produces."""
    dev = device or next(model.parameters()).device
    x = torch.as_tensor(np.asarray(noise, np.float32), device=dev)
    return from_model_space(meta, integrate(model, x, steps=steps))


def slerp(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    """Spherical interpolation per sample, flattened: the path between two
    noises that keeps the radius Gaussian noise concentrates on (a straight
    line would shrink it by up to 1/√2 and leave the prior's typical set)."""
    x, y = np.asarray(a, np.float32), np.asarray(b, np.float32)
    fx, fy = x.reshape(len(x), -1), y.reshape(len(y), -1)
    nx = np.linalg.norm(fx, axis=1, keepdims=True); ny = np.linalg.norm(fy, axis=1, keepdims=True)
    om = np.arccos(np.clip((fx / nx * fy / ny).sum(1, keepdims=True), -1, 1))
    s = np.where(om < 1e-6, 1.0, np.sin(om) + 1e-12)
    w1 = np.where(om < 1e-6, 1 - alpha, np.sin((1 - alpha) * om) / s)
    w2 = np.where(om < 1e-6, alpha, np.sin(alpha * om) / s)
    return (w1 * fx + w2 * fy).reshape(x.shape).astype(np.float32)


@torch.no_grad()
def refine(model: nn.Module, meta: dict, states: np.ndarray, *, t0: float = 0.6, steps: int = 20, device=None,
           generator: torch.Generator | None = None) -> np.ndarray:
    """A real state, taken to noise level `t0` and finished by the prior.

    The state-space analogue of image-to-image: x_t = (1−t₀)·ε + t₀·x₁ in the
    model's own space, then the flow is integrated from t₀ to 1. t₀ = 1 returns
    the state itself, t₀ = 0 is an unconditional sample; in between the prior
    keeps the coarse structure and rewrites the rest. This is the candidate
    mechanism for bringing an off-manifold state onto the learned manifold
    (ROADMAP 17.3), and it is measured, not assumed."""
    dev = device or next(model.parameters()).device
    x1 = to_model_space(meta, states, dev)
    eps = torch.randn(x1.shape, device=dev, generator=generator)
    return from_model_space(meta, integrate(model, (1 - t0) * eps + t0 * x1, steps=steps, t0=t0, t1=1.0))


def from_maps(maps: np.ndarray, layout, n_cells: int) -> np.ndarray:
    """(N, T, K, 721) maps -> (N, T, cells) states, the inverse of
    `learned.to_maps` (a bijection for the columnar T4/T5 types: one cell per
    column). Cells of other types stay zero, which is why only the deep types
    are ever scored from it."""
    pos, ch, col = layout
    out = np.zeros((maps.shape[0], maps.shape[1], n_cells), np.float32)
    out[:, :, pos] = np.asarray(maps, np.float32)[:, :, ch, col]
    return out


def unscale(maps: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """z-scored maps (the units 13B is conditioned in) -> raw activity."""
    return np.asarray(maps, np.float32) * np.asarray(std, np.float32)[None, None, :, None] \
        + np.asarray(mean, np.float32)[None, None, :, None]


def save(path, model: nn.Module, ema: EMA, meta: dict) -> None:
    torch.save({"state_dict": model.state_dict(), "ema": [p.detach().cpu() for p in ema.shadow], "meta": meta}, path)


def load(path, device) -> tuple[nn.Module, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = ck["meta"]
    model = build(frames=m["frames"], k=m.get("k", 8), width=m["width"], depth=m["depth"], heads=m.get("heads", 4),
                  n_classes=int(m.get("n_classes", 0) or 0))
    model.load_state_dict(ck["state_dict"])
    for p, s in zip(model.parameters(), ck["ema"]):
        p.data.copy_(s)                                                          # sample with the EMA weights
    return model.to(device).eval(), m
