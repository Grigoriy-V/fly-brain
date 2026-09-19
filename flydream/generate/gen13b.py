"""13B: the generative decoder `deep state + type mask + z -> video`
(`reports/2026-09-19_step13b_design.md`, revision 2).

Objective: the SiT-style linear interpolant (flow matching). With noise ε and
a video x₁, x_t = (1−t)·ε + t·x₁ and the model predicts the velocity
v = x₁ − ε; the loss is the MSE. Sampling integrates dx/dt = v from t = 0 to
1 (Euler, `steps` steps) with classifier-free guidance
v = v(∅) + s·(v(state) − v(∅)), the unconditional branch being the all-zero
mask. From x_t and v the clean estimate is x̂₁ = x_t + (1−t)·v.

Condition: the maps of the 8 T4/T5 types (B, T, 8, 721), z-scored per type,
times a type mask (B, 8); masked types are zero and the mask bits are given
to the model as well. Masks in training are structured (`sample_masks`):
full / T4 only / T5 only / one type / one direction / random subset /
unconditional, so "knob" regimes are seen regularly.

Two backbones, chosen by a benchmark (`bench`): `HexResNet` (ring-1 hex
mixing + temporal conv, FiLM by t) and `SiTColumns` (a transformer over the
721 columns, time as token features, adaLN-Zero by t).
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from flydream.generate.learned import ring_index

DEEP = ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"]
MASK_MODES = {"full": 0.40, "t4": 0.10, "t5": 0.10, "one": 0.15, "direction": 0.10, "random": 0.10, "none": 0.05}


# ----------------------------------------------------------------- masks


def sample_masks(n: int, rng: np.random.Generator, modes: dict | None = None) -> np.ndarray:
    """(n, 8) float32 type masks drawn from the structured modes."""
    modes = modes or MASK_MODES
    names, probs = list(modes), np.array(list(modes.values()), np.float64)
    probs /= probs.sum()
    out = np.zeros((n, 8), np.float32)
    for i, mode in enumerate(rng.choice(names, n, p=probs)):
        if mode == "full":
            out[i] = 1
        elif mode == "t4":
            out[i, :4] = 1
        elif mode == "t5":
            out[i, 4:] = 1
        elif mode == "one":
            out[i, rng.integers(8)] = 1
        elif mode == "direction":
            d = rng.integers(4); out[i, d] = 1; out[i, 4 + d] = 1
        elif mode == "random":
            m = rng.random(8) < 0.5
            if not m.any():
                m[rng.integers(8)] = True
            out[i] = m
    return out


def named_mask(name: str) -> np.ndarray:
    """A mask by name: 'full', 't4', 't5', 'T4a' … 'T5d', 'dir_a' … 'dir_d', 'none'."""
    m = np.zeros(8, np.float32)
    if name == "full":
        m[:] = 1
    elif name == "t4":
        m[:4] = 1
    elif name == "t5":
        m[4:] = 1
    elif name in DEEP:
        m[DEEP.index(name)] = 1
    elif name.startswith("dir_"):
        d = "abcd".index(name[-1]); m[d] = 1; m[4 + d] = 1
    elif name != "none":
        raise ValueError(name)
    return m


# ----------------------------------------------------------------- building blocks


def t_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of t in [0, 1] -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    a = t.float()[:, None] * 1000.0 * freqs[None]
    return torch.cat([a.sin(), a.cos()], 1)


class HexMix(nn.Module):
    """(B, C, T, n) -> (B, C', T, n): each column mixed with its ring-1 neighbours."""

    def __init__(self, c_in: int, c_out: int, ring: np.ndarray):
        super().__init__()
        self.register_buffer("ring", torch.as_tensor(ring))
        self.register_buffer("valid", torch.as_tensor(ring >= 0).float())
        self.m = ring.shape[1]
        self.weight = nn.Parameter(torch.randn(c_out, c_in * self.m) / math.sqrt(c_in * self.m))
        self.bias = nn.Parameter(torch.zeros(c_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, n = x.shape
        g = x[..., self.ring.clamp(min=0)] * self.valid                          # (B, C, T, n, m)
        g = g.permute(0, 2, 3, 1, 4).reshape(B * T * n, C * self.m)
        y = F.linear(g, self.weight, self.bias)
        return y.view(B, T, n, -1).permute(0, 3, 1, 2)


class HexResBlock(nn.Module):
    def __init__(self, width: int, ring: np.ndarray, t_dim: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, width)
        self.film = nn.Linear(t_dim, 2 * width)
        self.hex = HexMix(width, width, ring)
        self.temporal = nn.Conv1d(width, width, 3, padding=1)
        nn.init.zeros_(self.temporal.weight); nn.init.zeros_(self.temporal.bias)

    def forward(self, x: torch.Tensor, te: torch.Tensor) -> torch.Tensor:
        B, C, T, n = x.shape
        s, b = self.film(te).chunk(2, 1)
        h = self.norm(x) * (1 + s[:, :, None, None]) + b[:, :, None, None]
        h = F.silu(self.hex(F.silu(h)))
        h = self.temporal(h.permute(0, 3, 1, 2).reshape(B * n, C, T)).view(B, n, C, T).permute(0, 2, 3, 1)
        return x + h


class HexResNet(nn.Module):
    """Backbone A. Input channels: x_t (1) + masked state maps (8) + mask bits (8)."""

    def __init__(self, width: int = 64, depth: int = 6, k_cond: int = 8):
        super().__init__()
        ring = ring_index(1)
        self.t_dim = 128
        self.t_mlp = nn.Sequential(nn.Linear(self.t_dim, self.t_dim), nn.SiLU(), nn.Linear(self.t_dim, self.t_dim))
        self.inp = HexMix(1 + 2 * k_cond, width, ring)
        self.blocks = nn.ModuleList([HexResBlock(width, ring, self.t_dim) for _ in range(depth)])
        self.out_norm = nn.GroupNorm(8, width)
        self.out = HexMix(width, 1, ring)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x (B, T, n); cond (B, T, K, n); mask (B, K)
        B, T, n = x.shape
        c = cond * mask[:, None, :, None]
        mb = mask[:, None, :, None].expand(B, T, mask.shape[1], n)
        h = torch.cat([x[:, :, None], c, mb], 2).permute(0, 2, 1, 3)            # (B, 1+2K, T, n)
        te = self.t_mlp(t_embedding(t, self.t_dim))
        h = self.inp(h)
        for blk in self.blocks:
            h = blk(h, te)
        return self.out(F.silu(self.out_norm(h)))[:, 0]                           # (B, T, n)


class SiTBlock(nn.Module):
    def __init__(self, width: int, heads: int, t_dim: int):
        super().__init__()
        self.n1 = nn.LayerNorm(width, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.n2 = nn.LayerNorm(width, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(approximate="tanh"), nn.Linear(4 * width, width))
        self.ada = nn.Linear(t_dim, 6 * width)
        nn.init.zeros_(self.ada.weight); nn.init.zeros_(self.ada.bias)          # adaLN-Zero

    def forward(self, x: torch.Tensor, te: torch.Tensor) -> torch.Tensor:
        s1, b1, g1, s2, b2, g2 = self.ada(F.silu(te))[:, None].chunk(6, -1)
        h = self.n1(x) * (1 + s1) + b1
        x = x + g1 * self.attn(h, h, h, need_weights=False)[0]
        h = self.n2(x) * (1 + s2) + b2
        return x + g2 * self.mlp(h)


class SiTColumns(nn.Module):
    """Backbone B. Token = column; features = x_t's T frames, the masked state's
    K·T values, the K mask bits; learned position per column."""

    def __init__(self, frames: int = 40, width: int = 128, depth: int = 4, heads: int = 4, k_cond: int = 8, n: int = 721):
        super().__init__()
        self.t_dim = 128
        self.t_mlp = nn.Sequential(nn.Linear(self.t_dim, self.t_dim), nn.SiLU(), nn.Linear(self.t_dim, self.t_dim))
        self.inp = nn.Linear(frames * (1 + k_cond) + k_cond, width)
        self.pos = nn.Parameter(torch.randn(1, n, width) * 0.02)
        self.blocks = nn.ModuleList([SiTBlock(width, heads, self.t_dim) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.out_ada = nn.Linear(self.t_dim, 2 * width)
        self.out = nn.Linear(width, frames)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T, n = x.shape
        c = (cond * mask[:, None, :, None]).permute(0, 3, 2, 1).reshape(B, n, -1)   # (B, n, K·T)
        tok = torch.cat([x.permute(0, 2, 1), c, mask[:, None].expand(B, n, mask.shape[1])], 2)
        h = self.inp(tok) + self.pos
        te = self.t_mlp(t_embedding(t, self.t_dim))
        for blk in self.blocks:
            h = blk(h, te)
        s, b = self.out_ada(F.silu(te))[:, None].chunk(2, -1)
        return self.out(self.out_norm(h) * (1 + s) + b).permute(0, 2, 1)         # (B, T, n)


def build(kind: str, frames: int = 40, **kw) -> nn.Module:
    if kind == "hexresnet":
        return HexResNet(width=kw.get("width", 64), depth=kw.get("depth", 6))
    if kind == "sit":
        return SiTColumns(frames=frames, width=kw.get("width", 128), depth=kw.get("depth", 4), heads=kw.get("heads", 4))
    raise ValueError(kind)


# ----------------------------------------------------------------- the interpolant


def sample_t(n: int, device) -> torch.Tensor:
    """Logit-normal t (SD3 / SiT practice): more weight where the velocity is hardest."""
    return torch.sigmoid(torch.randn(n, device=device))


def loss_fn(model: nn.Module, x1: torch.Tensor, cond: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    eps = torch.randn_like(x1)
    t = sample_t(x1.shape[0], x1.device)
    xt = (1 - t)[:, None, None] * eps + t[:, None, None] * x1
    v = model(xt, t, cond, mask)
    return F.mse_loss(v, x1 - eps)


@torch.no_grad()
def sample(model: nn.Module, cond: torch.Tensor, mask: torch.Tensor, *, steps: int = 20, guidance: float = 1.0,
           generator: torch.Generator | None = None, clamp: bool = True) -> torch.Tensor:
    """(B, T, n) videos for (B, T, K, n) maps and (B, K) masks; guidance 1 = plain conditional."""
    B, T, _, n = cond.shape
    x = torch.randn(B, T, n, device=cond.device, generator=generator)
    zero = torch.zeros_like(mask)
    for i in range(steps):
        t = torch.full((B,), i / steps, device=cond.device)
        v = model(x, t, cond, mask)
        if guidance != 1.0:
            v = model(x, t, cond, zero) + guidance * (v - model(x, t, cond, zero))
        x = x + v / steps
    return x.clamp(0, 1) if clamp else x


# ----------------------------------------------------------------- training


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = [p.detach().clone() for p in model.parameters()]

    @torch.no_grad()
    def update(self, model: nn.Module):
        for s, p in zip(self.shadow, model.parameters()):
            s.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        for s, p in zip(self.shadow, model.parameters()):
            p.copy_(s)


def train(model: nn.Module, videos: torch.Tensor, maps: torch.Tensor, *, steps: int, batch: int, lr: float,
          warmup: int = 100, seed: int = 0, amp: bool = True, compile_model: bool = False, log_every: int = 50,
          log=print, val: tuple[torch.Tensor, torch.Tensor] | None = None, val_every: int = 500,
          mask_modes: dict | None = None) -> dict:
    """The optimised loop: data already on the device (float16), AMP fp16 with
    a GradScaler, fused AdamW, warmup + cosine, EMA, structured masks.
    `videos` (N, T, n), `maps` (N, T, K, n). Returns the history and the EMA."""
    dev = videos.device
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model.to(dev).train()
    fwd = torch.compile(model) if compile_model else model
    fused = dev.type == "cuda"
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.01, fused=fused)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, steps)))))
    use_amp = amp and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema = EMA(model)
    n = len(videos)
    hist, t0 = [], time.time()
    run_loss, run_n = 0.0, 0
    for step in range(steps):
        idx = torch.as_tensor(rng.integers(0, n, batch), device=dev)
        x1 = videos[idx].float(); cond = maps[idx].float()
        mask = torch.as_tensor(sample_masks(batch, rng, mask_modes), device=dev)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            loss = loss_fn(fwd, x1, cond, mask)
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
                rec["val_loss"] = validate(model, ema, *val, batch=batch, use_amp=use_amp)
            hist.append(rec); run_loss, run_n = 0.0, 0
            log(f"  step {step + 1:5d}  loss {rec['loss']:.4f}" + (f"  val {rec['val_loss']:.4f}" if "val_loss" in rec else "")
                + f"  lr {rec['lr']:.2e}  {rec['seconds']:.0f}s")
    return {"history": hist, "ema": ema, "seconds": round(time.time() - t0, 1)}


@torch.no_grad()
def validate(model: nn.Module, ema: EMA, videos: torch.Tensor, maps: torch.Tensor, *, batch: int, use_amp: bool) -> float:
    """Mean interpolant loss with the EMA weights, full mask, fixed t grid — the same every call."""
    backup = [p.detach().clone() for p in model.parameters()]
    ema.copy_to(model); model.eval()
    g = torch.Generator(device=videos.device).manual_seed(0)
    tot, n = 0.0, 0
    for i in range(0, len(videos), batch):
        x1 = videos[i:i + batch].float(); cond = maps[i:i + batch].float()
        eps = torch.randn(x1.shape, device=x1.device, generator=g)
        t = torch.linspace(0.05, 0.95, len(x1), device=x1.device)
        xt = (1 - t)[:, None, None] * eps + t[:, None, None] * x1
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            v = model(xt, t, cond, torch.ones(len(x1), cond.shape[2], device=x1.device))
        tot += float(F.mse_loss(v.float(), x1 - eps)) * len(x1); n += len(x1)
    for p, b in zip(model.parameters(), backup):
        p.copy_(b)
    model.train()
    return tot / n


def bench(kind: str, videos: torch.Tensor, maps: torch.Tensor, *, steps: int = 200, batch: int = 32, lr: float = 3e-4,
          compile_model: bool = False, log=print, **kw) -> dict:
    """One backbone for `steps` steps: seconds per step (after warm-up), peak
    memory, parameters, the loss over the run."""
    model = build(kind, frames=videos.shape[1], **kw)
    n_par = sum(p.numel() for p in model.parameters())
    if videos.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    r = train(model, videos, maps, steps=steps, batch=batch, lr=lr, warmup=20, compile_model=compile_model, log_every=steps // 4, log=log)
    if videos.device.type == "cuda":
        torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1e9 if videos.device.type == "cuda" else 0.0
    h = r["history"]
    per_step = (h[-1]["seconds"] - h[0]["seconds"]) / max(1, h[-1]["step"] - h[0]["step"])
    return {"kind": kind, "parameters": int(n_par), "seconds_per_step": round(per_step, 4), "peak_gb": round(peak, 2),
            "loss_first": h[0]["loss"], "loss_last": h[-1]["loss"], "batch": batch, "steps": steps, "compiled": compile_model}


# ----------------------------------------------------------------- checkpoints


def save(path, model: nn.Module, ema: EMA, meta: dict):
    torch.save({"state_dict": model.state_dict(), "ema": [s.cpu() for s in ema.shadow], "meta": meta}, path)


def load(path, device) -> tuple[nn.Module, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = ck["meta"]
    model = build(m["kind"], frames=m["frames"], width=m["width"], depth=m["depth"], heads=m.get("heads", 4))
    model.load_state_dict(ck["state_dict"])
    for p, s in zip(model.parameters(), ck["ema"]):
        p.data.copy_(s)                                                          # sample with the EMA weights
    return model.to(device).eval(), m
