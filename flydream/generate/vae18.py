"""18.24: энкодер и декодер состояния с латентом, который можно разыгрывать.

Решение человека, 2026-09-20: «VAE на 2 048». Обоснование и числа —
`reports/2026-09-20_the_seed_problem.md` §§ 7–8, запись в `DECISIONS.md`.

Зачем вообще. Во flow matching **нет ни одного члена**, который смотрит, куда
обращаются настоящие данные. Поэтому прообразы настоящих сцен и оказались на
радиусе 252 при оболочке 303,8 — на 73 стандартных отклонения внутрь, туда,
куда розыгрыш не попадает никогда. Член KL — ровно такой член: он заставляет
латенты настоящих данных лечь туда, откуда сэмплер и разыгрывает.

Форма латента. Токен = колонка (721 штука), поэтому латент тоже по колонкам:
`z_col` измерений на колонку, всего 721 · z_col. При z_col = 3 это **2 163** —
ближайший размер к 2 048, сохраняющий решётку. Глобальный вектор ломает
адресацию (декодеру пришлось бы разворачивать 721 разную колонку из одного
кода), поэтому он не берётся.

Критерий приёмки задан человеком и метрики не отменяет, а заменяет
(`reports/2026-09-20_the_seed_problem.md` § 5): отложенный клип → мозг →
состояние → энкодер → z → декодер → 13B → видео, сравнение **с сырым видео
корпуса**; и одновременно z от отложенных состояний обязан быть стандартным
нормальным — радиус √D ± 1,4 при ст. откл. по осям 0,99–1,01. По отдельности
каждое тривиально; смысл только в том, чтобы сошлось вместе.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from flydream.generate.gen13b import EMA


class Block(nn.Module):
    """Обычный блок трансформера — без adaLN: здесь нечем модулировать, у
    энкодера и декодера нет ни времени, ни условия."""

    def __init__(self, width: int, heads: int):
        super().__init__()
        self.n1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.n2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = self.n1(h)
        h = h + self.attn(x, x, x, need_weights=False)[0]
        return h + self.mlp(self.n2(h))


class StateVAE(nn.Module):
    """(B, T, K, n) состояние <-> (B, n, z_col) латент.

    T и K — оси представления (для DCT-16 это 16 коэффициентов и 8 типов),
    n — колонки. Признаки колонки T·K свёрнуты в один токен, как у приора.
    """

    def __init__(self, frames: int = 16, k: int = 8, n: int = 721, z_col: int = 3,
                 width: int = 192, depth: int = 4, heads: int = 4):
        super().__init__()
        self.frames, self.k, self.n, self.z_col = frames, k, n, z_col
        feat = frames * k
        self.e_in = nn.Linear(feat, width)
        self.e_pos = nn.Parameter(torch.randn(1, n, width) * 0.02)
        self.e_blocks = nn.ModuleList([Block(width, heads) for _ in range(depth)])
        self.e_norm = nn.LayerNorm(width)
        self.e_out = nn.Linear(width, 2 * z_col)                     # среднее и log-дисперсия
        # Начать у самого приора: mu ≈ 0, logvar ≈ 0 — то есть латент стартует
        # ровно стандартным нормальным, и KL в первом шаге равен нулю. Иначе
        # первые сотни шагов уходят на то, чтобы стянуть случайную инициализацию.
        nn.init.zeros_(self.e_out.weight); nn.init.zeros_(self.e_out.bias)

        self.d_in = nn.Linear(z_col, width)
        self.d_pos = nn.Parameter(torch.randn(1, n, width) * 0.02)
        self.d_blocks = nn.ModuleList([Block(width, heads) for _ in range(depth)])
        self.d_norm = nn.LayerNorm(width)
        self.d_out = nn.Linear(width, feat)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, K, n = x.shape
        h = self.e_in(x.permute(0, 3, 1, 2).reshape(B, n, T * K)) + self.e_pos
        for blk in self.e_blocks:
            h = blk(h)
        mu, logvar = self.e_out(self.e_norm(h)).chunk(2, -1)          # (B, n, z_col) каждая
        return mu, logvar.clamp(-8.0, 8.0)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        B, n, _ = z.shape
        h = self.d_in(z) + self.d_pos
        for blk in self.d_blocks:
            h = blk(h)
        y = self.d_out(self.d_norm(h))                                # (B, n, T·K)
        return y.reshape(B, n, self.frames, self.k).permute(0, 2, 3, 1)

    def forward(self, x: torch.Tensor, generator: torch.Generator | None = None):
        mu, logvar = self.encode(x)
        eps = torch.randn(mu.shape, device=mu.device, dtype=mu.dtype, generator=generator)
        z = mu + torch.exp(0.5 * logvar) * eps
        return self.decode(z), mu, logvar, z


def build(frames: int = 16, k: int = 8, **kw) -> nn.Module:
    return StateVAE(frames=frames, k=k, n=kw.get("n", 721), z_col=kw.get("z_col", 3),
                    width=kw.get("width", 192), depth=kw.get("depth", 4), heads=kw.get("heads", 4))


def kl_per_dim(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(q(z|x) || N(0, I)) на каждое измерение латента, (B, n, z_col)."""
    return 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)


def loss_fn(model: nn.Module, x: torch.Tensor, *, beta: float, free_bits: float = 0.0,
            generator: torch.Generator | None = None) -> dict:
    """Реконструкция плюс KL с полом на измерение.

    `free_bits` — стандартное средство против схлопывания апостериора: пока KL
    измерения ниже пола, за него не штрафуют, поэтому модели невыгодно гасить
    измерение целиком ради дешёвого KL. Пол берётся **до** усреднения по
    измерениям, иначе он ничего не ограничивает.
    """
    y, mu, logvar, z = model(x, generator=generator)
    rec = F.mse_loss(y, x)
    kl_d = kl_per_dim(mu, logvar)
    kl_raw = kl_d.sum(dim=(1, 2)).mean()                              # нат на образец, для отчёта
    kl_used = torch.clamp(kl_d.mean(dim=0), min=free_bits).sum()      # то, что попадает в градиент
    return {"loss": rec + beta * kl_used, "rec": rec.detach(), "kl": kl_raw.detach(),
            "kl_used": kl_used.detach(), "z_sd": z.detach().float().std(),
            "mu_sd": mu.detach().float().std(), "sigma": torch.exp(0.5 * logvar).detach().float().mean()}


def beta_at(step: int, steps: int, beta: float, warmup_frac: float = 0.3) -> float:
    """Линейный отжиг KL от нуля: сначала модель учится восстанавливать, и
    только потом её прижимают к приору. В обратном порядке она схлопывает
    латент раньше, чем декодер научится им пользоваться."""
    w = max(1, int(warmup_frac * steps))
    return beta * min(1.0, (step + 1) / w)


@torch.no_grad()
def encode_stats(model: nn.Module, states: torch.Tensor, *, batch: int = 64,
                 generator: torch.Generator | None = None) -> dict:
    """Главное число приёмки: похож ли латент настоящих данных на N(0, I).

    Гауссов шум в D измерениях садится на радиус √D шириной 0,707, поэтому
    сравниваются обе величины — ст. откл. по осям и радиус со своим разбросом.
    """
    model.eval()
    zs, recs = [], []
    for i in range(0, len(states), batch):
        x = states[i:i + batch].float()
        y, mu, logvar, z = model(x, generator=generator)
        zs.append(z.float().cpu())
        recs.append(float(F.mse_loss(y, x)) * len(x))
    model.train()
    Z = torch.cat(zs).reshape(len(states), -1)
    D = Z.shape[1]
    rad = Z.norm(dim=1)
    return {"dims": int(D), "typical_radius": math.sqrt(D), "z_sd": float(Z.std()),
            "radius_mean": float(rad.mean()), "radius_sd": float(rad.std()),
            "radius_sd_gaussian": math.sqrt(0.5), "rec": sum(recs) / len(states)}


def train(model: nn.Module, states: torch.Tensor, *, steps: int, batch: int, lr: float = 1e-3,
          beta: float = 1e-4, free_bits: float = 0.0, warmup_frac: float = 0.3, warmup: int = 100,
          seed: int = 0, amp: bool = True, log_every: int = 100, log=print,
          val: torch.Tensor | None = None, val_every: int = 500) -> dict:
    """Тот же оптимизированный цикл, что у приора: данные уже на карте, AMP
    fp16 с GradScaler, fused AdamW, warmup + косинус, EMA, клиппинг."""
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
    ema = EMA(model)
    hist, t0 = [], time.time()
    for step in range(steps):
        idx = torch.as_tensor(rng.integers(0, len(states), batch), device=dev)
        x = states[idx].float()
        b = beta_at(step, steps, beta, warmup_frac)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            out = loss_fn(model, x, beta=b, free_bits=free_bits)
        opt.zero_grad(set_to_none=True)
        scaler.scale(out["loss"]).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step(); ema.update(model)
        if (step + 1) % log_every == 0 or step == steps - 1:
            rec = {"step": step + 1, "beta": b, "loss": float(out["loss"]), "rec": float(out["rec"]),
                   "kl": float(out["kl"]), "z_sd": float(out["z_sd"]), "sigma": float(out["sigma"]),
                   "lr": sched.get_last_lr()[0], "seconds": round(time.time() - t0, 1)}
            if val is not None and ((step + 1) % val_every == 0 or step == steps - 1):
                with torch.no_grad():                                 # свап весов EMA — in-place по листам
                    backup = [p.detach().clone() for p in model.parameters()]
                    ema.copy_to(model)
                    g = torch.Generator(device=dev).manual_seed(0)
                    rec |= {f"val_{k}": v for k, v in encode_stats(model, val, generator=g).items()}
                    for p, q in zip(model.parameters(), backup):
                        p.copy_(q)
            hist.append(rec)
            log(f"  step {rec['step']:6d}  rec {rec['rec']:.4f}  kl {rec['kl']:8.1f}  "
                f"z_sd {rec['z_sd']:.3f}  sigma {rec['sigma']:.3f}  beta {b:.2e}  {rec['seconds']:.0f}s"
                + ("" if "val_rec" not in rec else
                   f"  | val rec {rec['val_rec']:.4f}, z_sd {rec['val_z_sd']:.3f}, "
                   f"радиус {rec['val_radius_mean']:.1f} (√D {rec['val_typical_radius']:.1f})"))
    return {"history": hist, "ema": ema, "seconds": round(time.time() - t0, 1)}


def save(path, model: nn.Module, ema: EMA, meta: dict) -> None:
    torch.save({"state_dict": model.state_dict(), "ema": [p.detach().cpu() for p in ema.shadow],
                "meta": meta}, path)


def load(path, device) -> tuple[nn.Module, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = ck["meta"]
    model = build(frames=m["frames"], k=m.get("k", 8), n=m.get("n", 721), z_col=m["z_col"],
                  width=m["width"], depth=m["depth"], heads=m.get("heads", 4))
    model.load_state_dict(ck["state_dict"])
    for p, s in zip(model.parameters(), ck["ema"]):
        p.data.copy_(s)                                               # оцениваем на весах EMA, как у приора
    return model.to(device).eval(), m


@torch.no_grad()
def sample_latent(model: nn.Module, n: int, *, device=None,
                  generator: torch.Generator | None = None) -> torch.Tensor:
    """Разыгранный латент — тот самый «вбитый руками сид», ради которого всё.
    Если KL сделал свою работу, это неотличимо от латента настоящей сцены."""
    dev = device or next(model.parameters()).device
    return torch.randn(n, model.n, model.z_col, device=dev, generator=generator)
