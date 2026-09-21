r"""22.4: обучаемый остаток поверх фиксированной линейной ступени.

Рецепт DC-AE в нашей постановке: линейная часть заморожена (PCA по блоку
T4a+T4b, k = 1536, отложенная дисперсия 85,3 %, r 0,806), а сеть учит только
то, чего линейная часть не взяла. Ключевое требование пришло из измерения
22.3: **остаток должен быть локальным**. Канальное сжатие сохраняет резкость
и теряет содержание (0,736 при 11 536 числах), глобальная PCA — наоборот
(0,824 при 2 048), и разница между ними в том, что PCA глобальная. Значит
сеть обязана смешивать соседей, а не всё поле.

Гекс-свёртка здесь — это вес на себя плюс шесть весов по направлениям
решётки (`neighbour_index`), то есть рецептивное поле ровно в один шаг на
слой. Два слоя кодировщика дают радиус 2, чего по 21б хватает: корреляция
колонок падает вдвое к пятому шагу.

    состояние блока x (721 x 32)
      -> PCA-1536 (заморожена) -> x_lin
      -> остаток e = x - x_lin
      -> кодировщик (гекс-свёртки) -> код (721 x m)
      -> декодер -> ê
      -> итог x_lin + ê

Итоговое число координат на клип: 1536 + 721·m.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from flydream.decode.hexraster import neighbour_index


class HexConv(nn.Module):
    """Свёртка по гексагональной решётке: себя плюс шесть соседей.

    Вне решётки соседа нет, и такие места берут ноль — это край поля зрения,
    а не пропуск в данных."""

    def __init__(self, c_in: int, c_out: int, n_cols: int = 721):
        super().__init__()
        nb = torch.as_tensor(np.asarray(neighbour_index(n_cols)), dtype=torch.long)
        self.register_buffer("nb", nb.clamp_min(0))
        self.register_buffer("ok", (nb >= 0).to(torch.float32))
        self.w = nn.Parameter(torch.empty(7, c_in, c_out))
        nn.init.normal_(self.w, std=(2.0 / (7 * c_in)) ** 0.5)
        self.b = nn.Parameter(torch.zeros(c_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:                # (N, n, c_in) -> (N, n, c_out)
        out = x @ self.w[0]
        for d in range(6):
            out = out + (x[:, self.nb[:, d]] * self.ok[:, d, None]) @ self.w[d + 1]
        return out + self.b


class ResidualAE(nn.Module):
    """Кодировщик и декодер остатка; между ними код (n, m) на колонку."""

    def __init__(self, c_in: int = 32, width: int = 64, code: int = 4, n_cols: int = 721):
        super().__init__()
        self.enc = nn.ModuleList([HexConv(c_in, width, n_cols), HexConv(width, width, n_cols)])
        self.to_code = nn.Linear(width, code)
        self.from_code = nn.Linear(code, width)
        self.dec = nn.ModuleList([HexConv(width, width, n_cols), HexConv(width, c_in, n_cols)])
        self.code = code

    def encode(self, e: torch.Tensor) -> torch.Tensor:
        h = e
        for layer in self.enc:
            h = F.gelu(layer(h))
        return self.to_code(h)

    def decode(self, c: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.from_code(c))
        h = F.gelu(self.dec[0](h))
        return self.dec[1](h)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(e))


@torch.no_grad()
def residual_of(X: torch.Tensor, p: dict, k: int, block: int = 8192) -> torch.Tensor:
    """x - PCA_k(x) в том же (N, D) — то, что сети остаётся выучить."""
    from flydream.generate import pca19 as P

    pk = P.truncate(p, k)
    return X - P.decode(P.encode(X, pk, block=block), pk, block=block)


def explained_fraction(e: torch.Tensor, hat: torch.Tensor) -> float:
    """Какую долю энергии остатка сеть сняла: 1 - ||e - ê||² / ||e||²."""
    return float(1.0 - (e - hat).pow(2).sum() / e.pow(2).sum().clamp_min(1e-12))


def train(model: nn.Module, E: torch.Tensor, Eval: torch.Tensor, *, steps: int = 4000, batch: int = 64,
          lr: float = 2e-3, warmup: int = 100, n_cols: int = 721, c_in: int = 32, use_amp: bool = True,
          log=print) -> dict:
    """Остаток — плотная регрессия, поэтому ни EMA, ни расписания шума не нужно:
    Adam, косинус, и лучший снимок по валидации, как в 19.2."""
    dev = E.device
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0, fused=(dev.type == "cuda"))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / steps))))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and dev.type == "cuda")
    g = torch.Generator(device=dev).manual_seed(0)
    best, best_state, best_step, nan_steps = -1e9, None, 0, 0
    t0 = time.time()
    hist = []
    for s in range(steps):
        idx = torch.randint(0, len(E), (batch,), device=dev, generator=g)
        e = E[idx].float().reshape(batch, n_cols, c_in)
        with torch.autocast("cuda", dtype=torch.float16, enabled=scaler.is_enabled()):
            loss = F.mse_loss(model(e), e)
        if not torch.isfinite(loss):
            nan_steps += 1
            opt.zero_grad(set_to_none=True)
            continue
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update(); sched.step()
        if (s + 1) % max(1, steps // 20) == 0 or s == steps - 1:
            model.eval()
            with torch.no_grad():
                ev = Eval.float().reshape(len(Eval), n_cols, c_in)
                frac = explained_fraction(ev, model(ev))
            model.train()
            hist.append({"step": s + 1, "train_mse": float(loss), "val_explained": frac})
            if frac > best:
                best, best_step = frac, s + 1
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            log(f"  шаг {s + 1:5}/{steps}: mse {float(loss):.4f}, остаток снят на {100 * frac:.1f} % "
                f"(лучшее {100 * best:.1f} % на {best_step}), {time.time() - t0:.0f} с")
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val_explained": best, "best_step": best_step, "nan_steps": nan_steps,
            "history": hist, "seconds": round(time.time() - t0, 1),
            "parameters": int(sum(p.numel() for p in model.parameters()))}


class HexPool(nn.Module):
    """Прореживание решётки: локальное смешивание, затем подвыборка подрешётки.

    22.3 измерил, что избыточность состояния лежит в пространстве, а не в
    каналах, а 22.4 показал, что сжатие каналов при сохранённых 721 колонке не
    окупается. Здесь наоборот: мест становится втрое меньше, каналов больше.
    Подрешётка `third` — изотропная √3 × √3, 241 колонка из 721, и у каждой
    выброшенной колонки есть хотя бы один оставленный сосед (проверено), так
    что обратный ход достижим за один шаг."""

    def __init__(self, keep: np.ndarray):
        super().__init__()
        self.register_buffer("idx", torch.as_tensor(np.where(keep)[0], dtype=torch.long))
        self.n_out = int(keep.sum())

    def forward(self, x: torch.Tensor) -> torch.Tensor:                # (N, n, c) -> (N, n_out, c)
        return x[:, self.idx]


class HexUnpool(nn.Module):
    """Обратный ход: разложить код по своим местам, остальные — нули.

    Заполняют их следующие гекс-свёртки, а не интерполяция: пусть сеть сама
    решает, как разносить значение по соседям."""

    def __init__(self, keep: np.ndarray):
        super().__init__()
        self.register_buffer("idx", torch.as_tensor(np.where(keep)[0], dtype=torch.long))
        self.n_full = int(len(keep))

    def forward(self, x: torch.Tensor) -> torch.Tensor:                # (N, n_out, c) -> (N, n, c)
        out = x.new_zeros(len(x), self.n_full, x.shape[2])
        out[:, self.idx] = x
        return out


class BlockAE(nn.Module):
    """Автоэнкодер блока состояния: 721 x c_in -> 241 x code -> 721 x c_in.

    Кодировщик смешивает соседей ДО подвыборки (это и есть свёртка с шагом),
    декодер раскладывает код по решётке и двумя свёртками разносит его по
    пропускам. Рецептивное поле каждого слоя — один шаг."""

    def __init__(self, keep: np.ndarray, c_in: int = 32, width: int = 64, code: int = 8,
                 n_cols: int = 721):
        super().__init__()
        self.enc = nn.ModuleList([HexConv(c_in, width, n_cols), HexConv(width, width, n_cols)])
        self.pool = HexPool(keep)
        self.to_code = nn.Linear(width, code)
        self.from_code = nn.Linear(code, width)
        self.unpool = HexUnpool(keep)
        self.dec = nn.ModuleList([HexConv(width, width, n_cols), HexConv(width, width, n_cols),
                                  HexConv(width, c_in, n_cols)])
        self.code, self.sites = code, self.pool.n_out

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.enc:
            h = F.gelu(layer(h))
        return self.to_code(self.pool(h))

    def decode(self, c: torch.Tensor) -> torch.Tensor:
        h = self.unpool(F.gelu(self.from_code(c)))
        h = F.gelu(self.dec[0](h))
        h = F.gelu(self.dec[1](h))
        return self.dec[2](h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))
