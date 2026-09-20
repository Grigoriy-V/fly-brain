"""19.0: линейная первая стадия — PCA на 2 048 компонент, отбелённая.

Решение человека, 2026-09-21 (`DECISIONS.md`): латент не обязан быть N(0, I),
разыгрываемость — работа потока над латентом, а первой стадии достаточно быть
линейной. Обоснование и источники: `reports/2026-09-21_research_how_vaes_are_trained.md`.

Что здесь считается. Состояния лежат в том же пространстве, в котором работает
поток (z-scored DCT-16, D = 92 288). PCA строится **только на обучающей части**;
доля дисперсии и геометрия меряются на отложенных — у выборки из N образцов
ковариация имеет ранг ≤ N−1, поэтому «объяснено 100 % своими же компонентами»
верно всегда и не значит ничего (то же правило, что в 18.22).

Так как N = 13 555 < D = 92 288, разложение берётся из матрицы Грама N × N, а не
из ковариации D × D: это то же самое разложение и в тысячи раз дешевле.
Если G v = μ v при |v| = 1, то собственный вектор ковариации — это
w = Xᵀv / √μ, а дисперсия вдоль него λ = μ / (N − 1).

Отбеливание (деление на √λ) — это тот самый шаг «записать mean/std латента и
поделить», который несёт каждый чекпойнт латентной диффузии (`scaling_factor`),
только в замкнутой форме и по каждой координате, а не одним скаляром.

**Названный заранее риск.** λ считается на обучающих, и для хвостовых компонент
это завышенная оценка (отбор по той же выборке). Значит на отложенных
отбелённые хвостовые координаты будут **уже** единичной дисперсии, латент сядет
внутрь оболочки √k — ровно та болезнь, из-за которой поток над состояниями не
дал сцену из розыгрыша. Поэтому `geometry` меряет профиль ст. отклонения по
блокам координат: это число решает, резать ли хвост (k = 1 024) до того, как за
поток заплачено.
"""
from __future__ import annotations

import time
from math import lgamma

import numpy as np
import torch

EPS = 1e-12


@torch.no_grad()
def fit(X: torch.Tensor, k: int, *, block: int = 8192, log=print) -> dict:
    """(N, D) в модельном пространстве -> {mean, basis (D, k), lam (k,), ...}.

    Всё считается блоками по признакам: X держится в fp16 (2,5 ГБ при
    13 555 × 92 288), а в fp32 живёт только текущий блок.
    """
    t0 = time.time()
    N, D = X.shape
    dev = X.device
    if k > min(N - 1, D):
        raise ValueError(f"компонент {k} больше ранга данных {min(N - 1, D)}")
    mean = torch.empty(D, device=dev, dtype=torch.float32)
    for i in range(0, D, block):
        mean[i:i + block] = X[:, i:i + block].float().mean(0)

    G = torch.zeros(N, N, device=dev, dtype=torch.float32)
    total = 0.0
    for i in range(0, D, block):
        B = X[:, i:i + block].float() - mean[i:i + block]
        G += B @ B.T                                                  # fp32: диагональ здесь ~D, в fp16 она бы переполнилась
        total += float(B.pow(2).sum())
    log(f"  матрица Грама {N}x{N} за {time.time() - t0:.0f} с")

    evals, evecs = torch.linalg.eigh(G)                               # по возрастанию
    del G                                                             # 735 МБ: eigh уже прочитал матрицу
    V = evecs[:, -k:].flip(1).contiguous()                            # срез сначала, поворот потом: копия на k, не на N
    del evecs
    evals = evals.flip(0).clamp_min(0.0)
    mu = evals[:k]
    if X.is_cuda:
        torch.cuda.empty_cache()
    log(f"  eigh за {time.time() - t0:.0f} с, mu[0] {float(mu[0]):.4g}, mu[{k - 1}] {float(mu[-1]):.4g}")

    scale = mu.clamp_min(EPS).sqrt()
    W = torch.empty(D, k, device=dev, dtype=torch.float32)
    for i in range(0, D, block):
        B = X[:, i:i + block].float() - mean[i:i + block]
        W[i:i + block] = (B.T @ V) / scale                            # w = X^T v / sqrt(mu)

    lam = mu / (N - 1)                                                # дисперсия вдоль компоненты на обучающих
    norm_min = float(W.norm(dim=0).min())                             # 1,000 у здорового базиса
    return {"mean": mean, "basis": W, "lam": lam, "eigvals_all": evals,
            "n_fit": N, "dims": D, "k": k,
            "train_total_var": total / (N - 1),
            "degenerate": int((mu <= EPS).sum()),
            "basis_norm_min": norm_min,                               # < 0,99 значит хвост базиса — шум fp16,
            "mu_last": float(mu[-1]),                                 # и тогда отбеливание его усиливает
            "seconds": round(time.time() - t0, 1)}


@torch.no_grad()
def encode(X: torch.Tensor, p: dict, *, whiten: bool = True, block: int = 8192) -> torch.Tensor:
    """(M, D) -> (M, k). `whiten=True` делит на √λ обучающих — единичная
    дисперсия по построению **на обучающих**, на отложенных это измеряется."""
    Z = torch.zeros(len(X), p["basis"].shape[1], device=p["basis"].device, dtype=torch.float32)
    for i in range(0, p["dims"], block):
        Z += (X[:, i:i + block].float() - p["mean"][i:i + block]) @ p["basis"][i:i + block]
    return Z / p["lam"].clamp_min(EPS).sqrt() if whiten else Z


@torch.no_grad()
def decode(Z: torch.Tensor, p: dict, *, whiten: bool = True, block: int = 8192,
           out_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """(M, k) -> (M, D). Обратное к `encode` с точностью до отброшенных компонент."""
    C = Z.float() * p["lam"].clamp_min(EPS).sqrt() if whiten else Z.float()
    out = torch.empty(len(Z), p["dims"], device=Z.device, dtype=out_dtype)
    for i in range(0, p["dims"], block):
        out[:, i:i + block] = (C @ p["basis"][i:i + block].T + p["mean"][i:i + block]).to(out_dtype)
    return out


@torch.no_grad()
def explained(X: torch.Tensor, p: dict, ks, *, block: int = 8192, z: torch.Tensor | None = None) -> dict:
    """Доля дисперсии, объяснённая первыми k компонентами, на данных X.

    Базис ортонормирован, поэтому энергия проекции считается без обратного
    преобразования: ||P_k x||² = ||z_k||², а остаток — разность с ||x − mean||².
    `z` — уже посчитанные НЕотбелённые координаты, чтобы не повторять матричное
    умножение на 2,6e12 операций ради того же числа.
    """
    Z = encode(X, p, whiten=False, block=block) if z is None else z
    tot = 0.0
    for i in range(0, p["dims"], block):
        tot += float((X[:, i:i + block].float() - p["mean"][i:i + block]).pow(2).sum())
    out = {}
    for k in ks:
        e = float(Z[:, :k].pow(2).sum())
        out[int(k)] = {"explained": e / max(tot, EPS),
                       "mse": (tot - e) / (len(X) * p["dims"])}
    return out


@torch.no_grad()
def geometry(Z: torch.Tensor, blocks: int = 4) -> dict:
    """Насколько отбелённый латент похож на стандартный гауссов — до всякого
    потока. Радиус √k при разбросе 0,707, ст. откл. по осям 1,0, эксцесс 3,0.

    Профиль по блокам координат отвечает на названный риск: если хвост
    систематически уже единицы, латент сидит внутри оболочки.

    **Малая выборка смещает ровно те числа, ради которых всё это считается.**
    Выборочное ст. отклонение по оси занижено множителем c₄(M) (при M = 6 это
    0,952), а выборочный эксцесс m₄/m₂² имеет матожидание 3(M−1)/(M+1) = 2,14.
    На шести клипах «0,95 и 2,1» — это идеальный гауссов латент, а не севший
    внутрь оболочки. Поэтому возвращаются и поправка, и ожидаемые при этом M
    значения, а `sd` (по всем M·k числам сразу) смещения не имеет."""
    Z = Z.float()
    M, k = Z.shape
    c4 = (np.exp(lgamma(M / 2) - lgamma((M - 1) / 2)) * np.sqrt(2.0 / (M - 1))) if M > 1 else float("nan")
    sd = Z.std(0)
    rad = Z.norm(dim=1)
    c = Z - Z.mean(0)
    kurt = (c.pow(4).mean(0) / c.pow(2).mean(0).clamp_min(EPS).pow(2))
    step = max(1, k // blocks)
    prof = [{"from": i, "to": min(i + step, k), "sd": float(sd[i:i + step].mean())}
            for i in range(0, k, step)]
    return {"n": int(M), "dims": int(k), "typical_radius": float(np.sqrt(k)),
            "sd": float(Z.std()), "sd_axis_mean": float(sd.mean()), "sd_axis_min": float(sd.min()),
            "sd_axis_max": float(sd.max()), "sd_axis_corrected": float(sd.mean()) / c4,
            "sd_axis_c4": float(c4), "radius_mean": float(rad.mean()),
            "radius_sd": float(rad.std()), "radius_sd_gaussian": float(np.sqrt(0.5)),
            "kurtosis_mean": float(kurt.mean()), "kurtosis_expected": 3.0 * (M - 1) / (M + 1),
            "profile": prof}


def to_numpy(p: dict, *, basis_dtype=np.float16) -> dict:
    """Что кладётся на том: базис в fp16 (378 МБ вместо 756), остальное в fp32.

    Относительная ошибка fp16 на элементе базиса ~1e-3, и при суммировании
    2 048 слагаемых со случайными знаками она такой и остаётся — на два порядка
    меньше, чем разница между k = 1 024 и k = 2 048 по рендеру."""
    return {"basis": p["basis"].cpu().numpy().astype(basis_dtype),
            "mean": p["mean"].cpu().numpy().astype(np.float32),
            "lam": p["lam"].cpu().numpy().astype(np.float32),
            "eigvals_all": p["eigvals_all"].cpu().numpy().astype(np.float32)}


def truncate(p: dict, k: int) -> dict:
    """Тот же базис, обрезанный до k — представлениями, без копии на 756 МБ."""
    if not k or k >= p["k"]:
        return p
    return {**p, "basis": p["basis"][:, :k], "lam": p["lam"][:k], "k": int(k)}


def from_numpy(z, device, *, k: int = 0) -> dict:
    """Обратно в словарь `fit`. Файл читается один раз: лестница k берётся
    из `truncate`, а не повторным `from_numpy` (базис — 378 МБ на диске)."""
    W = torch.as_tensor(np.asarray(z["basis"], np.float32), device=device)
    p = {"basis": W, "lam": torch.as_tensor(np.asarray(z["lam"], np.float32), device=device),
         "mean": torch.as_tensor(np.asarray(z["mean"], np.float32), device=device),
         "dims": int(W.shape[0]), "k": int(W.shape[1])}
    return truncate(p, k)


def as_tokens(Z, tokens: int = 16, k_axis: int = 8):
    """(N, k) отбелённых координат -> (N, T, K, n) для `prior17`.

    Поток устроен так, что токен — это последняя ось, а признаки токена —
    произведение первых двух. При k = 2 048 и 16 токенах выходит
    (N, 16, 8, 16): 16 токенов по 128 признаков, внимание по 16 позициям
    вместо 721.

    Раскладка C-порядка означает, что компоненты **чередуются** по токенам:
    токену j достаются компоненты j, j+16, j+32, … То есть каждый токен несёт
    полосу всего спектра, а не его кусок. Для обучающих это безразлично (после
    отбеливания все координаты единичны), а на отложенных полезно: там
    дисперсия по спектру неоднородна (0,74 у головы против 0,87 у хвоста,
    19.0), и при чередовании токены остаются однородными между собой.
    """
    N, k = Z.shape
    if k % (tokens * k_axis):
        raise ValueError(f"{k} координат не делится на {tokens} токенов по {k_axis}")
    return Z.reshape(N, k // (tokens * k_axis), k_axis, tokens)


def from_tokens(X):
    """(N, T, K, n) -> (N, k), обратное к `as_tokens`."""
    return X.reshape(len(X), -1)
