r"""21б: спадает ли ковариация состояния с гексагональным расстоянием.

    python -m flydream.generate.hexcov19          # локально, CPU, $0, ~2 мин

Вопрос решает выбор архитектуры для пункта 22 против 23. Опубликованный
механизм новизны у диффузии — **локальность плюс эквивариантность**: Kamb &
Ganguli (ICML 2025) воспроизводят выходы обученной свёрточной модели
аналитически с r² 0,94–0,96, а локальное окно внимания у DiT при N = 10⁴ даёт
FID 16,1 → 11,7 (`reports/2026-09-21_research_path_to_a_video_generator.md`
§ 4). Но там же есть возражение (arXiv 2509.09672): локальность — свойство
**ковариации данных**, а не свёртки. Значит вопрос измеряемый, и мерить надо
на своих данных: если ковариация колонок спадает с расстоянием, гекс-локальное
внимание уместно; если она плоская, локальность навязывать нечему.

Считается **без скачивания состояний**: PCA-базис уже лежит локально, а
ковариация обучающих состояний в ранге k — это Σ = B·diag(λ)·Bᵀ. Числа
относятся к этому приближению (94,7 % обучающей дисперсии, 19.0 § 3) и к
z-scored DCT-состоянию, а не к сырым напряжениям.

Контроль — перестановка номеров колонок: спектр тот же, пространство
разрушено, кривая должна стать плоской.

Три разреза, каждый по своей оси состояния (16 DCT × 8 типов × 721 колонка):
пространство (колонка к колонке), тип к типу в одной колонке, коэффициент к
коэффициенту в одной колонке. Последние два отвечают на риск «128 каналов на
сайт» из § 7.1 того же отчёта.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from flydream.decode.hexraster import hex_distance
from flydream.generate import pca19 as P

EPS = 1e-12


def corr_of(S: torch.Tensor) -> torch.Tensor:
    """Ковариация -> корреляция, с защитой от нулевой дисперсии (пустой тип)."""
    d = S.diagonal().clamp_min(EPS).sqrt()
    return S / d[:, None] / d[None, :]


def by_distance(C: torch.Tensor, dist: np.ndarray, dmax: int) -> tuple[np.ndarray, np.ndarray]:
    """Средняя корреляция и средний модуль корреляции по расстоянию, 0..dmax."""
    d = torch.as_tensor(dist, device=C.device)
    keep = d <= dmax
    idx = d[keep].reshape(-1).long()
    val = C[keep].reshape(-1)
    n = torch.zeros(dmax + 1, device=C.device).index_add_(0, idx, torch.ones_like(val))
    s = torch.zeros(dmax + 1, device=C.device).index_add_(0, idx, val)
    a = torch.zeros(dmax + 1, device=C.device).index_add_(0, idx, val.abs())
    n = n.clamp_min(1.0)
    return (s / n).cpu().numpy(), (a / n).cpu().numpy()


def locality_length(curve: np.ndarray) -> float:
    """Расстояние, на котором корреляция падает ниже половины значения на d = 1.

    Возвращает `nan`, если не падает вовсе на измеренном участке — это и есть
    «ковариация не локальна»."""
    if len(curve) < 3 or not np.isfinite(curve[1]) or curve[1] <= 0:
        return float("nan")
    half = curve[1] / 2.0
    for d in range(2, len(curve)):
        if curve[d] < half:
            return float(d)
    return float("nan")


def run(pca_path: Path, *, dmax: int = 12, n_cols: int = 721, types: int = 8, dct: int = 16,
        shuffle_control: bool = True, seed: int = 0, log=print) -> dict:
    t0 = time.time()
    dev = torch.device("cpu")
    z = np.load(pca_path)
    p = P.from_numpy(z, dev)
    D, k = p["dims"], p["k"]
    if D != dct * types * n_cols:
        raise ValueError(f"размерность {D} не равна {dct} x {types} x {n_cols}")
    lam = p["lam"]
    B = p["basis"]
    dist = hex_distance(n_cols)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_cols)
    log(f"базис {D} x {k}, λ от {float(lam[0]):.3g} до {float(lam[-1]):.3g}; "
        f"расстояния до {dmax} шагов; {time.time() - t0:.0f} с")

    # --- пространство: колонка к колонке, внутри (коэффициент, тип) ----------
    space, space_ctrl, weight = np.zeros(dmax + 1), np.zeros(dmax + 1), 0.0
    abs_space = np.zeros(dmax + 1)
    per_dct = {}
    mass = np.zeros(dmax + 1)
    for t in range(dct):
        acc, acc_abs, w = np.zeros(dmax + 1), np.zeros(dmax + 1), 0.0
        for c in range(types):
            i0 = (t * types + c) * n_cols
            Bb = B[i0:i0 + n_cols]
            S = (Bb * lam) @ Bb.T
            var = float(S.diagonal().sum())
            if var <= EPS:                                            # тип без клеток в экспорте
                continue
            C = corr_of(S)
            m, a = by_distance(C, dist, dmax)
            acc += m * var; acc_abs += a * var; w += var
            space += m * var; abs_space += a * var; weight += var
            A = S.abs()
            tot = float(A.sum())
            dd = torch.as_tensor(dist, device=S.device)
            for dv in range(dmax + 1):
                mass[dv] += float(A[dd == dv].sum()) / max(tot, EPS) * var
            if shuffle_control:
                Cs = C[np.ix_(perm, perm)]
                ms, _ = by_distance(Cs, dist, dmax)
                space_ctrl += ms * var
        if w > 0:
            per_dct[t] = {"corr": (acc / w).tolist(), "abs_corr": (acc_abs / w).tolist()}
        log(f"  коэффициент {t:2}: d=1 {acc[1] / max(w, EPS):+.3f}, d=4 {acc[4] / max(w, EPS):+.3f}, "
            f"d={dmax} {acc[dmax] / max(w, EPS):+.3f}, {time.time() - t0:.0f} с")
    space /= max(weight, EPS); abs_space /= max(weight, EPS)
    space_ctrl /= max(weight, EPS); mass /= max(weight, EPS)

    # --- канал к каналу в одной колонке: тип и коэффициент -------------------
    def channel_corr(axis: str) -> dict:
        n = types if axis == "types" else dct
        acc = torch.zeros(n, n, device=dev)
        w = 0.0
        outer = dct if axis == "types" else types
        for o in range(outer):
            rows = []
            for j in range(n):
                t, c = (o, j) if axis == "types" else (j, o)
                rows.append(B[(t * types + c) * n_cols:(t * types + c) * n_cols + n_cols])
            X = torch.stack(rows)                                     # (n, cols, k)
            S = torch.einsum("ack,k,bck->ab", X, lam, X)              # сумма по колонкам и компонентам
            var = float(S.diagonal().sum())
            if var <= EPS:
                continue
            acc += corr_of(S) * var; w += var
        M = (acc / max(w, EPS))
        off = M - torch.diag(M.diagonal())
        return {"matrix": M.tolist(),
                "mean_abs_off_diagonal": float(off.abs().sum() / max(n * (n - 1), 1)),
                "max_abs_off_diagonal": float(off.abs().max())}

    out = {"pca": str(pca_path), "k": int(k), "dims": int(D), "dmax": int(dmax),
           "space": {"corr": space.tolist(), "abs_corr": abs_space.tolist(),
                     "shuffled_control": space_ctrl.tolist() if shuffle_control else None,
                     "mass_fraction": mass.tolist(),
                     "mass_within_1": float(mass[:2].sum()), "mass_within_2": float(mass[:3].sum()),
                     "locality_length": locality_length(abs_space),
                     "locality_length_signed": locality_length(space)},
           "space_per_dct": per_dct,
           "types": channel_corr("types"), "dct": channel_corr("dct"),
           "n_pairs_at_distance": [int((dist == d).sum()) for d in range(dmax + 1)],
           "seconds": round(time.time() - t0, 1)}
    return out


def main(argv=None) -> int:
    from flydream.model import ROOT

    try:                                                              # вывод в файл под cp1251 иначе
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")    # падает на знаке корня
    except (AttributeError, OSError):
        pass
    p = argparse.ArgumentParser()
    p.add_argument("--pca", default=str(ROOT / "data" / "prior19" / "pca2048.npz"))
    p.add_argument("--out", default=str(ROOT / "data" / "prior19"))
    p.add_argument("--tag", default="hexcov19")
    p.add_argument("--dmax", type=int, default=12)
    p.add_argument("--no-control", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    S = run(Path(a.pca), dmax=a.dmax, shuffle_control=not a.no_control, seed=a.seed)
    (out / f"{a.tag}.json").write_text(json.dumps(S, indent=1, ensure_ascii=False), encoding="utf-8")

    sp = S["space"]
    print("\nкорреляция колонок по гексагональному расстоянию (16 DCT x 8 типов, вес — дисперсия):")
    print("{:>4} {:>10} {:>10} {:>12} {:>10} {:>9}".format(
        "d", "корр.", "|корр.|", "перемешано", "доля массы", "пар"))
    for d in range(len(sp["corr"])):
        c = sp["shuffled_control"]
        print("{:4} {:10.3f} {:10.3f} {:12} {:10.3f} {:9}".format(
            d, sp["corr"][d], sp["abs_corr"][d],
            f"{c[d]:.3f}" if c else "—", sp["mass_fraction"][d], S["n_pairs_at_distance"][d]))
    print(f"\nдлина локальности (где |корр.| падает вдвое от d = 1): {sp['locality_length']}")
    print(f"масса ковариации в пределах одного шага: {100 * sp['mass_within_1']:.1f} %, "
          f"двух шагов: {100 * sp['mass_within_2']:.1f} %")
    print(f"каналы в одной колонке — типы: |вне диагонали| в среднем "
          f"{S['types']['mean_abs_off_diagonal']:.3f}, максимум {S['types']['max_abs_off_diagonal']:.3f}; "
          f"коэффициенты DCT: {S['dct']['mean_abs_off_diagonal']:.3f} / "
          f"{S['dct']['max_abs_off_diagonal']:.3f}")
    print(f"\nwrote {out / a.tag}.json  ({S['seconds']:.0f} s, $0)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
