r"""23.x: fixdraw23 — можно ли сдвинуть розыгрыш к данным на шаге сэмплинга, без дообучения.

    python -m flydream.generate.fixdraw23        # локально, CPU, $0, ~15-20 минут

`accept23` намерил три расхождения розыгрыша с данными в одном и том же
замороженном базисе PCA-1536:

1. обратный проход потока кладёт настоящие отложенные блоки на радиус
   136,2 ± 4,21 (`seed23.json`), а шум, из которого рисуют, сидит на оболочке
   √23072 = 151,9 — то есть розыгрыш стартует с оболочки на ~10 % шире, чем
   там, где на самом деле живут прообразы данных;
2. свежий розыгрыш в PCA-1536 даёт радиус 37,78 ± 16,80 против 35,50 ± 1,09 у
   подвыборок обучающих того же размера (`accept23.json`) — перелёт на 6,4 %;
3. покоординатный эксцесс розыгрыша 5,36 против 8,31 ± 1,17 у обучающих
   (гауссиана читается 2,97) — самое важное число, потому что оно ловит
   тяжёлый хвост, а не только масштаб.

Всё это — геометрия шума, которую видно ДО интегрирования. Значит есть шанс
подправить её на входе, без переобучения потока. Здесь это не гипотеза, а
измерение: каждый рычаг разыгрывается на ОДНОМ и том же базовом шуме (сид
4242, как в `accept23`), только с одной модификацией `eps`, и меряется тем же
кодом, что и `accept23` — иначе числа между рычагами и с `accept23.json`
несравнимы.

Рычаги:

- **control** — обычный `torch.randn`, без изменений; должен воспроизвести
  `accept23.json` — это встроенная проверка, что харнесс не разошёлся;
- **radius=R** — каждый образец шума масштабируется РОВНО на радиус R (по всем
  23072 координатам сразу), R из {136,2; 140; 145; 151,9} — точка 1 в лоб;
- **scale=s** — шум умножается на s < 1 (обычная температура/обрезка,
  s из {0,90; 0,95}); в отличие от `radius=R` это не фиксирует радиус, а
  сжимает всё его распределение, форма оболочки остаётся прежней.

**Предупреждение, ещё до чисел.** Пункт 20 («sdproj») уже поправлял геометрию
латента похожим сэмплерным трюком и на рендере оказался ПРОИГРЫШЕМ. Поэтому
геометрическое улучшение здесь не значит победу — только число, с которым
можно спорить.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from flydream.generate import pca19 as P
from flydream.generate import prior17 as R
from flydream.generate.accept23 import geometry, kurtosis_of, reference


def make_eps(base: torch.Tensor, arm: str, value: float | None) -> torch.Tensor:
    """Применяет рычаг рычага к ОБЩЕМУ базовому шуму — рычаги парны, не независимы.

    `base` — один и тот же розыгрыш `torch.randn` для всех рычагов; каждый
    рычаг лишь переиначивает его, поэтому разница между рычагами — это разница
    самих преобразований, а не разных случайных выборок."""
    if arm == "control":
        return base
    n = len(base)
    flat = base.reshape(n, -1)
    if arm == "radius":
        norm = flat.norm(dim=1, keepdim=True)
        return (flat * (float(value) / norm)).reshape(base.shape)
    if arm == "scale":
        return base * float(value)
    raise ValueError(f"неизвестный рычаг {arm}")


def energy_outside_pca(X: torch.Tensor, pca: dict) -> float:
    """Доля энергии розыгрыша, которую базис PCA-1536 не ловит вовсе.

    `decode(encode(X))` — проекция на подпространство базиса; остаток —
    то, что рычаг на входе потока в принципе не может исправить, потому что
    вся эта метрика (как и `accept23`) судит только внутри базиса."""
    Z = P.encode(X, pca)
    rec = P.decode(Z, pca)
    return float(((X - rec) ** 2).sum() / (X ** 2).sum())


def measure(flow, pca: dict, eps: torch.Tensor, *, steps: int, batch: int, ref: dict, gauss: dict,
            log=print) -> dict:
    """Один рычаг: интегрирует `eps`, проецирует в PCA-1536, судит тем же кодом, что `accept23.run`."""
    t0 = time.time()
    draws = []
    with torch.no_grad():
        for i in range(0, len(eps), batch):
            draws.append(R.integrate(flow, eps[i:i + batch], steps=steps))
    X = torch.cat(draws)
    flat_eps = eps.reshape(len(eps), -1)
    flat_x = X.reshape(len(X), -1)
    moved = float((flat_x - flat_eps).norm(dim=1).mean() / flat_eps.norm(dim=1).mean())
    ra, rb = float(flat_eps.norm(dim=1).mean()), float(flat_x.norm(dim=1).mean())
    cos = (rb ** 2 + ra ** 2 - (moved * ra) ** 2) / (2 * ra * rb)
    Zd = P.encode(flat_x, pca).cpu().numpy().astype(np.float64)

    got = geometry(Zd)
    # тот же код, что внутри geometry() — если числа разошлись, харнесс сломан
    assert abs(float(kurtosis_of(Zd).mean()) - got["kurtosis"]) < 1e-9
    closed = (got["kurtosis"] - gauss["kurtosis"]) / (ref["kurtosis"]["mean"] - gauss["kurtosis"])
    outside = energy_outside_pca(flat_x, pca)
    out = {"draw": got, "kurtosis_closed": float(closed),
           "transport": {"moved": moved, "radius_in": ra, "radius_out": rb, "cos": float(cos),
                         "degrees": float(np.degrees(np.arccos(np.clip(cos, -1, 1))))},
           "energy_outside_pca": outside, "seconds": round(time.time() - t0, 1)}
    log(f"  эксцесс {got['kurtosis']:.2f} ({100 * closed:.0f} % пути к обучающим), "
        f"радиус {got['radius_mean']:.2f} ± {got['radius_sd']:.2f}, "
        f"перенос {100 * moved:.1f} % нормы, {out['transport']['degrees']:.1f}°, "
        f"вне PCA {100 * outside:.2f} %, {out['seconds']:.0f} с")
    return out


def sweep(flow_ckpt: Path, pca_path: Path, latent_path: Path, accept_path: Path | None, *,
          n: int = 256, steps: int = 100, k: int = 1536, batch: int = 64, repeats: int = 40,
          seed: int = 0, log=print) -> dict:
    t0 = time.time()
    dev = torch.device("cpu")
    flow, fmeta = R.load(flow_ckpt, dev)
    pz = np.load(pca_path)
    pca = P.truncate(P.from_numpy(pz, dev), k)
    zl = np.load(latent_path)
    Ztr = np.asarray(zl["z_train"][:, :k], np.float64)
    frames, kk, cols = int(fmeta["frames"]), int(fmeta["k"]), int(fmeta["n"])
    log(f"поток {fmeta.get('backbone') or 'sit'} {fmeta['width']}x{fmeta['depth']}, "
        f"({frames}, {kk}, {cols}); базис {pca['dims']} -> {k}; обучающих {len(Ztr)}")

    ref = reference(Ztr, n, repeats, seed)
    gauss = geometry(np.random.default_rng(seed).standard_normal((n, k)))
    log(f"обучающие подвыборки: эксцесс {ref['kurtosis']['mean']:.2f} ± {ref['kurtosis']['sd']:.2f} "
        f"[{ref['kurtosis']['min']:.2f}..{ref['kurtosis']['max']:.2f}], гаусс {gauss['kurtosis']:.2f}")

    # общий базовый розыгрыш — тот же вызов, что и в accept23.run(), так что
    # control обязан воспроизвести accept23.json бит в бит (Euler детерминирован)
    g = torch.Generator(device=dev).manual_seed(4242 + seed)
    base = torch.randn(n, frames, kk, cols, generator=g)

    arms = [("control", None)]
    arms += [("radius", r) for r in (136.2, 140.0, 145.0, 151.9)]
    arms += [("scale", s) for s in (0.90, 0.95)]

    results = {}
    for name, value in arms:
        label = name if value is None else f"{name}={value}"
        log(f"-- {label} --")
        eps = make_eps(base, name, value)
        results[label] = measure(flow, pca, eps, steps=steps, batch=batch, ref=ref, gauss=gauss, log=log)

    control_check = None
    if accept_path is not None and Path(accept_path).exists():
        prev = json.loads(Path(accept_path).read_text(encoding="utf-8"))
        c = results["control"]
        dk = abs(c["draw"]["kurtosis"] - prev["draw"]["kurtosis"])
        dr = abs(c["draw"]["radius_mean"] - prev["draw"]["radius_mean"])
        control_check = {"kurtosis_diff": dk, "radius_diff": dr,
                          "reproduced": bool(dk < 1e-3 and dr < 1e-3)}
        log(f"control против {accept_path}: Δэксцесс {dk:.2e}, Δрадиус {dr:.2e} -> "
            f"{'воспроизвёл' if control_check['reproduced'] else 'РАСХОДИТСЯ'}")

    out = {"flow": str(flow_ckpt), "k": k, "steps": steps, "n": n, "seed": seed,
           "training_subsamples": ref, "gaussian": gauss, "control_check": control_check,
           "arms": results, "seconds": round(time.time() - t0, 1)}
    return out


def print_table(out: dict) -> None:
    header = (f"{'рычаг':<14}{'эксцесс':>9}{'% пути':>8}{'радиус':>9}{'sd':>7}"
              f"{'перенос%':>10}{'угол°':>7}{'вне PCA%':>9}")
    print(header)
    print("-" * len(header))
    for label, r in out["arms"].items():
        d, t = r["draw"], r["transport"]
        print(f"{label:<14}{d['kurtosis']:>9.2f}{100 * r['kurtosis_closed']:>8.0f}"
              f"{d['radius_mean']:>9.2f}{d['radius_sd']:>7.2f}"
              f"{100 * t['moved']:>10.1f}{t['degrees']:>7.1f}{100 * r['energy_outside_pca']:>9.2f}")


def main(argv=None) -> int:
    ROOT = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser()
    p.add_argument("--flow", default=str(ROOT / "data" / "prior23" / "prior23_hex_ab.pt"))
    p.add_argument("--pca", default=str(ROOT / "data" / "prior19" / "pca_ab2048.npz"))
    p.add_argument("--latent", default=str(ROOT / "data" / "prior19" / "pca_ab2048_latent.npz"))
    p.add_argument("--accept", default=str(ROOT / "data" / "prior23" / "accept23.json"))
    p.add_argument("--out", default=str(ROOT / "data" / "prior23" / "fixdraw23.json"))
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--k", type=int, default=1536)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--repeats", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    out = sweep(Path(a.flow), Path(a.pca), Path(a.latent), Path(a.accept) if a.accept else None,
                n=a.n, steps=a.steps, k=a.k, batch=a.batch, repeats=a.repeats, seed=a.seed)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {a.out} за {out['seconds']} с")
    print_table(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
