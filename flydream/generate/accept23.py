r"""23: приёмка потока по распределению розыгрыша, без рендера и без карты.

Метрика найдена ревизией 22.7-22.8 и стоит перед всеми остальными: если
розыгрыши не похожи на данные **по распределению**, рисовать из них нечего, и
никакой сид-тест этого не исправит. Считается на процессоре за минуты.

Что меряется:

- **покоординатный эксцесс** розыгрышей против подвыборок ОБУЧАЮЩЕГО латента
  того же размера. Контроль по размеру выборки встроен, потому что эксцесс
  тяжёлого хвоста сильно занижается на малых выборках: по всем 13 555
  обучающим он читается 20,37, по 256 — 8,31 ± 1,17, а у гауссианы 2,97;
- **радиус и его разброс** против тех же подвыборок;
- **сдвиг точки** ‖z − ε‖ / ‖ε‖ и угол между входом и выходом: 22.7 намерил у
  плоского потока 19,0° против 90° у случайного поворота, то есть он почти
  тождественен по направлению, отчего интерполяция и выходила наложением.

Судить надо в одном пространстве, иначе числа несравнимы. Поток 23 работает
над сырым блоком 721 x 32, поток 22 работал над латентом PCA-1536; поэтому
розыгрыши 23 проецируются тем же замороженным базисом и сравниваются с теми
же точными обучающими латентами. Граница честности: проекция отбрасывает то,
чего в базисе нет, то есть эта метрика не видит содержания вне подпространства
PCA — она сравнима с 22.7 и не полна сама по себе.
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


def kurtosis_of(A: np.ndarray) -> np.ndarray:
    """Покоординатный эксцесс: четвёртый момент, стандартизованный по своей оси."""
    m, s = A.mean(0), A.std(0) + 1e-12
    return (((A - m) / s) ** 4).mean(0)


def geometry(A: np.ndarray) -> dict:
    r = np.linalg.norm(A, axis=1)
    return {"n": int(len(A)), "kurtosis": float(kurtosis_of(A).mean()),
            "radius_mean": float(r.mean()), "radius_sd": float(r.std()), "sd": float(A.std())}


def reference(Z: np.ndarray, n: int, repeats: int, seed: int = 0) -> dict:
    """Подвыборки обучающего латента того же размера, что и розыгрыш."""
    rng = np.random.default_rng(seed)
    rows = [geometry(Z[rng.choice(len(Z), n, replace=False)]) for _ in range(repeats)]
    out = {}
    for key in ("kurtosis", "radius_mean", "radius_sd", "sd"):
        a = np.array([r[key] for r in rows])
        out[key] = {"mean": float(a.mean()), "sd": float(a.std()),
                    "min": float(a.min()), "max": float(a.max())}
    out["repeats"], out["n"] = repeats, n
    return out


def run(flow_ckpt: Path, pca_path: Path, latent_path: Path, *, n: int = 256, steps: int = 100,
        k: int = 1536, repeats: int = 40, batch: int = 64, seed: int = 0, log=print) -> dict:
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

    g = torch.Generator(device=dev).manual_seed(4242 + seed)
    eps = torch.randn(n, frames, kk, cols, generator=g)
    draws = []
    with torch.no_grad():
        for i in range(0, n, batch):
            draws.append(R.integrate(flow, eps[i:i + batch], steps=steps))
            log(f"  розыгрышей {min(i + batch, n)}/{n}, {time.time() - t0:.0f} с")
    X = torch.cat(draws)
    flat_eps = eps.reshape(n, -1)
    flat_x = X.reshape(n, -1)
    moved = float((flat_x - flat_eps).norm(dim=1).mean() / flat_eps.norm(dim=1).mean())
    ra, rb = float(flat_eps.norm(dim=1).mean()), float(flat_x.norm(dim=1).mean())
    cos = (rb ** 2 + ra ** 2 - (moved * ra) ** 2) / (2 * ra * rb)
    Zd = P.encode(flat_x, pca).cpu().numpy().astype(np.float64)       # тот же базис, что судил 22.7

    ref = reference(Ztr, n, repeats, seed)
    got = geometry(Zd)
    gauss = geometry(np.random.default_rng(seed).standard_normal((n, k)))
    closed = (got["kurtosis"] - gauss["kurtosis"]) / (ref["kurtosis"]["mean"] - gauss["kurtosis"])
    out = {"flow": str(flow_ckpt), "k": k, "steps": steps, "n": n,
           "draw": got, "training_subsamples": ref, "gaussian": gauss,
           "kurtosis_closed": float(closed),
           "transport": {"moved": moved, "radius_in": ra, "radius_out": rb,
                         "cos": float(cos), "degrees": float(np.degrees(np.arccos(np.clip(cos, -1, 1))))},
           "block_sd": float(X.std()), "seconds": round(time.time() - t0, 1)}
    log(f"эксцесс: розыгрыш {got['kurtosis']:.2f}, обучающие {ref['kurtosis']['mean']:.2f} "
        f"± {ref['kurtosis']['sd']:.2f} [{ref['kurtosis']['min']:.2f}..{ref['kurtosis']['max']:.2f}], "
        f"гаусс {gauss['kurtosis']:.2f} -> пройдено {100 * closed:.0f} %")
    log(f"радиус: розыгрыш {got['radius_mean']:.2f} ± {got['radius_sd']:.2f}, обучающие "
        f"{ref['radius_mean']['mean']:.2f} ± {ref['radius_mean']['sd']:.2f}")
    log(f"перенос: сдвиг {100 * moved:.1f} % нормы, поворот {out['transport']['degrees']:.1f}° "
        f"(случайный дал бы 90°)")
    return out


def main(argv=None) -> int:
    ROOT = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser()
    p.add_argument("--flow", default=str(ROOT / "data" / "prior23" / "prior23_hex_ab.pt"))
    p.add_argument("--pca", default=str(ROOT / "data" / "prior19" / "pca_ab2048.npz"))
    p.add_argument("--latent", default=str(ROOT / "data" / "prior19" / "pca_ab2048_latent.npz"))
    p.add_argument("--out", default=str(ROOT / "data" / "prior23" / "accept23.json"))
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--k", type=int, default=1536)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--repeats", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    out = run(Path(a.flow), Path(a.pca), Path(a.latent), n=a.n, steps=a.steps, k=a.k,
              repeats=a.repeats, batch=a.batch, seed=a.seed)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {a.out} за {out['seconds']} с")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
