r"""29.6: геометрия розыгрышей статики против настоящих состояний.

    python -m flydream.generate.geom29            # локально, CPU, $0

Приёмка 22-23 читала эксцесс и радиус в базисе PCA-1536, потому что объект был
блок. Объект статики — 1 442 числа, и здесь ничего проецировать не нужно:
розыгрыш и настоящее состояние сравниваются покоординатно как есть.

Считается четыре величины, и третья — та, ради которой скрипт написан:

- **покоординатный эксцесс** и **радиус** — то же, что в `accept23`, эталон
  из подвыборок обучающего сплита того же размера, что и розыгрыш;
- **доля энергии в общей компоненте** — какую часть нормы несёт среднее по
  выборке. У статического состояния она велика: состояние в основном есть
  среднее корпуса плюс сравнительно небольшая индивидуальная часть. Число
  само по себе ничего не значит, значение имеет только разница с настоящими;
- **попарная корреляция после снятия общей компоненты** — разнообразие. Без
  снятия среднего она читается около 0,8 у любых состояний, настоящих в том
  числе, и ничего не говорит.

ВАЖНО про раскладку. Блок лежит в памяти как **(коэффициент, тип, колонка)**,
16 × 2 × 721 в C-порядке. Прочитанный как (колонка, канал), он даёт
правдоподобную, но неверную «статику» — при подготовке отчёта это дало 8 %
общей компоненты вместо 68 %.

Настоящие состояния берутся реконструкцией PCA-2048 (89,5 % дисперсии):
локально другого пути к ним нет, и это записано как ограничение отчёта.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def kurtosis_of(x: np.ndarray) -> float:
    """Покоординатный эксцесс: z-скор по выборке, четвёртый момент, среднее."""
    z = (x - x.mean(0)) / (x.std(0) + 1e-8)
    return float((z ** 4).mean())


def geometry(x: np.ndarray) -> dict:
    x = np.asarray(x, np.float64)
    centred = x - x.mean(0)
    pc = np.corrcoef(centred)
    iu = np.triu_indices(len(centred), 1)
    v = pc[iu]
    total = float(np.linalg.norm(x, axis=1).mean())
    common = float(np.linalg.norm(x.mean(0)))
    return {"n": int(len(x)),
            "kurtosis": kurtosis_of(x),
            "radius": total,
            "common_share": (common / total) ** 2,
            "pair_r_mean": float(v.mean()),
            "pair_r_p99": float(np.quantile(v, 0.99)),
            "pair_r_max": float(v.max())}


def real_static(pca, latent, split: str, n: int, rng=None) -> np.ndarray:
    """(n, 1442) настоящих статических состояний из латента PCA."""
    z = latent[split]
    idx = np.arange(n) if rng is None else rng.choice(len(z), n, replace=False)
    x = (z[idx] * np.sqrt(pca["lam"])) @ pca["basis"].astype(np.float32).T + pca["mean"]
    return x.reshape(len(idx), -1, 2, 721)[:, 0].reshape(len(idx), -1)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--draw", default=str(ROOT / "data" / "prior29" / "draw29.npz"))
    p.add_argument("--pca", default=str(ROOT / "data" / "prior19" / "pca_ab2048.npz"))
    p.add_argument("--latent", default=str(ROOT / "data" / "prior19" / "pca_ab2048_latent.npz"))
    p.add_argument("--repeats", type=int, default=20, help="подвыборок обучающего сплита для эталона")
    p.add_argument("--out", default=str(ROOT / "data" / "prior29" / "geom29.json"))
    a = p.parse_args(argv)

    draw = np.load(a.draw)["draw"]
    pca, latent = np.load(a.pca), np.load(a.latent)
    n = len(draw)
    rng = np.random.default_rng(0)

    out = {"draw": geometry(draw), "repeats": a.repeats}
    ref = [geometry(real_static(pca, latent, "z_train", n, rng)) for _ in range(a.repeats)]
    out["train"] = {k: {"mean": float(np.mean([r[k] for r in ref])),
                        "sd": float(np.std([r[k] for r in ref]))}
                    for k in ("kurtosis", "radius", "common_share", "pair_r_p99", "pair_r_max")}
    out["test"] = geometry(real_static(pca, latent, "z_test", n))
    out["gaussian_kurtosis"] = kurtosis_of(rng.standard_normal((n, draw.shape[1])))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    d, t = out["draw"], out["train"]
    print(f"эксцесс      розыгрыш {d['kurtosis']:.3f}   обучающие {t['kurtosis']['mean']:.3f} "
          f"± {t['kurtosis']['sd']:.3f}   гаусс {out['gaussian_kurtosis']:.3f}")
    print(f"радиус       розыгрыш {d['radius']:.2f}    обучающие {t['radius']['mean']:.2f} "
          f"± {t['radius']['sd']:.2f}")
    print(f"общая доля   розыгрыш {d['common_share']:.3f}    обучающие {t['common_share']['mean']:.3f}   "
          f"отложенные {out['test']['common_share']:.3f}")
    print(f"попарная p99 розыгрыш {d['pair_r_p99']:.3f}    обучающие {t['pair_r_p99']['mean']:.3f}   "
          f"отложенные {out['test']['pair_r_p99']:.3f}")
    print(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
