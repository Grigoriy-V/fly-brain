r"""22.7: что поток выучил из распределения латента, а что нет.

    python tools/fig_dist22.py

Никакого рендера: только поток на процессоре и сохранённые латенты. Судья —
обучающий сплит (тот, на котором поток учился), а не отложенный: сравнение с
отложенным и было дефектом ISS-0010. Контроль по размеру выборки встроен —
обучающие берутся подвыборками того же n, что и розыгрыши.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flydream.generate import pca19 as P          # noqa: E402
from flydream.generate import prior17 as R        # noqa: E402


def kurt(A: np.ndarray) -> np.ndarray:
    m, s = A.mean(0), A.std(0) + 1e-12
    return (((A - m) / s) ** 4).mean(0)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--flow", default=str(ROOT / "data" / "prior22" / "prior22_ab1536.pt"))
    p.add_argument("--latent", default=str(ROOT / "data" / "prior19" / "pca_ab2048_latent.npz"))
    p.add_argument("--k", type=int, default=1536)
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--steps", type=int, default=100)   # 20 шагов не сошлись: радиус 50,7 против 42,6
    p.add_argument("--repeats", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(ROOT / "reports" / "figures" / "2026-09-21_malecns_dist22"))
    a = p.parse_args(argv)
    t0, k, n = time.time(), a.k, a.n
    ru = lambda x: f"{x}".replace(".", ",")                           # noqa: E731 — decimal comma, numbers only

    dev = torch.device("cpu")
    flow, fmeta = R.load(Path(a.flow), dev)
    tokens = int(fmeta.get("n", 16))
    z = np.load(a.latent)
    Xtr = np.asarray(z["z_train"][:, :k], np.float64)
    print(f"поток {fmeta['width']}x{fmeta['depth']}, {tokens} токенов; обучающих {len(Xtr)}")

    g = torch.Generator(device=dev).manual_seed(4242 + a.seed)
    eps = torch.randn(n, k, generator=g)
    with torch.no_grad():
        D = R.integrate(flow, P.as_tokens(eps, tokens=tokens), steps=a.steps).reshape(n, -1).numpy()
    D = np.asarray(D, np.float64)
    # Контроль: промах по радиусу должен быть свойством модели, а не интегратора.
    with torch.no_grad():
        D2 = np.asarray(R.integrate(flow, P.as_tokens(eps, tokens=tokens),
                                    steps=2 * a.steps).reshape(n, -1).numpy(), np.float64)
    conv = {"steps": a.steps, "kurtosis": float(kurt(D).mean()), "kurtosis_2x": float(kurt(D2).mean()),
            "radius": float(np.linalg.norm(D, axis=1).mean()),
            "radius_2x": float(np.linalg.norm(D2, axis=1).mean())}
    print(f"{n} розыгрышей за {time.time() - t0:.0f} с; сходимость интегратора: эксцесс "
          f"{conv['kurtosis']:.2f} -> {conv['kurtosis_2x']:.2f}, радиус {conv['radius']:.2f} -> "
          f"{conv['radius_2x']:.2f} при удвоении шагов")

    rng = np.random.default_rng(a.seed)
    subs = [Xtr[rng.choice(len(Xtr), n, replace=False)] for _ in range(a.repeats)]
    ku_tr = np.array([kurt(S).mean() for S in subs])
    rad_tr = np.array([np.linalg.norm(S, axis=1).mean() for S in subs])
    G = rng.standard_normal((n, k))
    ku_d, ku_g = kurt(D).mean(), kurt(G).mean()
    closed = (ku_d - ku_g) / (ku_tr.mean() - ku_g)
    print(f"эксцесс: розыгрыш {ku_d:.2f}, обучающие {ku_tr.mean():.2f} ± {ku_tr.std():.2f}, "
          f"гаусс {ku_g:.2f} -> пройдено {100 * closed:.0f} %")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C = {"tr": "#1b5e20", "d": "#b71c1c", "g": "#666666"}
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.2))

    # 1. хвосты: все координаты, стандартизованные по своей оси
    def std_all(A):
        return ((A - A.mean(0)) / (A.std(0) + 1e-12)).reshape(-1)
    bins = np.linspace(-8, 8, 161)
    for A, c, lab in ((subs[0], C["tr"], "обучающие"), (D, C["d"], "розыгрыш"), (G, C["g"], "N(0, I)")):
        ax[0].hist(std_all(A), bins=bins, histtype="step", lw=1.8, color=c, density=True, label=lab)
    ax[0].set_yscale("log"); ax[0].set_xlim(-8, 8); ax[0].legend(frameon=False, fontsize=9)
    ax[0].set_title("хвосты координат (лог)", fontsize=10)
    ax[0].set_xlabel("значение координаты в своих ст. отклонениях", fontsize=8)

    # 2. радиус
    rb = np.linspace(0, 110, 70)
    for A, c, lab in ((Xtr[rng.choice(len(Xtr), 2000, replace=False)], C["tr"], "обучающие"),
                      (D, C["d"], "розыгрыш"), (G, C["g"], "N(0, I)")):
        ax[1].hist(np.linalg.norm(A, axis=1), bins=rb, histtype="step", lw=1.8, color=c,
                   density=True, label=lab)
    ax[1].set_yscale("log")                                          # N(0,I) — игла ± 0,7, иначе давит остальных
    ax[1].legend(frameon=False, fontsize=9); ax[1].set_title("радиус латента (лог)", fontsize=10)
    ax[1].set_xlabel("‖z‖ при √k = 39,2", fontsize=8)

    # 3. эксцесс: полоса подвыборок против розыгрыша
    ax[2].axhspan(ku_tr.mean() - ku_tr.std(), ku_tr.mean() + ku_tr.std(), color=C["tr"], alpha=0.18)
    ax[2].axhline(ku_tr.mean(), color=C["tr"], lw=2)
    jit = 0.30 + 0.40 * rng.random(len(ku_tr))
    ax[2].scatter(jit, ku_tr, s=16, color=C["tr"], zorder=3, alpha=0.75,
                  label=f"обучающие, {a.repeats} подвыборок по {n}")
    ax[2].axhline(ku_d, color=C["d"], lw=2.5, label=f"розыгрыш {ru(f'{ku_d:.2f}')}")
    ax[2].axhline(ku_g, color=C["g"], lw=1.6, ls="--", label=f"N(0, I) {ru(f'{ku_g:.2f}')}")
    ax[2].set_xlim(0, 1); ax[2].set_xticks([]); ax[2].set_ylim(2.4, 11.4)
    ax[2].legend(frameon=False, fontsize=8, loc="upper left")
    ax[2].set_title("покоординатный эксцесс при одном n", fontsize=10)

    title = (
        f"Поток выучил второй момент латента и не выучил четвёртый. Слева: обучающий латент РАЗРЕЖЕН — "
        f"координаты почти всегда у нуля и изредка выстреливают далеко (эксцесс {ru(f'{ku_tr.mean():.2f}')} ± "
        f"{ru(f'{ku_tr.std():.2f}')}), розыгрыш ПЛОТЕН и почти гауссов ({ru(f'{ku_d:.2f}')} при "
        f"{ru(f'{ku_g:.2f}')} у N(0, I)): пройдено {ru(f'{100 * closed:.0f}')} % пути от шума к данным. "
        f"В центре: разброс радиуса поток ВЫУЧИЛ — из тонкой оболочки ± 0,7 он сделал широкое облако, как у "
        f"данных, — но промахнулся положением на {ru(f'{100 * (np.linalg.norm(D, axis=1).mean() / rad_tr.mean() - 1):.0f}')} % вверх. "
        f"Справа: контроль по размеру выборки, без которого сравнение нечестно — эксцесс тяжёлого хвоста "
        f"занижается на малых выборках, и {ru('20,37')} по всем 13 555 обучающим съёживается до "
        f"{ru(f'{ku_tr.mean():.2f}')} при n = {n}; розыгрыш лежит ниже всех {a.repeats} повторов. Плотная смесь "
        f"всех 1 536 компонент PCA в ЛИНЕЙНОМ базисе — это наложение всего на всё: на дуге между двумя сидами "
        f"двойная экспозиция, из чистого шума текстурный ковёр. Судья — ОБУЧАЮЩИЙ сплит, не отложенный "
        f"(ISS-0010). Промах по радиусу — свойство модели, а не интегратора: при удвоении шагов с {a.steps} до "
        f"{2 * a.steps} радиус идёт {ru(f'{conv['radius']:.2f}')} -> {ru(f'{conv['radius_2x']:.2f}')}, эксцесс "
        f"{ru(f'{conv['kurtosis']:.2f}')} -> {ru(f'{conv['kurtosis_2x']:.2f}')} (на 20 шагах не сходится: радиус "
        f"{ru('50,68')}). Ни рендера, ни 13B, ни мозга: поток на процессоре, {int(time.time() - t0)} с, даром.")
    import textwrap
    fig.suptitle(textwrap.fill(title, 170), fontsize=8.5, y=0.985, va="top")
    fig.subplots_adjust(left=0.045, right=0.99, bottom=0.11, top=0.74, wspace=0.17)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=110)
    print(f"{out.with_suffix('.png')} за {time.time() - t0:.0f} с")
    return 0


if __name__ == "__main__":
    sys.exit(main())
