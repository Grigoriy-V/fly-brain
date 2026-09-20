r"""20: поправки сэмплера против перелёта нормы. Только латент, без 13B и мозга.

    python -m flydream.generate.fix19             # локально, CPU, $0, ~3 мин

19.4 измерил подпись: траектория из прообраза настоящего латента ложится на
идеал `√((1−t)² + t²·sd²)` до третьего знака, а траектория из свежего
розыгрыша отрывается уже на t = 0,1 и приходит на 59 % выше. В литературе это
называется **exposure bias** (Ning et al., ICLR 2024,
`reports/2026-09-21_research_path_to_a_video_generator.md` § 2): сеть на
входах, которых в обучении не было, отдаёт выход большей нормы, и ошибка
накапливается по шагам. Там же два лечения, оба без переобучения:

1. **Epsilon Scaling** — делить выход сети на одну константу λ (`vscale`).
2. **Сдвиг сетки** — переставить узлы Эйлера, τ = s·u/(1 + (s−1)u) (`shift`,
   Esser et al. 2024), чтобы шагов было больше там, где поле хуже выучено.

Третья поправка здесь — не из литературы, а прямо из нашего измерения 19.4, и
помечена как своя: **проекция на идеальную кривую** (`sdproj`) — после каждого
шага масштаб всей партии приводится к `√((1−t)² + t²·sd²)`. Она использует
только ту статистику, которую 19.4 уже измерил, и сохраняет разброс радиусов
внутри партии (масштаб общий на партию, не на образец).

Цель сравнения — **обучающий** латент, а не отложенный: модель училась на нём.
Отложенный слабее (десять невиданных классов), и оба числа печатаются рядом.

Что здесь НЕ измеряется: стала ли картинка сценой. Это делает
`seed19.py --fix`, и только его ворота и r к сырому видео — приёмка.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from flydream.generate import pca19 as P
from flydream.generate import prior17 as R

KNOBS = ("steps", "vscale", "shift", "sdproj")


def parse_spec(text: str) -> dict:
    """`"vscale=1.1,steps=100"` -> {"steps": 100, "vscale": 1.1, "shift": 1.0, "sdproj": 0}."""
    out = {"steps": 0, "vscale": 1.0, "shift": 1.0, "sdproj": 0}
    for part in str(text).split(","):
        part = part.strip()
        if not part or part in ("base", "базовый"):
            continue
        if part == "sdproj":                                          # единственный флаг без значения
            out["sdproj"] = 1
            continue
        if "=" not in part:
            raise ValueError(f"не разобрать поправку {part!r}: нужно ключ=значение из {KNOBS}")
        key, val = part.split("=", 1)
        key = key.strip()
        if key not in KNOBS:
            raise ValueError(f"неизвестная ручка {key!r}, есть только {KNOBS}")
        out[key] = int(float(val)) if key in ("steps", "sdproj") else float(val)
    return out


def name_of(spec: dict, steps: int) -> str:
    bits = []
    if spec["vscale"] != 1.0:
        bits.append(f"vscale={spec['vscale']:g}")
    if spec["shift"] != 1.0:
        bits.append(f"shift={spec['shift']:g}")
    if spec["sdproj"]:
        bits.append("sdproj")
    if (spec["steps"] or steps) != steps:
        bits.append(f"steps={spec['steps'] or steps}")
    return "+".join(bits) if bits else "базовый"


def ideal_sd(t: float, sd_data: float) -> float:
    """Ст. отклонение интерполянта `x_t = (1−t)ε + t·x₁` при независимых ε и x₁."""
    return float(np.sqrt((1.0 - t) ** 2 + (t * sd_data) ** 2))


@torch.no_grad()
def integrate_fixed(flow, x: torch.Tensor, *, steps: int, spec: dict, sd_data: float,
                    tokens: int) -> torch.Tensor:
    """Один розыгрыш через поток с поправками из `spec`. Без поправок — ровно
    `R.integrate`, чтобы базовая строка совпадала с 19.3 до последнего знака."""
    n = int(spec["steps"] or steps)
    model = R.scaled(flow, spec["vscale"])
    if not spec["sdproj"]:
        return R.integrate(model, x, steps=n, shift=spec["shift"])
    for i in range(n):                                                # своя поправка, не из литературы
        t0 = R.shifted(i / n, spec["shift"])
        t1 = R.shifted((i + 1) / n, spec["shift"])
        t = torch.full((len(x),), t0, device=x.device)
        x = x + model(x, t) * (t1 - t0)
        want = ideal_sd(t1, sd_data)
        have = float(x.std())
        if have > 0:
            x = x * (want / have)
    return x


@torch.no_grad()
def velocity_profile(flow, x: torch.Tensor, *, steps: int, tokens: int) -> dict:
    """‖v‖ вдоль траектории — версия рис. 2 у Ning et al. в скоростях.

    Их утверждение: норма выхода сети на сэмплировании всегда больше, чем на
    обучении. Здесь то же самое мерится двумя стартами — из прообраза
    настоящего латента (то, что было в обучении) и из свежего розыгрыша."""
    out, cur = [], x
    for i in range(steps):
        t = torch.full((len(cur),), i / steps, device=cur.device)
        v = flow(cur, t)
        out.append({"t": round(i / steps, 4), "v_norm": float(v.reshape(len(v), -1).norm(dim=1).mean()),
                    "sd": float(cur.std())})
        cur = cur + v / steps
    return {"steps": steps, "marks": out, "end_sd": float(cur.std())}


def default_specs() -> list[str]:
    """Лестница ручек: сначала по одной, комбинации — вторым проходом, по лучшим."""
    out = ["base", "steps=50", "steps=100"]
    out += [f"vscale={v:g}" for v in (1.05, 1.10, 1.15, 1.20, 1.25, 1.35)]
    out += [f"shift={s:g}" for s in (1.5, 2, 3, 6)]
    out += ["sdproj", "sdproj,steps=100"]
    return out


def run(flow_ckpt: Path, latent_path: Path, *, specs=None, n: int = 512, steps: int = 20,
        seed: int = 0, profile_n: int = 256, log=print) -> dict:
    t0 = time.time()
    torch.manual_seed(seed)
    dev = torch.device("cpu")
    flow, fmeta = R.load(flow_ckpt, dev)
    tokens = int(fmeta.get("n", 16))
    zf = np.load(latent_path)
    z_train = torch.as_tensor(np.asarray(zf["z_train"], np.float32))
    z_test = torch.as_tensor(np.asarray(zf["z_test"], np.float32))
    k = z_train.shape[1]
    train, test = P.geometry(z_train), P.geometry(z_test)
    sd_data = float(train["sd"])
    log(f"поток {fmeta['width']}x{fmeta['depth']}, {tokens} токенов, k = {k}; цель — обучающий латент: "
        f"ст. откл. {sd_data:.3f}, радиус {train['radius_mean']:.1f} ± {train['radius_sd']:.2f} "
        f"(отложенный: {test['sd']:.3f}, {test['radius_mean']:.1f}); {time.time() - t0:.0f} с")

    out = {"flow": str(flow_ckpt), "k": int(k), "tokens": tokens, "n_draws": int(n), "steps": steps,
           "train": train, "test": test, "arms": {}}

    # --- A. ручки против геометрии розыгрыша --------------------------------
    gen = torch.Generator(device=dev).manual_seed(4242 + seed)         # тот же сид, что у draw_many в 19.3
    eps = torch.randn(n, k, generator=gen)
    for text in (specs or default_specs()):
        spec = parse_spec(text)
        nm = name_of(spec, steps)
        z = torch.cat([integrate_fixed(flow, P.as_tokens(eps[i:i + 256], tokens=tokens), steps=steps,
                                       spec=spec, sd_data=sd_data, tokens=tokens).reshape(-1, k)
                       for i in range(0, len(eps), 256)])
        g = P.geometry(z)
        g |= {"spec": spec, "sd_ratio": g["sd"] / sd_data,
              "radius_ratio": g["radius_mean"] / train["radius_mean"],
              "moved": float((z - eps).norm(dim=1).mean() / eps.norm(dim=1).mean())}
        out["arms"][nm] = g
        log(f"  {nm:22} ст. откл. {g['sd']:.3f} ({g['sd_ratio']:.3f} от данных), радиус "
            f"{g['radius_mean']:.1f} ± {g['radius_sd']:.2f} ({g['radius_ratio']:.3f}), "
            f"эксцесс {g['kurtosis_mean']:.2f}")

    # --- B. норма скорости: розыгрыш против прообраза ------------------------
    m = min(profile_n, len(z_test))
    zt = z_test[:m]
    pre = torch.cat([R.invert(flow, P.as_tokens(zt[i:i + 128], tokens=tokens), steps=steps).reshape(-1, k)
                     for i in range(0, m, 128)])
    out["velocity"] = {
        "preimage": velocity_profile(flow, P.as_tokens(pre, tokens=tokens), steps=steps, tokens=tokens),
        "draw": velocity_profile(flow, P.as_tokens(eps[:m], tokens=tokens), steps=steps, tokens=tokens),
        "n": int(m), "preimage_geometry": P.geometry(pre)}
    pv = out["velocity"]["preimage"]["marks"]
    dv = out["velocity"]["draw"]["marks"]
    out["velocity"]["ratio"] = [{"t": a["t"], "draw_over_preimage": a2["v_norm"] / max(a["v_norm"], 1e-9)}
                               for a, a2 in zip(pv, dv)]
    out["seconds"] = round(time.time() - t0, 1)
    return out


def main(argv=None) -> int:
    from flydream.model import ROOT

    try:                                                              # вывод в файл под cp1251 иначе
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")    # падает на знаке корня
    except (AttributeError, OSError):
        pass
    p = argparse.ArgumentParser()
    p.add_argument("--flow", default=str(ROOT / "data" / "prior19" / "flow_pca2048_b256.pt"))
    p.add_argument("--latent", default=str(ROOT / "data" / "prior19" / "pca2048_latent.npz"))
    p.add_argument("--out", default=str(ROOT / "data" / "prior19"))
    p.add_argument("--tag", default="fix19")
    p.add_argument("--specs", default="", help="через ; — например \"vscale=1.1;shift=2,vscale=1.05\"")
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--profile-n", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    specs = [s for s in a.specs.split(";") if s.strip()] or None
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    S = run(Path(a.flow), Path(a.latent), specs=specs, n=a.n, steps=a.steps,
            seed=a.seed, profile_n=a.profile_n)
    (out / f"{a.tag}.json").write_text(json.dumps(S, indent=1, ensure_ascii=False), encoding="utf-8")

    print("\n{:24} {:>9} {:>8} {:>10} {:>8} {:>9} {:>8}".format(
        "поправка", "ст.откл.", "к данным", "радиус", "разброс", "эксцесс", "сдвиг"))
    for nm, g in S["arms"].items():
        print("{:24} {:9.3f} {:8.3f} {:10.1f} {:8.2f} {:9.2f} {:7.1f}%".format(
            nm, g["sd"], g["sd_ratio"], g["radius_mean"], g["radius_sd"],
            g["kurtosis_mean"], 100 * g["moved"]))
    tr, te = S["train"], S["test"]
    print("{:24} {:9.3f} {:8.3f} {:10.1f} {:8.2f} {:9.2f}".format(
        "обучающий латент", tr["sd"], 1.0, tr["radius_mean"], tr["radius_sd"], tr["kurtosis_mean"]))
    print("{:24} {:9.3f} {:8.3f} {:10.1f} {:8.2f} {:9.2f}".format(
        "отложенный латент", te["sd"], te["sd"] / tr["sd"], te["radius_mean"], te["radius_sd"],
        te["kurtosis_mean"]))

    v = S["velocity"]
    print(f"\nнорма скорости вдоль траектории, {v['n']} штук на арму "
          f"(Ning et al.: на сэмплировании она всегда больше):")
    print("{:>6} {:>12} {:>12} {:>9}".format("t", "из прообраза", "из розыгрыша", "во сколько"))
    for a1, a2 in zip(v["preimage"]["marks"], v["draw"]["marks"]):
        if round(a1["t"] * 20) % 4 == 0 or a1["t"] > 0.9:
            print("{:6.2f} {:12.1f} {:12.1f} {:9.3f}".format(
                a1["t"], a1["v_norm"], a2["v_norm"], a2["v_norm"] / max(a1["v_norm"], 1e-9)))
    print(f"\nwrote {out / a.tag}.json  ({S['seconds']:.0f} s, $0)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
