r"""19.4: где по времени сэмплер уезжает от истины.

    python -m flydream.generate.tprofile19        # локально, CPU, $0

У линейного интерполянта есть точный эталон. Обучение ставит
`x_t = (1 − t)·ε + t·x₁`, где ε стандартный нормальный, а x₁ — данные с
единичной дисперсией по осям и независимы от ε. Значит на каждом t дисперсия
по осям обязана быть

    sd(x_t) = √((1 − t)² + t²)·(с точностью до sd данных),

то есть 1,0 на концах и 0,707 в середине. Траектория сэмплера обязана идти по
этой кривой: если она уезжает, видно **на каком t** и в какую сторону.

Зачем это нужно. `sample_t` — логит-нормальное время (`sigmoid(N(0, 1))`,
практика SD3), поэтому обучение почти не заходит в t ≈ 0 и t ≈ 1, а сэмплер
стартует ровно в t = 0 равномерным шагом Эйлера. 19.2b измерил, что прообразы
отложенных лежат почти точно на оболочке (44,5 ± 0,92 при 45,3 ± 0,71), а
розыгрыш вперёд промахивается по масштабу на 22 %. Эта разница должна иметь
адрес во времени, и здесь он ищется.

Сравниваются три траектории:
* **вперёд из розыгрыша** — как сэмплит генератор;
* **вперёд из прообраза** настоящего латента — тот же путь, но старт в точке,
  которую поток сам породил;
* **истина** — интерполянт между настоящим латентом и его же прообразом.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from flydream.generate import pca19 as P
from flydream.generate import prior17 as R


@torch.no_grad()
def trajectory(flow, x0: torch.Tensor, *, steps: int, tokens: int, marks) -> dict:
    """Интегрирует от t = 0 до 1 и запоминает ст. откл. по осям на отметках."""
    x = P.as_tokens(x0, tokens=tokens)
    out, m = {}, sorted(set(marks) | {0.0, 1.0})
    if 0.0 in m:
        out[0.0] = float(x.float().std())
    for j in range(steps):
        t = j / steps
        x = x + flow(x, torch.full((len(x),), t, device=x.device)) / steps
        nt = (j + 1) / steps
        for mk in m:
            if abs(nt - mk) < 0.5 / steps:
                out[mk] = float(x.float().std())
    return {"sd": out, "final": x.reshape(len(x0), -1)}


def main(argv=None) -> int:
    from flydream.model import ROOT

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
    p = argparse.ArgumentParser()
    p.add_argument("--flow", default=str(ROOT / "data" / "prior19" / "flow_pca2048_b256.pt"))
    p.add_argument("--latent", default=str(ROOT / "data" / "prior19" / "pca2048_latent.npz"))
    p.add_argument("--out", default=str(ROOT / "data" / "prior19"))
    p.add_argument("--tag", default="tprofile19")
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--threads", type=int, default=16)
    a = p.parse_args(argv)
    torch.set_num_threads(a.threads)
    dev = torch.device("cpu")
    flow, fmeta = R.load(a.flow, dev)
    tokens = int(fmeta.get("n", 16))
    z = np.load(a.latent)
    zt = torch.as_tensor(np.asarray(z["z_train"][:a.n], np.float32))
    sd_data = float(zt.std())
    k = zt.shape[1]
    marks = [round(0.1 * i, 1) for i in range(11)]

    g = torch.Generator().manual_seed(4242)
    eps = torch.randn(a.n, k, generator=g)
    pre = R.invert(flow, P.as_tokens(zt, tokens=tokens), steps=a.steps).reshape(a.n, -1)

    draw = trajectory(flow, eps, steps=a.steps, tokens=tokens, marks=marks)
    back = trajectory(flow, pre, steps=a.steps, tokens=tokens, marks=marks)
    truth = {t: float(((1 - t) * pre + t * zt).std()) for t in marks}
    ideal = {t: float(np.sqrt((1 - t) ** 2 + (t * sd_data) ** 2)) for t in marks}

    print(f"данные: ст. откл. по осям {sd_data:.3f}; прообразы: {float(pre.std()):.3f}; "
          f"{a.n} образцов, {a.steps} шагов\n")
    print("{:>5} {:>10} {:>10} {:>12} {:>12}".format("t", "идеал", "истина", "из прообраза", "из розыгрыша"))
    for t in marks:
        print("{:5.1f} {:10.3f} {:10.3f} {:12.3f} {:12.3f}".format(
            t, ideal[t], truth[t], back["sd"].get(t, float("nan")), draw["sd"].get(t, float("nan"))))
    gd = P.geometry(draw["final"]); gb = P.geometry(back["final"]); gz = P.geometry(zt)
    print(f"\nконец пути: из розыгрыша ст. откл. {gd['sd']:.3f}, радиус {gd['radius_mean']:.1f} ± {gd['radius_sd']:.2f}")
    print(f"            из прообраза ст. откл. {gb['sd']:.3f}, радиус {gb['radius_mean']:.1f} ± {gb['radius_sd']:.2f}")
    print(f"            сами данные  ст. откл. {gz['sd']:.3f}, радиус {gz['radius_mean']:.1f} ± {gz['radius_sd']:.2f}")
    print(f"            замыкание прообраз→вперёд: {float((back['final'] - zt).norm(dim=1).mean() / zt.norm(dim=1).mean()) * 100:.2f} %")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rec = {"flow": a.flow, "n": a.n, "steps": a.steps, "sd_data": sd_data,
           "marks": marks, "ideal": ideal, "truth": truth,
           "from_preimage": back["sd"], "from_draw": draw["sd"],
           "end": {"draw": gd, "preimage": gb, "data": gz}}
    (out / f"{a.tag}.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out / a.tag}.json  ($0)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
