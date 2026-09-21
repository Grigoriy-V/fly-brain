r"""23: сид-тест для потока, который выдаёт блок состояния напрямую.

    python -m flydream.generate.seed23          # локально, CPU, $0

`seed19` умеет поток над латентом PCA: он разыгрывает координаты и декодирует
их базисом. Здесь базиса нет вовсе — поток 23 живёт в самом пространстве
блока (721 x 32), — поэтому ступень декодирования выпадает, а всё остальное
должно остаться тем же кодом, иначе числа перестанут быть сравнимыми с 19.3,
22.6 и с полами 22.8.

Группы:

- **настоящее состояние блока** — потолок цепочки: 13B из того, что снял мозг;
- **сид этого клипа через поток** — блок обращается в шум и возвращается
  обратно, то есть проверка, что поток ничего не теряет по дороге;
- **свежий розыгрыш** — та самая цель: видео без исходного клипа;
- рядом всегда сырое видео корпуса.

Ворота не считаются намеренно: шесть типов из восьми объявлены отсутствующими
честной маской 13B, и расстояние до такого заказа мерить незачем (21в).
Судьи — r к сырому, доля ровного поля, контраст и ближайшее обучающее видео,
все с измеренными полами (22.8).

Пачка 13B равна группе с самого начала: шум там берётся по пачке (ISS-0009).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from flydream.decode import pairs as P13
from flydream.generate import gen13b as G
from flydream.generate import learned as L
from flydream.generate import prior17 as R
from flydream.generate.edges18 import describe
from flydream.generate.gen13b import DEEP
from flydream.generate.invert import device_of, load_network, pixcorr_per_frame
from flydream.generate.pairs13 import simulate_states
from flydream.generate.prompts14 import Deep
from flydream.generate.roundtrip13 import build_states
from flydream.generate.vaeval18 import nearest


def geometry_of(eps: torch.Tensor) -> dict:
    """Геометрия прообразов: попадают ли они на гауссову оболочку."""
    e = eps.reshape(len(eps), -1).float()
    r = e.norm(dim=1)
    m, s = e.mean(0), e.std(0) + 1e-12
    return {"sd": float(e.std()), "radius_mean": float(r.mean()), "radius_sd": float(r.std()),
            "typical_radius": float(e.shape[1] ** 0.5),
            "kurtosis_mean": float((((e - m) / s) ** 4).mean(0).mean())}


def run(model: str, flow_ckpt: Path, pca_path: Path, gen_ckpt: Path, manifest: dict, columns: dict,
        corpus: Path, *, n_clips: int = 6, frames: int = 40, margin: int = 5, dt: float = 0.02,
        t_pre: float = 1.0, seed: int = 0, steps: int = 100, log=print) -> dict:
    t0 = time.time()
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    net = load_network(model); dev = device_of(net)
    _, index = P13.type_index(net.connectome)
    d = Deep(manifest, columns)
    gen, gmeta = G.load(gen_ckpt, dev)
    mean = np.array(gmeta["mean"], np.float32); std = np.array(gmeta["std"], np.float32)
    flow, fmeta = R.load(flow_ckpt, dev)
    if fmeta.get("backbone") != "hex":
        raise SystemExit("этот сид-тест для потока над блоком; для латента PCA есть seed19")
    btypes = list(fmeta["types"])
    ch = [DEEP.index(t) for t in btypes]
    # Мета нужна только ради DCT и масштаба коэффициентов — базис PCA здесь не
    # используется, но пространство обязано быть тем же, в котором учился поток.
    pmeta = json.loads(str(np.load(pca_path)["meta"]))
    log(f"поток {fmeta['width']}x{fmeta['depth']} над блоком {btypes} -> каналы {ch}; "
        f"{fmeta['parameters']} параметров, валидация {fmeta['best_val']:.4f}")

    built = build_states(net, index, frames=frames, margin=margin, dt=dt, t_pre=t_pre, seed=seed)
    sa = next(s for s in built if s["name"] == "clip_A")
    w0, w1 = sa["window"]

    cz = np.load(Path(corpus) / "videos.npz")
    cm = json.loads((Path(corpus) / "pairs_manifest.json").read_text(encoding="utf-8"))
    bank_all = np.asarray(cz["videos"][:, :frames], np.float16)
    clip_idx = rng.choice(np.asarray(cm["split"]["test"]), n_clips, replace=False)
    raw = np.asarray(cz["videos"][clip_idx][:, :frames], np.float32)
    st_real = simulate_states(net, np.asarray(cz["videos"], np.float16)[clip_idx], d.cells_all,
                              dt, t_pre, 8).astype(np.float32)
    maps = L.to_maps(st_real.astype(np.float16), d.layout, len(DEEP))[:, :frames].astype(np.float32)
    real = (maps - mean[None, None, :, None]) / std[None, None, :, None]
    # Мета базиса урезана до типов блока (`pca19 types=`), поэтому DCT и
    # масштаб коэффициентов применяются к блоку, а не к полному состоянию.
    block = R.to_model_space(pmeta, real[:, :, ch], dev)               # (n, dct_k, 2, 721)
    log(f"настоящие блоки {tuple(block.shape)}, ст. откл. {float(block.std()):.3f} "
        f"(обучение шло при 1,0); {time.time() - t0:.0f} с")

    def to_full(b: torch.Tensor) -> np.ndarray:
        """Блок на свои места в карте, остальное ноль — 13B знает это по маске."""
        m2 = R.from_model_space(pmeta, b)                              # (n, frames, 2, 721)
        full = np.zeros((len(m2), *real.shape[1:]), np.float32)
        full[:, :, ch] = m2
        return full

    with torch.no_grad():
        eps_clip = R.invert(flow, block, steps=steps)                  # блок -> шум
        b_clip = R.integrate(flow, eps_clip, steps=steps)              # и обратно
        e_draw = torch.randn(n_clips, *block.shape[1:], device=dev,
                             generator=torch.Generator(device=dev).manual_seed(9000 + seed))
        b_draw = R.integrate(flow, e_draw, steps=steps)
    out = {"flow": str(flow_ckpt), "types": btypes, "steps": steps, "n_clips": n_clips,
           "preimage": geometry_of(eps_clip),
           "round_trip_error": float((b_clip - block).norm() / block.norm()),
           "draw_geometry": geometry_of(b_draw), "block_geometry": geometry_of(block)}
    log(f"прообразы клипов: радиус {out['preimage']['radius_mean']:.1f} ± "
        f"{out['preimage']['radius_sd']:.2f} при оболочке {out['preimage']['typical_radius']:.1f}, "
        f"эксцесс {out['preimage']['kurtosis_mean']:.2f}; замыкание "
        f"{100 * out['round_trip_error']:.1f} %")

    groups = {"настоящее состояние блока": to_full(block),
              "сид от клипа через поток": to_full(b_clip),
              "свежий розыгрыш": to_full(b_draw)}
    names = [f"{g}|{i}" for g in groups for i in range(n_clips)]
    cond = torch.as_tensor(np.stack([groups[n.rsplit("|", 1)[0]][int(n.rsplit("|", 1)[1])]
                                     for n in names]), device=dev)
    bits = np.array([1.0 if t in btypes else 0.0 for t in DEEP], np.float32)
    mask = torch.as_tensor(np.tile(bits, (len(names), 1)), device=dev)
    vids = []
    with torch.no_grad():
        for i in range(0, len(names), n_clips):                        # пачка = группа (ISS-0009)
            gg = torch.Generator(device=dev).manual_seed(1000 + seed)
            vids.append(G.sample(gen, cond[i:i + n_clips], mask[i:i + n_clips],
                                 steps=20, generator=gg).cpu().numpy())
    videos = np.concatenate(vids).astype(np.float32)
    log(f"{len(videos)} видео отрисовано, {time.time() - t0:.0f} с")

    nb = np.asarray(L.neighbour_index(721))
    out |= {"clip_idx": clip_idx.tolist(), "groups": {}}
    for g in groups:
        sel = [i for i, n in enumerate(names) if n.startswith(g + "|")]
        s = describe([videos[i] for i in sel], nb)
        near = [nearest(videos[i], bank_all) for i in sel]
        out["groups"][g] = {
            "r_to_raw": float(np.mean([pixcorr_per_frame(videos[i], raw[j]).mean()
                                       for j, i in enumerate(sel)])),
            "frac_flat": s["frac_flat"]["mean"], "kurtosis": s["grad_kurtosis"]["mean"],
            "neigh_r": s["neigh_r"]["mean"], "sd": s["sd"]["mean"],
            "nearest_r": float(np.mean([r for _, r in near])),
            "nearest_idx": [int(j) for j, _ in near]}
        v = out["groups"][g]
        log(f"  {g}: r {v['r_to_raw']:.3f}, ровного {100 * v['frac_flat']:.1f} %, "
            f"контраст {v['sd']:.3f}, ближайшее {v['nearest_r']:.3f}")
    rw = describe(list(raw), nb)
    out["raw"] = {"frac_flat": rw["frac_flat"]["mean"], "kurtosis": rw["grad_kurtosis"]["mean"],
                  "neigh_r": rw["neigh_r"]["mean"], "sd": rw["sd"]["mean"]}
    out["seconds"] = round(time.time() - t0, 1)
    arrays = {f"video__{n}": videos[i] for i, n in enumerate(names)}
    arrays.update({f"raw__{i}": raw[i] for i in range(n_clips)})
    return {"summary": out, "arrays": arrays}


def main(argv=None) -> int:
    from flydream.generate.invert import settings
    from flydream.model import ROOT

    try:                                                              # вывод в файл под cp1251 иначе
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")    # падает на знаке корня
    except (AttributeError, OSError):
        pass
    g = settings().get("gen13b", {})
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="malecns")
    p.add_argument("--flow", default=str(ROOT / "data" / "prior23" / "prior23_hex_ab.pt"))
    p.add_argument("--pca", default=str(ROOT / "data" / "prior19" / "pca_ab2048.npz"))
    p.add_argument("--gen", default=str(ROOT / "data" / "gen13b" / "sit.pt"))
    p.add_argument("--pairs13", default=str(ROOT / "data" / "pairs13"))
    p.add_argument("--corpus", default=str(ROOT / "data" / "corpus18"))
    p.add_argument("--out", default=str(ROOT / "data" / "prior23"))
    p.add_argument("--tag", default="seed23")
    p.add_argument("--clips", type=int, default=6)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    pdir = Path(a.pairs13)
    manifest = json.loads((pdir / "manifest.json").read_text(encoding="utf-8"))
    columns = json.loads((pdir / "columns.json").read_text(encoding="utf-8"))
    r = run(a.model, Path(a.flow), Path(a.pca), Path(a.gen), manifest, columns, Path(a.corpus),
            n_clips=a.clips, frames=g.get("frames", 40), margin=g.get("margin", 5),
            dt=g.get("dt", 0.02), t_pre=g.get("t_pre", 1.0), steps=a.steps, seed=a.seed)
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{a.tag}.json").write_text(json.dumps(r["summary"], ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    np.savez_compressed(outdir / f"{a.tag}.npz", **r["arrays"])
    S = r["summary"]
    print("\n{:30} {:>11} {:>9} {:>9} {:>10}".format("группа", "r к сырому", "ровного", "контраст",
                                                     "ближайшее"))
    for kk, v in S["groups"].items():
        print("{:30} {:11.3f} {:8.1f}% {:9.3f} {:10.3f}".format(
            kk, v["r_to_raw"], 100 * v["frac_flat"], v["sd"], v["nearest_r"]))
    w = S["raw"]
    print("{:30} {:>11} {:8.1f}% {:9.3f}".format("сырое видео корпуса", "—", 100 * w["frac_flat"],
                                                 w["sd"]))
    print(f"\n-> {outdir / a.tag}.{{json,npz}} за {S['seconds']:.0f} с, $0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
