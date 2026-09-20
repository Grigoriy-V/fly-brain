"""19.1: приёмка первой стадии — отложенный клип через PCA и обратно.

    python -m flydream.generate.pcaval19          # локально, CPU, $0, ~2 мин

Критерий человека не меняется и метрики не отменяет, а заменяет
(`reports/2026-09-20_the_seed_problem.md` § 5): отложенный клип → мозг →
состояние → **PCA** → обратно → 13B → видео, сравнение **с сырым видео
корпуса**. Здесь проверяется только первая стадия: потока ещё нет, латент
задаётся самим клипом.

Заодно бесплатно берётся вся лестница k: один и тот же базис обрезается до
1 024 и 512, так что размен «точность против заполненности пространства» —
тот самый, который придётся делать, если розыгрыш сядет внутрь оболочки —
виден на картинке до того, как за что-то заплачено.

Опора сверху — 13B прямо из настоящего состояния: потолок цепочки, 0,952 на
этих же шести клипах (18.24c). Ворота считаются тем же кодом, что у всех
плеч пункта 18, и судьёй не являются.
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
from flydream.generate import pca19 as P
from flydream.generate import prior17 as R
from flydream.generate.edges18 import describe
from flydream.generate.gen13b import DEEP
from flydream.generate.invert import device_of, load_network, pixcorr_per_frame
from flydream.generate.pairs13 import simulate_states
from flydream.generate.prompts14 import Deep
from flydream.generate.roundtrip13 import build_states, round_trip
from flydream.generate.vaeval18 import nearest


def run(model: str, pca_path: Path, gen_ckpt: Path, manifest: dict, columns: dict, corpus: Path, *,
        ks=(2048, 1024, 512), n_clips: int = 6, frames: int = 40, margin: int = 5, dt: float = 0.02,
        t_pre: float = 1.0, seed: int = 0, log=print) -> dict:
    t0 = time.time()
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    net = load_network(model); dev = device_of(net)
    _, index = P13.type_index(net.connectome)
    d = Deep(manifest, columns)
    gen, gmeta = G.load(gen_ckpt, dev)
    mean = np.array(gmeta["mean"], np.float32); std = np.array(gmeta["std"], np.float32)

    z = np.load(pca_path)
    pmeta = json.loads(str(z["meta"]))
    fitted = json.loads(str(z["summary"])) if "summary" in z.files else {}
    p_full = P.from_numpy(z, dev)                                     # 378 МБ с диска — один раз на прогон
    ks = tuple(int(k) for k in ks if int(k) <= p_full["k"])
    log(f"базис {p_full['dims']} x {p_full['k']}, лестница k = {ks}, {time.time() - t0:.0f} с")

    built = build_states(net, index, frames=frames, margin=margin, dt=dt, t_pre=t_pre, seed=seed)
    sa = next(s for s in built if s["name"] == "clip_A")
    w0, w1 = sa["window"]
    ta = sa["target"][:, w0:w1, :][:, :, d.cells_all].cpu().numpy().astype(np.float32)
    var_ref = {t: float(ta[0][:, d.pos[t]].var()) + 1e-6 for t in DEEP}

    cz = np.load(Path(corpus) / "videos.npz")
    cm = json.loads((Path(corpus) / "pairs_manifest.json").read_text(encoding="utf-8"))
    bank_all = np.asarray(cz["videos"][:, :frames], np.float16)
    clip_idx = rng.choice(np.asarray(cm["split"]["test"]), n_clips, replace=False)
    raw = np.asarray(cz["videos"][clip_idx][:, :frames], np.float32)
    st = simulate_states(net, np.asarray(cz["videos"], np.float16)[clip_idx], d.cells_all,
                         dt, t_pre, 8).astype(np.float32)
    maps = L.to_maps(st.astype(np.float16), d.layout, len(DEEP))[:, :frames].astype(np.float32)
    real = (maps - mean[None, None, :, None]) / std[None, None, :, None]
    log(f"{n_clips} отложенных состояний, клипы {clip_idx.tolist()}, {time.time() - t0:.0f} с")

    x = R.to_model_space(pmeta, real, dev)                            # то же пространство, в котором строилась PCA
    flat = x.reshape(len(x), -1)
    groups, latent = {"настоящее состояние": real}, {}
    for k in ks:
        p = P.truncate(p_full, k)
        zk = P.encode(flat, p)
        latent[k] = P.geometry(zk)
        back = P.decode(zk, p).reshape(x.shape)
        groups[f"через PCA k = {k}"] = R.from_model_space(pmeta, back)
        log(f"  k = {k}: ст. откл. латента {latent[k]['sd']:.3f}, "
            f"радиус {latent[k]['radius_mean']:.1f} при √k = {latent[k]['typical_radius']:.1f}")

    jobs = {f"{k}|{i}": v[i] for k, v in groups.items() for i in range(n_clips)}
    names = list(jobs)
    cond = torch.as_tensor(np.stack([jobs[n] for n in names]), device=dev)
    mask = torch.ones(len(names), len(DEEP), device=dev)
    vids = []
    with torch.no_grad():
        for i in range(0, len(names), 8):
            gg = torch.Generator(device=dev).manual_seed(1000 + seed)  # один z у 13B везде
            vids.append(G.sample(gen, cond[i:i + 8], mask[i:i + 8], steps=20, generator=gg).cpu().numpy())
    videos = np.concatenate(vids).astype(np.float32)
    target_raw = np.stack([R.from_maps(R.unscale(jobs[n][None], mean, std), d.layout, len(d.cells_all))[0]
                           for n in names])
    rts = round_trip(net, videos, target_raw, d.cells_all, d.type_of, DEEP, dt, t_pre, margin, (0, frames), var_ref)
    log(f"{len(videos)} видео отрисовано и прогнано через мозг за {time.time() - t0:.0f} с")

    nb = np.asarray(L.neighbour_index(721))
    out = {"pca": str(pca_path), "ks": list(ks), "clip_idx": clip_idx.tolist(), "n_clips": n_clips,
           "latent": {str(k): v for k, v in latent.items()},
           "fitted_explained": fitted.get("explained", {}), "groups": {}}
    for k in groups:
        sel = [i for i, n in enumerate(names) if n.startswith(k + "|")]
        s = describe([videos[i] for i in sel], nb)
        r = float(np.mean([pixcorr_per_frame(videos[i], raw[j]).mean() for j, i in enumerate(sel)]))
        near = [nearest(videos[i], bank_all) for i in sel]
        out["groups"][k] = {"r_to_raw": r, "gate": float(np.median(rts[sel])),
                            "frac_flat": s["frac_flat"]["mean"], "kurtosis": s["grad_kurtosis"]["mean"],
                            "neigh_r": s["neigh_r"]["mean"], "sd": s["sd"]["mean"],
                            "nearest_r": float(np.mean([r_ for _, r_ in near])),
                            "nearest_idx": [int(j) for j, _ in near]}
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
    g = settings()
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="malecns")
    p.add_argument("--pca", default=str(ROOT / "data" / "prior19" / "pca2048.npz"))
    p.add_argument("--gen", default=str(ROOT / "data" / "gen13b" / "sit.pt"))
    p.add_argument("--pairs13", default=str(ROOT / "data" / "pairs13"))
    p.add_argument("--corpus", default=str(ROOT / "data" / "corpus18"))
    p.add_argument("--out", default=str(ROOT / "data" / "prior19"))
    p.add_argument("--tag", default="pcaval19")
    p.add_argument("--ks", default="2048,1024,512")
    p.add_argument("--clips", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    pdir = Path(a.pairs13)
    manifest = json.loads((pdir / "manifest.json").read_text(encoding="utf-8"))
    columns = json.loads((pdir / "columns.json").read_text(encoding="utf-8"))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    r = run(a.model, Path(a.pca), Path(a.gen), manifest, columns, Path(a.corpus),
            ks=tuple(int(x) for x in a.ks.split(",")), n_clips=a.clips,
            frames=g.get("frames", 40), margin=g.get("margin", 5), dt=g.get("dt", 0.02),
            t_pre=g.get("t_pre", 1.0), seed=a.seed)
    (out / f"{a.tag}.json").write_text(json.dumps(r["summary"], indent=1, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(out / f"{a.tag}.npz", **r["arrays"])
    S = r["summary"]
    print("\n{:24} {:>11} {:>8} {:>9} {:>9} {:>10}".format(
        "группа", "r к сырому", "ворота", "ровного", "контраст", "ближайшее"))
    for k, v in S["groups"].items():
        print("{:24} {:11.3f} {:8.4f} {:8.1f}% {:9.3f} {:10.3f}".format(
            k, v["r_to_raw"], v["gate"], 100 * v["frac_flat"], v["sd"], v["nearest_r"]))
    w = S["raw"]
    print("{:24} {:>11} {:>8} {:8.1f}% {:9.3f}".format("сырое видео корпуса", "—", "—",
                                                       100 * w["frac_flat"], w["sd"]))
    n = S["n_clips"]
    print(f"\nгеометрия отложенного латента (до всякого потока), {n} клипов — "
          f"судья здесь 19.0 на 1 246 отложенных, не эти шесть:")
    for k, v in S["latent"].items():
        print(f"  k = {k:>5}: ст. откл. {v['sd']:.3f} (по осям {v['sd_axis_mean']:.3f}, с поправкой "
              f"на выборку {v['sd_axis_corrected']:.3f}), радиус {v['radius_mean']:.1f} при "
              f"√k = {v['typical_radius']:.1f}")
    print(f"\nwrote {out / a.tag}.json / .npz  ({S['seconds']:.0f} s, $0)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
