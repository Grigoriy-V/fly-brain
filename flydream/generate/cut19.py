r"""21а: до какого среза состояния видео ещё держится.

    python -m flydream.generate.cut19             # локально, CPU, $0, ~6 мин

Вопрос человека (2026-09-21): «можно ли сократить параметры Т-уровня, чтобы у
нас была экономия… мб можно отбрасывать часть вообще и не задавать их, пока
видео будет держаться?» Состояние (92 288 чисел) БОЛЬШЕ клипа (81 920), и
розыгрыш состояния — это розыгрыш видео, поэтому каждое отброшенное число
уменьшает именно ту задачу, которая не решается
(`reports/2026-09-21_research_path_to_a_video_generator.md` § 7.1–7.2).

Здесь потока нет вовсе: берутся НАСТОЯЩИЕ состояния шести отложенных клипов,
срезаются, и видео 13B из срезанного состояния сравнивается с сырым видео
корпуса. Так измеряется только то, что теряет сама первая ступень.

Срезы задаются строками `вид=значение`:

* `pca=k` — оставить k главных компонент (лестница 19.1, продолженная вниз);
* `types=T4` — оставить одно семейство направлений, второе не задавать;
* `dct=k` — оставить k временных коэффициентов из 16;
* `rings=r` — оставить колонки в пределах r-го кольца от центра.

Суффикс `+complete` меняет смысл отброшенного: без него отброшенное остаётся
средним корпуса (ноль в z-scored координатах), с ним — **достраивается по PCA-
подпространству методом наименьших квадратов из того, что задано**. Это и есть
ответ на «не задавать»: если достройка держит видео, часть состояния можно не
генерировать, а вывести из остального.

Оговорки, которые нельзя опускать: 13B обучен на полном состоянии, поэтому
срез — это состояние, которого он не видел; а `rings` — это серая периферия, а
не меньший глаз.
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
from flydream.decode.hexraster import ring_of
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

KINDS = ("pca", "types", "dct", "rings")
DEFAULT = ("pca=2048", "pca=512", "pca=256", "pca=128", "types=T4", "types=T4+complete",
           "types=T5+complete", "dct=8", "dct=8+complete", "rings=10", "rings=10+complete")


def parse_cut(text: str) -> dict:
    """`"types=T4+complete"` -> {"kind": "types", "value": "T4", "complete": True}."""
    s = str(text).strip()
    complete = s.endswith("+complete")
    s = s[: -len("+complete")] if complete else s
    if "=" not in s:
        raise ValueError(f"не разобрать срез {text!r}: нужно вид=значение из {KINDS}")
    kind, value = (x.strip() for x in s.split("=", 1))
    if kind not in KINDS:
        raise ValueError(f"неизвестный срез {kind!r}, есть только {KINDS}")
    if kind == "pca" and complete:
        raise ValueError("pca=k уже наименьшие квадраты в подпространстве; +complete здесь не значит ничего")
    return {"kind": kind, "value": value, "complete": complete}


def mask_of(cut: dict, shape) -> torch.Tensor:
    """(D,) bool: какие числа состояния СЧИТАЮТСЯ ЗАДАННЫМИ этим срезом."""
    dct, types, cols = shape
    m = torch.zeros(dct, types, cols, dtype=torch.bool)
    if cut["kind"] == "types":
        keep = [i for i, t in enumerate(DEEP) if t.startswith(cut["value"])]
        if not keep:
            raise ValueError(f"нет типов, начинающихся на {cut['value']!r}, есть {list(DEEP)}")
        m[:, keep, :] = True
    elif cut["kind"] == "dct":
        k = int(cut["value"])
        if not 1 <= k <= dct:
            raise ValueError(f"dct={k} вне 1..{dct}")
        m[:k] = True
    elif cut["kind"] == "rings":
        r = int(cut["value"])
        ring = torch.as_tensor(np.asarray(ring_of(cols), np.int64))
        m[:, :, ring <= r] = True
    else:
        m[:] = True
    return m.reshape(-1)


def complete_in_subspace(x: torch.Tensor, mask: torch.Tensor, p: dict, *, ridge: float = 1e-6) -> tuple:
    """Достроить всё состояние по заданной части: наименьшие квадраты в базисе PCA.

    x = mean + B·c, и c берётся из тех координат, что заданы:
    c = (Bₒᵀ Bₒ + εI)⁻¹ Bₒᵀ (xₒ − meanₒ). Возвращает (x̂, относительная невязка
    на заданной части) — невязка показывает, вырожден ли этот подбор."""
    B, mu = p["basis"], p["mean"]
    Bo = B[mask]
    r = x[:, mask].float() - mu[mask]
    Gram = Bo.T @ Bo
    Gram += (ridge * float(Gram.diagonal().mean())) * torch.eye(Gram.shape[0], device=Gram.device)
    c = torch.linalg.solve(Gram, (r @ Bo).T).T
    fit = c @ Bo.T
    resid = float((fit - r).norm() / r.norm().clamp_min(1e-12))
    return mu + c @ B.T, resid


def apply_cut(flat: torch.Tensor, cut: dict, p_full: dict, shape) -> dict:
    """(N, D) -> словарь со срезанным состоянием, числом заданных чисел и невязкой."""
    D = flat.shape[1]
    if cut["kind"] == "pca":
        k = int(cut["value"])
        p = P.truncate(p_full, k)
        z = P.encode(flat, p)
        return {"x": P.decode(z, p), "given": int(k), "resid": None,
                "latent": P.geometry(z)}
    mask = mask_of(cut, shape).to(flat.device)
    given = int(mask.sum())
    if cut["complete"]:
        x, resid = complete_in_subspace(flat, mask, p_full)
        return {"x": x, "given": given, "resid": resid, "latent": None}
    x = torch.zeros_like(flat)                                        # ноль = среднее корпуса в z-scored
    x[:, mask] = flat[:, mask]
    return {"x": x, "given": given, "resid": None, "latent": None}


def run(model: str, pca_path: Path, gen_ckpt: Path, manifest: dict, columns: dict, corpus: Path, *,
        cuts=DEFAULT, n_clips: int = 6, frames: int = 40, margin: int = 5, dt: float = 0.02,
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
    p_full = P.from_numpy(z, dev)
    specs = [parse_cut(c) for c in cuts]
    log(f"базис {p_full['dims']} x {p_full['k']}, срезов {len(specs)}, {time.time() - t0:.0f} с")

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
    x = R.to_model_space(pmeta, real, dev)
    shape = tuple(x.shape[1:])
    flat = x.reshape(len(x), -1)
    log(f"{n_clips} отложенных состояний {shape}, клипы {clip_idx.tolist()}, {time.time() - t0:.0f} с")

    groups, info = {"настоящее состояние": real}, {}
    info["настоящее состояние"] = {"given": int(flat.shape[1]), "resid": None, "cut": None}
    for text, cut in zip(cuts, specs):
        c = apply_cut(flat, cut, p_full, shape)
        groups[text] = R.from_model_space(pmeta, c["x"].reshape(x.shape))
        info[text] = {"given": c["given"], "resid": c["resid"], "cut": cut,
                      "given_fraction": c["given"] / flat.shape[1],
                      "latent": c["latent"]}
        log(f"  {text:22} задано {c['given']:6} чисел ({100 * c['given'] / flat.shape[1]:5.1f} %)"
            + (f", невязка достройки {100 * c['resid']:.2f} %" if c["resid"] is not None else "")
            + f", {time.time() - t0:.0f} с")

    jobs = {f"{g}|{i}": v[i] for g, v in groups.items() for i in range(n_clips)}
    names = list(jobs)
    cond = torch.as_tensor(np.stack([jobs[n] for n in names]), device=dev)
    mask = torch.ones(len(names), len(DEEP), device=dev)
    vids = []
    with torch.no_grad():
        for i in range(0, len(names), 8):
            gg = torch.Generator(device=dev).manual_seed(1000 + seed)  # один z у 13B везде
            vids.append(G.sample(gen, cond[i:i + 8], mask[i:i + 8], steps=20, generator=gg).cpu().numpy())
            log(f"    13B {min(i + 8, len(names))}/{len(names)}, {time.time() - t0:.0f} с")
    videos = np.concatenate(vids).astype(np.float32)
    target_raw = np.stack([R.from_maps(R.unscale(jobs[n][None], mean, std), d.layout, len(d.cells_all))[0]
                           for n in names])
    rts = round_trip(net, videos, target_raw, d.cells_all, d.type_of, DEEP, dt, t_pre, margin, (0, frames), var_ref)
    log(f"{len(videos)} видео отрисовано и прогнано через мозг за {time.time() - t0:.0f} с")

    nb = np.asarray(L.neighbour_index(721))
    out = {"pca": str(pca_path), "cuts": list(cuts), "clip_idx": clip_idx.tolist(),
           "n_clips": n_clips, "dims": int(flat.shape[1]), "info": info, "groups": {}}
    for g in groups:
        sel = [i for i, n in enumerate(names) if n.startswith(g + "|")]
        s = describe([videos[i] for i in sel], nb)
        r = float(np.mean([pixcorr_per_frame(videos[i], raw[j]).mean() for j, i in enumerate(sel)]))
        near = [nearest(videos[i], bank_all) for i in sel]
        out["groups"][g] = {"r_to_raw": r, "gate": float(np.median(rts[sel])),
                            "frac_flat": s["frac_flat"]["mean"], "kurtosis": s["grad_kurtosis"]["mean"],
                            "neigh_r": s["neigh_r"]["mean"], "sd": s["sd"]["mean"],
                            "nearest_r": float(np.mean([r_ for _, r_ in near])),
                            "nearest_idx": [int(j) for j, _ in near],
                            "given": info[g]["given"]}
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
    p.add_argument("--tag", default="cut19")
    p.add_argument("--cuts", default=",".join(DEFAULT), help="через запятую: pca=512, types=T4+complete, …")
    p.add_argument("--clips", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    pdir = Path(a.pairs13)
    manifest = json.loads((pdir / "manifest.json").read_text(encoding="utf-8"))
    columns = json.loads((pdir / "columns.json").read_text(encoding="utf-8"))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    r = run(a.model, Path(a.pca), Path(a.gen), manifest, columns, Path(a.corpus),
            cuts=tuple(x.strip() for x in a.cuts.split(",") if x.strip()), n_clips=a.clips,
            frames=g.get("frames", 40), margin=g.get("margin", 5), dt=g.get("dt", 0.02),
            t_pre=g.get("t_pre", 1.0), seed=a.seed)
    (out / f"{a.tag}.json").write_text(json.dumps(r["summary"], indent=1, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(out / f"{a.tag}.npz", **r["arrays"])
    S = r["summary"]
    print("\n{:24} {:>9} {:>7} {:>11} {:>8} {:>9} {:>10}".format(
        "срез", "задано", "от D", "r к сырому", "ворота", "ровного", "ближайшее"))
    for k, v in S["groups"].items():
        print("{:24} {:9} {:6.1f}% {:11.3f} {:8.4f} {:8.1f}% {:10.3f}".format(
            k, v["given"], 100 * v["given"] / S["dims"], v["r_to_raw"], v["gate"],
            100 * v["frac_flat"], v["nearest_r"]))
    w = S["raw"]
    print("{:24} {:>9} {:>7} {:>11} {:>8} {:8.1f}%".format(
        "сырое видео корпуса", "—", "—", "—", "—", 100 * w["frac_flat"]))
    res = {k: v["resid"] for k, v in S["info"].items() if v.get("resid") is not None}
    if res:
        print("\nневязка достройки на заданной части (вырожден ли подбор):")
        for k, v in res.items():
            print(f"  {k:24} {100 * v:.2f} %")
    print(f"\nwrote {out / a.tag}.json / .npz  ({S['seconds']:.0f} s, $0)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
