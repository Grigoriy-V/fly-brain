r"""22.0: что ещё можно убрать ВНУТРИ половины T4.

    python -m flydream.generate.inside22          # локально, CPU, $0, ~5 мин

T5 убран и возвращается сам (`reports/2026-09-21_state_restoration.md` § 9).
Внутри оставшейся половины три оси, и каждую можно резать отдельно:

* **направления** — четыре канала T4a…T4d; маска 13B умеет объявить, что
  канала нет, и это его штатный вход;
* **время** — 16 коэффициентов DCT; маски для времени нет, поэтому обнуление
  верхних коэффициентов означает «медленное состояние», а не ложь про
  присутствие;
* **пространство** — 721 колонка кольцами от центра; маски тоже нет,
  обнуление означает «серая периферия», а не меньший глаз.

Арма задаётся как `типы/DCT/кольца`, типы через `+`:

    T4/16/15        все четыре направления, всё время, всё поле — опора
    T4a+T4c/16/15   одна ось движения из двух
    T4a/8/10        одно направление, половина времени, центр поля

Меряются две вещи сразу: держится ли **видео** (r к сырому клипу и доля
ровного поля) и возвращает ли цепочка **состояние** (состояние′ против
настоящего полного, как в 21в). Ворота не печатаются: для среза они меряют
расстояние до заказа, которого не бывает, и судьёй не являются.
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
from flydream.decode.hexraster import axial_coords, neighbour_index, ring_of
from flydream.generate import gen13b as G
from flydream.generate import learned as L
from flydream.generate import prior17 as R
from flydream.generate.back21 import per_type
from flydream.generate.edges18 import describe
from flydream.generate.gen13b import DEEP
from flydream.generate.invert import device_of, load_network, pixcorr_per_frame
from flydream.generate.pairs13 import simulate_states
from flydream.generate.prompts14 import Deep
from flydream.generate.roundtrip13 import build_states
from flydream.generate.vaeval18 import nearest

LATTICES = ("all", "half_rows", "half_rand", "third", "quarter")


def fill_from_neighbours(x: torch.Tensor, keep: torch.Tensor, rounds: int = 4) -> torch.Tensor:
    """Пустые колонки заполняются средним по имеющимся соседям на решётке.

    Маски по колонкам у 13B нет — объявить «этого гекса не дано» нечем, поэтому
    прорежённое состояние иначе подаётся как «здесь среднее корпуса», то есть
    ложь того же рода, что обнулённый T5. Достройка по соседям опирается на
    измеренное: корреляция соседних колонок 0,876 (21б)."""
    nb = torch.as_tensor(np.asarray(neighbour_index(x.shape[-1])), device=x.device)
    out = x.clone()
    have = keep.clone()
    for _ in range(rounds):
        if bool(have.all()):
            break
        idx = nb.clamp_min(0)                                          # (n, 6)
        ok = (nb >= 0) & have[idx]                                     # сосед есть и уже заполнен
        vals = out[..., idx] * ok.to(out.dtype)                        # (..., n, 6)
        cnt = ok.sum(1).to(out.dtype)
        filled = vals.sum(-1) / cnt.clamp_min(1)
        take = (~have) & (cnt > 0)
        out = torch.where(take, filled, out)
        have = have | take
    return out

DEFAULT = ("T4/16/15/all",
           # все комбинации направлений: четыре по одному, шесть пар, четыре тройки
           "T4a/16/15/all", "T4b/16/15/all", "T4c/16/15/all", "T4d/16/15/all",
           "T4a+T4b/16/15/all", "T4a+T4c/16/15/all", "T4a+T4d/16/15/all",
           "T4b+T4c/16/15/all", "T4b+T4d/16/15/all", "T4c+T4d/16/15/all",
           "T4a+T4b+T4c/16/15/all", "T4a+T4b+T4d/16/15/all", "T4a+T4c+T4d/16/15/all",
           "T4b+T4c+T4d/16/15/all",
           # решётка прорежена равномерно, поле зрения целое
           "T4/16/15/half_rows", "T4/16/15/half_rand", "T4/16/15/third", "T4/16/15/quarter",
           "T4a+T4b+T4c/16/15/half_rand")


def lattice_mask(kind: str, n: int = 721, seed: int = 0) -> np.ndarray:
    """(n,) bool: какие колонки остаются при равномерном прореживании решётки.

    Обрезка колец делает картинку МЕНЬШЕ; прореживание оставляет всё поле
    зрения и снижает разрешение. Гексагональная решётка треугольная, поэтому
    ровной двухцветной раскраски у неё нет: `half_rows` прореживает ряды (то
    есть анизотропно), `half_rand` берёт случайную половину, а изотропные
    подрешётки дают 1/3 (√3 × √3) и 1/4 (шаг решётки вдвое)."""
    a = axial_coords(n)
    u, v = a[:, 0], a[:, 1]
    if kind == "all":
        return np.ones(n, bool)
    if kind == "half_rows":
        return (v % 2) == 0
    if kind == "half_rand":
        keep = np.zeros(n, bool)
        keep[np.random.default_rng(seed).permutation(n)[: n // 2]] = True
        return keep
    if kind == "third":
        return ((u + 2 * v) % 3) == 0
    if kind == "quarter":
        return ((u % 2) == 0) & ((v % 2) == 0)
    raise ValueError(f"неизвестное прореживание {kind!r}, есть {LATTICES}")


def parse_arm(text: str, n_dct: int = 16, n_rings: int = 15) -> dict:
    """`"T4a+T4c/8/10/third"` -> {"types": [...], "dct": 8, "rings": 10, "lattice": "third"}."""
    parts = [p.strip() for p in str(text).split("/")]
    if not 1 <= len(parts) <= 4:
        raise ValueError(f"не разобрать арму {text!r}: нужно типы/DCT/кольца/решётка")
    names = []
    for token in parts[0].split("+"):
        token = token.strip()
        got = [t for t in DEEP if t.startswith(token)]
        if not got:
            raise ValueError(f"нет типов, начинающихся на {token!r}, есть {list(DEEP)}")
        names += got
    dct = int(parts[1]) if len(parts) > 1 and parts[1] else n_dct
    rings = int(parts[2]) if len(parts) > 2 and parts[2] else n_rings
    if not 1 <= dct <= n_dct:
        raise ValueError(f"DCT {dct} вне 1..{n_dct}")
    if not 0 <= rings <= n_rings:
        raise ValueError(f"кольца {rings} вне 0..{n_rings}")
    lat = parts[3] if len(parts) > 3 and parts[3] else "all"
    fill = lat.endswith("+fill")
    lat = lat[: -len("+fill")] if fill else lat
    if lat not in LATTICES:
        raise ValueError(f"неизвестное прореживание {lat!r}, есть {LATTICES}")
    return {"types": sorted(set(names), key=DEEP.index), "dct": dct, "rings": rings,
            "lattice": lat, "fill": fill, "name": text}


def apply_arm(x: torch.Tensor, arm: dict, cols: np.ndarray) -> tuple:
    """(N, DCT, K, 721) -> срезанное состояние и число заданных чисел.

    Одной булевой маской, а не цепочкой индексов: `out[:, :k][:, :, ch]` — это
    копия, и присваивание в неё ничего не записывает в `out`."""
    ch = torch.zeros(x.shape[2], dtype=torch.bool, device=x.device)
    ch[[DEEP.index(t) for t in arm["types"]]] = True
    col = torch.as_tensor(np.asarray((cols <= arm["rings"]) & lattice_mask(arm.get("lattice", "all"))),
                          device=x.device)
    tim = torch.zeros(x.shape[1], dtype=torch.bool, device=x.device)
    tim[:arm["dct"]] = True
    keep = tim[None, :, None, None] & ch[None, None, :, None] & col[None, None, None, :]
    given = int(tim.sum()) * int(ch.sum()) * int(col.sum())
    out = torch.where(keep, x, torch.zeros_like(x))
    if arm.get("fill"):
        out = fill_from_neighbours(out, col)
        out = torch.where(tim[None, :, None, None] & ch[None, None, :, None], out, torch.zeros_like(out))
    return out, given


def run(model: str, gen_ckpt: Path, manifest: dict, columns: dict, corpus: Path, *, arms=DEFAULT,
        n_clips: int = 6, frames: int = 40, margin: int = 5, dt: float = 0.02, t_pre: float = 1.0,
        seed: int = 0, nearest_to_bank: bool = True, log=print) -> dict:
    t0 = time.time()
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    net = load_network(model); dev = device_of(net)
    _, index = P13.type_index(net.connectome)
    d = Deep(manifest, columns)
    gen, gmeta = G.load(gen_ckpt, dev)
    mean = np.array(gmeta["mean"], np.float32); std = np.array(gmeta["std"], np.float32)
    cols = np.asarray(ring_of(721))

    built = build_states(net, index, frames=frames, margin=margin, dt=dt, t_pre=t_pre, seed=seed)
    sa = next(s for s in built if s["name"] == "clip_A")
    w0, w1 = sa["window"]
    ta = sa["target"][:, w0:w1, :][:, :, d.cells_all].cpu().numpy().astype(np.float32)
    var_ref = {t: float(ta[0][:, d.pos[t]].var()) + 1e-6 for t in DEEP}

    cz = np.load(Path(corpus) / "videos.npz")
    cm = json.loads((Path(corpus) / "pairs_manifest.json").read_text(encoding="utf-8"))
    bank_all = np.asarray(cz["videos"][:, :frames], np.float16) if nearest_to_bank else None
    clip_idx = rng.choice(np.asarray(cm["split"]["test"]), n_clips, replace=False)
    raw = np.asarray(cz["videos"][clip_idx][:, :frames], np.float32)
    st_real = simulate_states(net, np.asarray(cz["videos"], np.float16)[clip_idx], d.cells_all,
                              dt, t_pre, 8).astype(np.float32)[:, :frames]
    maps = L.to_maps(st_real.astype(np.float16), d.layout, len(DEEP))[:, :frames].astype(np.float32)
    real = (maps - mean[None, None, :, None]) / std[None, None, :, None]

    # DCT-пространство нужно только чтобы резать время; коэффициенты берутся из 13B-метаданных
    dct = R.dct_matrix(frames, 16)
    x = torch.as_tensor(np.einsum("kt,btcn->bkcn", dct, real), device=dev)
    specs = [parse_arm(a) for a in arms]
    log(f"{n_clips} настоящих состояний, клипы {clip_idx.tolist()}, арм {len(specs)}, "
        f"{time.time() - t0:.0f} с")

    groups, info = {"полное состояние": real}, {"полное состояние": {"given": int(real[0].size), "arm": None}}
    for spec in specs:
        cut, given = apply_arm(x, spec, cols)
        groups[spec["name"]] = np.einsum("kt,bkcn->btcn", dct, cut.cpu().numpy())
        info[spec["name"]] = {"given": given, "arm": spec}
        log(f"  {spec['name']:24} {len(spec['types'])} типов x {spec['dct']} DCT x "
            f"{given // max(len(spec['types']) * spec['dct'], 1)} колонок = {given} чисел, "
            f"{time.time() - t0:.0f} с")

    names = [f"{g}|{i}" for g in groups for i in range(n_clips)]
    cond = torch.as_tensor(np.stack([groups[n.rsplit("|", 1)[0]][int(n.rsplit("|", 1)[1])]
                                     for n in names]), device=dev)
    bits = {g: np.array([1.0 if (info[g]["arm"] is None or t in info[g]["arm"]["types"]) else 0.0
                         for t in DEEP], np.float32) for g in groups}
    mask = torch.as_tensor(np.stack([bits[n.rsplit("|", 1)[0]] for n in names]), device=dev)
    vids = []
    with torch.no_grad():
        for i in range(0, len(names), n_clips):                       # пачка = группа (ISS-0009)
            gg = torch.Generator(device=dev).manual_seed(1000 + seed)
            vids.append(G.sample(gen, cond[i:i + n_clips], mask[i:i + n_clips],
                                 steps=20, generator=gg).cpu().numpy())
    videos = np.concatenate(vids).astype(np.float32)
    log(f"{len(videos)} видео отрисовано, {time.time() - t0:.0f} с")

    vlong = np.concatenate([videos, np.repeat(videos[:, -1:], margin, 1)], 1) if margin else videos
    st_back = simulate_states(net, vlong.astype(np.float16), d.cells_all, dt, t_pre, 16).astype(np.float32)
    st_back = st_back[:, w0:w1][:, :frames] if st_back.shape[1] >= w1 else st_back[:, :frames]
    log(f"{len(videos)} видео прогнано через мозг, {time.time() - t0:.0f} с")

    nb = np.asarray(L.neighbour_index(721))
    out = {"arms": list(arms), "clip_idx": clip_idx.tolist(), "n_clips": n_clips,
           "dims": int(real[0].size), "groups": {}}
    for g in groups:
        sel = [i for i, n in enumerate(names) if n.startswith(g + "|")]
        s = describe([videos[i] for i in sel], nb)
        r = float(np.mean([pixcorr_per_frame(videos[i], raw[j]).mean() for j, i in enumerate(sel)]))
        near = [nearest(videos[i], bank_all) for i in sel] if bank_all is not None else [(0, float("nan"))]
        back = per_type(st_back[sel], st_real, d.type_of, var_ref)
        out["groups"][g] = {
            "given": info[g]["given"], "r_to_raw": r, "frac_flat": s["frac_flat"]["mean"],
            "sd": s["sd"]["mean"], "nearest_r": float(np.mean([r_ for _, r_ in near])),
            "state_prime_vs_real": back["среднее"],
            "state_prime_T4": {t: back[t] for t in DEEP if t.startswith("T4")},
            "arm": info[g]["arm"]}
        v = out["groups"][g]
        log(f"  {g:20} {v['given']:6} чисел, r {v['r_to_raw']:.3f}, ровного "
            f"{100 * v['frac_flat']:.1f} %, состояние′ {v['state_prime_vs_real']['err']:.4f} "
            f"(r {v['state_prime_vs_real']['r']:.3f})")
    rw = describe(list(raw), nb)
    out["raw"] = {"frac_flat": rw["frac_flat"]["mean"], "sd": rw["sd"]["mean"]}
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
    p.add_argument("--gen", default=str(ROOT / "data" / "gen13b" / "sit.pt"))
    p.add_argument("--pairs13", default=str(ROOT / "data" / "pairs13"))
    p.add_argument("--corpus", default=str(ROOT / "data" / "corpus18"))
    p.add_argument("--out", default=str(ROOT / "data" / "prior19"))
    p.add_argument("--tag", default="inside22")
    p.add_argument("--arms", default=",".join(DEFAULT))
    p.add_argument("--clips", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-nearest", action="store_true",
                   help="не искать ближайшее из 15 514: на НАСТОЯЩИХ состояниях это не про новизну")
    a = p.parse_args(argv)
    pdir = Path(a.pairs13)
    manifest = json.loads((pdir / "manifest.json").read_text(encoding="utf-8"))
    columns = json.loads((pdir / "columns.json").read_text(encoding="utf-8"))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    r = run(a.model, Path(a.gen), manifest, columns, Path(a.corpus),
            arms=tuple(x.strip() for x in a.arms.split(",") if x.strip()), n_clips=a.clips,
            frames=g.get("frames", 40), margin=g.get("margin", 5), dt=g.get("dt", 0.02),
            t_pre=g.get("t_pre", 1.0), seed=a.seed, nearest_to_bank=not a.no_nearest)
    (out / f"{a.tag}.json").write_text(json.dumps(r["summary"], indent=1, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(out / f"{a.tag}.npz", **r["arrays"])
    S = r["summary"]
    print("\n{:20} {:>8} {:>6} {:>11} {:>9} {:>14} {:>10}".format(
        "арма", "задано", "от D", "r к сырому", "ровного", "состояние′", "ближайшее"))
    for k, v in S["groups"].items():
        print("{:20} {:8} {:5.1f}% {:11.3f} {:8.1f}% {:9.4f}/{:.3f} {:10.3f}".format(
            k, v["given"], 100 * v["given"] / S["dims"], v["r_to_raw"], 100 * v["frac_flat"],
            v["state_prime_vs_real"]["err"], v["state_prime_vs_real"]["r"], v["nearest_r"]))
    print("{:20} {:>8} {:>6} {:>11} {:8.1f}%".format(
        "сырое видео корпуса", "—", "—", "—", 100 * S["raw"]["frac_flat"]))
    print(f"\nwrote {out / a.tag}.json / .npz  ({S['seconds']:.0f} s, $0)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
