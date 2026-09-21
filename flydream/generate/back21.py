r"""21в: что мозг вычитывает из видео, сделанного из урезанного состояния.

    python -m flydream.generate.back21            # локально, CPU, $0, ~3 мин

Вопрос человека (2026-09-21): «в чём разница между стартовым состоянием из
обучающего клипа, урезанным, и новым состоянием из сгенерированного от
урезанного — на глаз разница минимальная, так откуда 0,38 ворота?»

Ворота шага 21а меряли расстояние от состояния′ до **заказанного** (то есть
урезанного) состояния. Здесь меряется то, что имеет смысл:

* **по каждому типу отдельно** — где именно сидит ошибка;
* **против НАСТОЯЩЕГО полного состояния** клипа — вернула ли цепочка то, что
  мы выбросили;
* **положение в обучающем латенте** — лежит ли состояние′ там, где живут
  настоящие состояния, и лежит ли там заказанное.

Контроль — арма без среза: цепочка состояние → 13B → видео → мозг имеет свою
собственную ошибку, и без неё ни одно число из остальных арм не читается.

    настоящее состояние ──режем──> заказанное ──13B──> видео ──мозг──> состояние′
            │                          │                                  │
            └──────── r, ошибка ───────┴────────── r, ошибка ─────────────┘
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
from flydream.generate import cut19 as C
from flydream.generate import gen13b as G
from flydream.generate import learned as L
from flydream.generate import pca19 as P
from flydream.generate import prior17 as R
from flydream.generate.gen13b import DEEP
from flydream.generate.invert import device_of, load_network, pixcorr_per_frame
from flydream.generate.pairs13 import simulate_states
from flydream.generate.prompts14 import Deep
from flydream.generate.roundtrip13 import build_states

ARMS = ("полное", "types=T4", "types=T4+complete", "rings=10", "pca=128")


def per_type(a: np.ndarray, b: np.ndarray, type_of: np.ndarray, var_ref: dict) -> dict:
    """Нормированная ошибка и корреляция по каждому типу между двумя состояниями."""
    out = {}
    for t in DEEP:
        sel = np.where(type_of == t)[0]
        x, y = a[:, :, sel], b[:, :, sel]
        err = float(((x - y) ** 2).mean() / var_ref[t])
        xf, yf = x.reshape(-1), y.reshape(-1)
        xc, yc = xf - xf.mean(), yf - yf.mean()
        r = float((xc @ yc) / max(np.linalg.norm(xc) * np.linalg.norm(yc), 1e-12))
        out[t] = {"err": err, "r": r}
    out["среднее"] = {"err": float(np.mean([v["err"] for k, v in out.items() if k in DEEP])),
                      "r": float(np.mean([v["r"] for k, v in out.items() if k in DEEP]))}
    return out


def run(model: str, pca_path: Path, latent_path: Path, gen_ckpt: Path, manifest: dict, columns: dict,
        corpus: Path, *, arms=ARMS, n_clips: int = 6, frames: int = 40, margin: int = 5,
        dt: float = 0.02, t_pre: float = 1.0, seed: int = 0, show_type: str = "T5c",
        log=print) -> dict:
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
    zf = np.load(latent_path)
    z_test = torch.as_tensor(np.asarray(zf["z_test"], np.float32), device=dev)
    log(f"базис {p_full['dims']} x {p_full['k']}, арм {len(arms)}, {time.time() - t0:.0f} с")

    built = build_states(net, index, frames=frames, margin=margin, dt=dt, t_pre=t_pre, seed=seed)
    sa = next(s for s in built if s["name"] == "clip_A")
    w0, w1 = sa["window"]
    ta = sa["target"][:, w0:w1, :][:, :, d.cells_all].cpu().numpy().astype(np.float32)
    var_ref = {t: float(ta[0][:, d.pos[t]].var()) + 1e-6 for t in DEEP}

    cz = np.load(Path(corpus) / "videos.npz")
    cm = json.loads((Path(corpus) / "pairs_manifest.json").read_text(encoding="utf-8"))
    clip_idx = rng.choice(np.asarray(cm["split"]["test"]), n_clips, replace=False)
    raw = np.asarray(cz["videos"][clip_idx][:, :frames], np.float32)
    st_real = simulate_states(net, np.asarray(cz["videos"], np.float16)[clip_idx], d.cells_all,
                              dt, t_pre, 8).astype(np.float32)[:, :frames]
    maps = L.to_maps(st_real.astype(np.float16), d.layout, len(DEEP))[:, :frames].astype(np.float32)
    real = (maps - mean[None, None, :, None]) / std[None, None, :, None]
    x = R.to_model_space(pmeta, real, dev)
    shape = tuple(x.shape[1:])
    flat = x.reshape(len(x), -1)
    log(f"{n_clips} настоящих состояний {tuple(st_real.shape)}, клипы {clip_idx.tolist()}, "
        f"{time.time() - t0:.0f} с")

    def to_raw(cond_maps: np.ndarray) -> np.ndarray:
        """(N, T, 8, 721) в единицах 13B -> (N, T, cells) в единицах мозга."""
        return np.stack([R.from_maps(R.unscale(cond_maps[i][None], mean, std), d.layout, len(d.cells_all))[0]
                         for i in range(len(cond_maps))])

    def latent_of(raw_state: np.ndarray) -> torch.Tensor:
        """(N, T, cells) из мозга -> отбелённые координаты PCA."""
        m = L.to_maps(raw_state.astype(np.float16), d.layout, len(DEEP))[:, :frames].astype(np.float32)
        zs = (m - mean[None, None, :, None]) / std[None, None, :, None]
        return P.encode(R.to_model_space(pmeta, zs, dev).reshape(len(zs), -1), p_full)

    # --- заказанные состояния по армам --------------------------------------
    asked, info = {}, {}
    for arm in arms:
        if arm == "полное":
            asked[arm] = real
            info[arm] = {"given": int(flat.shape[1]), "resid": None}
            continue
        cut = C.parse_cut(arm)
        c = C.apply_cut(flat, cut, p_full, shape)
        asked[arm] = R.from_model_space(pmeta, c["x"].reshape(x.shape))
        info[arm] = {"given": c["given"], "resid": c["resid"]}
        log(f"  {arm:20} задано {c['given']} чисел, {time.time() - t0:.0f} с")

    # --- 13B на все армы сразу ------------------------------------------------
    names = [f"{a}|{i}" for a in arms for i in range(n_clips)]
    jobs = {f"{a}|{i}": asked[a][i] for a in arms for i in range(n_clips)}
    cond = torch.as_tensor(np.stack([jobs[n] for n in names]), device=dev)
    mask = torch.ones(len(names), len(DEEP), device=dev)
    vids = []
    with torch.no_grad():
        # Пачка РОВНО в группу: 13B берёт j-й шум для j-го элемента пачки, поэтому
        # «один z на клип во всех армах» верно только когда границы пачки совпадают
        # с границами группы. При пачке 8 и группе 6 они расходятся (ISS-0009).
        for i in range(0, len(names), n_clips):
            gg = torch.Generator(device=dev).manual_seed(1000 + seed)
            vids.append(G.sample(gen, cond[i:i + n_clips], mask[i:i + n_clips],
                                 steps=20, generator=gg).cpu().numpy())
    videos = np.concatenate(vids).astype(np.float32)
    log(f"{len(videos)} видео отрисовано, {time.time() - t0:.0f} с")

    # --- обратно через мозг ---------------------------------------------------
    vlong = np.concatenate([videos, np.repeat(videos[:, -1:], margin, 1)], 1) if margin else videos
    st_back = simulate_states(net, vlong.astype(np.float16), d.cells_all, dt, t_pre, 16).astype(np.float32)
    st_back = st_back[:, w0:w1][:, :frames] if st_back.shape[1] >= w1 else st_back[:, :frames]
    log(f"{len(videos)} видео прогнано через мозг, {time.time() - t0:.0f} с")

    out = {"pca": str(pca_path), "arms": list(arms), "clip_idx": clip_idx.tolist(), "n_clips": n_clips,
           "frames": frames, "info": info, "var_ref": var_ref,
           "latent_test": P.geometry(z_test), "по армам": {}}
    for a in arms:
        sel = [i for i, n in enumerate(names) if n.startswith(a + "|")]
        back = st_back[sel]
        ask_raw = to_raw(asked[a])
        z_back, z_ask = latent_of(back), latent_of(ask_raw)
        z_real = latent_of(st_real)
        rec = {
            "state_prime_vs_asked": per_type(back, ask_raw, d.type_of, var_ref),
            "state_prime_vs_real": per_type(back, st_real, d.type_of, var_ref),
            "asked_vs_real": per_type(ask_raw, st_real, d.type_of, var_ref),
            "latent": {"state_prime": P.geometry(z_back), "asked": P.geometry(z_ask)},
            "latent_distance": {
                "state_prime_to_real": float((z_back - z_real).norm(dim=1).mean()),
                "asked_to_real": float((z_ask - z_real).norm(dim=1).mean()),
                "real_norm": float(z_real.norm(dim=1).mean())},
            "r_video_to_raw": float(np.mean([pixcorr_per_frame(videos[i], raw[j]).mean()
                                             for j, i in enumerate(sel)])),
            "given": info[a]["given"]}
        out["по армам"][a] = rec
        p_, r_ = rec["state_prime_vs_asked"]["среднее"], rec["state_prime_vs_real"]["среднее"]
        log(f"  {a:20} ворота к заказу {p_['err']:.4f}, к настоящему {r_['err']:.4f}, "
            f"r к настоящему {r_['r']:.3f}, радиус латента {rec['latent']['state_prime']['radius_mean']:.1f}")
    out["latent_real"] = P.geometry(latent_of(st_real))
    out["seconds"] = round(time.time() - t0, 1)
    arrays = {f"video__{n}": videos[i] for i, n in enumerate(names)}
    arrays.update({f"raw__{i}": raw[i] for i in range(n_clips)})

    # Карты одного типа для фигуры: настоящая, заказанная и вычитанная из видео.
    # Это и есть предмет находки — канал, который выбросили, и то, чем он вернулся.
    ch = DEEP.index(show_type)
    out["show_type"] = show_type

    def zmaps(raw_state: np.ndarray) -> np.ndarray:
        m = L.to_maps(raw_state.astype(np.float16), d.layout, len(DEEP))[:, :frames].astype(np.float32)
        return ((m - mean[None, None, :, None]) / std[None, None, :, None])[:, :, ch]

    arrays.update({f"mapreal__{i}": zmaps(st_real)[i] for i in range(n_clips)})
    for a in arms:
        sel = [i for i, n in enumerate(names) if n.startswith(a + "|")]
        mb, ma = zmaps(st_back[sel]), asked[a][:, :, ch]
        arrays.update({f"mapback__{a}|{i}": mb[i] for i in range(n_clips)})
        arrays.update({f"mapask__{a}|{i}": ma[i] for i in range(n_clips)})
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
    p.add_argument("--latent", default=str(ROOT / "data" / "prior19" / "pca2048_latent.npz"))
    p.add_argument("--gen", default=str(ROOT / "data" / "gen13b" / "sit.pt"))
    p.add_argument("--pairs13", default=str(ROOT / "data" / "pairs13"))
    p.add_argument("--corpus", default=str(ROOT / "data" / "corpus18"))
    p.add_argument("--out", default=str(ROOT / "data" / "prior19"))
    p.add_argument("--tag", default="back21")
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--clips", type=int, default=6)
    p.add_argument("--show-type", default="T5c", help="какой тип сохранить картами для фигуры")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    pdir = Path(a.pairs13)
    manifest = json.loads((pdir / "manifest.json").read_text(encoding="utf-8"))
    columns = json.loads((pdir / "columns.json").read_text(encoding="utf-8"))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    r = run(a.model, Path(a.pca), Path(a.latent), Path(a.gen), manifest, columns, Path(a.corpus),
            arms=tuple(x.strip() for x in a.arms.split(",") if x.strip()), n_clips=a.clips,
            frames=g.get("frames", 40), margin=g.get("margin", 5), dt=g.get("dt", 0.02),
            t_pre=g.get("t_pre", 1.0), seed=a.seed, show_type=a.show_type)
    (out / f"{a.tag}.json").write_text(json.dumps(r["summary"], indent=1, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(out / f"{a.tag}.npz", **r["arrays"])
    S = r["summary"]
    print("\n{:20} {:>8} {:>12} {:>12} {:>10} {:>10}".format(
        "арма", "задано", "к заказу", "к настоящему", "r к наст.", "r видео"))
    for a_, v in S["по армам"].items():
        print("{:20} {:8} {:12.4f} {:12.4f} {:10.3f} {:10.3f}".format(
            a_, v["given"], v["state_prime_vs_asked"]["среднее"]["err"],
            v["state_prime_vs_real"]["среднее"]["err"], v["state_prime_vs_real"]["среднее"]["r"],
            v["r_video_to_raw"]))
    print("\nпо типам — ошибка состояния′ против ЗАКАЗАННОГО / против НАСТОЯЩЕГО:")
    print("{:20} {}".format("арма", " ".join(f"{t:>13}" for t in DEEP)))
    for a_, v in S["по армам"].items():
        row = " ".join(f"{v['state_prime_vs_asked'][t]['err']:6.3f}/"
                       f"{v['state_prime_vs_real'][t]['err']:6.3f}" for t in DEEP)
        print("{:20} {}".format(a_, row))
    print("\nчто сделал срез (заказанное против настоящего) и что вернула цепочка:")
    for a_, v in S["по армам"].items():
        print("  {:20} срез: ошибка {:7.4f}, r {:6.3f}   ->   после цепочки: ошибка {:7.4f}, r {:6.3f}".format(
            a_, v["asked_vs_real"]["среднее"]["err"], v["asked_vs_real"]["среднее"]["r"],
            v["state_prime_vs_real"]["среднее"]["err"], v["state_prime_vs_real"]["среднее"]["r"]))
    lr = S["latent_real"]
    print(f"\nлатент: настоящие состояния этих клипов радиус {lr['radius_mean']:.1f}, "
          f"отложенные все {S['latent_test']['radius_mean']:.1f} ± {S['latent_test']['radius_sd']:.2f}")
    print("{:20} {:>10} {:>10} {:>14} {:>14}".format(
        "арма", "рад. зак.", "рад. сост'", "|зак.−наст.|", "|сост'−наст.|"))
    for a_, v in S["по армам"].items():
        print("{:20} {:10.1f} {:10.1f} {:14.1f} {:14.1f}".format(
            a_, v["latent"]["asked"]["radius_mean"], v["latent"]["state_prime"]["radius_mean"],
            v["latent_distance"]["asked_to_real"], v["latent_distance"]["state_prime_to_real"]))
    print(f"\nwrote {out / a.tag}.json / .npz  ({S['seconds']:.0f} s, $0)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
