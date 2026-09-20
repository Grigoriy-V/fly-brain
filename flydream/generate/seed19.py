r"""19.3: тест сидов над двухступенчатой цепочкой.

    python -m flydream.generate.seed19            # локально, CPU, $0

Вопрос человека с самого начала (`reports/2026-09-20_the_seed_problem.md` § 1):
можно ли задать сид и получить видео, и даёт ли произвольный розыгрыш новое
осмысленное видео. В пункте 17 ответ был нет, и причина была измерена:
прообразы настоящих состояний лежали на радиусе 252 при оболочке 303,8 —
на 73 σ внутрь, туда, куда розыгрыш не попадает никогда.

Здесь то же самое меряется над цепочкой из двух ступеней:

    ε ~ N(0, I) → поток (19.2) → латент 2 048 → PCA⁻¹ (19.0) → состояние → 13B → видео

Три измерения, и первое — главное:

**A. Геометрия прообразов, на всех 1 246 отложенных.** Латент отложенного
клипа обращается потоком назад. Радиус ε\* против √2048 = 45,25 при разбросе
0,707 — прямой наследник числа 73 σ. Считается на 20 и на 100 шагах Эйлера,
потому что 18.23 показал: 20 шагов радиус завышают.

**B. Сид, заданный клипом.** ε\* того же клипа вперёд: возвращает ли он свой
клип, и насколько это лучше, чем чужой.

**C. Свежий розыгрыш.** ε ~ N(0, I) через ту же цепочку: против сырого видео
корпуса и против **ближайшего обучающего видео из всех 15 514** — без второго
числа «новое видео» заявлять нельзя (`AGENTS.md`).
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
from flydream.generate import fix19 as F
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


def geometry_of(eps: torch.Tensor) -> dict:
    """Геометрия прообразов в тех же полях, что у 19.0."""
    return P.geometry(eps.reshape(len(eps), -1))


@torch.no_grad()
def preimages(flow, z: torch.Tensor, *, steps: int, tokens: int, chunk: int = 256, log=print) -> torch.Tensor:
    """(N, k) латента -> (N, k) прообразов, кусками, чтобы не разложить память."""
    out = []
    t0 = time.time()
    for i in range(0, len(z), chunk):
        x = P.as_tokens(z[i:i + chunk], tokens=tokens)
        out.append(R.invert(flow, x, steps=steps).reshape(len(x), -1))
        if (i // chunk) % 2 == 0:
            log(f"    {i + len(x)}/{len(z)}, {time.time() - t0:.0f} с")
    return torch.cat(out)


def run(model: str, flow_ckpt: Path, pca_path: Path, latent_path: Path, gen_ckpt: Path,
        manifest: dict, columns: dict, corpus: Path, *, n_clips: int = 6, frames: int = 40,
        margin: int = 5, dt: float = 0.02, t_pre: float = 1.0, seed: int = 0,
        invert_steps=(20, 100), sample_steps: int = 20, draw_scale: float = 0.0,
        controls: bool = False, invert_n: int = 0, fixes=(), fix_n: int = 256, log=print) -> dict:
    t0 = time.time()
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    net = load_network(model); dev = device_of(net)
    _, index = P13.type_index(net.connectome)
    d = Deep(manifest, columns)
    gen, gmeta = G.load(gen_ckpt, dev)
    mean = np.array(gmeta["mean"], np.float32); std = np.array(gmeta["std"], np.float32)
    flow, fmeta = R.load(flow_ckpt, dev)
    tokens = int(fmeta.get("n", 16))
    zf = np.load(latent_path)
    pz = np.load(pca_path)
    pmeta = json.loads(str(pz["meta"]))
    pca = P.from_numpy(pz, dev)
    k = pca["k"]
    log(f"поток {fmeta['width']}x{fmeta['depth']}, {tokens} токенов; PCA {pca['dims']} -> {k}; "
        f"{time.time() - t0:.0f} с")

    # --- A. геометрия прообразов на всём отложенном сплите -------------------
    z_test = torch.as_tensor(np.asarray(zf["z_test"], np.float32), device=dev)
    idx_test = np.asarray(zf["index_test"])
    z_inv = z_test if not invert_n else z_test[:invert_n]              # сколько латентов обращать для геометрии
    out = {"flow": str(flow_ckpt), "pca": str(pca_path), "k": k, "tokens": tokens,
           "n_test": int(len(z_inv)), "latent_data": P.geometry(z_test), "preimage": {}}
    log(f"A. обращаю {len(z_inv)} отложенных латентов")
    for st in invert_steps:
        eps = preimages(flow, z_inv, steps=st, tokens=tokens, log=log)
        g = geometry_of(eps)
        back = R.integrate(flow, P.as_tokens(eps, tokens=tokens), steps=st).reshape(len(eps), -1)
        g["round_trip_error"] = float((back - z_inv).norm(dim=1).mean() / z_inv.norm(dim=1).mean())
        out["preimage"][st] = g
        log(f"  {st} шагов: ст. откл. {g['sd']:.3f}, радиус {g['radius_mean']:.1f} +- {g['radius_sd']:.2f} "
            f"при √k = {g['typical_radius']:.1f} +- 0,71, эксцесс {g['kurtosis_mean']:.2f}, "
            f"ошибка замыкания {100 * g['round_trip_error']:.1f} %")

    # --- клипы для картинки --------------------------------------------------
    built = build_states(net, index, frames=frames, margin=margin, dt=dt, t_pre=t_pre, seed=seed)
    sa = next(s for s in built if s["name"] == "clip_A")
    w0, w1 = sa["window"]
    ta = sa["target"][:, w0:w1, :][:, :, d.cells_all].cpu().numpy().astype(np.float32)
    var_ref = {t: float(ta[0][:, d.pos[t]].var()) + 1e-6 for t in DEEP}

    cz = np.load(Path(corpus) / "videos.npz")
    cm = json.loads((Path(corpus) / "pairs_manifest.json").read_text(encoding="utf-8"))
    bank_all = np.asarray(cz["videos"][:, :frames], np.float16)
    clip_idx = rng.choice(np.asarray(cm["split"]["test"]), n_clips, replace=False)
    rows = np.array([int(np.where(idx_test == c)[0][0]) for c in clip_idx])
    raw = np.asarray(cz["videos"][clip_idx][:, :frames], np.float32)
    st_real = simulate_states(net, np.asarray(cz["videos"], np.float16)[clip_idx], d.cells_all,
                              dt, t_pre, 8).astype(np.float32)
    maps = L.to_maps(st_real.astype(np.float16), d.layout, len(DEEP))[:, :frames].astype(np.float32)
    real = (maps - mean[None, None, :, None]) / std[None, None, :, None]
    x_real = R.to_model_space(pmeta, real, dev)
    z_here = P.encode(x_real.reshape(len(x_real), -1), pca)
    z_file = z_test[rows]
    agree = float((z_here - z_file).abs().max() / z_file.abs().max())
    log(f"сверка кодировщиков (локально против Modal): {100 * agree:.3f} % от размаха")

    def to_state(zl: torch.Tensor) -> np.ndarray:
        x = P.decode(zl, pca).reshape(len(zl), *x_real.shape[1:])
        return R.from_model_space(pmeta, x)

    # --- B. сид, заданный клипом; C. свежий розыгрыш -------------------------
    eps_clip = preimages(flow, z_file, steps=sample_steps, tokens=tokens, log=lambda *_: None)
    z_from_clip = R.integrate(flow, P.as_tokens(eps_clip, tokens=tokens),
                              steps=sample_steps).reshape(len(eps_clip), -1)
    gdraw = torch.Generator(device=dev).manual_seed(9000 + seed)
    e_draw = torch.randn(n_clips, k, device=dev, generator=gdraw)      # ровно N(0, I), без укорачивания
    z_draw = R.integrate(flow, P.as_tokens(e_draw, tokens=tokens), steps=sample_steps).reshape(n_clips, -1)
    scale = 1.0
    # Сколько переноса поток делает на самом деле. По шести розыгрышам этого не
    # увидеть: при n = 6 эксцесс смещён до 2,14 и идеальный гаусс читается как
    # «тяжёлых хвостов нет». Поэтому столько же розыгрышей, сколько отложенных.
    with torch.no_grad():
        e_many = torch.randn(len(z_inv), k, device=dev,
                             generator=torch.Generator(device=dev).manual_seed(4242 + seed))
        zm = torch.cat([R.integrate(flow, P.as_tokens(e_many[i:i + 256], tokens=tokens),
                                    steps=sample_steps).reshape(-1, k) for i in range(0, len(e_many), 256)])
    out["draw_geometry"] = P.geometry(z_draw)
    out["draw_many"] = P.geometry(zm) | {
        "n_draws": int(len(zm)),
        "moved": float((zm - e_many).norm(dim=1).mean() / e_many.norm(dim=1).mean()),
        "radius_needed": float(out["latent_data"]["radius_mean"]),
        "radius_start": float(P.geometry(e_many)["radius_mean"])}
    dm = out["draw_many"]
    if draw_scale:
        # Поправка масштаба, как `scaling_factor` у латентной диффузии: измеряем
        # ст. отклонение розыгрыша и приводим его к данным. `draw_scale < 0` —
        # взять поправку из самого замера, положительное число — задать руками.
        scale = float(out["latent_data"]["sd"] / dm["sd"]) if draw_scale < 0 else float(draw_scale)
        z_draw = z_draw * scale
        zm = zm * scale
        out["draw_scaled"] = P.geometry(zm) | {"scale": scale}
        log(f"масштаб розыгрыша поправлен в {scale:.3f} раза: ст. откл. {dm['sd']:.3f} -> "
            f"{out['draw_scaled']['sd']:.3f}, радиус {dm['radius_mean']:.1f} -> "
            f"{out['draw_scaled']['radius_mean']:.1f} при {out['latent_data']['radius_mean']:.1f} у данных")
    out["draw_scale"] = scale
    log(f"перенос: радиус {dm['radius_start']:.1f} -> {dm['radius_mean']:.1f}, нужно было "
        f"-> {dm['radius_needed']:.1f}; сдвиг точки {100 * dm['moved']:.1f} % нормы; "
        f"разброс {dm['radius_sd']:.2f} против {out['latent_data']['radius_sd']:.2f} у данных")
    out["clip_seed_geometry"] = geometry_of(eps_clip)
    groups = {"настоящее состояние": real,
              "только PCA (19.1)": to_state(z_file),
              "сид от клипа через поток": to_state(z_from_clip),
              "свежий розыгрыш": to_state(z_draw)}
    if fixes:
        # Пункт 20: те же самые ε, но через поправленный сэмплер. Цель — ОБУЧАЮЩИЙ
        # латент: на нём модель училась, отложенный слабее (десять новых классов).
        # Геометрия каждой поправки считается на `fix_n` розыгрышах, потому что на
        # шести её не видно (эксцесс при n = 6 смещён до 2,14, § 19.0).
        sd_train = float(torch.as_tensor(np.asarray(zf["z_train"], np.float32), device=dev).std())
        out["fix"] = {"sd_train": sd_train, "arms": {}}
        m = min(fix_n, len(e_many))
        for text in fixes:
            spec = F.parse_spec(text)
            nm = F.name_of(spec, sample_steps)
            zd = F.integrate_fixed(flow, P.as_tokens(e_draw, tokens=tokens), steps=sample_steps,
                                   spec=spec, sd_data=sd_train, tokens=tokens).reshape(n_clips, -1)
            zg = torch.cat([F.integrate_fixed(flow, P.as_tokens(e_many[i:i + 256], tokens=tokens),
                                              steps=sample_steps, spec=spec, sd_data=sd_train,
                                              tokens=tokens).reshape(-1, k) for i in range(0, m, 256)])
            out["fix"]["arms"][nm] = P.geometry(zg) | {"spec": spec, "n_draws": int(len(zg))}
            groups[f"розыгрыш + {nm}"] = to_state(zd)
            g = out["fix"]["arms"][nm]
            log(f"поправка {nm}: ст. откл. {g['sd']:.3f} при {sd_train:.3f} у обучающих, радиус "
                f"{g['radius_mean']:.1f} ± {g['radius_sd']:.2f} при "
                f"{out['latent_data']['radius_mean']:.1f} у отложенных, эксцесс {g['kurtosis_mean']:.2f}")
    if controls:
        # Латент отбелён: по осям дисперсия 1, среднее 0. Значит N(0, I) прямо
        # в PCA-обратно — уже модель первого порядка, и её надо побить, а не
        # предполагать. Второй контроль добавляет тяжёлый хвост: направление
        # равномерно, радиус — из эмпирического распределения обучающих.
        gc = torch.Generator(device=dev).manual_seed(7700 + seed)
        z_iso = torch.randn(n_clips, k, device=dev, generator=gc)
        z_tr = torch.as_tensor(np.asarray(zf["z_train"], np.float32), device=dev)
        rad = z_tr.norm(dim=1)
        pick = torch.randint(0, len(rad), (n_clips,), device=dev, generator=gc)
        u = torch.randn(n_clips, k, device=dev, generator=gc)
        z_rad = u / u.norm(dim=1, keepdim=True) * rad[pick][:, None]
        out["control_geometry"] = {"N(0,I)": P.geometry(z_iso), "радиус из данных": P.geometry(z_rad)}
        groups["контроль: N(0, I) без потока"] = to_state(z_iso)
        groups["контроль: радиус из данных"] = to_state(z_rad)
        log(f"контроли: N(0,I) радиус {out['control_geometry']['N(0,I)']['radius_mean']:.1f}, "
            f"радиус-из-данных {out['control_geometry']['радиус из данных']['radius_mean']:.1f} "
            f"± {out['control_geometry']['радиус из данных']['radius_sd']:.2f}")
        # Сид, которого нет ни у одного клипа, но и не экстраполяция: середина
        # сферического пути между прообразами ДВУХ РАЗНЫХ отложенных клипов.
        # Радиус при slerp сохраняется, то есть точка остаётся на той же
        # оболочке, где живут настоящие прообразы.
        pair = np.roll(np.arange(n_clips), 1)
        mid = torch.as_tensor(R.slerp(eps_clip.cpu().numpy(), eps_clip.cpu().numpy()[pair], 0.5), device=dev)
        z_mid = R.integrate(flow, P.as_tokens(mid.reshape(n_clips, -1), tokens=tokens),
                            steps=sample_steps).reshape(n_clips, -1)
        out["interp_geometry"] = {"сид": P.geometry(mid.reshape(n_clips, -1)),
                                  "латент": P.geometry(z_mid), "pairs": pair.tolist()}
        groups["середина двух прообразов"] = to_state(z_mid)
        log(f"середина двух прообразов: сид на радиусе {out['interp_geometry']['сид']['radius_mean']:.1f}, "
            f"латент {out['interp_geometry']['латент']['radius_mean']:.1f} "
            f"при {out['latent_data']['radius_mean']:.1f} у данных")

    jobs = {f"{g}|{i}": v[i] for g, v in groups.items() for i in range(n_clips)}
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
    out |= {"clip_idx": clip_idx.tolist(), "n_clips": n_clips, "encoder_agreement": agree, "groups": {}}
    for g in groups:
        sel = [i for i, n in enumerate(names) if n.startswith(g + "|")]
        s = describe([videos[i] for i in sel], nb)
        r = float(np.mean([pixcorr_per_frame(videos[i], raw[j]).mean() for j, i in enumerate(sel)]))
        near = [nearest(videos[i], bank_all) for i in sel]
        out["groups"][g] = {"r_to_raw": r, "gate": float(np.median(rts[sel])),
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
    p.add_argument("--flow", default=str(ROOT / "data" / "prior19" / "flow_pca2048.pt"))
    p.add_argument("--pca", default=str(ROOT / "data" / "prior19" / "pca2048.npz"))
    p.add_argument("--latent", default=str(ROOT / "data" / "prior19" / "pca2048_latent.npz"))
    p.add_argument("--gen", default=str(ROOT / "data" / "gen13b" / "sit.pt"))
    p.add_argument("--pairs13", default=str(ROOT / "data" / "pairs13"))
    p.add_argument("--corpus", default=str(ROOT / "data" / "corpus18"))
    p.add_argument("--out", default=str(ROOT / "data" / "prior19"))
    p.add_argument("--tag", default="seed19")
    p.add_argument("--clips", type=int, default=6)
    p.add_argument("--invert-steps", default="20,100")
    p.add_argument("--sample-steps", type=int, default=20)
    p.add_argument("--draw-scale", type=float, default=0.0, help="-1 — привести к ст. откл. данных")
    p.add_argument("--controls", action="store_true", help="добавить розыгрыши без потока и интерполяцию")
    p.add_argument("--fix", default="", help="поправки сэмплера через ; — \"vscale=1.1;shift=2\" (пункт 20)")
    p.add_argument("--fix-n", type=int, default=256, help="сколько розыгрышей на геометрию поправки")
    p.add_argument("--invert-n", type=int, default=0, help="сколько отложенных латентов обращать")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    pdir = Path(a.pairs13)
    manifest = json.loads((pdir / "manifest.json").read_text(encoding="utf-8"))
    columns = json.loads((pdir / "columns.json").read_text(encoding="utf-8"))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    r = run(a.model, Path(a.flow), Path(a.pca), Path(a.latent), Path(a.gen), manifest, columns,
            Path(a.corpus), n_clips=a.clips, frames=g.get("frames", 40), margin=g.get("margin", 5),
            dt=g.get("dt", 0.02), t_pre=g.get("t_pre", 1.0), seed=a.seed,
            invert_steps=tuple(int(x) for x in a.invert_steps.split(",")),
            sample_steps=a.sample_steps, draw_scale=a.draw_scale, controls=a.controls, invert_n=a.invert_n,
            fixes=[x for x in a.fix.split(';') if x.strip()], fix_n=a.fix_n)
    (out / f"{a.tag}.json").write_text(json.dumps(r["summary"], indent=1, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(out / f"{a.tag}.npz", **r["arrays"])
    S = r["summary"]
    print(f"\nA. прообразы {S['n_test']} отложенных латентов (√k = {S['preimage'][list(S['preimage'])[0]]['typical_radius']:.1f} ± 0,71):")
    for st, v in S["preimage"].items():
        print(f"  {st:>3} шагов: ст. откл. {v['sd']:.3f}, радиус {v['radius_mean']:.1f} ± {v['radius_sd']:.2f}, "
              f"эксцесс {v['kurtosis_mean']:.2f}, замыкание {100 * v['round_trip_error']:.1f} %")
    ld = S["latent_data"]
    print(f"  для сравнения сам латент: ст. откл. {ld['sd']:.3f}, радиус {ld['radius_mean']:.1f} ± {ld['radius_sd']:.2f}")
    print("\n{:28} {:>11} {:>8} {:>9} {:>9} {:>10}".format(
        "группа", "r к сырому", "ворота", "ровного", "контраст", "ближайшее"))
    for kk, v in S["groups"].items():
        print("{:28} {:11.3f} {:8.4f} {:8.1f}% {:9.3f} {:10.3f}".format(
            kk, v["r_to_raw"], v["gate"], 100 * v["frac_flat"], v["sd"], v["nearest_r"]))
    w = S["raw"]
    print("{:28} {:>11} {:>8} {:8.1f}% {:9.3f}".format("сырое видео корпуса", "—", "—",
                                                       100 * w["frac_flat"], w["sd"]))
    print(f"\nсверка кодировщиков: {100 * S['encoder_agreement']:.3f} % от размаха")
    print(f"wrote {out / a.tag}.json / .npz  ({S['seconds']:.0f} s, $0)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
