"""18.20: даёт ли назначенный клипу шум обратно этот клип.

    python -m flydream.generate.assigned18       # локально, CPU, $0

Проверка возможна только у модели, обученной с `couple="fixed"`: там у
каждого обучающего клипа есть **свой** ε, и этот ε — честный розыгрыш
N(0, I) на радиусе √D, а не вывернутый прообраз. То есть впервые появляется
пара «разыгрываемый сид ↔ настоящий клип» с эталоном.

Человек, 2026-09-20: «снимаем сид со стейта, генерим и смотрим».

Соответствие «строка обучения → клип корпуса» здесь не угадывается: каждое
сгенерированное видео сравнивается со **всеми** обучающими клипами корпуса, и
записывается, какой оказался ближайшим. Если ближайшим выходит именно тот, за
кем этот ε закреплён, — сцепка взялась и соответствие подтверждено разом.

Контроль обязателен: поиск лучшего из 13 555 всегда что-нибудь находит,
поэтому тем же способом ищется лучший клип для **свежего розыгрыша**. Его
корреляция и есть ноль этой шкалы.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from flydream.generate import gen13b as G
from flydream.generate import prior17 as R
from flydream.generate.gen13b import DEEP
from flydream.generate.invert import device_of
from flydream.generate.prompts14 import Deep


def best_match(v: np.ndarray, bank: np.ndarray) -> tuple[int, float, float]:
    """Ближайший клип банка по корреляции всего клипа, плюс второй по счёту."""
    a = np.asarray(v, np.float32).reshape(-1)
    a = (a - a.mean()) / (a.std() + 1e-8)
    B = bank.reshape(len(bank), -1).astype(np.float32)
    B = (B - B.mean(1, keepdims=True)) / (B.std(1, keepdims=True) + 1e-8)
    r = (B @ a) / len(a)
    o = np.argsort(-r)
    return int(o[0]), float(r[o[0]]), float(r[o[1]] if len(o) > 1 else np.nan)


def run(prior_ckpt, gen_ckpt, manifest: dict, columns: dict, corpus: Path, *,
        rows=(0, 1, 2, 3), frames: int = 40, steps: int = 100, seed: int = 0, log=print) -> dict:
    t0 = time.time()
    dev = torch.device("cpu")
    prior, pmeta = R.load(prior_ckpt, dev)
    gen, gmeta = G.load(gen_ckpt, dev)
    if pmeta.get("couple") != "fixed":
        raise ValueError(f"этот приор обучен со сцепкой {pmeta.get('couple')!r}; назначенного шума у клипов нет")
    d = Deep(manifest, columns)
    mean = np.array(gmeta["mean"], np.float32); std = np.array(gmeta["std"], np.float32)

    cz = np.load(Path(corpus) / "videos.npz")
    cm = json.loads((Path(corpus) / "pairs_manifest.json").read_text(encoding="utf-8"))
    train = np.sort(np.asarray(cm["split"]["train"]))                 # порядок строк обучения — по возрастанию
    bank = np.asarray(cz["videos"][train][:, :frames], np.float32)
    log(f"{len(train)} обучающих клипов в банке, {time.time() - t0:.0f} с")

    shape = (pmeta["frames"], pmeta.get("k", 8), 721)
    rows = list(rows)
    eps = R.fixed_noise(torch.as_tensor(rows), shape, dev, seed).cpu().numpy().astype(np.float32)
    g = torch.Generator(device=dev).manual_seed(5000 + seed)
    draw = torch.randn((len(rows),) + shape, device=dev, generator=g).cpu().numpy().astype(np.float32)
    D = int(np.prod(shape))
    log(f"назначенный ‖ε‖ {np.linalg.norm(eps.reshape(len(rows), -1), axis=1).mean():.1f}, "
        f"розыгрыш {np.linalg.norm(draw.reshape(len(rows), -1), axis=1).mean():.1f}, √D = {np.sqrt(D):.1f}")

    states = {"назначенный шум": R.from_noise(prior, pmeta, eps, steps=steps, device=dev),
              "свежий розыгрыш": R.from_noise(prior, pmeta, draw, steps=steps, device=dev)}
    names = [f"{k}|{i}" for k in states for i in range(len(rows))]
    cond = torch.as_tensor(np.stack([states[n.split("|")[0]][int(n.split("|")[1])] for n in names]), device=dev)
    mask = torch.ones(len(names), len(DEEP), device=dev)
    vids = []
    with torch.no_grad():
        for i in range(0, len(names), 8):
            gg = torch.Generator(device=dev).manual_seed(1000 + seed)
            vids.append(G.sample(gen, cond[i:i + 8], mask[i:i + 8], steps=20, generator=gg).cpu().numpy())
    videos = np.concatenate(vids).astype(np.float32)
    log(f"{len(videos)} видео отрисовано за {time.time() - t0:.0f} с")

    out = {"prior_ckpt": str(prior_ckpt), "couple": pmeta.get("couple"), "rows": rows, "steps": steps,
           "seed": seed, "dims": D, "typical_radius": float(np.sqrt(D)), "n_bank": int(len(train)), "groups": {}}
    for k in states:
        recs = []
        for i, j in enumerate(rows):
            v = videos[names.index(f"{k}|{i}")]
            b, r1, r2 = best_match(v, bank)
            recs.append({"row": int(j), "expected_clip": int(train[j]), "best_clip": int(train[b]),
                         "hit": bool(b == j), "r_best": r1, "r_second": r2,
                         "r_expected": float(best_match(v, bank[j:j + 1])[1])})
        out["groups"][k] = recs
    out["seconds"] = round(time.time() - t0, 1)
    arrays = {f"video__{n}": videos[i] for i, n in enumerate(names)}
    arrays.update({f"bank__{j}": bank[j] for j in rows})
    return {"summary": out, "arrays": arrays}


def main(argv=None) -> int:
    from flydream.generate.invert import settings
    from flydream.model import ROOT

    g = settings()
    p = argparse.ArgumentParser()
    p.add_argument("--prior", default=str(ROOT / "data" / "prior18" / "corpus_dct16_w192_lr1e3_pair_c.pt"))
    p.add_argument("--gen", default=str(ROOT / "data" / "gen13b" / "sit.pt"))
    p.add_argument("--pairs13", default=str(ROOT / "data" / "pairs13"))
    p.add_argument("--corpus", default=str(ROOT / "data" / "corpus18"))
    p.add_argument("--out", default=str(ROOT / "data" / "prior18"))
    p.add_argument("--tag", default="assigned18")
    p.add_argument("--rows", default="0,1,2,3")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    pdir = Path(a.pairs13)
    manifest = json.loads((pdir / "manifest.json").read_text(encoding="utf-8"))
    columns = json.loads((pdir / "columns.json").read_text(encoding="utf-8"))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    r = run(Path(a.prior), Path(a.gen), manifest, columns, Path(a.corpus),
            rows=tuple(int(x) for x in a.rows.split(",")), frames=g.get("frames", 40),
            steps=a.steps, seed=a.seed)
    (out / f"{a.tag}.json").write_text(json.dumps(r["summary"], indent=1, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(out / f"{a.tag}.npz", **r["arrays"])
    S = r["summary"]
    for k, recs in S["groups"].items():
        print(f"\n{k}:")
        print(f"  {'строка':>7} {'ждём клип':>10} {'ближайший':>10} {'попал':>6} "
              f"{'r лучшего':>10} {'r второго':>10} {'r ожидаемого':>13}")
        for w in recs:
            print(f"  {w['row']:7d} {w['expected_clip']:10d} {w['best_clip']:10d} {str(w['hit']):>6} "
                  f"{w['r_best']:10.3f} {w['r_second']:10.3f} {w['r_expected']:13.3f}")
    print(f"\nwrote {out / a.tag}.json / .npz  ({S['seconds']:.0f} s, $0)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
