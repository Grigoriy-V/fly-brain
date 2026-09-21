r"""22.8: полы и размах судей, которыми мы меряем сгенерированное видео.

    python -m flydream.generate.floors22          # локально, CPU, $0

Числа вроде «ближайшее 0,526 — значит не копия» и «r к сырому 0,017 — значит
не сцена» держатся только тогда, когда известно, что эти судьи читают на
заведомо плохом и на заведомо хорошем. Пол не измерялся ни разу.

Судьи берутся ТЕМИ ЖЕ функциями, что в `seed19.py`, иначе сравнение шкал
бессмысленно: `nearest` из `vaeval18`, `describe` из `edges18`,
`pixcorr_per_frame` из `invert`.

Арены:

- **настоящий клип** против банка без самого себя — сколько даёт настоящее
  видео, то есть верх шкалы `nearest_r`;
- **настоящий клип против только обучающего банка** — та же величина, но без
  соседей из своего же отложенного сплита;
- **клип, перемешанный во времени** — штатный контроль проекта: те же кадры,
  разрушенная сцена;
- **белый шум** и **шум по статистике клипа** — низ шкалы.

`r_to_raw` меряется между ДВУМЯ РАЗНЫМИ настоящими клипами: это пол той
колонки, в которой свежий розыгрыш читает 0,017.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from flydream.generate import learned as L
from flydream.generate.edges18 import describe
from flydream.generate.invert import pixcorr_per_frame
from flydream.generate.vaeval18 import nearest


def nearest_excluding(v: np.ndarray, bank: np.ndarray, skip: int) -> tuple[int, float]:
    """`nearest` без строки `skip`. Строка временно обнуляется, а не удаляется:
    так работает ровно та же функция, что и в замерах, и банк не копируется
    (гигабайт). Постоянный ряд даёт std 0 и корреляцию 0."""
    keep = bank[skip].copy()
    bank[skip] = 0
    try:
        return nearest(v, bank)
    finally:
        bank[skip] = keep


def run(corpus: Path, tag_json: Path, *, frames: int = 40, seed: int = 0, log=print) -> dict:
    t0 = time.time()
    rng = np.random.default_rng(seed)
    cz = np.load(Path(corpus) / "videos.npz")
    cm = json.loads((Path(corpus) / "pairs_manifest.json").read_text(encoding="utf-8"))
    bank = np.asarray(cz["videos"][:, :frames], np.float16)
    train_rows = np.asarray(cm["split"]["train"])
    clip_idx = json.loads(Path(tag_json).read_text(encoding="utf-8"))["clip_idx"]
    raw = np.asarray(cz["videos"][clip_idx][:, :frames], np.float32)
    nb = np.asarray(L.neighbour_index(721))
    log(f"банк {bank.shape}, обучающих {len(train_rows)}, клипов {len(clip_idx)}")

    arms: dict[str, np.ndarray] = {"настоящий клип": raw}
    sh = rng.permutation(frames)                                       # один и тот же порядок всем клипам
    arms["клип, перемешанный во времени"] = raw[:, sh]
    arms["белый шум"] = rng.standard_normal(raw.shape).astype(np.float32)
    arms["шум по статистике клипа"] = np.stack(
        [rng.standard_normal(raw.shape[1:]).astype(np.float32) * raw[i].std() + raw[i].mean()
         for i in range(len(raw))])

    out: dict = {"clip_idx": list(map(int, clip_idx)), "frames": frames, "n_bank": int(len(bank)),
                 "n_train": int(len(train_rows)), "arms": {}}
    for name, V in arms.items():
        # Своя строка исключается только у настоящего клипа: у остальных её нет.
        near = [nearest_excluding(V[i], bank, int(clip_idx[i])) if name == "настоящий клип"
                else nearest(V[i], bank) for i in range(len(V))]
        s = describe(list(V), nb)
        out["arms"][name] = {
            "nearest_r": float(np.mean([r for _, r in near])),
            "nearest_r_min": float(np.min([r for _, r in near])),
            "nearest_idx": [int(j) for j, _ in near],
            "frac_flat": s["frac_flat"]["mean"], "sd": s["sd"]["mean"],
            "kurtosis": s["grad_kurtosis"]["mean"], "neigh_r": s["neigh_r"]["mean"]}
        log(f"  {name}: ближайшее {out['arms'][name]['nearest_r']:.3f}, "
            f"ровного {100 * s['frac_flat']['mean']:.1f} %, контраст {s['sd']['mean']:.3f}")

    # То же для настоящего клипа, но банк — только обучающий сплит.
    tb = np.ascontiguousarray(bank[train_rows])
    nr = [nearest(raw[i], tb) for i in range(len(raw))]
    out["arms"]["настоящий клип, банк только обучающий"] = {
        "nearest_r": float(np.mean([r for _, r in nr])),
        "nearest_r_min": float(np.min([r for _, r in nr])),
        "nearest_idx": [int(train_rows[j]) for j, _ in nr]}
    log(f"  настоящий клип против только обучающих: "
        f"{out['arms']['настоящий клип, банк только обучающий']['nearest_r']:.3f}")
    del tb

    # Пол колонки `r к сырому`: два РАЗНЫХ настоящих клипа, той же функцией.
    pairs = [(i, j) for i in range(len(raw)) for j in range(len(raw)) if i != j]
    rr = [float(pixcorr_per_frame(raw[i], raw[j]).mean()) for i, j in pairs]
    sn = [float(pixcorr_per_frame(arms["белый шум"][i], raw[i]).mean()) for i in range(len(raw))]
    ss = [float(pixcorr_per_frame(arms["клип, перемешанный во времени"][i], raw[i]).mean())
          for i in range(len(raw))]
    out["r_to_raw"] = {"два разных клипа": {"mean": float(np.mean(rr)), "sd": float(np.std(rr)),
                                            "max": float(np.max(rr)), "n": len(rr)},
                       "белый шум против клипа": {"mean": float(np.mean(sn)), "sd": float(np.std(sn))},
                       "свой клип, перемешанный во времени": {"mean": float(np.mean(ss)),
                                                              "sd": float(np.std(ss))}}
    log(f"  r между двумя разными клипами: {np.mean(rr):.3f} ± {np.std(rr):.3f}, "
        f"наибольшее {np.max(rr):.3f}; шум против клипа {np.mean(sn):.3f}; "
        f"перемешанный во времени против своего же {np.mean(ss):.3f}")
    out["seconds"] = round(time.time() - t0, 1)
    return out


def main(argv=None) -> int:
    ROOT = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default=str(ROOT / "data" / "corpus18"))
    p.add_argument("--clips-from", default=str(ROOT / "data" / "prior19" / "seed22_ab1536.json"))
    p.add_argument("--out", default=str(ROOT / "data" / "prior19" / "floors22.json"))
    p.add_argument("--frames", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    out = run(Path(a.corpus), Path(a.clips_from), frames=a.frames, seed=a.seed)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {a.out} за {out['seconds']} с")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
