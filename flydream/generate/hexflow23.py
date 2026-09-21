r"""23: гекс-локальный поток над блоком состояния, без PCA.

Шаги 20 и 22 закрыли ось «форма задачи»: сэмплер починен и блокиратором не был,
объект уменьшен вчетверо и ничего не сдвинулось. Ревизия 22.7 показала, чего
потоку не хватает: он выучил второй момент латента и не выучил четвёртый —
покоординатный эксцесс розыгрышей 4,11 против 8,31 ± 1,17 у обучающих при том
же n, гауссов 2,97 — и поворачивает точку всего на 19,0° против 90° у
случайного поворота. Данные разрежены, розыгрыш плотен.

Остаётся ось «форма модели». По Kamb & Ganguli (ICML 2025) механизм, которым
диффузионная модель вообще обобщает, а не запоминает, — это **локальность плюс
эквивариантность** в архитектуре (r² 0,94-0,96 к предсказанию их аналитической
модели). У нашего потока нет ни того, ни другого: он работает над плоским
вектором PCA, где координата — глобальная мода по всем 721 колонке, и
соседство колонок не выражается никак.

Поэтому здесь:

- **объект** — блок T4a+T4b как есть, 721 × 32, решётка цела. Заодно потолок
  выше: r 0,884 против 0,806 у PCA-1536 (22.1);
- **трёхкратное укрупнение** — подрешётка √3 × √3 даёт 241 центр, и каждая из
  721 колонки лежит в одном шаге ровно от одного центра. Патчи выходят по
  2,99 колонки в среднем;
- **локальное внимание** — токен видит только патчи в радиусе R по подрешётке
  (R = 2 это себя плюс шесть соседей, ровно гексагональная окрестность);
- **эквивариантность** — вместо выученной позиции на токен относительное
  смещение внимания: таблица по 7 (при R = 2) или 19 (при R = 4) различным
  осевым смещениям, общая для всех положений. Абсолютной позиции по умолчанию
  нет вовсе: край поля модель узнаёт по усечённой окрестности, как свёрточная
  сеть узнаёт его по обрезанному окну;
- **глобальный канал** — G служебных токенов, которые видят всё и которых
  видят все. 21б измерил обе половины: локальность реальна (0,876 на одном
  шаге), но одним окном не обойтись — вне решётки остаётся плато 0,25-0,30,
  а внутри одного шага лежит лишь 4,6 % массы ковариации.

Контракт тензоров тот же, что у `prior17.SiTStates`: forward принимает
(B, T, K, n) и возвращает velocity той же формы, где T — коэффициенты DCT
(16), K — типы блока (2), n — колонки (721). Поэтому `integrate`, `invert`,
`sample` и вся приёмка `seed19` работают без единой правки.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from flydream.decode.hexraster import axial_coords, hex_distance
from flydream.generate.gen13b import t_embedding


@lru_cache(maxsize=4)
def patch_layout(n: int = 721, lattice: str = "third") -> dict:
    """Разбиение решётки на патчи вокруг центров подрешётки.

    Каждая колонка отходит ближайшему центру; ничьи достаются центру с меньшим
    индексом, что делает разбиение однозначным и воспроизводимым. Возвращается
    `members` (M, P) с −1 на пустых местах, `owner` (n,) и `slot` (n,) — куда
    именно в своём патче попала колонка, чтобы обратный ход был точным.
    """
    from flydream.generate.inside22 import lattice_mask

    D = hex_distance(n)
    centres = np.where(lattice_mask(lattice, n))[0]
    owner = D[:, centres].argmin(1).astype(np.int64)                   # (n,) индекс патча
    sizes = np.bincount(owner, minlength=len(centres))
    P = int(sizes.max())
    members = np.full((len(centres), P), -1, np.int64)
    slot = np.zeros(n, np.int64)
    fill = np.zeros(len(centres), np.int64)
    for col in range(n):                                               # порядок по индексу колонки — детерминирован
        m = owner[col]
        members[m, fill[m]] = col
        slot[col] = fill[m]
        fill[m] += 1
    return {"centres": centres, "members": members, "owner": owner, "slot": slot,
            "P": P, "M": len(centres), "n": n, "sizes": sizes}


@lru_cache(maxsize=8)
def neighbour_bias(n: int = 721, lattice: str = "third", radius: int = 2) -> dict:
    """Маска локального внимания и индекс относительного смещения на пару.

    Смещение берётся в осевых координатах исходной решётки; на подрешётке
    √3 × √3 оно принимает ровно 7 значений при radius 2 и 19 при radius 4, то
    есть окрестность сама по себе гексагональна. Одна таблица весов на смещение
    и есть эквивариантность: правило одно и то же во всех местах поля.
    """
    lay = patch_layout(n, lattice)
    a, c = axial_coords(n), lay["centres"]
    D = hex_distance(n)[np.ix_(c, c)]
    ok = D <= radius
    off = a[c][None, :, :] - a[c][:, None, :]                          # (M, M, 2)
    uniq = np.unique(off[ok], axis=0)
    key = {tuple(map(int, u)): i for i, u in enumerate(uniq)}
    idx = np.zeros(D.shape, np.int64)
    for i, j in zip(*np.where(ok)):
        idx[i, j] = key[(int(off[i, j, 0]), int(off[i, j, 1]))]
    return {"allowed": ok, "index": idx, "n_offsets": len(uniq),
            "neighbours_mean": float(ok.sum(1).mean()), "neighbours_max": int(ok.sum(1).max())}


class HexAttention(nn.Module):
    """Внимание с гекс-локальной маской, относительным смещением и служебными
    токенами. Служебные видят всё и видимы всем — это «глобальный канал» 21б."""

    def __init__(self, width: int, heads: int, allowed: np.ndarray, index: np.ndarray,
                 n_offsets: int, n_global: int = 4):
        super().__init__()
        assert width % heads == 0, "ширина должна делиться на число голов"
        self.h, self.d, self.G = heads, width // heads, int(n_global)
        M = allowed.shape[0]
        L = M + self.G
        # Индексы смещений: 0..n_offsets-1 — пары патчей, затем два общих
        # значения, «патч ↔ служебный» и «служебный ↔ что угодно».
        big = np.zeros((L, L), np.int64)
        big[:M, :M] = index
        big[:M, M:] = n_offsets
        big[M:, :] = n_offsets + 1
        allow = np.zeros((L, L), bool)
        allow[:M, :M] = allowed
        allow[:M, M:] = True
        allow[M:, :] = True
        self.register_buffer("bias_idx", torch.as_tensor(big))
        self.register_buffer("blocked", torch.as_tensor(~allow))
        self.table = nn.Parameter(torch.zeros(heads, n_offsets + 2))
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)

    def attn_bias(self) -> torch.Tensor:
        b = self.table[:, self.bias_idx.reshape(-1)].view(self.h, *self.bias_idx.shape)
        return b.masked_fill(self.blocked[None], float("-inf"))[None]  # (1, h, L, L)

    def forward(self, x: torch.Tensor) -> torch.Tensor:                # (B, L, width)
        B, L, W = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.h, self.d).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=self.attn_bias().to(q.dtype))
        return self.proj(o.transpose(1, 2).reshape(B, L, W))


class HexSiTBlock(nn.Module):
    """adaLN-Zero ровно как в `gen13b.SiTBlock`, внимание заменено локальным."""

    def __init__(self, width: int, heads: int, t_dim: int, attn_args: dict):
        super().__init__()
        self.n1 = nn.LayerNorm(width, elementwise_affine=False)
        self.attn = HexAttention(width, heads, **attn_args)
        self.n2 = nn.LayerNorm(width, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(approximate="tanh"),
                                 nn.Linear(4 * width, width))
        self.ada = nn.Linear(t_dim, 6 * width)
        nn.init.zeros_(self.ada.weight); nn.init.zeros_(self.ada.bias)

    def forward(self, x: torch.Tensor, te: torch.Tensor) -> torch.Tensor:
        s1, b1, g1, s2, b2, g2 = self.ada(F.silu(te))[:, None].chunk(6, -1)
        x = x + g1 * self.attn(self.n1(x) * (1 + s1) + b1)
        return x + g2 * self.mlp(self.n2(x) * (1 + s2) + b2)


class HexSiT(nn.Module):
    """Токен = патч из P колонок; признаки = T·K значений каждой из них."""

    def __init__(self, frames: int = 16, k: int = 2, width: int = 192, depth: int = 6, heads: int = 4,
                 n: int = 721, lattice: str = "third", radius: int = 2, n_global: int = 4,
                 abs_pos: bool = False):
        super().__init__()
        lay = patch_layout(n, lattice)
        nb = neighbour_bias(n, lattice, radius)
        self.frames, self.k, self.n = frames, k, n
        self.M, self.P, self.G = lay["M"], lay["P"], int(n_global)
        self.t_dim = 128
        feat = self.P * frames * k
        self.register_buffer("members", torch.as_tensor(np.clip(lay["members"], 0, None)))
        self.register_buffer("member_ok", torch.as_tensor((lay["members"] >= 0).astype(np.float32)))
        self.register_buffer("owner", torch.as_tensor(lay["owner"]))
        self.register_buffer("slot", torch.as_tensor(lay["slot"]))
        self.t_mlp = nn.Sequential(nn.Linear(self.t_dim, self.t_dim), nn.SiLU(),
                                   nn.Linear(self.t_dim, self.t_dim))
        self.inp = nn.Linear(feat, width)
        # Абсолютной позиции по умолчанию нет: она ломает эквивариантность,
        # ради которой всё и затевается. Оставлена флагом, чтобы её цену можно
        # было измерить, а не предполагать.
        self.pos = nn.Parameter(torch.randn(1, self.M, width) * 0.02) if abs_pos else None
        self.glob = nn.Parameter(torch.randn(1, self.G, width) * 0.02)
        args = {"allowed": nb["allowed"], "index": nb["index"], "n_offsets": nb["n_offsets"],
                "n_global": self.G}
        self.blocks = nn.ModuleList([HexSiTBlock(width, heads, self.t_dim, args) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.out_ada = nn.Linear(self.t_dim, 2 * width)
        self.out = nn.Linear(width, feat)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        self.geometry = {"M": self.M, "P": self.P, "radius": radius, "lattice": lattice,
                         "n_offsets": nb["n_offsets"], "neighbours_mean": nb["neighbours_mean"],
                         "patch_sizes": lay["sizes"].tolist()}

    def to_patches(self, x: torch.Tensor) -> torch.Tensor:             # (B, T, K, n) -> (B, M, P·T·K)
        B, T, K, n = x.shape
        col = x.permute(0, 3, 1, 2).reshape(B, n, T * K)               # (B, n, T·K)
        g = col[:, self.members.reshape(-1)].view(B, self.M, self.P, T * K)
        return (g * self.member_ok[None, :, :, None]).reshape(B, self.M, -1)

    def from_patches(self, y: torch.Tensor, T: int, K: int) -> torch.Tensor:
        B = y.shape[0]
        g = y.view(B, self.M, self.P, T * K)
        col = g[:, self.owner, self.slot]                              # (B, n, T·K) — каждая колонка ровно раз
        return col.view(B, self.n, T, K).permute(0, 2, 3, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        if y is not None:
            raise ValueError("этот поток безусловный; условия у него нет")
        B, T, K, n = x.shape
        h = self.inp(self.to_patches(x))
        if self.pos is not None:
            h = h + self.pos
        h = torch.cat([h, self.glob.expand(B, -1, -1)], 1)
        te = self.t_mlp(t_embedding(t, self.t_dim))
        for blk in self.blocks:
            h = blk(h, te)
        s, b = self.out_ada(F.silu(te))[:, None].chunk(2, -1)
        out = self.out(self.out_norm(h[:, :self.M]) * (1 + s) + b)
        return self.from_patches(out, T, K)


def build(frames: int = 16, k: int = 2, **kw) -> nn.Module:
    return HexSiT(frames=frames, k=k, width=kw.get("width", 192), depth=kw.get("depth", 6),
                  heads=kw.get("heads", 4), n=int(kw.get("n", 721) or 721),
                  lattice=kw.get("lattice", "third"), radius=int(kw.get("radius", 2)),
                  n_global=int(kw.get("n_global", 4)), abs_pos=bool(kw.get("abs_pos", False)))
