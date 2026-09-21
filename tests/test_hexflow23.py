"""Offline tests for the hex-local flow of 23. No download, no Modal."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from flydream.decode.hexraster import axial_coords, hex_distance
from flydream.generate import hexflow23 as H
from flydream.generate import prior17 as R


def test_patches_partition_the_lattice_exactly_once():
    """Каждая колонка лежит ровно в одном патче и на своём месте в нём."""
    lay = H.patch_layout(721, "third")
    m = lay["members"]
    got = sorted(int(c) for c in m.reshape(-1) if c >= 0)
    assert got == list(range(721))
    assert lay["M"] == 241
    # `owner`/`slot` должны указывать ровно туда, где колонка лежит.
    for col in range(721):
        assert int(m[lay["owner"][col], lay["slot"][col]]) == col


def test_every_column_is_one_step_from_its_centre_and_patches_are_three():
    """Трёхкратное укрупнение буквальное: 721 / 3. Наивная раздача давала
    патчи от 1 до 5 колонок, и это ломало эквивариантность — одно правило во
    всех местах поля. Сбалансированная даёт ровно по три, кроме двух патчей по
    две, потому что 721 = 241·3 − 2."""
    lay = H.patch_layout(721, "third")
    D = hex_distance(721)
    far = [c for c in range(721) if D[c, lay["centres"][lay["owner"][c]]] > 1]
    assert far == []
    assert lay["P"] == 3
    assert sorted(np.bincount(lay["sizes"]).tolist()) == [0, 0, 2, 239]
    assert lay["sizes"].mean() == pytest.approx(721 / 241, abs=1e-6)
    assert all(int(lay["members"][m, 0]) == int(c) for m, c in enumerate(lay["centres"]))


def test_the_patch_lattice_is_itself_hexagonal():
    """При радиусе 2 различных смещений ровно семь (себя плюс шесть
    направлений), при радиусе 4 — девятнадцать. Иначе подрешётка выбрана не
    та и относительное смещение перестанет быть эквивариантным."""
    assert H.neighbour_bias(721, "third", 2)["n_offsets"] == 7
    assert H.neighbour_bias(721, "third", 4)["n_offsets"] == 19
    nb = H.neighbour_bias(721, "third", 2)
    assert nb["neighbours_max"] == 7                                   # центр поля: себя и шесть соседей
    assert 6.0 < nb["neighbours_mean"] < 7.0                           # на краю соседей меньше


def test_relative_bias_is_shared_by_equal_offsets():
    """Эквивариантность: две пары с одинаковым осевым смещением обязаны брать
    один и тот же вес, где бы они ни стояли на поле."""
    nb = H.neighbour_bias(721, "third", 2)
    lay = H.patch_layout(721, "third")
    a = axial_coords(721)[lay["centres"]]
    ok, idx = nb["allowed"], nb["index"]
    seen: dict[tuple[int, int], set[int]] = {}
    for i, j in zip(*np.where(ok)):
        off = (int(a[j, 0] - a[i, 0]), int(a[j, 1] - a[i, 1]))
        seen.setdefault(off, set()).add(int(idx[i, j]))
    assert all(len(v) == 1 for v in seen.values())
    assert len({next(iter(v)) for v in seen.values()}) == nb["n_offsets"]


def test_to_patches_and_back_is_the_identity():
    m = H.HexSiT(frames=4, k=2, width=32, depth=1, heads=4)
    x = torch.randn(2, 4, 2, 721)
    assert torch.allclose(m.from_patches(m.to_patches(x), 4, 2), x, atol=1e-6)


def test_padding_slots_carry_no_signal():
    """У патчей меньше P колонок пустые места должны быть нулями, иначе в
    признак токена попадёт чужая колонка."""
    m = H.HexSiT(frames=3, k=2, width=32, depth=1, heads=4)
    lay = H.patch_layout(721, "third")
    small = int(np.argmin(lay["sizes"]))
    p = m.to_patches(torch.ones(1, 3, 2, 721)).view(1, m.M, m.P, 6)
    assert float(p[0, small, lay["sizes"][small]:].abs().max()) == 0.0
    assert float(p[0, small, : lay["sizes"][small]].min()) == 1.0


def test_attention_without_globals_reaches_only_the_neighbourhood():
    """Один блок без служебных токенов не имеет права вынести сигнал за
    радиус: это и есть локальность, ради которой шаг делается."""
    torch.manual_seed(0)
    att = H.HexAttention(16, 4, **{k: v for k, v in
                                   ((kk, H.neighbour_bias(721, "third", 2)[kk])
                                    for kk in ("allowed", "index", "n_offsets"))}, n_global=0)
    x = torch.zeros(1, att.bias_idx.shape[0], 16)
    base = att(x)
    x[0, 100] = 1.0
    moved = torch.nonzero((att(x) - base).abs().sum(-1)[0] > 1e-6).reshape(-1).tolist()
    allowed = set(np.where(H.neighbour_bias(721, "third", 2)["allowed"][:, 100])[0].tolist())
    assert set(moved) <= allowed
    assert len(moved) > 1                                              # соседи всё-таки видят


def test_global_tokens_are_the_only_long_range_path():
    """С служебными токенами дальняя связь появляется — это «глобальный
    канал» 21б, и он должен быть именно каналом, а не отменой локальности."""
    torch.manual_seed(0)
    nb = H.neighbour_bias(721, "third", 2)
    att = H.HexAttention(16, 4, allowed=nb["allowed"], index=nb["index"],
                         n_offsets=nb["n_offsets"], n_global=2)
    M = nb["allowed"].shape[0]
    assert bool(att.blocked[:M, :M].any())                             # между патчами связь не полная
    assert not bool(att.blocked[:M, M:].any())                         # к служебным — всегда
    assert not bool(att.blocked[M:].any())                             # от служебных — всегда


def test_forward_keeps_the_prior17_tensor_contract():
    """(B, T, K, n) на входе и на выходе — иначе integrate и seed19 сломаются."""
    m = H.HexSiT(frames=16, k=2, width=32, depth=2, heads=4)
    x = torch.randn(3, 16, 2, 721)
    t = torch.rand(3)
    assert m(x, t).shape == x.shape


def test_output_starts_at_zero_like_every_adaln_zero_model():
    m = H.HexSiT(frames=8, k=2, width=32, depth=2, heads=4)
    assert float(m(torch.randn(2, 8, 2, 721), torch.rand(2)).abs().max()) == 0.0


def test_absolute_position_is_off_by_default_and_optional():
    assert H.HexSiT(frames=4, k=2, width=32, depth=1).pos is None
    assert H.HexSiT(frames=4, k=2, width=32, depth=1, abs_pos=True).pos is not None


def test_prior17_builds_and_integrates_the_hex_backbone():
    """Поток должен ходить через ту же арифметику интегрирования."""
    m = R.build(frames=8, k=2, backbone="hex", width=32, depth=2, heads=4)
    assert isinstance(m, H.HexSiT)
    x = torch.randn(2, 8, 2, 721)
    out = R.integrate(m, x, steps=3)
    assert out.shape == x.shape
    assert torch.allclose(out, x)                                      # веса на нуле: скорость нулевая


def test_checkpoint_round_trip_rebuilds_the_hex_backbone(tmp_path):
    from flydream.generate.gen13b import EMA

    m = R.build(frames=8, k=2, backbone="hex", width=32, depth=2, heads=4, radius=4, n_global=2)
    with torch.no_grad():
        m.out.weight.normal_(std=0.02)                                 # чтобы выход не был тождественным нулём
    meta = {"frames": 8, "k": 2, "n": 721, "width": 32, "depth": 2, "heads": 4,
            "backbone": "hex", "lattice": "third", "radius": 4, "n_global": 2}
    p = tmp_path / "hex.pt"
    R.save(p, m, EMA(m, decay=0.0), meta)
    back, got = R.load(p, torch.device("cpu"))
    assert isinstance(back, H.HexSiT) and got["radius"] == 4
    x, t = torch.randn(2, 8, 2, 721), torch.rand(2)
    with torch.no_grad():
        assert torch.allclose(back(x, t), m(x, t), atol=1e-5)
