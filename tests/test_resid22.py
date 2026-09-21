"""Offline tests for the residual stage of 22.4. No download, no Modal."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from flydream.decode.hexraster import neighbour_index
from flydream.generate import pca19 as P
from flydream.generate import resid22 as Rs


def test_hexconv_touches_only_the_neighbourhood():
    """Рецептивное поле слоя — ровно один шаг решётки: единичный всплеск в
    колонке может изменить только её саму и её шесть соседей."""
    layer = Rs.HexConv(3, 5)
    with torch.no_grad():
        layer.b.zero_()
    x = torch.zeros(1, 721, 3)
    x[0, 100] = 1.0
    out = layer(x)[0]
    touched = torch.nonzero(out.abs().sum(1) > 1e-8).reshape(-1).tolist()
    nb = neighbour_index(721)[100]
    assert set(touched) <= {100, *[int(j) for j in nb if j >= 0]}
    assert 100 in touched


def test_hexconv_zeroes_the_missing_neighbours_not_wraps():
    """У краевой колонки соседей меньше шести, и отсутствующие дают ноль, а не
    заворачиваются на противоположный край."""
    nb = neighbour_index(721)
    edge = int(np.argmin((nb >= 0).sum(1)))
    assert (nb[edge] < 0).any()
    layer = Rs.HexConv(1, 1)
    with torch.no_grad():
        layer.b.zero_(); layer.w.fill_(1.0)
    x = torch.zeros(1, 721, 1)
    x[0, edge] = 1.0
    out = layer(x)[0, :, 0]
    assert float(out[edge]) == pytest.approx(1.0)
    assert int((out.abs() > 1e-8).sum()) == 1 + int((nb[edge] >= 0).sum())


def test_autoencoder_shapes_and_code_size():
    m = Rs.ResidualAE(c_in=8, width=16, code=3)
    e = torch.randn(2, 721, 8)
    c = m.encode(e)
    assert c.shape == (2, 721, 3)
    assert m.decode(c).shape == e.shape
    assert m(e).shape == e.shape


def test_residual_of_is_what_pca_did_not_take():
    """Остаток ортогонален взятому подпространству и обнуляется при полном k."""
    g = torch.Generator().manual_seed(0)
    X = torch.randn(40, 24, generator=g)
    p = P.fit(X, 8, log=lambda *_: None)
    e = Rs.residual_of(X, p, 8)
    assert e.shape == X.shape
    assert float(e.pow(2).sum()) < float(X.pow(2).sum())
    full = Rs.residual_of(X, p, p["k"])
    assert float(full.pow(2).sum()) <= float(e.pow(2).sum()) + 1e-6
    # Остаток не содержит взятых направлений. Проверять это через `encode`
    # нельзя: он вычитает среднее и делит на корень из собственного числа, и
    # оба действия к ортогональности отношения не имеют. Проекция — это
    # произведение на базис как есть.
    assert float((e @ p["basis"][:, :8]).abs().max()) < 1e-3 * float(X.abs().max())


def test_explained_fraction_reads_one_for_a_perfect_fit():
    e = torch.randn(3, 10, 4, generator=torch.Generator().manual_seed(1))
    assert Rs.explained_fraction(e, e) == pytest.approx(1.0)
    assert Rs.explained_fraction(e, torch.zeros_like(e)) == pytest.approx(0.0, abs=1e-6)
    assert Rs.explained_fraction(e, 0.5 * e) == pytest.approx(0.75, abs=1e-6)


def test_training_reduces_the_residual_on_a_tiny_problem():
    """Проверяется механика, а не качество: сеть должна уметь снять часть
    энергии простого локального остатка и сохранить лучший снимок."""
    g = torch.Generator().manual_seed(2)
    base = torch.randn(64, 721, 2, generator=g)
    E = (base + 0.5 * base.roll(1, dims=1)).reshape(64, -1)
    m = Rs.ResidualAE(c_in=2, width=8, code=2)
    before = Rs.explained_fraction(E[:8].reshape(8, 721, 2), m(E[:8].reshape(8, 721, 2)))
    out = Rs.train(m, E[:48], E[48:], steps=40, batch=8, lr=5e-3, warmup=5,
                   c_in=2, use_amp=False, log=lambda *_: None)
    assert out["nan_steps"] == 0
    assert out["parameters"] > 0 and out["best_step"] > 0
    assert out["best_val_explained"] > before
