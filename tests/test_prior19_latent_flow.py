"""Offline checks of 19.2's wiring: the flow over a 16-token latent.

The change is additive — `build` used to hard-code 721 tokens — so the point of
these tests is that the new path works *and* the old one is untouched, which is
what lets every checkpoint from items 17-18 still load.
"""
import numpy as np
import torch

from flydream.generate import pca19 as P
from flydream.generate import prior17 as R


def test_tokens_round_trip_and_interleave_the_spectrum():
    Z = np.arange(2 * 2048, dtype=np.float32).reshape(2, 2048)
    X = P.as_tokens(Z, tokens=16, k_axis=8)
    assert X.shape == (2, 16, 8, 16)                                  # 16 токенов по 128 признаков
    assert np.array_equal(P.from_tokens(X), Z)
    tok0 = X[0, :, :, 0].reshape(-1)                                  # компоненты, доставшиеся токену 0
    assert np.array_equal(tok0, np.arange(0, 2048, 16, dtype=np.float32))


def test_tokens_reject_a_size_that_does_not_divide():
    try:
        P.as_tokens(np.zeros((1, 2000), np.float32), tokens=16, k_axis=8)
    except ValueError as e:
        assert "2000" in str(e)
    else:
        raise AssertionError("должно падать: 2000 не делится на 16 x 8")


def test_the_flow_runs_on_sixteen_tokens_and_keeps_the_shape():
    m = R.build(frames=16, k=8, n=16, width=32, depth=2, heads=2)
    x = torch.randn(3, 16, 8, 16)
    t = torch.rand(3)
    v = m(x, t)
    assert v.shape == x.shape
    assert m.pos.shape == (1, 16, 32)                                 # позиционных вложений столько же, сколько токенов


def test_the_old_default_is_untouched():
    m = R.build(frames=40, k=8, width=32, depth=2, heads=2)           # без n — как во всех прежних вызовах
    assert m.n == 721 and m.pos.shape == (1, 721, 32)


def test_sampling_takes_the_token_count_from_the_checkpoint_meta():
    m = R.build(frames=16, k=8, n=16, width=32, depth=2, heads=2)
    meta = {"frames": 16, "k": 8, "n": 16, "dct_k": 0}
    out = R.sample_states(m, meta, 2, steps=2, device=torch.device("cpu"))
    assert out.shape == (2, 16, 8, 16)                                # 721 по умолчанию сломало бы это
    assert np.isfinite(out).all()


def test_one_training_step_on_a_latent_sized_batch():
    torch.manual_seed(0)
    states = torch.randn(24, 16, 8, 16)
    m = R.build(frames=16, k=8, n=16, width=32, depth=2, heads=2)
    r = R.train(m, states, steps=3, batch=8, lr=1e-3, log_every=100, log=lambda *_: None)
    assert len(r["history"]) >= 1 and np.isfinite(r["history"][-1]["loss"])
