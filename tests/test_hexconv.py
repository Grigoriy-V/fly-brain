"""Offline: rung two of the decoder ladder, on a small lattice and synthetic data.

No download and no credential: `Conv2dHexSpace` and `get_hex_coords` are pure
torch and integer geometry. The lattice here is extent 2, nineteen hexals,
rather than the 721 of the real runs.
"""
import numpy as np
import pytest
import torch

from flydream.decode.hexconv import HexDecoder, fit

EXTENT = 2
N_HEX = 19


def test_the_head_returns_one_channel_per_frame_on_the_same_lattice():
    net = HexDecoder(in_channels=1, out_channels=1, extent=EXTENT, shape=(4,))
    net.eval()
    with torch.no_grad():
        out = net(torch.rand(3, 7, 1, N_HEX))
    assert out.shape == (3, 7, 1, N_HEX)
    assert torch.isfinite(out).all()


def test_the_axial_map_is_the_bounding_box_of_the_lattice():
    net = HexDecoder(in_channels=1, extent=EXTENT)
    assert net.rows == 5 and net.cols == 5          # extent 2 spans u,v in -2..2
    assert len(net.u) == len(net.v) == N_HEX


def test_the_head_is_per_frame_and_carries_no_state():
    """Frames are folded into the batch dimension, so a change at frame 4 must
    not touch frames 0 to 3. This is why the head is comparable to ridge at
    lag 0 and to nothing else; in train mode batch norm couples the batch, so
    the check is made in eval mode, which is how it is always read."""
    net = HexDecoder(in_channels=1, extent=EXTENT, shape=(4,), p_dropout=0.0)
    net.eval()
    a = torch.rand(1, 6, 1, N_HEX)
    b = a.clone()
    b[:, 4:] = torch.rand(1, 2, 1, N_HEX)
    with torch.no_grad():
        oa, ob = net(a), net(b)
    assert torch.allclose(oa[:, :4], ob[:, :4], atol=1e-6)
    assert not torch.allclose(oa[:, 4:], ob[:, 4:], atol=1e-6)


def test_more_input_channels_means_more_free_parameters():
    from flyvis.utils.nn_utils import n_params

    small = int(n_params(HexDecoder(in_channels=1, extent=EXTENT)).free)
    large = int(n_params(HexDecoder(in_channels=8, extent=EXTENT)).free)
    assert large > small


def test_the_gradient_reaches_every_parameter():
    net = HexDecoder(in_channels=1, extent=EXTENT, shape=(4,), p_dropout=0.0)
    net(torch.rand(2, 3, 1, N_HEX)).sum().backward()
    missing = [n for n, p in net.named_parameters() if p.requires_grad and p.grad is None]
    assert missing == []


def test_fit_learns_a_lattice_map_and_keeps_its_best_epoch():
    """The target is a fixed linear mixing of the input over the lattice, which a
    convolution can represent; the point is that the loop trains and that the
    returned head is the best validation epoch, not the last."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(24, 5, N_HEX)).astype(np.float32)
    mix = rng.normal(size=(N_HEX, N_HEX)).astype(np.float32) / N_HEX
    y = (x @ mix).astype(np.float32)
    out = fit(x[:18], y[:18], x[18:], y[18:], epochs=12, batch=4, seed=0,
              shape=(6,), p_dropout=0.0)
    assert len(out.val_loss) == 12
    assert out.best_val == pytest.approx(min(out.val_loss))
    assert out.val_loss[-1] < out.val_loss[0]
    assert out.n_parameters > 0


def test_predict_returns_rows_of_hexals_for_scoring():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(10, 4, N_HEX)).astype(np.float32)
    y = x.copy()
    out = fit(x[:8], y[:8], x[8:], y[8:], epochs=2, batch=4, seed=0, shape=(4,), p_dropout=0.0)
    pred = out.predict(x[8:])
    assert pred.shape == (2 * 4, N_HEX)        # samples x frames flattened, as metrics want
    assert np.isfinite(pred).all()


def test_standardisation_is_one_scalar_not_one_per_cell():
    """A per-channel rescaling would equalise response amplitude across the
    lattice, the same intervention ridge.py refuses. One scalar only makes the
    optimiser's step comparable across cell types of very different loudness."""
    rng = np.random.default_rng(2)
    x = rng.normal(size=(8, 3, N_HEX)).astype(np.float32) * 100.0
    out = fit(x[:6], x[:6], x[6:], x[6:], epochs=1, batch=3, seed=0, shape=(4,), p_dropout=0.0)
    assert np.isscalar(out.x_mean) or np.ndim(out.x_mean) == 0
    assert np.ndim(out.x_std) == 0 and out.x_std > 1.0


def test_lattice_extent_accepts_a_lattice_and_refuses_a_strided_type():
    from flydream.decode.hexconv import lattice_extent

    assert lattice_extent(1) == 0
    assert lattice_extent(19) == 2
    assert lattice_extent(721) == 15
    with pytest.raises(ValueError, match="not a regular hex lattice"):
        lattice_extent(123)               # Lawf1 and Lawf2, strided


def test_pad_to_lattice_places_a_strided_type_and_zeroes_the_rest():
    from flyvis.utils.hex_utils import get_hex_coords
    from flydream.decode.hexconv import pad_to_lattice

    gu, gv = get_hex_coords(EXTENT)
    take = np.arange(0, N_HEX, 2)                      # a stand-in for a strided type
    vals = np.arange(1.0, len(take) + 1.0)[None, :]
    out = pad_to_lattice(vals, gu[take], gv[take], EXTENT)
    assert out.shape == (1, N_HEX)
    assert np.count_nonzero(out) == len(take)
    assert np.array_equal(out[0, take], vals[0])
