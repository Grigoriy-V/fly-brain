"""The generator's window: `frames + margin` fitted, `frames` saved (ROADMAP item 9)."""
import numpy as np

from flydream.generate.invert import extend_clip, pixcorr_per_frame, settings


def test_margin_from_the_next_chunk():
    clip = np.random.default_rng(0).random((40, 721), dtype=np.float32)
    nxt = np.random.default_rng(1).random((40, 721), dtype=np.float32)
    video, source = extend_clip(clip, nxt, 5)
    assert source == "next" and video.shape == (45, 721)
    assert np.array_equal(video[:40], clip) and np.array_equal(video[40:], nxt[:5])


def test_margin_holds_the_last_frame_without_a_next_chunk():
    clip = np.random.default_rng(0).random((40, 721), dtype=np.float32)
    video, source = extend_clip(clip, None, 5)
    assert source == "held" and video.shape == (45, 721)
    assert np.array_equal(video[40:], np.repeat(clip[-1:], 5, axis=0))
    video, source = extend_clip(clip, np.zeros((2, 721), np.float32), 5)   # too short a next chunk
    assert source == "held"


def test_zero_margin_is_the_clip():
    clip = np.zeros((7, 721), np.float32)
    video, source = extend_clip(clip, None, 0)
    assert source == "none" and video is clip


def test_settings_name_the_window():
    g = settings()
    assert g["frames"] == 40 and g["margin"] == 5 and g["dt"] == 0.02


def test_pixcorr_scores_the_shown_frames_only():
    a = np.random.default_rng(0).random((40, 721))
    assert pixcorr_per_frame(a, a).shape == (40,) and np.allclose(pixcorr_per_frame(a, a), 1.0)


def test_task_weights_give_each_task_its_own_mean():
    import torch
    from flydream.generate.invert import task_weights

    cells = [np.array([0, 1]), np.array([2, 3, 4])]
    w = task_weights(cells, 6, "cpu")
    diff = torch.arange(2 * 3 * 6, dtype=torch.float32).reshape(2, 3, 6)
    got = ((diff ** 2) * w[:, None, :]).sum(-1).mean(1)
    want = torch.stack([(diff[0][:, :2] ** 2).mean(), (diff[1][:, 2:5] ** 2).mean()])
    assert torch.allclose(got, want)


def test_batched_inversion_matches_the_single_one(tmp_path):
    """Two tasks in one pass give the videos two separate passes give (the
    pictures must not change under item 10')."""
    import os
    os.environ["FLYVIS_ROOT_DIR"] = str(tmp_path)
    import torch
    from flydream.model import patch_datamate_for_windows
    patch_datamate_for_windows()
    from flyvis import Network
    from flyvis.utils.config_utils import Namespace
    from flydream.generate import invert as I

    fixture = str(__import__("pathlib").Path(__file__).parent / "fixtures" / "mini_connectome.json")
    net = Network(connectome=Namespace(type="ConnectomeFromAvgFilters", file=fixture, extent=2, n_syn_fill=0))
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    H, T, dt = net.stimulus.n_input_elements, 4, 0.02
    torch.manual_seed(0)
    video = torch.rand(2, T, H)
    state1 = net.steady_state(0.2, dt, batch_size=1, value=0.5)
    with torch.no_grad():
        targets = torch.cat([I.simulate(net, video[[b]], dt, state1) for b in range(2)])
    cells = [np.arange(0, 5), np.arange(5, net.n_nodes)]
    single = [I.invert(net, targets[[b]], cells[b], dt=dt, state=state1, steps=5, lr=0.05, tv=0.02, init=None,
                       log_every=100)[0][0] for b in range(2)]
    state2 = net.steady_state(0.2, dt, batch_size=2, value=0.5)
    batched, trace, n = I.invert_batch(net, targets, cells, dt=dt, state=state2, steps=5, lr=0.05, tv=0.02, log_every=100)
    assert n == 5 and trace.shape == (5, 2)
    for b in range(2):
        assert torch.allclose(batched[b], single[b], atol=1e-5), b
