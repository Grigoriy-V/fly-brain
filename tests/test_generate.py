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


def test_dream_inputs_and_shuffle_and_noisy_simulation(tmp_path):
    import os
    os.environ["FLYVIS_ROOT_DIR"] = str(tmp_path)
    import torch
    from flydream.model import patch_datamate_for_windows
    patch_datamate_for_windows()
    from flyvis import Network
    from flyvis.utils.config_utils import Namespace
    from flydream.generate import dreams as D
    from flydream.generate import invert as I

    d = {"eye_noise_sd": 0.1, "flash_start": 2, "flash_frames": 2, "dark_frames": 4}
    rng = np.random.default_rng(0)
    v, tw = D.input_video("eye_noise", 6, 2, 0.02, d, rng)
    assert v.shape == (8, 721) and v.min() >= 0 and v.max() <= 1 and tw.sum() == 8
    v, tw = D.input_video("flash", 6, 2, 0.02, d, rng)
    assert v[1].max() == 0.5 and v[2].min() == 1.0 and v[4].max() == 0.5
    v, tw = D.input_video("neuron_noise", 6, 2, 0.02, d, rng)
    assert np.all(v == 0.5)

    fixture = str(__import__("pathlib").Path(__file__).parent / "fixtures" / "mini_connectome.json")
    net = Network(connectome=Namespace(type="ConnectomeFromAvgFilters", file=fixture, extent=2, n_syn_fill=0))
    net.eval()
    H, T, dt = net.stimulus.n_input_elements, 5, 0.02
    grey = torch.full((1, T, H), 0.5)
    state = net.steady_state(0.2, dt, batch_size=1, value=0.5)
    with torch.no_grad():
        quiet = I.simulate(net, grey, dt, state)
        noisy0 = D.simulate_noisy(net, grey, dt, state, 0.0, seed=0)
        noisy = D.simulate_noisy(net, grey, dt, state, 0.1, seed=0)
        again = D.simulate_noisy(net, grey, dt, state, 0.1, seed=0)
    assert torch.allclose(noisy0, quiet, atol=1e-6)        # sd 0 is the plain forward
    assert not torch.allclose(noisy, quiet) and torch.allclose(noisy, again)   # noise acts, and is seeded
    cells = np.arange(0, 4)
    sh = D.shuffled(quiet, cells, np.random.default_rng(1))
    assert torch.allclose(sh[:, :, 4:], quiet[:, :, 4:])
    assert torch.allclose(sh[:, :, :4].sort(-1).values, quiet[:, :, :4].sort(-1).values)


def test_type_weights_normalise_each_type_to_its_variance():
    import torch
    from flydream.generate.mix import type_weights

    index = {"a": np.arange(0, 4), "b": np.arange(4, 10)}
    target = torch.zeros(1, 5, 10)
    target[0, :, :4] = torch.arange(5.0)[:, None] * 2      # var over (time, cells) of a = 8
    target[0, :, 4:] = torch.arange(5.0)[:, None]          # var of b = 2
    cells, w = type_weights(index, ["a", "b"], target)
    assert cells.tolist() == list(range(10))
    # weight sums per type: 1/(k*var): a -> 1/(2*8), b -> 1/(2*2)
    assert np.isclose(w[:4].sum(), 1 / 16, rtol=1e-3) and np.isclose(w[4:].sum(), 1 / 4, rtol=1e-3)
    # the normalised error of a type is its MSE / var: equal errors in units of sd weigh the same
    err = torch.ones(10)
    assert np.isclose(float((err[:4] ** 2 * torch.as_tensor(w[:4])).sum()) * 8, float((err[4:] ** 2 * torch.as_tensor(w[4:])).sum()) * 2)


def test_cell_weights_reach_the_batched_fit():
    import torch
    from flydream.generate.invert import task_weights

    w = task_weights([np.array([1, 2])], 4, "cpu", weights=[np.array([0.25, 0.75], np.float32)])
    assert w.tolist() == [[0.0, 0.25, 0.75, 0.0]]


def test_pairs13_split_holds_out_whole_scenes_and_classes():
    from flydream.generate.pairs13 import split_indices

    meta = [{"source": "sintel", "scene": s} for s in ["a", "a", "b", "c"]] + \
           [{"source": "procedural", "class": c} for c in ["edge", "dots", "dots", "bar"]]
    sp = split_indices(meta, held_scenes=["b"], held_classes=["dots"], val_fraction=0.25, seed=0)
    assert sp["test"] == [2, 5, 6]
    assert sorted(sp["train"] + sp["val"]) == [0, 1, 3, 4, 7] and len(sp["val"]) == 1
    assert not (set(sp["train"]) & set(sp["val"]) & set(sp["test"]))


def test_learned_models_shapes_and_ring():
    import torch
    from flydream.generate import learned as L

    r1 = L.ring_index(1)
    assert r1.shape == (721, 7) and (r1[:, 0] == np.arange(721)).all()
    r2 = L.ring_index(2)
    assert r2.shape == (721, 19) and (r2[360] >= 0).all()          # the centre column has the full 2-ring
    x = torch.rand(2, 9, 3, 721)
    lin = L.LinearHexTemporal(3, rings=1, taps=5)
    assert lin(x, 6).shape == (2, 6, 721)
    cnn = L.HexTemporalCNN(3, width=8, depth=2, rings=1, taps=3)
    assert cnn(x, 9).shape == (2, 9, 721)                          # taps past the end are edge-padded
    n = sum(p.numel() for p in lin.parameters()); assert n == 3 * 7 * 5 + 1
    p = np.random.default_rng(0).random((4, 5, 721)); assert np.allclose(L.pixcorr(p, p), 1.0)


def test_gen13b_masks_backbones_and_interpolant():
    import torch
    from flydream.generate import gen13b as G

    rng = np.random.default_rng(0)
    m = G.sample_masks(500, rng)
    assert m.shape == (500, 8) and (m.sum(1) == 0).mean() < 0.15          # the unconditional share is small
    assert G.named_mask("dir_c").tolist() == [0, 0, 1, 0, 0, 0, 1, 0] and G.named_mask("t5").sum() == 4
    vids = torch.rand(6, 8, 721).half(); maps = torch.randn(6, 8, 8, 721).half()
    for kind, kw in (("hexresnet", dict(width=16, depth=1)), ("sit", dict(width=16, depth=1, heads=2))):
        mdl = G.build(kind, frames=8, **kw)
        v = mdl(vids[:2].float(), torch.tensor([0.1, 0.9]), maps[:2].float(), torch.ones(2, 8))
        assert v.shape == (2, 8, 721) and torch.isfinite(v).all()
        r = G.train(mdl, vids, maps, steps=3, batch=2, lr=1e-3, warmup=1, log_every=3, log=lambda s: None)
        assert r["history"][-1]["step"] == 3
        x = G.sample(mdl, maps[:2].float(), torch.ones(2, 8), steps=2, guidance=2.0)
        assert x.shape == (2, 8, 721) and x.min() >= 0 and x.max() <= 1


def test_baseline17_novelty_metrics():
    from flydream.generate import baseline17 as B

    rng = np.random.default_rng(0)
    bank_v = rng.random((5, 6, 721)).astype(np.float32)
    q = np.stack([bank_v[2], bank_v[4] + 0.001 * rng.random((6, 721))])      # a copy and a near-copy
    bank, query = B.flat(bank_v, 4), B.flat(q, 4)
    idx, r, d = B.nearest(query, bank)
    assert idx.tolist() == [2, 4]
    assert r[0] > 0.999 and d[0] < 1e-9                                      # an exact copy: r = 1, distance 0
    assert r[1] > 0.99 and 0 < d[1] < 0.01
    skipped = B.nearest(query, bank, skip=idx)[0]                            # leave-one-out picks something else
    assert (skipped != idx).all()
    c = B.pairwise_corr(bank)
    assert c.shape == (5, 5) and np.allclose(np.diag(c), 1.0) and np.allclose(c, c.T)
    t = B.triu_mean(c)
    assert t["median"] <= t["max"] and -1 <= t["mean"] <= 1
    assert np.allclose(B.flat(bank_v, 4).mean(1), 0, atol=1e-5)              # rows are centred
    assert B.flat(bank_v, 4).shape == (5, 4 * 721)


def test_prior17_state_flow_shapes_and_training():
    import torch
    from flydream.generate import prior17 as R

    model = R.build(frames=6, k=3, width=16, depth=1, heads=2)
    x = torch.randn(2, 6, 3, 721)
    v = model(x, torch.tensor([0.2, 0.8]))
    assert v.shape == x.shape and torch.isfinite(v).all()
    assert torch.allclose(v, torch.zeros_like(v))                         # adaLN-Zero: the fresh model outputs zero
    states = torch.randn(8, 6, 3, 721).half()
    r = R.train(model, states, steps=3, batch=2, lr=1e-3, warmup=1, log_every=3, log=lambda s: None,
                val=states[:2], val_every=3)
    assert r["history"][-1]["step"] == 3 and "val_loss" in r["history"][-1]
    g = torch.Generator().manual_seed(0)
    s = R.sample(model, 2, frames=6, k=3, steps=2, generator=g)
    assert s.shape == (2, 6, 3, 721) and torch.isfinite(s).all()
    s2 = R.sample(model, 2, frames=6, k=3, steps=2, generator=torch.Generator().manual_seed(0))
    assert torch.allclose(s, s2)                                          # same z, same state


def test_prior17_map_state_roundtrip():
    from flydream.generate import learned as L
    from flydream.generate import prior17 as R

    manifest = {"type_of_cell": ["T4a"] * 4 + ["T4b"] * 4}
    columns = {"column_of_cell": {"T4a": [0, 1, 2, 3], "T4b": [3, 2, 1, 0]}}
    layout = L.channel_layout(manifest, columns, ["T4a", "T4b"])
    st = np.arange(2 * 3 * 8, dtype=np.float16).reshape(2, 3, 8)
    maps = L.to_maps(st, layout, 2)
    assert maps.shape == (2, 3, 2, 721)
    back = R.from_maps(maps, layout, 8)
    assert np.allclose(back, st.astype(np.float32))                       # the columnar scatter is a bijection
    mean = np.array([1.0, -2.0], np.float32); std = np.array([2.0, 0.5], np.float32)
    raw = R.unscale((maps - mean[None, None, :, None]) / std[None, None, :, None], mean, std)
    assert np.allclose(raw, maps.astype(np.float32), atol=1e-4)


def test_prior17_dct_matrix_and_sample_states():
    import torch
    from flydream.generate import prior17 as R

    d40 = R.dct_matrix(40)
    assert d40.shape == (40, 40) and np.allclose(d40 @ d40.T, np.eye(40), atol=1e-5)   # orthonormal
    x = np.random.default_rng(0).standard_normal((2, 40, 3, 5)).astype(np.float32)
    assert np.allclose(R.from_dct(R.to_dct(x, d40), d40), x, atol=1e-4)                # full basis is lossless
    d16 = R.dct_matrix(40, 16)
    band = R.from_dct(R.to_dct(x, d16), d16)
    assert band.shape == x.shape and np.abs(band - x).mean() > 0                       # truncation loses something
    assert np.allclose(R.to_dct(band, d16), R.to_dct(x, d16), atol=1e-4)               # ...only above K
    model = R.build(frames=16, k=3, width=16, depth=1, heads=2)
    meta = {"frames": 16, "k": 3, "dct_k": 16, "time_frames": 40,
            "coef_mean": np.zeros((16, 3)).tolist(), "coef_std": np.ones((16, 3)).tolist()}
    s = R.sample_states(model, meta, 2, steps=2, generator=torch.Generator().manual_seed(0))
    assert s.shape == (2, 40, 3, 721) and np.isfinite(s).all()                         # back in frames


def test_prior17_refine_endpoints():
    import torch
    from flydream.generate import prior17 as R

    model = R.build(frames=6, k=2, width=16, depth=1, heads=2)
    meta = {"frames": 6, "k": 2, "dct_k": 0}
    st = np.random.default_rng(0).standard_normal((2, 6, 2, 721)).astype(np.float32)
    same = R.refine(model, meta, st, t0=1.0, steps=20, generator=torch.Generator().manual_seed(0))
    assert np.allclose(same, st, atol=1e-5)                      # t0 = 1: the state comes back untouched
    part = R.refine(model, meta, st, t0=0.5, steps=20, generator=torch.Generator().manual_seed(0))
    assert part.shape == st.shape and np.isfinite(part).all() and not np.allclose(part, st)


def test_prior17_noise_inversion_and_slerp():
    import torch
    from flydream.generate import prior17 as R

    torch.manual_seed(0)
    model = R.build(frames=6, k=2, width=16, depth=1, heads=2)
    R.train(model, torch.randn(8, 6, 2, 721).half(), steps=5, batch=4, lr=1e-2, warmup=1,
            log_every=5, log=lambda s: None)                                   # a flow that actually moves a point
    meta = {"frames": 6, "k": 2, "dct_k": 0}
    x0 = torch.randn(2, 6, 2, 721, generator=torch.Generator().manual_seed(0))
    x1 = R.integrate(model, x0.clone(), steps=8)
    assert not torch.allclose(x1, x0, atol=1e-4)
    def rel(a, b):
        return float((a - b).norm() / b.norm())

    back = R.invert(model, x1.clone(), steps=8, fixed_point=3)
    assert rel(back, x0) < 1e-3                                                # noise -> state -> the same noise
    assert rel(R.invert(model, x1.clone(), steps=8, fixed_point=1), x0) > rel(back, x0)   # the fixed point earns it
    st = R.from_noise(model, meta, R.to_noise(model, meta, x1.numpy(), steps=8), steps=8)
    assert rel(torch.as_tensor(st), x1) < 1e-3                                 # state -> noise -> the same state
    rng = np.random.default_rng(0)
    a = rng.standard_normal((1, 6, 2, 721)).astype(np.float32)
    b = rng.standard_normal((1, 6, 2, 721)).astype(np.float32)
    assert np.allclose(R.slerp(a, b, 0.0), a, atol=1e-4) and np.allclose(R.slerp(a, b, 1.0), b, atol=1e-4)
    mid = R.slerp(a, b, 0.5)
    assert abs(np.linalg.norm(mid) - np.linalg.norm(a)) / np.linalg.norm(a) < 0.05   # the radius survives the mix


def test_prior17_class_conditioning_changes_the_sample():
    """18.4e: a label reaches the sample, and the unconditional path is untouched."""
    import numpy as np
    import torch

    from flydream.generate import prior17 as R

    torch.manual_seed(3)
    T, K, C = 4, 3, 5
    x = torch.randn(24, T, K, 721) * 0.5
    y = torch.randint(0, C, (24,))
    model = R.build(frames=T, k=K, width=32, depth=2, heads=2, n_classes=C)
    assert model.y_emb is not None and model.y_emb.num_embeddings == C
    r = R.train(model, x, steps=8, batch=8, lr=1e-2, warmup=2, amp=False, log_every=99, log=lambda s: None,
                val=x[:8], val_every=8, labels=y, val_labels=y[:8])
    assert np.isfinite(r["history"][-1]["val_loss"])                           # the label reaches validation too

    def draw(label):                                                           # the same noise, a different label
        return R.sample(model, 2, frames=T, k=K, columns=721, steps=3,
                        generator=torch.Generator().manual_seed(7),
                        y=torch.full((2,), label, dtype=torch.long))

    assert float((draw(0) - draw(C - 1)).abs().mean()) > 1e-6
    meta = {"frames": T, "k": K, "dct_k": 0, "n_classes": C}
    g = torch.Generator().manual_seed(1)
    s = R.sample_states(model, meta, 4, steps=3, device=torch.device("cpu"), generator=g,
                        y=torch.randint(0, C, (4,), generator=g))
    assert s.shape == (4, T, K, 721) and np.isfinite(s).all()
    try:                                                                       # a conditional prior without a label
        R.sample_states(model, meta, 2, steps=2, device=torch.device("cpu"))
        raise AssertionError("a conditional prior must refuse to sample without a label")
    except ValueError:
        pass

    plain = R.build(frames=T, k=K, width=32, depth=2, heads=2)                 # and the old path still works
    assert plain.y_emb is None
    R.train(plain, x, steps=4, batch=8, lr=1e-2, warmup=2, amp=False, log_every=99, log=lambda s: None)
    assert R.sample_states(plain, {"frames": T, "k": K, "dct_k": 0}, 2, steps=3,
                           device=torch.device("cpu")).shape == (2, T, K, 721)
