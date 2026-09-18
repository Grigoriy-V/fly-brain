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
