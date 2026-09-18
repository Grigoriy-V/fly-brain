"""Offline: the lag-sweep join, on a synthetic pair of windows."""
import numpy as np
import pandas as pd
import pytest

from flydream.decode import sweep as S


def test_window_length_is_the_reach_of_the_lags_in_ms():
    assert S.window_ms("(0,)") == 0.0
    assert S.window_ms("(0, 2, 4)") == pytest.approx(80.0)          # 4 frames at 20 ms
    assert S.window_ms("(0, 1)", dt=0.01) == pytest.approx(10.0)


def test_stage_assignment_follows_the_hierarchy():
    assert S.stage_of("R7") == "1 photoreceptor"
    assert S.stage_of("Lawf2") == "2 lamina"
    assert S.stage_of("CT1(M10)") == "3 medulla Mi"
    assert S.stage_of("TmY15") == "4 medulla Tm/TmY"
    assert S.stage_of("T5d") == "5 T4/T5 motion"
    assert S.stage_of("LPi34") == "6 lobula/other"


def _table():
    rows = []
    for w, vals in ((0.0, {"L1": 0.8, "T5a": 0.2}), (80.0, {"L1": 0.7, "T5a": 0.75})):
        for t, real in vals.items():
            rows.append({"run": f"w{w:.0f}", "lags": "(0,)" if w == 0 else "(0, 2, 4)", "window_ms": w,
                         "cell_type": t, "stage": S.stage_of(t), "real": real,
                         "time_shuffle": 0.2, "sample_shuffle": 0.1,
                         "frame": real - 0.2, "scene": 0.1})
    return pd.DataFrame(rows)


def test_rank_shift_reports_gain_and_the_change_in_rank():
    """The question the sweep answers: does a type at the bottom at lag 0 rise
    above one at the top when the window opens (ISS-0004)?"""
    rs = S.rank_shift(_table())
    assert rs.loc["T5a", "gain"] == pytest.approx(0.55)
    assert rs.loc["L1", "gain"] == pytest.approx(-0.1)
    assert rs.loc["T5a", "rank_0ms"] == 2 and rs.loc["T5a", "rank_80ms"] == 1
    assert rs.loc["L1", "rank_0ms"] == 1 and rs.loc["L1", "rank_80ms"] == 2
    assert rs.loc["T5a", "rank_change"] == 1          # rose one place
    assert rs.index[0] == "T5a"                        # sorted by gain, largest first


def test_summary_is_a_median_per_stage_and_window():
    s = S.summarise(_table())
    lam = s[(s.stage == "2 lamina") & (s.window_ms == 80.0)].iloc[0]
    assert lam.frame == pytest.approx(0.5) and lam.n == 1
