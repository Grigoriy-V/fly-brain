"""The parameter transplant copies by key, rescales the gains, and defaults to the median.

No flyvis network is built here: the two networks are stand-ins carrying only
the parameter tables and the edge table `transplant` reads.
"""
import types

import numpy as np
import pytest
import torch

from flydream.model.zero import mean_n_syn, shuffle_strengths, total_n_syn, transplant


class _Param:
    def __init__(self, keys, values):
        self.keys = list(keys)
        self.raw_values = torch.nn.Parameter(torch.tensor(values, dtype=torch.float32))


class _Col:
    """An h5-like column: `col[:]` returns the array."""

    def __init__(self, values, dtype=None):
        self._v = np.array(values, dtype=dtype)

    def __getitem__(self, item):
        return self._v[item]

    def astype(self, dt):
        return self._v.astype(dt)


def _net(bias, tc, strength, edges=None):
    """edges: (source_type, target_type, n_syn[, target_index]); target_index
    defaults to the edge's position, i.e. one target cell per edge."""
    n = types.SimpleNamespace()
    n.node_params = {"bias": _Param(*bias), "time_const": _Param(*tc)}
    n.edge_params = {"syn_strength": _Param(*strength)}
    if edges is not None:
        src = [e[0] for e in edges]
        tar = [e[1] for e in edges]
        n_syn = [e[2] for e in edges]
        ti = [e[3] if len(e) > 3 else i for i, e in enumerate(edges)]
        n.connectome = types.SimpleNamespace(edges=types.SimpleNamespace(
            source_type=_Col(src, dtype=object), target_type=_Col(tar, dtype=object),
            n_syn=_Col(n_syn, dtype=float), target_index=_Col(ti, dtype=np.int64)))
    return n


def test_transplant_copies_matching_keys_and_defaults_others_to_median():
    """rescale=False is the pre-2026-09-18 behaviour, kept so the scaling error can
    be reproduced and measured against; it is not the default."""
    src = _net((["A", "B", "C"], [1.0, 2.0, 9.0]), (["A", "B", "C"], [0.1, 0.2, 0.3]),
               ([("A", "B"), ("B", "C"), ("C", "A")], [0.5, 1.5, 2.5]))
    dst = _net((["B", "Z", "A"], [0.0, 0.0, 0.0]), (["B", "Z", "A"], [0.0, 0.0, 0.0]),
               ([("B", "C"), ("Z", "A"), ("A", "B")], [0.0, 0.0, 0.0]))
    rep = transplant(src, dst, rescale=False)
    assert dst.node_params["bias"].raw_values.tolist() == [2.0, 2.0, 1.0]      # Z -> median of (1, 2, 9)
    assert torch.allclose(dst.node_params["time_const"].raw_values, torch.tensor([0.2, 0.2, 0.1]))
    assert dst.edge_params["syn_strength"].raw_values.tolist() == [1.5, 1.5, 0.5]
    assert rep["bias"] == {"matched": 2, "defaulted": 1}
    assert rep["syn_strength"]["matched"] == 2 and rep["syn_strength"]["rescaled"] == 0


def test_mean_n_syn_averages_the_edges_of_each_type_pair():
    net = _net((["A"], [0.0]), (["A"], [0.0]), ([("A", "B")], [0.0]),
               edges=[("A", "B", 10.0), ("A", "B", 20.0), ("B", "C", 4.0)])
    got = mean_n_syn(net)
    assert got[("A", "B")] == pytest.approx(15.0)
    assert got[("B", "C")] == pytest.approx(4.0)


def test_total_n_syn_sums_over_offsets_onto_the_same_target_cell():
    """Why the mean per edge was the wrong basis: two connectomes can give a
    target cell the same total input spread over a different number of offsets.
    Here cell 0 of type B receives 30 synapses from A either as one edge or as
    three; total_n_syn sees 30 both ways, mean_n_syn sees 30 against 10."""
    one = _net((["A"], [0.0]), (["A"], [0.0]), ([("A", "B")], [0.0]),
               edges=[("A", "B", 30.0, 0)])
    three = _net((["A"], [0.0]), (["A"], [0.0]), ([("A", "B")], [0.0]),
                 edges=[("A", "B", 20.0, 0), ("A", "B", 6.0, 0), ("A", "B", 4.0, 0)])
    assert total_n_syn(one)[("A", "B")] == pytest.approx(30.0)
    assert total_n_syn(three)[("A", "B")] == pytest.approx(30.0)
    assert mean_n_syn(three)[("A", "B")] == pytest.approx(10.0)
    # and it averages over target cells, not over edges
    two_cells = _net((["A"], [0.0]), (["A"], [0.0]), ([("A", "B")], [0.0]),
                     edges=[("A", "B", 30.0, 0), ("A", "B", 10.0, 1)])
    assert total_n_syn(two_cells)[("A", "B")] == pytest.approx(20.0)


def test_the_gain_is_rescaled_by_the_ratio_of_total_input_per_target_cell():
    """FlyVis stores syn_strength as `scale / <n_syn>` for the pair in ITS OWN
    connectome and builds the weight as `sign * n_syn * syn_strength`, so the raw
    number is not portable. Transplanting `syn_strength * total_src / total_dst`
    preserves the total input a target cell receives from the source type."""
    src = _net((["A"], [1.0]), (["A"], [0.1]), ([("A", "B"), ("B", "C")], [0.5, 1.0]),
               edges=[("A", "B", 12.0, 0), ("B", "C", 2.0, 0)])
    dst = _net((["A"], [0.0]), (["A"], [0.0]), ([("A", "B"), ("B", "C")], [0.0, 0.0]),
               edges=[("A", "B", 6.0, 0), ("B", "C", 4.0, 0)])
    rep = transplant(src, dst, rescale=True, rescale_cap=None)
    got = dst.edge_params["syn_strength"].raw_values.tolist()
    assert got[0] == pytest.approx(0.5 * 12.0 / 6.0)      # underweighted 2x without the fix
    assert got[1] == pytest.approx(1.0 * 2.0 / 4.0)       # overweighted 2x without the fix
    assert rep["syn_strength"]["rescaled"] == 2 and rep["syn_strength"]["capped"] == 0


def test_rescaling_preserves_the_total_input_of_a_target_cell():
    """The invariant the fix exists for: weight = sign * n_syn * syn_strength, so
    a target cell's total input from the source type must survive the move."""
    src = _net((["A"], [1.0]), (["A"], [0.1]), ([("A", "B")], [0.02]),
               edges=[("A", "B", 25.0, 0)])
    dst = _net((["A"], [0.0]), (["A"], [0.0]), ([("A", "B")], [0.0]),
               edges=[("A", "B", 5.0, 0), ("A", "B", 2.0, 0)])        # 7 synapses over two offsets
    transplant(src, dst, rescale=True, rescale_cap=None)
    w_src = 25.0 * 0.02
    w_dst = 7.0 * dst.edge_params["syn_strength"].raw_values[0].item()
    assert w_dst == pytest.approx(w_src, rel=1e-5)


def test_the_cap_bounds_the_factor_and_names_the_pair():
    """A thirtyfold wiring difference is an export defect, not a gain to apply:
    the factor is clipped to [1/cap, cap] and the pair is reported so the
    export can be fixed at the source."""
    src = _net((["A"], [1.0]), (["A"], [0.1]), ([("A", "B"), ("C", "D")], [1.0, 1.0]),
               edges=[("A", "B", 36.0, 0), ("C", "D", 1.0, 0)])
    dst = _net((["A"], [0.0]), (["A"], [0.0]), ([("A", "B"), ("C", "D")], [0.0, 0.0]),
               edges=[("A", "B", 1.2, 0), ("C", "D", 10.0, 0)])
    rep = transplant(src, dst, rescale=True, rescale_cap=3.0)
    got = dst.edge_params["syn_strength"].raw_values.tolist()
    assert got[0] == pytest.approx(3.0)                   # 30x asked, 3x applied
    assert got[1] == pytest.approx(1.0 / 3.0)             # 0.1x asked, 1/3 applied
    assert rep["syn_strength"]["capped"] == 2
    pairs = {tuple(c["pair"]): c["factor"] for c in rep["syn_strength"]["capped_pairs"]}
    assert pairs[("A", "B")] == pytest.approx(30.0)
    assert pairs[("C", "D")] == pytest.approx(0.1)
    assert rep["syn_strength"]["factor_max_applied"] == pytest.approx(3.0)


def test_shuffle_keeps_the_multiset_of_strengths():
    net = _net((["A"], [0.0]), (["A"], [0.0]), ([("A", "B"), ("B", "C"), ("C", "A"), ("A", "C")], [1.0, 2.0, 3.0, 4.0]))
    before = sorted(net.edge_params["syn_strength"].raw_values.tolist())
    shuffle_strengths(net, seed=1)
    after = net.edge_params["syn_strength"].raw_values.tolist()
    assert sorted(after) == before
