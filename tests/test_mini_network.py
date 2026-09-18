"""Offline: build a Network on the synthetic miniature connectome and simulate it.

No download, no credential, no Modal: flyvis is installed and the connectome is
`tests/fixtures/mini_connectome.json`. FLYVIS_ROOT_DIR is redirected to a
temporary directory so nothing is written into the project's data/.
"""
import json
import pathlib

import numpy as np
import pytest
import torch

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "mini_connectome.json"
EXTENT = 2          # 19 columns
N_COLUMNS = 19


@pytest.fixture(scope="module")
def net(tmp_path_factory):
    root = tmp_path_factory.mktemp("flyvis_root")
    import os

    os.environ["FLYVIS_ROOT_DIR"] = str(root)
    from flydream.model import patch_datamate_for_windows

    patch_datamate_for_windows()
    from flyvis import Network
    from flyvis.utils.config_utils import Namespace

    n = Network(connectome=Namespace(type="ConnectomeFromAvgFilters", file=str(FIXTURE),
                                     extent=EXTENT, n_syn_fill=0))
    n.eval()
    return n


def test_the_fixture_is_the_shape_the_builder_reads():
    spec = json.loads(FIXTURE.read_text())
    assert {"nodes", "edges", "input_units", "output_units"} <= set(spec)
    for e in spec["edges"]:
        assert {"src", "tar", "alpha", "offsets", "lambda_mult"} <= set(e)   # add_edges reads all five
        assert all(len(o) == 2 and len(o[0]) == 2 for o in e["offsets"])
    names = {n["name"] for n in spec["nodes"]}
    assert set(spec["input_units"]) <= names and set(spec["output_units"]) <= names


def test_network_has_one_cell_per_column_except_the_strided_type(net):
    types = net.connectome.nodes.type[:].astype(str)
    counts = {t: int((types == t).sum()) for t in set(types)}
    assert counts["R1"] == counts["L1"] == counts["Mi1"] == counts["T4a"] == N_COLUMNS
    assert counts["Wide"] < N_COLUMNS          # stride [2, 2] places fewer cells
    assert net.n_nodes == sum(counts.values())


def test_signs_and_offsets_survive_the_build(net):
    e = net.connectome.edges
    src, tar = e.source_type[:].astype(str), e.target_type[:].astype(str)
    sign, du, dv = e.sign[:], e.du[:], e.dv[:]
    m = (src == "R1") & (tar == "L1")
    assert m.sum() == N_COLUMNS and set(sign[m]) == {-1.0}
    m = (src == "Mi1") & (tar == "T4a")
    assert set(sign[m]) == {1.0}
    assert set(zip(du[m].tolist(), dv[m].tolist())) == {(0, 0), (1, 0), (-1, 0)}   # the three offsets, no hull fill


def set_parameters(net, biases, time_const=0.01, strength=0.02):
    """Pin every parameter so a test asserts the machinery, not the random init.

    The biases matter: the ON pathway here is a double inversion
    (R1 -| L1 -| Mi1 -> T4a), so L1 and Mi1 must rest well above zero or the
    ReLU clips the inhibition away and nothing propagates. This is the same
    reason a model with no baseline activity cannot carry an OFF channel.
    """
    import torch

    with torch.no_grad():
        b = net.node_params["bias"]
        for i, k in enumerate(b.keys):
            b.raw_values[i] = biases[k]
        tc = net.node_params["time_const"]
        tc.raw_values[:] = time_const
        net.edge_params["syn_strength"].raw_values[:] = strength


BIASES = {"R1": 1.0, "L1": 6.0, "Mi1": 6.0, "Wide": 0.5, "T4a": 0.5}


def test_simulation_returns_finite_activity_for_every_neuron(net):
    set_parameters(net, BIASES)
    frames = 5
    stim = torch.zeros(1, frames, 1, N_COLUMNS)
    stim[:, 2:, 0, N_COLUMNS // 2] = 1.0            # a flash in the central column
    a = np.asarray(net.simulate(stim, dt=1 / 100, initial_state=None).detach())
    assert a.shape[:2] == (1, frames)
    assert a.shape[-1] == net.n_nodes
    assert np.isfinite(a).all()


def test_a_flash_propagates_along_the_double_inversion_to_the_output_type(net):
    set_parameters(net, BIASES)
    frames = 40
    blank = torch.zeros(1, frames, 1, N_COLUMNS)
    flash = blank.clone()
    flash[:, 10:, 0, :] = 1.0
    types = net.connectome.nodes.type[:].astype(str)
    a_blank = np.asarray(net.simulate(blank, dt=1 / 100, initial_state=None).detach())
    a_flash = np.asarray(net.simulate(flash, dt=1 / 100, initial_state=None).detach())
    moved = {}
    for t in ("R1", "L1", "Mi1", "T4a"):
        m = types == t
        moved[t] = float(np.abs(a_flash[0, -1, m] - a_blank[0, -1, m]).max())
    assert moved["R1"] > 0.1, moved
    assert moved["L1"] > 1e-3, moved          # inhibited, stays above the ReLU floor
    assert moved["Mi1"] > 1e-4, moved         # disinhibited by the drop in L1
    assert moved["T4a"] > 1e-5, moved         # reached the output type
    # the pathway attenuates with depth, which is the ordering the project measures
    assert moved["R1"] > moved["L1"] > moved["Mi1"] > moved["T4a"], moved


def test_an_inhibited_type_without_headroom_blocks_the_pathway(net):
    """The control for the test above: with L1 resting at zero the ReLU clips the
    inhibition and the flash never reaches T4a."""
    set_parameters(net, {**BIASES, "L1": 0.0})
    frames = 40
    blank = torch.zeros(1, frames, 1, N_COLUMNS)
    flash = blank.clone()
    flash[:, 10:, 0, :] = 1.0
    types = net.connectome.nodes.type[:].astype(str)
    a_blank = np.asarray(net.simulate(blank, dt=1 / 100, initial_state=None).detach())
    a_flash = np.asarray(net.simulate(flash, dt=1 / 100, initial_state=None).detach())
    m = types == "T4a"
    assert float(np.abs(a_flash[0, -1, m] - a_blank[0, -1, m]).max()) < 1e-6
