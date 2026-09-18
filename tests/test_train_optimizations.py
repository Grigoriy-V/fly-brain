"""Offline equivalence on a synthetic connectome; no training/data/Modal calls."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from flydream.train.optimizations import activity_stats, adapted_train, optimized_network, optimized_solver
from flydream.train.benchmark import compare_states, plan, settings


@pytest.fixture
def network(tmp_path):
    from flydream.model import patch_datamate_for_windows
    patch_datamate_for_windows()
    from datamate import set_root_context
    from flyvis import Network
    from flyvis.utils.config_utils import Namespace
    torch.manual_seed(11)
    fixture = Path(__file__).parent / "fixtures" / "mini_connectome.json"
    with set_root_context(tmp_path):
        net = Network(connectome=Namespace(type="ConnectomeFromAvgFilters", file=str(fixture),
                                           extent=2, n_syn_fill=0))
    net.train()
    with torch.no_grad():
        net.node_params["bias"].raw_values[:] = torch.linspace(-0.4, 1.2, len(net.node_params["bias"].raw_values))
        net.node_params["time_const"].raw_values[:] = 0.06
        net.edge_params["syn_strength"].raw_values[:] = 0.04
    return net


@pytest.mark.parametrize("frames", [1, 40])
def test_node_relu_matches_reference_trajectory_and_all_parameter_gradients(network, frames):
    net = network
    torch.manual_seed(7)
    data = torch.randn(2, frames, net.n_nodes) * 0.4
    target = torch.randn_like(data)
    def evaluate(enabled):
        net.zero_grad(set_to_none=True)
        x = data.clone().requires_grad_()
        with optimized_network(net, enabled):
            y = net(x, 0.02)
            loss = (y - target).square().mean()
            loss.backward()
        return y.detach(), x.grad.clone(), {n: p.grad.clone() for n, p in net.named_parameters() if p.requires_grad}
    base, opt = evaluate(False), evaluate(True)
    torch.testing.assert_close(base[0], opt[0], rtol=0, atol=0)
    torch.testing.assert_close(base[1], opt[1], rtol=1e-5, atol=1e-7)
    assert base[2].keys() == opt[2].keys()
    assert base[2]
    for name in base[2]:
        torch.testing.assert_close(base[2][name], opt[2][name], rtol=1e-5, atol=1e-7)
    assert "write_state_velocity" not in net.dynamics.__dict__


def test_stats_detach_and_match_reference_with_nonfinite_values():
    for values in ([1., -3., 0., 7.], [1., float("nan")], [1., float("inf")]):
        x = torch.tensor(values, requires_grad=True)
        got = torch.stack(activity_stats(x))
        ref = torch.stack((x.detach().cpu().mean(), x.detach().cpu().min(), x.detach().cpu().max()))
        torch.testing.assert_close(got, ref, equal_nan=True)
        assert not got.requires_grad


def test_solver_adapter_only_changes_diagnostics_and_profile_boundary():
    from flyvis.solver import MultiTaskSolver
    # Compiles the actual installed method, fails if the guarded block drifts.
    result = adapted_train(MultiTaskSolver.train, stats=True, profiler=SimpleNamespace(step=lambda: None))
    assert result.__name__ == "train"
    assert "_flydream_activity_stats" in result.__globals__


def test_solver_restores_instance_after_exception(network):
    from flyvis.solver import MultiTaskSolver
    solver = object.__new__(MultiTaskSolver)
    solver.network = network
    with pytest.raises(RuntimeError, match="intentional"):
        with optimized_solver(solver, "stats_relu"):
            assert "train" in solver.__dict__
            raise RuntimeError("intentional")
    assert "train" not in solver.__dict__
    assert "write_state_velocity" not in network.dynamics.__dict__


def test_benchmark_plan_separates_profile_and_reverses_order():
    s = settings(Path(__file__).parents[1] / "config.toml")
    jobs = plan(s)
    assert [j["variant"] for j in jobs[:4]] == list(reversed([j["variant"] for j in jobs[4:8]]))
    assert sum(j["profile"] for j in jobs) == 1
    assert jobs[-1]["profile"]


def test_state_comparison_detects_numerical_and_structure_drift():
    a = {"network": {"w": torch.tensor([1., 2.])}}
    b = {"network": {"w": torch.tensor([1., 3.])}}
    assert compare_states(a, a, 0, 0)["close"]
    assert compare_states(a, b, 1e-4, 1e-6)["max_abs"] == 1
    assert not compare_states(a, b, 1e-4, 1e-6)["close"]
    with pytest.raises(ValueError):
        compare_states(a, {"network": {}}, 0, 0)


def test_member_timing_uses_actual_iteration_delta_including_resume(monkeypatch, tmp_path):
    from flydream.train import member
    import flyvis.solver
    class FakeSolver:
        def __init__(self, **kwargs):
            self.iteration = 0
            self.network = SimpleNamespace(dynamics=SimpleNamespace(), n_nodes=5, n_edges=8)
            self.task = SimpleNamespace(train_data=[1])
            self.dir = SimpleNamespace(path=tmp_path)
        def recover(self, **kwargs):
            self.iteration = 7
        def train(self, *args):
            self.iteration += 4  # requested end=10, completes its epoch at 11
    monkeypatch.setattr(flyvis.solver, "MultiTaskSolver", FakeSolver)
    monkeypatch.setattr(member, "compose", lambda *args: {})
    monkeypatch.setattr(member, "_fit_diagnostic_loaders", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    ticks = iter([10., 12.])
    monkeypatch.setattr(member.time, "perf_counter", lambda: next(ticks))
    got = member.train_member("unused.json", "test", 10, str(tmp_path), resume=True)
    assert got["iteration_start"] == 7
    assert got["iteration_end"] == 11
    assert got["completed_iters"] == 4
    assert got["s_per_iter"] == .5


def test_adapted_reference_loop_preserves_updates_penalties_and_checkpoints():
    from contextlib import nullcontext
    from flyvis.solver import MultiTaskSolver
    from flyvis.utils.config_utils import Namespace
    class Buffer:
        def zero(self, *args):
            self.value = None
        def add_input(self, x):
            self.value = x
        def __call__(self):
            return self.value
    class MiniNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(.7))
            self.stimulus = Buffer()
            self.dynamics = SimpleNamespace()
        def steady_state(self, **kwargs):
            return None
        def forward(self, x, dt, state=None):
            return self.weight * x
    class Batches(list):
        batch_size = 2
    class Record(SimpleNamespace):
        def __contains__(self, key):
            return key in self.__dict__
        def __setitem__(self, key, value):
            setattr(self, key, value)
    class MiniSolver(MultiTaskSolver):
        def __init__(self):
            self.network = MiniNet()
            self.decoder = {"lum": torch.nn.Identity()}
            self.optimizer = torch.optim.SGD(self.network.parameters(), lr=.01)
            self.iteration = 0
            self.dir = Record()
            self.config = Namespace(scheduler=Namespace(chkpt_every_epoch=1))
            self.checkpoints_seen, self.penalties_seen, self.scheduled = [], [], []
            data = Batches([{"lum": torch.arange(8.).reshape(2, 2, 1, 2) / 8},
                            {"lum": -torch.ones(2, 2, 1, 2)}])
            self.task = SimpleNamespace(n_iters=3, train_data=data,
                dataset=SimpleNamespace(tasks=["lum"], dt=.02, augmentation=lambda _: nullcontext()),
                loss=lambda a, b, task, **kw: (a - b).square().mean())
            self.penalty = lambda activity, iteration: self.penalties_seen.append(iteration)
            self.scheduler = self.scheduled.append
        def checkpoint(self):
            self.checkpoints_seen.append(self.iteration)
    baseline, candidate = MiniSolver(), MiniSolver()
    baseline.train()
    steps = []
    with optimized_solver(candidate, "stats", SimpleNamespace(step=lambda: steps.append(1))):
        candidate.train()
    torch.testing.assert_close(candidate.network.weight, baseline.network.weight, rtol=0, atol=0)
    assert candidate.iteration == baseline.iteration == len(steps) == 4
    assert candidate.checkpoints_seen == baseline.checkpoints_seen == [0, 2, 4]
    assert candidate.penalties_seen == baseline.penalties_seen == [0, 1, 2, 3]
    assert candidate.scheduled == baseline.scheduled
    for name in ("loss", "activity", "activity_min", "activity_max"):
        assert getattr(candidate.dir, name) == getattr(baseline.dir, name)
