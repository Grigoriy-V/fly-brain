"""Opt-in FlyVis 1.2.0 experiments; never edit the installed package.

See reports/2026-09-18_training_optimization_bench.md. The solver adapter
replaces exactly the diagnostic reduction block, keeping penalties, scheduling,
checkpointing and the reference training loop. Unexpected upstream source fails
closed. Changes are bound to one instance and restored even on exceptions.
"""
from contextlib import contextmanager
import hashlib
import inspect
import textwrap
from types import MethodType

import torch

VARIANTS = ("baseline", "stats", "relu", "stats_relu")


def activity_stats(activity):
    """Reduce on the activity device; only three detached scalars cross to CPU."""
    a = activity.detach()
    return torch.stack((a.mean(), a.min(), a.max())).cpu().unbind()


def node_relu_velocity(network):
    def velocity(self, vel, state, params, target_sum, x_t, dt, **kwargs):
        # ReLU commutes with gather. Shared source gradients are summed before
        # ReLU backward, so FP summation order can differ from the reference.
        rectified = self.activation(state.nodes.activity)
        current = target_sum(params.edges.weight * rectified.index_select(-1, network._source_indices))
        vel.nodes.activity = (
            1 / torch.max(params.nodes.time_const, torch.tensor(dt).float())
            * (-state.nodes.activity + params.nodes.bias + current + x_t)
        )
    return velocity


_OLD_STATS = """activity = activity.detach().cpu()
                        mean_activity = activity.mean()
                        activity_over_iters.append(mean_activity.item())
                        activity_min_over_iters.append(activity.min().item())
                        activity_max_over_iters.append(activity.max().item())"""
_NEW_STATS = """mean_activity, min_activity, max_activity = _flydream_activity_stats(activity)
                        activity_over_iters.append(mean_activity.item())
                        activity_min_over_iters.append(min_activity.item())
                        activity_max_over_iters.append(max_activity.item())"""


def adapted_train(original, *, stats=False, profiler=None):
    source = textwrap.dedent(inspect.getsource(original))
    # dedent removes the class's four spaces, including from the block.
    old = _OLD_STATS.replace("\n    ", "\n")
    new = _NEW_STATS.replace("\n    ", "\n")
    if stats:
        if source.count(old) != 1:
            raise RuntimeError("FlyVis diagnostic block changed; refusing to patch train")
        source = source.replace(old, new)
    if profiler is not None:
        needle = "self.iteration += 1"
        if source.count(needle) != 1:
            raise RuntimeError("FlyVis iteration boundary changed; refusing profiling adapter")
        source = source.replace(needle, needle + "\n                _flydream_profiler.step()")
    namespace = dict(original.__globals__)
    namespace.update(_flydream_activity_stats=activity_stats, _flydream_profiler=profiler)
    exec(compile(source, "<flydream-flyvis-train-adapter>", "exec"), namespace)
    return namespace[original.__name__]


@contextmanager
def optimized_network(network, enabled=True):
    from flyvis.network.dynamics import PPNeuronIGRSynapses
    dynamics = network.dynamics
    previous = dynamics.__dict__.get("write_state_velocity")
    if enabled:
        if type(dynamics) is not PPNeuronIGRSynapses or type(dynamics.activation) is not torch.nn.ReLU:
            raise ValueError("node-ReLU optimization requires stock PPNeuronIGRSynapses/ReLU")
        dynamics.write_state_velocity = MethodType(node_relu_velocity(network), dynamics)
    try:
        yield
    finally:
        if enabled:
            if previous is None:
                del dynamics.write_state_velocity
            else:
                dynamics.write_state_velocity = previous


@contextmanager
def optimized_solver(solver, variant="baseline", profiler=None):
    from importlib.metadata import version
    if variant not in VARIANTS:
        raise ValueError(f"Unknown optimization variant: {variant}")
    if version("flyvis") != "1.2.0":
        raise RuntimeError("Optimization adapter is verified only for FlyVis 1.2.0")
    original = type(solver).train
    provenance = hashlib.sha256(inspect.getsource(original).encode()).hexdigest()
    previous = solver.__dict__.get("train")
    patch = variant in ("stats", "stats_relu") or profiler is not None
    if patch:
        solver.train = MethodType(adapted_train(original, stats=variant in ("stats", "stats_relu"),
                                               profiler=profiler), solver)
    try:
        with optimized_network(solver.network, variant in ("relu", "stats_relu")):
            yield provenance
    finally:
        if patch:
            if previous is None:
                del solver.train
            else:
                solver.train = previous
