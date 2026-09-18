"""Paired (rendered stimulus, network activity) samples: the input to every decoder.

The rendered 721-hexal stimulus is the decoder's target, not the source pixels
(DECISIONS, 2026-09-18): the photoreceptors never saw the pixels, and a decoder
scored against them would be scored partly on the renderer. It comes back for
free as the first element of every `Network.stimulus_response` yield, frame
aligned 1:1 with the activity, because the pre-stimulus and fade-in frames go
into the initial state and never into the output.

Activity is kept raw. Nothing here divides by flyvis's per-cell-type
normalisation constants: see `ridge.py` for why, and note in passing that
flyvis's own trained decoder head does not use them either, it feeds rectified
raw voltages into a BatchNorm and learns the scale.

Samples are grouped by Sintel scene, so a split can hold out whole scenes. A
row-level random split leaks: the same scene reappears as three vertical by N
temporal by up to twelve geometric variants of itself, and a decoder tested on
one variant of a scene it trained on is being asked an easier question than the
map claims to answer.
"""
from __future__ import annotations

import ast
import pathlib
import re
from dataclasses import dataclass

import numpy as np


@dataclass
class Pairs:
    """Stimulus and activity for a set of samples, plus what is needed to split them."""

    stimulus: np.ndarray            # (S, T, H) float32, the rendered hexal input
    activity: np.ndarray            # (S, T, N) float32, the kept cells of the network
    groups: np.ndarray              # (S,) a label to hold out whole (e.g. a Sintel scene)
    cell_types: list[str]
    index: dict[str, np.ndarray]    # cell type -> indices into the N axis
    flow: np.ndarray | None = None  # (S, T, 2, H) float32 when the dataset has it
    meta: dict | None = None

    @property
    def n_samples(self) -> int:
        return self.stimulus.shape[0]

    @property
    def n_frames(self) -> int:
        return self.stimulus.shape[1]

    def by_type(self, cell_type: str) -> np.ndarray:
        """(S, T, n_cells_of_type). 63 of the 65 flyvis types have one cell per
        column; Lawf1 and Lawf2 are strided and come back with 123."""
        return self.activity[:, :, self.index[cell_type]]

    def frames(self, cell_type: str, sample_index: np.ndarray, lags=(0,)):
        """Flatten chosen samples into (rows, features), (rows, targets), (rows,) groups.

        A lag is an offset in frames added to the stimulus frame when selecting
        activity: lag 0 is the per-frame decoder, which is the shape of flyvis's
        own decoder head and therefore the honest baseline; a positive lag lets
        the decoder see the response after the stimulus frame, which a membrane
        with a time constant needs. Frames without every lag available are
        dropped from both sides.
        """
        lags = tuple(int(lag) for lag in lags)
        lo, hi = min(lags), max(lags)
        t0, t1 = max(0, -lo), self.n_frames - max(0, hi)
        if t1 <= t0:
            raise ValueError(f"lags {lags} leave no frames of {self.n_frames}")
        a = self.by_type(cell_type)[sample_index]
        y = self.stimulus[sample_index][:, t0:t1]
        x = np.concatenate([a[:, t0 + lag:t1 + lag] for lag in lags], axis=-1)
        rows = x.shape[0] * x.shape[1]
        return (x.reshape(rows, x.shape[-1]).astype(np.float64),
                y.reshape(rows, y.shape[-1]).astype(np.float64),
                np.repeat(self.groups[sample_index], t1 - t0))

    def save(self, path) -> pathlib.Path:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(self.index)
        np.savez_compressed(
            path,
            stimulus=self.stimulus,
            activity=self.activity,
            groups=self.groups.astype("U64"),
            cell_types=np.array(self.cell_types, dtype="U32"),
            index_keys=np.array(keys, dtype="U32"),
            index_values=np.concatenate([self.index[k] for k in keys]) if keys else np.zeros(0, np.int64),
            index_sizes=np.array([len(self.index[k]) for k in keys], dtype=np.int64),
            flow=self.flow if self.flow is not None else np.zeros(0, np.float32),
            meta=np.array(repr(self.meta or {}), dtype="U8192"),
        )
        return path

    @classmethod
    def load(cls, path) -> "Pairs":
        z = np.load(path, allow_pickle=False)
        keys = [str(k) for k in z["index_keys"]]
        bounds = np.concatenate([[0], np.cumsum(z["index_sizes"])])
        index = {k: z["index_values"][bounds[i]:bounds[i + 1]] for i, k in enumerate(keys)}
        flow = z["flow"]
        return cls(stimulus=z["stimulus"], activity=z["activity"], groups=z["groups"],
                   cell_types=[str(c) for c in z["cell_types"]], index=index,
                   flow=None if flow.size == 0 else flow,
                   meta=ast.literal_eval(str(z["meta"])))


def type_index(connectome) -> tuple[list[str], dict[str, np.ndarray]]:
    """Cell types and their node indices, in the connectome's own order."""
    types = [t.decode() if isinstance(t, bytes) else str(t)
             for t in connectome.unique_cell_types[:]]
    return types, {t: np.asarray(connectome.nodes.layer_index[t][:]) for t in types}


def scene_labels(dataset, indices: np.ndarray) -> np.ndarray:
    """One label per sample to hold out whole: the Sintel scene where there is one.

    `AugmentedSintel.arg_df.name` looks like `sequence_00_alley_1_split_00`; the
    scene is what survives stripping the sequence number and the split suffix.
    A dataset without that column falls back to one group per sample, which
    makes the split row-level; that is only honest for synthetic stimuli, where
    each sample is an independent condition rather than a crop of a neighbour.
    """
    df = getattr(dataset, "arg_df", None)
    if df is None or "name" not in getattr(df, "columns", []):
        return np.array([f"sample_{int(i)}" for i in indices], dtype="U64")
    out = []
    for i in indices:
        n = str(df.name.iloc[int(i)])
        s = re.sub(r"^sequence_\d+_", "", n.split("_split_")[0])
        out.append(s or n)
    return np.array(out, dtype="U64")


def build(network, dataset, *, dt: float, indices=None, batch_size: int = 4,
          t_pre: float = 0.0, t_fade_in: float = 2.0, stim_key: str = "lum",
          keep_types: list[str] | None = None, verbose: bool = True) -> Pairs:
    """Simulate `dataset` through `network` and return the paired samples.

    `keep_types` restricts what is stored: the activity of all 45,669 cells is
    about 7.3 MB per sample per 40 frames, so a hundred samples is 0.7 GB.
    """
    idx = np.arange(len(dataset)) if indices is None else np.asarray(indices)
    types, index = type_index(network.connectome)
    order = None
    if keep_types:
        missing = [t for t in keep_types if t not in index]
        if missing:
            raise KeyError(f"not cell types of this connectome: {missing}")
        order = np.concatenate([index[t] for t in keep_types])
        remap, pos = {}, 0
        for t in keep_types:
            n = len(index[t])
            remap[t] = np.arange(pos, pos + n)
            pos += n
        types, index = list(keep_types), remap

    stims, acts, seen = [], [], 0
    for stim, resp in network.stimulus_response(
            dataset, dt=dt, indices=idx, t_pre=t_pre, t_fade_in=t_fade_in,
            batch_size=batch_size, default_stim_key=stim_key):
        s = np.asarray(stim, dtype=np.float32)
        s = s.reshape(s.shape[0], s.shape[1], -1)        # (B, T, 1, H) -> (B, T, H)
        r = np.asarray(resp, dtype=np.float32)
        if order is not None:
            r = r[:, :, order]
        stims.append(s)
        acts.append(r)
        seen += s.shape[0]
        if verbose:
            print(f"  {seen}/{len(idx)} samples", flush=True)

    stimulus = np.concatenate(stims, axis=0)
    activity = np.concatenate(acts, axis=0)

    flow = None
    if stim_key == "lum":
        try:
            flow = np.stack([np.asarray(dataset[int(i)]["flow"], dtype=np.float32) for i in idx])
        except (KeyError, IndexError, TypeError):
            flow = None

    meta = {"dt": dt, "t_pre": t_pre, "t_fade_in": t_fade_in, "stim_key": stim_key,
            "n_cells": int(activity.shape[-1]), "dataset": type(dataset).__name__,
            "n_samples": int(stimulus.shape[0]), "n_frames": int(stimulus.shape[1])}
    return Pairs(stimulus=stimulus, activity=activity,
                 groups=scene_labels(dataset, idx), cell_types=types,
                 index=index, flow=flow, meta=meta)


def split_by_group(groups: np.ndarray, test_fraction: float = 0.2, seed: int = 0):
    """Hold out whole groups. Returns (train_index, test_index) over samples."""
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_test = max(1, int(round(test_fraction * len(uniq))))
    held = set(uniq[:n_test].tolist())
    mask = np.array([g in held for g in groups])
    return np.flatnonzero(~mask), np.flatnonzero(mask)
