

## summary

flyvis ships exactly one trainable decoder head, `DecoderGAVP` (D:/ML/Fly_Brain/.venv/Lib/site-packages/flyvis/task/decoder.py:190), plus its no-op base `ActivityDecoder` (same file:23) and the hex-masked conv layer it is built from, `Conv2dHexSpace` (same file:112). It is a fully convolutional, purely per-frame head: it scatters 721 hexals into a 31x31 cartesian "map storage" grid, runs 2 hex-masked 5x5 convs with BatchNorm+Softplus+Dropout, optionally divides by a softplus'd extra channel, and gathers back to 721 hexals — no temporal filter, no recurrence, no state (verified: perturbing frames 3..5 leaves output frames 0..2 bit-identical, and a single-frame call equals frame 2 of a batched call). Frames are folded into the batch dimension, so the only cross-frame coupling is BatchNorm's batch statistics during training. The shipped flow head (config in D:/ML/Fly_Brain/data/flyvis/results/flow/0000/000/_meta.yaml: `type: DecoderGAVP, shape: [8, 2], kernel_size: 5, const_weight: 0.001, n_out_features: null, p_dropout: 0.5`) has 7427 free parameters, verified by instantiating it and by loading `best_chkpt`. `Conv2dHexSpace` is a plain `nn.Conv2d` subclass and works standalone on any (in_channels, out_channels, odd kernel_size) — I ran it as 7->1 and built a k->1 head for k in {1, 8, 34}. The one real coupling to the Task machinery is that `DecoderGAVP.__init__` reads `in_channels = len(connectome.output_cell_types)` and `forward` expects the FULL network activity `(n_samples, n_frames, n_cells)` which it slices by cell type — both are bypassed either with a ~6-line connectome proxy that redefines `output_cell_types` (reuses the shipped class unmodified, verified for 1, 3, 4 and 8 cell types) or by reusing only `Conv2dHexSpace` + `get_hex_coords` and writing the 12-line forward.

## code_example

"""k channels of 721 hexals -> 1 channel of 721 hexals, on flyvis's shipped parts.
Both variants below were run end to end; printed output is at the bottom."""
import os, pathlib, tomllib
ROOT = pathlib.Path("D:/ML/Fly_Brain")
os.environ.setdefault(
    "FLYVIS_ROOT_DIR",
    str(ROOT / tomllib.loads((ROOT / "config.toml").read_text())["flyvis"]["root_dir"]),
)
from flydream.model import patch_datamate_for_windows  # noqa: F401  (must precede flyvis)

import numpy as np
import torch
import torch.nn.functional as nnf
from torch import nn

import flyvis
from flyvis.connectome import ConnectomeFromAvgFilters
from flyvis.task.decoder import Conv2dHexSpace, DecoderGAVP
from flyvis.utils.hex_utils import get_hex_coords
from flyvis.utils.nn_utils import n_params

connectome = ConnectomeFromAvgFilters(file="fib25-fib19_v2.2.json", extent=15, n_syn_fill=1)

# ------------------------------------------------- A: reuse DecoderGAVP unmodified
class OutputTypeView:
    """A connectome that reports a chosen set of cell types as the 'output' types.

    DecoderGAVP reads in_channels off len(connectome.output_cell_types) and slices
    those cells out of the full activity, so this is all it takes to pick k channels
    (k == 1 for a single cell type). Every selected type must have the same cell
    count (721 at extent 15) -- see the Lawf1/Lawf2 gotcha."""

    def __init__(self, connectome, output_cell_types):
        self._connectome = connectome
        self.output_cell_types = np.bytes_(list(output_cell_types))

    def __getattr__(self, name):
        return getattr(self._connectome, name)


cell_types = ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"]   # k = 8
decoder = DecoderGAVP(
    OutputTypeView(connectome, cell_types),
    shape=[8, 1],          # [hidden widths..., out_channels]; out_channels = 1
    kernel_size=5,
    const_weight=0.001,    # pass None for standard torch init
    n_out_features=None,   # None keeps the hex map; an int global-avg-pools it away
    p_dropout=0.5,
)
activity = torch.rand(2, 19, len(connectome.nodes.type))   # (samples, frames, n_cells)
decoder.eval()
with torch.no_grad():
    recon = decoder(activity)                               # (2, 19, 1, 721)
print("A:", tuple(recon.shape), decoder.num_parameters)

# ------------------------------------- B: standalone, takes hexals in directly
class HexDecoder(nn.Module):
    """DecoderGAVP's architecture, taking (samples, frames, k, n_hexals) directly."""

    def __init__(self, in_channels, out_channels=1, extent=15, shape=(8,), kernel_size=5,
                 p_dropout=0.5, batch_norm=True, const_weight=None,
                 normalize_last=True, activation="Softplus"):
        super().__init__()
        pad = (kernel_size - 1) // 2
        u, v = get_hex_coords(extent)
        u, v = u - u.min(), v - v.min()
        self.register_buffer("u", torch.tensor(u, dtype=torch.long), persistent=False)
        self.register_buffer("v", torch.tensor(v, dtype=torch.long), persistent=False)
        self.H, self.W = int(u.max() + 1), int(v.max() + 1)          # 31, 31
        self.out_channels, self.normalize_last = out_channels, normalize_last

        layers, c_in = [], in_channels
        for c in shape:
            layers.append(Conv2dHexSpace(c_in, c, kernel_size,
                                         const_weight=const_weight, padding=pad))
            if batch_norm:
                layers.append(nn.BatchNorm2d(c))
            layers.append(getattr(nn, activation)())
            if p_dropout:
                layers.append(nn.Dropout(p_dropout))
            c_in = c
        self.base = nn.Sequential(*layers)
        self.decoder = nn.Sequential(
            Conv2dHexSpace(c_in, out_channels + (1 if normalize_last else 0),
                           kernel_size, const_weight=const_weight, padding=pad)
        )
        self.num_parameters = n_params(self)

    def forward(self, x):                      # (samples, frames, in_channels, n_hexals)
        n, t, c_in, _ = x.shape
        hex_map = x.new_zeros(n, t, c_in, self.H, self.W)   # new_zeros: keeps dtype+device
        hex_map[:, :, :, self.u, self.v] = x
        out = self.decoder(self.base(hex_map.view(n * t, c_in, self.H, self.W)))
        if self.normalize_last:
            out = out[:, : self.out_channels] / (
                nnf.softplus(out[:, self.out_channels :]) + 1
            )
        return out.view(n, t, self.out_channels, self.H, self.W)[:, :, :, self.u, self.v]


for k in (1, 8, 34):
    head = HexDecoder(in_channels=k, out_channels=1)
    y = head(torch.rand(2, 19, k, 721))        # (2, 19, 1, 721)
    y.sum().backward()
    print(f"B: k={k:2d} -> {tuple(y.shape)} {head.num_parameters}")

# Actual output:
#   A: (2, 19, 1, 721) NumberOfParams(free=2026, fixed=0)
#   B: k= 1 -> (2, 19, 1, 721) NumberOfParams(free=626,  fixed=0)
#   B: k= 8 -> (2, 19, 1, 721) NumberOfParams(free=2026, fixed=0)
#   B: k=34 -> (2, 19, 1, 721) NumberOfParams(free=7226, fixed=0)


## api

[
 {
  "name": "ActivityDecoder",
  "signature": "ActivityDecoder(connectome: ConnectomeFromAvgFilters)",
  "file": "D:/ML/Fly_Brain/.venv/Lib/site-packages/flyvis/task/decoder.py:23 (__init__ :41, forward :51)",
  "returns": "forward(activity: (n_samples, n_frames, n_cells)) -> LayerActivity (a dict subclass). `.output` is (n_samples, n_frames, n_output_cell_types, n_hexals) = (N, T, 34, 721); `['T4a']` is (N, T, 721). Zero parameters.",
  "notes": "No-op / identity decoder; only reshapes by cell type. Sets self.u, self.v (numpy int arrays, 721 entries, shifted to >=0) and self.H, self.W = 31, 31 from connectome.config.extent=15. Stores activity as a weakref (LayerActivity(..., keepref=False)), so the caller must keep the tensor alive."
 },
 {
  "name": "DecoderGAVP",
  "signature": "DecoderGAVP(connectome, shape: List[int], kernel_size: int, p_dropout: float = 0.5, batch_norm: bool = True, n_out_features: Optional[int] = None, const_weight: Optional[float] = None, normalize_last: bool = True, activation: str = 'Softplus')",
  "file": "D:/ML/Fly_Brain/.venv/Lib/site-packages/flyvis/task/decoder.py:190 (__init__ :216, forward :285)",
  "returns": "forward(activity: (N, T, n_cells)) -> (N, T, shape[-1], 721) when n_out_features is None; -> (N, T, shape[-1], n_out_features) when n_out_features is an int (GlobalAvgPool over hexals). Verified: flow config gives (2, 19, 2, 721); shape=[8,3], n_out_features=2 gives (2, 19, 3, 2).",
  "notes": "in_channels is NOT a constructor argument: it is len(connectome.output_cell_types) (34 for fib25-fib19_v2.2.json). shape = [hidden widths..., out_channels]; a 0 entry in shape[:-1] is skipped. Input is relu'd (nnf.relu on dvs_channels.output) before the convs. Architecture: base = [Conv2dHexSpace(in,h,k,pad=k//2), BatchNorm2d(h), Softplus, Dropout(p)] per hidden width, then decoder = Conv2dHexSpace(h, out_channels + (1 if normalize_last else 0), k, pad=k//2). With normalize_last=True the extra channel is used as out[:, :C] / (softplus(out[:, C:]) + 1). Instantaneous per frame: frames are flattened into the batch dim at line 309 and unflattened at line 323."
 },
 {
  "name": "Conv2dHexSpace",
  "signature": "Conv2dHexSpace(in_channels: int, out_channels: int, kernel_size: int, const_weight: Optional[float] = 1e-3, stride: int = 1, padding: int = 0, **kwargs)",
  "file": "D:/ML/Fly_Brain/.venv/Lib/site-packages/flyvis/task/decoder.py:112 (__init__ :137, filter_to_hex :171, forward :175)",
  "returns": "forward(x: (B, in_channels, H, W)) -> (B, out_channels, H_out, W_out). Standard Conv2d semantics; verified 7->1 with kernel_size=5, padding=2 on (3, 7, 31, 31) -> (3, 1, 31, 31), gradients flow.",
  "notes": "Fully standalone and channel-agnostic — a subclass of Conv2dConstWeight (decoder.py:73) which is a subclass of nn.Conv2d. kernel_size must be ODD (raises ValueError('4 is even. Must be odd.')). The hexagonal shape is a MASK on the square kernel: get_hex_coords(kernel_size//2) marks get_num_hexals(kernel_size//2) of kernel_size**2 positions (19 of 25 for k=5) and forward calls self.weight.data.mul_(self.mask.to(flyvis.device)) every call. Masked weights still count as parameters and still receive gradients — they are simply re-zeroed before each forward. self.mask is a plain float64 tensor attribute (NOT a registered buffer): it is absent from state_dict() and is NOT moved by .to(device); filter_to_hex hardcodes the module-global flyvis.device. kernel_size=1 disables masking entirely (self._filter_to_hex = False, no .mask attribute)."
 },
 {
  "name": "GlobalAvgPool",
  "signature": "GlobalAvgPool()",
  "file": "D:/ML/Fly_Brain/.venv/Lib/site-packages/flyvis/task/decoder.py:66",
  "returns": "forward(x) -> x.mean(dim=-1)",
  "notes": "Only used by DecoderGAVP when n_out_features is not None; irrelevant for a per-hexal reconstruction head (keep n_out_features=None)."
 },
 {
  "name": "flyvis.task.decoder.init_decoder",
  "signature": "init_decoder(decoder_config: Namespace, connectome) -> nn.Module",
  "file": "D:/ML/Fly_Brain/.venv/Lib/site-packages/flyvis/task/decoder.py:335",
  "returns": "ONE decoder module. Pops 'type' from the config and looks it up in decoder.py's globals().",
  "notes": "Shadowed at import time: flyvis/task/__init__.py does `from .decoder import *` then `from .tasks import *`, so `flyvis.init_decoder` and `flyvis.task.init_decoder` are the tasks.py version (dict-valued). Import this one explicitly as `from flyvis.task.decoder import init_decoder as init_one_decoder`. Verified: flyvis.init_decoder is flyvis.task.tasks.init_decoder -> True."
 },
 {
  "name": "flyvis.task.tasks.init_decoder",
  "signature": "init_decoder(config: Dict, connectome) -> Dict[str, ActivityDecoder]",
  "file": "D:/ML/Fly_Brain/.venv/Lib/site-packages/flyvis/task/tasks.py:171",
  "returns": "{task_name: decoder module}, e.g. {'flow': DecoderGAVP(...)}",
  "notes": "valmap's forward_subclass(ActivityDecoder, {**conf, 'connectome': connectome}) over each task key. forward_subclass (flyvis/utils/class_utils.py:29) resolves 'type' by recursive __subclasses__ search on ActivityDecoder, so any subclass you define yourself is discoverable by name. Warns and falls back to ActivityDecoder if 'type' is missing or unknown — it will not raise."
 },
 {
  "name": "Task.init_decoder",
  "signature": "Task.init_decoder(self, connectome) -> Dict[str, ActivityDecoder]",
  "file": "D:/ML/Fly_Brain/.venv/Lib/site-packages/flyvis/task/tasks.py:123",
  "returns": "Same dict as tasks.init_decoder(self.decoder, connectome)",
  "notes": "Requires a whole Task (Sintel dataset, dataloaders, folds). Not needed for a decoder — skip it."
 },
 {
  "name": "NetworkView.init_decoder",
  "signature": "NetworkView.init_decoder(self, checkpoint='best', decoder=None) -> Dict[str, nn.Module]",
  "file": "D:/ML/Fly_Brain/.venv/Lib/site-packages/flyvis/network/network_view.py:307",
  "returns": "{'flow': DecoderGAVP} with trained weights loaded from the checkpoint",
  "notes": "This is how you get the TRAINED flow head: NetworkView(flyvis.results_dir / 'flow/0000/000').init_decoder()['flow'] -> NumberOfParams(free=7427, fixed=0). Uses dir.config.task.decoder + recover_decoder. Caches by checkpoint id."
 },
 {
  "name": "recover_decoder",
  "signature": "recover_decoder(decoder: Dict[str, nn.Module], state_dict: Union[Dict, Path], strict: bool = True) -> Dict[str, nn.Module]",
  "file": "D:/ML/Fly_Brain/.venv/Lib/site-packages/flyvis/utils/chkpt_utils.py:45",
  "returns": "the same dict, weights loaded in place",
  "notes": "Expects a dict of decoders, not a single module. Checkpoint layout verified for .../flow/0000/000/best_chkpt: top keys ['network', 'decoder']; ck['decoder']['flow'] keys = base.0.weight (8,34,5,5), base.0.bias, base.1.weight, base.1.bias, base.1.running_mean, base.1.running_var, base.1.num_batches_tracked, decoder.0.weight (3,8,5,5), decoder.0.bias. The trained base.0.weight already has exactly 6 zeros per (out,in) slice — the hex mask."
 },
 {
  "name": "get_hex_coords / get_num_hexals",
  "signature": "get_hex_coords(extent: int, astensor: bool = False) -> (u, v); get_num_hexals(extent: int) -> int",
  "file": "D:/ML/Fly_Brain/.venv/Lib/site-packages/flyvis/utils/hex_utils.py:15 and :218",
  "returns": "get_hex_coords(15) -> two arrays of 721 axial coords; get_num_hexals(15) == 721, get_num_hexals(2) == 19",
  "notes": "This is the only thing you need to do the hexal<->map scatter/gather yourself: u -= u.min(); v -= v.min(); H, W = u.max()+1, v.max()+1 == 31, 31. Same recipe used in ActivityDecoder.__init__ and in Conv2dHexSpace's mask."
 },
 {
  "name": "n_params",
  "signature": "n_params(nnmodule: nn.Module) -> NumberOfParams(free: int, fixed: int)",
  "file": "D:/ML/Fly_Brain/.venv/Lib/site-packages/flyvis/utils/nn_utils.py:62",
  "returns": "dataclass with .free and .fixed",
  "notes": "What DecoderGAVP stores in self.num_parameters and logs at init."
 },
 {
  "name": "LayerActivity (the cell-type slicer inside the decoder)",
  "signature": "LayerActivity(activity, connectome, keepref: bool = False, use_central: bool = True)",
  "file": "D:/ML/Fly_Brain/.venv/Lib/site-packages/flyvis/utils/activity_utils.py:204 (the 'output' branch of __getattr__ is at :93)",
  "returns": "activity[..., output_indices] where output_indices has shape (n_output_types, n_cells_per_type)",
  "notes": "output_indices is built as np.array([np.nonzero(types == t)[0] for t in connectome.output_cell_types]) — this is why every selected cell type must have the SAME cell count. In this connectome 63 of 65 types have 721 cells; Lawf1 and Lawf2 have 123. Mixing them raises ValueError('inhomogeneous shape'); Lawf1 alone raises RuntimeError('value tensor of shape [2,3,1,123] cannot be broadcast to indexing result of shape [2,3,1,721]')."
 },
 {
  "name": "flyvis.task.objectives.l2norm / epe",
  "signature": "l2norm(y_est, y_gt, **kwargs); epe(y_est, y_gt, **kwargs)",
  "file": "D:/ML/Fly_Brain/.venv/Lib/site-packages/flyvis/task/objectives.py:10 and :25",
  "returns": "scalar tensors",
  "notes": "l2norm = (((y_est-y_gt)**2).sum(dim=(1,2,3))).sqrt().mean() — sums over frames, channels AND hexals then sqrt per sample. This is the loss the shipped flow decoder was trained with (loss: {flow: l2norm} in _meta.yaml). Both assume 4-D (samples, frames, channels, hexals) and work unchanged for a 1-channel reconstruction target."
 },
 {
  "name": "shipped decoder config (ground truth for the baseline)",
  "signature": "config.task.decoder.flow",
  "file": "D:/ML/Fly_Brain/data/flyvis/results/flow/0000/000/_meta.yaml (identical defaults in .venv/Lib/site-packages/flyvis/config/task/task.yaml)",
  "returns": "{type: DecoderGAVP, shape: [8, 2], kernel_size: 5, const_weight: 0.001, n_out_features: null, p_dropout: 0.5}",
  "notes": "batch_norm, normalize_last and activation are NOT in the config, so they take their defaults True, True, 'Softplus'. Decoder learning rate in flyvis/config/optim/optim.yaml + scheduler.yaml: Adam, lr_dec stepwise 5e-5 -> 5e-6 in 10 steps, same schedule as the network."
 }
]

## gotchas

[
 "PARAMETER COUNT. The shipped flow head is 7427 free parameters (verified twice: fresh instantiation and NetworkView(...).init_decoder()['flow']). Breakdown: base.0.weight (8,34,5,5)=6800, base.0.bias=8, base.1 BatchNorm weight+bias=16, decoder.0.weight (3,8,5,5)=600, decoder.0.bias=3. Closed form for shape=[h, C_out], kernel k, batch_norm=True, normalize_last=True, n_out_features=None: h*k^2*C_in + 3h + (C_out+1)*(h*k^2 + 1). SCALING IN INPUT CHANNELS IS LINEAR with slope h*k^2 = 8*25 = 200 params per input channel; the intercept is 3h + (C_out+1)*(h*k^2+1). Measured for shape=[8,1], k=5: C_in=1 -> 626, 2 -> 826, 4 -> 1226, 8 -> 2026, 16 -> 3626, 34 -> 7226, 64 -> 13226. So a 1-cell-type -> 1-channel reconstruction head is 626 params, i.e. an order of magnitude cheaper than the 34-channel flow head.",
 "THE HEX MASK IS NOT A SPARSE CONV. Conv2dHexSpace keeps a full square kernel and multiplies weight.data by a 0/1 mask (19 of 25 positions live for kernel_size=5) at every forward call. Consequences: (a) masked weights ARE in the parameter count and DO get gradients and optimizer updates -- they are re-zeroed at the start of the next forward, so any L2/weight-decay or optimizer-state you report is over 25 positions while only 19 are effective; (b) self.mask is a plain attribute holding a float64 CPU tensor, NOT a registered buffer -- it is absent from state_dict() (so checkpoints stay clean) but is also NOT moved by module.to(device), and filter_to_hex() hardcodes the module-global flyvis.device, so moving the decoder to a device other than flyvis.device will break or silently cross devices; (c) kernel_size must be odd (ValueError otherwise) and kernel_size=1 skips masking entirely and never creates .mask.",
 "Conv2dHexSpace's DEFAULT const_weight IS 1e-3, NOT None (decoder.py:142), unlike its parent Conv2dConstWeight whose default is None. Instantiating Conv2dHexSpace(in, out, 5) with no const_weight gives you an all-1e-3 weight AND all-1e-3 bias -- a flat, degenerate filter, not Kaiming init. Pass const_weight=None explicitly if you want torch's standard init. The shipped flow decoder deliberately uses const_weight=0.001.",
 "DecoderGAVP DOES NOT TAKE in_channels. It is len(connectome.output_cell_types), read at decoder.py:230. And forward() does NOT accept hexal-shaped input -- it wants the FULL network activity (n_samples, n_frames, n_cells) with n_cells == 45669 for this connectome, then slices by cell type through LayerActivity. Two ways out, both verified: (1) the ~6-line OutputTypeView proxy in the snippet, which reuses the shipped class and forward verbatim and lets you pick any subset of cell types (including a single one, k=1) while still feeding it the whole activity tensor; (2) reuse only Conv2dHexSpace + get_hex_coords and write the 12-line forward (HexDecoder in the snippet) if you want to feed arbitrary k-channel hexal tensors that are not network activity at all.",
 "ALL SELECTED CELL TYPES MUST HAVE THE SAME CELL COUNT. LayerActivity builds output_indices as np.array([np.nonzero(types == t)[0] for t in output_cell_types]) (activity_utils.py:269). In fib25-fib19_v2.2 63 of 65 types have 721 cells but Lawf1 and Lawf2 have 123 (they are tiled, not one-per-column). ['T4a','Lawf1'] -> ValueError: setting an array element with a sequence ... inhomogeneous shape. ['Lawf1'] alone -> RuntimeError: value tensor of shape [2,3,1,123] cannot be broadcast to indexing result of shape [2,3,1,721]. Restrict the cell-type set to 721-cell types, or scatter Lawf activity yourself.",
 "THE DECODER IS STRICTLY INSTANTANEOUS PER FRAME (no temporal filtering, no recurrence, no state). forward flattens (samples, frames) into the conv batch dim at decoder.py:309 and unflattens at :323. Verified empirically: with batch_norm=False and p_dropout=0, perturbing frames 3..5 leaves output frames 0..2 bit-identical, and decoder(a[:, 2:3]) equals decoder(a)[:, 2] to 1e-6. THE ONE EXCEPTION is BatchNorm2d in train() mode: its statistics are computed over n_samples*n_frames rows, so during training frames and samples in a batch are coupled through the normalization and the running stats. If you want a genuinely frame-independent baseline, either keep the decoder in eval() or pass batch_norm=False.",
 "DTYPE AND DEVICE OF THE INTERNAL SCATTER BUFFER. decoder.py:304 is `torch.zeros([...])` with no dtype/device -- it takes torch's DEFAULT dtype and device. Importing flyvis calls torch.set_default_device(cuda if available else cpu) at flyvis/__init__.py:13-14, so the buffer lands on flyvis's device regardless of where your activity tensor lives, and float64 input raises `RuntimeError: Index put requires the source and destination dtypes match, got Float for the destination and Double for the source` (verified). If you write your own forward, use x.new_zeros(...) instead, as HexDecoder does.",
 "TWO DIFFERENT FUNCTIONS ARE BOTH NAMED init_decoder, and the name you get by the obvious import is the dict-valued one. flyvis/task/__init__.py does `from .decoder import *` then `from .tasks import *`, so flyvis.init_decoder and flyvis.task.init_decoder both resolve to tasks.py:171 which returns Dict[str, ActivityDecoder] via valmap. The single-module version is decoder.py:335 -- import it as `from flyvis.task.decoder import init_decoder`. Also note forward_subclass (utils/class_utils.py:29) only WARNS on a missing or unrecognized 'type' and silently falls back to the base ActivityDecoder (which has zero parameters and returns a LayerActivity dict, not a tensor) -- a typo in the config will not raise, it will hand you a no-op decoder.",
 "activity is stored as a WEAKREF. ActivityDecoder builds LayerActivity(None, connectome, use_central=False) with keepref=False (activity_utils.py:112 wraps the value in weakref.ref). Inside DecoderGAVP.forward the local argument keeps it alive so this is safe, but if you call decoder.dvs_channels.update(some_temporary) yourself and then read .output, __getattr__ silently RETURNS None (activity_utils.py:80-81) rather than raising -- keep your own reference.",
 "THE TRAINED FLOW WEIGHTS ARE NOT TRANSFERABLE TO A 1-CHANNEL RECONSTRUCTION HEAD AS-IS. base.0.weight is (8, 34, 5, 5) so it only loads if you keep all 34 output cell types as input channels; decoder.0.weight is (3, 8, 5, 5) (out_channels=2 plus the normalize_last channel) whereas out_channels=1 needs (2, 8, 5, 5). You can warm-start base.* from the checkpoint when k=34 and must re-initialize decoder.*; recover_decoder(..., strict=False) is the escape hatch but it will silently skip the mismatched keys.",
 "normalize_last=True (the shipped default) silently adds one output channel and divides: out[:, :C] / (softplus(out[:, C:]) + 1) (decoder.py:318-320). So decoder.0 has C+1 output channels, the divisor is >= ~0.693+1 (never zero, no numerical hazard), and the head's output is bounded relative to the raw conv -- worth keeping in mind when comparing reconstruction magnitudes against a plain conv baseline.",
 "The connectome is a datamate Directory built ON DISK and cached BY CONFIG (not by file content). D:/ML/Fly_Brain/data/flyvis/connectome/ already holds 6 built directories; ConnectomeFromAvgFilters_0000 is the fib25 one (extent 15, n_syn_fill 1) and 0001/0002 are this project's MaleCNS exports. FLYVIS_ROOT_DIR must be set BEFORE importing flyvis (flydream.model.configure_flyvis_root does this) and flydream.model.patch_datamate_for_windows must be imported before any datamate write or you hit WinError 32. Note config.toml's [flyvis] root_dir='data/flyvis' is read by flydream, not by flyvis itself."
]

## unknowns

[
 "Whether the 'lum' target in MultiTaskSintel is the right reconstruction target and what its normalization/range is. I confirmed 'lum' is one of MultiTaskSintel.valid_tasks = ['lum', 'flow', 'depth'] (datasets/sintel.py:227), that it is always loaded because it is also the network input (sintel.py:259-260), that it is 4-D with (samples, frames, 1, n_hexals) from the solver's unpacking (solver.py:312), and that it goes through a transform_lum at sintel.py:599 -- but I did not read transform_lum, the boxfilter/eye rendering, or the value range, so I cannot state what a reconstruction loss would actually be regressing onto.",
 "Whether the trained flow decoder's base.* convolutions are a useful warm start for reconstruction. The shapes match only for k=34 input channels; I did not run any transfer experiment and there is nothing in the code that speaks to it.",
 "Whether flyvis ever trained or evaluated a reconstruction/autoencoding head. I grepped the whole package for decoder classes and found only ActivityDecoder and DecoderGAVP; the only shipped decoder configs are flow (shape [8,2]), and the docstring examples for depth (shape [8,1]) and lum (shape [8,3], n_out_features=2) at tasks.py:183-210. No 1-channel-per-hexal stimulus-reconstruction config exists in the package or in data/flyvis/results.",
 "Whether Conv2dHexSpace's mask semantics are correct for stride > 1 or for even-sized inputs. I only exercised stride=1 with padding=kernel_size//2 on the 31x31 map, which is the only configuration flyvis itself uses (DecoderGAVP hardcodes padding=(k-1)//2 and never passes stride).",
 "The behaviour under CUDA. flyvis.device resolved to cpu on this machine (no CUDA available), so the mask device mismatch I flagged in the gotchas is read off the code (filter_to_hex using the module-global flyvis.device on a non-buffer tensor) and was not reproduced.",
 "How well the 34-channel flow head's 7427 parameters compare to alternatives as a baseline capacity -- I established the counts and the scaling law but ran no training, so I have no statement about whether 626 params (k=1) is enough for stimulus reconstruction."
]