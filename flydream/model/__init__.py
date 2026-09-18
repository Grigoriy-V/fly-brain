"""Model: the FlyVis deep mechanistic network on a chosen connectome export."""
import os
import pathlib
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[2]


def configure_flyvis_root() -> pathlib.Path:
    """Point flyvis at the project's data root before flyvis is imported."""
    cfg = tomllib.loads((ROOT / "config.toml").read_text())
    root = ROOT / cfg.get("flyvis", {}).get("root_dir", "data/flyvis")
    os.environ.setdefault("FLYVIS_ROOT_DIR", str(root))
    return root


def patch_datamate_for_windows() -> None:
    """datamate 1.0.0's `_write_h5` unlinks a file it still holds open, which
    Windows refuses (WinError 32). Replace it with a version that closes first."""
    import h5py as h5
    import numpy as np
    import datamate.io as dio
    import datamate.directory as ddir

    def _write_h5(path, val):
        val = np.asarray(val)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_dir():
            path.rmdir()
        elif path.exists():
            path.unlink()
        f = h5.File(path, libver="latest", mode="w")
        f["data"] = val
        f.swmr_mode = True
        f.close()

    dio._write_h5 = _write_h5
    if hasattr(ddir, "_write_h5"):
        ddir._write_h5 = _write_h5


patch_datamate_for_windows()
