"""Ingest the IEEE DataPort 'Three-phase PMSM with ITSC faults' dataset
(doi:10.21227/4jpc-qh81, MAT files). Subscription-gated — place the .mat files under
data/raw/ieee_pmsm/ then run:  .venv/bin/python -m python.ingest_ieee_pmsm

ADAPT POINTS (the dataset's exact layout isn't public): after downloading, inspect
one file's keys and set CURRENT_KEYS and NATIVE_FS below.
    python -c "import scipy.io,sys; print(scipy.io.loadmat(sys.argv[1]).keys())" <file.mat>

The loader: finds the stator-current channel, resamples NATIVE_FS -> target_fs,
segments, labels by filename ('0'/healthy vs inter-turn), writes the manifest.
"""
import os
import glob
import re
import numpy as np
from scipy.signal import resample_poly

from python.config import load_config
from python.manifest import new_manifest, load_manifest, add_record, save_manifest
from python.ingest_real import segment_signal

# --- adapt after inspecting a real file -----------------------------------
CURRENT_KEYS = ("Ia", "ia", "current", "I_abc", "i_abc", "Is", "stator_current")
NATIVE_FS = 10000          # Hz — SET to the dataset's true rate after inspection
# --------------------------------------------------------------------------


def _load_mat(path):
    """Return a {name: ndarray} dict for classic or v7.3 (HDF5) MAT files."""
    try:
        import scipy.io
        d = scipy.io.loadmat(path)
        return {k: np.asarray(v) for k, v in d.items() if not k.startswith("__")}
    except NotImplementedError:                       # v7.3 = HDF5
        import h5py
        out = {}
        with h5py.File(path, "r") as f:
            for k in f.keys():
                try:
                    out[k] = np.asarray(f[k]).squeeze()
                except Exception:
                    pass
        return out


def _pick_current(mat):
    """Choose the current channel: a named key if present, else the longest 1-D array."""
    for k in CURRENT_KEYS:
        if k in mat and np.asarray(mat[k]).size > 100:
            return np.asarray(mat[k], dtype=float).squeeze().ravel()
    best, best_len = None, 0
    for v in mat.values():
        a = np.asarray(v, dtype=float).squeeze()
        if a.ndim >= 1 and a.size > best_len:
            best, best_len = (a if a.ndim == 1 else a.reshape(a.shape[0], -1)[:, 0]), a.size
    return None if best is None else np.asarray(best, dtype=float).ravel()


def _label(fname):
    f = os.path.basename(fname).lower()
    sev = 0.0
    m = re.search(r"(\d+(?:\.\d+)?)\s*%?", f)
    if m:
        sev = float(m.group(1))
    if "health" in f or "normal" in f or sev == 0.0:
        return "Healthy", 0.0
    return "InterTurn", sev / 100.0 if sev > 1 else sev


def ingest(cfg, raw_subdir="ieee_pmsm"):
    manifest_path = cfg["paths"]["manifest"]
    df = load_manifest(manifest_path) if os.path.exists(manifest_path) else new_manifest()
    seg_dir = os.path.join(cfg["paths"]["raw"], "segments")
    os.makedirs(seg_dir, exist_ok=True)
    target_fs, cap = cfg["target_fs"], cfg.get("max_segments_per_recording")

    for path in sorted(glob.glob(os.path.join(cfg["paths"]["raw"], raw_subdir, "**", "*.mat"),
                                 recursive=True)):
        sig = _pick_current(_load_mat(path))
        if sig is None or sig.size == 0:
            print("skip (no current channel found):", os.path.basename(path)); continue
        if NATIVE_FS != target_fs:
            sig = resample_poly(sig, target_fs, NATIVE_FS)
        klass, severity = _label(path)
        rec_id = "ieee-" + os.path.splitext(os.path.basename(path))[0]
        segs = list(segment_signal(sig, fs=target_fs,
                    window_seconds=cfg["window_seconds"], overlap=cfg["overlap"]))
        if cap:
            step = max(1, len(segs) // cap)
            segs = segs[::step][:cap]
        for k, seg in enumerate(segs):
            sid = f"{rec_id}-seg{k}"
            np.save(os.path.join(seg_dir, sid + ".npy"), seg)
            df = add_record(df, signal_id=sid, source="real", signal_type="current",
                            klass=klass, severity=severity, fs=target_fs,
                            dataset_name="ieee_4jpc-qh81", recording_id=rec_id)
    save_manifest(df, manifest_path)
    return df


if __name__ == "__main__":
    ingest(load_config())
