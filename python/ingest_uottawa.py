"""Ingest the University of Ottawa UOEMD-VAFCVS dataset (Mendeley msxs4vj48g).

INDUCTION-motor vibration/acoustic data (NOT PMSM) — use only for a cross-machine
transfer/robustness experiment. CSVs are 5 columns @ 42 kHz:
    accel1, acoustic, accel2, accel3, temperature   (no header)

Place CSVs under data/raw/uottawa/ then run:  .venv/bin/python -m python.ingest_uottawa
Resamples 42 kHz -> cfg['target_fs'], segments, tags dataset_name='uottawa'.

Label mapping is by filename token; edit MAP after checking the real filenames.
"""
import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from python.config import load_config
from python.manifest import new_manifest, load_manifest, add_record, save_manifest
from python.ingest_real import segment_signal

NATIVE_FS = 42000
VIB_COL = 0  # accel1


def map_label(fname):
    """Map a filename to one of our classes; return None to skip (e.g. bearing)."""
    f = os.path.basename(fname).lower()
    if any(t in f for t in ("health", "normal", "good", "baseline")):
        return "Healthy"
    if any(t in f for t in ("stator", "winding", "turn")):
        return "InterTurn"
    return None  # other faults (bearing, unbalance, ...) not in our class set


def _read_csv_channel(path, col):
    df = pd.read_csv(path, header=None)
    return np.asarray(df.iloc[:, col].values, dtype=float)


def ingest(cfg, raw_subdir="uottawa", vib_col=VIB_COL):
    manifest_path = cfg["paths"]["manifest"]
    df = load_manifest(manifest_path) if os.path.exists(manifest_path) else new_manifest()
    seg_dir = os.path.join(cfg["paths"]["raw"], "segments")
    os.makedirs(seg_dir, exist_ok=True)
    target_fs, cap = cfg["target_fs"], cfg.get("max_segments_per_recording")

    for path in sorted(glob.glob(os.path.join(cfg["paths"]["raw"], raw_subdir, "**", "*.csv"),
                                 recursive=True)):
        klass = map_label(path)
        if klass is None:
            print("skip (class not mapped):", os.path.basename(path)); continue
        sig = _read_csv_channel(path, vib_col)
        if sig.size == 0:
            continue
        sig = resample_poly(sig, target_fs, NATIVE_FS)        # 42 kHz -> target_fs
        rec_id = "uottawa-" + os.path.splitext(os.path.basename(path))[0]
        segs = list(segment_signal(sig, fs=target_fs,
                    window_seconds=cfg["window_seconds"], overlap=cfg["overlap"]))
        if cap:
            step = max(1, len(segs) // cap)
            segs = segs[::step][:cap]
        for k, seg in enumerate(segs):
            sid = f"{rec_id}-seg{k}"
            np.save(os.path.join(seg_dir, sid + ".npy"), seg)
            df = add_record(df, signal_id=sid, source="real", signal_type="vibration",
                            klass=klass, severity=0.0, fs=target_fs,
                            dataset_name="uottawa", recording_id=rec_id)
    save_manifest(df, manifest_path)
    return df


if __name__ == "__main__":
    ingest(load_config())
