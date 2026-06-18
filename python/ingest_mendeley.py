"""Ingest the KAIST PMSM stator-fault dataset (Mendeley rgn5brrgrn).

Dataset: "Vibration and Current Dataset of Three-Phase PMSM with Stator Faults"
(DOI 10.17632/rgn5brrgrn.5, CC-BY-4.0). Current @ 100 kHz, vibration @ 25.6 kHz,
stored as TDMS files named:

    <cap>W_<sev_int>_<sev_frac>_<modality>_<faulttype>.tdms
    e.g. 1000W_0_00_current_interturn.tdms   (healthy, 0.00% severity)
         1500W_5_70_vibration_intercoil.tdms (5.70% inter-coil fault)

Place the downloaded .tdms files under data/raw/mendeley_pmsm/ then run this.
Long recordings are decimated to cfg['target_fs'] before segmentation to keep
scalogram sizes tractable.
"""
import os
import glob
import re
import numpy as np
from scipy.signal import decimate

from python.config import load_config
from python.manifest import new_manifest, load_manifest, add_record, save_manifest
from python.ingest_real import segment_signal

FNAME_RE = re.compile(
    r"(?P<cap>\d+)W_(?P<sev_i>\d+)_(?P<sev_f>\d+)_(?P<modality>current|vibration)_(?P<fault>\w+)",
    re.IGNORECASE,
)

# Dataset fault token -> our class label. "normal"/0% severity maps to Healthy.
FAULT_TO_CLASS = {"interturn": "InterTurn", "intercoil": "InterTurn", "normal": "Healthy"}
NATIVE_FS = {"current": 100_000, "vibration": 25_600}


def parse_mendeley_filename(fname):
    """Return (signal_type, klass, severity, cap_w) or None if it doesn't match."""
    m = FNAME_RE.search(os.path.basename(fname))
    if not m:
        return None
    severity = float(f"{m.group('sev_i')}.{m.group('sev_f')}") / 100.0
    fault = m.group("fault").lower()
    klass = "Healthy" if severity == 0.0 else FAULT_TO_CLASS.get(fault, "InterTurn")
    return m.group("modality").lower(), klass, severity, int(m.group("cap"))


def _read_tdms(path):
    from nptdms import TdmsFile
    tdms = TdmsFile.read(path)
    for group in tdms.groups():
        for ch in group.channels():
            data = ch[:]
            if len(data):
                return np.asarray(data, dtype=float)
    return np.array([])


def ingest(cfg, raw_subdir="mendeley_pmsm"):
    manifest_path = cfg["paths"]["manifest"]
    df = load_manifest(manifest_path) if os.path.exists(manifest_path) else new_manifest()
    seg_dir = os.path.join(cfg["paths"]["raw"], "segments")
    os.makedirs(seg_dir, exist_ok=True)
    target_fs = cfg["target_fs"]

    paths = glob.glob(os.path.join(cfg["paths"]["raw"], raw_subdir, "**", "*.tdms"), recursive=True)
    for path in sorted(paths):
        parsed = parse_mendeley_filename(path)
        if parsed is None:
            print("skip (unparsed):", os.path.basename(path))
            continue
        modality, klass, severity, cap = parsed
        sig = _read_tdms(path)
        if sig.size == 0:
            continue
        native_fs = NATIVE_FS.get(modality, target_fs)
        q = max(1, int(round(native_fs / target_fs)))
        if q > 1:
            sig = decimate(sig, q, ftype="fir")
        rec_id = "mendeley-" + os.path.splitext(os.path.basename(path))[0]
        cap = cfg.get("max_segments_per_recording")
        segs = list(segment_signal(sig, fs=target_fs,
                    window_seconds=cfg["window_seconds"], overlap=cfg["overlap"]))
        if cap:
            # Even stride across the recording so kept segments span the whole run,
            # not just its first seconds.
            step = max(1, len(segs) // cap)
            segs = segs[::step][:cap]
        for k, seg in enumerate(segs):
            sid = f"{rec_id}-seg{k}"
            np.save(os.path.join(seg_dir, sid + ".npy"), seg)
            df = add_record(df, signal_id=sid, source="real", signal_type=modality,
                            klass=klass, severity=severity, fs=target_fs,
                            dataset_name="mendeley_rgn5brrgrn", recording_id=rec_id)
    save_manifest(df, manifest_path)
    return df


if __name__ == "__main__":
    ingest(load_config())
