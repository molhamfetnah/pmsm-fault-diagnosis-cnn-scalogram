import os
import glob
import re
import numpy as np
from python.config import load_config
from python.manifest import new_manifest, load_manifest, add_record, save_manifest
from python.ingest_real import segment_signal


def parse_sim_filename(fname):
    stem = os.path.splitext(os.path.basename(fname))[0]
    klass = stem.split("_")[0]
    m = re.search(r"sigma(\d+)", stem)
    severity = int(m.group(1)) / 100.0 if m else 0.0
    return klass, severity


def ingest(cfg):
    manifest_path = cfg["paths"]["manifest"]
    df = load_manifest(manifest_path) if os.path.exists(manifest_path) else new_manifest()
    seg_dir = os.path.join(cfg["paths"]["raw"], "segments")
    os.makedirs(seg_dir, exist_ok=True)
    for path in glob.glob(os.path.join(cfg["paths"]["raw"], "sim", "**", "*.npy"), recursive=True):
        klass, severity = parse_sim_filename(path)
        sig = np.load(path)
        rec_id = "sim-" + os.path.splitext(os.path.basename(path))[0]
        for k, seg in enumerate(segment_signal(sig, fs=cfg["target_fs"],
                                window_seconds=cfg["window_seconds"], overlap=cfg["overlap"])):
            sid = f"{rec_id}-seg{k}"
            np.save(os.path.join(seg_dir, sid + ".npy"), seg)
            df = add_record(df, signal_id=sid, source="sim", signal_type="current",
                            klass=klass, severity=severity, fs=cfg["target_fs"],
                            dataset_name="simulation", recording_id=rec_id)
    save_manifest(df, cfg["paths"]["manifest"])


if __name__ == "__main__":
    ingest(load_config())
