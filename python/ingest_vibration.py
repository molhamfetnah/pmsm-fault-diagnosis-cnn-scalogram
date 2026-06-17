import os
import numpy as np
from python.config import load_config
from python.manifest import new_manifest, load_manifest, add_record, save_manifest
from python.ingest_real import segment_signal

# EDIT to match the chosen vibration dataset's raw label strings:
LABEL_MAP = {"normal": "Healthy", "demag": "Demagnetization",
             "interturn": "InterTurn", "overload": "Overload"}


def map_label(raw):
    return LABEL_MAP[raw.lower()]


def load_recordings(cfg):
    # TODO-FOR-USER: parse the vibration dataset; yield dicts
    # {samples, fs, raw_label, recording_id, severity}
    raise NotImplementedError


def ingest(cfg):
    manifest_path = cfg["paths"]["manifest"]
    df = load_manifest(manifest_path) if os.path.exists(manifest_path) else new_manifest()
    seg_dir = os.path.join(cfg["paths"]["raw"], "segments")
    os.makedirs(seg_dir, exist_ok=True)
    for rec in load_recordings(cfg):
        for k, seg in enumerate(segment_signal(rec["samples"], fs=rec["fs"],
                                window_seconds=cfg["window_seconds"], overlap=cfg["overlap"])):
            sid = f"vib-{rec['recording_id']}-seg{k}"
            np.save(os.path.join(seg_dir, sid + ".npy"), seg)
            df = add_record(df, signal_id=sid, source="real", signal_type="vibration",
                            klass=map_label(rec["raw_label"]), severity=rec["severity"],
                            fs=rec["fs"], dataset_name="vibration_ds", recording_id=rec["recording_id"])
    save_manifest(df, cfg["paths"]["manifest"])


if __name__ == "__main__":
    ingest(load_config())
