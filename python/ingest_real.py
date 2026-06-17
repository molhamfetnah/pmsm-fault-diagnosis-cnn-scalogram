import os
import numpy as np
from python.config import load_config
from python.manifest import new_manifest, add_record, save_manifest, load_manifest


def segment_signal(samples, *, fs, window_seconds, overlap):
    win = int(round(window_seconds * fs))
    step = max(1, int(round(win * (1 - overlap))))
    segs = []
    i = 0
    while i + win <= len(samples):
        segs.append(np.asarray(samples[i:i + win]))
        i += step
    return segs


def ingest(cfg):
    """Read raw recordings, segment, save .npy, and append manifest rows.
    EDIT load_recordings() to match the chosen dataset's file format/labels."""
    df = load_manifest(cfg["paths"]["manifest"]) if os.path.exists(cfg["paths"]["manifest"]) else new_manifest()
    seg_dir = os.path.join(cfg["paths"]["raw"], "segments")
    os.makedirs(seg_dir, exist_ok=True)
    for rec in load_recordings(cfg):          # rec: dict(samples, fs, class, dataset, recording_id, severity)
        segs = segment_signal(rec["samples"], fs=rec["fs"],
                              window_seconds=cfg["window_seconds"], overlap=cfg["overlap"])
        for k, seg in enumerate(segs):
            sid = f"{rec['recording_id']}-seg{k}"
            np.save(os.path.join(seg_dir, sid + ".npy"), seg)
            df = add_record(df, signal_id=sid, source="real", signal_type="current",
                            klass=rec["class"], severity=rec["severity"], fs=rec["fs"],
                            dataset_name=rec["dataset"], recording_id=rec["recording_id"])
    save_manifest(df, cfg["paths"]["manifest"])
    return df


def load_recordings(cfg):
    # TODO-FOR-USER: implement parsing for the dataset chosen in docs/data-audit.md.
    # Yield dicts: {samples: 1D array, fs: int, class: str, dataset: str,
    #               recording_id: str, severity: float}
    raise NotImplementedError("Implement load_recordings() for your dataset format")


if __name__ == "__main__":
    ingest(load_config())
