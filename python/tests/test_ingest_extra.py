import os
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy")


def _cfg(tmp_path):
    return {"paths": {"raw": str(tmp_path), "manifest": str(tmp_path / "manifest.csv")},
            "target_fs": 10000, "window_seconds": 0.5, "overlap": 0.5,
            "max_segments_per_recording": 5}


def test_uottawa_loader(tmp_path):
    from python.ingest_uottawa import ingest
    d = tmp_path / "uottawa"; d.mkdir()
    n = int(42000 * 0.6)                                   # 0.6 s @ 42 kHz
    arr = np.random.default_rng(0).normal(size=(n, 5))     # accel1,acoustic,accel2,accel3,temp
    pd.DataFrame(arr).to_csv(d / "motor_Healthy_run1.csv", header=False, index=False)
    df = ingest(_cfg(tmp_path))
    rows = df[df["dataset_name"] == "uottawa"]
    assert len(rows) >= 1
    assert set(rows["class"]) == {"Healthy"} and set(rows["signal_type"]) == {"vibration"}
    seg = np.load(os.path.join(str(tmp_path), "segments", rows.iloc[0]["signal_id"] + ".npy"))
    assert len(seg) == int(10000 * 0.5)                    # resampled to 10 kHz, 0.5 s window


def test_ieee_pmsm_loader(tmp_path):
    import scipy.io
    from python.ingest_ieee_pmsm import ingest
    d = tmp_path / "ieee_pmsm"; d.mkdir()
    scipy.io.savemat(str(d / "pmsm_health_0.mat"),
                     {"Ia": np.random.default_rng(1).normal(size=int(10000 * 0.6))})
    df = ingest(_cfg(tmp_path))
    rows = df[df["dataset_name"] == "ieee_4jpc-qh81"]
    assert len(rows) >= 1
    assert set(rows["class"]) == {"Healthy"} and set(rows["signal_type"]) == {"current"}
