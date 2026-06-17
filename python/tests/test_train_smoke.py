import pytest

pytest.importorskip("tensorflow", reason="requires TensorFlow (run on Python 3.10-3.12)")
pytest.importorskip("PIL", reason="requires Pillow")

import numpy as np
from PIL import Image
from python.manifest import new_manifest, add_record
from python.train import train_from_df


def _tiny_dataset(tmp_path):
    df = new_manifest()
    for klass in ["Healthy", "InterTurn"]:
        for r in range(4):
            p = tmp_path / f"{klass}-{r}.png"
            Image.fromarray((np.random.rand(32, 32, 3) * 255).astype("uint8")).save(p)
            df = add_record(df, signal_id=f"{klass}-{r}", source="real",
                            signal_type="current", klass=klass, severity=0.0, fs=10000,
                            dataset_name="ds", recording_id=f"{klass}-rec{r}")
            df.loc[df.index[-1], "scalogram_path"] = str(p)
            df.loc[df.index[-1], "split"] = "train" if r < 3 else "val"
    return df


def test_train_runs_one_epoch(tmp_path):
    df = _tiny_dataset(tmp_path)
    model, hist = train_from_df(df, classes=["Healthy", "InterTurn"], signal_type="current",
                                image_size=32, batch_size=2, epochs=1, seed=42)
    assert "loss" in hist.history
