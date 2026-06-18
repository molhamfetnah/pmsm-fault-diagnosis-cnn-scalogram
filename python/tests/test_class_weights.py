import pytest

pytest.importorskip("tensorflow", reason="requires TensorFlow (run on Python 3.10-3.12)")

from python.manifest import new_manifest, add_record
from python.train import compute_class_weights

CLASSES = ["Healthy", "InterTurn", "Demagnetization", "Overload"]


def _df(counts):
    df = new_manifest()
    for klass, n in counts.items():
        for r in range(n):
            df = add_record(df, signal_id=f"{klass}-{r}", source="real",
                            signal_type="current", klass=klass, severity=0.0, fs=10000,
                            dataset_name="ds", recording_id=f"{klass}-rec{r}")
            df.loc[df.index[-1], "split"] = "train"
    return df


def test_minority_class_gets_higher_weight():
    df = _df({"Healthy": 10, "InterTurn": 160})
    w = compute_class_weights(df, classes=CLASSES, signal_type="current")
    assert w[0] > w[1]  # Healthy (minority) up-weighted relative to InterTurn


def test_absent_classes_default_to_one():
    df = _df({"Healthy": 10, "InterTurn": 160})
    w = compute_class_weights(df, classes=CLASSES, signal_type="current")
    assert w[2] == 1.0 and w[3] == 1.0  # Demagnetization, Overload absent


def test_balanced_classes_get_equal_weight():
    df = _df({"Healthy": 50, "InterTurn": 50})
    w = compute_class_weights(df, classes=CLASSES, signal_type="current")
    assert w[0] == pytest.approx(w[1])
