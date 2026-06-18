import numpy as np
from python.evaluate import metrics_from_predictions


def test_metrics_from_predictions():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    cm, report = metrics_from_predictions(y_true, y_pred, ["A", "B"])
    assert cm.shape == (2, 2)
    assert "accuracy" in report


def test_subset_of_classes_present():
    # Real data may only contain 2 of the 4 configured classes; the report must
    # still align with all class_names instead of raising.
    names = ["Healthy", "InterTurn", "Demagnetization", "Overload"]
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 1])
    cm, report = metrics_from_predictions(y_true, y_pred, names)
    assert cm.shape == (4, 4)
    assert all(n in report for n in names)
