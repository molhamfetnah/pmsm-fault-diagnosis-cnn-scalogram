import numpy as np
from python.evaluate import metrics_from_predictions


def test_metrics_from_predictions():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    cm, report = metrics_from_predictions(y_true, y_pred, ["A", "B"])
    assert cm.shape == (2, 2)
    assert "accuracy" in report
