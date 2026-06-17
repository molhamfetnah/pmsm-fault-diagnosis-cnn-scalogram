import pytest

# TensorFlow has no wheel for Python 3.13/3.14; skip cleanly where it is absent.
pytest.importorskip("tensorflow", reason="requires TensorFlow (run on Python 3.10-3.12)")

from python.model import build_cnn


def test_output_shape():
    m = build_cnn(input_shape=(224, 224, 3), num_classes=4)
    assert m.output_shape == (None, 4)


def test_accepts_custom_classes():
    m = build_cnn(input_shape=(64, 64, 3), num_classes=2)
    assert m.output_shape == (None, 2)
