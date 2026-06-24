import pytest

# TensorFlow has no wheel for Python 3.13/3.14; skip cleanly where it is absent.
pytest.importorskip("tensorflow", reason="requires TensorFlow (run on Python 3.10-3.12)")

from python.model import build_cnn, build_modern_cnn, build_transfer_cnn


def test_output_shape():
    m = build_cnn(input_shape=(224, 224, 3), num_classes=4)
    assert m.output_shape == (None, 4)


def test_accepts_custom_classes():
    m = build_cnn(input_shape=(64, 64, 3), num_classes=2)
    assert m.output_shape == (None, 2)


def test_depth_configurable():
    assert len([l for l in build_cnn(filters=(8, 16)).layers if "Conv2D" in type(l).__name__]) == 2
    assert len([l for l in build_cnn(filters=(8, 16, 32, 64)).layers if "Conv2D" in type(l).__name__]) == 4


def test_modern_cnn_builds():
    m = build_modern_cnn(input_shape=(224, 224, 3), num_classes=4)
    assert m.output_shape == (None, 4)
    assert any("BatchNormalization" in type(l).__name__ for l in m.layers)


def test_transfer_cnn_builds_without_download():
    m = build_transfer_cnn(input_shape=(224, 224, 3), num_classes=3, weights=None)
    assert m.output_shape == (None, 3)
    assert len(m.inputs) == 1
