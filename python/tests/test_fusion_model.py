import pytest

pytest.importorskip("tensorflow", reason="requires TensorFlow (run on Python 3.10-3.12)")

from python.model import build_fusion_cnn


def test_fusion_two_inputs_one_output():
    m = build_fusion_cnn(input_shape=(224, 224, 3), num_classes=4)
    assert len(m.inputs) == 2
    assert m.output_shape == (None, 4)
