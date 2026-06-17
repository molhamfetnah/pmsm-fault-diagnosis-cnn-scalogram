from python.ingest_vibration import map_label


def test_map_label_known():
    assert map_label("demag") == "Demagnetization"
    assert map_label("normal") == "Healthy"
