from python.data_loader import class_to_index


def test_class_to_index_stable_order():
    classes = ["Healthy", "InterTurn", "Demagnetization", "Overload"]
    idx = class_to_index(classes)
    assert idx["Healthy"] == 0 and idx["Overload"] == 3
    assert len(idx) == 4
