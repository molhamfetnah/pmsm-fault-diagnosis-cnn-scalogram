from python.ingest_sim import parse_sim_filename


def test_parse_fault_filename():
    klass, severity = parse_sim_filename("InterTurn_sigma20.npy")
    assert klass == "InterTurn" and severity == 0.20


def test_parse_healthy_filename():
    klass, severity = parse_sim_filename("Healthy_load1.npy")
    assert klass == "Healthy" and severity == 0.0
