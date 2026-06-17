from python.ingest_mendeley import parse_mendeley_filename


def test_parse_healthy_current():
    st, klass, sev, cap = parse_mendeley_filename("1000W_0_00_current_interturn.tdms")
    assert st == "current" and klass == "Healthy" and sev == 0.0 and cap == 1000


def test_parse_interturn_vibration():
    st, klass, sev, cap = parse_mendeley_filename("1500W_5_70_vibration_interturn.tdms")
    assert st == "vibration" and klass == "InterTurn" and round(sev, 4) == 0.057 and cap == 1500


def test_parse_unmatched_returns_none():
    assert parse_mendeley_filename("readme.txt") is None
