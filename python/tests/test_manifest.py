from python.manifest import new_manifest, add_record, save_manifest, load_manifest, MANIFEST_COLUMNS


def test_new_manifest_columns():
    df = new_manifest()
    assert list(df.columns) == MANIFEST_COLUMNS


def test_add_record_appends_row():
    df = new_manifest()
    df = add_record(df, signal_id="s1", source="real", signal_type="current",
                    klass="Healthy", severity=0.0, fs=10000,
                    dataset_name="ds", recording_id="r1")
    assert len(df) == 1
    assert df.iloc[0]["class"] == "Healthy"
    assert df.iloc[0]["split"] == ""


def test_save_load_roundtrip(tmp_path):
    df = new_manifest()
    df = add_record(df, signal_id="s1", source="real", signal_type="current",
                    klass="Healthy", severity=0.0, fs=10000,
                    dataset_name="ds", recording_id="r1")
    p = tmp_path / "m.csv"
    save_manifest(df, str(p))
    out = load_manifest(str(p))
    assert out.iloc[0]["signal_id"] == "s1"
