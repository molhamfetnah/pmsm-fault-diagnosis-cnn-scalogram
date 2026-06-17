from python.manifest import new_manifest, add_record
from python.split import assign_splits


def _df():
    df = new_manifest()
    # 10 recordings per class, 3 segments each
    for klass in ["Healthy", "InterTurn"]:
        for r in range(10):
            for seg in range(3):
                df = add_record(df, signal_id=f"{klass}-{r}-{seg}", source="real",
                                signal_type="current", klass=klass, severity=0.0,
                                fs=10000, dataset_name="ds", recording_id=f"{klass}-rec{r}")
    return df


def test_no_recording_spans_splits():
    df = assign_splits(_df(), seed=42)
    per_rec_splits = df.groupby("recording_id")["split"].nunique()
    assert (per_rec_splits == 1).all()


def test_all_rows_assigned():
    df = assign_splits(_df(), seed=42)
    assert set(df["split"].unique()) <= {"train", "val", "test"}
    assert (df["split"] != "").all()
