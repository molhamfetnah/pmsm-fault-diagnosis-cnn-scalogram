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


def _small_df(n_recordings):
    df = new_manifest()
    for r in range(n_recordings):
        df = add_record(df, signal_id=f"H-{r}", source="real", signal_type="current",
                        klass="Healthy", severity=0.0, fs=10000, dataset_name="ds",
                        recording_id=f"H-rec{r}")
    return df


def test_small_class_reaches_every_split():
    # 4 recordings must still land one each in val and test, not all in train.
    df = assign_splits(_small_df(4), seed=42)
    assert set(df["split"].unique()) == {"train", "val", "test"}


def test_three_recordings_one_per_split():
    df = assign_splits(_small_df(3), seed=42)
    assert sorted(df["split"].tolist()) == ["test", "train", "val"]


def test_two_recordings_no_empty_split_forced():
    # Too few to fill all splits; must not crash or duplicate.
    df = assign_splits(_small_df(2), seed=42)
    assert len(df) == 2
    assert df.groupby("recording_id")["split"].nunique().eq(1).all()
