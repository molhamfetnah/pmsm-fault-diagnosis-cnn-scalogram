from python.manifest import new_manifest, add_record
from python.balance import balance_df


def _df():
    df = new_manifest()
    spec = {("current", "train"): {"Healthy": 100, "InterTurn": 900},
            ("current", "val"): {"Healthy": 50, "InterTurn": 200},
            ("current", "test"): {"Healthy": 50, "InterTurn": 200}}
    for (st, split), counts in spec.items():
        for cls, n in counts.items():
            for i in range(n):
                df = add_record(df, signal_id=f"{st}-{split}-{cls}-{i}", source="real",
                                signal_type=st, klass=cls, severity=0.0, fs=10000,
                                dataset_name="ds", recording_id=f"{cls}-rec")
                df.loc[df.index[-1], "split"] = split
    return df


def test_train_and_val_balanced():
    out = balance_df(_df(), splits=("train", "val"), seed=42)
    for split in ("train", "val"):
        counts = out[out["split"] == split]["class"].value_counts()
        assert counts["Healthy"] == counts["InterTurn"]


def test_test_split_left_natural():
    df = _df()
    out = balance_df(df, splits=("train", "val"), seed=42)
    test = out[out["split"] == "test"]["class"].value_counts().to_dict()
    assert test == {"InterTurn": 200, "Healthy": 50}


def test_no_rows_invented():
    df = _df()
    out = balance_df(df, seed=42)
    # every kept row exists in the original
    assert set(out["signal_id"]) <= set(df["signal_id"])
