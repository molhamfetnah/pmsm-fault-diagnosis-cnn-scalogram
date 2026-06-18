from python.manifest import new_manifest, add_record
from python.train_fusion import pair_manifest, balance_pairs


def _df():
    df = new_manifest()
    # 6 conditions per class, each with current + vibration, 4 segments each
    for cls in ["Healthy", "InterTurn"]:
        for c in range(6):
            for modality in ["current", "vibration"]:
                rec = f"mendeley-{cls}{c}_{modality}_x"
                for s in range(4):
                    sid = f"{rec}-seg{s}"
                    df = add_record(df, signal_id=sid, source="real", signal_type=modality,
                                    klass=cls, severity=0.0, fs=10000, dataset_name="ds",
                                    recording_id=rec)
                    df.loc[df.index[-1], "scalogram_path"] = f"/x/{sid}.png"
    return df


def test_pairs_match_condition_and_segment():
    pairs = pair_manifest(_df(), seed=42)
    # 12 conditions * 4 segments = 48 pairs
    assert len(pairs) == 48
    # current and vibration in a pair share condition + segment index
    for _, row in pairs.iterrows():
        assert f"-seg{row['seg']}.png" in row["current_path"]
        assert f"-seg{row['seg']}.png" in row["vibration_path"]
        assert "current" in row["current_path"] and "vibration" in row["vibration_path"]


def test_condition_never_spans_splits():
    pairs = pair_manifest(_df(), seed=42)
    assert pairs.groupby("condition")["split"].nunique().eq(1).all()
    assert set(pairs["split"]) == {"train", "val", "test"}


def test_balance_pairs_equalizes_train():
    pairs = pair_manifest(_df(), seed=42)
    bal = balance_pairs(pairs, seed=42)
    tr = bal[bal["split"] == "train"]["class"].value_counts()
    assert tr["Healthy"] == tr["InterTurn"]
