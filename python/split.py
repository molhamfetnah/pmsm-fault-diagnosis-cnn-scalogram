import random


def assign_splits(df, *, seed=42, ratios=(0.7, 0.15, 0.15)):
    rng = random.Random(seed)
    df = df.copy()
    rec_to_class = df.groupby("recording_id")["class"].first().to_dict()
    by_class = {}
    for rec, klass in rec_to_class.items():
        by_class.setdefault(klass, []).append(rec)
    rec_split = {}
    for klass, recs in by_class.items():
        recs = sorted(recs)
        rng.shuffle(recs)
        n = len(recs)
        n_tr = int(round(n * ratios[0]))
        n_val = int(round(n * ratios[1]))
        for i, rec in enumerate(recs):
            rec_split[rec] = "train" if i < n_tr else "val" if i < n_tr + n_val else "test"
    df["split"] = df["recording_id"].map(rec_split)
    return df
