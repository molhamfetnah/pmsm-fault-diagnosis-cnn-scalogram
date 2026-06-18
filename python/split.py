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
        counts = [n_tr, n_val, n - n_tr - n_val]
        # Guarantee every split gets at least one recording when the class has
        # enough of them — otherwise rounding can starve a small class (e.g. only
        # 4 healthy recordings) out of val/test, leaving evaluation blind to it.
        n_splits = sum(1 for r in ratios if r > 0)
        if n >= n_splits:
            for j in range(len(counts)):
                if counts[j] == 0:
                    counts[counts.index(max(counts))] -= 1
                    counts[j] += 1
        bounds = (counts[0], counts[0] + counts[1])
        for i, rec in enumerate(recs):
            rec_split[rec] = "train" if i < bounds[0] else "val" if i < bounds[1] else "test"
    df["split"] = df["recording_id"].map(rec_split)
    return df
