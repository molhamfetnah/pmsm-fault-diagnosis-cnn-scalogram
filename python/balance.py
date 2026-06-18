"""Class balancing for the training pipeline.

Real PMSM datasets are heavily imbalanced (few healthy recordings vs. many fault
recordings). Training on the raw distribution lets the CNN collapse to predicting
the majority class. We undersample the majority class *only in the splits used for
fitting* (train/val), per signal_type, so the model gets a balanced learning
signal. The test split is deliberately left at its natural distribution so reported
metrics reflect reality — report macro-F1 / per-class recall, not raw accuracy.
"""
import pandas as pd


def balance_df(df, *, splits=("train", "val"), seed=42):
    """Return DF with the named splits undersampled to equal per-class counts.

    Balancing is done independently per (signal_type, split). Rows in splits not
    listed (e.g. ``test``) are passed through untouched.
    """
    keep = []
    for (_st, split), g in df.groupby(["signal_type", "split"], sort=False):
        if split not in splits:
            keep.append(g)
            continue
        n = g["class"].value_counts().min()
        for _cls, gg in g.groupby("class", sort=False):
            keep.append(gg.sample(n=min(len(gg), n), random_state=seed))
    return pd.concat(keep).reset_index(drop=True)
