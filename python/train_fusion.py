"""Dual-branch current+vibration fusion CNN (real KAIST data).

Pairs each current scalogram with the vibration scalogram from the *same
operating condition and segment index*, then trains the two-input model from
python.model.build_fusion_cnn.

Pairing notes
- The KAIST dataset stores current and vibration as separate recordings per
  condition (same cap/severity/fault). Both are decimated to the same target_fs
  and segmented identically, so segment index k of the current recording and of
  the vibration recording cover the same time fraction — an approximate temporal
  pairing (not sample-synchronised).
- The train/val/test split is assigned at the *condition* level so a condition's
  current and vibration scalograms always land in the same split (no leakage).
- Same limitation as the single-channel models: only 4 distinct healthy
  conditions exist, so fusion cannot be validated beyond that diversity.
"""
import os
import json
import numpy as np
import pandas as pd

from python.config import load_config
from python.manifest import load_manifest
from python.split import assign_splits
from python.data_loader import class_to_index


def _condition(recording_id):
    """Recording id with the modality token removed, e.g.
    'mendeley-1000W_0_00_current_interturn' -> 'mendeley-1000W_0_00_interturn'."""
    return recording_id.replace("_current_", "_").replace("_vibration_", "_")


def _seg_index(signal_id):
    return signal_id.rsplit("-seg", 1)[1]


def pair_manifest(df, *, seed=42):
    """Return paired rows (current_path, vibration_path, class, condition, split).

    Split is assigned per condition so both modalities share it (leakage-free).
    """
    df = df.copy()
    df["condition"] = df["recording_id"].map(_condition)
    df["seg"] = df["signal_id"].map(_seg_index)
    cur = (df[df["signal_type"] == "current"][["condition", "seg", "class", "scalogram_path"]]
           .rename(columns={"scalogram_path": "current_path"}))
    vib = (df[df["signal_type"] == "vibration"][["condition", "seg", "scalogram_path"]]
           .rename(columns={"scalogram_path": "vibration_path"}))
    pairs = cur.merge(vib, on=["condition", "seg"], how="inner").reset_index(drop=True)
    # Condition-level split: reuse assign_splits with recording_id = condition.
    tmp = pairs.rename(columns={"condition": "recording_id"})
    pairs["split"] = assign_splits(tmp, seed=seed)["split"].values
    return pairs


def balance_pairs(pairs, *, splits=("train", "val"), seed=42):
    """Undersample the majority class in the named splits; others pass through."""
    keep = []
    for split, g in pairs.groupby("split", sort=False):
        if split not in splits:
            keep.append(g)
            continue
        n = g["class"].value_counts().min()
        for _cls, gg in g.groupby("class", sort=False):
            keep.append(gg.sample(n=min(len(gg), n), random_state=seed))
    return pd.concat(keep).reset_index(drop=True)


def make_fusion_dataset(pairs, split, classes, image_size, batch_size, seed, augment=False):
    import tensorflow as tf
    idx = class_to_index(classes)
    sub = pairs[pairs["split"] == split]
    cpaths = sub["current_path"].tolist()
    vpaths = sub["vibration_path"].tolist()
    labels = [idx[c] for c in sub["class"].tolist()]

    def _img(p):
        im = tf.io.decode_png(tf.io.read_file(p), channels=3)
        return tf.image.resize(im, [image_size, image_size]) / 255.0

    def _load(cp, vp, y):
        return {"current": _img(cp), "vibration": _img(vp)}, y

    ds = tf.data.Dataset.from_tensor_slices((cpaths, vpaths, labels))
    if split == "train":
        ds = ds.shuffle(len(cpaths) or 1, seed=seed)
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def train(cfg, epochs=30):
    from tensorflow import keras
    from python.model import build_fusion_cnn

    df = load_manifest(cfg["paths"]["manifest"])
    pairs = pair_manifest(df, seed=cfg["seed"])
    if cfg.get("balance_train", True):
        pairs = balance_pairs(pairs, seed=cfg["seed"])
    classes, size = cfg["classes"], cfg["image_size"]
    tr = make_fusion_dataset(pairs, "train", classes, size, 16, cfg["seed"], augment=True)
    va = make_fusion_dataset(pairs, "val", classes, size, 16, cfg["seed"])
    model = build_fusion_cnn(input_shape=(size, size, 3), num_classes=len(classes))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(tr, validation_data=va, epochs=epochs,
              callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
    os.makedirs(cfg["paths"]["models"], exist_ok=True)
    model.save(os.path.join(cfg["paths"]["models"], "cnn_fusion.keras"))
    return model, pairs


def evaluate(cfg, model=None, pairs=None):
    from tensorflow import keras
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from python.evaluate import metrics_from_predictions

    df = load_manifest(cfg["paths"]["manifest"])
    if pairs is None:
        pairs = pair_manifest(df, seed=cfg["seed"])
    classes, size = cfg["classes"], cfg["image_size"]
    ds = make_fusion_dataset(pairs, "test", classes, size, 16, cfg["seed"])
    if model is None:
        model = keras.models.load_model(os.path.join(cfg["paths"]["models"], "cnn_fusion.keras"))
    y_true, y_pred = [], []
    for x, y in ds:
        y_pred.extend(model.predict(x, verbose=0).argmax(axis=1).tolist())
        y_true.extend(y.numpy().tolist())
    cm, report = metrics_from_predictions(np.array(y_true), np.array(y_pred), classes)
    os.makedirs(cfg["paths"]["results"], exist_ok=True)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right"); ax.set_yticklabels(classes)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, int(v), ha="center", va="center")
    ax.set_title("Fusion (current+vibration) — test"); fig.colorbar(im); fig.tight_layout()
    fig.savefig(os.path.join(cfg["paths"]["results"], "confusion_fusion.png"))
    with open(os.path.join(cfg["paths"]["results"], "report_fusion.json"), "w") as f:
        json.dump(report, f, indent=2)
    acc = report.get("accuracy", report.get("micro avg", {}).get("f1-score"))
    print(f"fusion accuracy={acc:.3f}")
    return report


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    a = ap.parse_args()
    cfg = load_config()
    model, pairs = train(cfg, epochs=a.epochs)
    evaluate(cfg, model=model, pairs=pairs)
