import json
import os
from tensorflow import keras
from python.config import load_config
from python.manifest import load_manifest
from python.data_loader import make_dataset
from python.model import build_cnn, build_modern_cnn, build_transfer_cnn
from python.balance import balance_df


def compute_class_weights(df, *, classes, signal_type):
    """Inverse-frequency weights over the train split, keyed by class index.

    Real datasets are heavily imbalanced (few healthy recordings vs. many fault
    recordings); absent classes default to weight 1.0 so the 4-class config still
    trains on data that only covers a subset of classes.
    """
    from python.data_loader import class_to_index
    idx = class_to_index(classes)
    counts = (df[(df["split"] == "train") & (df["signal_type"] == signal_type)]["class"]
              .value_counts().to_dict())
    total = sum(counts.values())
    n_present = len(counts) or 1
    return {idx[c]: (total / (n_present * counts[c]) if c in counts else 1.0)
            for c in classes}


def _build(arch, image_size, n_classes, filters):
    shape = (image_size, image_size, 3)
    if arch == "modern":
        return build_modern_cnn(input_shape=shape, num_classes=n_classes, filters=tuple(filters))
    if arch == "transfer":
        return build_transfer_cnn(input_shape=shape, num_classes=n_classes)
    return build_cnn(input_shape=shape, num_classes=n_classes, filters=tuple(filters))


def train_from_df(df, *, classes, signal_type, image_size, batch_size, epochs, seed,
                  filters=(32, 64, 128), arch="baseline", aug_mode="flip"):
    train_ds, _ = make_dataset(df, "train", signal_type, classes, image_size, batch_size, seed,
                               augment=True, aug_mode=aug_mode)
    val_ds, _ = make_dataset(df, "val", signal_type, classes, image_size, batch_size, seed)
    class_weight = compute_class_weights(df, classes=classes, signal_type=signal_type)
    model = _build(arch, image_size, len(classes), filters)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs, class_weight=class_weight,
                     callbacks=[
                         keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                         keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-5),
                     ])
    return model, hist


def train(cfg, signal_type="current", epochs=30, arch="baseline"):
    df = load_manifest(cfg["paths"]["manifest"])
    if cfg.get("balance_train", True):
        # Undersample the majority class in train/val so the CNN can't collapse to
        # it; test stays natural (see python/balance.py).
        df = balance_df(df, seed=cfg["seed"])
    model, hist = train_from_df(df, classes=cfg["classes"], signal_type=signal_type,
                                image_size=cfg["image_size"], batch_size=32,
                                epochs=epochs, seed=cfg["seed"], arch=arch)
    os.makedirs(cfg["paths"]["models"], exist_ok=True)
    os.makedirs(cfg["paths"]["results"], exist_ok=True)
    suffix = "" if arch == "baseline" else f"_{arch}"   # keep baseline filenames stable
    model.save(os.path.join(cfg["paths"]["models"], f"cnn_{signal_type}{suffix}.keras"))
    with open(os.path.join(cfg["paths"]["results"], f"history_{signal_type}{suffix}.json"), "w") as f:
        json.dump(hist.history, f)
    return model, hist


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal-type", default="current")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--arch", default="baseline", choices=["baseline", "modern", "transfer"])
    a = ap.parse_args()
    train(load_config(), signal_type=a.signal_type, epochs=a.epochs, arch=a.arch)
