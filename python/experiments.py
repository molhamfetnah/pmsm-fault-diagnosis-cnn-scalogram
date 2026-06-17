import json
import os
from python.config import load_config
from python.manifest import load_manifest
from python.train import train_from_df

GRID = [{"lr": 1e-3, "dropout": 0.5, "augment": True},
        {"lr": 1e-4, "dropout": 0.3, "augment": True},
        {"lr": 1e-3, "dropout": 0.5, "augment": False}]


def main(cfg, signal_type="current"):
    df = load_manifest(cfg["paths"]["manifest"])
    rows = []
    for g in GRID:
        _, hist = train_from_df(df, classes=cfg["classes"], signal_type=signal_type,
                                image_size=cfg["image_size"], batch_size=32, epochs=20, seed=cfg["seed"])
        rows.append({**g, "val_acc": max(hist.history["val_accuracy"])})
    os.makedirs(cfg["paths"]["results"], exist_ok=True)
    with open(f"{cfg['paths']['results']}/experiments_{signal_type}.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(rows)


if __name__ == "__main__":
    main(load_config())
