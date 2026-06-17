import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from python.config import load_config
from python.manifest import load_manifest
from python.data_loader import make_dataset


def metrics_from_predictions(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    report = classification_report(y_true, y_pred, target_names=class_names,
                                   output_dict=True, zero_division=0)
    return cm, report


def confusion_and_report(model, ds, class_names):
    y_true, y_pred = [], []
    for x, y in ds:
        p = model.predict(x, verbose=0).argmax(axis=1)
        y_true.extend(y.numpy().tolist())
        y_pred.extend(p.tolist())
    return metrics_from_predictions(np.array(y_true), np.array(y_pred), class_names)


def main(cfg, signal_type="current"):
    # Lazy import: keeps metrics_from_predictions testable without TensorFlow.
    from tensorflow import keras

    df = load_manifest(cfg["paths"]["manifest"])
    ds, names = make_dataset(df, "test", signal_type, cfg["classes"],
                             cfg["image_size"], 32, cfg["seed"])
    model = keras.models.load_model(os.path.join(cfg["paths"]["models"], f"cnn_{signal_type}.keras"))
    cm, report = confusion_and_report(model, ds, names)
    fig, ax = plt.subplots()
    im = ax.imshow(cm)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(os.path.join(cfg["paths"]["results"], f"confusion_{signal_type}.png"))
    with open(os.path.join(cfg["paths"]["results"], f"report_{signal_type}.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"accuracy={report['accuracy']:.3f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal-type", default="current")
    a = ap.parse_args()
    main(load_config(), signal_type=a.signal_type)
