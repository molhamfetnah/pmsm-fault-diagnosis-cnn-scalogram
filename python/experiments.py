"""Ablation experiments on the real KAIST data (requirement subtasks 5 & 6).

Each run trains the baseline CNN and evaluates on the **natural (imbalanced)
held-out test set**, reporting balanced accuracy and 2-class macro-F1 (raw
accuracy is misleading under imbalance). Three studies:

  1. balancing      — train/val balanced vs. raw (shows majority-class collapse)
  2. image_size     — 96 vs 160 vs 224 px (scalograms are downscaled at load)
  3. learning curve — 25 / 50 / 100 % of the training set (effect of data volume)

Each configuration is trained in its **own subprocess** so TensorFlow releases
all memory between runs (training ~18 models in one process exhausts RAM).

Results -> results/experiments_real.json and results/learning_curve.png.
Run: .venv/bin/python -m python.experiments
"""
import json
import os
import subprocess
import sys

import pandas as pd

from python.config import load_config
from python.manifest import load_manifest
from python.balance import balance_df
from python.train import train_from_df


# ---------------------------------------------------------------- single run --
def _subsample_train(df, frac, seed):
    if frac >= 1.0:
        return df
    tr = df[df["split"] == "train"]
    keep = tr.groupby("class", group_keys=False).apply(
        lambda g: g.sample(frac=frac, random_state=seed))
    return pd.concat([keep, df[df["split"] != "train"]], ignore_index=True)


def run_single(cfg, *, signal_type, image_size, balance, train_frac, epochs=25):
    """Train one model and return its test metrics (run inside a subprocess)."""
    from python.evaluate import confusion_and_report
    from python.data_loader import make_dataset

    full = load_manifest(cfg["paths"]["manifest"])
    classes, seed = cfg["classes"], cfg["seed"]
    df = balance_df(full, seed=seed) if balance else full.copy()
    df = _subsample_train(df, train_frac, seed)
    n_train = int(((df["split"] == "train") & (df["signal_type"] == signal_type)).sum())
    model, _ = train_from_df(df, classes=classes, signal_type=signal_type,
                             image_size=image_size, batch_size=32, epochs=epochs, seed=seed)
    ds, _ = make_dataset(full, "test", signal_type, classes, image_size, 32, seed)
    _cm, rep = confusion_and_report(model, ds, classes)
    acc = rep.get("accuracy", rep.get("micro avg", {}).get("f1-score"))
    return {
        "channel": signal_type, "image_size": image_size, "balanced": balance,
        "train_frac": train_frac, "n_train": n_train,
        "accuracy": round(acc, 3),
        "balanced_acc": round((rep["Healthy"]["recall"] + rep["InterTurn"]["recall"]) / 2, 3),
        "macro_f1": round((rep["Healthy"]["f1-score"] + rep["InterTurn"]["f1-score"]) / 2, 3),
        "healthy_recall": round(rep["Healthy"]["recall"], 3),
        "interturn_recall": round(rep["InterTurn"]["recall"], 3),
    }


# -------------------------------------------------------------- orchestration --
def _run_in_subprocess(spec):
    """Invoke this module with --single to train one config in a fresh process."""
    cmd = [sys.executable, "-m", "python.experiments", "--single", json.dumps(spec)]
    out = subprocess.run(cmd, capture_output=True, text=True)
    for line in out.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):])
    print(out.stdout[-2000:]); print(out.stderr[-2000:])
    raise RuntimeError(f"subprocess failed for {spec}")


def _specs():
    specs = []
    # 1) balancing on/off (both channels, 224 px)
    for st in ["current", "vibration"]:
        for bal in [False, True]:
            specs.append({"study": "balancing", "signal_type": st, "image_size": 224,
                          "balance": bal, "train_frac": 1.0})
    # 2) image size (both channels, balanced)
    for st in ["current", "vibration"]:
        for sz in [96, 160, 224]:
            specs.append({"study": "image_size", "signal_type": st, "image_size": sz,
                          "balance": True, "train_frac": 1.0})
    # 3) learning curve (both channels, balanced, 224 px)
    for st in ["current", "vibration"]:
        for frac in [0.25, 0.5, 1.0]:
            specs.append({"study": "learning_curve", "signal_type": st, "image_size": 224,
                          "balance": True, "train_frac": frac})
    return specs


def main(cfg):
    out = {"balancing": [], "image_size": [], "learning_curve": []}
    for spec in _specs():
        study = spec["study"]
        res = _run_in_subprocess(spec)
        out[study].append(res)
        print(study, res, flush=True)
    os.makedirs(cfg["paths"]["results"], exist_ok=True)
    with open(os.path.join(cfg["paths"]["results"], "experiments_real.json"), "w") as f:
        json.dump(out, f, indent=2)
    _plot_learning_curve(out["learning_curve"], cfg["paths"]["results"])
    return out


def _plot_learning_curve(rows, results_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    for st in ["current", "vibration"]:
        pts = sorted([r for r in rows if r["channel"] == st], key=lambda r: r["n_train"])
        ax.plot([r["n_train"] for r in pts], [r["balanced_acc"] for r in pts],
                marker="o", label=st)
    ax.set_xlabel("training images"); ax.set_ylabel("balanced accuracy (test)")
    ax.set_ylim(0.4, 1.05); ax.set_title("Learning curve — effect of training-set size")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "learning_curve.png"), dpi=120)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", help="JSON spec for a single run (internal)")
    a = ap.parse_args()
    cfg = load_config()
    if a.single:
        spec = json.loads(a.single)
        spec.pop("study", None)
        res = run_single(cfg, **spec)
        print("RESULT " + json.dumps(res))
    else:
        main(cfg)
