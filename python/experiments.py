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


def run_single(cfg, *, signal_type, image_size, balance, train_frac, epochs=25,
               filters=(32, 64, 128), arch="baseline"):
    """Train one model and return its test metrics (run inside a subprocess)."""
    from python.evaluate import confusion_and_report
    from python.data_loader import make_dataset

    full = load_manifest(cfg["paths"]["manifest"])
    classes, seed = cfg["classes"], cfg["seed"]
    df = balance_df(full, seed=seed) if balance else full.copy()
    df = _subsample_train(df, train_frac, seed)
    n_train = int(((df["split"] == "train") & (df["signal_type"] == signal_type)).sum())
    model, _ = train_from_df(df, classes=classes, signal_type=signal_type,
                             image_size=image_size, batch_size=32, epochs=epochs, seed=seed,
                             filters=tuple(filters), arch=arch)
    ds, _ = make_dataset(full, "test", signal_type, classes, image_size, 32, seed)
    _cm, rep = confusion_and_report(model, ds, classes)
    acc = rep.get("accuracy", rep.get("micro avg", {}).get("f1-score"))
    return {
        "channel": signal_type, "image_size": image_size, "balanced": balance,
        "train_frac": train_frac, "n_train": n_train, "arch": arch,
        "n_blocks": len(filters), "filters": list(filters),
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
    # 4) depth — number of conv blocks (both channels, balanced, 224 px)
    for st in ["current", "vibration"]:
        for filt in [[32, 64], [32, 64, 128], [32, 64, 128, 256]]:
            specs.append({"study": "depth", "signal_type": st, "image_size": 224,
                          "balance": True, "train_frac": 1.0, "filters": filt})
    # 5) architecture — baseline vs modernized vs transfer (both channels, balanced)
    for st in ["current", "vibration"]:
        for arch in ["baseline", "modern", "transfer"]:
            specs.append({"study": "architecture", "signal_type": st, "image_size": 224,
                          "balance": True, "train_frac": 1.0, "arch": arch})
    return specs


def _spec_key(study, d):
    # identify a run from either a spec or a stored result (keys differ slightly)
    st = d.get("signal_type", d.get("channel"))
    bal = d.get("balance", d.get("balanced"))
    return (study, st, d.get("image_size"), bal, d.get("train_frac"),
            tuple(d.get("filters") or ()), d.get("arch", "baseline"))


def main(cfg):
    out = {"balancing": [], "image_size": [], "learning_curve": [], "depth": [], "architecture": []}
    os.makedirs(cfg["paths"]["results"], exist_ok=True)
    out_path = os.path.join(cfg["paths"]["results"], "experiments_real.json")
    # Resume: reload prior checkpoint and skip runs already completed successfully.
    done = set()
    if os.path.exists(out_path):
        try:
            prev = json.load(open(out_path))
            for study, rows in prev.items():
                out.setdefault(study, [])
                for r in rows:
                    out[study].append(r)
                    if "balanced_acc" in r:  # only count successful runs as done
                        done.add(_spec_key(study, r))
            print(f"[resume] {len(done)} runs already complete; skipping them", flush=True)
        except Exception:
            out = {"balancing": [], "image_size": [], "learning_curve": [], "depth": [], "architecture": []}
    for spec in _specs():
        study = spec["study"]
        if _spec_key(study, spec) in done:
            continue
        try:
            res = _run_in_subprocess(spec)
        except Exception as e:  # one bad run shouldn't sink the whole sweep
            res = {**{k: spec[k] for k in ("signal_type", "image_size", "balance", "train_frac")},
                   "error": str(e)[:200]}
        out[study].append(res)
        print(study, res, flush=True)
        # Checkpoint after every run so a suspend/kill never loses completed work.
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
    _plot_learning_curve([r for r in out["learning_curve"] if "balanced_acc" in r],
                         cfg["paths"]["results"])
    _plot_depth([r for r in out.get("depth", []) if "balanced_acc" in r], cfg["paths"]["results"])
    _plot_arch([r for r in out.get("architecture", []) if "balanced_acc" in r], cfg["paths"]["results"])
    return out


def _plot_xy(rows, xkey, xlabel, title, fname, results_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    for st in ["current", "vibration"]:
        pts = sorted([r for r in rows if r["channel"] == st], key=lambda r: r[xkey])
        if pts:
            ax.plot([r[xkey] for r in pts], [r["balanced_acc"] for r in pts], marker="o", label=st)
    ax.set_xlabel(xlabel); ax.set_ylabel("balanced accuracy (test)")
    ax.set_ylim(0.4, 1.05); ax.set_title(title)
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(results_dir, fname), dpi=120)


def _plot_learning_curve(rows, results_dir):
    _plot_xy(rows, "n_train", "training images",
             "Learning curve — effect of training-set size", "learning_curve.png", results_dir)


def _plot_depth(rows, results_dir):
    if rows:
        _plot_xy(rows, "n_blocks", "number of conv blocks (depth)",
                 "Depth ablation — effect of #conv layers", "depth_ablation.png", results_dir)


def _plot_arch(rows, results_dir):
    if not rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    archs = ["baseline", "modern", "transfer"]
    chans = ["current", "vibration"]
    import numpy as np
    fig, ax = plt.subplots(figsize=(6.5, 4))
    w = 0.35
    for i, ch in enumerate(chans):
        vals = [next((r["balanced_acc"] for r in rows if r["channel"] == ch and r.get("arch") == a), 0) for a in archs]
        ax.bar(np.arange(len(archs)) + i * w, vals, w, label=ch)
    ax.set_xticks(np.arange(len(archs)) + w / 2); ax.set_xticklabels(archs)
    ax.set_ylabel("balanced accuracy (test)"); ax.set_ylim(0, 1.05)
    ax.set_title("Architecture ablation — baseline vs modern vs transfer")
    ax.legend(); ax.grid(alpha=0.3, axis="y"); fig.tight_layout()
    fig.savefig(__import__("os").path.join(results_dir, "architecture_ablation.png"), dpi=120)


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
