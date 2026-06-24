"""Does *generated* data help the weak real current channel? Benchmarks, on the
real KAIST current test set (held-out recordings), three conditions vs the baseline:

  baseline        : train build_cnn on real current (standard flip aug)
  augment         : same, with SpecAugment (time/freq masking + noise)
  synth_pretrain  : pre-train build_cnn on a rich, diverse SYNTHETIC dataset
                    (varied load/speed/SNR/severity across 4 classes), then
                    fine-tune on the real current train split

Honest scope: synthetic data cannot validate the real result (it's our own model of
"healthy"); this measures only whether it helps as pretraining/augmentation.

Run: .venv/bin/python -m python.transfer_experiment   ->  results/transfer_experiment.json (+ .png)
"""
import os
import json
import numpy as np

from python.config import load_config
from python.manifest import load_manifest
from python.balance import balance_df
from python.simulate import build_signal
from python.scalogram import compute_scalogram


def _synth_xy(cfg, n_per_class=80, duration=0.5, seed=123):
    """Generate a diverse synthetic scalogram set (many 'virtual motors')."""
    import matplotlib.cm as cm
    from PIL import Image
    classes, fs, size = cfg["classes"], cfg["target_fs"], cfg["image_size"]
    wav, nsc = cfg.get("wavelet_py", "cmor1.5-1.0"), cfg.get("n_scales", 128)
    X, Y = [], []
    rng = np.random.default_rng(seed)
    for ci, klass in enumerate(classes):
        for _ in range(n_per_class):
            sev = float(rng.uniform(0.1, 0.9)) if klass in ("InterTurn", "Demagnetization") else 0.0
            sig = build_signal(klass, sev, fs=fs, duration=duration, rng=rng,
                               f0=float(rng.uniform(45, 55)), fr=float(rng.uniform(8, 16)),
                               load=float(rng.uniform(0.6, 1.4)), noise=float(rng.uniform(0.03, 0.12)))
            A = compute_scalogram(sig, fs, wavelet=wav, n_scales=nsc)
            A = A / (A.max() + 1e-12)
            img = Image.fromarray((cm.jet(A)[..., :3] * 255).astype("uint8")).resize((size, size))
            X.append(np.array(img) / 255.0); Y.append(ci)
    return np.asarray(X, "float32"), np.asarray(Y, "int64")


def _eval(model, df, cfg, signal_type="current"):
    from python.evaluate import confusion_and_report
    from python.data_loader import make_dataset
    ds, _ = make_dataset(df, "test", signal_type, cfg["classes"], cfg["image_size"], 32, cfg["seed"])
    _cm, rep = confusion_and_report(model, ds, cfg["classes"])
    return {"balanced_acc": round((rep["Healthy"]["recall"] + rep["InterTurn"]["recall"]) / 2, 3),
            "macro_f1": round((rep["Healthy"]["f1-score"] + rep["InterTurn"]["f1-score"]) / 2, 3),
            "healthy_recall": round(rep["Healthy"]["recall"], 3),
            "interturn_recall": round(rep["InterTurn"]["recall"], 3)}


def run(cfg, signal_type="current", epochs=25, pre_epochs=12):
    from tensorflow import keras
    from python.model import build_cnn
    from python.train import train_from_df

    full = load_manifest(cfg["paths"]["manifest"])
    bal = balance_df(full, seed=cfg["seed"])
    classes, size = cfg["classes"], cfg["image_size"]
    out = {}

    # 1) baseline (flip aug)
    m, _ = train_from_df(bal, classes=classes, signal_type=signal_type, image_size=size,
                         batch_size=32, epochs=epochs, seed=cfg["seed"], aug_mode="flip")
    out["baseline"] = _eval(m, full, cfg, signal_type)

    # 2) SpecAugment
    m, _ = train_from_df(bal, classes=classes, signal_type=signal_type, image_size=size,
                         batch_size=32, epochs=epochs, seed=cfg["seed"], aug_mode="spec")
    out["augment"] = _eval(m, full, cfg, signal_type)

    # 3) synthetic pre-train -> real fine-tune
    Xs, Ys = _synth_xy(cfg)
    pre = build_cnn(input_shape=(size, size, 3), num_classes=len(classes))
    pre.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    pre.fit(Xs, Ys, validation_split=0.15, epochs=pre_epochs, batch_size=32,
            callbacks=[keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)], verbose=0)
    from python.data_loader import make_dataset
    tr, _ = make_dataset(bal, "train", signal_type, classes, size, 32, cfg["seed"], augment=True)
    va, _ = make_dataset(bal, "val", signal_type, classes, size, 32, cfg["seed"])
    pre.compile(optimizer=keras.optimizers.Adam(1e-4), loss="sparse_categorical_crossentropy",
                metrics=["accuracy"])  # lower LR for fine-tuning
    pre.fit(tr, validation_data=va, epochs=epochs,
            callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)], verbose=0)
    out["synth_pretrain"] = _eval(pre, full, cfg, signal_type)

    os.makedirs(cfg["paths"]["results"], exist_ok=True)
    json.dump(out, open(os.path.join(cfg["paths"]["results"], "transfer_experiment.json"), "w"), indent=2)
    _plot(out, cfg["paths"]["results"])
    for k, v in out.items():
        print(k, v)
    return out


def _plot(out, results_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    conds = ["baseline", "augment", "synth_pretrain"]
    vals = [out[c]["balanced_acc"] for c in conds]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(conds, vals, color=["#7f8c8d", "#2c6fbb", "#2ecc71"])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center")
    ax.set_ylim(0, 1.05); ax.set_ylabel("balanced accuracy (real current test)")
    ax.set_title("Generated-data strategies on the real current channel")
    fig.tight_layout(); fig.savefig(os.path.join(results_dir, "transfer_experiment.png"), dpi=120)


if __name__ == "__main__":
    run(load_config())
