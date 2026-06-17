# PMSM Fault Diagnosis using Wavelet Scalograms and CNNs

A reproducible research pipeline that diagnoses faults in **Permanent Magnet
Synchronous Motors (PMSM)** by turning motor signals into **Continuous Wavelet
Transform (CWT) scalogram images** and classifying them with a **Convolutional
Neural Network (CNN)**.

> **Status.** The complete software pipeline is implemented, tested, and runs
> end-to-end without MATLAB. It ships with a synthetic signal generator for
> instant validation, and loaders for real public PMSM datasets. See
> [Results](#results) for the important caveat about the synthetic numbers.

Authors: **Mulham Fetna**, **Mohammad Zein Qabbani** — university research project.

---

## 1. Method

```
 signals ──► segment ──► CWT ──► scalogram PNG ──► CNN ──► fault class
 (current /            (window)  (Morlet)  (224×224)        Healthy / InterTurn /
  vibration)                                                Demagnetization / Overload
```

Signals come from three interchangeable sources, unified by a single
`data/manifest.csv` that links every signal segment → its scalogram → its label
and train/val/test split:

| Source | Channel(s) | Tooling | Status |
|---|---|---|---|
| **Synthetic generator** | current | Python (`python/simulate.py`) | included, runs anywhere |
| **Real datasets** | current + vibration | Python loaders | loaders ready; data downloaded by user |
| **Physics simulation** | current | MATLAB (FOC + Simscape) | scripts in `matlab/` |

Scalograms can be rendered either in **Python** (`python/scalogram.py`,
PyWavelets — no toolboxes needed) or in **MATLAB** (`matlab/scalogram/`, Wavelet
Toolbox). Both write the same `data/scalograms/<channel>/<class>/<id>.png` layout.

### Example scalograms (synthetic current)

![Example scalograms](results/example_scalograms.png)

The 50 Hz fundamental dominates the low-frequency band; the demagnetization and
overload signatures add visible sidebands and harmonics.

---

## 2. Quickstart

Requires **Python 3.10–3.12** (TensorFlow has no wheel for 3.13/3.14 yet).

```bash
make setup     # create .venv on python3.10 and install requirements
make test      # run the test suite (should be all green)
make demo      # synthetic end-to-end: simulate → scalograms → train → evaluate
```

`make demo` writes a trained model to `models/`, and metrics + a confusion
matrix to `results/`.

Run stages individually:

```bash
make simulate                       # synthetic signals  -> data/raw/sim/
make scalograms                     # CWT PNGs           -> data/scalograms/
make split                          # leakage-free splits in the manifest
make train SIGNAL=current EPOCHS=20 # -> models/cnn_current.keras
make evaluate SIGNAL=current        # -> results/confusion_current.png, report
make report                         # -> results/summary.md
```

---

## 3. Using real data

### 3.1 KAIST PMSM stator-fault dataset (recommended — current + vibration)

"Vibration and Current Dataset of Three-Phase PMSM with Stator Faults"
([Mendeley `rgn5brrgrn`](https://data.mendeley.com/datasets/rgn5brrgrn/5),
DOI `10.17632/rgn5brrgrn.5`, CC-BY-4.0). Current @ 100 kHz, vibration @ 25.6 kHz,
conditions: normal / inter-turn / inter-coil short, motors at 1.0/1.5/3.0 kW,
stored as `.tdms`.

```bash
# 1. Download .tdms files into data/raw/mendeley_pmsm/
#    (filenames like 1000W_0_00_current_interturn.tdms)
# 2. Ingest, decimating current 100 kHz -> target_fs:
.venv/bin/python -m python.ingest_mendeley
make scalograms split
make train SIGNAL=current   && make evaluate SIGNAL=current
make train SIGNAL=vibration && make evaluate SIGNAL=vibration
```

### 3.2 Inverter-fault dataset (Zenodo, tabular)

"Comprehensive Dataset for Fault Detection and Diagnosis in Inverter-Driven PMSM
Systems" ([Zenodo `13974503`](https://zenodo.org/records/13974503), CC-BY-4.0).
Useful as a real-world reference, **but sampled at 10 Hz** — too low for
meaningful scalograms — so it is documented in `docs/data-audit.md` rather than
wired into the CWT pipeline.

### 3.3 MATLAB physics simulation

`matlab/sim/` exports stator current from the Field-Oriented-Control model
(`FOC_PMSM-main/`) and the Simscape inter-turn fault model (`simscape-pmsm/`).
See `docs/superpowers/plans/2026-06-17-pmsm-cnn-scalogram.md` Phase 2 for the
exact steps and the model caveats (1 pole-pair validity, open-loop drive).

---

## 4. Repository layout

```
python/              # the pipeline (one responsibility per module)
  config.py          # config.yaml loader + validation
  manifest.py        # the data manifest (single source of truth)
  split.py           # leakage-free, group-wise, stratified train/val/test split
  simulate.py        # synthetic PMSM signal generator (MCSA fault signatures)
  ingest_sim.py      # ingest simulated/synthetic .npy signals
  ingest_real.py     # template loader for a generic real current dataset
  ingest_mendeley.py # loader for the KAIST TDMS dataset (current + vibration)
  ingest_vibration.py# template loader for a generic vibration dataset
  scalogram.py       # Python CWT scalogram rendering (PyWavelets)
  data_loader.py     # manifest -> tf.data pipeline
  model.py           # baseline CNN + dual-branch fusion CNN
  train.py           # training entrypoint
  evaluate.py        # confusion matrix + classification report
  experiments.py     # hyperparameter / overfitting grid
  report_metrics.py  # consolidated results/summary.md
  tests/             # pytest suite
matlab/              # MATLAB alternatives: CWT scalograms + FOC/Simscape export
docs/                # design spec, implementation plan, data audit, report/slides
FOC_PMSM-main/       # third-party Simulink FOC model (signal source)
simscape-pmsm/       # Simscape inter-turn fault model (separate git repo)
config.yaml          # all pipeline parameters
Makefile             # convenience targets
```

---

## 5. Results

On the **synthetic** dataset (4 classes, 456 segments, leakage-free split) the
baseline CNN reaches **100% test accuracy**.

![Confusion matrix](results/confusion_current.png)

**Interpretation — read this.** A perfect score here means the *software
pipeline is correct end-to-end* (signal → scalogram → CNN → metrics, with no
train/test leakage), **not** that the problem is hard. The synthetic fault
signatures are deliberately separable. Meaningful difficulty — and the numbers
worth reporting in the paper — come from the **real** datasets in §3. The
pipeline is identical for synthetic and real data; only the ingestion step
differs.

---

## 6. Testing

```bash
make test          # or: .venv/bin/python -m pytest -q
```

Pure-logic modules (config, manifest, split, segmentation, filename parsing,
metrics, scalogram shape) are unit-tested. TensorFlow-dependent tests run when
TF is installed and skip cleanly otherwise.

---

## 7. Configuration

All parameters live in `config.yaml`: class set, channels, window length /
overlap, target sampling rate, image size, wavelet, number of CWT scales, RNG
seed, and paths. Every module reads from it — change behaviour there, not in code.

---

## 8. Citation & references

If you use the real datasets, cite their authors:

- KAIST PMSM stator-fault dataset — *Vibration and current dataset of
  three-phase PMSM with stator faults*, Data in Brief (2023). DOI
  `10.17632/rgn5brrgrn.5`.
- Inverter-fault dataset — Zenodo `10.5281/zenodo.13974503`.

Background reading and the full source list are in `references.md`; the design
rationale is in `docs/superpowers/specs/` and `docs/superpowers/plans/`.

## 9. License

Code: [MIT](LICENSE). Datasets retain their own (CC-BY-4.0) licenses.
