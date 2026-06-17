# Design Spec — PMSM Fault Diagnosis via Wavelet Scalogram + CNN

**Date:** 2026-06-17
**Authors / students:** Mulham Fetna (ملهم فتنة), Mohammad Zein Qabbani (محمد زين قباني)
**Source requirements:** `project-discription-ar.md`

## 1. Goal

Build an intelligent fault-diagnosis pipeline for Permanent Magnet Synchronous Motors (PMSM):
convert motor signals → **Continuous Wavelet Transform (CWT)** → **Wavelet Scalogram images** →
train a **Convolutional Neural Network (CNN)** to classify operating state and detect faults.

Deliverables (from the spec): labelled image dataset, training code, final CNN model, results/plots,
a 25–35 page report, and a 15–20 slide presentation. Total timeline: 6–8 weeks.

## 2. Locked decisions

| Decision | Choice |
|---|---|
| Toolchain | **Hybrid**: MATLAB for signal simulation + CWT scalogram generation; Python (TensorFlow/Keras) for the CNN |
| Data source | **Both** simulation (FOC + Simscape models in this repo) **and** real public datasets |
| Signal types | **Both** stator current (MCSA) and vibration — handled as separate channels |
| Fault classes | **Multi-fault (4+)**: `Healthy`, `InterTurn`, `Demagnetization`, `Overload` (+ optional inter-turn severity sub-levels) |
| Data fallback | **Per-channel datasets** — if no single real dataset has both current and vibration, use one dataset per channel and report each model independently |
| Deliverable format | Design spec (this doc) + a detailed phased implementation plan |

## 3. Architecture

Five stages, MATLAB → Python handoff via image folders + a manifest file.

```
                 ┌─ Simulation (MATLAB) ──┐
  Stage 1        │ FOC_PMSM-main (current)│      Stage 2 (MATLAB)        Stage 3
  SIGNAL  ───────┤ simscape-pmsm (fault)  ├────► segment + CWT  ────►  scalogram PNGs
  SOURCES        │                        │      (Wavelet Toolbox)     foldered by class
                 └─ Real datasets ────────┘                                  │
                   (current / vibration)                                     │
                                                                    Stage 4 (Python)
                                                              ┌───────────────┴───────────┐
                                                              │ CNN: train / eval / tune  │
                                                              │ per channel, then fusion  │
                                                              └───────────────┬───────────┘
                                                                       Stage 5
                                                               results, report, slides
```

### 3.1 Components & interfaces

- **Signal sources (Stage 1).** Two origins, one normalized output format.
  - *Simulation:* `FOC_PMSM-main/` (run `Motor_script.m` then `FOCsimulation.slx`; logged signal at `out.simout.signals.values`) for current under controlled load; `simscape-pmsm/` `FaultyPMSM.ssc` for inter-turn fault (`sigma` = ratio of shorted turns) after `ssc_build`. Caveats: faulty model is validated only for 1 pole-pair and may not converge in sensorless FOC — see §6.
  - *Real datasets:* selected in Phase 0 from `references.md` (Kaggle/Mendeley/Zenodo/IEEE-DataPort). Per-channel: a current-fault dataset and (separately) a vibration-fault dataset.
  - **Interface:** every signal is exported as a uniform record — a 1-D time series + metadata (sampling rate `fs`, class, severity, source, signal_type).

- **Manifest (`data/manifest.csv`).** Single source of truth. One row per signal segment:
  `signal_id, source(sim|real), signal_type(current|vibration), class, severity, fs, dataset_name, split(train|val|test), scalogram_path`.
  Train/val/test split is assigned here, **grouped by raw recording** to prevent leakage (segments from one recording never span splits).

- **CWT / Scalogram generator (Stage 2, MATLAB).** Input: a signal record. Process: segment into fixed-length windows (with overlap), apply CWT (Morlet/`amor` default), render the magnitude scalogram to a fixed-size RGB PNG. Output: PNG written to `data/scalograms/<signal_type>/<class>/<signal_id>.png` and the path recorded in the manifest. Deterministic params captured in a config block (window length, overlap, wavelet, frequency range, image size).

- **CNN trainer (Stage 4, Python/Keras).** Input: scalogram folders + manifest split. Loads images via the manifest (not just folder walking, so splits are honored). Trains per channel first; fusion (dual-branch or 2-channel stack) only after single-channel models work. Output: trained model, training curves, confusion matrix, classification report.

### 3.2 Repository layout (to be created)

```
uni-plc-lab/
├── data/
│   ├── raw/            # downloaded real datasets + exported sim signals (gitignored / large)
│   ├── scalograms/     # generated PNGs, foldered by signal_type/class
│   └── manifest.csv
├── matlab/
│   ├── sim/            # scripts to run FOC/Simscape and export signals
│   └── scalogram/      # CWT + image generation scripts
├── python/
│   ├── data_loader.py  # reads manifest, builds tf.data pipelines
│   ├── model.py        # CNN architecture(s)
│   ├── train.py
│   └── evaluate.py
├── models/             # saved CNN weights
├── results/            # plots, confusion matrices, metrics
└── docs/               # this spec, report drafts
```

## 4. Phasing (de-risked, MVP-first)

- **Phase 0 — Foundations & data audit.** Confirm real datasets actually contain the needed signals/labels (highest risk). Pick the current dataset and the vibration dataset. Set conventions (sampling, window length, class names, image size). Create repo structure + empty manifest.
- **Phase 1 — MVP.** One real current dataset → scalograms → baseline CNN on 3 classes. Proves the entire chain end-to-end before adding complexity.
- **Phase 2 — Simulation data.** Generate current signals from FOC + faulty Simscape across fault types/severities; export and merge into the manifest. Compare sim vs real scalograms visually.
- **Phase 3 — Expand classes + vibration channel.** Add Demagnetization + severities; build the vibration scalogram set and a vibration CNN.
- **Phase 4 — Fusion, tuning, evaluation.** Channel fusion; hyperparameter tuning; overfitting control (augmentation, dropout, early stopping); full evaluation (accuracy, confusion matrix, per-class metrics, ablations).
- **Phase 5 — Deliverables.** Report (25–35 pp) and presentation (15–20 slides).

## 5. Success criteria

- End-to-end pipeline reproducible from a documented command sequence.
- A current-signal CNN with reported accuracy + confusion matrix on a held-out test set (leakage-controlled split).
- At least the 4 target classes represented across the combined dataset.
- A vibration-signal CNN reported independently (per-channel fallback honored).
- Simulation contributes controlled fault/severity data, visually and quantitatively compared to real.
- Report + slides covering theory (PMSM, faults, wavelet/scalogram, CNN), method, and results.

## 6. Risks & mitigations

- **No single real dataset with both channels** → per-channel datasets, independent models (accepted).
- **Simscape faulty model limits** (1 pole-pair, sensorless non-convergence) → use it open-loop / simple-drive for fault current generation; do not block on integrating it into full sensorless FOC. Real data covers what sim can't.
- **Data leakage** across segments → group-wise split enforced in the manifest.
- **Class imbalance** (sim easy to over-generate) → cap per-class counts; report class distribution; use stratified metrics.
- **Overfitting** on scalogram texture → augmentation, dropout, early stopping, and sim-vs-real cross-checks.
- **MATLAB not runnable in this environment** → all MATLAB steps delivered as scripts the user runs; confirm Wavelet, Deep Learning, Simscape Electrical toolboxes are licensed.

## 7. Out of scope

Real-time/embedded deployment, hardware-in-the-loop, sensorless-FOC integration of the faulty model, and non-PMSM motors.
