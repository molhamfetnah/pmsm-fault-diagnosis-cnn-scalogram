# PMSM Fault Diagnosis using Wavelet Scalograms and CNN

> Status: **OUTLINE — to be written once results exist.** Target length 25–35 pages.
> Pull theory from `notes.md`, figures from `results/`, and experiment data from
> `docs/experiments.md`. Structure mirrors the requirements in
> `project-discription-ar.md`.

## 1. Introduction
- Motivation: intelligent fault diagnosis for electric motors.
- PMSM operating principle (BLDC vs PMSM, FOC). _Source: notes.md._
- Common PMSM fault types: inter-turn (stator winding short), demagnetization,
  overload, bearing/mechanical.

## 2. Signal Processing Background
- Time vs frequency domain; the uncertainty principle.
- Fourier transform limitations for non-stationary signals.

## 3. Wavelet Transform and Scalograms
- Continuous Wavelet Transform; mother/daughter wavelets; Morlet.
- The scalogram as a time–frequency power image; Heisenberg boxes / blur.
- Example scalograms per class (figure: `results/sim_vs_real_grid.png`).

## 4. Convolutional Neural Networks
- Conv / ReLU / pooling / flatten / dense pipeline. _Source: notes.md._
- Why CNNs suit scalogram images (spatial patterns, parameter sharing).

## 5. Data Preparation
- Signal sources: simulation (FOC + Simscape inter-turn fault) and real datasets.
- Per-channel handling (current, vibration); segmentation; CWT → 224×224 PNG.
- Manifest, leakage-free group-wise train/val/test split.
- Dataset composition table (from `results/summary.md`).

## 6. CNN Model and Training
- Architecture (baseline CNN; dual-branch fusion). Diagram.
- Training setup: optimizer, loss, augmentation, early stopping.

## 7. Results and Analysis

### 7.1 Real data — KAIST PMSM (Healthy vs Inter-turn)

Source: Mendeley `rgn5brrgrn` (1.0 + 1.5 kW), 3,150 scalogram segments (current
1,550 / vibration 1,600). Train/val class-balanced by majority undersampling;
**test split natural and on held-out recordings** (grouped by `recording_id`,
leakage-free). Headline metrics are balanced accuracy and 2-class macro-F1
because the test set is imbalanced (50 Healthy / 200 Inter-turn).

| Channel   | Test acc | Balanced acc | Macro-F1 | Healthy recall | Inter-turn recall |
|-----------|----------|--------------|----------|----------------|-------------------|
| Vibration | 1.00     | 1.00         | 1.00     | 1.00           | 1.00              |
| Current   | 0.50     | 0.69         | 0.49     | 1.00           | 0.37              |

Confusion matrices: `results/confusion_real_2class.png`. Example scalograms:
`results/example_scalograms_real.png`. Raw metrics: `results/real_metrics.json`.

**Analysis.** Vibration scalograms separate the two states perfectly on held-out
recordings; current scalograms do not — the current-channel model detects all
healthy cases but misses ~63 % of inter-turn faults, i.e. the inter-turn
signature is weak in the stator current at the available severities relative to
vibration. This channel ranking matches the motor-fault-diagnosis literature.

**Limitation.** Only 4 distinct healthy recordings exist (2/1/1 across splits),
so the perfect vibration score cannot exclude a recording-identity shortcut;
generalization needs more independent healthy recordings. Demagnetization and
Overload do not appear in the real dataset.

### 7.2 Synthetic data — software validation (4 classes)

The synthetic generator (Healthy / Inter-turn / Demagnetization / Overload, 456
segments) yields 100 % test accuracy. This validates the end-to-end software
(signal → scalogram → CNN → metrics, no leakage), not task difficulty: the
synthetic fault signatures are deliberately separable.

### 7.3 Effect of class balancing

Without balancing, the current-channel model collapses to the majority class
(accuracy 0.80 = base rate, Healthy recall 0.00). Undersampling the majority in
train/val removes the collapse (Healthy recall → 1.00) at the cost of inter-turn
recall, exposing the channel's true weak separability rather than hiding it
behind a misleading accuracy. Future work: focal loss / threshold tuning,
severity-aware multi-class targets, and current+vibration fusion.

## 8. Conclusions
- Vibration-based CWT scalograms + CNN detect inter-turn stator faults with high
  accuracy on this dataset; current-based scalograms are markedly weaker.
- Reporting balanced accuracy / macro-F1 (not raw accuracy) is essential given
  the class imbalance.
- Main limitation: too few independent healthy recordings to claim
  generalization; main next step is acquiring more healthy/fault recordings and a
  paired multi-channel fusion dataset (`build_fusion_cnn`).

## References
- See `references.md` and the two PDFs in the repo root.
