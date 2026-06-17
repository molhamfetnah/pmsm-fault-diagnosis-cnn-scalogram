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
- Accuracy, confusion matrices (`results/confusion_*.png`).
- Per-class precision/recall/F1.
- Effect of dataset size/quality; sim vs real comparison.
- Hyperparameter / overfitting experiments (`docs/experiments.md`).

## 8. Conclusions
- Summary, limitations, future work (paired multi-channel fusion dataset).

## References
- See `references.md` and the two PDFs in the repo root.
