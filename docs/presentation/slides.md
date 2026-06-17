---
marp: true
title: PMSM Fault Diagnosis — Wavelet Scalogram + CNN
---

> Status: OUTLINE — 15–20 slides target. One idea per slide. Use figures from `results/`.

# PMSM Fault Diagnosis
## Wavelet Scalograms + CNN
Mulham Fetna · Mohammad Zein Qabbani

---

## The Problem
Detect & classify PMSM faults from motor signals.

---

## Idea / Pipeline
Signals → CWT → Scalogram images → CNN → fault class.

---

## System Diagram
(architecture figure)

---

## PMSM & Faults
Healthy · Inter-turn · Demagnetization · Overload.

---

## Why Wavelets (not Fourier)
Time–frequency localization; the uncertainty principle.

---

## Scalogram Examples
(figure: `results/sim_vs_real_grid.png`)

---

## Data: Simulation + Real
FOC + Simscape fault model; public datasets; per-channel.

---

## CNN Architecture
Conv → ReLU → Pool → Flatten → Dense → Softmax.

---

## Results
Accuracy + confusion matrix (`results/confusion_current.png`).

---

## Analysis
Per-class F1; sim vs real; overfitting controls.

---

## Conclusions & Future Work
Per-channel models today; paired fusion dataset next.

---

# Thank You
Questions?
