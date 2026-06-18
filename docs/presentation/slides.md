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

## Results (real KAIST data)
Held-out recordings, Healthy vs Inter-turn. Balanced acc / macro-F1 (test set is imbalanced).

| Channel | Bal-acc | Macro-F1 | Healthy rec. | Inter-turn rec. |
|---|---|---|---|---|
| **Vibration** | **1.00** | **1.00** | 1.00 | 1.00 |
| Fusion (cur.+vib.) | 0.88 | 0.76 | 1.00 | 0.75 |
| Current | 0.69 | 0.49 | 1.00 | 0.37 |

![](../../results/confusion_real_2class.png)

---

## Analysis
- **Vibration ≫ current** for inter-turn detection (richer time-frequency content).
- Class balancing prevents majority collapse (current: 0.80 "accuracy" → honest 0.69 bal-acc).
- Report balanced accuracy / macro-F1, not raw accuracy, under imbalance.
- ⚠️ Only 4 healthy recordings → can't yet exclude a recording-identity shortcut.

---

## Conclusions & Future Work
- Vibration scalograms + CNN detect inter-turn faults; current is weak.
- Next: more independent healthy/fault recordings; current+vibration **fusion**; severity-aware classes.

---

# Thank You
Questions?
