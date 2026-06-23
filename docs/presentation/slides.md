---
marp: true
title: PMSM Fault Diagnosis — Wavelet Scalogram + CNN
paginate: true
---

# PMSM Fault Diagnosis using Wavelet Scalograms + CNN

**«University Name — Faculty»** · **«Department / Program»**

Students: **Mulham Fetna** · **Mohammad Zein Qabbani**

Supervisor: **«Dr. Supervisor Name»**  ·  Academic year: **«2025–2026»**

*Research seminar — CNN classification of PMSM operating states from
time–frequency images*

---

## 1. The problem

- **PMSM** motors run EVs, robots, industrial servos, aircraft actuators.
- The most common stator fault — an **inter-turn short circuit** — can escalate
  to total winding failure within minutes.
- Goal: **detect faults automatically and early** from the motor's own signals.
- Classical methods (e.g. current spectrum analysis) need an expert to know *which*
  frequency to watch and break down when speed/load vary.

---

## 2. Our idea

Turn a **signal** problem into an **image** problem.

```
 signal ─► segment ─► CWT ─► scalogram image ─► CNN ─► fault class
(current/                  (Morlet)  (224×224)         Healthy / InterTurn
 vibration)                                            (+ Demag / Overload)
```

- The CNN **learns** the fault signatures itself — no hand-designed features.
- Two signal channels studied: **stator current** and **vibration**.

---

## 3. System pipeline

- **Sources:** real KAIST dataset (current + vibration), a synthetic generator,
  and MATLAB FOC/Simscape simulation.
- **One config + one manifest** drive everything (signal → scalogram → label →
  split): fully reproducible.
- Implemented in Python (TensorFlow/Keras, PyWavelets), MATLAB-free, **38 unit
  tests + CI**.

---

## 4. PMSM & its faults

- PMSM: permanent-magnet rotor locks onto the stator's rotating field → rotates
  **synchronously**, no slip → high efficiency.
- Driven by **Field-Oriented Control (FOC)** — current sensors are already present.
- Faults studied:
  - **Inter-turn short** (primary, real data)
  - **Demagnetization**, **Overload** (synthetic)

---

## Motor & control system

![w:780](../figures/fig_foc_block.png)

- Stator (3-phase winding) + PM rotor; driven by an **inverter** under **FOC**.
- The current loop **rejects disturbances → it masks the fault in the current**.
- That is *why vibration detects inter-turn faults far better* (our key result).

---

## PMSM vs its relatives

![w:880](../figures/fig_motor_comparison.png)

- vs **PMDC**: same easy torque control, but no brushes → less maintenance.
- vs **BLDC**: sinusoidal back-EMF + FOC → smoother torque, higher efficiency.
- vs **Induction**: no rotor current/slip → higher efficiency & torque density.

---

## Detecting the fault — today vs this work

![w:820](../figures/fig_detection_taxonomy.png)

- Manual/offline (thermal, **Megger**, surge) → need shutdown, periodic.
- Online MCSA/vibration FFT → fixed thresholds, fooled by load/speed.
- **CWT scalogram + CNN** → learns features, robust, detects **earlier**.

---

## 5. Why wavelets, not Fourier

- Fourier is **blind to time** — it can't say *when* a frequency occurs.
- Motor fault signals are **non-stationary** (transient, load-dependent).
- Uncertainty principle: `Δt · Δf ≥ const` — can't have perfect time *and*
  frequency resolution.
- **Wavelets** = short, localised waves → frequency-dependent resolution, ideal
  for transients.

---

## 6. The CWT & scalogram

- Scale the **Morlet** wavelet (frequency knob) and slide it (time knob).
- Each coefficient = **similarity** of the signal to that wave at that time.
- **Scalogram** = colour image of energy vs. time (x) and frequency (y).
- Bright bands & side-bands = fault signatures the CNN can *see*.

---

## 7. Example scalograms (real KAIST data)

![w:1000](../../results/example_scalograms_real.png)

Inter-turn fault visibly **enriches the vibration time–frequency content**.

---

## 8. The dataset

- **KAIST PMSM stator-fault dataset** (Mendeley `rgn5brrgrn`, CC-BY-4.0).
- Current @ **100 kHz**, vibration @ **25.6 kHz** → decimated to 10 kHz.
- **3,150** scalogram segments (0.5 s windows, 50 % overlap).

| Channel | Healthy | InterTurn |
|---|---|---|
| current | 200 (4 rec.) | 1350 (27 rec.) |
| vibration | 200 (4 rec.) | 1400 (28 rec.) |

---

## 9. Leakage-free split

- Split by **recording**, not by segment → no correlated windows across
  train/test (no data leakage).
- Stratified by class, **per channel**, 70 / 15 / 15.
- Special care: only **4 healthy recordings** → force ≥1 into each split so every
  class is evaluable.

---

## 10. Class imbalance — the key issue

- Only **4 healthy** vs ~60 fault recordings.
- Naïve training → model predicts "faulty" for everything:
  **80 % "accuracy" but 0 % healthy detection.**
- Fix: **undersample the majority in train/val**, keep **test natural**.
- Report **balanced accuracy** & **macro-F1**, not raw accuracy.

---

## 11. CNN architecture

```
224×224×3
 → Conv32 → Pool → Conv64 → Pool → Conv128 → Pool
 → Flatten → Dropout(0.5) → Dense128 → Softmax
```

- Adam, sparse cross-entropy, early stopping, data augmentation.
- **Fusion** model: two branches (current + vibration) → global avg pool →
  concatenate → dense.

---

## 12. Results — real data

| Channel | Balanced acc | Macro-F1 | Healthy rec. | InterTurn rec. |
|---|---|---|---|---|
| **Vibration** | **1.00** | **1.00** | 1.00 | 1.00 |
| Fusion (cur.+vib.) | 0.88 | 0.76 | 1.00 | 0.75 |
| Current | 0.69 | 0.49 | 1.00 | 0.37 |

Held-out recordings, Healthy vs Inter-turn.

---

## 13. Confusion matrices

![w:900](../../results/confusion_real_2class.png)

Vibration is perfect; current misses most faults; fusion is in between.

---

## 14. Analysis

- **Vibration ≫ current** for inter-turn detection — physically expected
  (vibration responds directly; FOC suppresses current signatures).
- **Fusion** beats current but not vibration alone on this small data.
- Right **metric matters**: balanced accuracy exposed the majority-class collapse
  that raw accuracy hid.

---

## 15. Improvement experiments

- **Balancing** removes the majority-class collapse (current healthy recall
  0.00 → 1.00).
- **Image size** & **training-set size** studied (see `docs/experiments.md`).
- Overfitting controlled by dropout, global average pooling, augmentation and
  early stopping.

---

## 16. Limitations

- Only **4 distinct healthy recordings** → perfect vibration score can't yet
  exclude a *recording-identity* shortcut.
- Demagnetization / Overload only synthetic.
- Fusion comparison indicative, not strictly controlled.

---

## 17. Conclusions & future work

- A reproducible scalogram + CNN pipeline detects inter-turn PMSM faults; **vibration
  reaches balanced accuracy 1.00** on held-out recordings.
- Future: more independent recordings; paired multi-channel fusion; severity-aware
  targets; explainability (Grad-CAM); on-line deployment in the drive.

---

# Thank you

Questions?

Code, data instructions and full report:
`github.com/molhamfetnah/pmsm-fault-diagnosis-cnn-scalogram`
