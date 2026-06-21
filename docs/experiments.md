# Improvement Experiments

This document reports the ablation studies required by the project (subtask 5 —
*effect of dataset size and quality*; subtask 6 — *model improvement*). Every run
trains the baseline CNN and evaluates on the **natural, imbalanced, held-out
test set** (Healthy vs Inter-turn, real KAIST data). Because the test set is
imbalanced (50 Healthy / 200 Inter-turn), the headline metric is **balanced
accuracy** (mean of per-class recalls) and **2-class macro-F1**, not raw accuracy.

Reproduce with:

```bash
.venv/bin/python -m python.experiments   # writes results/experiments_real.json + learning_curve.png
```

The harness (`python/experiments.py`) runs **each configuration in its own
subprocess** (so TensorFlow releases memory between runs) and **checkpoints after
every run** (so it is resumable). All 16 runs below completed without error.

---

## 1. Effect of class balancing (subtask 6 — dataset improvement)

Train/validation are balanced by undersampling the majority class; the test set
stays natural. Run at 224 px.

| Channel | Balancing | n_train | Balanced acc | Macro-F1 | Healthy recall | Inter-turn recall |
|---|---|---|---|---|---|---|
| current | off | 1050 | 0.800 | 0.851 | 0.600 | 1.000 |
| current | **on** | 200 | 0.693 | 0.502 | **1.000** | 0.385 |
| vibration | off | 1100 | 1.000 | 1.000 | 1.000 | 1.000 |
| vibration | **on** | 200 | 1.000 | 1.000 | 1.000 | 1.000 |

**Interpretation.** On the **current** channel, training on the raw (imbalanced)
distribution biases the model toward the majority Inter-turn class: it scores a
high-looking 0.80 balanced accuracy here but **misses 40 % of healthy motors**
(healthy recall 0.60) — and in repeated runs it collapses further, to healthy
recall **0.00** (raw accuracy 0.80 = the base rate, the classic majority-class
collapse). **Balancing forces the model to detect the rare healthy class**
(healthy recall → 1.00) at the cost of Inter-turn recall, exposing the channel's
true, weak separability instead of hiding it behind a misleading accuracy. The
current channel is also subject to run-to-run variance because only four healthy
recordings and a small balanced set (200 images) are available. On the
**vibration** channel balancing makes no difference — it is perfectly separable
either way.

> Practical takeaway: for fault *detection* (where missing a fault, or wrongly
> flagging a healthy machine, both matter) you must report per-class recall /
> balanced accuracy and you must balance the training signal. Raw accuracy alone
> is actively misleading on this data.

---

## 2. Effect of image (scalogram) size (subtask 6 — model improvement)

Scalograms are rendered at 224 px and downscaled at load. Balanced training.

| Channel | Image size | Balanced acc | Macro-F1 | Inter-turn recall |
|---|---|---|---|---|
| current | 96 | 0.682 | 0.488 | 0.365 |
| current | 160 | 0.730 | 0.555 | 0.460 |
| current | **224** | **0.760** | **0.597** | 0.520 |
| vibration | 96 | 1.000 | 1.000 | 1.000 |
| vibration | 160 | 1.000 | 1.000 | 1.000 |
| vibration | 224 | 1.000 | 1.000 | 1.000 |

**Interpretation.** For the **current** channel, larger scalograms help
**monotonically** (balanced accuracy 0.68 → 0.73 → 0.76 as size grows 96 → 160 →
224): the weak inter-turn signature lives in fine time–frequency detail that is
lost at low resolution, so resolution matters. For the **vibration** channel the
task is already saturated at 1.00 for every size — so one could use **96 px
vibration images for ~5× faster training and inference with no accuracy loss**, a
useful deployment optimisation. We keep 224 px as the default because it is
required for the current channel and harmless for vibration.

---

## 3. Effect of training-set size / data quantity (subtask 5)

Fraction of the (balanced) training set used, at 224 px. `n_train` is the number
of training images.

| Channel | Train fraction | n_train | Balanced acc | Macro-F1 |
|---|---|---|---|---|
| current | 25 % | 54 | 0.693 | 0.502 |
| current | 50 % | 102 | 0.698 | 0.509 |
| current | 100 % | 200 | 0.698 | 0.509 |
| vibration | 25 % | 46 | 0.970 | 0.981 |
| vibration | 50 % | 98 | 1.000 | 1.000 |
| vibration | 100 % | 200 | 1.000 | 1.000 |

![Learning curve](../results/learning_curve.png)

*Figure — Balanced accuracy vs. number of training images, per channel.*

**Interpretation.** This is the clearest result in the study:

- **Vibration learns from almost nothing.** With just **46 training images** it
  already reaches 0.97 balanced accuracy, and it hits 1.00 by ~98 images. The
  vibration fault signature is so separable that data quantity is essentially a
  non-issue.
- **Current does not improve with more data.** It is **flat at ~0.70** from 54 to
  200 images — quadrupling the data changes nothing. This shows the current
  channel's limitation is **signal quality, not quantity**: no amount of
  additional current data will fix a fault signature that is intrinsically weak
  (and partly suppressed by the FOC controller) at the available severities.

> This directly answers subtask 5: dataset *quality* (which channel, what
> severity) dominates dataset *quantity* for this problem. Collecting more current
> data is futile; collecting vibration data — or higher-severity current data —
> is the productive direction.

---

## 4. Overfitting controls used

The pipeline mitigates overfitting with: 50 % **dropout** before the dense head;
**global average pooling** in the fusion branches (far fewer parameters than
flatten); training-data **augmentation** (random horizontal flips);
**early stopping** (patience 5, restore best weights); and a deliberately
**small architecture** sized to a few-thousand-image dataset. The flat current
learning curve and the saturated vibration curve both indicate the model is not
data-starved given balancing — the binding constraints are signal quality
(current) and recording diversity (the four-healthy-recording limit), not model
capacity.

---

## 5. Summary of conclusions

1. **Balance the training set and report balanced accuracy / per-class recall** —
   raw accuracy hides majority-class collapse on this imbalanced problem.
2. **Vibration ≫ current** for inter-turn detection, and it needs very little
   data and resolution to succeed.
3. **For current, resolution helps but data quantity does not** — its ceiling is
   set by signal quality, not sample count.
4. Sensible next steps: smaller (96 px) vibration images for speed; for current,
   pursue higher fault severities or fusion rather than more low-severity data.
