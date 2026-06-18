# Results Summary

Real KAIST PMSM dataset (Mendeley `rgn5brrgrn`, 1.0 + 1.5 kW), Healthy vs
Inter-turn fault. Training/validation are class-balanced by undersampling the
majority class; the **test split is left at its natural distribution** and uses
held-out recordings (leakage-free, grouped by `recording_id`). Because raw
accuracy is misleading on the imbalanced test set, the headline metrics are
**balanced accuracy** and **macro-F1** (2-class).

## Dataset composition (real, post-ingest)

| signal_type | class      | segments | recordings (train/val/test) |
|-------------|------------|----------|------------------------------|
| current     | Healthy    | 200      | 4 (2/1/1)                    |
| current     | InterTurn  | 1350     | 27                           |
| vibration   | Healthy    | 200      | 4 (2/1/1)                    |
| vibration   | InterTurn  | 1400     | 28                           |

Segments are 0.5 s windows (50 % overlap) decimated to 10 kHz, rendered as
224×224 complex-Morlet CWT scalograms. Demagnetization and Overload are **not**
present in the real data (synthetic-only; see README §5).

## Real test-set results (held-out recordings)

| channel   | test accuracy | balanced accuracy | macro-F1 (2-cls) | Healthy recall | InterTurn recall |
|-----------|---------------|-------------------|------------------|----------------|------------------|
| current   | 0.50          | 0.69              | 0.49             | 1.00           | 0.37             |
| vibration | **1.00**      | **1.00**          | **1.00**         | 1.00           | 1.00             |

Confusion matrices: `results/confusion_real_2class.png`. Raw machine-readable
metrics: `results/real_metrics.json`, `results/report_{current,vibration}.json`.

## Reading the result

- **Vibration scalograms cleanly separate healthy from inter-turn faulty motors**
  on held-out recordings; current scalograms do not (the current model detects
  every healthy case but misses ~63 % of faults — inter-turn signatures are weak
  in the stator current at these severities relative to vibration).
- **Caveat — few healthy recordings.** Only 4 distinct healthy recordings exist
  (2 train / 1 val / 1 test). A perfect vibration score therefore cannot fully
  exclude the model keying on recording-specific characteristics rather than
  generalizable fault features. More independent healthy recordings (more motors,
  loads, sessions) are needed to confirm generalization. This is the main
  limitation for the report.
