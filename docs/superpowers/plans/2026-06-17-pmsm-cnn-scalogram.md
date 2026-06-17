# PMSM Fault Diagnosis (Wavelet Scalogram + CNN) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a hybrid MATLAB→Python pipeline that turns PMSM current and vibration signals (simulated + real) into Wavelet Scalogram images and trains a CNN to classify Healthy / InterTurn / Demagnetization / Overload faults.

**Architecture:** MATLAB generates signals (FOC + Simscape fault models) and renders CWT scalogram PNGs; a `manifest.csv` links every signal → image → label/split (group-wise, leakage-free); Python/Keras trains and evaluates per-channel CNNs, then fuses channels. Built MVP-first: a real current dataset proves the whole chain before simulation, vibration, and extra classes are layered on.

**Tech Stack:** MATLAB R2023+ (Wavelet Toolbox, Deep Learning Toolbox, Simscape Electrical), Python 3.10+, TensorFlow/Keras, pandas, scikit-learn, pytest.

## Global Constraints

- Class label set (exact strings, used identically in MATLAB folders, manifest, and Python): `Healthy`, `InterTurn`, `Demagnetization`, `Overload`.
- Manifest is the single source of truth: `data/manifest.csv` with columns `signal_id, source, signal_type, class, severity, fs, dataset_name, recording_id, split, scalogram_path`. `source` ∈ {`sim`,`real`}; `signal_type` ∈ {`current`,`vibration`}; `split` ∈ {`train`,`val`,`test`}.
- Train/val/test split is assigned **grouped by `recording_id`** — segments from one physical recording never appear in two splits.
- Scalogram images: fixed **224×224 RGB PNG**, written to `data/scalograms/<signal_type>/<class>/<signal_id>.png`.
- Signal segmentation: fixed window length and overlap defined once in `config.yaml`; every stage reads that config (no hardcoded duplicates).
- Default wavelet: analytic Morlet (`amor` in MATLAB `cwt`).
- All randomness seeded: `SEED = 42` in Python; document any MATLAB RNG seed.
- MATLAB cannot run in the dev environment — MATLAB tasks deliver `.m` scripts plus a manual verification command the user runs; Python tasks are TDD with pytest.
- `data/raw/` and `data/scalograms/` are gitignored (large); the manifest and code are committed.

---

## Phase 0 — Foundations & Data Audit

### Task 0.1: Repository scaffold + Python environment

**Files:**
- Create: `requirements.txt`, `.gitignore`, `config.yaml`, `python/__init__.py`, `matlab/.gitkeep`, `data/.gitkeep`, `models/.gitkeep`, `results/.gitkeep`
- Create: `pytest.ini`

**Interfaces:**
- Produces: `config.yaml` keys consumed by every later task: `classes`, `signal_types`, `window_seconds`, `overlap`, `target_fs`, `image_size`, `wavelet`, `seed`, `paths.{raw,scalograms,manifest,models,results}`.

- [ ] **Step 1: Create `.gitignore`**

```gitignore
data/raw/
data/scalograms/
models/*.keras
models/*.h5
__pycache__/
.venv/
*.slxc
slprj/
```

- [ ] **Step 2: Create `requirements.txt`**

```
tensorflow==2.16.*
pandas==2.2.*
numpy==1.26.*
scikit-learn==1.4.*
matplotlib==3.8.*
pyyaml==6.*
pillow==10.*
pytest==8.*
```

- [ ] **Step 3: Create `config.yaml`**

```yaml
seed: 42
classes: [Healthy, InterTurn, Demagnetization, Overload]
signal_types: [current, vibration]
window_seconds: 0.5      # length of each signal segment
overlap: 0.5             # fractional overlap between consecutive windows
target_fs: 10000         # Hz; signals resampled to this before CWT
image_size: 224          # square PNG side
wavelet: amor            # MATLAB cwt wavelet name
paths:
  raw: data/raw
  scalograms: data/scalograms
  manifest: data/manifest.csv
  models: models
  results: results
```

- [ ] **Step 4: Create `pytest.ini`**

```ini
[pytest]
testpaths = python/tests
python_files = test_*.py
```

- [ ] **Step 5: Create dirs and install**

Run:
```bash
mkdir -p python/tests data/raw data/scalograms models results matlab/sim matlab/scalogram
touch python/__init__.py python/tests/__init__.py matlab/.gitkeep data/.gitkeep models/.gitkeep results/.gitkeep
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
```
Expected: pip installs without error; `python -c "import tensorflow, pandas, sklearn, yaml, PIL"` prints nothing (success).

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "chore: scaffold pmsm-cnn repo, config, deps"
```

---

### Task 0.2: Config loader (Python)

**Files:**
- Create: `python/config.py`
- Test: `python/tests/test_config.py`

**Interfaces:**
- Produces: `load_config(path="config.yaml") -> dict`; `CLASSES: list[str]` helper read from config; raises `ValueError` if any required key is missing.

- [ ] **Step 1: Write the failing test**

```python
# python/tests/test_config.py
import pytest
from python.config import load_config

def test_load_config_has_required_keys(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "seed: 42\nclasses: [Healthy, InterTurn]\nsignal_types: [current]\n"
        "window_seconds: 0.5\noverlap: 0.5\ntarget_fs: 10000\nimage_size: 224\n"
        "wavelet: amor\npaths: {raw: data/raw, scalograms: s, manifest: m.csv, models: mo, results: r}\n"
    )
    cfg = load_config(str(cfg_file))
    assert cfg["classes"] == ["Healthy", "InterTurn"]
    assert cfg["image_size"] == 224

def test_load_config_missing_key_raises(tmp_path):
    cfg_file = tmp_path / "bad.yaml"
    cfg_file.write_text("seed: 42\n")
    with pytest.raises(ValueError):
        load_config(str(cfg_file))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest python/tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'python.config'`

- [ ] **Step 3: Write minimal implementation**

```python
# python/config.py
import yaml

REQUIRED = ["seed", "classes", "signal_types", "window_seconds", "overlap",
            "target_fs", "image_size", "wavelet", "paths"]

def load_config(path="config.yaml"):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise ValueError(f"config missing keys: {missing}")
    return cfg
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest python/tests/test_config.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add python/config.py python/tests/test_config.py && git commit -m "feat: config loader with required-key validation"
```

---

### Task 0.3: Dataset audit (research deliverable, no code)

**Files:**
- Create: `docs/data-audit.md`

**Interfaces:**
- Produces: a chosen **current** dataset and a chosen **vibration** dataset, each with a documented download URL, license, signal type, sampling rate, and which of the 4 classes it covers.

- [ ] **Step 1: Audit each source in `references.md`**

For each of these, record in `docs/data-audit.md` a row `name | url | signal_type | classes covered | fs | license | notes`:
- Kaggle `ziya07/pmsm-smart-control-dataset`
- Mendeley `rgn5brrgrn/1`
- Zenodo `13974503`
- IEEE-DataPort `three-phase-pmsm-itsc-faults-stator-winding-dataset`
- ResearchGate ResNet PMSM faults paper (for the dataset it references)

- [ ] **Step 2: Select datasets**

Pick the **current** dataset that covers the most of {Healthy, InterTurn, Overload} and one **vibration** dataset (mechanical/demag faults). Record the decision and the class→label mapping (their label names → our `classes` strings) at the top of `docs/data-audit.md`. If Demagnetization is unavailable in real data, note it will come from simulation only.

- [ ] **Step 3: Verify access**

Run (example for a direct-download dataset):
```bash
curl -sI "<chosen-current-dataset-url>" | head -5
```
Expected: HTTP 200/302 (reachable). For Kaggle/login-gated sources, note the manual download step instead.

- [ ] **Step 4: Commit**

```bash
git add docs/data-audit.md && git commit -m "docs: dataset audit and selection (current + vibration)"
```

---

## Phase 1 — MVP (real current dataset → scalograms → baseline CNN)

### Task 1.1: Manifest schema + record API (Python)

**Files:**
- Create: `python/manifest.py`
- Test: `python/tests/test_manifest.py`

**Interfaces:**
- Produces:
  - `MANIFEST_COLUMNS: list[str]`
  - `new_manifest() -> pandas.DataFrame` (empty, typed columns)
  - `add_record(df, *, signal_id, source, signal_type, klass, severity, fs, dataset_name, recording_id) -> DataFrame` (split/scalogram_path left blank, filled later)
  - `save_manifest(df, path)` / `load_manifest(path) -> DataFrame`

- [ ] **Step 1: Write the failing test**

```python
# python/tests/test_manifest.py
from python.manifest import new_manifest, add_record, save_manifest, load_manifest, MANIFEST_COLUMNS

def test_new_manifest_columns():
    df = new_manifest()
    assert list(df.columns) == MANIFEST_COLUMNS

def test_add_record_appends_row():
    df = new_manifest()
    df = add_record(df, signal_id="s1", source="real", signal_type="current",
                    klass="Healthy", severity=0.0, fs=10000,
                    dataset_name="ds", recording_id="r1")
    assert len(df) == 1
    assert df.iloc[0]["class"] == "Healthy"
    assert df.iloc[0]["split"] == ""

def test_save_load_roundtrip(tmp_path):
    df = new_manifest()
    df = add_record(df, signal_id="s1", source="real", signal_type="current",
                    klass="Healthy", severity=0.0, fs=10000,
                    dataset_name="ds", recording_id="r1")
    p = tmp_path / "m.csv"
    save_manifest(df, str(p))
    out = load_manifest(str(p))
    assert out.iloc[0]["signal_id"] == "s1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest python/tests/test_manifest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'python.manifest'`

- [ ] **Step 3: Write minimal implementation**

```python
# python/manifest.py
import pandas as pd

MANIFEST_COLUMNS = ["signal_id", "source", "signal_type", "class", "severity",
                    "fs", "dataset_name", "recording_id", "split", "scalogram_path"]

def new_manifest():
    return pd.DataFrame({c: pd.Series(dtype="object") for c in MANIFEST_COLUMNS})

def add_record(df, *, signal_id, source, signal_type, klass, severity, fs,
               dataset_name, recording_id):
    row = {"signal_id": signal_id, "source": source, "signal_type": signal_type,
           "class": klass, "severity": severity, "fs": fs,
           "dataset_name": dataset_name, "recording_id": recording_id,
           "split": "", "scalogram_path": ""}
    return pd.concat([df, pd.DataFrame([row])], ignore_index=True)

def save_manifest(df, path):
    df.to_csv(path, index=False)

def load_manifest(path):
    return pd.read_csv(path, dtype={"split": "string"}).fillna("")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest python/tests/test_manifest.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add python/manifest.py python/tests/test_manifest.py && git commit -m "feat: manifest schema and record API"
```

---

### Task 1.2: Group-wise train/val/test split (Python)

**Files:**
- Create: `python/split.py`
- Test: `python/tests/test_split.py`

**Interfaces:**
- Consumes: a manifest DataFrame with `recording_id`, `class`.
- Produces: `assign_splits(df, *, seed=42, ratios=(0.7,0.15,0.15)) -> DataFrame` — fills `split` so all rows sharing a `recording_id` get the same split, and class stratification is approximated at the recording level.

- [ ] **Step 1: Write the failing test**

```python
# python/tests/test_split.py
from python.manifest import new_manifest, add_record
from python.split import assign_splits

def _df():
    df = new_manifest()
    # 10 recordings per class, 3 segments each
    for klass in ["Healthy", "InterTurn"]:
        for r in range(10):
            for seg in range(3):
                df = add_record(df, signal_id=f"{klass}-{r}-{seg}", source="real",
                                signal_type="current", klass=klass, severity=0.0,
                                fs=10000, dataset_name="ds", recording_id=f"{klass}-rec{r}")
    return df

def test_no_recording_spans_splits():
    df = assign_splits(_df(), seed=42)
    per_rec_splits = df.groupby("recording_id")["split"].nunique()
    assert (per_rec_splits == 1).all()

def test_all_rows_assigned():
    df = assign_splits(_df(), seed=42)
    assert set(df["split"].unique()) <= {"train", "val", "test"}
    assert (df["split"] != "").all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest python/tests/test_split.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'python.split'`

- [ ] **Step 3: Write minimal implementation**

```python
# python/split.py
import random

def assign_splits(df, *, seed=42, ratios=(0.7, 0.15, 0.15)):
    rng = random.Random(seed)
    df = df.copy()
    rec_to_class = df.groupby("recording_id")["class"].first().to_dict()
    by_class = {}
    for rec, klass in rec_to_class.items():
        by_class.setdefault(klass, []).append(rec)
    rec_split = {}
    for klass, recs in by_class.items():
        recs = sorted(recs)
        rng.shuffle(recs)
        n = len(recs)
        n_tr = int(round(n * ratios[0]))
        n_val = int(round(n * ratios[1]))
        for i, rec in enumerate(recs):
            rec_split[rec] = "train" if i < n_tr else "val" if i < n_tr + n_val else "test"
    df["split"] = df["recording_id"].map(rec_split)
    return df
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest python/tests/test_split.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add python/split.py python/tests/test_split.py && git commit -m "feat: leakage-free group-wise stratified split"
```

---

### Task 1.3: Ingest the real current dataset into the manifest (Python)

**Files:**
- Create: `python/ingest_real.py`
- Test: `python/tests/test_ingest_real.py`

**Interfaces:**
- Consumes: downloaded raw files under `data/raw/<dataset_name>/`, the label mapping from `docs/data-audit.md`.
- Produces: `segment_signal(samples, fs, window_seconds, overlap) -> list[np.ndarray]`; a CLI `python -m python.ingest_real` that reads raw signals, segments them, and writes/extends `data/manifest.csv` (current rows, `source=real`). Exported segment arrays saved as `data/raw/segments/<signal_id>.npy` for the MATLAB scalogram stage to read.

- [ ] **Step 1: Write the failing test (segmentation is the testable core)**

```python
# python/tests/test_ingest_real.py
import numpy as np
from python.ingest_real import segment_signal

def test_segment_count_and_length():
    fs = 1000
    samples = np.arange(2000)            # 2 seconds
    segs = segment_signal(samples, fs=fs, window_seconds=0.5, overlap=0.5)
    # window = 500 samples, step = 250 -> (2000-500)/250 + 1 = 7 segments
    assert len(segs) == 7
    assert all(len(s) == 500 for s in segs)

def test_segment_no_overlap():
    segs = segment_signal(np.arange(1000), fs=1000, window_seconds=0.5, overlap=0.0)
    assert len(segs) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest python/tests/test_ingest_real.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'python.ingest_real'`

- [ ] **Step 3: Write minimal implementation (segmentation + CLI shell)**

```python
# python/ingest_real.py
import os, sys
import numpy as np
from python.config import load_config
from python.manifest import new_manifest, add_record, save_manifest, load_manifest

def segment_signal(samples, *, fs, window_seconds, overlap):
    win = int(round(window_seconds * fs))
    step = max(1, int(round(win * (1 - overlap))))
    segs = []
    i = 0
    while i + win <= len(samples):
        segs.append(np.asarray(samples[i:i + win]))
        i += step
    return segs

def ingest(cfg):
    """Read raw recordings, segment, save .npy, and append manifest rows.
    EDIT load_recordings() to match the chosen dataset's file format/labels."""
    df = load_manifest(cfg["paths"]["manifest"]) if os.path.exists(cfg["paths"]["manifest"]) else new_manifest()
    seg_dir = os.path.join(cfg["paths"]["raw"], "segments")
    os.makedirs(seg_dir, exist_ok=True)
    for rec in load_recordings(cfg):          # rec: dict(samples, fs, class, dataset, recording_id, severity)
        segs = segment_signal(rec["samples"], fs=rec["fs"],
                              window_seconds=cfg["window_seconds"], overlap=cfg["overlap"])
        for k, seg in enumerate(segs):
            sid = f"{rec['recording_id']}-seg{k}"
            np.save(os.path.join(seg_dir, sid + ".npy"), seg)
            df = add_record(df, signal_id=sid, source="real", signal_type="current",
                            klass=rec["class"], severity=rec["severity"], fs=rec["fs"],
                            dataset_name=rec["dataset"], recording_id=rec["recording_id"])
    save_manifest(df, cfg["paths"]["manifest"])
    return df

def load_recordings(cfg):
    # TODO-FOR-USER: implement parsing for the dataset chosen in docs/data-audit.md.
    raise NotImplementedError("Implement load_recordings() for your dataset format")

if __name__ == "__main__":
    ingest(load_config())
```

Note: `load_recordings` is the one dataset-specific seam — its body is implemented when the actual dataset format is known (Task 0.3 output). The tested logic (`segment_signal`) is complete.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest python/tests/test_ingest_real.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Implement `load_recordings` for the chosen dataset, then run ingest**

Run:
```bash
python -m python.ingest_real
python -c "import pandas as pd; d=pd.read_csv('data/manifest.csv'); print(d['class'].value_counts())"
```
Expected: per-class segment counts printed; `data/raw/segments/*.npy` created.

- [ ] **Step 6: Commit**

```bash
git add python/ingest_real.py python/tests/test_ingest_real.py && git commit -m "feat: real current dataset ingestion + segmentation"
```

---

### Task 1.4: Assign splits and freeze them into the manifest (Python)

**Files:**
- Create: `python/run_split.py`
- Test: covered by Task 1.2 (`assign_splits`)

**Interfaces:**
- Consumes: `assign_splits` (1.2), manifest on disk.
- Produces: manifest with `split` filled; printed split distribution.

- [ ] **Step 1: Write the script**

```python
# python/run_split.py
from python.config import load_config
from python.manifest import load_manifest, save_manifest
from python.split import assign_splits

if __name__ == "__main__":
    cfg = load_config()
    df = load_manifest(cfg["paths"]["manifest"])
    df = assign_splits(df, seed=cfg["seed"])
    save_manifest(df, cfg["paths"]["manifest"])
    print(df.groupby(["split", "class"]).size())
```

- [ ] **Step 2: Run it**

Run: `python -m python.run_split`
Expected: a table of split×class counts; every recording confined to one split.

- [ ] **Step 3: Commit**

```bash
git add python/run_split.py && git commit -m "feat: freeze train/val/test splits into manifest"
```

---

### Task 1.5: CWT scalogram generator (MATLAB)

**Files:**
- Create: `matlab/scalogram/generate_scalograms.m`
- Create: `matlab/scalogram/scalogram_from_signal.m`

**Interfaces:**
- Consumes: `data/raw/segments/*.npy` (or a MATLAB-readable export), the manifest, `config.yaml` values.
- Produces: 224×224 RGB PNGs at `data/scalograms/<signal_type>/<class>/<signal_id>.png`; updates `scalogram_path` column in the manifest.

- [ ] **Step 1: Write the per-signal function**

```matlab
% matlab/scalogram/scalogram_from_signal.m
function scalogram_from_signal(sig, fs, wname, imgSize, outPath)
% Render the magnitude CWT scalogram of SIG to an imgSize x imgSize RGB PNG.
    [cfs, ~] = cwt(double(sig(:)), wname, fs);
    A = abs(cfs);
    A = A / max(A(:) + eps);                 % normalize 0..1
    rgb = ind2rgb(im2uint8(A), jet(256));    % colour map -> RGB
    rgb = imresize(rgb, [imgSize imgSize]);
    folder = fileparts(outPath);
    if ~exist(folder, 'dir'); mkdir(folder); end
    imwrite(rgb, outPath);
end
```

- [ ] **Step 2: Write the batch driver**

```matlab
% matlab/scalogram/generate_scalograms.m
% Reads config.yaml + data/manifest.csv, renders a PNG per signal, writes scalogram_path back.
cfg = yaml_read('config.yaml');           % use a YAML reader or hardcode the few values
wname   = cfg.wavelet;                     % 'amor'
imgSize = cfg.image_size;                  % 224
M = readtable('data/manifest.csv', 'TextType','string');
segDir = fullfile(cfg.paths.raw, 'segments');
for i = 1:height(M)
    sid = M.signal_id(i);
    sig = readNPY(fullfile(segDir, sid + ".npy"));   % npy-matlab on path
    outPath = fullfile(cfg.paths.scalograms, M.signal_type(i), M.class(i), sid + ".png");
    scalogram_from_signal(sig, M.fs(i), wname, imgSize, char(outPath));
    M.scalogram_path(i) = outPath;
end
writetable(M, 'data/manifest.csv');
disp('Scalograms generated.');
```

Dependencies the user adds to the MATLAB path: `npy-matlab` (read `.npy`) and a small YAML reader, or replace `yaml_read`/`readNPY` with hardcoded values and `.mat` exports.

- [ ] **Step 3: Manual verification (run in MATLAB)**

Run in MATLAB:
```matlab
scalogram_from_signal(sin(2*pi*50*(0:1/10000:0.5)), 10000, 'amor', 224, 'results/_smoke.png');
imfinfo('results/_smoke.png')   % expect Width=224 Height=224, ColorType truecolor
```
Expected: a 224×224 RGB PNG with a clear band around 50 Hz.

- [ ] **Step 4: Run the batch (in MATLAB) and spot-check**

Run: `generate_scalograms` in MATLAB, then in Python:
```bash
python -c "import pandas as pd,os; d=pd.read_csv('data/manifest.csv'); print((d['scalogram_path']!='').mean(), 'have images'); print(all(os.path.exists(p) for p in d['scalogram_path'][:20]))"
```
Expected: `1.0 have images` and `True`.

- [ ] **Step 5: Commit**

```bash
git add matlab/scalogram/*.m && git commit -m "feat: MATLAB CWT scalogram generation"
```

---

### Task 1.6: Keras data pipeline from manifest (Python)

**Files:**
- Create: `python/data_loader.py`
- Test: `python/tests/test_data_loader.py`

**Interfaces:**
- Consumes: manifest with `scalogram_path` + `split`, `config.yaml`.
- Produces: `make_dataset(df, split, signal_type, classes, image_size, batch_size, seed, augment=False) -> (tf.data.Dataset, class_names)`; `class_to_index(classes) -> dict`.

- [ ] **Step 1: Write the failing test (label mapping is the pure-logic core)**

```python
# python/tests/test_data_loader.py
from python.data_loader import class_to_index

def test_class_to_index_stable_order():
    classes = ["Healthy", "InterTurn", "Demagnetization", "Overload"]
    idx = class_to_index(classes)
    assert idx["Healthy"] == 0 and idx["Overload"] == 3
    assert len(idx) == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest python/tests/test_data_loader.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'python.data_loader'`

- [ ] **Step 3: Write minimal implementation**

```python
# python/data_loader.py
import tensorflow as tf

def class_to_index(classes):
    return {c: i for i, c in enumerate(classes)}

def make_dataset(df, split, signal_type, classes, image_size, batch_size, seed, augment=False):
    idx = class_to_index(classes)
    sub = df[(df["split"] == split) & (df["signal_type"] == signal_type)]
    paths = sub["scalogram_path"].tolist()
    labels = [idx[c] for c in sub["class"].tolist()]

    def _load(path, label):
        img = tf.io.decode_png(tf.io.read_file(path), channels=3)
        img = tf.image.resize(img, [image_size, image_size]) / 255.0
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if split == "train":
        ds = ds.shuffle(len(paths), seed=seed)
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(lambda x, y: (tf.image.random_flip_left_right(x), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE), list(classes)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest python/tests/test_data_loader.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add python/data_loader.py python/tests/test_data_loader.py && git commit -m "feat: manifest-driven keras data pipeline"
```

---

### Task 1.7: CNN model definition (Python)

**Files:**
- Create: `python/model.py`
- Test: `python/tests/test_model.py`

**Interfaces:**
- Produces: `build_cnn(input_shape=(224,224,3), num_classes=4) -> keras.Model` (compiled-ready, softmax output).

- [ ] **Step 1: Write the failing test**

```python
# python/tests/test_model.py
from python.model import build_cnn

def test_output_shape():
    m = build_cnn(input_shape=(224, 224, 3), num_classes=4)
    assert m.output_shape == (None, 4)

def test_accepts_custom_classes():
    m = build_cnn(input_shape=(64, 64, 3), num_classes=2)
    assert m.output_shape == (None, 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest python/tests/test_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'python.model'`

- [ ] **Step 3: Write minimal implementation**

```python
# python/model.py
from tensorflow import keras
from tensorflow.keras import layers

def build_cnn(input_shape=(224, 224, 3), num_classes=4):
    return keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest python/tests/test_model.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add python/model.py python/tests/test_model.py && git commit -m "feat: baseline CNN architecture"
```

---

### Task 1.8: Training entrypoint (Python)

**Files:**
- Create: `python/train.py`
- Test: `python/tests/test_train_smoke.py`

**Interfaces:**
- Consumes: `make_dataset`, `build_cnn`, config, manifest.
- Produces: `train(cfg, signal_type="current", epochs=...) -> (model, history)`; saves `models/cnn_<signal_type>.keras` and `results/history_<signal_type>.json`.

- [ ] **Step 1: Write the failing smoke test (tiny synthetic data, 1 epoch)**

```python
# python/tests/test_train_smoke.py
import numpy as np, os
from PIL import Image
from python.manifest import new_manifest, add_record
from python.train import train_from_df

def _tiny_dataset(tmp_path):
    df = new_manifest()
    for klass in ["Healthy", "InterTurn"]:
        for r in range(4):
            p = tmp_path / f"{klass}-{r}.png"
            Image.fromarray((np.random.rand(32, 32, 3) * 255).astype("uint8")).save(p)
            df = add_record(df, signal_id=f"{klass}-{r}", source="real",
                            signal_type="current", klass=klass, severity=0.0, fs=10000,
                            dataset_name="ds", recording_id=f"{klass}-rec{r}")
            df.loc[df.index[-1], "scalogram_path"] = str(p)
            df.loc[df.index[-1], "split"] = "train" if r < 3 else "val"
    return df

def test_train_runs_one_epoch(tmp_path):
    df = _tiny_dataset(tmp_path)
    model, hist = train_from_df(df, classes=["Healthy", "InterTurn"], signal_type="current",
                                image_size=32, batch_size=2, epochs=1, seed=42)
    assert "loss" in hist.history
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest python/tests/test_train_smoke.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'python.train'`

- [ ] **Step 3: Write minimal implementation**

```python
# python/train.py
import json, os
from tensorflow import keras
from python.config import load_config
from python.manifest import load_manifest
from python.data_loader import make_dataset
from python.model import build_cnn

def train_from_df(df, *, classes, signal_type, image_size, batch_size, epochs, seed):
    train_ds, _ = make_dataset(df, "train", signal_type, classes, image_size, batch_size, seed, augment=True)
    val_ds, _ = make_dataset(df, "val", signal_type, classes, image_size, batch_size, seed)
    model = build_cnn(input_shape=(image_size, image_size, 3), num_classes=len(classes))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                     callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
    return model, hist

def train(cfg, signal_type="current", epochs=30):
    df = load_manifest(cfg["paths"]["manifest"])
    model, hist = train_from_df(df, classes=cfg["classes"], signal_type=signal_type,
                                image_size=cfg["image_size"], batch_size=32,
                                epochs=epochs, seed=cfg["seed"])
    os.makedirs(cfg["paths"]["models"], exist_ok=True)
    os.makedirs(cfg["paths"]["results"], exist_ok=True)
    model.save(os.path.join(cfg["paths"]["models"], f"cnn_{signal_type}.keras"))
    with open(os.path.join(cfg["paths"]["results"], f"history_{signal_type}.json"), "w") as f:
        json.dump(hist.history, f)
    return model, hist

if __name__ == "__main__":
    train(load_config())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest python/tests/test_train_smoke.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Train the MVP for real**

Run: `python -m python.train`
Expected: model trains; `models/cnn_current.keras` and `results/history_current.json` written.

- [ ] **Step 6: Commit**

```bash
git add python/train.py python/tests/test_train_smoke.py && git commit -m "feat: CNN training entrypoint + smoke test"
```

---

### Task 1.9: Evaluation (confusion matrix + report) (Python)

**Files:**
- Create: `python/evaluate.py`
- Test: `python/tests/test_evaluate.py`

**Interfaces:**
- Consumes: a saved model, test split, config.
- Produces: `confusion_and_report(model, ds, class_names) -> (np.ndarray, dict)`; CLI saves `results/confusion_<signal_type>.png` and `results/report_<signal_type>.json`.

- [ ] **Step 1: Write the failing test**

```python
# python/tests/test_evaluate.py
import numpy as np
from python.evaluate import metrics_from_predictions

def test_metrics_from_predictions():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    cm, report = metrics_from_predictions(y_true, y_pred, ["A", "B"])
    assert cm.shape == (2, 2)
    assert "accuracy" in report
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest python/tests/test_evaluate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'python.evaluate'`

- [ ] **Step 3: Write minimal implementation**

```python
# python/evaluate.py
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from python.config import load_config
from python.manifest import load_manifest
from python.data_loader import make_dataset
from tensorflow import keras

def metrics_from_predictions(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    report = classification_report(y_true, y_pred, target_names=class_names,
                                   output_dict=True, zero_division=0)
    return cm, report

def confusion_and_report(model, ds, class_names):
    y_true, y_pred = [], []
    for x, y in ds:
        p = model.predict(x, verbose=0).argmax(axis=1)
        y_true.extend(y.numpy().tolist()); y_pred.extend(p.tolist())
    return metrics_from_predictions(np.array(y_true), np.array(y_pred), class_names)

def main(cfg, signal_type="current"):
    df = load_manifest(cfg["paths"]["manifest"])
    ds, names = make_dataset(df, "test", signal_type, cfg["classes"],
                             cfg["image_size"], 32, cfg["seed"])
    model = keras.models.load_model(os.path.join(cfg["paths"]["models"], f"cnn_{signal_type}.keras"))
    cm, report = confusion_and_report(model, ds, names)
    fig, ax = plt.subplots()
    im = ax.imshow(cm); ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right"); ax.set_yticklabels(names)
    for (i, j), v in np.ndenumerate(cm): ax.text(j, i, str(v), ha="center", va="center")
    fig.colorbar(im); fig.tight_layout()
    fig.savefig(os.path.join(cfg["paths"]["results"], f"confusion_{signal_type}.png"))
    with open(os.path.join(cfg["paths"]["results"], f"report_{signal_type}.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"accuracy={report['accuracy']:.3f}")

if __name__ == "__main__":
    main(load_config())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest python/tests/test_evaluate.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Evaluate the MVP**

Run: `python -m python.evaluate`
Expected: prints `accuracy=...`; writes `results/confusion_current.png` and `results/report_current.json`.

- [ ] **Step 6: Commit**

```bash
git add python/evaluate.py python/tests/test_evaluate.py && git commit -m "feat: evaluation, confusion matrix, classification report"
```

**MVP COMPLETE** — full chain (real current → scalograms → CNN → metrics) is proven end-to-end. Review checkpoint before Phase 2.

---

## Phase 2 — Simulation data

### Task 2.1: Export current signals from the FOC model (MATLAB)

**Files:**
- Create: `matlab/sim/export_foc_current.m`

**Interfaces:**
- Consumes: `FOC_PMSM-main/Motor_script.m` + `FOCsimulation.slx`.
- Produces: `data/raw/sim/foc/<recording_id>.npy` (or `.mat`) per load case, with metadata, for `Healthy` and `Overload` (via load/torque setpoints).

- [ ] **Step 1: Write the export script**

```matlab
% matlab/sim/export_foc_current.m
% Run Motor_script first, then this. Sweeps load cases, logs phase current, saves arrays.
run(fullfile('FOC_PMSM-main','Motor_script.m'));
cases = struct('name', {'Healthy_load1','Overload_load1'}, 'load', {0.5, 1.5});  % PU torque
outDir = fullfile('data','raw','sim','foc'); if ~exist(outDir,'dir'); mkdir(outDir); end
for c = 1:numel(cases)
    % set load via model workspace / input case here (model-specific), then:
    out = sim('FOC_PMSM-main/FOCsimulation.slx');
    ia = out.simout.signals.values(:,1);     % phase-A current (adjust index)
    writeNPY(ia, fullfile(outDir, [cases(c).name '.npy']));  % npy-matlab
end
disp('FOC current exported.');
```

- [ ] **Step 2: Manual verification (MATLAB)**

Run the script; then check a saved file loads and is non-trivial:
```matlab
x = readNPY('data/raw/sim/foc/Healthy_load1.npy'); fprintf('len=%d std=%.4f\n', numel(x), std(x));
```
Expected: non-zero length, non-zero std (a real waveform).

- [ ] **Step 3: Commit**

```bash
git add matlab/sim/export_foc_current.m && git commit -m "feat: export FOC simulated current (healthy/overload)"
```

---

### Task 2.2: Build & export inter-turn fault current from Simscape (MATLAB)

**Files:**
- Create: `matlab/sim/export_faulty_current.m`

**Interfaces:**
- Consumes: `simscape-pmsm/FaultyPMSM.ssc` (`sigma` = shorted-turn ratio), a simple drive harness.
- Produces: `data/raw/sim/fault/InterTurn_sigmaXX_*.npy` across severities, plus a `Healthy` baseline (`sigma=0`).

- [ ] **Step 1: Build the Simscape component**

Run in MATLAB (from a `+pmsm` package folder containing the `.ssc`):
```matlab
ssc_build pmsm
```
Expected: generates the `FaultyPMSM` block library without errors.

- [ ] **Step 2: Write the sweep/export script**

```matlab
% matlab/sim/export_faulty_current.m
% Drives the FaultyPMSM block open-loop across inter-turn severities, logs stator current.
sigmas = [0.0 0.05 0.10 0.20];   % 0 = healthy baseline
outDir = fullfile('data','raw','sim','fault'); if ~exist(outDir,'dir'); mkdir(outDir); end
for s = sigmas
    % set block param 'sigma' = s in the harness model, run sim, capture current:
    out = sim('matlab/sim/fault_harness.slx');     % create a minimal drive harness model
    ia = out.simout.signals.values(:,1);
    klass = "InterTurn"; if s == 0; klass = "Healthy"; end
    name = sprintf('%s_sigma%02d', klass, round(s*100));
    writeNPY(ia, fullfile(outDir, [char(name) '.npy']));
end
disp('Faulty current exported.');
```

Note: honor the model limits from `simscape-pmsm/README.md` — use a **1 pole-pair** config and a simple open-loop drive (not full sensorless FOC), per the spec's risk mitigation.

- [ ] **Step 3: Manual verification (MATLAB)**

Run; confirm severity changes the signal:
```matlab
h = readNPY('data/raw/sim/fault/Healthy_sigma00.npy');
f = readNPY('data/raw/sim/fault/InterTurn_sigma20.npy');
fprintf('rms healthy=%.4f faulty=%.4f\n', rms(h), rms(f));
```
Expected: the two RMS values differ (fault is visible in the current).

- [ ] **Step 4: Commit**

```bash
git add matlab/sim/export_faulty_current.m && git commit -m "feat: export Simscape inter-turn fault current sweep"
```

---

### Task 2.3: Ingest simulated signals into the manifest (Python)

**Files:**
- Modify: `python/ingest_sim.py` (new)
- Test: `python/tests/test_ingest_sim.py`

**Interfaces:**
- Consumes: `data/raw/sim/**/*.npy`, the filename→class/severity convention from 2.1/2.2, `segment_signal` (reuse from 1.3).
- Produces: appends `source=sim, signal_type=current` rows; reuses `segment_signal`. CLI `python -m python.ingest_sim`.

- [ ] **Step 1: Write the failing test (filename parsing is the testable core)**

```python
# python/tests/test_ingest_sim.py
from python.ingest_sim import parse_sim_filename

def test_parse_fault_filename():
    klass, severity = parse_sim_filename("InterTurn_sigma20.npy")
    assert klass == "InterTurn" and severity == 0.20

def test_parse_healthy_filename():
    klass, severity = parse_sim_filename("Healthy_load1.npy")
    assert klass == "Healthy" and severity == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest python/tests/test_ingest_sim.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'python.ingest_sim'`

- [ ] **Step 3: Write minimal implementation**

```python
# python/ingest_sim.py
import os, glob, re
import numpy as np
from python.config import load_config
from python.manifest import load_manifest, add_record, save_manifest
from python.ingest_real import segment_signal

def parse_sim_filename(fname):
    stem = os.path.splitext(os.path.basename(fname))[0]
    klass = stem.split("_")[0]
    m = re.search(r"sigma(\d+)", stem)
    severity = int(m.group(1)) / 100.0 if m else 0.0
    return klass, severity

def ingest(cfg):
    df = load_manifest(cfg["paths"]["manifest"])
    seg_dir = os.path.join(cfg["paths"]["raw"], "segments")
    os.makedirs(seg_dir, exist_ok=True)
    for path in glob.glob(os.path.join(cfg["paths"]["raw"], "sim", "**", "*.npy"), recursive=True):
        klass, severity = parse_sim_filename(path)
        sig = np.load(path)
        rec_id = "sim-" + os.path.splitext(os.path.basename(path))[0]
        for k, seg in enumerate(segment_signal(sig, fs=cfg["target_fs"],
                                window_seconds=cfg["window_seconds"], overlap=cfg["overlap"])):
            sid = f"{rec_id}-seg{k}"
            np.save(os.path.join(seg_dir, sid + ".npy"), seg)
            df = add_record(df, signal_id=sid, source="sim", signal_type="current",
                            klass=klass, severity=severity, fs=cfg["target_fs"],
                            dataset_name="simulation", recording_id=rec_id)
    save_manifest(df, cfg["paths"]["manifest"])

if __name__ == "__main__":
    ingest(load_config())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest python/tests/test_ingest_sim.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Ingest, re-split, regenerate scalograms, retrain**

Run:
```bash
python -m python.ingest_sim
python -m python.run_split          # re-freeze splits over the enlarged manifest
# (MATLAB) generate_scalograms       # render the new sim segments
python -m python.train
python -m python.evaluate
```
Expected: manifest now has `sim` + `real` current rows; accuracy reported on the combined set.

- [ ] **Step 6: Commit**

```bash
git add python/ingest_sim.py python/tests/test_ingest_sim.py && git commit -m "feat: simulated current ingestion + sim/real merge"
```

---

### Task 2.4: Sim-vs-real scalogram comparison (report figure)

**Files:**
- Create: `python/compare_sim_real.py`

**Interfaces:**
- Produces: `results/sim_vs_real_grid.png` — a grid of example scalograms (sim vs real, per class) for the report.

- [ ] **Step 1: Write the script**

```python
# python/compare_sim_real.py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from python.config import load_config
from python.manifest import load_manifest

def main(cfg):
    df = load_manifest(cfg["paths"]["manifest"])
    df = df[df["signal_type"] == "current"]
    sources = ["sim", "real"]; classes = cfg["classes"]
    fig, axes = plt.subplots(len(sources), len(classes), figsize=(3*len(classes), 6))
    for r, src in enumerate(sources):
        for c, klass in enumerate(classes):
            sub = df[(df["source"] == src) & (df["class"] == klass) & (df["scalogram_path"] != "")]
            ax = axes[r][c]; ax.axis("off"); ax.set_title(f"{src}/{klass}", fontsize=8)
            if len(sub): ax.imshow(Image.open(sub.iloc[0]["scalogram_path"]))
    fig.tight_layout(); fig.savefig(f"{cfg['paths']['results']}/sim_vs_real_grid.png", dpi=150)

if __name__ == "__main__":
    main(load_config())
```

- [ ] **Step 2: Run it**

Run: `python -m python.compare_sim_real`
Expected: `results/sim_vs_real_grid.png` shows scalograms per source×class.

- [ ] **Step 3: Commit**

```bash
git add python/compare_sim_real.py && git commit -m "feat: sim-vs-real scalogram comparison figure"
```

---

## Phase 3 — Expand classes + vibration channel

### Task 3.1: Ingest the vibration dataset (Python)

**Files:**
- Create: `python/ingest_vibration.py`
- Test: reuse `segment_signal`; add `python/tests/test_ingest_vibration.py` for its label-mapping seam.

**Interfaces:**
- Consumes: the vibration dataset chosen in Task 0.3.
- Produces: manifest rows with `signal_type=vibration`; `.npy` segments; mirrors `ingest_real` structure (separate file because the dataset format and label map differ).

- [ ] **Step 1: Write the failing test for the label map**

```python
# python/tests/test_ingest_vibration.py
from python.ingest_vibration import map_label

def test_map_label_known():
    assert map_label("demag") == "Demagnetization"
    assert map_label("normal") == "Healthy"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest python/tests/test_ingest_vibration.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# python/ingest_vibration.py
import os
import numpy as np
from python.config import load_config
from python.manifest import load_manifest, add_record, save_manifest
from python.ingest_real import segment_signal

# EDIT to match the chosen vibration dataset's raw label strings:
LABEL_MAP = {"normal": "Healthy", "demag": "Demagnetization",
             "interturn": "InterTurn", "overload": "Overload"}

def map_label(raw):
    return LABEL_MAP[raw.lower()]

def load_recordings(cfg):
    # TODO-FOR-USER: parse the vibration dataset; yield dict(samples, fs, raw_label, recording_id, severity)
    raise NotImplementedError

def ingest(cfg):
    df = load_manifest(cfg["paths"]["manifest"])
    seg_dir = os.path.join(cfg["paths"]["raw"], "segments"); os.makedirs(seg_dir, exist_ok=True)
    for rec in load_recordings(cfg):
        for k, seg in enumerate(segment_signal(rec["samples"], fs=rec["fs"],
                                window_seconds=cfg["window_seconds"], overlap=cfg["overlap"])):
            sid = f"vib-{rec['recording_id']}-seg{k}"
            np.save(os.path.join(seg_dir, sid + ".npy"), seg)
            df = add_record(df, signal_id=sid, source="real", signal_type="vibration",
                            klass=map_label(rec["raw_label"]), severity=rec["severity"],
                            fs=rec["fs"], dataset_name="vibration_ds", recording_id=rec["recording_id"])
    save_manifest(df, cfg["paths"]["manifest"])

if __name__ == "__main__":
    ingest(load_config())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest python/tests/test_ingest_vibration.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Implement `load_recordings`, ingest, generate scalograms, train vibration model**

Run:
```bash
python -m python.ingest_vibration
python -m python.run_split
# (MATLAB) generate_scalograms     # vibration PNGs land under data/scalograms/vibration/
python -m python.train            # then edit __main__ or call train(cfg, signal_type="vibration")
```
Expected: a `models/cnn_vibration.keras` plus `results/report_vibration.json`.

- [ ] **Step 6: Commit**

```bash
git add python/ingest_vibration.py python/tests/test_ingest_vibration.py && git commit -m "feat: vibration dataset ingestion + per-channel model"
```

---

### Task 3.2: Per-signal-type training CLI flag (Python)

**Files:**
- Modify: `python/train.py` (add argparse), `python/evaluate.py` (add argparse)

**Interfaces:**
- Produces: `python -m python.train --signal-type vibration` and `--signal-type current`; same for evaluate.

- [ ] **Step 1: Add argparse to `train.py`**

Replace the `__main__` block:
```python
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--signal-type", default="current")
    ap.add_argument("--epochs", type=int, default=30)
    a = ap.parse_args()
    train(load_config(), signal_type=a.signal_type, epochs=a.epochs)
```

- [ ] **Step 2: Add the same to `evaluate.py`**

```python
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--signal-type", default="current")
    a = ap.parse_args()
    main(load_config(), signal_type=a.signal_type)
```

- [ ] **Step 3: Verify both channels run**

Run:
```bash
python -m python.train --signal-type current --epochs 1
python -m python.train --signal-type vibration --epochs 1
```
Expected: both produce a saved model without error.

- [ ] **Step 4: Commit**

```bash
git add python/train.py python/evaluate.py && git commit -m "feat: per-channel train/evaluate CLI flags"
```

---

## Phase 4 — Fusion, tuning, evaluation

### Task 4.1: Dual-branch fusion model (Python)

**Files:**
- Modify: `python/model.py` (add `build_fusion_cnn`)
- Test: `python/tests/test_fusion_model.py`

**Interfaces:**
- Produces: `build_fusion_cnn(input_shape, num_classes) -> keras.Model` with two image inputs (current, vibration) and one softmax output.

- [ ] **Step 1: Write the failing test**

```python
# python/tests/test_fusion_model.py
from python.model import build_fusion_cnn

def test_fusion_two_inputs_one_output():
    m = build_fusion_cnn(input_shape=(224, 224, 3), num_classes=4)
    assert len(m.inputs) == 2
    assert m.output_shape == (None, 4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest python/tests/test_fusion_model.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_fusion_cnn'`

- [ ] **Step 3: Implement**

```python
# add to python/model.py
from tensorflow.keras import Model, Input

def _branch(x):
    for f in (32, 64, 128):
        x = layers.Conv2D(f, 3, activation="relu")(x)
        x = layers.MaxPooling2D()(x)
    return layers.Flatten()(x)

def build_fusion_cnn(input_shape=(224, 224, 3), num_classes=4):
    ic, iv = Input(shape=input_shape, name="current"), Input(shape=input_shape, name="vibration")
    merged = layers.concatenate([_branch(ic), _branch(iv)])
    z = layers.Dropout(0.5)(merged)
    z = layers.Dense(128, activation="relu")(z)
    out = layers.Dense(num_classes, activation="softmax")(z)
    return Model(inputs=[ic, iv], outputs=out)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest python/tests/test_fusion_model.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add python/model.py python/tests/test_fusion_model.py && git commit -m "feat: dual-branch current+vibration fusion CNN"
```

Note: fusion requires paired current+vibration samples. If the per-channel datasets are not paired (the accepted fallback), document fusion as "future work / requires a paired dataset" in the report and rely on the two single-channel models — do not fabricate pairings.

---

### Task 4.2: Hyperparameter + overfitting experiments (report data)

**Files:**
- Create: `python/experiments.py`
- Create: `docs/experiments.md`

**Interfaces:**
- Produces: a small grid (learning rate, dropout, with/without augmentation, image size) run per channel; results table in `docs/experiments.md` + curves in `results/`.

- [ ] **Step 1: Write the experiment runner**

```python
# python/experiments.py
import json, os
from python.config import load_config
from python.manifest import load_manifest
from python.train import train_from_df

GRID = [{"lr": 1e-3, "dropout": 0.5, "augment": True},
        {"lr": 1e-4, "dropout": 0.3, "augment": True},
        {"lr": 1e-3, "dropout": 0.5, "augment": False}]

def main(cfg, signal_type="current"):
    df = load_manifest(cfg["paths"]["manifest"])
    rows = []
    for g in GRID:
        _, hist = train_from_df(df, classes=cfg["classes"], signal_type=signal_type,
                                image_size=cfg["image_size"], batch_size=32, epochs=20, seed=cfg["seed"])
        rows.append({**g, "val_acc": max(hist.history["val_accuracy"])})
    os.makedirs(cfg["paths"]["results"], exist_ok=True)
    with open(f"{cfg['paths']['results']}/experiments_{signal_type}.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(rows)

if __name__ == "__main__":
    main(load_config())
```

(For a true LR/dropout sweep, extend `train_from_df` to accept `lr`/`dropout`; the grid above is the harness — wire those params through `build_cnn`/`compile` as the experiment requires.)

- [ ] **Step 2: Run and record**

Run: `python -m python.experiments`
Expected: prints val-acc per config; write the best config + the overfitting analysis (train vs val gap) into `docs/experiments.md`.

- [ ] **Step 3: Commit**

```bash
git add python/experiments.py docs/experiments.md && git commit -m "feat: hyperparameter/overfitting experiments + writeup"
```

---

### Task 4.3: Final evaluation pass + results bundle

**Files:**
- Create: `python/report_metrics.py`
- Create: `results/summary.md`

**Interfaces:**
- Produces: a consolidated `results/summary.md` aggregating accuracy, per-class F1, and confusion matrices for current and vibration models, plus dataset composition (sim/real counts, class balance).

- [ ] **Step 1: Write the summary generator**

```python
# python/report_metrics.py
import json, os
from python.config import load_config
from python.manifest import load_manifest

def main(cfg):
    df = load_manifest(cfg["paths"]["manifest"])
    lines = ["# Results Summary\n", "## Dataset composition\n"]
    lines.append(df.groupby(["signal_type", "source", "class"]).size().to_string())
    for st in cfg["signal_types"]:
        p = f"{cfg['paths']['results']}/report_{st}.json"
        if os.path.exists(p):
            r = json.load(open(p))
            lines.append(f"\n## {st} model\n- accuracy: {r.get('accuracy')}\n")
            for k in cfg["classes"]:
                if k in r: lines.append(f"- {k} F1: {r[k]['f1-score']:.3f}\n")
    open(f"{cfg['paths']['results']}/summary.md", "w").write("\n".join(map(str, lines)))
    print("wrote results/summary.md")

if __name__ == "__main__":
    main(load_config())
```

- [ ] **Step 2: Run it**

Run: `python -m python.report_metrics`
Expected: `results/summary.md` with dataset composition + per-model metrics.

- [ ] **Step 3: Run the full Python test suite**

Run: `pytest -v`
Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add python/report_metrics.py results/summary.md && git commit -m "feat: consolidated results summary"
```

---

## Phase 5 — Deliverables

### Task 5.1: Final report (25–35 pages)

**Files:**
- Create: `docs/report/report.md` (or LaTeX)

**Interfaces:**
- Produces: the written report. Pull figures from `results/` and content from `notes.md`, the spec, and `docs/experiments.md`.

- [ ] **Step 1: Draft the report following the spec's required structure**

Sections (from `project-discription-ar.md`): theoretical intro (PMSM, fault types), signal processing, Wavelet Transform + Scalogram explanation, CNN fundamentals, data preparation steps (sim + real, per-channel), training/testing results, performance analysis (accuracy, confusion matrix, effect of dataset size/quality), conclusions. Embed `results/sim_vs_real_grid.png`, `results/confusion_*.png`, and the experiments table.

- [ ] **Step 2: Verify length and figure references**

Run: `wc -w docs/report/report.md` and confirm every `results/*.png` referenced exists.
Expected: ~25–35 pages of content; no broken figure links.

- [ ] **Step 3: Commit**

```bash
git add docs/report/report.md && git commit -m "docs: final report draft"
```

---

### Task 5.2: Presentation (15–20 slides)

**Files:**
- Create: `docs/presentation/slides.md` (Marp/Reveal-compatible) or `.pptx`

**Interfaces:**
- Produces: 15–20 slides: project idea, system diagram, scalogram examples, CNN architecture diagram, results, conclusions.

- [ ] **Step 1: Draft slides**

Build from the report's key figures and the architecture diagram in the design spec. Keep one idea per slide; use `results/` figures directly.

- [ ] **Step 2: Verify slide count**

Run: `grep -c '^---' docs/presentation/slides.md` (for Marp slide separators)
Expected: 15–20 slides.

- [ ] **Step 3: Commit**

```bash
git add docs/presentation/slides.md && git commit -m "docs: final presentation"
```

---

## Self-Review (completed)

- **Spec coverage:** §3 architecture → Tasks 1.1–1.9, 4.1; §4 phasing → Phases 0–5; theory study → Task 5.1; sim source → 2.1/2.2; real per-channel → 1.3/3.1; scalograms → 1.5; CNN/metrics → 1.7–1.9; tuning/overfitting → 4.2; deliverables → 5.1/5.2; risks (leakage 1.2, sim limits 2.2, fusion pairing 4.1) addressed.
- **Placeholder scan:** the only deliberate seams are `load_recordings()` (real/vibration dataset parsers) and the LR/dropout wiring in 4.2 — both are dataset-/experiment-specific and explicitly flagged as TODO-FOR-USER because they cannot be written until Task 0.3 fixes the dataset format. All algorithmic logic is complete and tested.
- **Type consistency:** `segment_signal`, `add_record`, `make_dataset`, `build_cnn`, `metrics_from_predictions`, `assign_splits` signatures are reused identically across tasks; manifest columns match the Global Constraints everywhere.
