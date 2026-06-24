# Data Expansion — adding more datasets to the pipeline

Three complementary datasets were identified (see `data-audit.md`). None is openly
scriptable, so each needs a manual download/grant; this guide gives the exact steps
and the loader to run so each flows through the existing pipeline
(`ingest → scalograms → split → train/evaluate`).

The pipeline contract: a loader writes 0.5 s, 10 kHz `.npy` segments to
`data/raw/segments/` and rows to `data/manifest.csv` (columns in `python/manifest.py`).
After any loader runs: `make scalograms && .venv/bin/python -m python.run_split`,
then `make train SIGNAL=current` / `evaluate`.

---

## 1. Zenodo PMSM-elevator (15613954) — adds healthy *current* diversity

**Why:** raw current @ 4 kHz, healthy at 4 loads × 2 directions + stator short —
exactly the healthy diversity we lack, in a real elevator context. CC-BY-4.0.
**Access:** restricted — request it on Zenodo. Paste this when you click
*"Request access"* at <https://zenodo.org/records/15613954>:

> Subject: Access request — Multimodal PMSM-elevator dataset (10.5281/zenodo.15613954)
>
> Dear authors,
> I am an undergraduate Mechatronics Engineering student at Aleppo University working
> on a research-seminar project on PMSM stator-fault diagnosis using wavelet
> scalograms and CNNs. I would be grateful for access to your dataset to use the
> 4 kHz stator-current recordings (healthy and short-circuit) as an additional,
> independent source of healthy operating conditions. It will be used for
> non-commercial academic research and cited as per CC-BY-4.0. Thank you.
> — Mulham Fetna

**Once you have the `.xlsx`:** put it in `data/raw/zenodo_elevator/`. The Excel
layout (sheet/column names) isn't public, so this loader is the one piece not yet
written — send me one sheet's header row and I'll finalise
`python/ingest_elevator.py` (it will read the 4 kHz current column, resample to the
config `target_fs`, segment, and tag `dataset_name="elevator"`). The other two
loaders below are ready now.

---

## 2. IEEE DataPort PMSM-ITSC (10.21227/4jpc-qh81) — richest ITSC diversity

**Why:** 3-phase voltage+current, 12 torque-speed × 9 severities × 3 resistances —
huge operating-condition coverage for the exact fault. MAT format, ~67 GB.
**Access:** IEEE DataPort **subscription / IEEE membership**. Download via the site
or its AWS mirror.
**Integrate:** put the `.mat` files under `data/raw/ieee_pmsm/`, then
`.venv/bin/python -m python.ingest_ieee_pmsm`. The loader auto-detects current
channels in each MAT (handles both classic and v7.3/HDF5 MAT) and decimates to
`target_fs`. **Adapt point:** if the field names differ from the auto-detection,
set `CURRENT_KEYS`/`FS` at the top of `python/ingest_ieee_pmsm.py` after inspecting
one file (`python -c "import scipy.io,sys;print(scipy.io.loadmat(sys.argv[1]).keys())" <file>`).

---

## 3. University of Ottawa UOEMD-VAFCVS (Mendeley msxs4vj48g) — transfer test

**Why:** open, 42 kHz vibration+acoustic, healthy + stator/bearing/unbalance/… —
but **induction motors, not PMSM**. Use only for a **cross-machine transfer test**
(does our scalogram+CNN method generalise to another machine?). Not for the main
PMSM results.
**Access:** open on Mendeley — use the website "Download All" (same manual flow as
KAIST; the API needs auth).
**Integrate:** put the CSVs under `data/raw/uottawa/`, then
`.venv/bin/python -m python.ingest_uottawa`. Columns are
`accel1, acoustic, accel2, accel3, temperature` @ 42 kHz; the loader takes the
vibration channels, decimates, segments, and tags `dataset_name="uottawa"` so you
can train/evaluate on it separately or as a transfer set.

---

## Notes
- Keep new raw data **gitignored** (it already is: `data/raw/`).
- Mixing datasets: the manifest's `dataset_name` and `source` columns let you train
  on one set and test on another (true cross-dataset generalisation) — the most
  rigorous use of the extra data.
- Report it honestly: more *healthy* recordings is the one change that could move the
  current channel and firm up the vibration result (see `report.md` §8.4).
