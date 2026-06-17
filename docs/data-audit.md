# Dataset Audit

Findings from auditing the candidate sources in `references.md` (June 2026).

## Decision

- **Primary real dataset (current + vibration):** KAIST *Vibration and Current
  Dataset of Three-Phase PMSM with Stator Faults* — Mendeley `rgn5brrgrn` v5,
  DOI `10.17632/rgn5brrgrn.5`, **CC-BY-4.0**.
  - Current @ **100 kHz**, vibration @ **25.6 kHz** — high enough for CWT scalograms.
  - Conditions: **normal**, **inter-turn** short, **inter-coil** short; motors at
    1.0 / 1.5 / 3.0 kW; 8 severities. Format: **TDMS**.
  - Loader: `python/ingest_mendeley.py` (parses the filename convention, decimates
    to `target_fs`, segments, writes the manifest).
  - Label mapping: `normal`/0% → `Healthy`; `interturn`/`intercoil` (>0%) → `InterTurn`.
  - Download is via Mendeley's authenticated web flow ("Download All") into
    `data/raw/mendeley_pmsm/`.

- **Secondary real dataset (tabular reference only):** *Comprehensive Dataset for
  Fault Detection and Diagnosis in Inverter-Driven PMSM Systems* — Zenodo
  `13974503`, **CC-BY-4.0**. Downloaded to `data/raw/zenodo_pmsm/`.
  - **Sampled at 10 Hz** (tabular sensor features: Ia, Ib, VDC, IDC, T1–T3, VD;
    label `FDD`). Covers inverter faults (open-circuit, short-circuit,
    over-temperature) of an inverter-driven PMSM.
  - **Not used for scalograms** — 10 Hz (Nyquist 5 Hz) carries no useful
    time-frequency content for CWT. Kept as a real-world tabular-ML reference and
    for the report's discussion of data quality vs. method fit.

- **Demagnetization class:** not present in the real datasets above. It is
  represented by the **synthetic generator** and can be added from the **MATLAB**
  simulation; document this scope in the report (or drop to a 3-class problem if
  only real data is used).

## Candidate sources (audited)

| name | url | signal | classes | fs | license | verdict |
|---|---|---|---|---|---|---|
| KAIST PMSM stator faults | https://data.mendeley.com/datasets/rgn5brrgrn/5 | current + vibration | normal, inter-turn, inter-coil | 100 kHz / 25.6 kHz | CC-BY-4.0 | **chosen (primary)** |
| Inverter-driven PMSM FDD | https://zenodo.org/records/13974503 | tabular sensors | normal + inverter OC/SC/over-temp | 10 Hz | CC-BY-4.0 | reference only (too slow for CWT) |
| Kaggle pmsm-smart-control | https://www.kaggle.com/datasets/ziya07/pmsm-smart-control-dataset | control/tabular | — | low | — | not needed (login-gated, control data) |
| IEEE-DataPort PMSM ITSC | https://ieee-dataport.org/documents/three-phase-pmsm-itsc-faults-stator-winding-dataset | current | inter-turn | — | gated | alternative if Mendeley unavailable |

## How to verify access

```bash
# Zenodo (open, scriptable):
curl -sL "https://zenodo.org/api/records/13974503" | python3 -c "import sys,json;d=json.load(sys.stdin);[print(f['key'],f['size']) for f in d['files']]"
# Mendeley: use the website "Download All" button (API requires auth).
```
