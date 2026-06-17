# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A **university research project**, not a conventional software codebase. The goal (see `project-discription-ar.md`, in Arabic) is a fault-diagnosis pipeline for **PMSM** (Permanent Magnet Synchronous Motors):

> simulate/collect motor signals → apply Continuous Wavelet Transform → render **Wavelet Scalogram** images → train a **CNN** to classify operating states and detect faults (e.g. healthy, overload, inter-turn fault).

Students: ملهم فتنة (Mulham Fetna) and محمد زين قباني. Deliverables are a labelled image dataset, training code, a final CNN model, a 25–35 page report, and a 15–20 slide presentation.

**The full pipeline is now implemented in Python** (`python/`), runs end-to-end without MATLAB, and is unit-tested. MATLAB scripts (`matlab/`) remain as an alternative scalogram/simulation path. See `README.md` for the architecture and `docs/superpowers/{specs,plans}/` for the design rationale and step-by-step plan.

### Working on the Python pipeline

- Environment: **Python 3.10–3.12** in a venv (TensorFlow has no wheel for 3.13/3.14). `make setup` builds `.venv` on `python3.10`.
- Commands (via `Makefile`): `make test`, `make demo` (synthetic end-to-end), `make train SIGNAL=current EPOCHS=20`, `make evaluate SIGNAL=current`. Direct: `.venv/bin/python -m python.<module>`.
- Single test: `.venv/bin/python -m pytest python/tests/test_split.py::test_no_recording_spans_splits -v`.
- `config.yaml` holds every parameter; all modules read it. `data/manifest.csv` is the single source of truth linking signal → scalogram → label/split.
- TF-dependent tests use `pytest.importorskip("tensorflow")` so they skip when TF is absent; keep TF imports lazy inside functions where a pure-logic helper must stay importable without TF.
- `data/`, `models/*.keras`, and `simscape-pmsm/` are gitignored. The KAIST dataset goes in `data/raw/mendeley_pmsm/` (see `docs/data-audit.md`).

Most root files remain research artifacts (PDFs, saved MathWorks doc pages, notes).

## Repository layout

- `project-discription-ar.md` — authoritative project plan and task breakdown (Arabic). Read this first to understand scope and the intended next steps.
- `notes.md` — study notes on PMSM/FOC control, Fourier vs. Wavelet transforms, and CNNs. Conceptual background; contains many typos but captures the intended theory.
- `references.md` — source links: tutorials, MathWorks docs, and datasets (Kaggle, Mendeley, Zenodo, IEEE DataPort) to draw real PMSM signals from.
- `*.pdf`, `pmsm-matlab*.html` — reference papers and saved MathWorks documentation pages on PMSM modelling and fault simulation.
- `FOC_PMSM-main/` — Simulink Field-Oriented-Control model of a PMSM. A source of simulated current/speed signals.
- `simscape-pmsm/` — **a separate git repo** (the only version-controlled directory here). Custom Simscape components modelling a PMSM with **inter-turn stator fault** injection.

## FOC_PMSM-main (Simulink FOC model)

- `Motor_script.m` — MATLAB parameter script. **Run this first** to populate the workspace (`pmsm`, `inverter`, `target`, `PU_System`, `PI_params` structs) before opening the model. Note: the README says to run `Motor_Parameter.m`, but the actual file is `Motor_script.m`.
- `FOCsimulation.slx` — open in Simulink, pick a pre-set input case (simulation time is encoded in each case name), and watch the "Speed tracking scope".
- `slprj/`, `*.slxc` — Simulink build cache; regenerated artifacts, not source.
- To capture data for the scalogram stage: the script notes that `out.simout.signals.values` holds the logged signal array after a run.

## simscape-pmsm (Simscape fault model — independent git repo)

`cd simscape-pmsm` before running any `git` command; the workspace root is not a git repo.

- `FaultyPMSM.ssc` / `NonFaultyPMSM.ssc` — Simscape language component definitions for a PMSM, with and without inter-turn fault. The faulty model adds an `sigma` parameter (ratio of shorted turns) plus thermal dependence of `Rs` and PM flux. Based on Otava et al., "Interior PMSM Stator Winding Fault Modelling" (doi:10.1016/j.ifacol.2015.07.055).
- `.ssc` files are Simscape source: they must be in a `+package` namespace folder and built with `ssc_build` in MATLAB to produce usable Simulink blocks.
- **Known limitations** (from its README): the model is only valid for **1 pole-pair** motors and does not converge in complex sensorless-FOC setups (suspected ill-conditioned inductance matrices). Keep this in mind before wiring it into the FOC model.

## Working notes for future agents

- The two motor models are alternative **signal sources** for the dataset, not an integrated system — and they conflict on key assumptions (`FaultyPMSM.ssc` defaults to 6 pole pairs but documents 1-pole-pair validity; `Motor_script.m` uses 4 pole pairs). Verify pole-pair and unit assumptions before combining or comparing their outputs.
- When implementing the missing stages, follow the plan in `project-discription-ar.md` and the datasets in `references.md` rather than inventing a new pipeline.
- MATLAB `.slx`/`.ssc` work cannot be executed here; produce scripts the user runs in MATLAB, and confirm toolbox availability (Deep Learning, Wavelet, Simscape Electrical).
