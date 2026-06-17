# PMSM fault-diagnosis pipeline — convenience targets.
# Use the project virtualenv interpreter (Python 3.10-3.12; TensorFlow has no
# wheel for 3.13/3.14).
PY := .venv/bin/python
PIP := .venv/bin/pip
SIGNAL ?= current
EPOCHS ?= 20

.PHONY: help setup test demo simulate scalograms split train evaluate report clean

help:
	@echo "Targets:"
	@echo "  setup      create .venv (py3.10) and install requirements"
	@echo "  test       run the pytest suite"
	@echo "  demo       full synthetic end-to-end run (simulate -> ... -> evaluate)"
	@echo "  simulate   generate synthetic PMSM signals"
	@echo "  scalograms render CWT scalograms for every manifest row (Python/pywt)"
	@echo "  split      assign leakage-free train/val/test splits"
	@echo "  train      train the CNN     (SIGNAL=current|vibration EPOCHS=20)"
	@echo "  evaluate   evaluate on test  (SIGNAL=current|vibration)"
	@echo "  report     write results/summary.md"
	@echo "  clean      remove generated data/, models/, results artifacts"

setup:
	python3.10 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

test:
	$(PY) -m pytest -q

simulate:
	$(PY) -m python.simulate

scalograms:
	$(PY) -m python.scalogram

split:
	$(PY) -m python.run_split

train:
	$(PY) -m python.train --signal-type $(SIGNAL) --epochs $(EPOCHS)

evaluate:
	$(PY) -m python.evaluate --signal-type $(SIGNAL)

report:
	$(PY) -m python.report_metrics

demo: simulate
	$(PY) -m python.ingest_sim
	$(MAKE) scalograms
	$(MAKE) split
	$(MAKE) train SIGNAL=current EPOCHS=$(EPOCHS)
	$(MAKE) evaluate SIGNAL=current
	$(MAKE) report

clean:
	rm -rf data/raw/sim data/raw/segments data/scalograms data/manifest.csv
	rm -f models/cnn_*.keras results/history_*.json results/report_*.json \
	      results/confusion_*.png results/summary.md
