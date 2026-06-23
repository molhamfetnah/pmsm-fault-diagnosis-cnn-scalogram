# PMSM fault-diagnosis pipeline — convenience targets.
# Use the project virtualenv interpreter (Python 3.10-3.12; TensorFlow has no
# wheel for 3.13/3.14).
PY := .venv/bin/python
PIP := .venv/bin/pip
SIGNAL ?= current
EPOCHS ?= 20

.PHONY: help setup test demo simulate scalograms split train train-fusion evaluate report docs docs-ar clean

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
	@echo "  train-fusion  train+eval dual-branch current+vibration model (EPOCHS=20)"
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

train-fusion:
	$(PY) -m python.train_fusion --epochs $(EPOCHS)

report:
	$(PY) -m python.report_metrics

# Build presentable report (PDF/DOCX) + slides (PPTX/PDF) into docs/build/.
# Needs: pandoc, xelatex (texlive-xetex). Run pandoc from each doc's dir so the
# relative image paths resolve and figures embed.
docs:
	mkdir -p docs/build
	$(PY) -m python.figures_engineering
	cd docs/report && pandoc report.md -o ../build/report.pdf --pdf-engine=xelatex \
		-V geometry:margin=2.5cm -V fontsize=12pt -V linestretch=1.5 \
		-V mainfont="DejaVu Sans" -V monofont="DejaVu Sans Mono" --toc -V colorlinks=true
	cd docs/report && pandoc engineering-background.md -o ../build/engineering-background.pdf \
		--pdf-engine=xelatex -V geometry:margin=2.3cm -V fontsize=11pt -V linestretch=1.3 \
		-V mainfont="DejaVu Sans" -V monofont="DejaVu Sans Mono" --toc -V colorlinks=true
	cd docs/report && pandoc engineering-background.md -o ../build/engineering-background.docx --toc
	cd docs/report && pandoc report.md -o ../build/report.docx --toc
	cd docs/presentation && pandoc slides.md -o ../build/slides.pptx
	cd docs/presentation && pandoc slides.md -t beamer -o ../build/slides.pdf \
		--pdf-engine=xelatex -V mainfont="DejaVu Sans" -V monofont="DejaVu Sans Mono" -V fontsize=9pt
	for f in build-walkthrough defense-study-guide; do \
		pandoc docs/$$f.md -o docs/build/$$f.pdf --pdf-engine=xelatex \
			-V geometry:margin=2.2cm -V fontsize=11pt -V mainfont="DejaVu Sans" \
			-V monofont="DejaVu Sans Mono" --toc -V colorlinks=true; done

# Arabic deliverables (RTL). DOCX/PPTX keep original symbols; the PDF is built from
# a symbol-sanitized copy because Amiri lacks math/arrow glyphs. Needs the Amiri font.
docs-ar:
	mkdir -p docs/build
	$(PY) -m python.figures_engineering
	cd docs/report && pandoc report-ar.md -o ../build/report-ar.docx --toc
	$(PY) docs/_sanitize_for_pdf.py docs/report/engineering-background-ar.md docs/report/_ar_tmp.md
	cd docs/report && pandoc _ar_tmp.md -o ../build/engineering-background-ar.pdf --pdf-engine=xelatex \
		-V mainfont="Amiri" -V monofont="Amiri" -V geometry:margin=2.3cm -V fontsize=12pt \
		-V linestretch=1.4 -V dir=rtl -V lang=ar --toc -V colorlinks=true; rm -f docs/report/_ar_tmp.md
	cd docs/presentation && pandoc slides-ar.md -o ../build/slides-ar.pptx
	$(PY) docs/_sanitize_for_pdf.py docs/report/report-ar.md docs/report/_ar_tmp.md
	cd docs/report && pandoc _ar_tmp.md -o ../build/report-ar.pdf --pdf-engine=xelatex \
		-V mainfont="Amiri" -V monofont="Amiri" -V geometry:margin=2.5cm -V fontsize=12pt \
		-V linestretch=1.5 -V dir=rtl -V lang=ar --toc -V colorlinks=true; rm -f docs/report/_ar_tmp.md
	$(PY) docs/_sanitize_for_pdf.py docs/presentation/slides-ar.md docs/presentation/_ar_tmp.md
	cd docs/presentation && pandoc _ar_tmp.md -o ../build/slides-ar.pdf --pdf-engine=xelatex \
		-V mainfont="Amiri" -V monofont="Amiri" -V geometry:margin=2cm -V dir=rtl -V lang=ar; rm -f docs/presentation/_ar_tmp.md
	for f in build-walkthrough-ar defense-study-guide-ar; do \
		$(PY) docs/_sanitize_for_pdf.py docs/$$f.md docs/_tmp.md; \
		pandoc docs/_tmp.md -o docs/build/$$f.pdf --pdf-engine=xelatex \
			-V mainfont="Amiri" -V monofont="Amiri" -V geometry:margin=2.2cm -V fontsize=12pt \
			-V linestretch=1.4 -V dir=rtl -V lang=ar --toc -V colorlinks=true; rm -f docs/_tmp.md; done

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
