#!/usr/bin/env bash
# Final stages (run after ./run_all.sh and ./run_models.sh): test periods, anomalies, PCA/K-Means, report and deck data.
set -euo pipefail
python -m src.analysis.timestamp_offset_by_month
python -m src.analysis.test_periods
python -m src.anomaly.detect
python -m src.anomaly.plots
python -m src.analysis.pca_kmeans
python -m tests.smoke_app
python -m src.analysis.build_deck_data
python -m src.analysis.build_ieee_report         # needs pdflatex, bibtex and IEEEtran
(cd presentation && node build_deck.js)          # needs Node.js + pptxgenjs (and the pptx skill's apply_theme.js path in build_deck.js)
