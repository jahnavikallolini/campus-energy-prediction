#!/usr/bin/env bash
# Runs the pipeline up to the end of preprocessing/EDA (modeling stages are added later).
set -euo pipefail
python -m src.data.audit
python -m src.data.preprocess
python -m src.features.build_features
python -m tests.test_features
python -m src.analysis.lecture_diagnostics
python -m src.analysis.eda
python -m src.data.coverage_report
