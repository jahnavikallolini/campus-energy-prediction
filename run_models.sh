#!/usr/bin/env bash
# Modeling stage (run after ./run_all.sh). Tuning/evaluation checkpoints make reruns resumable;
# delete results/tuning_checkpoints and results/eval_checkpoints to recompute from scratch.
set -euo pipefail
python -m src.models.tune
python -m src.models.evaluate
python -m tests.test_models
python -m src.analysis.model_report
python -m src.analysis.timestamp_shift_diagnostic
python -m src.analysis.build_eval_report
