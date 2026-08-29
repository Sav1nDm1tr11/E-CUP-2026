# LSTM Automatic Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the current best LSTM notebook into one restartable Colab GPU pipeline that trains the unchanged LSTM, exports exact heads, automatically selects an early-stopped temporal ML calibrator, and writes a calibrated submission.

**Architecture:** Keep neural training in `07_LSTM.ipynb`; place deterministic feature construction, temporal selection, bounded gate correction, and bundle persistence in a tested Python module. The notebook becomes thin orchestration: save all temporal fold predictions, export per-seed inference heads, call the calibration module, and report baseline versus candidate.

**Tech Stack:** Python, NumPy, pandas, scikit-learn, LightGBM, joblib, PyTorch, Jupyter/Colab.

**Spec:** `docs/superpowers/specs/2026-08-30-lstm-auto-calibration.md`

## Global Constraints

- Preserve the current `JointHurdleLSTM-v4-expanded-es` architecture and its three-seed log-space ensemble.
- Use chronological temporal evaluation and bounded search.
- Use LightGBM early stopping for every hyperparameter candidate.
- Never use identity, target, residual, or absolute cutoff fields as ML features.
- Preserve unrelated changes in `06_LSTM_Data_Preparation.ipynb` and untracked `graphify-out/`.
- Do not require LSTM retraining merely to rerun calibration when valid exported heads already exist.

---

### Task 1: Calibration core and tests

**Files:**
- Create: `notebooks/modeling/lstm_gate_calibrator.py`
- Create: `notebooks/modeling/tests/test_lstm_gate_calibrator.py`

**Interfaces:**
- Produces `CalibrationSearchConfig`, `build_meta_frame`, `select_temporal_calibrator`, `fit_production_calibrator`, `apply_calibrator_per_seed`, `save_calibrator_bundle`, and artifact validators.
- Consumes OOF and inference component DataFrames plus `data/lstm` static snapshots.

- [x] Write failing tests for exact user alignment, forbidden features, temporal ordering, zero-correction identity, early-stopping callbacks, guardrail fallback, and per-seed log aggregation.
- [x] Run the focused tests and confirm failures are caused by the missing module behavior.
- [x] Implement the minimal core with Beta/Platt baseline, bounded LightGBM candidate search, RF baseline, temporal reports, and artifact manifests.
- [x] Run focused and existing proxy-gate tests.
- [x] Review the public interfaces and commit the independently testable core.

### Task 2: Make the existing LSTM notebook emit exact calibration artifacts

**Files:**
- Modify: `notebooks/modeling/07_LSTM.ipynb`
- Create: `notebooks/modeling/tests/test_lstm_auto_calibration_notebook.py`

**Interfaces:**
- Produces `oof_best_epoch.parquet` and `inference_components_by_seed.parquet` for Task 1 interfaces.
- Calls Task 1 orchestration only after baseline inference succeeds.

- [x] Write a structural notebook test asserting every temporal fold saves epoch predictions, selected-epoch OOF is exported, every final seed exports all heads, and calibration runs after baseline submission.
- [x] Run the test and confirm it fails on the current notebook.
- [x] Modify notebook cells mechanically while preserving current outputs and architecture.
- [x] Add a final documented calibration section with dependency install, bounded search configuration, automatic resume/cache behavior, report display, and calibrated submission generation.
- [x] Run structural tests, JSON validation, and focused calibration tests.
- [x] Commit the notebook integration separately.

### Task 3: End-to-end verification and review

**Files:**
- Review: all files changed by Tasks 1-2

**Interfaces:**
- Validates the complete Colab execution contract without retraining the full LSTM locally.

- [x] Run the synthetic temporal end-to-end calibration test.
- [x] Run all modeling calibration/proxy tests.
- [x] Parse every notebook cell and compile every code cell that is valid outside notebook magics.
- [x] Check the git diff for unrelated notebook/output changes and secrets.
- [x] Perform final spec and code-quality review; fix any load-bearing finding and rerun affected checks.
