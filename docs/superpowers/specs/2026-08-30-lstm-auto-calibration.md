# LSTM Automatic Gate Calibration Specification

## Goal

Extend the current `JointHurdleLSTM-v4-expanded-es` notebook so one GPU run performs the existing temporal CV and three-seed final training, exports all LSTM heads, automatically selects a classical-ML activity calibrator with temporal validation and early stopping, and writes both baseline and calibrated submissions.

## Required behavior

- Preserve the current LSTM architecture, losses, temporal folds, selected-epoch policy, three final seeds, and baseline submission.
- During temporal CV, persist per-user predictions for every fold and epoch required to export the globally selected `BEST_EPOCH` without retraining LSTM folds.
- Export an exact OOF table with `user_id`, cutoff, target, source/provenance, and the `pred_log`, `gate_prob`, `positive_log`, `direct_log`, and `hurdle_log` heads.
- During final inference, export the same components separately for seeds 42, 143, and 2026.
- Build point-in-time meta-features from the LSTM heads, `history_length`, and the prepared static snapshot. Never use `user_id`, target values, residuals, or absolute cutoff identifiers as model features.
- Compare a conservative probability-calibration baseline, LightGBM candidates, and a Random Forest robustness baseline.
- Use chronological validation. Hyperparameter search and tree-count selection must use early stopping and must not use the final temporal audit slice as an ordinary shuffled fold.
- Correct the gate in logit space with a bounded correction. Applying zero correction must reproduce the original per-seed `pred_log`.
- Apply the selected calibrator to every final seed separately, then average corrected `pred_log` and convert with `expm1`.
- Save a versioned bundle, feature contract, selection report, temporal metrics, and calibrated submission.
- If no candidate satisfies guardrails, write diagnostics and keep the baseline submission as the selected output.

## Runtime constraints

- Target environment: Google Colab GPU for LSTM training; calibration may run on CPU after GPU inference.
- Install missing `lightgbm`, `pyarrow`, `joblib`, and compatible scikit-learn dependencies inside the notebook.
- Search must be bounded by an explicit trial count and LightGBM early stopping.
- The notebook must resume existing valid LSTM checkpoints and skip already completed stages when artifacts match the current signatures.

## Verification

- Unit tests cover feature leakage guards, temporal split construction, probability correction identity, early-stopping configuration, model selection guardrails, per-seed aggregation, and artifact schema validation.
- A synthetic end-to-end test trains the calibration pipeline on small temporal data without GPU.
- Notebook structure tests confirm the OOF export, per-seed component export, calibration invocation, and two submission paths are present in execution order.
