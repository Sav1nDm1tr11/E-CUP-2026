# Task 3 report (hardened review pass)

## Implemented

Created the seven owned core modules under `notebooks/modeling/two_stage_v2/src/`:

- immutable `ExperimentConfig`/`ModelSearchSpace` and canonical SHA-256 fingerprints;
- Arrow schema contract, project-root discovery, cutoff-filtered loading, stable feature-column loading, and dtype casting;
- exact ordered feature/target/submission/OOF validation;
- nested expanding cutoff folds (with `train_dates` as the complete outer-train history and `train_dates[:-1]` as inner train);
- cutoff-based `ExpandingCutoffSplit`;
- RMSLE and deterministic user-cluster paired bootstrap with equal-fold RMSLE averaging;
- sklearn-compatible sigmoid calibration;
- deterministic convex simplex blending, soft-log/expm1 prediction, and serializable past-only `BlendState`.

Review hardening additionally makes the blend boundary an explicit trusted input, requires `cutoff_date` plus both `p_lgbm` and `p_catboost`, validates binary target consistency, enforces configurable logloss/Brier degradation limits, and records JSON-safe candidate diagnostics. Feature validation now permits NaN but rejects infinity without a full float64 matrix copy; bootstrap samples retain every fold; and default configuration contains the approved cutoff/report dates and explicit search budgets.

## Verification

Focused command (Python 3.14):

```powershell
Set-Location notebooks/modeling/two_stage_v2
& 'C:\Users\savin\AppData\Local\Programs\Python\Python314\python.exe' -m unittest tests.test_data_contract tests.test_temporal_split tests.test_metrics tests.test_calibration tests.test_blending -v
```

Result: **19 tests passed** (including review regression tests).

Full v2 command:

```powershell
& 'C:\Users\savin\AppData\Local\Programs\Python\Python314\python.exe' -m unittest discover -s tests -v
```

Result: Task 3 tests passed; remaining collection errors are the expected unimplemented Task 4 modules (`src.models`, `src.artifacts`, `src.inference`). The environment test also remains red because the pre-existing `interpret` distribution is not installed. Full suite now discovers 23 tests: 19 Task 3 passes, 3 expected Task 4 import errors, and the one environment dependency failure.

## Test correction

`tests/test_metrics.py` contained a demonstrable contradiction: its hand-written expected delta compared each prediction to itself (therefore exactly zero) while the same test required a strictly negative delta. The expectation now compares candidate and baseline predictions against the actual values, matching the documented `RMSLE_new - RMSLE_strict_baseline` contract. No production behavior was changed for this correction.

## Concerns

- `interpret` installation is outside Task 3 ownership.
- Full suite cannot become green until Task 4 adds its modules.
- No production data or legacy `two_stage` files were modified.
