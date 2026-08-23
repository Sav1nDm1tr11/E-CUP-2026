# Task 4 report — model adapters, artifacts, inference

## Raw verification summaries

- Focused command: `python -m unittest tests.test_models tests.test_artifacts tests.test_inference -v`
  - Latest run: **18/18 tests passed**, including strict CatBoost objective, positive-only refit, exact 2D EBM bags/search/governance, RSS-tree overhead, true bundle persistence/provenance and 91-feature inference regressions.
- Full v2 command: `python -m unittest discover -s tests -v`
  - Latest run: **39 passed, 1 failed**. The only failure is the expected `test_required_distributions_are_installed`: the optional `interpret` distribution is absent. `models.py` does not import InterpretML at package import time; `fit_ebm_classifier` imports it only when EBM fitting is requested.
- `python -m compileall -q src`: passed.
- Toy contract smoke: positive-only regressor fit used only positive rows for inner and outer refit, applied `log1p`, and returned a positive selected iteration; the 70% EBM memory gate returned a structured `ResourceRejection` before estimator construction.

## Implemented contract

- LightGBM classifier and positive-only regressor use separate inner early-stopping and fresh fixed-iteration outer refits. Outer report rows are prediction-only.
- CatBoost requires strict inner end-to-end RMSLE inputs, exact spec distributions/settings, stable temporal/user ordering, retains winning `ParameterSampler` parameters, and refits a fresh estimator. Ordered trials are capped at three and require finite Plain RMSLE/time/peak-memory governance with the 3x/0.0005 rule.
- EBM searches every bounded candidate (≤12) on strict inner RMSLE, applies a monotonic 90-minute callback, uses exact 2D temporal bags `(n_outer_train, 8)`, native `estimate_mem(X_outer_train, y_outer_train, data_multiplier=1)`, ndarray best-iteration aggregation by maximum positive stage/bag, and process-tree RSS toy-fit overhead. Missing/invalid overhead or temporal-bags `TypeError` is a structured rejection.
- Checkpoints, JSON, Parquet, and model artifacts use sibling temporary paths followed by flush/close and atomic replace. Checkpoints carry a fingerprint and reject corruption or mismatch.
- Versioned manifests/bundles persist classifier files, positive regressor, calibrator, config/feature/data hashes, package versions, trained-through cutoff, blend history, and pre/post-January configs; `load_bundle` requires expected provenance hashes/version and validates alignment before use.
- Inference consumes loaded bundle components, removes reserved fields from estimator matrices, validates exactly 91 unique ordered features/hash and model-weight alignment, rejects probabilities outside `[0,1]`, preserves expected user order, and emits raw base, weighted/calibrated probabilities, positive-log, prediction-log, and `expm1` GMV audit outputs.

## Concerns / follow-up

- Installing `interpret==0.7.8` is an environment action outside this implementation task; no dependency was added or imported eagerly.
- Native model round-trip functions require the corresponding optional runtime package at call time (LightGBM/CatBoost/pyarrow/joblib), as intended by the artifact format.
