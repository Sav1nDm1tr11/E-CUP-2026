# Task 4 report — model adapters, artifacts, inference

## Raw verification summaries

- Focused command: `python -m unittest tests.test_models tests.test_artifacts tests.test_inference -v`
  - Result: **6/6 tests passed**.
- Full v2 command: `python -m unittest discover -s tests -v`
  - Result: **27 passed, 1 failed**.
  - The only failure is `test_required_distributions_are_installed`: the optional `interpret` distribution is absent in the runtime. `models.py` does not import InterpretML at package import time; `fit_ebm_classifier` imports it only when EBM fitting is requested.
- `python -m compileall -q src`: passed.
- Toy contract smoke: positive-only regressor fit used only positive rows for inner and outer refit, applied `log1p`, and returned a positive selected iteration; the 70% EBM memory gate returned a structured `ResourceRejection` before estimator construction.

## Implemented contract

- LightGBM classifier and positive-only regressor use separate inner early-stopping and fresh fixed-iteration outer refits. Outer report rows are prediction-only.
- CatBoost uses sequential `ParameterSampler` candidates, Plain/Ordered flags, native missing values, `allow_writing_files=False`, deterministic seed 42, and at most 10 estimator threads.
- EBM has lazy import, explicit temporal bags, fixed-round outer refit settings, and a strict estimated-memory-plus-overhead gate.
- Checkpoints, JSON, Parquet, and model artifacts use sibling temporary paths followed by flush/close and atomic replace. Checkpoints carry a fingerprint and reject corruption or mismatch.
- Inference validates exact ordered features, retains raw base probabilities and weighted/calibrated probabilities, preserves expected user order, and emits positive-log, prediction-log, and `expm1` GMV outputs.

## Concerns / follow-up

- Installing `interpret==0.7.8` is an environment action outside this implementation task; no dependency was added or imported eagerly.
- Native model round-trip functions require the corresponding optional runtime package at call time (LightGBM/CatBoost/pyarrow/joblib), as intended by the artifact format.
