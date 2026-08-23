# Task 4 report — model adapters, artifacts, inference

## Raw verification summaries

- Focused command: `python -m unittest tests.test_models tests.test_artifacts tests.test_inference -v`
  - Latest run: **13/13 tests passed**, including strict CatBoost objective, positive-only refit, EBM gate/bags, manifest/native-save and invalid-inference regressions.
- Full v2 command: `python -m unittest discover -s tests -v`
  - Latest run: **34 passed, 1 failed**. The only failure is the expected `test_required_distributions_are_installed`: the optional `interpret` distribution is absent. `models.py` does not import InterpretML at package import time; `fit_ebm_classifier` imports it only when EBM fitting is requested.
- `python -m compileall -q src`: passed.
- Toy contract smoke: positive-only regressor fit used only positive rows for inner and outer refit, applied `log1p`, and returned a positive selected iteration; the 70% EBM memory gate returned a structured `ResourceRejection` before estimator construction.

## Implemented contract

- LightGBM classifier and positive-only regressor use separate inner early-stopping and fresh fixed-iteration outer refits. Outer report rows are prediction-only.
- CatBoost uses strict inner end-to-end RMSLE when positive-log/GMV arrays are supplied, retains winning `ParameterSampler` parameters, and refits a fresh estimator. Plain/Ordered flags, native missing values, `allow_writing_files=False`, deterministic seed 42, and at most 10 estimator threads are enforced; Ordered trials are capped at three with governance hooks.
- EBM has lazy import, native `estimate_mem(data_multiplier=1)` support, explicit temporal bags, fixed-round outer refit settings, measured overhead hooks, and a strict estimated-memory-plus-overhead gate. A temporal-bags `TypeError` is a structured rejection, never a random fallback.
- Checkpoints, JSON, Parquet, and model artifacts use sibling temporary paths followed by flush/close and atomic replace. Checkpoints carry a fingerprint and reject corruption or mismatch.
- Versioned manifests/bundles carry config/feature/data hashes, package versions, trained-through cutoff, blend history, and pre/post-January configs; loaders validate alignment before use.
- Inference removes reserved fields from estimator matrices, validates exact feature order/hash and model-weight alignment when a bundle is supplied, rejects probabilities outside `[0,1]`, preserves expected user order, and emits raw base, weighted/calibrated probabilities, positive-log, prediction-log, and `expm1` GMV audit outputs.

## Concerns / follow-up

- Installing `interpret==0.7.8` is an environment action outside this implementation task; no dependency was added or imported eagerly.
- Native model round-trip functions require the corresponding optional runtime package at call time (LightGBM/CatBoost/pyarrow/joblib), as intended by the artifact format.
