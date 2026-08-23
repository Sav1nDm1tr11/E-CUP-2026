# Task 2 report: RED behavior contract

## Files

Created and committed exactly these eight files under `notebooks/modeling/two_stage_v2/tests/`:

- `test_data_contract.py`
- `test_temporal_split.py`
- `test_metrics.py`
- `test_calibration.py`
- `test_blending.py`
- `test_models.py`
- `test_artifacts.py`
- `test_inference.py`

## Contract coverage

- Exact ordered 91-feature contract, cutoff loading dtypes, and submission validation.
- Nested cutoff construction and expanding split exclusion of outer report rows.
- RMSLE and deterministic paired cluster-bootstrap result contract.
- Sigmoid calibration shape, finite probabilities, normalization, and fitted coefficients.
- Unique convex simplex weights and soft-log/expm1 GMV formula.
- Two-phase estimator fit/refit separation with a local recording fake.
- Atomic checkpoint round trip and fingerprint mismatch rejection.
- Inference user-order preservation and a separate 250,000-row validation test without feature-matrix allocation.
- Review hardening: permutation rejection, cutoff filtering, int8 binary targets, negative/NaN/inf/duplicate/row-count submission rejection, hand-checked RMSLE, deterministic clustered bootstrap sign and seed, shuffled-row temporal index coverage, calibration bounds/monotonicity/sklearn cloning, recording estimator eval/refit separation, past-only blend state, feature alignment and auditable inference fakes, and corrupted-checkpoint rejection.

## Raw RED evidence

Command (Python 3.14):

```powershell
Set-Location notebooks/modeling/two_stage_v2
& 'C:\Users\savin\AppData\Local\Programs\Python\Python314\python.exe' -m unittest discover -s tests -v
```

Result after review hardening: 9 discovered tests; 8 new test modules fail at import solely with `ModuleNotFoundError` for planned `src` modules (`artifacts`, `blending`, `calibration`, `data_io`, `inference`, `metrics`, `models`, `temporal_split`). The pre-existing `test_environment` has one unrelated failure because the `interpret` distribution is absent. `python -m compileall -q tests` completed successfully. No production modules were added or modified.

## Commit

Original commit: `5ac708b269bcd56c5e561ab6854afbe0ab84b061` — `test: define two-stage ensemble v2 behavior`. Review fixes are in a separate follow-up commit.

## Concerns

- The repository-wide environment test remains red due to the missing `interpret` package; this is outside Task 2 ownership.
- Model and inference dependency-injection signatures were not fully enumerated in the design. Tests explicitly document minimal seams: `fit_lgbm_classifier(frame, target, fold=..., estimator_factory=...)` and `predict_ensemble(frame, classifiers=..., positive_regressor=..., calibrator=..., weights=..., feature_names=..., expected_user_ids=...)`.
- The artifact test validates corrupted-checkpoint rejection and does not claim an injected pre-replace failure simulation because no failure-injection hook is specified by the design.
