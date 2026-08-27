# Two-Stage Ensemble v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, fully train, execute, and verify a notebook-driven LightGBM/CatBoost/EBM probability ensemble for two-stage GMV prediction without changing the existing 91-feature dataset.

**Architecture:** `10_Two_Stage_Ensemble_v2.ipynb` is a thin narrative orchestrator. Reusable data, temporal split, model, calibration, blending, diagnostic, artifact, and inference logic lives in a neighboring tested `src` package. Long-running model/fold work is checkpointed under `data/two_stage_v2_artifacts` and reused only when data/config/feature fingerprints match.

**Tech Stack:** CPython 3.14.0, NumPy 2.3.3, pandas 2.3.3, PyArrow 25.0.1, scikit-learn 1.8.0, LightGBM 4.6.0, CatBoost 1.2.10, InterpretML 0.7.8, Optuna 4.8.0, Matplotlib 3.10.7, Seaborn, nbformat 5.10.4, nbclient 0.10.4, unittest.

**Spec:** `docs/superpowers/specs/2026-08-24-two-stage-ensemble-v2-design.md`

## Global Constraints

- Do not modify `notebooks/modeling/two_stage`, `data/two_stage_artifacts`, the 91 features, rows, targets, or cutoff definitions.
- Create implementation only under `notebooks/modeling/two_stage_v2`; create runtime outputs only under `data/two_stage_v2_artifacts`.
- Use nested temporal model selection: inner cutoff selects parameters/iterations, full outer-train refit uses fixed iterations, outer report labels never enter fit/model selection.
- Rebuild strict OOF for the frozen LightGBM classifier and frozen positive-only LightGBM regressor.
- CatBoost 1.2.10 uses 20 sequential Plain trials; Ordered is at most 3 pilot trials.
- EBM 0.7.8 uses at most 12 configurations and must pass the exact resource gate from the design.
- Meta-layer is past-only: fold 1 → fold 2, folds 1–2 → fold 3, folds 1–3 → January.
- Final log prediction is `p_calibrated * max(predicted_positive_log, 0)` and final GMV is `expm1` of that value.
- All new Python APIs are typed, deterministic, concise, sklearn-compatible where applicable, and covered by behavior tests.
- Trial-level parallelism is 1; estimator thread count is at most 10.
- Preserve every unrelated dirty-worktree change. Never reset, clean, delete, or overwrite user files.
- Each implementer is not alone in the repository, owns only the files listed in its task, and must not revert edits made by others.

---

### Task 1: Scaffold, dependency contract, and environment smoke check

**Files:**
- Create: `notebooks/modeling/two_stage_v2/README.md`
- Create: `notebooks/modeling/two_stage_v2/requirements-v2.txt`
- Create: `notebooks/modeling/two_stage_v2/src/__init__.py`
- Create: `notebooks/modeling/two_stage_v2/tests/__init__.py`
- Create: `notebooks/modeling/two_stage_v2/tests/test_environment.py`

**Interfaces:**
- Produces exact package/version requirements used by every later task.
- Produces an importable neighboring `src` package and environment test command.

- [ ] **Step 1: Create the package directories and failing dependency test**

```python
from importlib.metadata import PackageNotFoundError, version
import unittest


class EnvironmentTests(unittest.TestCase):
    def test_required_distributions_are_installed(self):
        required = {
            "lightgbm": "4.6.0",
            "catboost": "1.2.10",
            "interpret": "0.7.8",
            "scikit-learn": "1.8.0",
            "nbclient": "0.10.4",
            "nbformat": "5.10.4",
        }
        missing = []
        mismatched = []
        for distribution, expected in required.items():
            try:
                actual = version(distribution)
            except PackageNotFoundError:
                missing.append(distribution)
                continue
            if actual != expected:
                mismatched.append((distribution, expected, actual))
        self.assertEqual(missing, [])
        self.assertEqual(mismatched, [])
```

- [ ] **Step 2: Run the environment test and record the expected RED state**

Run from `notebooks/modeling/two_stage_v2`:

```powershell
& 'C:\Users\savin\AppData\Local\Programs\Python\Python314\python.exe' -m unittest tests.test_environment -v
```

Expected before dependency installation: FAIL listing `interpret` as missing.

- [ ] **Step 3: Write exact direct dependencies**

```text
numpy==2.3.3
pandas==2.3.3
pyarrow==25.0.1
scikit-learn==1.8.0
lightgbm==4.6.0
catboost==1.2.10
interpret==0.7.8
optuna==4.8.0
matplotlib==3.10.7
seaborn==0.13.2
scipy==1.16.2
shap==0.52.0
nbformat==5.10.4
nbclient==0.10.4
ipykernel==7.0.1
psutil==7.1.0
joblib==1.5.3
```

- [ ] **Step 4: Document the exact working-directory and smoke-test commands**

README must state that the notebook runs from `notebooks/modeling/two_stage_v2`, reads `data/Prepared_data.parquet`, writes only `data/two_stage_v2_artifacts`, and uses the Python 3.14 interpreter above.

- [ ] **Step 5: Commit the dependency scaffold**

```powershell
git add -- notebooks/modeling/two_stage_v2
git commit -m "chore: scaffold two-stage ensemble v2"
```

---

### Task 2: Write the RED behavior contract for temporal modeling

**Files:**
- Create: `notebooks/modeling/two_stage_v2/tests/test_data_contract.py`
- Create: `notebooks/modeling/two_stage_v2/tests/test_temporal_split.py`
- Create: `notebooks/modeling/two_stage_v2/tests/test_metrics.py`
- Create: `notebooks/modeling/two_stage_v2/tests/test_calibration.py`
- Create: `notebooks/modeling/two_stage_v2/tests/test_blending.py`
- Create: `notebooks/modeling/two_stage_v2/tests/test_models.py`
- Create: `notebooks/modeling/two_stage_v2/tests/test_artifacts.py`
- Create: `notebooks/modeling/two_stage_v2/tests/test_inference.py`

**Interfaces:**
- Consumes the exact API signatures from the design.
- Produces failing tests that Task 3 must satisfy without changing test expectations.

- [ ] **Step 1: Test nested cutoff construction**

```python
def test_nested_fold_uses_last_past_cutoff_as_inner_validation():
    dates = pd.to_datetime([
        "2025-04-19", "2025-05-19", "2025-06-18", "2025-07-18"
    ])
    fold = build_nested_folds(dates, [pd.Timestamp("2025-07-18")])[0]
    assert fold.inner_valid_date == pd.Timestamp("2025-06-18")
    assert fold.outer_valid_date == pd.Timestamp("2025-07-18")
    assert max(fold.inner_train_dates) < fold.inner_valid_date
```

- [ ] **Step 2: Test that outer report rows never enter inner or refit indices**

```python
def test_expanding_split_keeps_outer_report_labels_out_of_fit():
    frame = toy_cutoff_frame()
    split = ExpandingCutoffSplit(build_nested_folds(
        frame["cutoff_date"].unique(), [pd.Timestamp("2025-07-18")]
    ))
    train_idx, report_idx = next(split.split(frame, groups=frame["cutoff_date"]))
    assert set(frame.iloc[train_idx]["cutoff_date"]) == {
        pd.Timestamp("2025-04-19"),
        pd.Timestamp("2025-05-19"),
        pd.Timestamp("2025-06-18"),
    }
    assert set(frame.iloc[report_idx]["cutoff_date"]) == {
        pd.Timestamp("2025-07-18")
    }
```

- [ ] **Step 3: Test probability calibration**

```python
def test_sigmoid_calibrator_returns_two_class_probabilities():
    calibrator = SigmoidCalibrator().fit(
        np.array([0.1, 0.2, 0.8, 0.9]),
        np.array([0, 0, 1, 1]),
    )
    result = calibrator.predict_proba(np.array([0.0, 0.5, 1.0]))
    assert result.shape == (3, 2)
    np.testing.assert_allclose(result.sum(axis=1), 1.0)
    assert np.isfinite(result).all()
```

- [ ] **Step 4: Test the convex simplex and final GMV formula**

```python
def test_simplex_grid_is_unique_and_sums_to_one():
    weights = simplex_grid(model_count=3, step=0.5)
    assert len(weights) == 6
    assert len({tuple(row) for row in weights}) == 6
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)
    assert (weights >= 0).all()


def test_soft_log_prediction_matches_manual_formula():
    result = combine_predictions(
        probability=np.array([0.25, 1.0]),
        positive_log=np.array([2.0, -1.0]),
    )
    np.testing.assert_allclose(result.prediction_log, [0.5, 0.0])
    np.testing.assert_allclose(result.prediction, np.expm1([0.5, 0.0]))
```

- [ ] **Step 5: Test checkpoint invalidation and atomic round trip**

```python
def test_checkpoint_is_rejected_when_config_fingerprint_changes(self):
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "fold_1.json"
        save_checkpoint(path, {"value": 1}, fingerprint="abc")
        self.assertEqual(
            load_checkpoint(path, expected_fingerprint="abc")["value"],
            1,
        )
        with self.assertRaises(CheckpointMismatchError):
            load_checkpoint(path, expected_fingerprint="different")
```

Use `unittest.TestCase` and `tempfile.TemporaryDirectory` throughout.

- [ ] **Step 6: Test estimator fit/refit separation with a recording fake**

The fake estimator must record inner-validation indices and refit indices. Assert that the outer report date appears in neither and that refit has more rows than the inner fit.

- [ ] **Step 7: Test inference order and 250,000-row validation separately**

Use a small unit test for order preservation and a validation test that passes a 250,000-length array without allocating a feature matrix.

- [ ] **Step 8: Run the new suite and verify RED for missing `src` APIs**

```powershell
& 'C:\Users\savin\AppData\Local\Programs\Python\Python314\python.exe' -m unittest discover -s tests -v
```

Expected: imports fail only because planned modules/functions do not exist. Syntax errors or collection errors inside tests must be fixed before commit.

- [ ] **Step 9: Commit the RED behavior contract**

```powershell
git add -- notebooks/modeling/two_stage_v2/tests
git commit -m "test: define two-stage ensemble v2 behavior"
```

---

### Task 3: Implement core data, split, metric, calibration, and blend APIs

**Files:**
- Create: `notebooks/modeling/two_stage_v2/src/config.py`
- Create: `notebooks/modeling/two_stage_v2/src/data_io.py`
- Create: `notebooks/modeling/two_stage_v2/src/validation.py`
- Create: `notebooks/modeling/two_stage_v2/src/temporal_split.py`
- Create: `notebooks/modeling/two_stage_v2/src/metrics.py`
- Create: `notebooks/modeling/two_stage_v2/src/calibration.py`
- Create: `notebooks/modeling/two_stage_v2/src/blending.py`
- Modify only if a test contains a demonstrable test bug: Task 2 test file concerned, with an appended report explaining the correction.

**Interfaces:**
- Produces the exact public interfaces described in design sections 5–6.
- `fit_walk_forward_blend` consumes past OOF columns `p_lgbm`, `p_catboost`, optionally `p_ebm`, `predicted_positive_log`, `target_gmv_30d`, and `target_nonzero`.
- Produces serializable `BlendState(weights, calibrator_a, calibrator_b, epsilon, objective, trained_through)`.

- [ ] **Step 1: Implement frozen dataclasses and stable SHA-256 fingerprinting**

`config_fingerprint` must serialize dataclasses with sorted JSON keys, UTF-8 encoding, and SHA-256.

- [ ] **Step 2: Implement Arrow contract reads and cutoff-filtered `float32` loading**

Use `pyarrow.dataset.dataset(...).to_table(filter=..., columns=...)`; reject a feature contract other than exactly 91 ordered feature names.

- [ ] **Step 3: Implement `CutoffFold`, `build_nested_folds`, and `ExpandingCutoffSplit`**

Raise a descriptive `ValueError` if fewer than two past cutoff dates exist for a requested outer fold.

- [ ] **Step 4: Implement strict metrics and paired cluster bootstrap by user**

Use local `numpy.random.default_rng(seed)`; sample user IDs with replacement, select all their rows, calculate per-fold RMSLE, and average folds equally. Return point delta, 2.5/97.5 percentiles, and resample count.

- [ ] **Step 5: Implement sklearn-compatible `SigmoidCalibrator`**

Use `BaseEstimator`, `ClassifierMixin`, `LogisticRegression(C=1e6, solver="lbfgs", random_state=42)`, `check_is_fitted`, and epsilon `1e-6`.

- [ ] **Step 6: Implement the deterministic convex grid and past-only fit**

For each grid weight, blend raw probabilities, fit sigmoid on the past pool, compute end-to-end RMSLE on that same fit pool only as the fit objective, enforce logloss/Brier constraints, then optionally refine inside the winning simplex neighborhood with SLSQP. The report fold is not accepted by this function.

- [ ] **Step 7: Run focused tests and make them GREEN**

```powershell
& 'C:\Users\savin\AppData\Local\Programs\Python\Python314\python.exe' -m unittest tests.test_data_contract tests.test_temporal_split tests.test_metrics tests.test_calibration tests.test_blending -v
```

- [ ] **Step 8: Run the full new suite**

Expected remaining failures: only unimplemented model/artifact/inference APIs owned by Task 4.

- [ ] **Step 9: Commit core APIs**

```powershell
git add -- notebooks/modeling/two_stage_v2/src notebooks/modeling/two_stage_v2/tests
git commit -m "feat: add temporal ensemble core"
```

---

### Task 4: Implement model adapters, resource gates, artifacts, and inference

**Files:**
- Create: `notebooks/modeling/two_stage_v2/src/models.py`
- Create: `notebooks/modeling/two_stage_v2/src/artifacts.py`
- Create: `notebooks/modeling/two_stage_v2/src/inference.py`
- Update only the Task 2 tests for these modules if a test bug is demonstrated.

**Interfaces:**
- Consumes `ExperimentConfig`, `CutoffFold`, metrics, and blend state from Task 3.
- Produces `FittedFoldModel`, model-specific fit functions, checkpoint APIs, model save/load, and auditable ensemble inference.

- [ ] **Step 1: Implement `FittedFoldModel` and common two-phase fit protocol**

```python
@dataclass
class FittedFoldModel:
    estimator: Any
    best_iteration: int
    fit_seconds: float
    model_name: str
    fold_name: str
    report_prediction: np.ndarray
```

Model-specific functions must build fresh estimators and never call `clone` on already fitted native boosters.

- [ ] **Step 2: Implement strict LightGBM classifier and positive regressor fits**

Inner fit uses early stopping. Refit uses the selected positive integer `best_iteration` as fixed `n_estimators`, no eval set, and full outer-train. Positive regressor filters `target_gmv_30d > 0` before both inner fit and refit.

- [ ] **Step 3: Implement CatBoost Plain fit and Ordered pilot support**

Set `allow_writing_files=False`, `thread_count<=10`, `random_seed=42`, `auto_class_weights=None`, native NaN, and no outer eval set. Parameter candidates come from sklearn `ParameterSampler`.

- [ ] **Step 4: Implement lazy EBM import and resource gate**

Import InterpretML inside `fit_ebm_classifier`, not at package import time. Return a structured `ResourceRejection` before fit when the exact 70% gate fails. Search uses explicit bags; refit sets `validation_size=0`, `outer_bags=1`, fixed rounds.

- [ ] **Step 5: Implement atomic JSON/Parquet and model round trips**

Write to a sibling temporary path, flush/close, then `Path.replace`. Store fold-specific weights/calibrator history and both pre-January/post-January configs.

- [ ] **Step 6: Implement ensemble inference**

Validate exact feature order; generate base probabilities; apply weights; calibrate; obtain frozen regressor positive log; call the shared soft-log combination; preserve expected user order.

- [ ] **Step 7: Run focused model/artifact/inference tests**

```powershell
& 'C:\Users\savin\AppData\Local\Programs\Python\Python314\python.exe' -m unittest tests.test_models tests.test_artifacts tests.test_inference -v
```

- [ ] **Step 8: Run all v2 tests and require GREEN except environment if InterpretML is still absent**

- [ ] **Step 9: Commit model and persistence APIs**

```powershell
git add -- notebooks/modeling/two_stage_v2/src notebooks/modeling/two_stage_v2/tests
git commit -m "feat: add ensemble training and inference"
```

---

### Task 5: Implement diagnostics and visualization APIs

**Files:**
- Create: `notebooks/modeling/two_stage_v2/src/diagnostics.py`
- Create: `notebooks/modeling/two_stage_v2/tests/test_diagnostics.py`

**Interfaces:**
- Consumes tidy OOF and metrics frames.
- Produces tidy diagnostic tables and Matplotlib `Figure` objects; each plot function optionally saves to a supplied path.

- [ ] **Step 1: Write failing tests for tidy residual and error-case tables**

Test exact columns, deterministic bin ordering, and that TP/FP/FN/TN squared-log contributions sum to total squared-log error.

- [ ] **Step 2: Write failing smoke tests for every required plot function**

Use the non-interactive `Agg` backend and assert each function returns a `matplotlib.figure.Figure` and writes a non-empty PNG when a path is supplied.

- [ ] **Step 3: Implement table builders**

Implement cutoff summary, missing summary, fold timeline, model/fold metrics, reliability, residual correlations, confusion contribution, probability bins, blend weights, simplex landscape, bootstrap interval, segment heatmap, and inference comparison.

- [ ] **Step 4: Implement plot functions with Russian titles/labels**

Do not call `plt.show()` inside library functions; notebook owns display.

- [ ] **Step 5: Run diagnostics tests and full suite**

- [ ] **Step 6: Commit diagnostics**

```powershell
git add -- notebooks/modeling/two_stage_v2/src/diagnostics.py notebooks/modeling/two_stage_v2/tests/test_diagnostics.py
git commit -m "feat: add ensemble diagnostics"
```

---

### Task 6: Build the narrative notebook and notebook contract

**Files:**
- Create: `notebooks/modeling/two_stage_v2/10_Two_Stage_Ensemble_v2.ipynb`
- Create: `notebooks/modeling/two_stage_v2/tests/test_notebook_contract.py`
- Modify: `notebooks/modeling/two_stage_v2/README.md`

**Interfaces:**
- Consumes only public `src` APIs from Tasks 3–5.
- Produces the 19-section notebook from design section 9 and every required visualization from section 10.

- [ ] **Step 1: Write notebook contract tests before creating the notebook**

Tests must assert exact section headings, neighboring `src` bootstrap, absence of repeated function definitions for training/calibration/blending, presence of checkpoint/config paths, and required plot call names.

- [ ] **Step 2: Verify contract tests fail because notebook is absent**

- [ ] **Step 3: Generate notebook as nbformat 4.5 with deterministic cell order**

The first code cell verifies working directory and imports neighboring `src`. The second prints package versions, CPU/RAM, config fingerprint, and data path. Each long-running cell delegates to checkpoint-aware `src` functions.

- [ ] **Step 4: Add Markdown interpretation after every result/figure cell**

Use concise Russian explanations: what was measured, whether it passes the gate, and what the next stage will use. Do not assert an improvement before the corresponding output exists.

- [ ] **Step 5: Add explicit RUN_MODE control**

`RUN_MODE="full"` is the committed default. `"smoke"` may reduce rows/trials only for development, must use a distinct fingerprint/artifact namespace, and cannot produce production artifacts.

- [ ] **Step 6: Run notebook contract and full unit suite**

- [ ] **Step 7: Commit notebook source**

```powershell
git add -- notebooks/modeling/two_stage_v2
git commit -m "feat: add two-stage ensemble v2 notebook"
```

---

### Task 7: Install the missing dependency and execute smoke integration

**Files:**
- Runtime outputs only: `data/two_stage_v2_artifacts/smoke/`
- Modify source/tests only to fix reproduced integration defects.

**Interfaces:**
- Consumes all committed code and notebook.
- Produces a clean environment test, green unit suite, smoke OOF/blend/inference artifacts, and an executed smoke notebook copy outside the production artifact namespace.

- [ ] **Step 1: Install only the missing pinned dependency into the Python 3.14 environment**

```powershell
& 'C:\Users\savin\AppData\Local\Programs\Python\Python314\python.exe' -m pip install interpret==0.7.8
```

- [ ] **Step 2: Run environment and full v2 tests**

- [ ] **Step 3: Execute smoke mode from a fresh kernel via nbclient**

Use an execution timeout of 90 minutes per cell and save the executed smoke notebook to `data/two_stage_v2_artifacts/smoke/10_Two_Stage_Ensemble_v2_smoke.ipynb`.

- [ ] **Step 4: Scan outputs**

Reject any output with `output_type="error"`, strings `Traceback`, non-finite metrics, missing figures, invalid checkpoints, or production paths used by smoke mode.

- [ ] **Step 5: For every defect, write a focused reproduction test, observe RED, fix, and rerun the affected integration stage**

- [ ] **Step 6: Commit only code/test corrections, not smoke runtime outputs**

---

### Task 8: Run full strict training, selection, confirmation, and production inference

**Files:**
- Runtime outputs: `data/two_stage_v2_artifacts/**`
- Executed notebook: `notebooks/modeling/two_stage_v2/10_Two_Stage_Ensemble_v2.ipynb`
- Modify code/tests only for reproduced defects.

**Interfaces:**
- Produces strict base OOF, model search records, blend history, pre-January confirmation, post-January production config, models, metrics, figures, and final 250,000-row submission.

- [ ] **Step 1: Record pre-run environment, free RAM, config fingerprint, and git commit**

- [ ] **Step 2: Execute the committed notebook in `RUN_MODE="full"` with nbclient**

Use checkpoints to tolerate process interruption. Communicate progress at model/fold boundaries, not trial spam.

- [ ] **Step 3: Verify strict LightGBM classifier/regressor OOF**

Require complete folds 1–3, no duplicate report rows, no future cutoff in fit metadata, and finite probabilities/predictions.

- [ ] **Step 4: Verify CatBoost and EBM decision records**

CatBoost must complete strict OOF. EBM must either complete strict OOF or have a structured resource/admission rejection satisfying the exact gate.

- [ ] **Step 5: Verify walk-forward blend history**

Require separate weights/calibrator for fold 2, fold 3, pre-January, and post-January production. Hash each input OOF pool.

- [ ] **Step 6: Verify acceptance and bootstrap**

Report strict baseline, candidate, delta, confidence interval, per-fold changes, logloss/Brier constraints, and admission decisions. Do not claim improvement if the configured gate fails.

- [ ] **Step 7: Verify final artifacts and submission**

Load every saved model/config in a new process, reproduce a fixed prediction sample, then validate exactly 250,000 ordered rows with finite non-negative `predict`.

- [ ] **Step 8: Scan the executed notebook for errors and non-monotonic execution counts**

- [ ] **Step 9: Commit the executed notebook and lightweight JSON manifests/metrics only if repository policy permits; do not commit large model/OOF/submission files unless already expected by the repository**

---

### Task 9: Independent code/data-science review, fix wave, and final verification

**Files:**
- Review the complete plan diff and runtime manifests.
- Modify only files implicated by accepted findings.

**Interfaces:**
- Produces a reviewer verdict covering correctness, temporal leakage, statistical claims, sklearn API quality, memory safety, persistence, notebook readability, and result interpretation.

- [ ] **Step 1: Package the complete diff, plan, design, test evidence, notebook execution evidence, and artifact manifests for an independent reviewer**

- [ ] **Step 2: Require findings by severity with exact file/line or notebook cell references**

- [ ] **Step 3: Dispatch one lightweight fix agent with the full accepted finding list**

Every bug fix must begin with a focused failing regression test where behavior is testable.

- [ ] **Step 4: Run one scoped re-review of the fix wave**

- [ ] **Step 5: Re-run affected unit/integration tests, then the full suite**

- [ ] **Step 6: Re-execute the notebook if any fix changes runtime behavior or serialized outputs**

- [ ] **Step 7: Run final completion gate**

Evidence must show: all tests green, zero notebook error outputs, valid model round trips, valid 250,000-row submission, documented EBM decision, and no unresolved Critical/Important findings.
