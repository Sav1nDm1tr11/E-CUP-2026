"""Small, deterministic adapters for the v2 model families.

The adapters deliberately keep the temporal protocol here instead of relying on
an estimator's random split.  Native boosters are always constructed twice:
once for inner early stopping and once, fresh, for the outer-train refit.
"""

from __future__ import annotations

import gc
import importlib
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .temporal_split import CutoffFold
from .metrics import rmsle


_RESERVED_COLUMNS = frozenset({
    "user_id", "cutoff_date", "target_nonzero", "target_gmv_30d",
    "prediction", "sample_weight", "fold", "group", "row_id",
})


class ModelFitRejection(RuntimeError):
    """Raised when an estimator cannot honor the temporal fit contract."""


@dataclass
class FittedFoldModel:
    estimator: Any
    best_iteration: int
    fit_seconds: float
    model_name: str
    fold_name: str
    report_prediction: np.ndarray
    peak_memory_bytes: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResourceRejection:
    """Structured result used when an optional model cannot fit safely."""

    reason: str
    estimated_model_memory: int
    measured_overhead: int
    available_memory: int
    gate_limit: int
    accepted: bool = False

    @property
    def total_estimated_memory(self) -> int:
        return self.estimated_model_memory + self.measured_overhead

    # Stable descriptive aliases for reports and notebook JSON serialization.
    @property
    def estimated_bytes(self) -> int:
        return self.estimated_model_memory

    @property
    def overhead_bytes(self) -> int:
        return self.measured_overhead

    @property
    def available_bytes(self) -> int:
        return self.available_memory

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "estimated_model_memory": self.estimated_model_memory,
            "measured_overhead": self.measured_overhead,
            "available_memory": self.available_memory,
            "gate_limit": self.gate_limit,
            "accepted": self.accepted,
        }


def _dates(frame: Any, groups: Sequence[Any] | None = None) -> pd.DatetimeIndex:
    if groups is not None:
        dates = pd.DatetimeIndex(pd.to_datetime(np.asarray(groups)))
        if len(dates) != len(frame):
            raise ValueError("groups must have one cutoff date per row")
        return dates
    if isinstance(frame, pd.DataFrame) and "cutoff_date" in frame.columns:
        return pd.DatetimeIndex(pd.to_datetime(frame["cutoff_date"]))
    index = getattr(frame, "index", None)
    if isinstance(index, pd.MultiIndex) and "cutoff_date" in index.names:
        return pd.DatetimeIndex(pd.to_datetime(index.get_level_values("cutoff_date")))
    if isinstance(index, pd.DatetimeIndex):
        return index
    raise ValueError("X must have a cutoff_date column or DatetimeIndex")


def _feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove IDs/temporal labels before passing a matrix to an estimator."""
    columns = [column for column in frame.columns if column not in _RESERVED_COLUMNS]
    return frame.loc[:, columns]


def _mask(frame: Any, date_values: Sequence[pd.Timestamp]) -> np.ndarray:
    dates = _dates(frame)
    wanted = pd.DatetimeIndex(pd.to_datetime(list(date_values)))
    return np.asarray(dates.isin(wanted), dtype=bool)


def _slice(value: Any, mask: np.ndarray) -> Any:
    if isinstance(value, (pd.Series, pd.DataFrame)):
        return value.loc[mask]
    return np.asarray(value)[mask]


def _factory_default(kind: str) -> Callable[..., Any]:
    if kind.startswith("lgbm"):
        module = importlib.import_module("lightgbm")
        return module.LGBMRegressor if kind.endswith("regressor") else module.LGBMClassifier
    module = importlib.import_module("catboost")
    return module.CatBoostClassifier


def _params(config: Any, name: str) -> dict[str, Any]:
    if config is None:
        return {}
    value = getattr(config, name, None)
    if value is None and isinstance(config, Mapping):
        value = config.get(name)
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, (tuple, list)):
        return dict(value)
    return {}


def _positive_iteration(estimator: Any, fallback: int) -> int:
    for name in ("best_iteration_", "best_iteration", "tree_count_", "get_best_iteration"):
        value = getattr(estimator, name, None)
        if callable(value):
            try:
                value = value()
            except TypeError:
                value = None
        if value is not None:
            try:
                result = int(value)
            except (TypeError, ValueError):
                continue
            if result > 0:
                # CatBoost's get_best_iteration is zero based.
                return result + 1 if name == "get_best_iteration" else result
    return max(1, int(fallback))


def _positive_proba(estimator: Any, X: Any) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        values = np.asarray(estimator.predict_proba(X), dtype=float)
        if values.ndim == 2:
            values = values[:, -1]
    else:
        values = np.asarray(estimator.predict(X), dtype=float).reshape(-1)
    values = values.reshape(-1)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("estimator returned probabilities outside [0, 1]")
    return values


def _fit_with_optional_callbacks(estimator: Any, X: Any, y: Any, *, eval_set=None, callbacks=None, **kwargs):
    if eval_set is None:
        return estimator.fit(X, y, **kwargs)
    fit_kwargs = dict(kwargs, eval_set=eval_set)
    if callbacks:
        fit_kwargs["callbacks"] = callbacks
    try:
        return estimator.fit(X, y, **fit_kwargs)
    except TypeError as callback_error:
        if "callbacks" not in fit_kwargs:
            raise ModelFitRejection("LightGBM estimator rejected required eval_set") from callback_error
        # Compatibility retry is allowed only for the callbacks argument;
        # eval_set is never removed because that would leak report labels.
        fit_kwargs.pop("callbacks", None)
        try:
            return estimator.fit(X, y, **fit_kwargs)
        except TypeError as eval_error:
            raise ModelFitRejection(
                "LightGBM estimator cannot fit with required inner eval_set"
            ) from eval_error


def _lgbm_callbacks(rounds: int):
    try:
        lgb = importlib.import_module("lightgbm")
        return [lgb.early_stopping(rounds, verbose=False)]
    except (ImportError, AttributeError):
        return None


def _two_phase_lgbm(
    X: pd.DataFrame,
    y: Sequence,
    *,
    fold: CutoffFold,
    estimator_factory: Callable[..., Any] | None,
    params: Mapping[str, Any] | None,
    model_name: str,
    default_n_estimators: int,
    early_stopping_rounds: int,
    groups: Sequence[Any] | None = None,
    transform_y: Callable[[Any], np.ndarray] | None = None,
    positive_mask: np.ndarray | None = None,
) -> FittedFoldModel:
    if not isinstance(X, pd.DataFrame) or len(X) != len(y):
        raise ValueError("X and y must be equally sized")
    dates = _dates(X, groups)
    model_X = _feature_frame(X)
    outer_train = np.asarray(dates.isin(pd.DatetimeIndex(fold.train_dates)), dtype=bool)
    inner_valid = np.asarray(dates == pd.Timestamp(fold.inner_valid_date), dtype=bool)
    inner_train = outer_train & ~inner_valid
    outer_report = np.asarray(dates == pd.Timestamp(fold.outer_valid_date), dtype=bool)
    if not inner_train.any() or not inner_valid.any() or not outer_train.any():
        raise ValueError(f"Fold {fold.name} has no inner train, inner validation, or outer train rows")
    y_values = np.asarray(y)
    if positive_mask is not None:
        positive_mask = np.asarray(positive_mask, dtype=bool)
        if len(positive_mask) != len(X):
            raise ValueError("positive_mask must match X")
        inner_train &= positive_mask
        inner_valid &= positive_mask
        outer_train &= positive_mask
        if not inner_train.any() or not inner_valid.any() or not outer_train.any():
            raise ValueError(
                f"Fold {fold.name} has no positive rows in inner train, inner validation, or outer train"
            )
    y_values = transform_y(y_values) if transform_y is not None else y_values
    factory = estimator_factory or _factory_default(model_name)
    base = dict(params or {})
    if "n_jobs" in base:
        base["n_jobs"] = min(10, int(base["n_jobs"]))
    base.setdefault("n_estimators", max(1, int(default_n_estimators)))
    inner_estimator = factory(**base)
    started = time.perf_counter()
    callbacks = _lgbm_callbacks(early_stopping_rounds)
    _fit_with_optional_callbacks(
        inner_estimator,
        _slice(model_X, inner_train),
        _slice(y_values, inner_train),
        eval_set=[(_slice(model_X, inner_valid), _slice(y_values, inner_valid))],
        callbacks=callbacks,
    )
    best_iteration = _positive_iteration(inner_estimator, base["n_estimators"])
    refit_params = dict(base)
    refit_params["n_estimators"] = best_iteration
    refit_estimator = factory(**refit_params)
    # Deliberately no eval_set: outer report labels are never used by fit or
    # iteration selection.
    refit_estimator.fit(_slice(model_X, outer_train), _slice(y_values, outer_train))
    if not outer_report.any():
        report_prediction = np.empty(0, dtype=float)
    elif model_name.endswith("classifier"):
        report_prediction = _positive_proba(refit_estimator, _slice(model_X, outer_report))
    else:
        report_prediction = np.asarray(refit_estimator.predict(_slice(model_X, outer_report)), dtype=float).reshape(-1)
    elapsed = time.perf_counter() - started
    del inner_estimator
    gc.collect()
    return FittedFoldModel(refit_estimator, best_iteration, float(elapsed), model_name, fold.name,
                           report_prediction, metadata={"inner_valid_date": str(fold.inner_valid_date)})


def fit_lgbm_classifier(
    X: pd.DataFrame,
    y: Sequence,
    fold: CutoffFold,
    *,
    estimator_factory: Callable[..., Any] | None = None,
    params: Mapping[str, Any] | None = None,
    config: Any = None,
    early_stopping_rounds: int = 150,
    max_estimators: int | None = None,
    groups: Sequence[Any] | None = None,
) -> FittedFoldModel:
    chosen = dict(params or {})
    if not chosen:
        chosen = _params(config, "frozen_classifier_params")
    default = max_estimators or int(getattr(config, "lgbm_classifier_max_estimators", 5000))
    return _two_phase_lgbm(X, y, fold=fold, estimator_factory=estimator_factory,
                           params=chosen, model_name="lgbm_classifier",
                           default_n_estimators=default, early_stopping_rounds=early_stopping_rounds,
                           groups=groups)


def fit_positive_lgbm_regressor(
    X: pd.DataFrame,
    target_gmv_30d: Sequence,
    fold: CutoffFold,
    *,
    estimator_factory: Callable[..., Any] | None = None,
    params: Mapping[str, Any] | None = None,
    config: Any = None,
    early_stopping_rounds: int = 150,
    max_estimators: int | None = None,
    groups: Sequence[Any] | None = None,
) -> FittedFoldModel:
    target = np.asarray(target_gmv_30d, dtype=float)
    if len(target) != len(X) or not np.isfinite(target).all() or (target < 0).any():
        raise ValueError("target_gmv_30d must be finite, non-negative, and match X")
    chosen = dict(params or {}) or _params(config, "frozen_regressor_params")
    default = max_estimators or int(getattr(config, "lgbm_regressor_max_estimators", 5000))
    return _two_phase_lgbm(X, target, fold=fold, estimator_factory=estimator_factory,
                           params=chosen, model_name="lgbm_regressor",
                           default_n_estimators=default, early_stopping_rounds=early_stopping_rounds,
                           groups=groups,
                           transform_y=lambda values: np.log1p(np.asarray(values, dtype=float)),
                           positive_mask=target > 0)


def fit_catboost_classifier(
    X: pd.DataFrame,
    y: Sequence,
    fold: CutoffFold,
    *,
    estimator_factory: Callable[..., Any] | None = None,
    param_distributions: Mapping[str, Any] | None = None,
    params: Mapping[str, Any] | None = None,
    config: Any = None,
    trials: int | None = None,
    ordered: bool = False,
    random_seed: int = 42,
    groups: Sequence[Any] | None = None,
    inner_positive_log: Sequence[float] | None = None,
    actual_gmv: Sequence[float] | None = None,
    actual_gmv_30d: Sequence[float] | None = None,
    governance_hook: Callable[[Mapping[str, Any]], ResourceRejection | None] | None = None,
    plain_metrics: Mapping[str, float] | None = None,
    minimum_ordered_improvement: float = 0.0005,
) -> FittedFoldModel | ResourceRejection:
    """Search CatBoost candidates sequentially, then perform a fresh refit."""
    from sklearn.model_selection import ParameterSampler
    from sklearn.metrics import log_loss

    dates = _dates(X, groups)
    model_X = _feature_frame(X)
    outer_train = np.asarray(dates.isin(fold.train_dates), dtype=bool)
    inner_valid = np.asarray(dates == fold.inner_valid_date, dtype=bool)
    inner_train = outer_train & ~inner_valid
    outer_report = np.asarray(dates == fold.outer_valid_date, dtype=bool)
    factory = estimator_factory or _factory_default("catboost_classifier")
    base = dict(params or {})
    if not base:
        base.update(_params(config, "catboost_params"))
    base.update({"allow_writing_files": False, "thread_count": min(10, int(base.get("thread_count", 10))),
                 "random_seed": random_seed, "auto_class_weights": None,
                 "boosting_type": "Ordered" if ordered else "Plain", "grow_policy": "SymmetricTree",
                 "nan_mode": "Min"})
    base.setdefault("iterations", int(getattr(config, "catboost_max_iterations", 5000)))
    # Keep candidate generation behind ParameterSampler even for the small
    # default search; this makes trial ordering and the random seed auditable.
    candidate_space = dict(param_distributions or {
        "depth": [6, 7, 8, 9, 10], "learning_rate": [0.02, 0.03, 0.04, 0.05],
    })
    requested_trials = int(trials if trials is not None else getattr(config, "catboost_trials", 20))
    n_trials = max(1, min(requested_trials, 3) if ordered else requested_trials)
    candidates = list(ParameterSampler(candidate_space, n_iter=n_trials, random_state=random_seed))
    if not candidates:
        raise ValueError("ParameterSampler produced no CatBoost candidates")
    base.setdefault("loss_function", "Logloss")
    base.setdefault("eval_metric", "Logloss")
    base.setdefault("bootstrap_type", "Bayesian")
    base.setdefault("depth", 6)
    base.setdefault("learning_rate", 0.05)
    best_score = float("inf")
    best_candidate: dict[str, Any] = {}
    best_iteration = int(base["iterations"])
    objective_actual = actual_gmv_30d if actual_gmv_30d is not None else actual_gmv
    positive = None if inner_positive_log is None else np.asarray(inner_positive_log, dtype=float)
    actual = None if objective_actual is None else np.asarray(objective_actual, dtype=float)
    if positive is None or actual is None:
        raise ValueError("CatBoost requires inner_positive_log and actual_gmv for end-to-end objective")
    if positive is not None and not (
        len(positive) == len(X) == len(actual)
        or len(positive) == int(inner_valid.sum()) == len(actual)
    ):
        raise ValueError("inner objective arrays must match X or inner validation rows")
    process = None
    try:
        import psutil
        process = psutil.Process()
        peak_memory = int(process.memory_info().rss)
    except ImportError:
        peak_memory = 0
        if ordered:
            return ResourceRejection("ordered_memory_measurement_unavailable", 0, 0, 0, 0)
    started = time.perf_counter()
    for candidate in candidates:
        trial_params = dict(base, **candidate)
        trial = factory(**trial_params)
        fit_kwargs = {"eval_set": [(_slice(model_X, inner_valid), _slice(y, inner_valid))],
                      "early_stopping_rounds": int(getattr(config, "catboost_early_stopping_rounds", 150))}
        try:
            trial.fit(_slice(model_X, inner_train), _slice(y, inner_train), **fit_kwargs)
        except TypeError:
            # Compatibility retry may remove only early_stopping_rounds;
            # inner eval_set remains mandatory.
            trial.fit(_slice(model_X, inner_train), _slice(y, inner_train), eval_set=fit_kwargs["eval_set"])
        valid_probability = _positive_proba(trial, _slice(model_X, inner_valid))
        if process is not None:
            peak_memory = max(peak_memory, int(process.memory_info().rss))
        if len(positive) == len(X):
            valid_actual = actual[inner_valid]
            valid_log = positive[inner_valid]
        else:
            valid_actual = actual
            valid_log = positive
        prediction = np.expm1(valid_probability * np.maximum(valid_log, 0.0))
        score = float(rmsle(valid_actual, prediction))
        objective_name = "inner_end_to_end_rmsle"
        if score < best_score:
            best_score = score
            best_candidate = dict(candidate)
            best_iteration = _positive_iteration(trial, trial_params["iterations"])
        del trial
    governance = {
        "ordered": bool(ordered), "selected_score": best_score,
        "objective": objective_name, "trials": len(candidates),
        "fit_seconds": float(time.perf_counter() - started),
        "peak_memory_bytes": peak_memory,
    }
    if ordered and plain_metrics is None:
        return ResourceRejection("ordered_governance_metrics_unavailable", 0, 0, 0, 0)
    if plain_metrics:
        governance.update({f"plain_{key}": value for key, value in plain_metrics.items()})
        if ordered and "rmsle" in plain_metrics:
            improvement = float(plain_metrics["rmsle"]) - best_score
            governance["improvement"] = improvement
            plain_seconds = float(plain_metrics.get("fit_seconds", np.inf))
            plain_memory = float(plain_metrics.get("memory_bytes", plain_metrics.get("plain_memory_bytes", np.inf)))
            current_memory = float(governance.get("memory_bytes", governance.get("peak_memory_bytes", 0)))
            if improvement < minimum_ordered_improvement and (
                governance["fit_seconds"] > 3 * plain_seconds
                or current_memory > 3 * plain_memory
            ):
                return ResourceRejection("ordered_governance_rejected", 0, 0, 0, 0)
    if governance_hook is not None:
        rejection = governance_hook(governance)
        if rejection is not None:
            return rejection
    refit_params = dict(base, **best_candidate, iterations=best_iteration, use_best_model=False)
    refit = factory(**refit_params)
    refit.fit(_slice(model_X, outer_train), _slice(y, outer_train))
    report = _positive_proba(refit, _slice(model_X, outer_report)) if outer_report.any() else np.empty(0, dtype=float)
    return FittedFoldModel(refit, best_iteration, float(time.perf_counter() - started),
                           "catboost_classifier_ordered" if ordered else "catboost_classifier",
                           fold.name, report, metadata={"trials": len(candidates), "selection_score": best_score,
                                                         "objective": objective_name,
                                                         "selected_params": best_candidate,
                                                         "governance": governance})


def estimate_ebm_memory(data: Any, *, data_multiplier: float = 1.0) -> int:
    """Conservative byte estimate used by the pre-fit EBM gate."""
    if hasattr(data, "memory_usage"):
        usage = data.memory_usage(index=True, deep=True)
        return int(np.ceil(float(np.asarray(usage).sum()) * float(data_multiplier) * 4.0))
    return int(np.ceil(np.asarray(data).nbytes * float(data_multiplier) * 4.0))


def _measure_ebm_toy_overhead(
    estimator_factory: Callable[..., Any],
    params: Mapping[str, Any],
    X: pd.DataFrame,
    y: Sequence,
    bags: np.ndarray,
) -> int | None:
    """Measure peak RSS increase for a same-settings, bounded toy fit."""
    try:
        import psutil
    except ImportError:
        return None
    process = psutil.Process()
    rows = min(len(X), 256)
    toy_X, toy_y, toy_bags = X.iloc[:rows], np.asarray(y)[:rows], bags[:rows]
    try:
        toy = estimator_factory(**dict(params))
    except Exception:
        return None
    baseline = int(process.memory_info().rss)
    peak = baseline
    stop = threading.Event()

    def sample() -> None:
        nonlocal peak
        while not stop.wait(0.01):
            try:
                peak = max(peak, int(process.memory_info().rss))
            except Exception:
                return

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        toy.fit(toy_X, toy_y, bags=toy_bags)
    except Exception:
        stop.set()
        sampler.join(timeout=1)
        return None
    stop.set()
    sampler.join(timeout=1)
    try:
        peak = max(peak, int(process.memory_info().rss))
    except Exception:
        return None
    overhead = peak - baseline
    return int(overhead) if overhead >= 0 else None


def fit_ebm_classifier(
    X: pd.DataFrame,
    y: Sequence,
    fold: CutoffFold,
    *,
    params: Mapping[str, Any] | None = None,
    config: Any = None,
    estimator_factory: Callable[..., Any] | None = None,
    available_memory: int | None = None,
    memory_estimator: Callable[..., int] | None = None,
    measured_overhead: int | None = None,
    overhead_estimator: Callable[..., int] | None = None,
    max_rounds: int = 20_000,
    groups: Sequence[Any] | None = None,
    inner_bags: int | None = None,
    n_jobs: int | None = None,
    early_stopping_rounds: int | None = None,
    max_configurations: int = 12,
    pilot_wall_clock_seconds: int = 5400,
    candidate_configs: Sequence[Mapping[str, Any]] | None = None,
) -> FittedFoldModel | ResourceRejection:
    """Fit InterpretML EBM only after the exact 70% RAM gate passes.

    InterpretML is imported here (and nowhere at module import time), which
    keeps the core package usable in environments without the optional wheel.
    """
    available = int(available_memory) if available_memory is not None else None
    if available is None:
        try:
            import psutil
            available = int(psutil.virtual_memory().available)
        except ImportError:
            available = 0
    fraction = float(getattr(config, "resource_gate_fraction", 0.70))
    if candidate_configs is not None:
        candidates = list(candidate_configs)[:min(12, max(1, int(max_configurations)))]
        if not candidates:
            return ResourceRejection("ebm_no_candidates", 0, 0, available, int(available * fraction))
        if len(candidates) > 1:
            started_search = time.perf_counter()
            last_rejection: ResourceRejection | None = None
            for candidate in candidates:
                if time.perf_counter() - started_search > pilot_wall_clock_seconds:
                    break
                outcome = fit_ebm_classifier(
                    X, y, fold, params=candidate, config=config,
                    estimator_factory=estimator_factory, available_memory=available,
                    memory_estimator=memory_estimator, measured_overhead=measured_overhead,
                    overhead_estimator=overhead_estimator, max_rounds=max_rounds, groups=groups,
                    inner_bags=inner_bags, n_jobs=n_jobs, early_stopping_rounds=early_stopping_rounds,
                    max_configurations=1, pilot_wall_clock_seconds=pilot_wall_clock_seconds,
                    candidate_configs=None,
                )
                if isinstance(outcome, FittedFoldModel):
                    outcome.metadata["candidate_count"] = len(candidates)
                    outcome.metadata["candidate_configs_capped"] = len(candidates) <= 12
                    outcome.metadata["pilot_wall_clock_seconds"] = pilot_wall_clock_seconds
                    return outcome
                last_rejection = outcome
            return last_rejection or ResourceRejection("ebm_pilot_timeout", 0, 0, available, int(available * fraction))
    estimate_fn = memory_estimator or estimate_ebm_memory
    estimated = int(estimate_fn(X, data_multiplier=1))
    # Lazy optional dependency boundary.
    if estimator_factory is None:
        module = importlib.import_module("interpret.glassbox")
        estimator_factory = module.ExplainableBoostingClassifier
    dates = _dates(X, groups)
    model_X = _feature_frame(X)
    outer_train = np.asarray(dates.isin(fold.train_dates), dtype=bool)
    inner_valid = np.asarray(dates == fold.inner_valid_date, dtype=bool)
    inner_train = outer_train & ~inner_valid
    base = dict(params or {})
    base.setdefault("inner_bags", inner_bags if inner_bags is not None else int(getattr(config, "ebm_inner_bags", 0)))
    base.setdefault("n_jobs", min(10, int(n_jobs if n_jobs is not None else getattr(config, "estimator_threads", 10))))
    base.setdefault("max_rounds", max_rounds)
    base.setdefault("outer_bags", 8)
    base.setdefault("validation_size", 0)
    base.setdefault("early_stopping_rounds", early_stopping_rounds if early_stopping_rounds is not None else 100)
    base.setdefault("random_state", 42)
    if base["n_jobs"] > 10:
        base["n_jobs"] = 10
    started = time.perf_counter()
    probe = estimator_factory(**base)
    # InterpretML accepts integer bags: +1 train and -1 validation. This is
    # explicit and avoids its random temporal split.
    bags = np.where(inner_train, 1, np.where(inner_valid, -1, 0)).astype(np.int8)
    bags_outer = bags[outer_train]
    overhead_fn = overhead_estimator
    if measured_overhead is not None:
        overhead = int(measured_overhead)
    elif overhead_fn is not None:
        try:
            overhead = int(overhead_fn(model_X, bags=bags_outer, n_jobs=base["n_jobs"]))
        except TypeError:
            overhead = int(overhead_fn(model_X, bags_outer, base["n_jobs"]))
    else:
        overhead = _measure_ebm_toy_overhead(estimator_factory, base, model_X.iloc[outer_train],
                                              _slice(y, outer_train), bags_outer)
        if overhead is None:
            return ResourceRejection("ebm_overhead_measurement_unavailable", estimated, 0, available,
                                     int(available * fraction))
    if overhead < 0:
        return ResourceRejection("ebm_overhead_measurement_invalid", estimated, overhead, available,
                                 int(available * fraction))
    native_estimator = getattr(probe, "estimate_mem", None)
    if native_estimator is not None and memory_estimator is None:
        try:
            estimated = int(native_estimator(model_X.iloc[outer_train], _slice(y, outer_train), data_multiplier=1))
        except TypeError as exc:
            return ResourceRejection("ebm_estimate_mem_signature_rejected", estimated, overhead, available,
                                     int(available * fraction))
    limit = int(available * fraction)
    if estimated + overhead > limit:
        return ResourceRejection("memory_gate_exceeded", estimated, overhead, available, limit)
    inner = probe
    try:
        inner.fit(_slice(model_X, outer_train), _slice(y, outer_train), bags=bags_outer)
    except TypeError as exc:
        return ResourceRejection("ebm_temporal_bags_rejected", estimated, overhead, available, limit)
    raw_rounds = getattr(inner, "best_iteration_", getattr(inner, "max_rounds", max_rounds))
    rounds_array = np.asarray(raw_rounds)
    positive_rounds = rounds_array[np.isfinite(rounds_array) & (rounds_array > 0)] if rounds_array.ndim else rounds_array
    chosen_rounds = int(np.max(positive_rounds)) if np.size(positive_rounds) else int(max_rounds)
    chosen_rounds = max(1, chosen_rounds)
    refit_params = dict(base, max_rounds=chosen_rounds, validation_size=0, outer_bags=1,
                        early_stopping_rounds=0)
    refit = estimator_factory(**refit_params)
    refit.fit(_slice(model_X, outer_train), _slice(y, outer_train))
    report = _positive_proba(refit, _slice(model_X, dates == fold.outer_valid_date))
    return FittedFoldModel(refit, chosen_rounds, float(time.perf_counter() - started), "ebm_classifier", fold.name, report,
                           metadata={"estimated_model_memory": estimated, "measured_overhead": overhead,
                                     "available_memory": available, "resource_gate_fraction": fraction,
                                     "inner_bags": base["inner_bags"], "n_jobs": base["n_jobs"],
                                     "max_configurations": min(12, max_configurations),
                                     "pilot_wall_clock_seconds": pilot_wall_clock_seconds,
                                     "temporal_bags": True,
                                     "selected_rounds_aggregation": "max_positive_stage_bag"})
