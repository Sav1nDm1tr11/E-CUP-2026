"""Past-only convex probability blending and soft-log predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.utils.validation import check_is_fitted

from .calibration import SigmoidCalibrator
from .metrics import rmsle


@dataclass(frozen=True)
class BlendState:
    weights: tuple[float, ...]
    calibrator_a: float
    calibrator_b: float
    epsilon: float
    objective: float
    trained_through: pd.Timestamp
    diagnostics: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": list(self.weights),
            "calibrator_a": self.calibrator_a,
            "calibrator_b": self.calibrator_b,
            "epsilon": self.epsilon,
            "objective": self.objective,
            "trained_through": pd.Timestamp(self.trained_through).isoformat(),
            "diagnostics": [dict(item) for item in self.diagnostics],
        }


@dataclass(frozen=True)
class BlendPrediction:
    probability: np.ndarray
    prediction_log: np.ndarray
    prediction: np.ndarray


def simplex_grid(model_count: int = 2, step: float = 0.05) -> np.ndarray:
    if model_count < 1 or not 0 < step <= 1:
        raise ValueError("model_count must be positive and step must be in (0, 1]")
    units = max(1, int(round(1.0 / step)))
    if not np.isclose(units * step, 1.0, atol=1e-8):
        raise ValueError("step must divide one exactly")
    rows: list[list[float]] = []

    def compositions(total: int, count: int, prefix: list[int]):
        if count == 1:
            rows.append([*prefix, total])
            return
        for value in range(total + 1):
            compositions(total - value, count - 1, [*prefix, value])

    compositions(units, model_count, [])
    return np.asarray(rows, dtype=float) / units


class ConvexProbabilityBlender(BaseEstimator):
    def __init__(self, weights=None):
        self.weights = weights

    def fit(self, X, y=None):
        values = np.asarray(X if self.weights is None else self.weights, dtype=float)
        if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
            raise ValueError("weights must be a finite one-dimensional vector")
        if (values < 0).any() or not np.isclose(values.sum(), 1.0):
            raise ValueError("weights must be non-negative and sum to one")
        self.weights_ = values.copy()
        return self

    def transform(self, probabilities):
        check_is_fitted(self, "weights_")
        weights = self.weights_
        values = np.asarray(probabilities, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(weights):
            raise ValueError("probabilities shape does not match weights")
        if not np.isfinite(values).all():
            raise ValueError("probabilities must be finite")
        return values @ weights

    def predict_proba(self, probabilities):
        blended = np.clip(self.transform(probabilities), 0.0, 1.0)
        return np.column_stack([1.0 - blended, blended])


def combine_predictions(probability, positive_log) -> BlendPrediction:
    p = np.asarray(probability, dtype=float).reshape(-1)
    positive = np.asarray(positive_log, dtype=float).reshape(-1)
    if p.shape != positive.shape or not np.isfinite(p).all() or not np.isfinite(positive).all():
        raise ValueError("probability and positive_log must be finite and equally sized")
    p = np.clip(p, 0.0, 1.0)
    prediction_log = p * np.maximum(positive, 0.0)
    return BlendPrediction(p, prediction_log, np.expm1(prediction_log))


def _config_value(config: Any, key: str, default: Any = None):
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def fit_walk_forward_blend(past_oof, positive_log, actual_gmv, config, *, trained_through=None) -> BlendState:
    if not isinstance(past_oof, pd.DataFrame) or past_oof.empty:
        raise ValueError("past_oof must be a non-empty DataFrame")
    required_probability_columns = ("p_lgbm", "p_catboost")
    if not all(column in past_oof.columns for column in required_probability_columns):
        raise ValueError("past_oof must contain p_lgbm and p_catboost")
    model_columns = [c for c in (*required_probability_columns, "p_ebm") if c in past_oof.columns]
    positive = np.asarray(positive_log, dtype=float).reshape(-1)
    actual = np.asarray(actual_gmv, dtype=float).reshape(-1)
    if len(positive) != len(past_oof) or len(actual) != len(past_oof):
        raise ValueError("OOF and target lengths must match")
    if not np.isfinite(positive).all() or not np.isfinite(actual).all() or (actual < 0).any():
        raise ValueError("OOF values and actual_gmv must be finite; actual_gmv non-negative")
    if "cutoff_date" not in past_oof.columns:
        raise ValueError("past_oof must contain cutoff_date")
    cutoff = pd.to_datetime(past_oof["cutoff_date"], errors="coerce")
    if cutoff.isna().any():
        raise ValueError("past_oof cutoff_date must be non-null and valid")
    explicit_trained_through = trained_through
    if explicit_trained_through is None:
        explicit_trained_through = _config_value(config, "trained_through", None)
    if explicit_trained_through is None:
        raise ValueError("trusted trained_through must be supplied explicitly")
    trained_through = pd.Timestamp(explicit_trained_through)
    if pd.isna(trained_through) or (cutoff > trained_through).any():
        raise ValueError("past_oof contains rows later than trusted trained_through")
    target = (actual > 0).astype(np.int8)
    if "target_nonzero" in past_oof:
        target = np.asarray(past_oof["target_nonzero"])
    from .validation import validate_target_contract
    validate_target_contract(actual, target)
    probabilities = past_oof[model_columns].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all():
        raise ValueError("OOF probabilities must be finite")
    step = float(_config_value(config, "simplex_step", 0.05))
    max_logloss_degradation = float(_config_value(config, "max_logloss_degradation", 0.0010))
    max_brier_degradation = float(_config_value(config, "max_brier_degradation", 0.0005))
    cutoff_values = tuple(pd.DatetimeIndex(cutoff).sort_values().unique())
    diagnostics: list[dict[str, Any]] = []
    best: tuple[float, np.ndarray, SigmoidCalibrator] | None = None
    for weights in simplex_grid(len(model_columns), step):
        raw = np.clip(probabilities @ weights, 0.0, 1.0)
        calibrator = SigmoidCalibrator().fit(raw, target)
        calibrated = calibrator.predict_proba(raw)[:, 1]
        prediction = combine_predictions(calibrated, positive).prediction
        fold_metrics: list[dict[str, Any]] = []
        for cutoff_value in cutoff_values:
            mask = cutoff.to_numpy() == cutoff_value
            current_probability = np.clip(probabilities[mask, 0], 0.0, 1.0)
            candidate_probability = calibrated[mask]
            current_logloss = float(log_loss(target[mask], current_probability, labels=[0, 1]))
            candidate_logloss = float(log_loss(target[mask], candidate_probability, labels=[0, 1]))
            current_brier = float(brier_score_loss(target[mask], current_probability))
            candidate_brier = float(brier_score_loss(target[mask], candidate_probability))
            fold_metrics.append({
                "cutoff_date": pd.Timestamp(cutoff_value).isoformat(),
                "rmsle": rmsle(actual[mask], prediction[mask]),
                "logloss_current": current_logloss, "logloss_candidate": candidate_logloss,
                "brier_current": current_brier, "brier_candidate": candidate_brier,
                "accepted": bool(
                    candidate_logloss <= current_logloss + max_logloss_degradation
                    and candidate_brier <= current_brier + max_brier_degradation
                ),
            })
        fold_rmsle = np.asarray([item["rmsle"] for item in fold_metrics], dtype=float)
        stability = fold_rmsle.std(ddof=1) if len(fold_rmsle) > 1 else 0.0
        objective = float(fold_rmsle.mean() + 0.25 * stability)
        accepted = all(item["accepted"] for item in fold_metrics)
        diagnostics.append({
            "weights": [float(value) for value in weights], "objective": objective,
            "fold_metrics": fold_metrics,
            "accepted": bool(accepted),
        })
        if not accepted:
            continue
        if best is None or objective < best[0]:
            best = (objective, weights.copy(), calibrator)
    if best is None:
        raise ValueError("No blend candidate satisfied logloss/Brier degradation constraints")
    objective, weights, calibrator = best
    return BlendState(tuple(float(v) for v in weights), float(calibrator.a), float(calibrator.b),
                      float(calibrator.epsilon), float(objective), trained_through,
                      tuple(diagnostics))
