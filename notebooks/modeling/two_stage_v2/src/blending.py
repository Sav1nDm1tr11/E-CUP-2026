"""Past-only convex probability blending and soft-log predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.base import BaseEstimator

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
        weights = getattr(self, "weights_", np.asarray(self.weights, dtype=float))
        values = np.asarray(probabilities, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(weights):
            raise ValueError("probabilities shape does not match weights")
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


def fit_walk_forward_blend(past_oof, positive_log, actual_gmv, config) -> BlendState:
    if not isinstance(past_oof, pd.DataFrame) or past_oof.empty:
        raise ValueError("past_oof must be a non-empty DataFrame")
    model_columns = [c for c in ("p_lgbm", "p_catboost", "p_ebm") if c in past_oof.columns]
    if not model_columns:
        raise ValueError("past_oof must contain at least one base probability column")
    positive = np.asarray(positive_log, dtype=float).reshape(-1)
    actual = np.asarray(actual_gmv, dtype=float).reshape(-1)
    if len(positive) != len(past_oof) or len(actual) != len(past_oof):
        raise ValueError("OOF and target lengths must match")
    if not np.isfinite(positive).all() or not np.isfinite(actual).all() or (actual < 0).any():
        raise ValueError("OOF values and actual_gmv must be finite; actual_gmv non-negative")
    if "cutoff_date" in past_oof:
        cutoff = pd.to_datetime(past_oof["cutoff_date"])
        trained_through = pd.Timestamp(_config_value(config, "trained_through", cutoff.max()))
        if (cutoff > trained_through).any():
            raise ValueError("past_oof contains rows later than trained_through")
    else:
        trained_through = pd.Timestamp(_config_value(config, "trained_through", pd.Timestamp.max))
    target = (actual > 0).astype(np.int8)
    if "target_nonzero" in past_oof:
        target = np.asarray(past_oof["target_nonzero"], dtype=np.int8)
    probabilities = past_oof[model_columns].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all():
        raise ValueError("OOF probabilities must be finite")
    step = float(_config_value(config, "simplex_step", 0.05))
    best: tuple[float, np.ndarray, SigmoidCalibrator] | None = None
    for weights in simplex_grid(len(model_columns), step):
        raw = np.clip(probabilities @ weights, 0.0, 1.0)
        calibrator = SigmoidCalibrator().fit(raw, target)
        calibrated = calibrator.predict_proba(raw)[:, 1]
        objective = rmsle(actual, combine_predictions(calibrated, positive).prediction)
        if best is None or objective < best[0]:
            best = (objective, weights.copy(), calibrator)
    assert best is not None
    objective, weights, calibrator = best
    return BlendState(tuple(float(v) for v in weights), float(calibrator.a), float(calibrator.b),
                      float(calibrator.epsilon), float(objective), trained_through)
