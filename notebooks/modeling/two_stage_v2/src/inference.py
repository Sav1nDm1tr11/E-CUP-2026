"""Auditable, order-preserving inference for the two-stage ensemble."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .blending import combine_predictions


@dataclass(frozen=True)
class InferenceResult:
    user_id: np.ndarray
    raw_base_probabilities: np.ndarray
    blended_probability: np.ndarray
    calibrated_probability: np.ndarray
    positive_log: np.ndarray
    prediction_log: np.ndarray
    prediction: np.ndarray

    @property
    def probability(self) -> np.ndarray:
        return self.calibrated_probability

    def to_frame(self) -> pd.DataFrame:
        output: dict[str, Any] = {"user_id": self.user_id}
        for index in range(self.raw_base_probabilities.shape[1]):
            output[f"p_base_{index}"] = self.raw_base_probabilities[:, index]
        output.update({
            "blended_probability": self.blended_probability,
            "calibrated_probability": self.calibrated_probability,
            "positive_log": self.positive_log,
            "prediction_log": self.prediction_log,
            "prediction": self.prediction,
        })
        return pd.DataFrame(output)


def _positive_probability(model: Any, features: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        value = np.asarray(model.predict_proba(features), dtype=float)
        if value.ndim == 2:
            value = value[:, -1]
    else:
        value = np.asarray(model.predict(features), dtype=float).reshape(-1)
    value = value.reshape(-1)
    if len(value) != len(features) or not np.isfinite(value).all():
        raise ValueError("classifier returned invalid probabilities")
    return np.clip(value, 0.0, 1.0)


def _calibrated(calibrator: Any, raw: np.ndarray) -> np.ndarray:
    if calibrator is None:
        return np.clip(raw, 0.0, 1.0)
    if hasattr(calibrator, "predict_proba"):
        result = np.asarray(calibrator.predict_proba(raw), dtype=float)
        if result.ndim == 2:
            result = result[:, -1]
    elif callable(calibrator):
        result = np.asarray(calibrator(raw), dtype=float)
    else:
        raise TypeError("calibrator must implement predict_proba or be callable")
    result = result.reshape(-1)
    if len(result) != len(raw) or not np.isfinite(result).all():
        raise ValueError("calibrator returned invalid probabilities")
    return np.clip(result, 0.0, 1.0)


def predict_ensemble(
    frame: pd.DataFrame,
    *,
    classifiers: Sequence[Any],
    positive_regressor: Any,
    calibrator: Any = None,
    weights: Sequence[float] | None = None,
    feature_names: Sequence[str],
    expected_user_ids: Sequence[Any] | None = None,
) -> InferenceResult:
    """Predict without changing row order and expose every intermediate value."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("frame must be a non-empty DataFrame")
    names = tuple(feature_names)
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    actual_feature_order = tuple(column for column in frame.columns if column in names)
    if actual_feature_order != names:
        raise ValueError("Feature names or order do not match the contract")
    if not classifiers:
        raise ValueError("at least one classifier is required")
    if weights is None:
        weight_values = np.full(len(classifiers), 1.0 / len(classifiers), dtype=float)
    else:
        weight_values = np.asarray(weights, dtype=float).reshape(-1)
    if len(weight_values) != len(classifiers) or not np.isfinite(weight_values).all() \
            or (weight_values < 0).any() or not np.isclose(weight_values.sum(), 1.0):
        raise ValueError("weights must be finite, non-negative, and sum to one")
    user_ids = frame["user_id"].to_numpy(copy=True) if "user_id" in frame.columns else np.arange(len(frame))
    if expected_user_ids is not None:
        expected = np.asarray(expected_user_ids)
        if len(expected) != len(user_ids) or not np.array_equal(user_ids, expected):
            raise ValueError("frame user_id values or order do not match expected IDs")
    features = frame.loc[:, list(names)]
    base = np.column_stack([_positive_probability(model, features) for model in classifiers])
    blended = np.clip(base @ weight_values, 0.0, 1.0)
    calibrated = _calibrated(calibrator, blended)
    positive_log = np.asarray(positive_regressor.predict(features), dtype=float).reshape(-1)
    if len(positive_log) != len(frame) or not np.isfinite(positive_log).all():
        raise ValueError("positive regressor returned invalid predictions")
    combined = combine_predictions(calibrated, positive_log)
    return InferenceResult(user_ids, base, blended, calibrated, positive_log,
                           combined.prediction_log, combined.prediction)
