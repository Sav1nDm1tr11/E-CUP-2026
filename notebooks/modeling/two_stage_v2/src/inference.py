"""Auditable, order-preserving inference for the two-stage ensemble."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .blending import combine_predictions


_RESERVED_COLUMNS = frozenset({
    "user_id", "cutoff_date", "target_nonzero", "target_gmv_30d", "prediction",
    "sample_weight", "fold", "group", "row_id",
})


@dataclass(frozen=True)
class InferenceResult:
    user_id: np.ndarray
    raw_base_probabilities: np.ndarray
    blended_probability: np.ndarray
    calibrated_probability: np.ndarray
    positive_log: np.ndarray
    prediction_log: np.ndarray
    prediction: np.ndarray
    model_names: tuple[str, ...] = ()
    feature_names: tuple[str, ...] = ()
    bundle_version: int | None = None

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
    if (value < 0).any() or (value > 1).any():
        raise ValueError("classifier returned probabilities outside [0, 1]")
    return value


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
    if (result < 0).any() or (result > 1).any():
        raise ValueError("calibrator returned probabilities outside [0, 1]")
    return result


def _feature_hash(names: Sequence[str]) -> str:
    return hashlib.sha256(json.dumps(list(names), separators=(",", ":")).encode("utf-8")).hexdigest()


def _bundle_values(bundle: Any) -> dict[str, Any]:
    if bundle is None:
        return {}
    if isinstance(bundle, dict):
        value = dict(bundle)
    else:
        value = {name: getattr(bundle, name) for name in (
            "manifest", "feature_names", "classifiers", "positive_regressor", "calibrator"
        ) if hasattr(bundle, name)}
    manifest = value.get("manifest")
    if manifest is not None:
        value.setdefault("weights", getattr(manifest, "weights", None))
        value.setdefault("model_names", getattr(manifest, "model_names", None))
        value.setdefault("bundle_version", getattr(manifest, "version", None))
        value.setdefault("feature_sha256", getattr(manifest, "feature_sha256", None))
    return value


def predict_ensemble(
    frame: pd.DataFrame,
    *,
    classifiers: Sequence[Any] | None = None,
    positive_regressor: Any = None,
    calibrator: Any = None,
    weights: Sequence[float] | None = None,
    feature_names: Sequence[str] | None = None,
    expected_user_ids: Sequence[Any] | None = None,
    bundle: Any = None,
) -> InferenceResult:
    """Predict without changing row order and expose every intermediate value."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("frame must be a non-empty DataFrame")
    bundle_values = _bundle_values(bundle)
    if bundle_values:
        manifest = bundle_values.get("manifest")
        if manifest is None or getattr(manifest, "version", None) != 1:
            raise ValueError("inference requires a validated version-1 bundle")
        if classifiers is None:
            classifiers = bundle_values.get("classifiers")
        if positive_regressor is None:
            positive_regressor = bundle_values.get("positive_regressor")
        if calibrator is None:
            calibrator = bundle_values.get("calibrator")
        if feature_names is None:
            feature_names = bundle_values.get("feature_names")
    if classifiers is None or positive_regressor is None or feature_names is None:
        raise ValueError("classifiers, positive_regressor, and feature_names or a validated bundle are required")
    if bundle_values:
        if bundle_values.get("feature_names") is not None and tuple(feature_names) != tuple(bundle_values["feature_names"]):
            raise ValueError("feature names do not match validated bundle")
        if bundle_values.get("classifiers") is not None and tuple(classifiers) != tuple(bundle_values["classifiers"]):
            raise ValueError("classifiers do not match validated bundle")
        if bundle_values.get("positive_regressor") is not None and positive_regressor is not bundle_values["positive_regressor"]:
            raise ValueError("positive regressor does not match validated bundle")
        if weights is not None and bundle_values.get("weights") is not None and not np.allclose(weights, bundle_values["weights"]):
            raise ValueError("weights do not match validated bundle")
        if weights is None:
            weights = bundle_values.get("weights")
    names = tuple(feature_names)
    if len(names) != 91 or len(set(names)) != 91:
        raise ValueError("inference requires exactly 91 unique ordered feature names")
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    actual_feature_order = tuple(column for column in frame.columns if column not in _RESERVED_COLUMNS)
    if actual_feature_order != names:
        raise ValueError("Feature names or order do not match the contract")
    expected_hash = bundle_values.get("feature_sha256")
    if expected_hash is not None and expected_hash != _feature_hash(names):
        raise ValueError("feature contract hash does not match bundle")
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
    blended = base @ weight_values
    if (blended < 0).any() or (blended > 1).any() or not np.isfinite(blended).all():
        raise ValueError("blended probability outside [0, 1]")
    calibrated = _calibrated(calibrator, blended)
    positive_log = np.asarray(positive_regressor.predict(features), dtype=float).reshape(-1)
    if len(positive_log) != len(frame) or not np.isfinite(positive_log).all():
        raise ValueError("positive regressor returned invalid predictions")
    combined = combine_predictions(calibrated, positive_log)
    model_names = tuple(bundle_values.get("model_names") or (f"model_{i}" for i in range(len(classifiers))))
    if len(model_names) != len(classifiers):
        raise ValueError("model names and classifiers are not aligned")
    return InferenceResult(user_ids, base, blended, calibrated, positive_log,
                           combined.prediction_log, combined.prediction, model_names, names,
                           bundle_values.get("bundle_version"))
