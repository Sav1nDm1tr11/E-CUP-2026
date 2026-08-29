"""Lightweight post-hoc activity recalibration for frozen LSTM predictions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROBABILITY_EPS = 1e-6

ORACLE_STATIC_FEATURES = (
    "customer_age_days",
    "never_purchased",
    "days_since_last_purchase",
    "days_since_last_activity",
    "days_since_last_search",
    "days_since_last_cart",
    "gmv_7d",
    "gmv_30d",
    "gmv_90d",
    "purchased_items_7d",
    "purchased_items_30d",
    "purchased_items_90d",
    "searches_7d",
    "searches_30d",
    "searches_90d",
    "active_days_30d",
    "active_days_90d",
    "gmv_trend_log",
    "intent_trend_log",
    "whale_score",
    "purchase_frequency",
    "recency_ratio",
    "active_day_rate_lifetime",
    "no_recent_engagement",
    "gmv_per_item",
    "gmv_daily_mean",
    "search_to_purchase_freq",
    "searches_trend_log",
    "purchased_items_trend_log",
    "active_days_trend_log",
)


def _clip_probability(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("probability values must be finite")
    return np.clip(
        values,
        PROBABILITY_EPS,
        1.0 - PROBABILITY_EPS,
    )


def _logit(values: np.ndarray) -> np.ndarray:
    probability = _clip_probability(values)
    return np.log(probability) - np.log1p(-probability)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-values))


def compose_proxy_log_prediction(
    baseline_log,
    baseline_probability,
    oracle_probability,
    probability_weight,
    correction_weight,
    max_ratio,
) -> np.ndarray:
    """Correct a frozen log prediction using only an activity-probability ratio."""
    if not 0.0 <= probability_weight <= 1.0:
        raise ValueError("probability_weight must be in [0, 1]")
    if not 0.0 <= correction_weight <= 1.0:
        raise ValueError("correction_weight must be in [0, 1]")
    if max_ratio < 1.0:
        raise ValueError("max_ratio must be at least 1")

    baseline_log = np.asarray(baseline_log, dtype=np.float64)
    baseline_probability = _clip_probability(baseline_probability)
    oracle_probability = _clip_probability(oracle_probability)
    if not (
        baseline_log.shape
        == baseline_probability.shape
        == oracle_probability.shape
    ):
        raise ValueError("prediction and probability arrays must have equal shapes")
    if not np.isfinite(baseline_log).all() or (baseline_log < 0.0).any():
        raise ValueError("baseline_log must be finite and nonnegative")

    blended_probability = _sigmoid(
        (1.0 - probability_weight) * _logit(baseline_probability)
        + probability_weight * _logit(oracle_probability)
    )
    ratio = np.clip(
        blended_probability / baseline_probability,
        1.0 / max_ratio,
        max_ratio,
    )
    result = baseline_log * np.exp(correction_weight * np.log(ratio))
    return np.clip(result, 0.0, None)


def select_best_correction(
    baseline_log,
    target_log,
    baseline_probability,
    oracle_probability,
    probability_weights,
    correction_weights,
    max_ratios,
) -> dict[str, float]:
    """Select the least invasive candidate among equal tuning RMSLE values."""
    baseline_log = np.asarray(baseline_log, dtype=np.float64)
    target_log = np.asarray(target_log, dtype=np.float64)
    candidates = []
    for max_ratio in max_ratios:
        for probability_weight in probability_weights:
            for correction_weight in correction_weights:
                prediction_log = compose_proxy_log_prediction(
                    baseline_log=baseline_log,
                    baseline_probability=baseline_probability,
                    oracle_probability=oracle_probability,
                    probability_weight=float(probability_weight),
                    correction_weight=float(correction_weight),
                    max_ratio=float(max_ratio),
                )
                error = target_log - prediction_log
                candidates.append(
                    {
                        "probability_weight": float(probability_weight),
                        "correction_weight": float(correction_weight),
                        "max_ratio": float(max_ratio),
                        "rmsle": float(np.sqrt(np.mean(error * error))),
                    }
                )
    return min(
        candidates,
        key=lambda row: (
            row["rmsle"],
            row["probability_weight"],
            row["correction_weight"],
            row["max_ratio"],
        ),
    )


def load_project_inputs(project_root: str | Path) -> dict[str, object]:
    """Load aligned January labels/features and February frozen predictions."""
    project_root = Path(project_root).resolve()
    data_dir = project_root / "data" / "lstm"
    model_dir = project_root / "models" / "lstm_hurdle_v4_robust"
    with (data_dir / "meta.json").open(encoding="utf-8") as stream:
        metadata = json.load(stream)

    jan_dir = data_dir / metadata["labeled_cutoffs"][-1]
    feb_dir = data_dir / metadata["inference_cutoff"]
    diagnostics = pd.read_csv(model_dir / "jan_holdout_diagnostics.csv")
    baseline_submission = pd.read_csv(
        project_root / "submissions" / "lstm_hurdle_v4_robust.csv"
    )

    jan_user_ids = np.load(jan_dir / "user_id.npy")
    feb_user_ids = np.load(feb_dir / "user_id.npy")
    submission_user_ids = baseline_submission["user_id"].to_numpy()
    diagnostic_user_ids = diagnostics["user_id"].to_numpy()
    if not np.array_equal(jan_user_ids, diagnostic_user_ids):
        raise ValueError("January diagnostics and static arrays are misaligned")
    if not np.array_equal(feb_user_ids, submission_user_ids):
        raise ValueError("February arrays and baseline submission are misaligned")
    if not np.array_equal(jan_user_ids, feb_user_ids):
        raise ValueError("January and February user order differs")

    static_names = list(metadata["static_features"])
    missing = sorted(set(ORACLE_STATIC_FEATURES) - set(static_names))
    if missing:
        raise KeyError(f"Missing static features: {missing}")
    static_indices = [static_names.index(name) for name in ORACLE_STATIC_FEATURES]

    jan_static = np.load(jan_dir / "static.npy", mmap_mode="r")
    feb_static = np.load(feb_dir / "static.npy", mmap_mode="r")
    jan_base_log = diagnostics["pred_log"].to_numpy(dtype=np.float64)
    feb_base_log = np.log1p(
        baseline_submission["predict"].to_numpy(dtype=np.float64)
    )
    jan_history = np.load(jan_dir / "history_length.npy")
    feb_history = np.load(feb_dir / "history_length.npy")

    jan_features = np.column_stack(
        [jan_base_log, jan_history, jan_static[:, static_indices]]
    ).astype(np.float32, copy=False)
    feb_features = np.column_stack(
        [feb_base_log, feb_history, feb_static[:, static_indices]]
    ).astype(np.float32, copy=False)
    feature_names = ("lstm_pred_log", "history_length", *ORACLE_STATIC_FEATURES)

    return {
        "jan_user_ids": jan_user_ids,
        "feb_user_ids": feb_user_ids,
        "submission_user_ids": submission_user_ids,
        "jan_features": jan_features,
        "feb_features": feb_features,
        "feature_names": feature_names,
        "jan_base_log": jan_base_log,
        "feb_base_log": feb_base_log,
        "target_log": diagnostics["target_log"].to_numpy(dtype=np.float64),
        "target_active": (diagnostics["y_true"].to_numpy() > 0).astype(np.int8),
    }


def build_submission(expected_user_ids, predictions) -> pd.DataFrame:
    """Build the exact competition schema while preserving the supplied ID order."""
    user_ids = np.asarray(expected_user_ids)
    predictions = np.asarray(predictions, dtype=np.float64)
    if user_ids.ndim != 1 or predictions.ndim != 1:
        raise ValueError("user IDs and predictions must be one-dimensional")
    if user_ids.shape != predictions.shape:
        raise ValueError("user IDs and predictions must have equal shapes")
    if user_ids.size == 0 or np.unique(user_ids).size != user_ids.size:
        raise ValueError("user IDs must be nonempty and unique")
    if not np.isfinite(predictions).all():
        raise ValueError("predictions must be finite")
    if (predictions < 0.0).any():
        raise ValueError("predictions must be nonnegative")
    return pd.DataFrame({"user_id": user_ids, "predict": predictions})


def passes_submission_gate(
    delta,
    bootstrap_upper,
    better_folds,
    guardrails_pass,
) -> bool:
    """Accept only an improvement that is consistent and statistically one-sided."""
    return bool(
        delta < 0.0
        and bootstrap_upper < 0.0
        and better_folds >= 4
        and guardrails_pass
    )
