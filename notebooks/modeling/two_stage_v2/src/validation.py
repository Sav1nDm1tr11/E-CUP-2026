"""Runtime validation for data, OOF tables, and submissions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


def validate_feature_contract(actual: Sequence[str], expected: Sequence[str]) -> tuple[str, ...]:
    actual_tuple, expected_tuple = tuple(actual), tuple(expected)
    if len(expected_tuple) != 91:
        raise ValueError(f"Expected feature contract to contain 91 names, got {len(expected_tuple)}")
    if actual_tuple != expected_tuple:
        raise ValueError("Feature names or order do not match the contract")
    return actual_tuple


def validate_feature_matrix(features: pd.DataFrame, expected_columns: Sequence[str]) -> None:
    if not isinstance(features, pd.DataFrame) or features.empty:
        raise ValueError("features must be a non-empty DataFrame")
    validate_feature_contract(features.columns, expected_columns)
    bad = [c for c in features.columns if not is_numeric_dtype(features[c])]
    if bad:
        raise ValueError(f"Non-numeric features: {bad}")
    for column in features.columns:
        values = features[column].to_numpy(copy=False)
        if np.isinf(values).any():
            raise ValueError(f"Feature {column} contains infinity")


def validate_target_contract(target_gmv: Sequence[float], target_nonzero: Sequence[int]) -> None:
    gmv, binary = np.asarray(target_gmv, dtype=float), np.asarray(target_nonzero)
    if gmv.ndim != 1 or binary.ndim != 1 or len(gmv) != len(binary):
        raise ValueError("Targets must be equally sized one-dimensional arrays")
    if not np.isfinite(gmv).all() or (gmv < 0).any():
        raise ValueError("target_gmv_30d must be finite and non-negative")
    if not np.isin(binary, [0, 1]).all():
        raise ValueError("target_nonzero must be binary")
    if not np.array_equal((gmv > 0).astype(np.int8), binary.astype(np.int8)):
        raise ValueError("target_nonzero disagrees with target_gmv_30d")


def validate_cutoff_order(cutoff_dates: Sequence[pd.Timestamp]) -> None:
    dates = pd.DatetimeIndex(pd.to_datetime(cutoff_dates))
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("cutoff dates must be unique and strictly increasing")


def validate_oof_coverage(report_ids: Sequence, prediction_count: Sequence | None = None) -> None:
    values = np.asarray(report_ids)
    if values.ndim != 1 or len(values) == 0 or pd.isna(values).any() or pd.Series(values).duplicated().any():
        raise ValueError("Each report row must have exactly one non-null prediction")
    if prediction_count is not None and not np.array_equal(np.asarray(prediction_count), np.ones(len(values), dtype=int)):
        raise ValueError("OOF rows must be predicted exactly once")


def validate_submission(submission: pd.DataFrame, expected_user_ids: Sequence, expected_rows: int | None = None) -> None:
    if not isinstance(submission, pd.DataFrame):
        raise TypeError("submission must be a DataFrame")
    if list(submission.columns) != ["user_id", "prediction"]:
        raise ValueError("Submission must contain user_id and prediction columns")
    expected = np.asarray(expected_user_ids)
    if expected_rows is not None and len(submission) != expected_rows:
        raise ValueError("Unexpected submission row count")
    actual = submission["user_id"].to_numpy()
    if len(actual) != len(expected) or not np.array_equal(actual, expected):
        raise ValueError("Submission user_id values or order do not match expected IDs")
    values = submission["prediction"].to_numpy(dtype=float)
    if not submission["user_id"].is_unique:
        raise ValueError("Submission contains duplicate user_id")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Submission predictions must be finite and non-negative")
