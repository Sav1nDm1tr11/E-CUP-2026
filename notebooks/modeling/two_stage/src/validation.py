"""Исполняемые контракты данных двухстадийной модели."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


def validate_feature_matrix(
    features: pd.DataFrame,
    expected_columns: Sequence[str],
) -> None:
    """Проверяет порядок, типы и конечность определённых значений признаков."""
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features должен быть pandas.DataFrame")
    if features.empty:
        raise ValueError("Матрица признаков не должна быть пустой")

    expected_columns = tuple(expected_columns)
    actual_columns = tuple(features.columns)
    if actual_columns != expected_columns:
        raise ValueError(
            "Не совпали состав или порядок признаков: "
            f"ожидалось {expected_columns}, получено {actual_columns}"
        )

    non_numeric = [
        column
        for column in features.columns
        if not is_numeric_dtype(features[column].dtype)
    ]
    if non_numeric:
        raise ValueError(f"Нечисловые признаки: {non_numeric}")

    for column in features.select_dtypes(include=["floating"]).columns:
        if np.isinf(features[column].to_numpy(copy=False)).any():
            raise ValueError(f"Признак {column} содержит infinity")


def validate_temporal_split(train_dates, valid_dates) -> None:
    """Проверяет, что validation строго позже всех train-наблюдений."""
    train_dates = pd.to_datetime(train_dates)
    valid_dates = pd.to_datetime(valid_dates)

    if len(train_dates) == 0 or len(valid_dates) == 0:
        raise ValueError("Train и validation периоды не должны быть пустыми")
    if train_dates.max() >= valid_dates.min():
        raise ValueError("Validation период должен быть строго позже train")


def make_submission(expected_user_ids, predictions) -> pd.DataFrame:
    """Строит submission в обязательном порядке ожидаемых пользователей."""
    expected_user_ids = np.asarray(expected_user_ids)
    predictions = np.asarray(predictions, dtype=np.float64)

    if expected_user_ids.ndim != 1 or predictions.ndim != 1:
        raise ValueError("user_id и predictions должны быть одномерными")
    if expected_user_ids.shape != predictions.shape:
        raise ValueError("Число user_id не совпадает с числом predictions")
    if expected_user_ids.size == 0:
        raise ValueError("Submission не должен быть пустым")
    if len(np.unique(expected_user_ids)) != len(expected_user_ids):
        raise ValueError("expected_user_ids содержит дубликаты")
    if not np.isfinite(predictions).all():
        raise ValueError("predictions содержит NaN или infinity")
    if (predictions < 0).any():
        raise ValueError("predictions содержит отрицательные значения")

    submission = pd.DataFrame(
        {
            "user_id": expected_user_ids,
            "predict": predictions,
        }
    )
    validate_submission(submission, expected_user_ids)
    return submission


def validate_submission(
    submission: pd.DataFrame,
    expected_user_ids,
) -> None:
    """Проверяет schema, порядок пользователей и значения submission."""
    if not isinstance(submission, pd.DataFrame):
        raise TypeError("submission должен быть pandas.DataFrame")
    if submission.columns.tolist() != ["user_id", "predict"]:
        raise ValueError("Submission должен содержать только user_id и predict")

    expected_user_ids = np.asarray(expected_user_ids)
    actual_user_ids = submission["user_id"].to_numpy()
    predictions = submission["predict"].to_numpy(dtype=np.float64)

    if actual_user_ids.shape != expected_user_ids.shape:
        raise ValueError("Число пользователей в submission не совпало с ожидаемым")
    if not np.array_equal(actual_user_ids, expected_user_ids):
        raise ValueError("Состав или порядок user_id не совпал с sample submission")
    if not submission["user_id"].is_unique:
        raise ValueError("Submission содержит дубликаты user_id")
    if not np.isfinite(predictions).all():
        raise ValueError("Submission содержит NaN или infinity")
    if (predictions < 0).any():
        raise ValueError("Submission содержит отрицательные predictions")
