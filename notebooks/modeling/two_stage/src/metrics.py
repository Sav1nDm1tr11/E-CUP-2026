"""Метрики и LightGBM eval-contracts двухстадийной модели."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)


def rmsle(actual, prediction) -> float:
    """Вычисляет RMSLE и строго отклоняет некорректные входы."""
    actual = np.asarray(actual, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)

    if actual.shape != prediction.shape:
        raise ValueError("actual и prediction должны иметь одинаковую форму")
    if actual.size == 0:
        raise ValueError("Нельзя вычислить RMSLE на пустом массиве")
    if not np.isfinite(actual).all():
        raise ValueError("actual содержит NaN или infinity")
    if not np.isfinite(prediction).all():
        raise ValueError("prediction содержит NaN или infinity")
    if (actual < 0).any():
        raise ValueError("actual содержит отрицательные значения")
    if (prediction < 0).any():
        raise ValueError("prediction содержит отрицательные значения")

    log_error = np.log1p(prediction) - np.log1p(actual)
    return float(np.sqrt(np.mean(np.square(log_error))))


def classification_metrics(actual, probability, threshold=0.5) -> dict[str, float]:
    """Возвращает метрики первой стадии при заданном пороге."""
    actual = np.asarray(actual, dtype=np.int8)
    probability = np.asarray(probability, dtype=np.float64)

    if actual.shape != probability.shape:
        raise ValueError("actual и probability должны иметь одинаковую форму")
    if actual.size == 0:
        raise ValueError("Пустой массив классификации")
    if not np.isin(actual, [0, 1]).all():
        raise ValueError("actual должен содержать только 0 и 1")
    if not np.isfinite(probability).all():
        raise ValueError("probability содержит NaN или infinity")
    if ((probability < 0) | (probability > 1)).any():
        raise ValueError("probability должна находиться в диапазоне [0, 1]")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold должен находиться в диапазоне [0, 1]")

    predicted_class = probability >= threshold
    return {
        "average_precision": float(average_precision_score(actual, probability)),
        "logloss": float(log_loss(actual, probability, labels=[0, 1])),
        "brier": float(brier_score_loss(actual, probability)),
        "precision": float(precision_score(actual, predicted_class, zero_division=0)),
        "recall": float(recall_score(actual, predicted_class, zero_division=0)),
        "f1": float(f1_score(actual, predicted_class, zero_division=0)),
    }


def positive_regression_metrics(actual, predicted_log) -> dict[str, float]:
    """Возвращает метрики регрессора на строках с положительным GMV."""
    actual = np.asarray(actual, dtype=np.float64)
    predicted_log = np.asarray(predicted_log, dtype=np.float64)

    if actual.shape != predicted_log.shape:
        raise ValueError("actual и predicted_log должны иметь одинаковую форму")
    if actual.size == 0:
        raise ValueError("Пустой массив положительного GMV")
    if not np.isfinite(actual).all():
        raise ValueError("actual содержит NaN или infinity")
    if not np.isfinite(predicted_log).all():
        raise ValueError("predicted_log содержит NaN или infinity")
    if (actual <= 0).any():
        raise ValueError("Регрессионный actual должен быть строго положительным")

    actual_log = np.log1p(actual)
    log_error = predicted_log - actual_log
    predicted_gmv = np.expm1(np.maximum(predicted_log, 0.0))

    return {
        "rmse_log": float(np.sqrt(np.mean(np.square(log_error)))),
        "mae_log": float(np.mean(np.abs(log_error))),
        "log_bias": float(np.mean(log_error)),
        "sum_ratio": float(predicted_gmv.sum() / actual.sum()),
        "negative_log_prediction_share": float(np.mean(predicted_log < 0)),
    }


def pipeline_metrics(actual, prediction) -> dict[str, float]:
    """Возвращает end-to-end метрики итогового GMV-прогноза."""
    actual = np.asarray(actual, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)

    score = rmsle(actual, prediction)
    actual_sum = actual.sum()
    log_error = np.log1p(prediction) - np.log1p(actual)

    return {
        "rmsle": score,
        "sum_ratio": (
            float(prediction.sum() / actual_sum) if actual_sum > 0 else np.nan
        ),
        "log_bias": float(np.mean(log_error)),
    }


def make_soft_pipeline_rmsle_metric(
    probability,
) -> Callable[[np.ndarray, np.ndarray], tuple[str, float, bool]]:
    """Создаёт LightGBM eval metric для soft-log объединения стадий."""
    probability = np.asarray(probability, dtype=np.float64)

    if probability.size == 0:
        raise ValueError("probability не должна быть пустой")
    if not np.isfinite(probability).all():
        raise ValueError("probability содержит NaN или infinity")
    if ((probability < 0) | (probability > 1)).any():
        raise ValueError("probability должна находиться в диапазоне [0, 1]")

    def soft_pipeline_rmsle_metric(actual_log, predicted_log):
        actual_log = np.asarray(actual_log, dtype=np.float64)
        predicted_log = np.asarray(predicted_log, dtype=np.float64)

        if actual_log.shape != predicted_log.shape:
            raise ValueError("actual_log и predicted_log должны иметь одинаковую форму")
        if actual_log.shape != probability.shape:
            raise ValueError("Длина probability не совпадает с eval-набором")
        if not np.isfinite(actual_log).all() or not np.isfinite(predicted_log).all():
            raise ValueError("eval metric получила NaN или infinity")

        pipeline_predicted_log = probability * np.maximum(predicted_log, 0.0)
        metric_value = float(
            np.sqrt(np.mean(np.square(pipeline_predicted_log - actual_log)))
        )
        return "soft_pipeline_rmsle", metric_value, False

    return soft_pipeline_rmsle_metric
