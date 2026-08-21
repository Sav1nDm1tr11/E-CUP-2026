"""Единая формула inference для двухстадийной модели."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TwoStagePredictions:
    """Компоненты и итоговый прогноз двухстадийной модели."""

    probability: np.ndarray
    predicted_positive_log: np.ndarray
    prediction_log: np.ndarray
    prediction: np.ndarray


def combine_two_stage_predictions(
    probability,
    predicted_positive_log,
    *,
    scale: float,
    gamma: float,
) -> TwoStagePredictions:
    """Объединяет вероятность и условный log1p(GMV) в итоговый прогноз."""
    probability = np.asarray(probability, dtype=np.float64)
    predicted_positive_log = np.asarray(
        predicted_positive_log,
        dtype=np.float64,
    )

    if probability.shape != predicted_positive_log.shape:
        raise ValueError(
            "probability и predicted_positive_log должны иметь одинаковую форму"
        )
    if probability.size == 0:
        raise ValueError("Нельзя построить прогноз для пустого массива")
    if not np.isfinite(probability).all():
        raise ValueError("probability содержит NaN или infinity")
    if not np.isfinite(predicted_positive_log).all():
        raise ValueError("predicted_positive_log содержит NaN или infinity")
    if ((probability < 0) | (probability > 1)).any():
        raise ValueError("probability должна находиться в диапазоне [0, 1]")
    if not np.isfinite(scale) or scale < 0:
        raise ValueError("scale должен быть конечным и неотрицательным")
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma должен быть конечным и положительным")

    predicted_positive_log = np.maximum(predicted_positive_log, 0.0)
    prediction_log = (
        float(scale) * np.power(probability, float(gamma)) * predicted_positive_log
    )
    prediction = np.expm1(prediction_log)

    if not np.isfinite(prediction).all():
        raise ValueError("Итоговый prediction содержит NaN или infinity")

    return TwoStagePredictions(
        probability=probability,
        predicted_positive_log=predicted_positive_log,
        prediction_log=prediction_log,
        prediction=prediction,
    )


def predict_two_stage(
    classifier,
    regressor,
    features,
    *,
    scale: float,
    gamma: float,
) -> TwoStagePredictions:
    """Выполняет обе стадии и возвращает все компоненты прогноза."""
    probabilities = np.asarray(classifier.predict_proba(features))
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("classifier.predict_proba должен вернуть матрицу N × 2")

    if not hasattr(classifier, "classes_"):
        raise ValueError("classifier должен предоставлять classes_")
    classes = np.asarray(classifier.classes_)
    if classes.ndim != 1 or classes.size != probabilities.shape[1]:
        raise ValueError(
            "classifier.classes_ должен быть одномерным и соответствовать "
            "числу колонок predict_proba"
        )
    positive_indices = np.flatnonzero(classes == 1)
    if positive_indices.size != 1:
        raise ValueError("classifier.classes_ должен содержать ровно один класс 1")

    predicted_positive_log = regressor.predict(features)
    return combine_two_stage_predictions(
        probabilities[:, positive_indices[0]],
        predicted_positive_log,
        scale=scale,
        gamma=gamma,
    )
