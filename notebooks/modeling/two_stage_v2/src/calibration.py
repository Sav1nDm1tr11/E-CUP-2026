"""One-dimensional Platt (sigmoid) calibration."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted


class IdentityCalibrator(ClassifierMixin, BaseEstimator):
    """A cloneable calibrator that preserves already-calibrated probabilities."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    @staticmethod
    def _probability(values):
        p = np.asarray(values, dtype=float).reshape(-1)
        if len(p) == 0 or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
            raise ValueError("probability must be non-empty, finite, and in [0, 1]")
        return p

    def fit(self, probability, y=None):
        p = self._probability(probability)
        if not np.isfinite(float(self.threshold)) or not 0 <= float(self.threshold) <= 1:
            raise ValueError("threshold must be finite and in [0, 1]")
        if y is not None:
            target = np.asarray(y).reshape(-1)
            if len(target) != len(p) or set(np.unique(target).tolist()) - {0, 1}:
                raise ValueError("probability and y must have equal length and binary labels")
        self.classes_ = np.array([0, 1], dtype=np.int8)
        self.n_features_in_ = 1
        return self

    def predict_proba(self, probability):
        check_is_fitted(self, "classes_")
        p = self._probability(probability)
        return np.column_stack((1.0 - p, p))

    def predict(self, probability):
        check_is_fitted(self, "classes_")
        p = self._probability(probability)
        return (p >= float(self.threshold)).astype(np.int8)


class SigmoidCalibrator(ClassifierMixin, BaseEstimator):
    def __init__(self, epsilon: float = 1e-6):
        self.epsilon = epsilon

    def fit(self, probability, y):
        p = np.asarray(probability, dtype=float).reshape(-1)
        target = np.asarray(y).reshape(-1)
        if len(p) != len(target) or len(p) == 0:
            raise ValueError("probability and y must have equal non-zero length")
        if not np.isfinite(p).all() or not np.isfinite(target.astype(float)).all():
            raise ValueError("calibration inputs must be finite")
        if not 0 < self.epsilon < 0.5:
            raise ValueError("epsilon must be between zero and 0.5")
        if set(np.unique(target).tolist()) != {0, 1}:
            raise ValueError("Sigmoid calibration labels must be exactly {0, 1}")
        clipped = np.clip(p, self.epsilon, 1.0 - self.epsilon)
        logit = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
        self.model_ = LogisticRegression(C=1e6, solver="lbfgs", random_state=42)
        self.model_.fit(logit, target)
        self.a = float(self.model_.coef_[0, 0])
        self.b = float(self.model_.intercept_[0])
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, probability):
        check_is_fitted(self, ("model_", "a", "b"))
        p = np.asarray(probability, dtype=float).reshape(-1)
        if not np.isfinite(p).all():
            raise ValueError("probability must be finite")
        clipped = np.clip(p, self.epsilon, 1.0 - self.epsilon)
        result = self.model_.predict_proba(np.log(clipped / (1.0 - clipped)).reshape(-1, 1))
        return np.clip(result, 0.0, 1.0)

    def predict(self, probability):
        check_is_fitted(self, "model_")
        p = np.asarray(probability, dtype=float).reshape(-1)
        clipped = np.clip(p, self.epsilon, 1.0 - self.epsilon)
        return self.model_.predict(np.log(clipped / (1.0 - clipped)).reshape(-1, 1))
