"""Metrics used by the two-stage ensemble and its uncertainty report."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def rmsle(actual, predicted) -> float:
    y_true = np.asarray(actual, dtype=float)
    y_pred = np.asarray(predicted, dtype=float)
    if y_true.shape != y_pred.shape or y_true.ndim != 1:
        raise ValueError("actual and predicted must be equally sized one-dimensional arrays")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("RMSLE inputs must be finite")
    if (y_true < 0).any() or (y_pred < 0).any():
        raise ValueError("RMSLE inputs must be non-negative")
    return float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)))


@dataclass(frozen=True)
class BootstrapDelta:
    delta: float
    point_delta: float
    ci_low: float
    ci_high: float
    n_resamples: int


def paired_cluster_bootstrap_delta(
    actual, candidate, baseline, group, fold, n_resamples: int = 2000, seed: int = 42
) -> BootstrapDelta:
    actual = np.asarray(actual, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    groups = np.asarray(group)
    folds = np.asarray(fold)
    if not (actual.ndim == candidate.ndim == baseline.ndim == groups.ndim == folds.ndim == 1):
        raise ValueError("Bootstrap inputs must be one-dimensional")
    n = len(actual)
    if any(len(values) != n for values in (candidate, baseline, groups, folds)):
        raise ValueError("Bootstrap inputs must have equal length")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    unique_groups = np.unique(groups)
    unique_folds = np.unique(folds)

    def score(indices: np.ndarray) -> float:
        by_fold = []
        for current_fold in unique_folds:
            selected = indices[folds[indices] == current_fold]
            if len(selected) == 0:
                continue
            by_fold.append(rmsle(actual[selected], candidate[selected]) - rmsle(actual[selected], baseline[selected]))
        if not by_fold:
            raise ValueError("Bootstrap sample contains no folds")
        return float(np.mean(by_fold))

    point = score(np.arange(n))
    rng = np.random.default_rng(seed)
    values = np.empty(n_resamples, dtype=float)
    group_rows = {key: np.flatnonzero(groups == key) for key in unique_groups}
    for i in range(n_resamples):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([group_rows[key] for key in sampled])
        values[i] = score(indices)
    low, high = np.percentile(values, [2.5, 97.5])
    return BootstrapDelta(point, point, float(low), float(high), int(n_resamples))
