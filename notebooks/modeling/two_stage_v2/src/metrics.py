"""Metrics used by the two-stage ensemble and its uncertainty report."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def rmsle(actual, predicted) -> float:
    y_true = np.asarray(actual, dtype=float)
    y_pred = np.asarray(predicted, dtype=float)
    if y_true.shape != y_pred.shape or y_true.ndim != 1:
        raise ValueError("actual and predicted must be equally sized one-dimensional arrays")
    if len(y_true) == 0:
        raise ValueError("RMSLE inputs must not be empty")
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
    if n == 0:
        raise ValueError("Bootstrap inputs must not be empty")
    if not np.isfinite(actual).all() or not np.isfinite(candidate).all() or not np.isfinite(baseline).all():
        raise ValueError("Bootstrap inputs must be finite")
    if (actual < 0).any() or (candidate < 0).any() or (baseline < 0).any():
        raise ValueError("Bootstrap inputs must be non-negative")

    # Reduce rows once to user x fold sufficient statistics.  A bootstrap draw
    # only needs the number of times each user was sampled; materialising row
    # indices for every draw is both quadratic in practice and memory-heavy.
    unique_groups, group_codes = np.unique(groups, return_inverse=True)
    unique_folds, fold_codes = np.unique(folds, return_inverse=True)
    n_groups, n_folds = len(unique_groups), len(unique_folds)
    flat_codes = group_codes * n_folds + fold_codes
    cell_count = n_groups * n_folds
    counts = np.bincount(flat_codes, minlength=cell_count).reshape(n_groups, n_folds).astype(np.float64)
    candidate_error = (np.log1p(actual) - np.log1p(candidate)) ** 2
    baseline_error = (np.log1p(actual) - np.log1p(baseline)) ** 2
    candidate_sse = np.bincount(flat_codes, weights=candidate_error, minlength=cell_count).reshape(n_groups, n_folds)
    baseline_sse = np.bincount(flat_codes, weights=baseline_error, minlength=cell_count).reshape(n_groups, n_folds)

    def score_from_sufficient_statistics(sample_counts: np.ndarray) -> np.ndarray:
        selected_counts = sample_counts @ counts
        if selected_counts.ndim == 1:
            selected_counts = selected_counts[None, :]
        if not np.all(selected_counts > 0, axis=1).all():
            raise ValueError("Bootstrap sample must contain every fold")
        selected_candidate = sample_counts @ candidate_sse
        selected_baseline = sample_counts @ baseline_sse
        candidate_rmsle = np.sqrt(selected_candidate / selected_counts)
        baseline_rmsle = np.sqrt(selected_baseline / selected_counts)
        return np.mean(candidate_rmsle - baseline_rmsle, axis=1)

    point = float(score_from_sufficient_statistics(np.ones((1, n_groups), dtype=np.float64))[0])
    rng = np.random.default_rng(seed)
    values = np.empty(n_resamples, dtype=float)
    # Bound the multinomial matrix to ~16 MiB so 250k-user runs never create a
    # giant list/indices object.  The result is immediately reduced and freed.
    max_chunk = max(1, min(n_resamples, (16 * 1024 * 1024) // max(8 * n_groups, 1)))
    filled = 0
    probabilities = np.full(n_groups, 1.0 / n_groups)
    attempts = 0
    max_attempts = n_resamples * 1000
    while filled < n_resamples and attempts < max_attempts:
        remaining = n_resamples - filled
        chunk = min(max_chunk, remaining, max_attempts - attempts)
        sample_counts = rng.multinomial(n_groups, probabilities, size=chunk)
        attempts += chunk
        selected_counts = sample_counts @ counts
        valid = np.all(selected_counts > 0, axis=1)
        if valid.any():
            valid_scores = score_from_sufficient_statistics(sample_counts[valid])
            take = min(len(valid_scores), n_resamples - filled)
            values[filled:filled + take] = valid_scores[:take]
            filled += take
    if filled < n_resamples:
        raise ValueError("Unable to construct a bootstrap sample containing every fold")
    low, high = np.percentile(values, [2.5, 97.5])
    return BootstrapDelta(point, point, float(low), float(high), int(n_resamples))
