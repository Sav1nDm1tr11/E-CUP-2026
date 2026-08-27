"""Deterministic expanding cutoff splits for nested temporal validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator


@dataclass(frozen=True)
class CutoffFold:
    name: str
    train_dates: tuple[pd.Timestamp, ...]
    inner_valid_date: pd.Timestamp
    outer_valid_date: pd.Timestamp


def _unique_sorted(dates) -> tuple[pd.Timestamp, ...]:
    values = tuple(pd.Timestamp(d) for d in dates)
    if not values:
        return ()
    result = tuple(pd.DatetimeIndex(values).sort_values().unique().to_pydatetime())
    return tuple(pd.Timestamp(d) for d in result)


def build_nested_folds(cutoff_dates, report_dates) -> tuple[CutoffFold, ...]:
    available = _unique_sorted(cutoff_dates)
    reports = _unique_sorted(report_dates)
    folds: list[CutoffFold] = []
    for report in reports:
        past = tuple(d for d in available if d < report)
        if len(past) < 2:
            raise ValueError(f"Outer fold {report.date()} requires at least two past cutoff dates")
        folds.append(CutoffFold(
            name=f"outer_{report.date().isoformat()}",
            train_dates=past,
            inner_valid_date=past[-1],
            outer_valid_date=report,
        ))
    return tuple(folds)


class ExpandingCutoffSplit(BaseCrossValidator):
    def __init__(self, folds):
        self.folds = tuple(folds)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return len(self.folds)

    def split(self, X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if groups is None:
            if isinstance(X, pd.DataFrame) and "cutoff_date" in X:
                groups = X["cutoff_date"].to_numpy()
            elif isinstance(getattr(X, "index", None), pd.DatetimeIndex):
                groups = X.index.to_numpy()
            else:
                raise ValueError("groups or a cutoff_date column/index is required")
        group_dates = pd.DatetimeIndex(pd.to_datetime(np.asarray(groups)))
        for fold in self.folds:
            train = np.flatnonzero(np.asarray(group_dates.isin(fold.train_dates)))
            report = np.flatnonzero(np.asarray(group_dates == fold.outer_valid_date))
            if len(train) == 0 or len(report) == 0:
                raise ValueError(f"Fold {fold.name} has no train or report rows")
            if np.intersect1d(train, report).size:
                raise ValueError(f"Fold {fold.name} has overlapping train/report rows")
            yield train, report
