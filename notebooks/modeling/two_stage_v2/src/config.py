"""Immutable configuration for the v2 two-stage experiment."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ModelSearchSpace:
    """Small, serialisable search spaces used by the model stages."""

    lightgbm_classifier: tuple[tuple[str, tuple[Any, ...]], ...] = (
        ("learning_rate", (0.03, 0.05)), ("num_leaves", (15, 31)),
    )
    lightgbm_regressor: tuple[tuple[str, tuple[Any, ...]], ...] = (
        ("learning_rate", (0.03, 0.05)), ("num_leaves", (15, 31)),
    )
    catboost_classifier: tuple[tuple[str, tuple[Any, ...]], ...] = (
        ("depth", (5, 7)), ("learning_rate", (0.03, 0.05)),
    )
    ebm_classifier: tuple[tuple[str, tuple[Any, ...]], ...] = (
        ("max_bins", (64, 128)), ("interactions", (0, 10)),
    )


@dataclass(frozen=True)
class ExperimentConfig:
    project_root: Path
    dataset_path: Path
    artifacts_dir: Path
    feature_columns_path: Path | None = None
    sample_submission_path: Path | None = None
    cutoff_dates: tuple[pd.Timestamp, ...] = ()
    report_dates: tuple[pd.Timestamp, ...] = ()
    random_seed: int = 42
    n_jobs: int = 1
    estimator_threads: int = 10
    trial_parallelism: int = 1
    bootstrap_resamples: int = 2000
    simplex_step: float = 0.05
    resource_gate_fraction: float = 0.70
    catboost_trials: int = 20
    catboost_ordered_pilot_trials: int = 3
    catboost_max_iterations: int = 5000
    catboost_early_stopping_rounds: int = 150
    ebm_max_configurations: int = 12
    ebm_pilot_wall_clock_seconds: int = 5400
    max_logloss_degradation: float = 0.02
    max_brier_degradation: float = 0.02
    frozen_classifier_n_estimators: int = 722
    frozen_regressor_n_estimators: int = 1666
    frozen_classifier_params: tuple[tuple[str, Any], ...] = (
        ("random_state", 42), ("n_jobs", 10), ("deterministic", True),
        ("force_col_wise", True), ("verbosity", -1), ("max_depth", 10),
        ("learning_rate", 0.02111084036846094), ("num_leaves", 209),
        ("min_child_samples", 102), ("subsample", 0.7233468086168436),
        ("colsample_bytree", 0.809790174827837), ("reg_alpha", 9.941450502972573),
        ("reg_lambda", 0.687875101121605), ("subsample_freq", 1),
    )
    frozen_regressor_params: tuple[tuple[str, Any], ...] = (
        ("random_state", 42), ("n_jobs", 10), ("deterministic", True),
        ("force_col_wise", True), ("verbosity", -1), ("max_depth", 6),
        ("learning_rate", 0.026346090172912194), ("num_leaves", 44),
        ("min_child_samples", 1774), ("subsample", 0.7843800634757309),
        ("colsample_bytree", 0.9759305296305512), ("reg_alpha", 0.001068152168205832),
        ("reg_lambda", 0.0031698605715400154), ("subsample_freq", 1),
    )
    model_search_space: ModelSearchSpace = field(default_factory=ModelSearchSpace)


def build_default_config(project_root: Path) -> ExperimentConfig:
    root = Path(project_root).expanduser().resolve()
    cutoff_dates = tuple(pd.to_datetime([
        "2025-04-19", "2025-05-19", "2025-06-18", "2025-07-18",
        "2025-08-17", "2025-09-16", "2025-10-16", "2025-11-15",
        "2025-12-15", "2026-01-14", "2026-02-13",
    ]))
    report_dates = tuple(pd.to_datetime([
        "2025-10-16", "2025-11-15", "2025-12-15", "2026-01-14",
    ]))
    return ExperimentConfig(
        project_root=root,
        dataset_path=root / "data" / "Prepared_data.parquet",
        artifacts_dir=root / "data" / "two_stage_v2_artifacts",
        feature_columns_path=root / "data" / "two_stage_v2_artifacts" / "feature_columns.json",
        sample_submission_path=root / "data" / "sample_submission.csv",
        cutoff_dates=cutoff_dates,
        report_dates=report_dates,
    )


def _json_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {k: _json_value(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if isinstance(value, tuple):
        return [_json_value(v) for v in value]
    if isinstance(value, list):
        return [_json_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
    return value


def config_fingerprint(config: ExperimentConfig) -> str:
    """Return a stable SHA-256 hash of the canonical UTF-8 JSON config."""
    payload = json.dumps(_json_value(config), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
