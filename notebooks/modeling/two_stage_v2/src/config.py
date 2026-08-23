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
    bootstrap_resamples: int = 2000
    simplex_step: float = 0.05
    resource_gate_fraction: float = 0.70
    model_search_space: ModelSearchSpace = field(default_factory=ModelSearchSpace)


def build_default_config(project_root: Path) -> ExperimentConfig:
    root = Path(project_root).expanduser().resolve()
    return ExperimentConfig(
        project_root=root,
        dataset_path=root / "data" / "Prepared_data.parquet",
        artifacts_dir=root / "data" / "two_stage_v2_artifacts",
        feature_columns_path=root / "data" / "two_stage_v2_artifacts" / "feature_columns.json",
        sample_submission_path=root / "data" / "sample_submission.csv",
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
