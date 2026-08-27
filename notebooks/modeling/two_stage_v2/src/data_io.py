"""Arrow-backed data loading with the v2 dataset contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow.dataset as ds


@dataclass(frozen=True)
class DatasetContract:
    path: Path
    columns: tuple[str, ...]
    feature_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    cutoff_column: str = "cutoff_date"
    user_column: str = "user_id"


def find_project_root(start: str | Path) -> Path:
    current = Path(start).expanduser().resolve()
    if current.is_file():
        current = current.parent
    for directory in (current, *current.parents):
        if (directory / "data" / "Prepared_data.parquet").is_file():
            return directory
    raise FileNotFoundError("Не найден маркер проекта: data/Prepared_data.parquet")


def _feature_columns(names: Sequence[str]) -> tuple[str, ...]:
    reserved = {"user_id", "cutoff_date", "target_gmv_30d", "target_nonzero"}
    return tuple(name for name in names if name not in reserved)


def read_dataset_contract(path: Path) -> DatasetContract:
    path = Path(path)
    schema = ds.dataset(path).schema
    names = tuple(schema.names)
    required = {"user_id", "cutoff_date", "target_gmv_30d", "target_nonzero"}
    missing_required = sorted(required.difference(names))
    if missing_required:
        raise ValueError(f"Dataset is missing required contract columns: {missing_required}")
    features = _feature_columns(names)
    if len(features) != 91:
        raise ValueError(f"Expected exactly 91 ordered features, got {len(features)}")
    targets = tuple(name for name in ("target_gmv_30d", "target_nonzero") if name in names)
    return DatasetContract(path, names, features, targets)


def load_feature_columns(path: Path) -> tuple[str, ...]:
    import json

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    columns = payload.get("feature_columns") if isinstance(payload, dict) else payload
    if not isinstance(columns, list) or not all(isinstance(c, str) for c in columns):
        raise ValueError("feature_columns must be a JSON list of strings")
    return tuple(columns)


def load_cutoff_frame(
    path: str | Path,
    cutoff_dates: Iterable[pd.Timestamp],
    columns: Sequence[str],
    *,
    positive_only: bool = False,
) -> pd.DataFrame:
    requested = tuple(columns)
    dates = tuple(pd.Timestamp(d) for d in cutoff_dates)
    if not dates:
        raise ValueError("cutoff_dates must not be empty")
    dataset = ds.dataset(Path(path))
    missing = [column for column in requested if column not in dataset.schema.names]
    if missing:
        raise ValueError(f"Unknown dataset columns: {missing}")
    expression = ds.field("cutoff_date").isin(list(dates))
    if positive_only:
        if "target_gmv_30d" in dataset.schema.names:
            expression = expression & (ds.field("target_gmv_30d") > 0)
        elif "target_nonzero" in dataset.schema.names:
            expression = expression & (ds.field("target_nonzero") == 1)
        else:
            raise ValueError("positive_only requires target_gmv_30d or target_nonzero")
    frame = dataset.to_table(filter=expression, columns=list(requested)).to_pandas()
    if "cutoff_date" in frame:
        frame["cutoff_date"] = pd.to_datetime(frame["cutoff_date"])
    for column in requested:
        if column in {"user_id", "cutoff_date"}:
            continue
        if column == "target_nonzero":
            frame[column] = frame[column].astype(np.int8)
        else:
            frame[column] = frame[column].astype(np.float32)
    return frame.loc[:, list(requested)]
