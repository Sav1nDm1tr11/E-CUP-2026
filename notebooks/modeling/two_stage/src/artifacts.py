"""Persistence helpers for model artifacts and submissions."""

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Объект типа {type(value).__name__} не поддерживает JSON")


def save_json(value: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_lightgbm_model(model: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    booster = getattr(model, "booster_", model)
    destination.write_text(booster.model_to_string(), encoding="utf-8")


def save_submission_csv(frame: pd.DataFrame, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)


def load_submission_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)
