"""Input data helpers for the two-stage model."""

from pathlib import Path

import pandas as pd


def find_project_root(start: str | Path) -> Path:
    """Find the nearest ancestor containing ``data/Prepared_data.parquet``."""
    current = Path(start).expanduser().resolve()
    if current.is_file():
        current = current.parent
    for directory in (current, *current.parents):
        if (directory / "data" / "Prepared_data.parquet").is_file():
            return directory
    raise FileNotFoundError("Не найден маркер проекта: data/Prepared_data.parquet")


def load_expected_user_ids(path: str | Path) -> pd.Series:
    """Load submission user IDs in file order and reject duplicates."""
    frame = pd.read_csv(path)
    if "user_id" not in frame.columns:
        raise ValueError("В CSV отсутствует колонка user_id")
    user_ids = frame["user_id"]
    if user_ids.duplicated().any():
        raise ValueError("В CSV обнаружены дубликаты user_id")
    return user_ids
