"""Atomic, fingerprinted persistence helpers for model run artifacts."""

from __future__ import annotations

import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any


class CheckpointMismatchError(ValueError):
    """Raised when an artifact was produced for another config/data fingerprint."""


def _json_default(value: Any):
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        Path(tmp_name).replace(path)
    except BaseException:
        try:
            Path(tmp_name).unlink(missing_ok=True)
        finally:
            raise


def save_checkpoint(path: Path, payload: dict[str, Any], *, fingerprint: str) -> None:
    if not fingerprint:
        raise ValueError("fingerprint must be non-empty")
    envelope = {"fingerprint": str(fingerprint), "payload": payload}
    data = json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=_json_default).encode("utf-8")
    _atomic_bytes(Path(path), data)


def load_checkpoint(path: Path, *, expected_fingerprint: str) -> dict[str, Any]:
    try:
        envelope = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CheckpointMismatchError(f"Invalid checkpoint {path}: {exc}") from exc
    if not isinstance(envelope, dict) or envelope.get("fingerprint") != expected_fingerprint:
        actual = envelope.get("fingerprint") if isinstance(envelope, dict) else None
        raise CheckpointMismatchError(f"Checkpoint fingerprint mismatch: expected {expected_fingerprint!r}, got {actual!r}")
    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        raise CheckpointMismatchError("Checkpoint payload is not a JSON object")
    return payload


def save_json(path: Path, value: Any) -> None:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, default=_json_default).encode("utf-8")
    _atomic_bytes(Path(path), data)


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# Explicit aliases make the atomic contract discoverable to notebook callers.
atomic_write_json = save_json
read_json = load_json


def save_parquet(path: Path, frame: Any, **kwargs: Any) -> None:
    """Write a parquet table through a sibling temp path and atomic replace."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    table = frame if isinstance(frame, pa.Table) else pa.Table.from_pandas(frame, preserve_index=kwargs.pop("preserve_index", False))
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(fd)
    try:
        pq.write_table(table, tmp_name, **kwargs)
        with open(tmp_name, "rb") as handle:
            os.fsync(handle.fileno())
        Path(tmp_name).replace(target)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


atomic_write_parquet = save_parquet


def load_parquet(path: Path, **kwargs: Any):
    import pandas as pd
    return pd.read_parquet(path, **kwargs)


def save_model(path: Path, estimator: Any, *, model_kind: str | None = None) -> None:
    """Persist native boosters in their native format, pickle everything else."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = (model_kind or target.suffix).lower()
    if not suffix.startswith("."):
        suffix = "." + suffix
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=target.suffix or ".tmp", dir=target.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        if suffix in {".txt", ".cbm", ".lightgbm", ".catboost"} and hasattr(estimator, "save_model"):
            estimator.save_model(str(tmp))
        else:
            import joblib
            joblib.dump(estimator, tmp)
        with open(tmp, "rb") as handle:
            os.fsync(handle.fileno())
        tmp.replace(target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def load_model(path: Path, *, model_kind: str | None = None) -> Any:
    source = Path(path)
    kind = (model_kind or source.suffix.lower().lstrip(".")).lower()
    if kind in {"txt", "lightgbm", "lgbm"}:
        module = __import__("lightgbm", fromlist=["Booster"])
        return module.Booster(model_file=str(source))
    if kind in {"cbm", "catboost", "catboost_classifier"}:
        module = __import__("catboost", fromlist=["CatBoostClassifier"])
        model = module.CatBoostClassifier()
        model.load_model(str(source))
        return model
    try:
        import joblib
        return joblib.load(source)
    except ImportError:
        with source.open("rb") as handle:
            return pickle.load(handle)


save_model_artifact = save_model
load_model_artifact = load_model
