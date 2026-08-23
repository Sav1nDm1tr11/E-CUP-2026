"""Atomic, fingerprinted persistence helpers for model run artifacts."""

from __future__ import annotations

import json
import os
import pickle
import tempfile
import hashlib
import importlib.metadata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


class CheckpointMismatchError(ValueError):
    """Raised when an artifact was produced for another config/data fingerprint."""


@dataclass(frozen=True)
class ArtifactManifest:
    """Versioned provenance required before a model bundle can be inferred."""

    version: int = 1
    config_sha256: str = ""
    feature_sha256: str = ""
    data_sha256: str = ""
    data_path: str = ""
    data_size: int | None = None
    data_mtime_ns: int | None = None
    package_versions: dict[str, str] = field(default_factory=dict)
    trained_through: str = ""
    model_names: tuple[str, ...] = ()
    weights: tuple[float, ...] = ()
    blend_history: tuple[dict[str, Any], ...] = ()
    pre_january_config: dict[str, Any] = field(default_factory=dict)
    post_january_config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ArtifactBundle:
    manifest: ArtifactManifest
    feature_names: tuple[str, ...]
    classifiers: tuple[Any, ...]
    positive_regressor: Any
    calibrator: Any = None


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default).encode("utf-8"))


def build_manifest(
    *,
    config_sha256: str,
    feature_sha256: str,
    data_sha256: str,
    trained_through: str,
    model_names: Sequence[str],
    weights: Sequence[float],
    data_path: Path | str = "",
    blend_history: Sequence[Mapping[str, Any]] = (),
    pre_january_config: Mapping[str, Any] | None = None,
    post_january_config: Mapping[str, Any] | None = None,
    package_versions: Mapping[str, str] | None = None,
) -> ArtifactManifest:
    path = Path(data_path) if data_path else None
    return ArtifactManifest(
        config_sha256=str(config_sha256), feature_sha256=str(feature_sha256), data_sha256=str(data_sha256),
        data_path=str(path or ""), data_size=path.stat().st_size if path and path.exists() else None,
        data_mtime_ns=path.stat().st_mtime_ns if path and path.exists() else None,
        package_versions=dict(package_versions or {}), trained_through=str(trained_through),
        model_names=tuple(str(name) for name in model_names), weights=tuple(float(value) for value in weights),
        blend_history=tuple(dict(item) for item in blend_history),
        pre_january_config=dict(pre_january_config or {}), post_january_config=dict(post_january_config or {}),
    )


def _manifest_dict(manifest: ArtifactManifest) -> dict[str, Any]:
    value = asdict(manifest)
    value["model_names"] = list(manifest.model_names)
    value["weights"] = list(manifest.weights)
    value["blend_history"] = [dict(item) for item in manifest.blend_history]
    return value


def save_manifest(path: Path, manifest: ArtifactManifest) -> None:
    save_json(path, _manifest_dict(manifest))


def load_manifest(path: Path, *, expected_version: int = 1) -> ArtifactManifest:
    raw = load_json(path)
    if not isinstance(raw, dict) or int(raw.get("version", -1)) != expected_version:
        raise CheckpointMismatchError("Unsupported or malformed artifact manifest version")
    return ArtifactManifest(
        version=int(raw["version"]), config_sha256=str(raw.get("config_sha256", "")),
        feature_sha256=str(raw.get("feature_sha256", "")), data_sha256=str(raw.get("data_sha256", "")),
        data_path=str(raw.get("data_path", "")), data_size=raw.get("data_size"), data_mtime_ns=raw.get("data_mtime_ns"),
        package_versions=dict(raw.get("package_versions", {})), trained_through=str(raw.get("trained_through", "")),
        model_names=tuple(raw.get("model_names", ())), weights=tuple(float(v) for v in raw.get("weights", ())),
        blend_history=tuple(dict(item) for item in raw.get("blend_history", ())),
        pre_january_config=dict(raw.get("pre_january_config", {})), post_january_config=dict(raw.get("post_january_config", {})),
    )


def validate_manifest(
    manifest: ArtifactManifest,
    *,
    config_sha256: str | None = None,
    feature_sha256: str | None = None,
    data_sha256: str | None = None,
    feature_names: Sequence[str] | None = None,
    model_names: Sequence[str] | None = None,
    weights: Sequence[float] | None = None,
) -> None:
    if manifest.version != 1 or not manifest.config_sha256 or not manifest.feature_sha256 or not manifest.data_sha256:
        raise CheckpointMismatchError("Manifest is missing required provenance hashes")
    for label, expected, actual in (("config", config_sha256, manifest.config_sha256),
                                    ("feature", feature_sha256, manifest.feature_sha256),
                                    ("data", data_sha256, manifest.data_sha256)):
        if expected is not None and expected != actual:
            raise CheckpointMismatchError(f"{label} hash mismatch")
    if feature_names is not None and sha256_json(list(feature_names)) != manifest.feature_sha256:
        raise CheckpointMismatchError("feature order/hash mismatch")
    if model_names is not None and tuple(model_names) != manifest.model_names:
        raise CheckpointMismatchError("model names do not match manifest")
    if weights is not None:
        values = tuple(float(value) for value in weights)
        if values != manifest.weights:
            raise CheckpointMismatchError("model weights do not match manifest")
    if len(manifest.model_names) != len(manifest.weights) or not manifest.model_names:
        raise CheckpointMismatchError("manifest model names/weights are not aligned")
    if any(value < 0 for value in manifest.weights) or abs(sum(manifest.weights) - 1.0) > 1e-8:
        raise CheckpointMismatchError("manifest weights must be non-negative and sum to one")


def save_bundle(path: Path, bundle: ArtifactBundle) -> None:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    save_manifest(target / "manifest.json", bundle.manifest)
    save_json(target / "bundle.json", {"feature_names": list(bundle.feature_names), "model_names": list(bundle.manifest.model_names),
                                        "weights": list(bundle.manifest.weights)})


def load_bundle_metadata(path: Path) -> dict[str, Any]:
    manifest = load_manifest(Path(path) / "manifest.json")
    raw = load_json(Path(path) / "bundle.json")
    validate_manifest(manifest, feature_names=raw.get("feature_names"), model_names=raw.get("model_names"), weights=raw.get("weights"))
    return {"manifest": manifest, **raw}


def load_bundle(
    path: Path,
    *,
    classifiers: Sequence[Any] = (),
    positive_regressor: Any = None,
    calibrator: Any = None,
) -> ArtifactBundle:
    """Load and validate bundle metadata before handing models to inference."""
    metadata = load_bundle_metadata(path)
    manifest = metadata["manifest"]
    return ArtifactBundle(
        manifest=manifest,
        feature_names=tuple(metadata["feature_names"]),
        classifiers=tuple(classifiers),
        positive_regressor=positive_regressor,
        calibrator=calibrator,
    )


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
        with open(tmp_name, "ab") as handle:
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
        if suffix in {".txt", ".lightgbm"} and getattr(estimator, "booster_", None) is not None:
            estimator.booster_.save_model(str(tmp))
        elif suffix in {".txt", ".cbm", ".lightgbm", ".catboost"} and hasattr(estimator, "save_model"):
            estimator.save_model(str(tmp))
        else:
            import joblib
            joblib.dump(estimator, tmp)
        with open(tmp, "ab") as handle:
            os.fsync(handle.fileno())
        tmp.replace(target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def load_model(path: Path, *, model_kind: str | None = None) -> Any:
    source = Path(path)
    kind = (model_kind or source.suffix.lower().lstrip(".")).lower()
    if kind in {"txt", "lightgbm", "lgbm", "lightgbm_sklearn"}:
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
