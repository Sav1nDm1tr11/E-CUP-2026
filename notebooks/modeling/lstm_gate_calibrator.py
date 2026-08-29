"""Leak-free, temporal gate calibration for exported LSTM heads.

The module deliberately keeps the LSTM prediction untouched.  A selected
activity model only supplies a bounded logit-space delta to ``pred_log``;
therefore a zero correction is an exact identity operation.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import pickle
import tempfile
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


EPS = 1e-6
REQUIRED_HEADS = ("pred_log", "gate_prob", "positive_log", "direct_log", "hurdle_log")
META_COLUMNS = ("user_id", "cutoff", "target", "target_active", "source", "seed")
FORBIDDEN_FEATURES = {
    "user_id", "target", "target_active", "residual", "cutoff", "source", "seed",
    "absolute_cutoff", "fold", "provenance",
}


@dataclass(frozen=True)
class CalibrationSearchConfig:
    """Small, explicit search budget and all temporal/guardrail policy knobs."""

    max_trials: int = 8
    random_state: int = 42
    fit_cutoff: str = "2025-11-01"
    validation_cutoff: str = "2025-12-01"
    audit_cutoff: str = "2026-01-01"
    early_stopping_rounds: int = 40
    n_estimators: int = 400
    max_depth: int = 4
    learning_rate: float = 0.03
    correction_weight: float = 1.0
    max_ratio: float = 2.0
    min_improvement: float = 0.0
    max_audit_regression: float = 0.0
    min_better_folds: int = 1
    candidates: tuple[str, ...] = ("identity", "platt", "beta", "lightgbm", "random_forest")

    def __post_init__(self) -> None:
        if int(self.max_trials) < 1:
            raise ValueError("max_trials must be positive")
        if int(self.early_stopping_rounds) < 1:
            raise ValueError("early_stopping_rounds must be positive")
        if not 0.0 <= float(self.correction_weight) <= 1.0:
            raise ValueError("correction_weight must be in [0, 1]")
        if float(self.max_ratio) < 1.0:
            raise ValueError("max_ratio must be at least one")


def _finite(values: Any, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0 or not np.isfinite(arr).all():
        raise ValueError(f"{name} must be non-empty and finite")
    return arr


def _clip_probability(values: Any) -> np.ndarray:
    p = _finite(values, "probability")
    if (p < 0).any() or (p > 1).any():
        raise ValueError("probability must be in [0, 1]")
    return np.clip(p, EPS, 1.0 - EPS)


def _logit(values: Any) -> np.ndarray:
    p = _clip_probability(values)
    return np.log(p) - np.log1p(-p)


def _sigmoid(values: Any) -> np.ndarray:
    x = np.clip(np.asarray(values, dtype=np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def _target_active(frame: pd.DataFrame) -> np.ndarray:
    if "target_active" in frame:
        target = frame["target_active"].to_numpy()
    elif "target" in frame:
        target = (frame["target"].to_numpy(dtype=float) > 0).astype(np.int8)
    else:
        raise ValueError("meta frame needs target or target_active")
    if len(target) == 0 or set(np.unique(target).tolist()) - {0, 1, False, True}:
        raise ValueError("target_active must be binary")
    return np.asarray(target, dtype=np.int8)


def validate_meta_frame(frame: pd.DataFrame, *, require_target: bool = False) -> tuple[str, ...]:
    """Validate the metadata/feature contract and return ordered feature names."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("meta frame must be a non-empty DataFrame")
    if "user_id" not in frame or frame["user_id"].isna().any():
        raise ValueError("meta frame requires non-null user_id")
    # Temporal OOF tables legitimately contain one row per user and cutoff;
    # reject only duplicate observations at the same cutoff/seed.
    key_columns = ["user_id"]
    if "cutoff" in frame:
        key_columns.append("cutoff")
    if "seed" in frame:
        key_columns.append("seed")
    if frame.duplicated(key_columns).any():
        raise ValueError("duplicate user/cutoff/seed observations")
    if require_target and not ({"target", "target_active"} & set(frame.columns)):
        raise ValueError("meta frame requires target")
    names = tuple(frame.attrs.get("feature_names", ()))
    if not names:
        names = tuple(c for c in frame.columns if c not in META_COLUMNS and c not in FORBIDDEN_FEATURES)
    if not names:
        raise ValueError("meta frame has no model features")
    forbidden = FORBIDDEN_FEATURES.intersection(names)
    if forbidden:
        raise ValueError(f"forbidden feature(s): {sorted(forbidden)}")
    missing = [c for c in names if c not in frame]
    if missing:
        raise ValueError(f"feature columns missing: {missing}")
    for name in names:
        if not pd.api.types.is_numeric_dtype(frame[name]):
            raise TypeError(f"feature {name!r} must be numeric")
        values = frame[name].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"feature {name!r} must be finite")
    return names


def _static_frame(static_snapshot: Any, expected_ids: np.ndarray, static_feature_names: Sequence[str] | None) -> pd.DataFrame:
    if static_snapshot is None:
        return pd.DataFrame(index=np.arange(len(expected_ids)))
    if isinstance(static_snapshot, (str, Path)):
        path = Path(static_snapshot)
        if path.suffix.lower() in {".csv", ".parquet"}:
            static_snapshot = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_parquet(path)
        else:
            static_snapshot = np.load(path, mmap_mode="r")
    if isinstance(static_snapshot, Mapping):
        static_snapshot = pd.DataFrame(static_snapshot)
    elif not isinstance(static_snapshot, pd.DataFrame):
        values = np.asarray(static_snapshot)
        if values.ndim != 2 or values.shape[0] != len(expected_ids):
            raise ValueError("static snapshot row count/order does not match component IDs")
        names = tuple(static_feature_names or (f"static_{i}" for i in range(values.shape[1])))
        return pd.DataFrame(values, columns=names)
    static = static_snapshot.copy()
    if "user_id" in static:
        ids = static["user_id"].to_numpy()
        if not np.array_equal(ids, expected_ids):
            raise ValueError("static snapshot user order does not match component order")
        static = static.drop(columns=["user_id"])
    elif len(static) != len(expected_ids):
        raise ValueError("static snapshot row count does not match component IDs")
    return static.reset_index(drop=True)


def build_meta_frame(
    components: pd.DataFrame,
    static_snapshot: Any = None,
    *,
    static_feature_names: Sequence[str] | None = None,
    expected_user_ids: Sequence[Any] | None = None,
) -> pd.DataFrame:
    """Create deterministic meta-features while retaining only safe provenance columns."""
    if not isinstance(components, pd.DataFrame):
        raise TypeError("components must be a DataFrame")
    missing = [name for name in REQUIRED_HEADS if name not in components]
    if missing:
        raise ValueError(f"component columns missing: {missing}")
    if "user_id" not in components:
        raise ValueError("components require user_id")
    result = components.copy().reset_index(drop=True)
    ids = result["user_id"].to_numpy()
    if expected_user_ids is not None and not np.array_equal(ids, np.asarray(expected_user_ids)):
        raise ValueError("component user order does not match expected user order")
    static = _static_frame(static_snapshot, ids, static_feature_names)
    for name in static.columns:
        if name in FORBIDDEN_FEATURES or name in result.columns:
            raise ValueError(f"forbidden or duplicate static feature: {name}")
        if not pd.api.types.is_numeric_dtype(static[name]):
            raise TypeError(f"static feature {name!r} must be numeric")
    feature_names = ["lstm_pred_log", "lstm_gate_prob", "lstm_positive_log", "lstm_direct_log", "lstm_hurdle_log"]
    result[feature_names] = result[list(REQUIRED_HEADS)].to_numpy(dtype=np.float64)
    if "history_length" in result:
        feature_names.append("history_length")
    result = pd.concat([result, static], axis=1)
    feature_names.extend(static.columns.tolist())
    result.attrs["feature_names"] = tuple(feature_names)
    validate_meta_frame(result)
    return result


class IdentityGateCalibrator:
    name = "identity"

    def fit(self, X: Any, y: Any = None) -> "IdentityGateCalibrator":
        self.feature_names_ = tuple(X.attrs.get("feature_names", ())) if isinstance(X, pd.DataFrame) else ()
        return self

    def predict_probability(self, X: Any) -> np.ndarray:
        values = (X["lstm_gate_prob"] if "lstm_gate_prob" in X else X["gate_prob"]) if isinstance(X, pd.DataFrame) else np.asarray(X)[:, 1]
        return _clip_probability(values)

    def predict_proba(self, X: Any) -> np.ndarray:
        p = self.predict_probability(X)
        return np.column_stack((1.0 - p, p))


class PlattGateCalibrator:
    """Dependency-free Platt calibration of the exported gate probability."""

    name = "platt"

    def fit(self, X: Any, y: Any) -> "PlattGateCalibrator":
        p = _clip_probability((X["lstm_gate_prob"] if "lstm_gate_prob" in X else X["gate_prob"]) if isinstance(X, pd.DataFrame) else X)
        y = np.asarray(y, dtype=float).reshape(-1)
        if len(p) != len(y) or set(np.unique(y).tolist()) != {0.0, 1.0}:
            raise ValueError("Platt calibration requires both binary classes")
        z = _logit(p)
        design = np.column_stack((np.ones(len(z)), z))
        coef = np.zeros(2, dtype=float)
        for _ in range(60):
            q = _sigmoid(design @ coef)
            hessian = design.T @ (design * (q * (1.0 - q))[:, None]) + np.eye(2) * 1e-6
            step = np.linalg.solve(hessian, design.T @ (q - y))
            coef -= step
            if np.max(np.abs(step)) < 1e-7:
                break
        self.intercept_, self.coef_ = float(coef[0]), float(coef[1])
        self.feature_names_ = tuple(X.attrs.get("feature_names", ())) if isinstance(X, pd.DataFrame) else ()
        return self

    def predict_probability(self, X: Any) -> np.ndarray:
        p = _clip_probability((X["lstm_gate_prob"] if "lstm_gate_prob" in X else X["gate_prob"]) if isinstance(X, pd.DataFrame) else X)
        return _sigmoid(self.intercept_ + self.coef_ * _logit(p))

    def predict_proba(self, X: Any) -> np.ndarray:
        p = self.predict_probability(X)
        return np.column_stack((1.0 - p, p))


class BetaGateCalibrator(PlattGateCalibrator):
    """Beta-calibration baseline using log(p) and -log(1-p) features."""

    name = "beta"

    def fit(self, X: Any, y: Any) -> "BetaGateCalibrator":
        p = _clip_probability((X["lstm_gate_prob"] if "lstm_gate_prob" in X else X["gate_prob"]) if isinstance(X, pd.DataFrame) else X)
        target = np.asarray(y, dtype=float).reshape(-1)
        if len(p) != len(target) or set(np.unique(target).tolist()) != {0.0, 1.0}:
            raise ValueError("Beta calibration requires both binary classes")
        design = np.column_stack((np.ones(len(p)), np.log(p), -np.log1p(-p)))
        coef = np.zeros(3, dtype=float)
        for _ in range(60):
            q = _sigmoid(design @ coef)
            hessian = design.T @ (design * (q * (1.0 - q))[:, None]) + np.eye(3) * 1e-6
            step = np.linalg.solve(hessian, design.T @ (q - target))
            coef -= step
            if np.max(np.abs(step)) < 1e-7:
                break
        self.intercept_, self.coef_ = float(coef[0]), coef[1:]
        self.feature_names_ = tuple(X.attrs.get("feature_names", ())) if isinstance(X, pd.DataFrame) else ()
        return self

    def predict_probability(self, X: Any) -> np.ndarray:
        p = _clip_probability((X["lstm_gate_prob"] if "lstm_gate_prob" in X else X["gate_prob"]) if isinstance(X, pd.DataFrame) else X)
        return _sigmoid(self.intercept_ + self.coef_[0] * np.log(p) + self.coef_[1] * -np.log1p(-p))


def _predict_probability(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_probability"):
        return _clip_probability(model.predict_probability(X))
    names = tuple(getattr(model, "feature_names_", X.attrs.get("feature_names", ())))
    values = X[list(names)] if names and set(names).issubset(X.columns) else X
    if hasattr(model, "predict_proba"):
        raw = np.asarray(model.predict_proba(values))
        return _clip_probability(raw[:, 1] if raw.ndim == 2 else raw)
    raw = np.asarray(model.predict(values)).reshape(-1)
    return _clip_probability(raw)


def _corrected_log(frame: pd.DataFrame, model: Any, correction_weight: float, max_ratio: float) -> np.ndarray:
    baseline = _finite(frame["pred_log"], "pred_log")
    gate = _clip_probability(frame["gate_prob"])
    calibrated = _predict_probability(model, frame)
    # Bound the gate movement in logit space first, then retain the existing
    # proxy-gate composition: prediction logs are scaled by the bounded
    # calibrated/original activity-probability ratio.
    logit_delta = np.clip(_logit(calibrated) - _logit(gate), -np.log(max_ratio), np.log(max_ratio))
    corrected_gate = _sigmoid(_logit(gate) + logit_delta)
    ratio = np.clip(corrected_gate / gate, 1.0 / max_ratio, max_ratio)
    return np.clip(baseline * np.exp(float(correction_weight) * np.log(ratio)), 0.0, None)


def _rmsle(frame: pd.DataFrame, prediction_log: np.ndarray) -> float:
    if "target_log" in frame:
        target = frame["target_log"].to_numpy(dtype=float)
    elif "target" in frame:
        target = np.log1p(np.clip(frame["target"].to_numpy(dtype=float), 0.0, None))
    else:
        raise ValueError("scoring requires target or target_log")
    return float(np.sqrt(np.mean((target - prediction_log) ** 2)))


def _cutoff_parts(frame: pd.DataFrame, config: CalibrationSearchConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "cutoff" not in frame:
        raise ValueError("temporal calibration requires cutoff")
    labels = frame["cutoff"].astype(str)
    try:
        temporal = pd.to_datetime([config.fit_cutoff, config.validation_cutoff, config.audit_cutoff])
    except (TypeError, ValueError):
        temporal = None
    if temporal is not None and not (temporal[0] < temporal[1] < temporal[2]):
        raise ValueError("fit, validation, and audit cutoffs must be chronological")
    fit = frame[labels == str(config.fit_cutoff)]
    validation = frame[labels == str(config.validation_cutoff)]
    audit = frame[labels == str(config.audit_cutoff)]
    if fit.empty or validation.empty or audit.empty:
        ordered = sorted(labels.unique())
        if len(ordered) < 3:
            raise ValueError("at least three chronological cutoffs are required")
        fit, validation, audit = (frame[labels == ordered[i]] for i in range(3))
    return fit, validation, audit


def _make_lightgbm(config: CalibrationSearchConfig, seed: int) -> Any:
    import lightgbm as lgb  # type: ignore
    return lgb.LGBMClassifier(
        n_estimators=int(config.n_estimators), learning_rate=float(config.learning_rate),
        max_depth=int(config.max_depth), random_state=seed, n_jobs=-1, verbosity=-1,
    )


def _fit_candidate(name: str, fit: pd.DataFrame, validation: pd.DataFrame, feature_names: Sequence[str], config: CalibrationSearchConfig) -> Any:
    X_fit, X_val = fit[list(feature_names)], validation[list(feature_names)]
    y_fit, y_val = _target_active(fit), _target_active(validation)
    if name == "identity":
        return IdentityGateCalibrator().fit(X_fit, y_fit)
    if name == "platt":
        return PlattGateCalibrator().fit(fit, y_fit)
    if name == "beta":
        return BetaGateCalibrator().fit(fit, y_fit)
    if name == "lightgbm":
        model = _make_lightgbm(config, config.random_state)
        import lightgbm as lgb  # type: ignore
        model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], eval_metric="binary_logloss",
                  callbacks=[lgb.early_stopping(config.early_stopping_rounds, verbose=False)])
        model.best_iteration_ = int(getattr(model, "best_iteration_", config.n_estimators) or config.n_estimators)
        return model
    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier  # type: ignore
        model = RandomForestClassifier(n_estimators=min(300, int(config.n_estimators)), max_depth=config.max_depth,
                                       random_state=config.random_state, n_jobs=-1)
        model.fit(X_fit, y_fit)
        model.feature_names_ = tuple(feature_names)
        return model
    raise ValueError(f"unknown calibrator candidate: {name}")


def select_temporal_calibrator(meta_frame: pd.DataFrame, config: CalibrationSearchConfig | None = None) -> dict[str, Any]:
    """Bounded chronological search; January is only an audit/guardrail slice."""
    config = config or CalibrationSearchConfig()
    feature_names = validate_meta_frame(meta_frame, require_target=True)
    fit, validation, audit = _cutoff_parts(meta_frame, config)
    baseline_val = _rmsle(validation, validation["pred_log"].to_numpy(dtype=float))
    baseline_audit = _rmsle(audit, audit["pred_log"].to_numpy(dtype=float))
    rows: list[dict[str, Any]] = []
    models: dict[str, Any] = {}
    candidates = list(dict.fromkeys(config.candidates))[: int(config.max_trials)]
    if "identity" not in candidates:
        candidates.insert(0, "identity")
    for name in candidates[: int(config.max_trials)]:
        try:
            model = _fit_candidate(name, fit, validation, feature_names, config)
            val_pred = _corrected_log(validation, model, config.correction_weight, config.max_ratio)
            audit_pred = _corrected_log(audit, model, config.correction_weight, config.max_ratio)
            val_score, audit_score = _rmsle(validation, val_pred), _rmsle(audit, audit_pred)
            row = {"name": name, "validation_rmsle": val_score, "audit_rmsle": audit_score,
                   "delta": val_score - baseline_val, "audit_delta": audit_score - baseline_audit,
                   "guardrails_pass": audit_score <= baseline_audit + config.max_audit_regression}
            rows.append(row); models[name] = model
        except (ImportError, ModuleNotFoundError):
            continue
    if not rows:
        raise RuntimeError("no calibration candidate could be fitted")
    eligible = [r for r in rows if r["guardrails_pass"] and r["delta"] < -config.min_improvement]
    selected = min(eligible, key=lambda r: (r["validation_rmsle"], r["audit_rmsle"])) if eligible else next(r for r in rows if r["name"] == "identity")
    if selected["name"] != "identity" and selected["delta"] >= -config.min_improvement:
        selected = next(r for r in rows if r["name"] == "identity")
    selected["best_iteration"] = getattr(models[selected["name"]], "best_iteration_", None)
    return {"selected_name": selected["name"], "selected_model": models[selected["name"]],
            "feature_names": tuple(feature_names), "candidates": rows, "trials": len(rows),
            "selection_cutoff": str(config.validation_cutoff), "audit_cutoff": str(config.audit_cutoff),
            "guardrails_pass": bool(selected["guardrails_pass"] and selected["name"] != "identity"),
            "baseline_validation_rmsle": baseline_val, "baseline_audit_rmsle": baseline_audit,
            "config": config}


def build_temporal_splits(meta_frame: pd.DataFrame, config: CalibrationSearchConfig | None = None) -> dict[str, pd.DataFrame]:
    """Return named chronological slices for callers that need inspection/testing."""
    config = config or CalibrationSearchConfig()
    fit, validation, audit = _cutoff_parts(meta_frame, config)
    return {"fit": fit.copy(), "validation": validation.copy(), "audit": audit.copy()}


def fit_production_calibrator(meta_frame: pd.DataFrame, selection: Mapping[str, Any] | None = None,
                              config: CalibrationSearchConfig | None = None) -> Any:
    """Refit the chosen activity model on November+December, excluding January."""
    config = config or (selection.get("config") if selection else None) or CalibrationSearchConfig()
    feature_names = validate_meta_frame(meta_frame, require_target=True)
    fit, validation, _ = _cutoff_parts(meta_frame, config)
    train = pd.concat([fit, validation], ignore_index=True)
    name = str(selection.get("selected_name", "identity")) if selection else "identity"
    if name == "identity":
        return IdentityGateCalibrator().fit(train[list(feature_names)], _target_active(train))
    # Production fitting uses validation as eval_set for LightGBM, preserving callback semantics.
    # Reuse the selected early-stopping tree count when the search supplied it.
    if name == "lightgbm" and selection and selection.get("best_iteration"):
        config = replace(config, n_estimators=max(1, int(selection["best_iteration"])))
    return _fit_candidate(name, train, validation, feature_names, config)


def apply_calibrator_per_seed(inference_components: pd.DataFrame, calibrator: Any = None, *,
                              correction_weight: float = 1.0, max_ratio: float = 2.0,
                              config: CalibrationSearchConfig | None = None,
                              static_snapshot: Any = None,
                              static_feature_names: Sequence[str] | None = None) -> pd.DataFrame:
    """Apply one calibrator independently to each seed and average corrected logs."""
    # Accept both natural call forms: (components, calibrator) and
    # (calibrator, components), which is convenient for notebook orchestration.
    if not isinstance(inference_components, pd.DataFrame) and isinstance(calibrator, pd.DataFrame):
        inference_components, calibrator = calibrator, inference_components
    if isinstance(calibrator, Mapping):
        calibrator = calibrator.get("selected_model", calibrator.get("calibrator"))
    if config is not None:
        correction_weight, max_ratio = config.correction_weight, config.max_ratio
    if not 0.0 <= float(correction_weight) <= 1.0 or float(max_ratio) < 1.0:
        raise ValueError("invalid correction bounds")
    missing = [name for name in ("user_id", "pred_log", "gate_prob") if name not in inference_components]
    if missing:
        raise ValueError(f"inference component columns missing: {missing}")
    result = inference_components.copy().reset_index(drop=True)
    model_names = tuple(getattr(calibrator, "feature_names_", ())) if calibrator is not None else ()
    if model_names and not set(model_names).issubset(result.columns) and "lstm_gate_prob" not in result:
        result = build_meta_frame(result, static_snapshot, static_feature_names=static_feature_names)
    seed_column = result["seed"] if "seed" in result else pd.Series(np.zeros(len(result), dtype=int))
    corrected = np.empty(len(result), dtype=float)
    for _, indices in seed_column.groupby(seed_column).groups.items():
        part = result.loc[indices]
        if float(correction_weight) == 0.0:
            corrected[np.asarray(indices)] = _finite(part["pred_log"], "pred_log")
        else:
            corrected[np.asarray(indices)] = _corrected_log(part, calibrator or IdentityGateCalibrator(), correction_weight, max_ratio)
    result["corrected_log"] = corrected
    result["prediction_log"] = result.groupby("user_id")["corrected_log"].transform("mean")
    result["corrected_prediction"] = np.expm1(result["prediction_log"].clip(lower=0.0))
    result.attrs["feature_names"] = tuple(getattr(calibrator, "feature_names_", ()))
    return result


def _sha256(value: Any) -> str:
    payload = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def save_calibrator_bundle(path: str | Path, calibrator: Any, *, config: CalibrationSearchConfig | None = None,
                           feature_names: Sequence[str] = (), selection_report: Mapping[str, Any] | None = None,
                           correction_parameters: Mapping[str, Any] | None = None,
                           cutoff_report: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Persist model and provenance in a versioned, atomically-written bundle."""
    target = Path(path); target.mkdir(parents=True, exist_ok=True)
    config = config or CalibrationSearchConfig()
    names = tuple(feature_names or getattr(calibrator, "feature_names_", ()))
    manifest = {"version": 1, "feature_names": list(names), "feature_sha256": _sha256(list(names)),
                "config": asdict(config), "config_sha256": _sha256(asdict(config)),
                "selection": dict(selection_report or {}), "correction_parameters": dict(correction_parameters or {}),
                "cutoff_report": dict(cutoff_report or {}),
                "package_versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas") if _has_package(name)}}
    _atomic_write(target / "calibrator.pkl", pickle.dumps(calibrator, protocol=pickle.HIGHEST_PROTOCOL))
    _atomic_write(target / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True, default=str).encode())
    return manifest


def _has_package(name: str) -> bool:
    try:
        importlib.metadata.version(name); return True
    except importlib.metadata.PackageNotFoundError:
        return False


def _atomic_write(path: Path, payload: bytes) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        Path(temporary).replace(path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def validate_calibrator_bundle(path: str | Path, *, expected_feature_names: Sequence[str] | None = None) -> dict[str, Any]:
    target = Path(path)
    if not (target / "manifest.json").is_file() or not (target / "calibrator.pkl").is_file():
        raise ValueError("calibrator bundle requires manifest.json and calibrator.pkl")
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("version") != 1 or not isinstance(manifest.get("feature_names"), list):
        raise ValueError("unsupported or malformed calibrator bundle")
    names = tuple(manifest["feature_names"])
    if len(names) != len(set(names)) or FORBIDDEN_FEATURES.intersection(names):
        raise ValueError("invalid feature contract in calibrator bundle")
    if _sha256(list(names)) != manifest.get("feature_sha256"):
        raise ValueError("feature hash mismatch")
    if expected_feature_names is not None and names != tuple(expected_feature_names):
        raise ValueError("feature names do not match bundle")
    return manifest


def validate_artifact_bundle(path: str | Path, **kwargs: Any) -> dict[str, Any]:
    return validate_calibrator_bundle(path, **kwargs)


validate_artifact_manifest = validate_calibrator_bundle


__all__ = ["CalibrationSearchConfig", "build_meta_frame", "validate_meta_frame", "select_temporal_calibrator",
           "build_temporal_splits",
           "fit_production_calibrator", "apply_calibrator_per_seed", "save_calibrator_bundle",
           "validate_calibrator_bundle", "validate_artifact_bundle", "validate_artifact_manifest",
           "IdentityGateCalibrator", "PlattGateCalibrator", "BetaGateCalibrator"]
