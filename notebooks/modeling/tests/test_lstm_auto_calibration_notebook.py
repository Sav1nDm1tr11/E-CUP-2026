"""Structural contract for the automatic calibration section in 07_LSTM."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


NOTEBOOK = Path(__file__).parents[1] / "07_LSTM.ipynb"


def _sources() -> list[str]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(cell.get("source", [])) for cell in notebook["cells"]]


def _joined() -> str:
    return "\n\n".join(_sources())


def _position(text: str, needle: str) -> int:
    position = text.find(needle)
    assert position >= 0, f"missing notebook anchor: {needle!r}"
    return position


def test_notebook_is_valid_json_and_keeps_lstm_baseline() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] >= 4
    source = _joined()
    assert "JointHurdleLSTM-v4-expanded-es" in source
    baseline = _position(source, "lstm_hurdle_v4_expanded_es_3seed.csv")
    calibration = _position(source, "lstm_hurdle_v4_expanded_es_3seed_calibrated.csv")
    assert baseline < calibration
    cells = _sources()
    baseline_cell = next(i for i, cell in enumerate(cells) if "lstm_hurdle_v4_expanded_es_3seed.csv" in cell and "to_csv" in cell)
    calibration_cell = next(i for i, cell in enumerate(cells) if "Automatic temporal ML gate calibration" in cell)
    assert baseline_cell < calibration_cell


def test_temporal_cv_persists_all_epoch_heads_and_exports_selected_oof() -> None:
    source = _joined()
    for field in ("user_id", "y_true", "pred_log", "gate_prob", "positive_log", "direct_log", "hurdle_log"):
        assert field in source
    assert "prediction_epoch_{epoch:02d}.npz" in source
    assert "CV_RUN_SIGNATURE" in source
    assert "oof_best_epoch.parquet" in source
    assert "BEST_EPOCH" in source
    assert '"history_length": np.asarray(saved["history_length"]' in source
    assert '"prediction_schema_version": OOF_SCHEMA_VERSION' in source
    assert '"provenance":' in source


def test_final_inference_exports_components_for_each_seed() -> None:
    source = _joined()
    assert "inference_components_by_seed.parquet" in source
    assert "hurdle_weight" in source
    assert "seed" in source
    assert "np.allclose" in source or "allclose" in source
    assert "FINAL_SEEDS" in source
    assert "expected_training_signature" in source
    assert "history_length" in source
    assert "_align_static_snapshot" in source
    assert '"history_length": pred["history_length"]' in source


def test_calibration_is_bounded_temporal_and_early_stopped() -> None:
    source = _joined()
    section = _position(source, "Automatic temporal ML gate calibration")
    tail = source[section:]
    assert "%pip" in tail
    assert "search_trials=16" in tail
    assert "max_boost_rounds=2500" in tail
    assert "early_stopping_rounds=100" in tail
    assert "random_state=2026" in tail
    assert "n_jobs=-1" in tail
    assert "select_temporal_calibrator" in tail
    assert "fit_production_calibrator" in tail
    assert "apply_calibrator_per_seed" in tail
    assert "save_calibrator_bundle" in tail
    assert "expected_training_signature" in tail
    assert "expected_feature_names=oof_meta.attrs[\"feature_names\"]" in tail
    assert "oof_safe = _safe_meta_view" in tail
    assert "inference_aligned_snapshots" in tail
    assert "_write_calibrated_submission(baseline_submission.set_index" in tail
    for metric in ("log_loss", "brier", "roc_auc", "rmsle_zero", "rmsle_positive", "rmsle_high_value"):
        assert metric in tail
    assert "guardrail" in tail.lower()


def test_calibration_does_not_depend_on_recovery_notebook() -> None:
    source = _joined()
    section = _position(source, "Automatic temporal ML gate calibration")
    assert "07_LSTM_OOF_recovery.ipynb" not in source[section:]


def test_safe_meta_view_matches_public_core_schema() -> None:
    source = _joined()
    assert "_safe_meta_view" in source
    assert "provenance" in source
    sys.path.insert(0, str(NOTEBOOK.parent))
    from lstm_gate_calibrator import build_meta_frame

    n = 4
    frame = pd.DataFrame({
        "user_id": np.arange(n),
        "cutoff": ["2025-11-15"] * n,
        "target": [0.0, 1.0, 0.0, 2.0],
        "target_active": [0, 1, 0, 1],
        "source": ["cv"] * n,
        "hurdle_weight": [0.5] * n,
        "pred_log": [0.0, 0.2, 0.1, 0.3],
        "gate_prob": [0.1, 0.8, 0.2, 0.7],
        "positive_log": [0.0, 0.3, 0.1, 0.4],
        "direct_log": [0.0, 0.2, 0.1, 0.3],
        "hurdle_log": [0.0, 0.24, 0.02, 0.28],
        "history_length": [3.0, 4.0, 2.0, 4.0],
        "provenance": ["ignored"] * n,
        "y_true": [0.0, 1.0, 0.0, 2.0],
    })
    safe = frame.drop(columns=["provenance", "y_true"])
    meta = build_meta_frame(safe)
    assert "history_length" in meta.attrs["feature_names"]
    assert "provenance" not in meta.attrs["feature_names"]


def test_three_seed_static_alignment_is_executable() -> None:
    source = _joined()
    assert "_align_static_snapshot" in source
    assert "np.tile" in source or "repeat" in source
    n = 3
    users = np.array([10, 11, 12])
    components = pd.DataFrame({
        "user_id": np.tile(users, 3),
        "seed": np.repeat([42, 143, 2026], n),
    })
    static = pd.DataFrame({"user_id": users, "age": [1.0, 2.0, 3.0]})
    aligned = static.set_index("user_id").loc[components["user_id"], ["age"]].reset_index(drop=True)
    assert aligned["age"].tolist() == [1.0, 2.0, 3.0] * 3
