"""Structural contract for the automatic calibration section in 07_LSTM."""

from __future__ import annotations

import json
from pathlib import Path


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


def test_final_inference_exports_components_for_each_seed() -> None:
    source = _joined()
    assert "inference_components_by_seed.parquet" in source
    assert "hurdle_weight" in source
    assert "seed" in source
    assert "np.allclose" in source or "allclose" in source
    assert "FINAL_SEEDS" in source
    assert "expected_training_signature" in source


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
    assert "guardrail" in tail.lower()


def test_calibration_does_not_depend_on_recovery_notebook() -> None:
    source = _joined()
    section = _position(source, "Automatic temporal ML gate calibration")
    assert "07_LSTM_OOF_recovery.ipynb" not in source[section:]
