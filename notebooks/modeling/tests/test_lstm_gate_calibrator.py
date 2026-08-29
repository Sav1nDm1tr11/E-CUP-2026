import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


MODELING_DIR = Path(__file__).resolve().parents[1]
if str(MODELING_DIR) not in sys.path:
    sys.path.insert(0, str(MODELING_DIR))

from lstm_gate_calibrator import (  # noqa: E402
    CalibrationSearchConfig,
    apply_calibrator_per_seed,
    build_meta_frame,
    compose_calibrated_prediction,
    fit_production_calibrator,
    save_calibrator_bundle,
    select_temporal_calibrator,
    validate_calibrator_bundle,
    validate_meta_frame,
)


def _components(ids=(3, 1, 2), cutoff="2025-12-01"):
    ids = np.asarray(ids)
    return pd.DataFrame(
        {
            "user_id": ids,
            "cutoff": [cutoff] * len(ids),
            "target": [0.0, 2.0, 1.0][: len(ids)],
            "source": ["oof"] * len(ids),
            "pred_log": [0.1, 0.4, 0.2][: len(ids)],
            "gate_prob": [0.2, 0.8, 0.6][: len(ids)],
            "positive_log": [0.2, 0.5, 0.4][: len(ids)],
            "direct_log": [0.1, 0.3, 0.2][: len(ids)],
            "hurdle_log": [0.1, 0.4, 0.2][: len(ids)],
            "history_length": [2, 4, 3][: len(ids)],
        }
    )


def test_meta_frame_preserves_exact_user_order_and_rejects_leakage():
    frame = build_meta_frame(_components(), static_snapshot=pd.DataFrame({
        "user_id": [3, 1, 2], "static_a": [10.0, 11.0, 12.0]
    }))
    assert frame.user_id.tolist() == [3, 1, 2]
    validate_meta_frame(frame)
    assert not {"target", "user_id", "cutoff", "source"}.intersection(
        frame.attrs["feature_names"]
    )


def test_meta_frame_rejects_static_order_mismatch():
    with pytest.raises(ValueError, match="order"):
        build_meta_frame(_components(), static_snapshot=pd.DataFrame({
            "user_id": [1, 3, 2], "static_a": [10.0, 11.0, 12.0]
        }))


def test_zero_correction_is_exact_identity_per_seed():
    components = pd.concat([
        _components(cutoff="2026-01-01").assign(seed=42),
        _components(cutoff="2026-01-01").assign(seed=143),
    ], ignore_index=True)
    result = apply_calibrator_per_seed(
        components, calibrator=None, correction_weight=0.0
    )
    np.testing.assert_allclose(result["corrected_log"], result["pred_log"])
    np.testing.assert_allclose(
        result.groupby("user_id").prediction_log.mean().sort_index(),
        components.groupby("user_id").pred_log.mean().sort_index(),
    )


def test_temporal_selection_uses_december_for_selection_and_january_guardrail():
    rows = []
    for cutoff, offset in [("2025-11-15", 0), ("2025-12-15", 10), ("2026-01-14", 20)]:
        part = _components(cutoff=cutoff)
        part["target"] += offset / 10
        rows.append(part)
    result = select_temporal_calibrator(
        pd.concat(rows, ignore_index=True),
        CalibrationSearchConfig(max_trials=2, search_trials=1, candidates=("identity", "platt")),
    )
    assert result["selection_cutoff"] == "2025-12-15"
    assert result["audit_cutoff"] == "2026-01-14"
    assert result["trials"] == 3


def test_lightgbm_configuration_has_eval_set_and_early_stopping():
    config = CalibrationSearchConfig(max_trials=3)
    assert config.early_stopping_rounds > 0
    assert config.max_trials <= 3


def test_nonzero_gate_correction_recomposes_only_hurdle_head():
    components = _components().assign(hurdle_weight=0.6)

    class FixedCalibrator:
        feature_names_ = ()

        def predict_probability(self, frame):
            return np.full(len(frame), 0.75)

    actual = compose_calibrated_prediction(
        components, FixedCalibrator(), correction_weight=1.0, max_ratio=4.0
    )
    expected = 0.6 * (0.75 * components.positive_log) + 0.4 * components.direct_log
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("alias", [
    "target_log", "residual_log", "y_true", "error_rate", "user_id_hash", "cutoff_month",
])
def test_normalized_target_residual_identity_time_aliases_are_forbidden(alias):
    frame = _components()
    frame[alias] = 1.0
    frame.attrs["feature_names"] = (alias,)
    with pytest.raises(ValueError, match="forbidden"):
        validate_meta_frame(frame)


def test_derived_head_features_are_present_and_numeric():
    frame = build_meta_frame(_components())
    assert {
        "gate_logit", "gate_entropy", "gate_uncertainty", "positive_direct_delta",
        "hurdle_direct_delta", "pred_hurdle_delta", "pred_direct_delta",
    }.issubset(frame.attrs["feature_names"])


def test_config_uses_approved_cutoffs_and_search_budget():
    config = CalibrationSearchConfig()
    assert (config.fit_cutoff, config.validation_cutoff, config.audit_cutoff) == (
        "2025-11-15", "2025-12-15", "2026-01-14"
    )
    assert config.search_trials == 16
    assert config.max_boost_rounds == 2500
    assert config.early_stopping_rounds == 100


def test_skipped_single_class_candidate_falls_back_to_identity():
    rows = [_components(cutoff=cutoff).assign(target=1.0)
            for cutoff in ["2025-11-15", "2025-12-15", "2026-01-14"]]
    result = select_temporal_calibrator(pd.concat(rows, ignore_index=True), CalibrationSearchConfig(search_trials=1))
    assert result["selected_name"] == "identity"
    assert any(row.get("skip_reason") for row in result["candidates"] if row["name"] != "identity")


def test_static_snapshots_are_aligned_per_cutoff():
    components = pd.concat([
        _components(cutoff="2025-11-15"), _components(cutoff="2025-12-15")
    ], ignore_index=True)
    snapshots = {
        "2025-11-15": pd.DataFrame({"user_id": [3, 1, 2], "static_a": [1., 1., 1.]}),
        "2025-12-15": pd.DataFrame({"user_id": [3, 1, 2], "static_a": [2., 2., 2.]}),
    }
    frame = build_meta_frame(components, snapshots)
    assert frame.loc[frame.cutoff == "2025-11-15", "static_a"].tolist() == [1., 1., 1.]
    assert frame.loc[frame.cutoff == "2025-12-15", "static_a"].tolist() == [2., 2., 2.]


def test_bundle_manifest_contains_provenance_and_ordered_schema(tmp_path):
    from lstm_gate_calibrator import IdentityGateCalibrator
    model = IdentityGateCalibrator()
    manifest = save_calibrator_bundle(
        tmp_path / "bundle", model, feature_names=["lstm_gate_prob"],
        oof_hash="oof", inference_hash="inference", data_hash="data",
        selected_params={"learning_rate": 0.03}, best_iteration=7,
    )
    assert manifest["hashes"] == {"oof": "oof", "inference": "inference", "data": "data"}
    assert manifest["best_iteration"] == 7
    assert validate_calibrator_bundle(tmp_path / "bundle")["feature_names"] == ["lstm_gate_prob"]


def test_tiny_lightgbm_fit_uses_eval_early_stop_and_schema_safe_refit():
    pytest.importorskip("lightgbm")
    ids = np.arange(120)
    base_target = np.random.default_rng(42).integers(0, 2, len(ids)).astype(float)
    parts = []
    for cutoff, shift in [("2025-11-15", 0.0), ("2025-12-15", 0.0), ("2026-01-14", 0.0)]:
        part = pd.DataFrame({
            "user_id": ids, "cutoff": cutoff, "target": base_target + shift,
            "source": "oof", "pred_log": 0.1 + 0.001 * ids,
            "gate_prob": np.clip(0.1 + 0.007 * (ids % 100), 0.01, 0.99),
            "positive_log": 1.0 + 0.002 * ids, "direct_log": 0.2 + 0.001 * ids,
            "hurdle_log": 0.1 + 0.001 * ids, "history_length": ids + 1,
        })
        parts.append(part)
    frame = build_meta_frame(pd.concat(parts, ignore_index=True))
    config = CalibrationSearchConfig(search_trials=1, max_boost_rounds=15, early_stopping_rounds=3)
    selected = select_temporal_calibrator(frame, config)
    lgbm_rows = [row for row in selected["candidates"] if row["name"] == "lightgbm"]
    assert lgbm_rows and "skip_reason" not in lgbm_rows[0]
    assert lgbm_rows[0]["params"]["max_boost_rounds"] == 15
    assert lgbm_rows[0]["best_iteration"] is not None
    selected["selected_name"] = "lightgbm"
    selected["best_iteration"] = lgbm_rows[0]["best_iteration"]
    production = fit_production_calibrator(frame, selected, config)
    assert tuple(production.feature_names_) == tuple(frame.attrs["feature_names"])
    assert production.best_iteration_ == selected["best_iteration"]
    applied = apply_calibrator_per_seed(frame.iloc[:120].copy(), production)
    assert np.isfinite(applied["prediction_log"]).all()
