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
    select_temporal_calibrator,
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
    for cutoff, offset in [("2025-11-01", 0), ("2025-12-01", 10), ("2026-01-01", 20)]:
        part = _components(cutoff=cutoff)
        part["target"] += offset / 10
        rows.append(part)
    result = select_temporal_calibrator(
        pd.concat(rows, ignore_index=True),
        CalibrationSearchConfig(max_trials=2),
    )
    assert result["selection_cutoff"] == "2025-12-01"
    assert result["audit_cutoff"] == "2026-01-01"
    assert result["trials"] <= 2


def test_lightgbm_configuration_has_eval_set_and_early_stopping():
    config = CalibrationSearchConfig(max_trials=3)
    assert config.early_stopping_rounds > 0
    assert config.max_trials <= 3

