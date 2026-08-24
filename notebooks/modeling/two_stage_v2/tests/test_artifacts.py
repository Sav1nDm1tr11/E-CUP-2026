import tempfile
import unittest
from pathlib import Path
import sys
import types
from unittest.mock import patch

import numpy as np

from src.artifacts import (
    ArtifactBundle,
    ArtifactManifest,
    CheckpointMismatchError,
    build_manifest,
    load_checkpoint,
    load_manifest,
    save_checkpoint,
    save_manifest,
    save_model,
    sha256_json,
    validate_manifest,
)


class BundleClassifier:
    def predict_proba(self, X):
        import numpy as np
        return np.column_stack([np.full(len(X), .75), np.full(len(X), .25)])


class BundleRegressor:
    def predict(self, X):
        import numpy as np
        return np.full(len(X), 2.0)


class BundleCalibrator:
    def predict_proba(self, p):
        import numpy as np
        p = np.asarray(p)
        return np.column_stack([1 - p, p])


class ArtifactTests(unittest.TestCase):
    def test_checkpoint_is_rejected_when_config_fingerprint_changes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "fold_1.json"
            save_checkpoint(path, {"value": 1}, fingerprint="abc")
            self.assertEqual(load_checkpoint(path, expected_fingerprint="abc")["value"], 1)
            with self.assertRaises(CheckpointMismatchError):
                load_checkpoint(path, expected_fingerprint="different")

    def test_corrupted_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "fold_1.json"
            save_checkpoint(path, {"value": 1}, fingerprint="abc")
            path.write_bytes(b"{not valid json")
            with self.assertRaises((ValueError, CheckpointMismatchError)):
                load_checkpoint(path, expected_fingerprint="abc")

    def test_manifest_round_trip_and_hash_validation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "data.parquet"
            data_path.write_bytes(b"data")
            manifest = build_manifest(config_sha256="c", feature_sha256="f", data_sha256="d",
                                     trained_through="2025-12-15T00:00:00", model_names=("lgbm", "cat"),
                                     weights=(0.75, 0.25), data_path=data_path,
                                     package_versions={"numpy": "1", "pandas": "1", "scikit-learn": "1"},
                                     blend_history=({"fold": 1},), pre_january_config={"a": 1},
                                     post_january_config={"a": 2})
            path = Path(temp_dir) / "manifest.json"
            save_manifest(path, manifest)
            loaded = load_manifest(path)
            validate_manifest(loaded, config_sha256="c", feature_sha256="f", data_sha256="d",
                              model_names=("lgbm", "cat"), weights=(0.75, 0.25))
            self.assertIsInstance(loaded, ArtifactManifest)
            with self.assertRaises(CheckpointMismatchError):
                validate_manifest(loaded, feature_sha256="wrong")

    def test_incomplete_manifest_is_rejected(self):
        with self.assertRaises(CheckpointMismatchError):
            validate_manifest(ArtifactManifest(config_sha256="c", feature_sha256="f", data_sha256="d"))

    def test_sklearn_lightgbm_adapter_uses_native_booster_save(self):
        class Booster:
            def model_to_string(self):
                return "native-booster"
        class Wrapper:
            booster_ = Booster()
            def save_model(self, path):
                raise AssertionError("wrapper save_model should not be selected")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "classifier.txt"
            save_model(path, Wrapper())
            self.assertEqual(path.read_text(encoding="utf-8"), "native-booster")

    def test_lightgbm_text_round_trip_is_utf8_safe_for_unicode_paths(self):
        class NativeBooster:
            def model_to_string(self):
                return "модель\nfeature=0\n"

        class Wrapper:
            booster_ = NativeBooster()

        class LoadedBooster:
            def __init__(self, *, model_str=None, model_file=None):
                if model_file is not None or model_str is None:
                    raise AssertionError("LightGBM must be loaded from UTF-8 model_str")
                self.model_str = model_str

            def predict(self, X):
                return np.full(len(X), 0.25 if "модель" in self.model_str else 0.0)

        fake_lightgbm = types.SimpleNamespace(Booster=LoadedBooster)
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(sys.modules, {"lightgbm": fake_lightgbm}):
            path = Path(temp_dir) / "модели" / "классификатор.txt"
            save_model(path, Wrapper())
            self.assertIn("модель", path.read_text(encoding="utf-8"))
            from src.artifacts import load_model
            loaded = load_model(path)
            np.testing.assert_allclose(loaded.predict(np.zeros((3, 1))), [0.25, 0.25, 0.25])

    def test_bundle_save_load_persists_all_components_and_provenance(self):
        names = tuple(f"feature_{i}" for i in range(91))
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "data.parquet"
            data_path.write_bytes(b"data")
            manifest = build_manifest(config_sha256="config", feature_sha256=sha256_json(list(names)),
                                      data_sha256="data", trained_through="2025-12-15T00:00:00",
                                      model_names=("classifier",), weights=(1.0,), data_path=data_path,
                                      package_versions={"numpy": "1", "pandas": "1", "scikit-learn": "1"},
                                      blend_history=({"fold": "outer_1"},),
                                      pre_january_config={"phase": "pre"}, post_january_config={"phase": "post"})
            bundle = ArtifactBundle(manifest, names, (BundleClassifier(),), BundleRegressor(), BundleCalibrator())
            path = Path(temp_dir) / "bundle"
            from src.artifacts import load_bundle, save_bundle
            save_bundle(path, bundle)
            loaded = load_bundle(path, expected_config_sha256="config",
                                 expected_feature_sha256=sha256_json(list(names)), expected_data_sha256="data")
            self.assertEqual(loaded.feature_names, names)
            self.assertEqual(len(loaded.classifiers), 1)
            self.assertIsNotNone(loaded.positive_regressor)
            self.assertIsNotNone(loaded.calibrator)
            self.assertEqual(loaded.manifest.blend_history[0]["fold"], "outer_1")
            with self.assertRaises(CheckpointMismatchError):
                load_bundle(path, expected_config_sha256="wrong",
                            expected_feature_sha256=sha256_json(list(names)), expected_data_sha256="data")
