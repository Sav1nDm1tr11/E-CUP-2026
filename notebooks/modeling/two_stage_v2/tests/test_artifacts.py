import tempfile
import unittest
from pathlib import Path

from src.artifacts import (
    ArtifactManifest,
    CheckpointMismatchError,
    build_manifest,
    load_checkpoint,
    load_manifest,
    save_checkpoint,
    save_manifest,
    save_model,
    validate_manifest,
)


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
        manifest = build_manifest(config_sha256="c", feature_sha256="f", data_sha256="d",
                                 trained_through="2025-12-15", model_names=("lgbm", "cat"),
                                 weights=(0.75, 0.25), pre_january_config={"a": 1},
                                 post_january_config={"a": 2})
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "manifest.json"
            save_manifest(path, manifest)
            loaded = load_manifest(path)
            validate_manifest(loaded, config_sha256="c", feature_sha256="f", data_sha256="d",
                              model_names=("lgbm", "cat"), weights=(0.75, 0.25))
            self.assertIsInstance(loaded, ArtifactManifest)
            with self.assertRaises(CheckpointMismatchError):
                validate_manifest(loaded, feature_sha256="wrong")

    def test_sklearn_lightgbm_adapter_uses_native_booster_save(self):
        class Booster:
            def save_model(self, path):
                Path(path).write_bytes(b"native-booster")
        class Wrapper:
            booster_ = Booster()
            def save_model(self, path):
                raise AssertionError("wrapper save_model should not be selected")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "classifier.txt"
            save_model(path, Wrapper())
            self.assertEqual(path.read_bytes(), b"native-booster")
