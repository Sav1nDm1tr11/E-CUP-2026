import tempfile
import unittest
from pathlib import Path

from src.artifacts import CheckpointMismatchError, load_checkpoint, save_checkpoint


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
