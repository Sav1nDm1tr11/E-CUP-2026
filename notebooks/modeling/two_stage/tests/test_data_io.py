import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data_io import find_project_root, load_expected_user_ids


class ProjectRootTests(unittest.TestCase):
    def test_finds_marker_in_parent_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            marker = root / "data" / "Prepared_data.parquet"
            marker.parent.mkdir()
            marker.touch()
            nested = root / "notebooks" / "modeling" / "two_stage"
            nested.mkdir(parents=True)

            self.assertEqual(find_project_root(nested), root)

    def test_raises_when_marker_is_missing(self):
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            self.assertRaises(FileNotFoundError),
        ):
            find_project_root(Path(temp_dir))


class ExpectedUserIdsTests(unittest.TestCase):
    def test_loads_user_ids_in_file_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample_submit.csv"
            pd.DataFrame({"user_id": [7, 2], "predict": [0.0, 0.0]}).to_csv(
                path,
                index=False,
            )

            self.assertEqual(load_expected_user_ids(path).tolist(), [7, 2])

    def test_rejects_duplicate_user_ids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample_submit.csv"
            pd.DataFrame({"user_id": [2, 2]}).to_csv(path, index=False)

            with self.assertRaisesRegex(ValueError, "дубликаты"):
                load_expected_user_ids(path)


if __name__ == "__main__":
    unittest.main()
