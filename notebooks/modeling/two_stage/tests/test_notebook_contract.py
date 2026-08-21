import json
import unittest
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).parents[1] / "09_Two_Staged_Model.ipynb"


def _code_cells():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    return [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]


class NotebookContractTests(unittest.TestCase):
    def test_last_code_cell_delegates_json_serialization(self):
        last_cell = _code_cells()[-1]

        self.assertNotIn("json_default", last_cell)
        self.assertNotIn("json.dump", last_cell)
        self.assertIn("save_json(model_metadata, METADATA_PATH)", last_cell)

    def test_bootstrap_uses_neighbor_src_package(self):
        notebook_source = "\n".join(_code_cells())

        self.assertIn(
            "MODEL_DIR = Path.cwd().resolve()",
            notebook_source,
        )
        self.assertIn(
            'SRC_DIR = MODEL_DIR / "src"',
            notebook_source,
        )
        self.assertIn(
            "sys.path.insert(0, str(MODEL_DIR))",
            notebook_source,
        )
        self.assertIn(
            "from src.metrics import",
            notebook_source,
        )
        self.assertNotIn(
            "two_stage_model_helpers",
            notebook_source,
        )


if __name__ == "__main__":
    unittest.main()
