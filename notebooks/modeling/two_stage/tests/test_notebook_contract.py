import ast
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


def _artifact_cells(code_cells):
    artifact_cells = []
    for source in code_cells:
        tree = ast.parse(source)
        names = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
        }
        if {"production_metadata", "CONFIGURATION_PATH"}.issubset(names):
            artifact_cells.append(tree)
    return artifact_cells


class NotebookContractTests(unittest.TestCase):
    def test_artifact_cell_delegates_json_serialization(self):
        code_cells = _code_cells()
        notebook_source = "\n".join(code_cells)

        self.assertNotIn("json_default", notebook_source)
        self.assertNotIn("json.dump", notebook_source)
        artifact_cells = _artifact_cells(code_cells)
        self.assertEqual(len(artifact_cells), 1)

        save_json_calls = [
            node
            for node in ast.walk(artifact_cells[0])
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "save_json"
                and len(node.args) == 2
                and all(isinstance(arg, ast.Name) for arg in node.args)
                and node.args[0].id == "production_metadata"
                and node.args[1].id == "CONFIGURATION_PATH"
            )
        ]
        self.assertEqual(len(save_json_calls), 1)

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
