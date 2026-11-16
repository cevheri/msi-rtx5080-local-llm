"""A change to one runtime's files must not change any command generated for the other runtime."""

import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path

from launcher.catalog import load_catalog
from launcher.tests.fakes import REPO
from launcher.tests.regen_golden import GOLDEN, golden_argv


def edit_runtime(directory: Path, port: int, extra_arg: str) -> None:
    """Change the runtime file and every recipe of one runtime."""
    runtime = directory / "_runtime.toml"
    text = runtime.read_text()
    text = re.sub(r'(?m)^image = "([^"]+):[^"]+"$', r'image = "\1:edited"', text)
    text = re.sub(r"(?m)^host_port = \d+$", f"host_port = {port}", text)
    text = re.sub(r"(?m)^start_timeout_s = \d+$", "start_timeout_s = 42", text)
    runtime.write_text(text)
    for recipe in directory.glob("*.toml"):
        if recipe.name == "_runtime.toml":
            continue
        text = recipe.read_text()
        assert "args = [\n" in text, recipe
        text = text.replace("args = [\n", f'args = [\n  "{extra_arg}",\n', 1)
        text = re.sub(r"(?m)^memory_gib = \d+$", "memory_gib = 7", text)
        recipe.write_text(text)


class IsolationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.recipes = Path(self.tmp.name) / "recipes"
        shutil.copytree(REPO / "recipes", self.recipes)

    def tearDown(self):
        self.tmp.cleanup()

    def assert_runtime_untouched(self, edited: str, untouched: str) -> None:
        catalog = load_catalog(self.recipes, None)
        self.assertEqual([p.message for p in catalog.runtimes[edited].problems], [])
        changed = 0
        for recipe_id in catalog.runtimes[edited].recipes:
            expected = json.loads((GOLDEN / edited / f"{recipe_id}.json").read_text())
            changed += golden_argv(recipe_id, edited, self.recipes) != expected
        self.assertEqual(changed, len(catalog.runtimes[edited].recipes), "the edit did not reach every recipe")
        for recipe_id in catalog.runtimes[untouched].recipes:
            expected = json.loads((GOLDEN / untouched / f"{recipe_id}.json").read_text())
            self.assertEqual(golden_argv(recipe_id, untouched, self.recipes), expected, f"{untouched}/{recipe_id}")

    def test_llamacpp_edits_leave_vllm_alone(self):
        edit_runtime(self.recipes / "llamacpp", 8090, "--metrics")
        self.assert_runtime_untouched("llamacpp", "vllm")

    def test_vllm_edits_leave_llamacpp_alone(self):
        edit_runtime(self.recipes / "vllm", 8001, "--disable-log-requests")
        self.assert_runtime_untouched("vllm", "llamacpp")


if __name__ == "__main__":
    unittest.main()
