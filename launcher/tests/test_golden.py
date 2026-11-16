import json
import unittest

from launcher.catalog import load_catalog
from launcher.tests.fakes import REPO
from launcher.tests.regen_golden import GOLDEN, golden_argv

HINT = "run `python3 -m launcher.tests.regen_golden` and review the diff of launcher/tests/golden/"


class GoldenTest(unittest.TestCase):
    def test_every_recipe_matches_golden(self):
        catalog = load_catalog(REPO / "recipes", None)
        checked = 0
        for runtime, entry in catalog.runtimes.items():
            for recipe_id in sorted(entry.recipes):
                path = GOLDEN / runtime / f"{recipe_id}.json"
                self.assertTrue(path.is_file(), f"no golden file {path}; {HINT}")
                expected = json.loads(path.read_text())
                self.assertEqual(golden_argv(recipe_id, runtime, REPO / "recipes"), expected,
                                 f"{runtime}/{recipe_id} changed; {HINT}")
                checked += 1
        self.assertEqual(checked, 19)

    def test_no_orphan_golden(self):
        catalog = load_catalog(REPO / "recipes", None)
        for path in sorted(GOLDEN.glob("*/*.json")):
            self.assertIn(path.stem, catalog.runtimes[path.parent.name].recipes, f"orphan golden file {path}")


if __name__ == "__main__":
    unittest.main()
