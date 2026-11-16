import json
import unittest

from launcher import hermes_config, pi_config
from launcher.catalog import load_catalog
from launcher.tests.fakes import REPO
from launcher.tests.regen_golden import CLIENT_GOLDEN, GOLDEN, golden_argv, target_for

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

    def test_every_recipe_matches_client_golden(self):
        catalog = load_catalog(REPO / "recipes", None)
        checked = {"hermes": 0, "pi": 0}
        for runtime, entry in catalog.runtimes.items():
            for recipe_id in sorted(entry.recipes):
                target = target_for(recipe_id, runtime, REPO / "recipes")
                for client, suffix, text in (("hermes", ".hermes.yaml", hermes_config.render_block(target)),
                                             ("pi", ".pi.json", pi_config.new_models_text(target))):
                    path = CLIENT_GOLDEN / runtime / f"{recipe_id}{suffix}"
                    self.assertTrue(path.is_file(), f"no client golden file {path}; {HINT}")
                    self.assertEqual(text, path.read_text(), f"{runtime}/{recipe_id}{suffix} changed; {HINT}")
                    checked[client] += 1
        self.assertEqual(checked, {"hermes": 19, "pi": 19})

    def test_no_orphan_client_golden(self):
        catalog = load_catalog(REPO / "recipes", None)
        for path in sorted(CLIENT_GOLDEN.glob("*/*")):
            name = path.name
            recipe_id = name.removesuffix(".hermes.yaml") if name.endswith(".hermes.yaml") else name.removesuffix(".pi.json")
            self.assertNotEqual(recipe_id, name, f"unexpected client golden file {path}")
            self.assertIn(path.parent.name, catalog.runtimes, f"orphan client golden folder {path.parent}")
            self.assertIn(recipe_id, catalog.runtimes[path.parent.name].recipes, f"orphan client golden file {path}")


if __name__ == "__main__":
    unittest.main()
