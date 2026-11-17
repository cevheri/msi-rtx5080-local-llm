import json
import unittest

from launcher import dsh_config, hermes_config, kimi_config, pi_config
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
        checked = {"hermes": 0, "pi": 0, "dsh": 0, "kimi": 0}
        for runtime, entry in catalog.runtimes.items():
            for recipe_id in sorted(entry.recipes):
                target = target_for(recipe_id, runtime, REPO / "recipes")
                for client, suffix, text in (("hermes", ".hermes.yaml", hermes_config.render_block(target)),
                                             ("pi", ".pi.json", pi_config.new_models_text(target)),
                                             ("dsh", ".dsh.yml", dsh_config.render_block(target, default=True)),
                                             ("kimi", ".kimi.toml", kimi_config.render_tables(target))):
                    path = CLIENT_GOLDEN / runtime / f"{recipe_id}{suffix}"
                    self.assertTrue(path.is_file(), f"no client golden file {path}; {HINT}")
                    self.assertEqual(text, path.read_text(), f"{runtime}/{recipe_id}{suffix} changed; {HINT}")
                    checked[client] += 1
        self.assertEqual(checked, {"hermes": 19, "pi": 19, "dsh": 19, "kimi": 19})

    def test_no_orphan_client_golden(self):
        catalog = load_catalog(REPO / "recipes", None)
        for path in sorted(CLIENT_GOLDEN.glob("*/*")):
            name = path.name
            suffix = next((suffix for suffix in (".hermes.yaml", ".pi.json", ".dsh.yml", ".kimi.toml")
                           if name.endswith(suffix)), "")
            recipe_id = name.removesuffix(suffix) if suffix else name
            self.assertNotEqual(recipe_id, name, f"unexpected client golden file {path}")
            self.assertIn(path.parent.name, catalog.runtimes, f"orphan client golden folder {path.parent}")
            self.assertIn(recipe_id, catalog.runtimes[path.parent.name].recipes, f"orphan client golden file {path}")

    def test_dsh_golden_first_line_is_the_block_without_default(self):
        catalog = load_catalog(REPO / "recipes", None)
        checked = 0
        for runtime, entry in catalog.runtimes.items():
            for recipe_id in sorted(entry.recipes):
                target = target_for(recipe_id, runtime, REPO / "recipes")
                path = CLIENT_GOLDEN / runtime / f"{recipe_id}.dsh.yml"
                self.assertTrue(path.is_file(), f"no client golden file {path}; {HINT}")
                first = path.read_text().splitlines(keepends=True)[0]
                self.assertEqual(first, dsh_config.render_block(target, default=False), f"{runtime}/{recipe_id}")
                checked += 1
        self.assertEqual(checked, 19)


if __name__ == "__main__":
    unittest.main()
