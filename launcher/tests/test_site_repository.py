"""The runtime matrix page built from the real repository: invariants only, never today's values (design D14)."""

import unittest

from launcher import clients
from launcher.catalog import load_catalog
from launcher.e2e import summary
from launcher.runtimes import ADAPTERS
from launcher.site import data
from launcher.tests.fakes import REPO


class SiteRepositoryTest(unittest.TestCase):
    def test_the_repository_builds(self):
        matrix = data.load(REPO)
        catalog = data.shipped_catalog(REPO)
        shipped = sorted({recipe_id for entry in catalog.runtimes.values() for recipe_id in entry.recipes})
        self.assertEqual([model.id for model in matrix.models], shipped)
        for model in matrix.models:
            with self.subTest(model=model.id):
                self.assertEqual([cell.runtime for cell in model.cells], list(ADAPTERS))
                for cell in model.cells:
                    if cell.status == "runs":
                        self.assertIsNotNone(cell.recipe, cell.runtime)
                    else:
                        self.assertIsNone(cell.recipe, cell.runtime)

    def test_supported_rule_matches_the_e2e_plan(self):
        from launcher.e2e import clients as e2e
        catalog = load_catalog(REPO / "recipes", None)
        for runtime, entry in catalog.runtimes.items():
            for recipe_id, recipe in entry.recipes.items():
                steps = e2e.plan_steps(catalog, [(recipe_id, runtime)], round_trip=False, keep_running=True,
                                       dsh=True, kimi=True)
                planned = {summary.step_client(s.name.split(" ")[-1], clients.NAMES) for s in steps} - {None}
                for client in clients.NAMES:
                    with self.subTest(recipe=recipe_id, runtime=runtime, client=client):
                        self.assertEqual(client in planned, data.SUPPORTED.get(client, lambda r: True)(recipe))


if __name__ == "__main__":
    unittest.main()
