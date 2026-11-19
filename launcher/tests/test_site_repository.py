"""The runtime matrix page built from the real repository: invariants only, never today's values (design D14)."""

import json
import unittest

from launcher import clients
from launcher.catalog import load_catalog
from launcher.e2e import summary
from launcher.runtimes import ADAPTERS
from launcher.site import data, render
from launcher.tests.fakes import REPO
from launcher.tests.site_fixture import well_formed_problems


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
        page = render.page(matrix, data.Build("https://github.com/cevheri/msi-rtx5080-local-llm", "0" * 40, 0), REPO)
        self.assertEqual(well_formed_problems(page), [])
        self.assertNotIn(str(REPO), page)

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

    def test_opencode_default_reads_the_block_the_docs_tests_read(self):
        from launcher.tests.test_opencode_config import doc_config_text
        self.assertEqual(data.opencode_default(REPO), json.loads(doc_config_text())["model"])
        matrix = data.load(REPO)
        pairs = [(model.id, cell.runtime) for model in matrix.models for cell in model.cells
                 if cell.status == "runs" and cell.recipe.opencode == matrix.opencode_default]
        self.assertEqual(pairs, [matrix.default])

    def test_no_view_names_a_pair_twice(self):
        matrix = data.load(REPO)
        for use in data.USE_CASES:
            for sort, _ in data.SORTS:
                with self.subTest(use=use.id, sort=sort):
                    picks = data.view(matrix, use.id, data.Filters(), sort).picks
                    self.assertLessEqual(len(picks), 3)
                    self.assertEqual(len({(pick.model, pick.runtime) for pick in picks}), len(picks))


if __name__ == "__main__":
    unittest.main()
