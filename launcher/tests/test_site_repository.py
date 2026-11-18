"""The runtime matrix page built from the real repository: invariants only, never today's values (design D14)."""

import unittest

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


if __name__ == "__main__":
    unittest.main()
