"""The model guide built from the real repository: invariants only, never today's values (design D14)."""

import json
import re
import unittest

from launcher import clients, markdown
from launcher.catalog import load_catalog
from launcher.e2e import summary
from launcher.runtimes import ADAPTERS
from launcher.site import data, fit, render, state, words
from launcher.tests.fakes import REPO
from launcher.tests.site_fixture import well_formed_problems
from launcher.tests.test_site_state import HERO_TEXTS, text_of


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
        pages = render.site(matrix, data.Build("https://github.com/cevheri/msi-rtx5080-local-llm", "0" * 40, 0), REPO)
        self.assertGreater(len(pages), 0)
        for path, page in pages.items():
            with self.subTest(page=path):
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

    def test_every_setup_names_its_model_alike(self):
        for model in data.load(REPO).models:
            for cell in model.cells:
                if cell.status == "runs":
                    with self.subTest(model=model.id, runtime=cell.runtime):
                        self.assertEqual(data.display_name(cell.recipe.name, cell.recipe.file), model.name)

    def test_every_state_has_a_pick_that_agrees_with_its_row(self):
        b = fit.bench(data.load(REPO))
        for m in fit.machines():
            for agent in (*b.agents, fit.NONE):
                with self.subTest(machine=m, agent=agent):
                    pick = fit.hero_pick(b, m, agent)
                    row = fit.row_setup(b, pick.setup.model, m)
                    if row == pick.setup:
                        self.assertEqual(fit.verdict(row, m, b.test_gpu).key, pick.verdict.key)
                    if pick.verdict.key == "ok":
                        self.assertEqual(fit.verdict(row, m, b.test_gpu).key, "ok")

    def test_refusal_messages_are_llmctls(self):
        _, rows = markdown.table(markdown.section(REPO, "docs/llmctl.md#when-up-refuses"), "Message starts with")
        starts = {span for row in rows for span in re.findall(r"`([^`]+)`", row[0])}
        self.assertEqual([message for message, _ in words.REFUSALS if message not in starts], [])

    def test_the_ranking_answer_rests_on_the_decision_summary(self):
        summary = " ".join(markdown.section(REPO, "docs/model-selection.md#decision-summary"))
        self.assertIn("one forced swap so that the DeepSeek family is represented", summary)

    def test_pages_stay_within_the_size_budget(self):
        pages = render.site(data.load(REPO), data.Build("https://github.com/cevheri/msi-rtx5080-local-llm", "0" * 40, 0), REPO)
        sizes = {path: len(page.encode("utf-8")) for path, page in pages.items()}
        self.assertLessEqual(sizes.pop("index.html"), 220_000)
        self.assertEqual([p for p, n in sizes.items() if n > 120_000], [])

    def test_the_hero_never_says_setups_or_your_before_input(self):
        matrix = data.load(REPO)
        b, labels = fit.bench(matrix), words.labels(matrix)
        st = state.home_state(matrix, b, labels, render.Links(REPO, data.Build("https://github.com/o/r", "0" * 40, 0)))
        hero = {record[i] for row in st["hero"] for record in row for i in HERO_TEXTS if record[i] >= 0}
        lap = {record[i] for record in st["heroLap"] for i in HERO_TEXTS if record[i] >= 0}
        self.assertFalse([t for t in map(text_of, (st["S"][i] for i in hero | lap)) if re.search(r"\bsetups\b", t)])
        self.assertFalse([t for t in map(text_of, (st["S"][i] for i in lap)) if re.search(r"\byour\b", t, re.I)])


if __name__ == "__main__":
    unittest.main()
