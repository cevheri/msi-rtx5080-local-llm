"""The runtime matrix page's sources: the shipped catalog, the runtime coverage table and the Top 10 (design D4, D5,
D13), loaded from a small fixture repository."""

import dataclasses
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from launcher import clients, markdown, ops, preflight
from launcher.runtimes import ADAPTERS
from launcher.site import data
from launcher.tests.site_fixture import RECIPES, SiteFixture, cell_of, model_of


def strings(value):
    """Every string held by a dataclass, tuple or string value, depth first."""
    if isinstance(value, str):
        yield value
    elif dataclasses.is_dataclass(value):
        for field in dataclasses.fields(value):
            yield from strings(getattr(value, field.name))
    elif isinstance(value, tuple):
        for item in value:
            yield from strings(item)


class SiteDataTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.fixture = SiteFixture(Path(self._tmp.name))

    def fresh(self) -> SiteFixture:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return SiteFixture(Path(tmp.name))

    def coverage_lines(self, fixture: SiteFixture) -> list[str]:
        lines = markdown.section(fixture.root, "docs/model-selection.md#runtime-coverage")
        return [line for line in lines if line.startswith("|")]

    def assertStops(self, fixture: SiteFixture, text: str) -> str:
        with self.assertRaises(data.SiteError) as caught:
            fixture.load()
        self.assertIn(text, str(caught.exception))
        return str(caught.exception)

    def test_cells_take_their_status_from_the_coverage_table(self):
        matrix = self.fixture.load()
        failed = cell_of(matrix, "deepseek-r1-0528-qwen3-8b", "sglang")
        self.assertEqual((failed.status, failed.measured, failed.recipe), ("failed", "2026-11-17", None))
        self.assertEqual(failed.note, "out of memory, measured 2026-11-17 "
                         "([record](../research/2026-11-17/fixture-failure/README.md))")
        self.assertEqual(cell_of(matrix, "k2-horizon-7b", "vllm").status, "does-not-fit")
        self.assertEqual(cell_of(matrix, "qwen3.6-35b-a3b", "sglang").status, "runs")
        self.assertEqual([c.runtime for c in model_of(matrix, "gpt-oss-20b").cells], ["llamacpp", "vllm", "sglang"])
        self.assertEqual([m.id for m in matrix.models], sorted({i for _, i in RECIPES}))
        self.assertEqual([r.heading for r in matrix.runtimes], ["llama.cpp b12407", "vLLM v0.33.0", "SGLang v0.5.23"])
        catalog = data.shipped_catalog(self.fixture.root)
        self.assertEqual({m.id: m.chat_only for m in matrix.models},
                         {m.id: any(entry.recipes[m.id].chat_only for entry in catalog.runtimes.values()
                                    if m.id in entry.recipes) for m in matrix.models})
        fixture = self.fresh()
        fixture.set_value("llamacpp", "gpt-oss-20b", "aliases", '["gpt-oss", "gpt-oss-b"]')
        fixture.set_value("vllm", "gpt-oss-20b", "aliases", '["gpt-oss-a", "gpt-oss"]')
        self.assertEqual(model_of(fixture.load(), "gpt-oss-20b").aliases, ("gpt-oss", "gpt-oss-b", "gpt-oss-a"))

    def test_runs_cells_and_measured_recipes_match(self):
        self.fixture.coverage({("k2-horizon-7b", "vllm"): "Runs: 64k, 94.8 tok/s, measured 2026-11-17"})
        self.assertStops(self.fixture, "row k2-horizon-7b, column vLLM")
        fixture = self.fresh()
        fixture.coverage({("gpt-oss-20b", "vllm"): "Failed: crashed, measured 2026-11-17"})
        self.assertStops(fixture, "row gpt-oss-20b, column vLLM")

    def test_coverage_rows_must_match_the_shipped_models(self):
        lines = self.coverage_lines(self.fixture)
        self.fixture.coverage_table([line for line in lines if not line.startswith("| `mellum2.1-12b` |")])
        self.assertStops(self.fixture, "mellum2.1-12b")
        fixture = self.fresh()
        fixture.coverage_table(lines + ["| `unknown-model` | Does not fit: x (estimate) | Does not fit: x (estimate) "
                                        "| Does not fit: x (estimate) |"])
        self.assertStops(fixture, "unknown-model")
        fixture = self.fresh()
        fixture.coverage_table(lines + [line for line in lines if line.startswith("| `gpt-oss-20b` |")])
        self.assertStops(fixture, "gpt-oss-20b")
        for cell, text in (("gpt-oss-20b", "row gpt-oss-20b: the Model ID must be one code span"),
                           ("", "a row with an empty Model ID cell")):
            with self.subTest(id_cell=cell):
                fixture = self.fresh()
                fixture.coverage_table([line.replace("| `gpt-oss-20b` |", f"| {cell} |") for line in lines])
                self.assertStops(fixture, text)
        for key in (("no-such-model", "vllm"), ("gpt-oss-20b", "no-such-runtime")):
            with self.subTest(key=key), self.assertRaises(LookupError):
                self.fresh().coverage({key: "Runs: x"})

    def test_coverage_columns_follow_the_adapters(self):
        lines = self.coverage_lines(self.fixture)
        self.fixture.coverage_table(["| Model ID | vLLM v0.33.0 | llama.cpp b12407 | SGLang v0.5.23 |", *lines[1:]])
        self.assertStops(self.fixture, "llama.cpp")
        fixture = self.fresh()
        fixture.coverage_table(["| Model ID | llama.cpp b12407 | vLLM v0.33.0 |", "|---|---|---|"]
                               + ["|" + "|".join(line.split("|")[1:4]) + "|" for line in lines[2:]])
        self.assertStops(fixture, "column")
        fixture = self.fresh()
        fixture.coverage_table(["| Model ID | llama.cppx b12407 | vLLM v0.33.0 | SGLang v0.5.23 |", *lines[1:]])
        self.assertStops(fixture, "llama.cpp")

    def test_failed_and_does_not_fit_cells_need_their_labels(self):
        for text in ("Failed: crashed", "Does not fit: too big", "Unknown: x", ""):
            with self.subTest(cell=text):
                fixture = self.fresh()
                fixture.coverage({("deepseek-r1-0528-qwen3-8b", "sglang"): text})
                self.assertStops(fixture, "row deepseek-r1-0528-qwen3-8b, column SGLang")

    def test_catalog_problems_stop_the_build(self):
        (self.fixture.root / "recipes/llamacpp/broken.toml").write_text("id = [\n", encoding="utf-8")
        message = self.assertStops(self.fixture, "broken")
        self.assertTrue(message.startswith("recipes/llamacpp/broken.toml: "), message)
        self.assertNotIn(str(self.fixture.root), message)
        readme = "# Fixture\n\n## Quick start\n\nRun a model with llmctl.\n"
        for name, text, expected in (
                ("no License section", readme, "README.md: no section #license"),
                ("empty License section", readme + "\n## License\n\n", "README.md: the License section has no text")):
            with self.subTest(source=name):
                fixture = self.fresh()
                (fixture.root / "README.md").write_text(text, encoding="utf-8")
                self.assertEqual(self.assertStops(fixture, expected), expected)
        for source in ("README.md", "docs/model-selection.md"):
            with self.subTest(source=source):
                fixture = self.fresh()
                with open(fixture.root / source, "ab") as handle:
                    handle.write(b"\xff\xfe")
                self.assertStops(fixture, f"{source}: not valid UTF-8")

    def test_unknown_thinking_type_stops_the_build(self):
        with mock.patch.dict(data.THINKING):
            del data.THINKING["effort"]
            self.assertStops(self.fixture, "effort")

    def test_rank_joins_the_top_10_by_repository_link(self):
        head = ["| Rank | Model | Checkpoint |", "|---|---|---|"]
        rows = ["| 1 | [A](https://huggingface.co/org/A) | [A GGUF](https://huggingface.co/org/A-GGUF) |",
                "| 2 | [B](https://huggingface.co/org/b) | none |"]
        repos = {"org/A-GGUF": ("a",), "org/B": ("b",)}
        self.assertEqual(data.rank_of(head + rows, repos), ({"a": 1}, ((2, "[B](https://huggingface.co/org/b)"),)))
        for bad in (["| 1 | [A](https://huggingface.co/org/A-GGUF) [B](https://huggingface.co/org/B) | x |"],
                    [rows[0], "| 1 | [B](https://huggingface.co/org/B) | x |"],             # rank used twice
                    [rows[0], "| 2 | [A](https://huggingface.co/org/A-GGUF) | x |"],        # one model ranked twice
                    ["| x | [B](https://huggingface.co/org/B) | x |"]):                     # not a positive integer
            with self.subTest(bad=bad), self.assertRaises(data.SiteError):
                data.rank_of(head + bad, repos)
        with self.assertRaisesRegex(data.SiteError, "^docs/model-selection.md: the Top 10 table has no Model column$"):
            data.rank_of(["| Rank | Name | x |", "|---|---|---|", "| 1 | [A](https://huggingface.co/org/A-GGUF) | x |"],
                         repos)
        with self.assertRaisesRegex(data.SiteError, "^docs/model-selection.md: a row with an empty Rank cell: "):
            data.rank_of(head + ["|  | [B](https://huggingface.co/org/B) | x |"], repos)

    def test_client_letters_must_be_unique(self):
        labels = {"pi": "Pi", "pio": "Pio", "kimi": "Kimi"}
        self.assertEqual(data.client_letters(("pi", "kimi"), labels, {}), {"pi": "P", "kimi": "K"})
        with self.assertRaisesRegex(data.SiteError, r"pi.*pio.*LETTERS"):
            data.client_letters(("pi", "pio"), labels, {})
        self.assertEqual(data.client_letters(("pi", "pio"), labels, {"pio": "Q"}), {"pi": "P", "pio": "Q"})

    def test_model_holds_no_absolute_path(self):
        self.assertEqual([s for s in strings(self.fixture.load()) if s.startswith("/")], [])

    def test_recipe_facts_come_from_the_recipe_and_llmctl(self):
        catalog = data.shipped_catalog(self.fixture.root)
        recipe, runtime = catalog.runtimes["sglang"].recipes["qwen3.6-35b-a3b"], catalog.runtimes["sglang"].runtime
        target = ops.client_target(recipe, runtime)
        facts = cell_of(self.fixture.load(), "qwen3.6-35b-a3b", "sglang").recipe
        self.assertEqual((facts.file, facts.doc, facts.context, facts.decode, facts.plain_runtime),
                         ("recipes/sglang/qwen3.6-35b-a3b.toml", recipe.doc, 131072, 151.3, "llamacpp"))
        self.assertEqual((facts.up, facts.pull, facts.gpu, facts.served, facts.opencode),
                         (ops.up_command(recipe), preflight.pull_hint(recipe),
                          ADAPTERS["sglang"].gpu_need_text(recipe, runtime),
                          target.served, f"{runtime.opencode_provider}/qwen3.6-35b-a3b"))
        self.assertEqual(facts.thinking_texts, tuple(zip(clients.NAMES, clients.thinking_texts(target))))
        self.assertIsNone(cell_of(self.fixture.load(), "deepseek-r1-0528-qwen3-8b", "vllm").recipe.opencode)
        self.fixture.set_value("llamacpp", "k2-horizon-7b", "context", "100000")
        with self.assertRaisesRegex(data.SiteError, "k2-horizon-7b"):
            self.fixture.load()


if __name__ == "__main__":
    unittest.main()
