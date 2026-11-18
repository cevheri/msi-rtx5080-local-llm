"""The runtime matrix page's sources: the shipped catalog, the runtime coverage table and the Top 10 (design D4, D5,
D13), loaded from a small fixture repository."""

import dataclasses
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from launcher import clients, markdown, ops, preflight
from launcher.runtimes import ADAPTERS
from launcher.site import data
from launcher.tests.site_fixture import (RECIPES, SiteFixture, cell_of, default_rows, model_of, prefix, row,
                                        version_record)

QWEN = prefix("llamacpp", "qwen3.6-35b-a3b")
QWEN_SGLANG = prefix("sglang", "qwen3.6-35b-a3b")


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

    def result(self, matrix, model_id: str, runtime: str, client: str) -> tuple:
        found = next(r for r in cell_of(matrix, model_id, runtime).clients if r.client == client)
        return found.cls, found.passed_runs, found.runs, found.latest, found.not_reached

    def note(self, fixture: SiteFixture, model_id: str, runtime: str, text: str) -> None:
        fixture.coverage({(model_id, runtime): fixture._default_cell(model_id, runtime) + "; " + text})

    def run_of(self, cell, path: str):
        return next(cell_run for cell_run in cell.history if cell_run.run.path == path)

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

    def test_client_classes(self):
        R = data.ClientRun
        marker = "e2e 2026-11-18: `hermes-z` and `hermes-chat` aimed hello.py outside the work folder"
        cases = [  # (supported, history, note, (cls, passed_runs, runs, latest, not_reached))
            (False, [R("2026-11-18", "b", False, ("hermes-z",), None)], marker, ("not-supported", 0, 0, None, None)),
            (True, [], marker, ("not-tested", 0, 0, None, None)),
            (True, [R("2026-11-17", "a", False, ("hermes-z",), None), R("2026-11-18", "b", True, (), None)], marker,
             ("passed", 1, 2, "2026-11-18", None)),
            (True, [R("2026-11-18", "b", False, ("hermes-z", "hermes-chat"), None)], marker,
             ("known-issue", 0, 1, "2026-11-18", None)),
            (True, [R("2026-11-18", "b", False, ("hermes-z", "hermes-config-model"), None)], marker,      # one unnamed
             ("failed", 0, 1, "2026-11-18", None)),
            (True, [R("2026-11-18", "b", False, ("pi",), None)], "e2e 2026-11-18: pi failed on reads",   # plain text
             ("failed", 0, 1, "2026-11-18", None)),
            (True, [R("2026-11-18", "b", False, ("pi",), None)], "e2e 2026-11-18: `pi and kimi` failed",  # longer span
             ("failed", 0, 1, "2026-11-18", None)),
            (True, [R("2026-11-18", "b", False, ("pi",), None)], "e2e 2026-11-17: `pi` failed",          # older date
             ("failed", 0, 1, "2026-11-18", None)),
            (True, [R("2026-11-18", "b", False, ("pi",), None)], "`pi` reads first; e2e 2026-11-18: it failed",  # ruling 3
             ("failed", 0, 1, "2026-11-18", None)),
            (True, [R("2026-11-25", "c", False, ("pi",), None)], "e2e 2026-11-25: `kimi` failed; e2e 2026-11-18: `pi`",
             ("failed", 0, 1, "2026-11-25", None)),                                                       # ruling 3
            (True, [R("2026-11-18", "a", True, (), None), R("2026-11-18", "b", False, ("pi",), None)],     # same date
             "e2e 2026-11-18: `pi` failed in one run", ("known-issue", 1, 2, "2026-11-18", None)),
            (True, [R("2026-11-18", "a", True, (), None), R("2026-11-18", "b", False, ("pi",), None)],     # same date
             "e2e 2026-11-18: every session passed", ("failed", 1, 2, "2026-11-18", None)),
            (True, [R("2026-11-18", "b", False, ("up",), "up")], "e2e 2026-11-18: `up` failed",
             ("known-issue", 0, 1, "2026-11-18", "up")),
        ]
        for supported, history, note, expected in cases:
            with self.subTest(history=history, note=note):
                self.assertEqual(data.classify("hermes", supported, history, note), data.ClientResult("hermes", *expected))

    def test_runs_are_listed_by_date_then_folder(self):
        self.fixture.run("2026-11-17", "llmctl-b-e2e", default_rows())
        self.fixture.run("2026-11-17", "llmctl-a-e2e", default_rows(), readme=None)
        matrix = self.fixture.load()
        self.assertEqual([r.path for r in matrix.runs], ["research/2026-11-17/llmctl-a-e2e",
                                                         "research/2026-11-17/llmctl-b-e2e",
                                                         "research/2026-11-18/llmctl-fixture-e2e"])
        default = matrix.runs[2]
        self.assertEqual((default.date, default.folder, default.rows, default.passed_rows, default.recipes,
                          default.orphan_rows, default.readme, default.no_rows, default.records, default.clients),
                         ("2026-11-18", "llmctl-fixture-e2e", len(default_rows()), len(default_rows()), 10, 0,
                          "research/2026-11-18/llmctl-fixture-e2e/README.md", True, True, clients.NAMES))
        self.assertEqual([(other.readme, other.no_rows, other.records) for other in matrix.runs[:2]],
                         [(None, False, False), ("research/2026-11-17/llmctl-b-e2e/README.md", False, False)])
        cell = cell_of(matrix, "qwen3.6-35b-a3b", "llamacpp")
        self.assertEqual([cell_run.run.path for cell_run in cell.history],
                         ["research/2026-11-18/llmctl-fixture-e2e", "research/2026-11-17/llmctl-a-e2e",
                          "research/2026-11-17/llmctl-b-e2e"])
        self.assertEqual(cell.covered, "2026-11-18")
        self.assertEqual((cell_of(matrix, "k2-horizon-7b", "vllm").history,
                          cell_of(matrix, "k2-horizon-7b", "vllm").covered,
                          cell_of(matrix, "k2-horizon-7b", "vllm").clients), ((), None, ()))
        self.assertEqual([r.client for r in cell.clients], list(clients.NAMES))
        fixture = self.fresh()
        untested = (prefix("llamacpp", "k2-horizon-7b"), prefix("vllm", "deepseek-r1-0528-qwen3-8b"))
        fixture.run("2026-11-18", "llmctl-fixture-e2e", [line for line in default_rows()
                                                         if not line.startswith(tuple(f"| {p} |" for p in untested))],
                    readme=None)
        matrix = fixture.load()
        for model_id, runtime in (("k2-horizon-7b", "llamacpp"), ("deepseek-r1-0528-qwen3-8b", "vllm")):
            with self.subTest(model=model_id, runtime=runtime):
                cell = cell_of(matrix, model_id, runtime)
                self.assertEqual((cell.status, cell.history, cell.covered, cell.setup), ("runs", (), None, ()))
                self.assertEqual([r.client for r in cell.clients], list(clients.NAMES))
                for client in clients.NAMES:
                    unsupported = (client, runtime) == ("opencode", "vllm")
                    self.assertEqual(self.result(matrix, model_id, runtime, client),
                                     ("not-supported" if unsupported else "not-tested", 0, 0, None, None))

    def test_a_client_not_reached_failed_that_run(self):
        self.fixture.run("2026-11-17", "llmctl-old-e2e", default_rows())
        rows = [line for line in default_rows() if not line.startswith(f"| {QWEN} |")
                and not line.startswith(f"| {QWEN_SGLANG} |")]
        rows = [row(QWEN, "up", "exit status 4"), row(QWEN_SGLANG, "up"), row(QWEN_SGLANG, "status"),
                row(QWEN_SGLANG, "long-grammar", "the server stopped"), *rows]
        self.fixture.run("2026-11-19", "llmctl-new-e2e", rows)
        matrix = self.fixture.load()
        self.assertEqual(self.result(matrix, "qwen3.6-35b-a3b", "llamacpp", "hermes"),
                         ("failed", 2, 3, "2026-11-19", "up"))
        self.assertEqual(self.result(matrix, "qwen3.6-35b-a3b", "sglang", "hermes"),
                         ("failed", 2, 3, "2026-11-19", "long-grammar"))
        self.assertEqual(self.result(matrix, "gpt-oss-20b", "llamacpp", "hermes"), ("passed", 3, 3, "2026-11-19", None))
        self.assertEqual((cell_of(matrix, "qwen3.6-35b-a3b", "llamacpp").setup,
                          cell_of(matrix, "qwen3.6-35b-a3b", "sglang").setup),
                         ((data.Failure("up", "exit status 4", ()),),
                          (data.Failure("long-grammar", "the server stopped", ()),)))
        self.note(self.fixture, "qwen3.6-35b-a3b", "llamacpp", "e2e 2026-11-19: `up` failed")
        self.note(self.fixture, "qwen3.6-35b-a3b", "sglang", "e2e 2026-11-19: `long-grammar` failed")
        matrix = self.fixture.load()
        self.assertEqual(self.result(matrix, "qwen3.6-35b-a3b", "llamacpp", "hermes"),
                         ("known-issue", 2, 3, "2026-11-19", "up"))
        self.assertEqual(self.result(matrix, "qwen3.6-35b-a3b", "sglang", "hermes"),
                         ("known-issue", 2, 3, "2026-11-19", "long-grammar"))
        self.fixture.run("2026-11-20", "llmctl-pi-e2e", [line for line in default_rows()
                                                         if not line.startswith(f"| {QWEN} | pi |")])
        matrix = self.fixture.load()
        self.assertEqual(self.result(matrix, "qwen3.6-35b-a3b", "llamacpp", "pi"),
                         ("known-issue", 2, 3, "2026-11-19", "up"))
        self.assertEqual(self.result(matrix, "qwen3.6-35b-a3b", "llamacpp", "hermes"),
                         ("passed", 3, 4, "2026-11-20", None))
        kimi = self.result(matrix, "qwen3.6-35b-a3b", "llamacpp", "kimi")
        self.assertEqual(kimi, ("passed", 3, 4, "2026-11-20", None))
        self.fixture.run("2026-11-21", "llmctl-no-kimi-e2e", [row(QWEN, "up", "exit status 4"), *(
            line for line in default_rows() if not line.startswith(f"| {QWEN} |") and " | kimi | " not in line)])
        matrix = self.fixture.load()
        self.assertEqual(self.result(matrix, "qwen3.6-35b-a3b", "llamacpp", "kimi"), kimi)
        self.assertEqual(self.result(matrix, "qwen3.6-35b-a3b", "llamacpp", "hermes"),
                         ("failed", 3, 5, "2026-11-21", "up"))
        fixture = self.fresh()
        others = [line for line in default_rows() if not line.startswith(f"| {QWEN_SGLANG} |")]
        fixture.run("2026-11-19", "llmctl-a-e2e", [row(QWEN_SGLANG, "up"), row(QWEN_SGLANG, "status"),
                                                  row(QWEN_SGLANG, "long-grammar", "the server stopped"), *others])
        fixture.run("2026-11-19", "llmctl-b-e2e", [row(QWEN_SGLANG, "up", "exit status 4"), *others])
        self.assertEqual(self.result(fixture.load(), "qwen3.6-35b-a3b", "sglang", "hermes"),
                         ("failed", 1, 3, "2026-11-19", "long-grammar"))

    def test_setup_checks_mark_the_cell(self):
        rows = [row(QWEN, "up"), row(QWEN, "status", "no model listed"), row(QWEN, "config", "bad config"),
                row(QWEN, "hermes-z"), row(QWEN, "probes", "probe A failed")]
        self.fixture.run("2026-11-19", "llmctl-setup-e2e", rows)
        self.assertEqual(cell_of(self.fixture.load(), "qwen3.6-35b-a3b", "llamacpp").setup,
                         (data.Failure("status", "no model listed", ()), data.Failure("config", "bad config", ()),
                          data.Failure("probes", "probe A failed", ())))
        fixture = self.fresh()
        fixture.run("2026-11-17", "llmctl-setup-e2e", rows)
        self.assertEqual(cell_of(fixture.load(), "qwen3.6-35b-a3b", "llamacpp").setup, ())
        fixture = self.fresh()
        fixture.run("2026-11-19", "llmctl-gateway-e2e", [row(QWEN, "up"), row(QWEN, "gateway", "no gateway")])
        cell = cell_of(fixture.load(), "qwen3.6-35b-a3b", "llamacpp")
        self.assertEqual((cell.setup, cell.covered), ((), "2026-11-19"))

    def test_client_results_ignore_round_trip_and_gateway_rows(self):
        rows = [row(QWEN, "gateway", "gateway failed") if line == row(QWEN, "gateway") else line
                for line in default_rows()]
        rows += [row("round-trip", "opencode", "no answer"), row("-", "gateway", "no up created the profile")]
        self.assertIn(row(QWEN, "gateway", "gateway failed"), rows)
        self.fixture.run("2026-11-19", "llmctl-trip-e2e", rows)
        matrix = self.fixture.load()
        for model in matrix.models:
            for cell in model.cells:
                for result in cell.clients:
                    with self.subTest(model=model.id, runtime=cell.runtime, client=result.client):
                        self.assertIn(result.cls, ("passed", "not-supported"))
        cell = cell_of(matrix, "qwen3.6-35b-a3b", "llamacpp")
        cell_run = self.run_of(cell, "research/2026-11-19/llmctl-trip-e2e")
        self.assertEqual((cell_run.steps, cell_run.failures),
                         (len([line for line in rows if line.startswith(f"| {QWEN} |")]) - 1, ()))
        self.assertEqual(self.result(matrix, "deepseek-r1-0528-qwen3-8b", "vllm", "opencode"),
                         ("not-supported", 0, 0, None, None))

    def test_records_are_found(self):
        rows = [row(QWEN, "up"), row(QWEN, "opencode", "no answer"), row(QWEN, "probes", "probe failed"),
                row(QWEN, "pi", "no answer")]
        records = {f"012-{QWEN}-opencode": "x\n", f"021-{QWEN}-probes-B": "x\n", f"020-{QWEN}-probes-A": "x\n",
                   f"015-{QWEN}-opencode-extra": "x\n"}
        path = self.fixture.run("2026-11-19", "llmctl-records-e2e", rows, records=records)
        cell_run = self.run_of(cell_of(self.fixture.load(), "qwen3.6-35b-a3b", "llamacpp"), path)
        self.assertEqual([(f.step, f.records) for f in cell_run.failures],
                         [("opencode", (f"{path}/records/012-{QWEN}-opencode",)),
                          ("probes", (f"{path}/records/020-{QWEN}-probes-A", f"{path}/records/021-{QWEN}-probes-B")),
                          ("pi", ())])
        fixture = self.fresh()
        path = fixture.run("2026-11-19", "llmctl-records-e2e", rows)
        cell_run = self.run_of(cell_of(fixture.load(), "qwen3.6-35b-a3b", "llamacpp"), path)
        self.assertEqual([f.records for f in cell_run.failures], [(), (), ()])
        fixture = self.fresh()
        fixture.run("2026-11-19", "llmctl-records-e2e", rows, records={**records, f"013-{QWEN}-opencode": "x\n"})
        message = self.assertStops(fixture, "opencode")
        self.assertTrue(message.startswith("research/2026-11-19/llmctl-records-e2e/records: "), message)

    def test_versions_are_read_from_client_records_only(self):
        from launcher.e2e import clients as e2e
        self.assertEqual(data.WITHHELD, e2e.WITHHELD)
        self.fixture.run("2026-11-17", "llmctl-old-e2e", default_rows(), records={
            "001-hermes-version": version_record("", 127), "002-pi-version": version_record("1.5.2", None),
            "003-opencode-version": version_record(""), "004-dsh-version": '{"exit": false, "stdout": "0.2.2"}\n',
            "005-kimi-version": version_record(data.WITHHELD), "006-nvidia-smi-version": version_record("[N/A]")})
        matrix = self.fixture.load()
        self.assertEqual(matrix.runs[1].versions, (("opencode", "opencode v2.3.0"), ("hermes", "Hermes Agent v0.23.1"),
                                                   ("pi", "1.5.2"), ("dsh", "0.2.2-rc.3"), ("kimi", "2.5.3")))
        self.assertEqual(matrix.runs[0].versions, (("opencode", None), ("hermes", None), ("pi", None), ("dsh", None),
                                                   ("kimi", None)))
        versions = {client.id: client.versions for client in matrix.clients}
        self.assertEqual(versions["opencode"], (("opencode v2.3.0", "research/2026-11-18/llmctl-fixture-e2e"),))
        self.fixture.run("2026-11-18", "llmctl-second-e2e", default_rows(), records={
            "003-opencode-version": version_record("opencode v2.4.0\nmore\n"),
            "001-hermes-version": version_record("Hermes Agent v0.23.1")})
        versions = {client.id: client.versions for client in self.fixture.load().clients}
        self.assertEqual(versions["opencode"], (("opencode v2.3.0", "research/2026-11-18/llmctl-fixture-e2e"),
                                                ("opencode v2.4.0", "research/2026-11-18/llmctl-second-e2e")))
        self.assertEqual(versions["hermes"], (("Hermes Agent v0.23.1", "research/2026-11-18/llmctl-fixture-e2e"),))
        fixture = self.fresh()
        older = fixture.run("2026-11-17", "llmctl-older-e2e", default_rows(), records={
            "001-hermes-version": version_record("Hermes old"),
            "003-opencode-version": version_record("opencode v2.2.0 \n")})
        fixture.run("2026-11-18", "llmctl-fixture-e2e", default_rows(), readme=None,
                    records={"001-hermes-version": version_record("", 127)})
        matrix = fixture.load()
        self.assertEqual(matrix.runs[0].versions, (("opencode", "opencode v2.2.0"), ("hermes", "Hermes old")))
        versions = {client.id: client.versions for client in matrix.clients}
        self.assertEqual((versions["opencode"], versions["hermes"]),
                         ((("opencode v2.3.0", "research/2026-11-18/llmctl-fixture-e2e"),), (("Hermes old", older),)))
        for name, records in (("not JSON", {"001-hermes-version": "not json\n"}),
                              ("no stdout", {"001-hermes-version": '{"exit": 0}\n'}),
                              ("no exit", {"001-hermes-version": '{"stdout": "x"}\n'}),
                              ("not an object", {"001-hermes-version": '["x"]\n'}),
                              ("twice", {"001-hermes-version": version_record("a"),
                                         "007-hermes-version": version_record("b")})):
            with self.subTest(record=name):
                fixture = self.fresh()
                fixture.run("2026-11-17", "llmctl-bad-e2e", default_rows(), records=records)
                self.assertStops(fixture, "research/2026-11-17/llmctl-bad-e2e/records/")
                self.assertStops(fixture, "hermes-version")

    def test_only_e2e_summaries_are_read(self):
        notes = self.fixture.root / "research/2026-11-17/notes/summary.md"
        notes.parent.mkdir(parents=True)
        notes.write_bytes(b"# Benchmark summary\n\nNothing h\xe9re.\n")
        self.assertEqual(len(self.fixture.load().runs), 1)
        for path in ("research/x/summary.md", "research/2026-11-17/a/b/summary.md", "research/latest/a/summary.md"):
            with self.subTest(path=path):
                fixture = self.fresh()
                (fixture.root / path).parent.mkdir(parents=True)
                shutil.copyfile(fixture.root / "research/2026-11-18/llmctl-fixture-e2e/summary.md", fixture.root / path)
                message = self.assertStops(fixture, path)
                self.assertTrue(message.startswith(path + ": "), message)
        shutil.rmtree(self.fixture.root / "research/2026-11-18/llmctl-fixture-e2e")
        self.assertStops(self.fixture, "no e2e summary")

    def test_rows_of_recipes_that_no_longer_ship_are_counted(self):
        path = self.fixture.run("2026-11-19", "llmctl-orphan-e2e",
                                [row("03-removed-model-llamacpp", "up"), row("04-qwen3.6-35b-a3b-oldengine", "up"),
                                 row(QWEN, "up")])
        matrix = self.fixture.load()
        run = next(r for r in matrix.runs if r.path == path)
        self.assertEqual((run.orphan_rows, run.recipes, run.rows), (2, 3, 3))
        held = [(model.id, cell.runtime) for model in matrix.models for cell in model.cells
                for cell_run in cell.history if cell_run.run.path == path]
        self.assertEqual(held, [("qwen3.6-35b-a3b", "llamacpp")])

    def test_unknown_steps_and_malformed_summaries_stop_the_build(self):
        self.fixture.run("2026-11-19", "llmctl-bad-e2e", [row(QWEN, "up"), row(QWEN, "foo-bar")])
        message = self.assertStops(self.fixture, "foo-bar")
        self.assertTrue(message.startswith("research/2026-11-19/llmctl-bad-e2e/summary.md: line 6: "), message)
        for name, line in (("4 cells", f"| {QWEN} | up | Yes | 1 |"),
                           ("carriage return", f"| {QWEN} | pi | No | a\rb | 1 |")):
            with self.subTest(row=name):
                fixture = self.fresh()
                fixture.run("2026-11-19", "llmctl-bad-e2e", [line])
                message = self.assertStops(fixture, "line 5")
                self.assertTrue(message.startswith("research/2026-11-19/llmctl-bad-e2e/summary.md: "), message)
        fixture = self.fresh()
        path = fixture.run("2026-11-19", "llmctl-bad-e2e", [row(QWEN, "up")])
        with open(fixture.root / path / "summary.md", "ab") as handle:
            handle.write(b"\xff\xfe")
        self.assertStops(fixture, f"{path}/summary.md: not valid UTF-8")
        fixture = self.fresh()
        path = fixture.run("2026-11-19", "llmctl-bad-e2e", [row(QWEN, "up")])
        (fixture.root / path / "README.md").write_bytes(b"# e2e\n\xff\n")
        message = self.assertStops(fixture, f"{path}/README.md: not valid UTF-8")
        self.assertTrue(message.startswith(f"{path}/README.md: "), message)


if __name__ == "__main__":
    unittest.main()
