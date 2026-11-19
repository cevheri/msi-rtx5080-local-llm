"""The runtime matrix page's sources: the shipped catalog, the runtime coverage table and the Top 10 (design D4, D5,
D13), loaded from a small fixture repository."""

import dataclasses
import os
import re
import shutil
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

from launcher import clients, markdown, ops, preflight
from launcher.runtimes import ADAPTERS
from launcher.site import data
from launcher.tests.fakes import REPO
from launcher.tests.site_fixture import (RECIPES, TOP10_IDS, TOP10_ROW, SiteFixture, cell_of, config_section,
                                        default_rows, model_of, prefix, row, version_record)

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


def order(matrix, key):
    """The Runs cells, as (model ID, runtime), sorted by their place in one cell order."""
    places = {(m.id, c.runtime): dict(c.positions)[key] for m in matrix.models for c in m.cells if c.positions}
    return sorted(places, key=places.get)


def cards(matrix, use, filters=data.Filters(), sort=None):
    """(label, model, runtime) of each pick of a view, computed over that view's cells and its effective sort."""
    key = sort or next(u.sort for u in data.USE_CASES if u.id == use)
    found = data.view(matrix, use, filters, sort)
    return [(p.label, p.model, p.runtime) for p in data.picks(matrix, use, found.cells, key)]


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
        self.assertEqual(self.fixture.load().top, 3)
        catalog = data.shipped_catalog(self.fixture.root)
        rows = [TOP10_ROW.format(rank=rank, id=model_id, repo=catalog.runtimes["llamacpp"].recipes[model_id].model.repo)
                for rank, model_id in enumerate(TOP10_IDS, start=1)]
        self.fixture.top10([*rows, "| 4 | [Unshipped](https://huggingface.co/org/unshipped) | llama.cpp |"])
        self.assertEqual(self.fixture.load().top, 4)

    def test_opencode_default_is_read_from_the_validated_config(self):
        matrix = self.fixture.load()
        self.assertEqual(data.opencode_default(self.fixture.root), "llamacpp-docker/qwen3.6-35b-a3b")
        self.assertEqual((matrix.opencode_default, matrix.default),
                         ("llamacpp-docker/qwen3.6-35b-a3b", ("qwen3.6-35b-a3b", "llamacpp")))
        self.fixture.opencode(config_section("sglang-docker/gpt-oss-20b") + "\n" + config_section("vllm-docker/gpt-oss-20b"))
        self.assertEqual(self.fixture.load().default, ("gpt-oss-20b", "sglang"))           # the first json block wins
        self.fixture.opencode('```jsonc\n{"model": "vllm-docker/gpt-oss-20b"}\n```\n\n' + config_section())
        self.assertEqual(self.fixture.load().default, ("qwen3.6-35b-a3b", "llamacpp"))     # only "```json" opens one
        where = "docs/opencode.md#the-validated-config: "
        no_model = where + 'the validated config has no top-level "model" string'
        for section, message in ((None, "docs/opencode.md: no section #the-validated-config"),
                                 ("No block here.\n", where + "the section has no complete json block"),
                                 ('```json\n{"model": "x"}\n', where + "the section has no complete json block"),
                                 ('```json\n["x"]\n```\n', no_model), ('```json\n{"model": 1}\n```\n', no_model),
                                 ('```json\n{"providers": {}}\n```\n', no_model)):
            with self.subTest(section=section):
                self.fixture.opencode(section)
                with self.assertRaises(data.SiteError) as caught:
                    self.fixture.load()
                self.assertEqual(str(caught.exception), message)
        with self.subTest(section="a json block under the next section"):           # the section ends at its heading
            (self.fixture.root / "docs/opencode.md").write_text(
                "# OpenCode\n\n## The validated config\n\nNo block here.\n\n## Context budget\n\n"
                + config_section().split("\n\n", 1)[1], encoding="utf-8")
            with self.assertRaises(data.SiteError) as caught:
                self.fixture.load()
            self.assertEqual(str(caught.exception), where + "the section has no complete json block")
        with self.subTest(section="a closing fence that carries text"):              # any later ``` line closes it
            self.fixture.opencode('```json\n{"model": "sglang-docker/gpt-oss-20b"}\n```text\n```\n')
            self.assertEqual(self.fixture.load().default, ("gpt-oss-20b", "sglang"))
        self.fixture.opencode('```json\n{"model": }\n```\n')
        with self.assertRaisesRegex(data.SiteError, "^" + re.escape(where + "the validated config is not JSON: ") + "."):
            self.fixture.load()
        (self.fixture.root / "docs/opencode.md").unlink()
        with self.assertRaisesRegex(data.SiteError, "^docs/opencode.md: missing$"):
            self.fixture.load()

    def test_opencode_default_must_name_one_pair_that_runs(self):
        where = "docs/opencode.md#the-validated-config: "
        target = self.fixture.root / "recipes/vllm/k2-horizon-7b.toml"    # an estimate recipe in a Does not fit cell
        shutil.copyfile(REPO / "recipes/vllm/k2-horizon-7b.toml", target)
        doc = tomllib.loads(target.read_text(encoding="utf-8"))["doc"]
        target.write_text("".join(line + "\n" for line in target.read_text(encoding="utf-8").splitlines()
                                  if not line.startswith("decode_tok_s = "))
                          .replace('kind = "measured"', f'kind = "estimate"\ndoc = "{doc}"'), encoding="utf-8")
        path, anchor = doc.split("#", 1)
        with (self.fixture.root / path).open("a", encoding="utf-8") as models:
            models.write(f"## {anchor}\n\n")
        self.assertEqual(data.shipped_catalog(self.fixture.root).runtimes["vllm"].recipes["k2-horizon-7b"].opencode, True)
        self.assertEqual(cell_of(self.fixture.load(), "k2-horizon-7b", "vllm").status, "does-not-fit")
        for model in ("vllm-docker/k2-horizon-7b",                 # the entry of a Does not fit cell
                      "vllm-docker/deepseek-r1-0528-qwen3-8b",     # a recipe with opencode = false
                      "llamacpp-docker/no-such-model"):
            with self.subTest(model=model):
                self.fixture.opencode(config_section(model))
                with self.assertRaises(data.SiteError) as caught:
                    self.fixture.load()
                self.assertEqual(str(caught.exception), f"{where}the default model {model} names 0 OpenCode entries of "
                                                        "pairs that run; it must name exactly one")
        self.fixture.opencode(config_section(""))
        with self.assertRaisesRegex(data.SiteError,
                                    "^" + re.escape(where + 'the validated config has no top-level "model" string') + "$"):
            self.fixture.load()
        runtime = self.fixture.root / "recipes/sglang/_runtime.toml"         # two engines with one provider name
        runtime.write_text(runtime.read_text(encoding="utf-8").replace('"sglang-docker"', '"llamacpp-docker"'),
                           encoding="utf-8")
        self.fixture.opencode(config_section())
        with self.assertRaisesRegex(data.SiteError, re.escape("llamacpp-docker/qwen3.6-35b-a3b names 2 OpenCode entries")):
            self.fixture.load()

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
        # The driver's own _Run.record names the record, so a change to its naming breaks this test too.
        fixture = self.fresh()
        path = fixture.run("2026-11-19", "llmctl-records-e2e", rows, records={})
        from launcher.e2e import clients as e2e
        from launcher.tests.fakes import FakeHost
        out = fixture.root / path
        e2e._Run(out, None, FakeHost(env={}), {"HOME": str(out)}, out).record(f"{QWEN} opencode", "x\n")
        cell_run = self.run_of(cell_of(fixture.load(), "qwen3.6-35b-a3b", "llamacpp"), path)
        self.assertEqual(cell_run.failures[0].records, (f"{path}/records/001-{QWEN}-opencode",))
        # A record of the longer step kimi-config is not a record of a failed kimi; its docker-logs record is.
        fixture = self.fresh()
        path = fixture.run("2026-11-19", "llmctl-records-e2e",
                           [row(QWEN, "up"), row(QWEN, "kimi-config"), row(QWEN, "kimi", "no answer")],
                           records={f"014-{QWEN}-kimi-config": "x\n", f"016-{QWEN}-kimi-docker-logs": "x\n"})
        cell_run = self.run_of(cell_of(fixture.load(), "qwen3.6-35b-a3b", "llamacpp"), path)
        self.assertEqual([(f.step, f.records) for f in cell_run.failures],
                         [("kimi", (f"{path}/records/016-{QWEN}-kimi-docker-logs",))])

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
        # A summary that starts with a byte order mark is an e2e summary still, and stops at line 1 rather than
        # leaving the page in silence; so does a research folder the build cannot list.
        fixture = self.fresh()
        bom = fixture.root / "research/2026-11-18/llmctl-fixture-e2e/summary.md"
        bom.write_bytes(b"\xef\xbb\xbf" + bom.read_bytes())
        message = self.assertStops(fixture, "llmctl-fixture-e2e/summary.md")
        self.assertTrue(message.startswith("research/2026-11-18/llmctl-fixture-e2e/summary.md: line 1: "), message)
        if os.geteuid() != 0:                                                        # root lists any folder
            fixture = self.fresh()
            locked = fixture.root / "research/2026-11-17/locked"
            locked.mkdir(parents=True)
            locked.chmod(0)
            self.addCleanup(locked.chmod, 0o755)
            self.assertEqual(self.assertStops(fixture, "locked"), "research/2026-11-17/locked: Permission denied")
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

    def test_fastest_tags(self):
        self.assertEqual(data.fastest_tags({"llamacpp": 122.5, "sglang": 151.3}), {"sglang": "Fastest by 24%"})
        self.assertEqual(data.fastest_tags({"a": 102.5, "b": 100.0}), {"a": "Fastest by 3%"})
        self.assertEqual(data.fastest_tags({"a": 201.0, "b": 200.0}), {"a": "Fastest by 1%"})
        self.assertEqual(data.fastest_tags({"a": 100.1, "b": 100.0}), {"a": "Fastest by under 1%"})
        self.assertEqual(data.fastest_tags({"a": 90.0, "b": 90.0, "c": 50.0}), {"a": "Tied fastest", "b": "Tied fastest"})
        self.assertEqual(data.fastest_tags({"a": 166.2, "b": 83.5, "c": 124.6}), {"a": "Fastest by 33%"})
        self.assertEqual((data.fastest_tags({"a": 90.0}), data.fastest_tags({"a": 90.0, "b": None})), ({}, {}))

    def test_speed_text_keeps_at_least_one_decimal(self):
        self.assertEqual([data.speed_text(x) for x in (66.0, 257.2, 70.55, 100.0, 36.4)],
                         ["66.0", "257.2", "70.55", "100.0", "36.4"])

    def test_lead_gap_and_bar(self):
        m = self.fixture.load()
        labels = {runtime.id: runtime.label for runtime in m.runtimes}
        lead = lambda matrix, model_id, runtime: data.lead(model_of(matrix, model_id), cell_of(matrix, model_id, runtime), labels)
        gap = lambda matrix, model_id: data.gap_text(model_of(matrix, model_id), labels)
        self.assertEqual((lead(m, "qwen3.6-35b-a3b", "sglang"), lead(m, "gpt-oss-20b", "llamacpp"),
                          lead(m, "qwen3.6-35b-a3b", "llamacpp")),
                         (("24%", "fastest, 24% ahead of llama.cpp"), ("33%", "fastest, 33% ahead of SGLang"), None))
        self.assertEqual([gap(m, i) for i in ("qwen3.6-35b-a3b", "gpt-oss-20b", "deepseek-r1-0528-qwen3-8b", "k2-horizon-7b")],
                         ["SGLang 24% ahead of llama.cpp", "llama.cpp 99% ahead of vLLM", "vLLM 3% ahead of llama.cpp", None])
        self.assertEqual(data.top_speed(m), 188.2)
        self.assertEqual([data.bar_width(x, 188.2) for x in (188.2, 151.3, 122.5, 83.5, None)], [100, 80, 65, 44, 0])
        self.assertEqual([data.bar_width(1.0, 8.0), data.bar_width(3.0, 8.0), data.bar_width(50.0, None),
                          data.bar_width(250.0, 200.0), data.bar_width(33.8, 104.0)],
                         [13, 38, 0, 100, 33])                    # half up, clamped, exact (32 in floats)
        with self.assertRaisesRegex(ValueError, "unknown fastest tag 'Quickest'"):
            data.lead(model_of(m, "gpt-oss-20b"),
                      dataclasses.replace(cell_of(m, "gpt-oss-20b", "llamacpp"), fastest="Quickest"), labels)
        cases = (  # gpt-oss-20b's speeds on llama.cpp, vLLM and SGLang; its gap words; the lead of each engine
            (("100.0", "112.0", "112.0"), "vLLM and SGLang 12% ahead of llama.cpp",
             (None, ("tied", "tied fastest with SGLang"), ("tied", "tied fastest with vLLM"))),
            (("150.0", "100.0", "100.0"), "llama.cpp 50% ahead of vLLM and SGLang",
             (("50%", "fastest, 50% ahead of vLLM and SGLang"), None, None)),
            (("100.4", "100.0", "100.2"), "llama.cpp under 1% ahead of vLLM",
             (("under 1%", "fastest, under 1% ahead of SGLang"), None, None)),
            (("90.0", "90.0", "90.0"), "the same speed on llama.cpp, vLLM and SGLang",
             (("tied", "tied fastest with vLLM and SGLang"), ("tied", "tied fastest with llama.cpp and SGLang"),
              ("tied", "tied fastest with llama.cpp and vLLM"))),
            (("112.5", "100.0", None), "llama.cpp 13% ahead of vLLM", (("13%", "fastest, 13% ahead of vLLM"), None, None)),
            (("51.0", "40.0", "45.0"), "llama.cpp 28% ahead of vLLM",    # exact: 27% in floats
             (("13%", "fastest, 13% ahead of SGLang"), None, None)))
        for speeds, words, leads in cases:
            fixture = self.fresh()
            for runtime, speed in zip(("llamacpp", "vllm", "sglang"), speeds):
                fixture.set_value(runtime, "gpt-oss-20b", "decode_tok_s", speed)
            matrix = fixture.load()
            with self.subTest(speeds=speeds):
                self.assertEqual(gap(matrix, "gpt-oss-20b"), words)
                self.assertEqual(tuple(lead(matrix, "gpt-oss-20b", r) for r in ("llamacpp", "vllm", "sglang")), leads)
        for runtime, recipe_id in RECIPES:
            self.fixture.set_value(runtime, recipe_id, "decode_tok_s", None)
        self.assertIsNone(data.top_speed(self.fixture.load()))

    def test_cell_and_row_orders(self):
        matrix = self.fixture.load()
        self.assertEqual(order(matrix, "rank"), [("qwen3.6-35b-a3b", "sglang"), ("qwen3.6-35b-a3b", "llamacpp"),
            ("gpt-oss-20b", "llamacpp"), ("gpt-oss-20b", "sglang"), ("gpt-oss-20b", "vllm"),
            ("deepseek-r1-0528-qwen3-8b", "vllm"), ("deepseek-r1-0528-qwen3-8b", "llamacpp"), ("mellum2.1-12b", "llamacpp"),
            ("qwen3-coder-next", "llamacpp"), ("k2-horizon-7b", "llamacpp")])
        self.assertEqual(order(matrix, "context")[5:], [("qwen3-coder-next", "llamacpp"), ("k2-horizon-7b", "llamacpp"),
            ("gpt-oss-20b", "vllm"), ("deepseek-r1-0528-qwen3-8b", "vllm"), ("deepseek-r1-0528-qwen3-8b", "llamacpp")])
        catalog, engines = data.shipped_catalog(self.fixture.root), list(ADAPTERS)
        size = lambda key: (catalog.runtimes[key[1]].recipes[key[0]].model.size_bytes, key[0], engines.index(key[1]))
        self.assertEqual(order(matrix, "size"), sorted(order(matrix, "size"), key=size))
        rows = {m.id: dict(m.positions) for m in matrix.models}
        self.assertEqual(sorted(rows, key=lambda i: rows[i]["gap"]), ["gpt-oss-20b", "qwen3.6-35b-a3b",
            "deepseek-r1-0528-qwen3-8b", "k2-horizon-7b", "mellum2.1-12b", "qwen3-coder-next"])
        self.assertEqual(sorted(rows, key=lambda i: rows[i]["name"]), sorted(rows))
        self.fixture.set_value("sglang", "gpt-oss-20b", "decode_tok_s", None)         # no speed: last within its group
        self.fixture.set_value("llamacpp", "qwen3-coder-next", "decode_tok_s", "70.6")  # a tie: model ID decides
        matrix = self.fixture.load()
        self.assertEqual(order(matrix, "rank")[2:5], [("gpt-oss-20b", "llamacpp"), ("gpt-oss-20b", "vllm"), ("gpt-oss-20b", "sglang")])
        self.assertEqual(order(matrix, "speed")[-4:], [("deepseek-r1-0528-qwen3-8b", "llamacpp"), ("qwen3-coder-next", "llamacpp"),
            ("k2-horizon-7b", "llamacpp"), ("gpt-oss-20b", "sglang")])
        self.assertEqual(order(matrix, "context")[:6], [("mellum2.1-12b", "llamacpp"), ("gpt-oss-20b", "llamacpp"),
            ("qwen3.6-35b-a3b", "sglang"), ("qwen3.6-35b-a3b", "llamacpp"), ("qwen3-coder-next", "llamacpp"), ("gpt-oss-20b", "sglang")])
        self.fixture.set_value("llamacpp", "gpt-oss-20b", "decode_tok_s", "72.5")  # a tie across models: ID before engine
        self.assertEqual(order(self.fixture.load(), "speed")[3:6], [("gpt-oss-20b", "vllm"),
            ("deepseek-r1-0528-qwen3-8b", "vllm"), ("gpt-oss-20b", "llamacpp")])

    def test_use_case_membership(self):
        matrix = self.fixture.load()
        model, cell = model_of(matrix, "qwen3.6-35b-a3b"), cell_of(matrix, "qwen3.6-35b-a3b", "llamacpp")
        recipe = lambda **changes: dataclasses.replace(cell, recipe=dataclasses.replace(cell.recipe, **changes))
        classes = lambda **changes: dataclasses.replace(cell, clients=tuple(
            dataclasses.replace(r, cls=changes.get(r.client, r.cls)) for r in cell.clients))
        uses = lambda c, m=model: data.use_cases_of(m, c)
        self.assertEqual(uses(cell), ("opencode", "quality", "fastest", "long-context", "every-client", "compare"))
        self.assertEqual([u.id for u in data.USE_CASES], ["", "opencode", "quality", "fastest", "long-context",
                                                          "every-client", "compare"])
        self.assertIn("opencode", uses(recipe(context=65536)))
        self.assertNotIn("opencode", uses(recipe(context=65535)))
        self.assertNotIn("opencode", uses(recipe(chat_only=True)))
        self.assertNotIn("opencode", uses(classes(opencode="known-issue")))
        self.assertIn("quality", uses(cell, dataclasses.replace(model, rank=1)))
        self.assertNotIn("quality", uses(cell, dataclasses.replace(model, rank=None)))
        self.assertIn("fastest", uses(recipe(decode=None, context=1024, chat_only=True)))
        self.assertIn("long-context", uses(recipe(context=131072)))
        self.assertNotIn("long-context", uses(recipe(context=130048)))
        self.assertTrue(all(r.cls == "passed" for r in cell.clients))
        self.assertIn("every-client", uses(cell))
        self.assertNotIn("every-client", uses(classes(kimi="not-supported")))
        self.assertIn("compare", uses(cell))
        unmeasured = tuple(dataclasses.replace(c, recipe=dataclasses.replace(c.recipe, decode=None))
                           if c.runtime == "sglang" else c for c in model.cells)                # Runs cells, not speeds
        self.assertIn("compare", uses(cell, dataclasses.replace(model, cells=unmeasured)))
        self.assertNotIn("compare", uses(cell, dataclasses.replace(model, cells=tuple(
            c for c in model.cells if c.runtime != "sglang"))))
        self.assertEqual(uses(cell_of(matrix, "deepseek-r1-0528-qwen3-8b", "sglang"),
                              model_of(matrix, "deepseek-r1-0528-qwen3-8b")), ())
        self.assertEqual(uses(recipe(context=65536), dataclasses.replace(model, rank=None)),
                         ("opencode", "fastest", "every-client", "compare"))
        self.assertEqual(cell.uses, uses(cell))
        self.assertEqual(data.rule_line(data.USE_CASES[5], matrix),
                         "Pairs whose latest session passed in OpenCode, Hermes, Pi, dsh and Kimi.")
        self.assertEqual(data.rule_line(data.USE_CASES[3], matrix), "Every pair that runs, fastest decode first.")

    def test_view_names_rules_and_sort_labels_come_from_the_data(self):
        m = self.fixture.load()
        self.assertEqual([data.view_name(u, m) for u in data.USE_CASES],
                         ["All models", "Coding in OpenCode", "Top 3 models", "Fastest decode", "128k context",
                          "Passed in all 5 clients", "Compare engines"])
        self.assertEqual([data.rule_line(u, m) for u in data.USE_CASES],
                         ["", "Tool calling, at least 64k of context (OpenCode's practical minimum), and the latest OpenCode "
                          "session passed.", "Models in the Top 3 that run here.", "Every pair that runs, fastest decode first.",
                          "Pairs that hold 128k tokens.", "Pairs whose latest session passed in OpenCode, Hermes, Pi, dsh and Kimi.",
                          "Models that run on two or more engines, biggest speed difference first."])
        self.assertEqual([(u.id, u.sort, u.roles) for u in data.USE_CASES],
                         [("", "rank", ("rank", "fastest", "fastest-top")), ("opencode", "rank", ("default", "rank", "fastest")),
                          ("quality", "rank", ("rank", "rank-long", "fastest-top")), ("fastest", "speed", ("engines",)),
                          ("long-context", "rank", ("rank", "fastest", "fastest-top")),
                          ("every-client", "rank", ("rank", "fastest", "size")), ("compare", "gap", ("gap", "rank", "fastest"))])
        self.assertEqual([data.sort_label(key, m) for key, _ in data.SORTS],
                         ["Top 3 rank, then speed", "Fastest first", "Longest context first", "Smallest download first",
                          "Biggest engine difference first", "Model ID, A to Z"])
        self.assertEqual({role: data.role_words(role, m) for role in data.ROLES},
                         {"rank": "Highest ranked", "fastest": "Fastest", "fastest-top": "Fastest in the Top 3",
                          "rank-long": "Highest ranked at 128k", "default": "Default in the OpenCode config",
                          "size": "Smallest download", "gap": "Biggest engine difference", "next-rank": "Next by Top 3 rank",
                          "next-speed": "Next by speed", "next-context": "Next by context", "next-size": "Next by download size",
                          "next-gap": "Next by engine difference", "next-name": "Next by model ID"})
        self.assertEqual(list(data.ROLES), ["rank", "fastest", "fastest-top", "rank-long", "default", "size", "gap", "next-rank",
                                            "next-speed", "next-context", "next-size", "next-gap", "next-name"])
        self.assertEqual([data.role_words(f"engine-{r.id}", m) for r in m.runtimes],
                         ["Fastest on llama.cpp", "Fastest on vLLM", "Fastest on SGLang"])
        for roles, label in ((("rank", "fastest", "fastest-top"), "Highest ranked and fastest"), (("rank", "rank-long"), "Highest ranked"),
                             (("gap", "rank", "fastest"), "Biggest engine difference, highest ranked and fastest"),
                             (("fastest", "engine-vllm"), "Fastest"), (("default", "rank"), "Default in the OpenCode config and highest ranked"),
                             (("next-gap",), "Next by engine difference"),
                             (("rank-long", "fastest-top"), "Highest ranked at 128k and fastest in the Top 3"),
                             (("engine-vllm",), "Fastest on vLLM")):
            self.assertEqual(data.role_label(roles, m), label)
        for bad in (lambda: data.role_words("bogus", m), lambda: data.role_words("engine-bogus", m),
                    lambda: data.sort_label("bogus", m)):
            with self.assertRaises(ValueError):
                bad()
        twelve = dataclasses.replace(m, top=12)
        self.assertEqual((data.view_name(data.USE_CASES[2], twelve), data.rule_line(data.USE_CASES[2], twelve),
                          data.sort_label("rank", twelve), data.role_words("fastest-top", twelve), data.role_words("next-rank", twelve)),
                         ("Top 12 models", "Models in the Top 12 that run here.", "Top 12 rank, then speed", "Fastest in the Top 12",
                          "Next by Top 12 rank"))
        six = dataclasses.replace(m, clients=m.clients + (data.Client("zed", "Zed", "Z", ()),))
        self.assertEqual((data.view_name(data.USE_CASES[5], six), data.rule_line(data.USE_CASES[5], six)),
                         ("Passed in all 6 clients", "Pairs whose latest session passed in OpenCode, Hermes, Pi, dsh, Kimi and Zed."))
        oc = dataclasses.replace(m, clients=tuple(dataclasses.replace(c, label="OC") if c.id == "opencode" else c for c in m.clients))
        self.assertEqual((data.view_name(data.USE_CASES[1], oc), data.rule_line(data.USE_CASES[1], oc), data.role_words("default", oc)),
                         ("Coding in OC", "Tool calling, at least 64k of context (OC's practical minimum), and the latest OC session "
                          "passed.", "Default in the OC config"))

    def test_script_strings(self):
        m, F = self.fixture.load(), data.Filters
        each = (F(engines=("vllm",)), F(engines=()), F(statuses=("runs",)), F(must_pass="kimi"), F(context=128),
                F(thinking=("toggle",)), F(tools="yes"), F(ram=20))
        self.assertEqual([data.filter_count(f) for f in (F(), F(search="qwen"), F(search=" \t "), *each)], [0, 0, 0] + [1] * 8)
        self.assertEqual(data.filter_count(F(search="q", engines=("vllm",), statuses=("runs",), must_pass="every", context=64,
                                             thinking=(), tools="no", ram=16)), 7)
        self.assertEqual((data.filters_label(F()), data.filters_label(F(ram=20, tools="no"))), ("Filters", "Filters (2)"))
        self.assertEqual([data.show_label(n) for n in (15, 1, 0)], ["Show 15 models", "Show 1 model", "Show 0 models"])
        self.assertEqual([data.match_line(n) for n in (10, 1, 0)], ["10 models match.", "1 model matches.", "0 models match."])
        head = "Decode speed, context and client results of each model on each engine. View: "
        self.assertEqual(data.caption_line(data.USE_CASES[0], "rank", m), head + "All models; sorted by: Top 3 rank, then speed.")
        self.assertEqual(data.caption_line(data.USE_CASES[6], "name", m), head + "Compare engines; sorted by: Model ID, A to Z.")
        for filters, tail in ((F(), "."), (F(search=" \t "), "."), (F(search="qwen"), "; 1 filter active."),
                              (F(search="\xa0"), "; 1 filter active."), (F(ram=20), "; 1 filter active."),
                              (F(search="qwen", ram=20, tools="no"), "; 3 filters active.")):
            with self.subTest(filters=filters):
                self.assertEqual(data.print_line(data.USE_CASES[3], "speed", filters, m),
                                 "View: Fastest decode; sorted by: Fastest first" + tail)
        two = (data.Pick(("rank", "fastest"), "Highest ranked and fastest", "qwen3.6-35b-a3b", "sglang"),
               data.Pick(("next-rank",), "Next by Top 3 rank", "qwen3.6-35b-a3b", "llamacpp"))
        self.assertEqual(data.picks_line(two, m), "Picks: Highest ranked and fastest: qwen3.6-35b-a3b on SGLang; "
                                                  "Next by Top 3 rank: qwen3.6-35b-a3b on llama.cpp.")
        self.assertEqual(data.picks_line((), m), "")

    def test_picks_follow_the_roles_of_each_view(self):
        m = self.fixture.load()
        best = ("Highest ranked", "qwen3.6-35b-a3b", "sglang")
        fast = ("Fastest", "mellum2.1-12b", "llamacpp")
        top = ("Fastest in the Top 3", "gpt-oss-20b", "llamacpp")
        runs = [k for k in data.view(m, "every-client", data.Filters(), None).cells if cell_of(m, *k).status == "runs"]
        smallest = min(runs, key=lambda k: dict(cell_of(m, *k).positions)["size"])        # ruling 20
        self.assertEqual(cards(m, ""), [best, fast, top])
        self.assertEqual(cards(m, "opencode"), [("Default in the OpenCode config", "qwen3.6-35b-a3b", "llamacpp"), best, fast])
        self.assertEqual(cards(m, "quality"), [best, top, ("Next by Top 3 rank", "qwen3.6-35b-a3b", "llamacpp")])
        self.assertEqual(cards(m, "fastest"), [("Fastest on llama.cpp", "mellum2.1-12b", "llamacpp"),
                                               ("Fastest on SGLang", "qwen3.6-35b-a3b", "sglang"), ("Fastest on vLLM", "gpt-oss-20b", "vllm")])
        self.assertEqual(cards(m, "long-context"), [best, fast, top])                       # ruling 6
        self.assertEqual(cards(m, "every-client"), [best, fast, ("Smallest download", *smallest)])
        self.assertEqual(cards(m, "compare"), [("Biggest engine difference and fastest", "gpt-oss-20b", "llamacpp"), best,
                                               ("Next by engine difference", "deepseek-r1-0528-qwen3-8b", "vllm")])
        self.assertEqual(len({frozenset(card[1:] for card in cards(m, use)) for use in ("", "opencode", "quality")}), 3)
        roles = lambda use: [p.roles for p in data.picks(m, use, data.view(m, use, data.Filters(), None).cells,
                                                          next(u.sort for u in data.USE_CASES if u.id == use))]
        self.assertEqual(roles("quality"), [("rank", "rank-long"), ("fastest-top",), ("next-rank",)])
        self.assertEqual(roles("compare"), [("gap", "fastest"), ("rank",), ("next-gap",)])
        self.assertEqual(roles("fastest"), [("engine-llamacpp",), ("engine-sglang",), ("engine-vllm",)])

    def test_picks_merge_roles_and_fill_from_the_view_order(self):
        m, F = self.fixture.load(), data.Filters
        gpt = data.picks(m, "", data.view(m, "", F(search="gpt-oss"), None).cells, "rank")
        self.assertEqual([(p.roles, p.label, p.runtime) for p in gpt],
                         [(("rank", "fastest", "fastest-top"), "Highest ranked and fastest", "llamacpp"),
                          (("next-rank",), "Next by Top 3 rank", "sglang"), (("next-rank",), "Next by Top 3 rank", "vllm")])
        cells = data.view(m, "compare", F(search="gpt-oss"), None).cells
        self.assertEqual([(p.roles, p.label, p.model, p.runtime) for p in data.picks(m, "compare", cells, "gap")],
                         [(("gap", "rank", "fastest"), "Biggest engine difference, highest ranked and fastest", "gpt-oss-20b",
                           "llamacpp")])                                                  # next-gap: one cell per model
        self.assertEqual([(p.roles, p.runtime) for p in data.picks(m, "compare", cells, "rank")][1:],
                         [(("next-rank",), "sglang"), (("next-rank",), "vllm")])
        self.assertEqual(cards(m, "quality", sort="name")[2], ("Next by model ID", "deepseek-r1-0528-qwen3-8b", "vllm"))
        self.assertEqual(cards(m, "every-client", sort="gap"), cards(m, "every-client"))
        best, fast = ("Highest ranked", "qwen3.6-35b-a3b", "sglang"), ("Fastest", "mellum2.1-12b", "llamacpp")
        sized = tuple(dataclasses.replace(u, roles=("size", "rank", "fastest", "fastest-top")) if not u.id else u
                      for u in data.USE_CASES)
        with mock.patch.object(data, "USE_CASES", sized):
            found = cards(m, "")
        self.assertEqual((len(found), found[0][0], found[1:]), (3, "Smallest download", [best, fast]))
        for use in data.USE_CASES:
            for key, _ in data.SORTS:
                with self.subTest(use=use.id, sort=key):
                    found = cards(m, use.id, sort=key)
                    self.assertLessEqual(len(found), 3)
                    self.assertEqual(len({card[1:] for card in found}), len(found))
        for use, sort in (("bogus", "rank"), ("", "bogus")):
            with self.subTest(use=use, sort=sort):
                with self.assertRaises(ValueError):
                    data.picks(m, use, (), sort)
        bogus = tuple(dataclasses.replace(u, roles=("bogus",)) if not u.id else u for u in data.USE_CASES)
        with mock.patch.object(data, "USE_CASES", bogus):
            with self.assertRaises(ValueError):
                data.picks(m, "", (), "rank")                                             # the role loop runs without cells

    def test_picks_of_filtered_views(self):
        m, F = self.fixture.load(), data.Filters
        self.assertEqual(cards(m, "", F(search="qwen3.6")), [("Highest ranked and fastest", "qwen3.6-35b-a3b", "sglang"),
                                                              ("Next by Top 3 rank", "qwen3.6-35b-a3b", "llamacpp")])
        self.assertEqual(cards(m, "", F(search="mellum")), [("Fastest", "mellum2.1-12b", "llamacpp")])  # unranked, one pair
        self.assertEqual(cards(m, "fastest", F(engines=("llamacpp",))), [("Fastest on llama.cpp", "mellum2.1-12b", "llamacpp"),
                         ("Next by speed", "gpt-oss-20b", "llamacpp"), ("Next by speed", "qwen3.6-35b-a3b", "llamacpp")])
        self.assertEqual(cards(m, "opencode", F(engines=("sglang",))),                      # the default pair out of view
                         [("Highest ranked and fastest", "qwen3.6-35b-a3b", "sglang"), ("Next by Top 3 rank", "gpt-oss-20b", "sglang")])
        self.assertEqual(cards(m, "", F(engines=("vllm",))), [("Highest ranked and fastest", "gpt-oss-20b", "vllm"),
                                                               ("Next by Top 3 rank", "deepseek-r1-0528-qwen3-8b", "vllm")])
        self.assertEqual(cards(m, "", F(statuses=("failed", "does-not-fit"))), [])
        self.assertEqual(cards(m, "", F(search="zzz")), [])
        self.assertEqual(cards(m, "compare", F(engines=("llamacpp",))),                    # no row has two in-view speeds
                         [("Highest ranked", "qwen3.6-35b-a3b", "llamacpp"), ("Fastest", "gpt-oss-20b", "llamacpp"),
                          ("Next by engine difference", "deepseek-r1-0528-qwen3-8b", "llamacpp")])
        self.assertEqual(cards(m, "", F(engines=("llamacpp",), search="qwen3"), sort="gap"),  # fill in the row order
                         [("Highest ranked and fastest", "qwen3.6-35b-a3b", "llamacpp"),
                          ("Next by engine difference", "deepseek-r1-0528-qwen3-8b", "llamacpp"),
                          ("Next by engine difference", "qwen3-coder-next", "llamacpp")])
        fixture = self.fresh()
        fixture.set_value("vllm", "gpt-oss-20b", "decode_tok_s", None)
        m2 = fixture.load()
        self.assertEqual(cards(m2, "compare", F(engines=("vllm", "sglang"))),               # a gap needs two speeds
                         [("Highest ranked and fastest", "qwen3.6-35b-a3b", "sglang"),
                          ("Next by engine difference", "gpt-oss-20b", "sglang"),
                          ("Next by engine difference", "deepseek-r1-0528-qwen3-8b", "vllm")])
        self.assertEqual(cards(m2, "fastest", F(engines=("vllm",), search="gpt-oss")),      # an engine pick needs a speed
                         [("Next by speed", "gpt-oss-20b", "vllm")])

    def test_view_orders_by_in_view_cells(self):
        m = self.fixture.load()
        v = data.view(m, "", data.Filters(engines=("vllm",)), "speed")
        self.assertEqual(v.rows, ("gpt-oss-20b", "deepseek-r1-0528-qwen3-8b", "k2-horizon-7b", "mellum2.1-12b",
                                  "qwen3-coder-next", "qwen3.6-35b-a3b"))
        self.assertEqual(v.answer, "All models: 6 models, 2 recipes that run. Best quality: gpt-oss-20b on vLLM "
                         "(rank 2), 83.5 tok/s, 64k. Fastest: gpt-oss-20b on vLLM, 83.5 tok/s, 64k.")
        self.assertEqual(data.view(m, "", data.Filters(), None).answer, "All models: 6 models, 10 recipes that run. "
                         "Best quality: qwen3.6-35b-a3b on SGLang (rank 1), 151.3 tok/s, 128k. "
                         "Fastest: mellum2.1-12b on llama.cpp, 188.2 tok/s, 128k.")
        self.assertEqual(data.view(m, "", data.Filters(engines=("llamacpp",), search=" K2 "), None).answer,
                         "All models: 1 model, 1 recipe that runs. Best quality: no ranked model is in this view. "
                         "Fastest: k2-horizon-7b on llama.cpp, 66.9 tok/s, 80k.")
        self.assertEqual(data.view(m, "", data.Filters(statuses=("failed",)), None).answer,
                         "All models: 1 model, 0 recipes that run. Best quality: no ranked model is in this view.")
        empty = data.view(m, "", data.Filters(search="no such model"), None)
        self.assertEqual((empty.rows, empty.answer), ((), "No recipe matches this view."))
        self.assertEqual(data.view(m, "compare", data.Filters(), None).answer, "Compare engines: 3 models. "
                         "Biggest difference: gpt-oss-20b, 166.2 tok/s on llama.cpp against 83.5 on vLLM.")
        best = data.view(m, "", data.Filters(), None).phrases[0]
        self.assertEqual((best.kind, best.target, best.link), ("best", "d-sglang-qwen3.6-35b-a3b", "qwen3.6-35b-a3b on SGLang"))
        cells = lambda **f: data.view(m, "", data.Filters(**f), None).cells
        deepseek = ("deepseek-r1-0528-qwen3-8b", "vllm")
        self.assertEqual(cells(tools="no"), (deepseek, ("deepseek-r1-0528-qwen3-8b", "llamacpp")))  # the Failed cell leaves
        self.assertEqual(len(cells(tools="yes")), 8)
        self.assertEqual((len(cells(context=128)), len(cells(context=64))), (6, 10))           # 65536 tokens is 64k
        self.assertIn(("qwen3.6-35b-a3b", "sglang"), cells(ram=27))                             # a cap equal to the value
        self.assertNotIn(("qwen3.6-35b-a3b", "sglang"), cells(ram=20))
        self.assertEqual(len(cells(ram=20)), 6)
        self.assertEqual(cells(thinking=("none",)), (("qwen3-coder-next", "llamacpp"),))
        self.assertEqual(len(cells(thinking=("always", "effort"))), 6)
        self.assertNotIn(deepseek, cells(must_pass="opencode"))                                # OpenCode not supported
        self.assertIn(deepseek, cells(must_pass="hermes"))
        self.assertEqual((len(cells(must_pass="every")), len(cells(must_pass="hermes"))), (9, 10))
        self.assertEqual(data.view(m, "", data.Filters(search=" 3.6 "), None).rows, ("qwen3.6-35b-a3b",))
        # The search trims ASCII whitespace only, as page.js does; other spaces stay part of the search.
        self.assertEqual(data.view(m, "", data.Filters(search="\t\n3.6\r\f\v"), None).rows, ("qwen3.6-35b-a3b",))
        for search in ("\ufeff3.6", "\x853.6", "\xa03.6"):
            with self.subTest(search=search):
                self.assertEqual(data.view(m, "", data.Filters(search=search), None).rows, ())
        gpt = lambda use, sort: [c[1] for c in data.view(m, use, data.Filters(), sort).cells if c[0] == "gpt-oss-20b"]
        self.assertEqual(gpt("", "speed"), ["llamacpp", "sglang", "vllm"])
        self.assertEqual(gpt("compare", None), ["llamacpp", "vllm", "sglang"])
        self.assertEqual(data.view(m, "", data.Filters(), "gap").answer, data.view(m, "", data.Filters(), None).answer)
        self.assertTrue(data.view(m, "compare", data.Filters(), "rank").answer.startswith("Compare engines: 3 models. "))
        for use, sort, filters in (("bogus", None, data.Filters()), ("", "bogus", data.Filters()),
                                   ("", None, data.Filters(must_pass="bogus")), ("", None, data.Filters(tools="maybe"))):
            with self.subTest(use=use, sort=sort, filters=filters):
                with self.assertRaises(ValueError):
                    data.view(m, use, filters, sort)
        found = data.phrases(m)
        kinds = [p.kind for p in found]
        self.assertEqual((kinds.count("best"), kinds.count("fastest"), kinds.count("gap")), (7, 10, 3))
        self.assertEqual([p.target for p in found if p.kind == "gap"],
                         ["m-deepseek-r1-0528-qwen3-8b", "m-gpt-oss-20b", "m-qwen3.6-35b-a3b"])
        self.assertEqual(kinds[-2:], ["no-best", "empty"])
        runs = {f"d-{c.runtime}-{model.id}" for model in m.models for c in model.cells if c.status == "runs"}
        self.assertTrue(all(p.target in runs for p in found if p.kind in ("best", "fastest")))
        self.assertIn(best, found)
        self.fixture.set_value("vllm", "gpt-oss-20b", "decode_tok_s", "124.6")  # a tie for slowest: the last engine
        gap = [p.text for p in data.phrases(self.fixture.load()) if p.target == "m-gpt-oss-20b"]
        self.assertEqual(gap, ["Biggest difference: gpt-oss-20b, 166.2 tok/s on llama.cpp against 124.6 on SGLang."])

    def test_missing_speed_is_shown_as_absent(self):
        self.fixture.set_value("vllm", "gpt-oss-20b", "decode_tok_s", None)
        m = self.fixture.load()
        cell = cell_of(m, "gpt-oss-20b", "vllm")
        self.assertEqual((cell.recipe.decode, cell.fastest), (None, None))
        self.assertEqual(order(m, "speed")[-1], ("gpt-oss-20b", "vllm"))
        speeds = {c.runtime: c.recipe.decode for c in model_of(m, "gpt-oss-20b").cells if c.recipe}
        self.assertEqual(data.fastest_tags(speeds), {"llamacpp": "Fastest by 33%"})
        self.assertEqual(cell_of(m, "gpt-oss-20b", "llamacpp").fastest, "Fastest by 33%")
        fastest = [p for p in data.view(m, "", data.Filters(engines=("vllm",)), "speed").phrases if p.kind == "fastest"]
        self.assertEqual([p.link for p in fastest], ["deepseek-r1-0528-qwen3-8b on vLLM"])
        self.assertEqual(data.view(m, "", data.Filters(engines=("vllm",), search="gpt-oss"), None).answer,
                         "All models: 1 model, 1 recipe that runs. "
                         "Best quality: gpt-oss-20b on vLLM (rank 2), speed not recorded, 64k.")
        self.assertNotIn("d-vllm-gpt-oss-20b", [p.target for p in data.phrases(m) if p.kind == "fastest"])

    def test_opencode_use_case_follows_the_context_budget(self):
        text = "\n".join(markdown.section(REPO, "docs/opencode.md#context-budget"))
        self.assertIn("The practical minimum for OpenCode is 64k tokens", text,
                      "docs/opencode.md#context-budget changed: update data.OPENCODE_MINIMUM")
        self.assertEqual(data.OPENCODE_MINIMUM, 64 * 1024, "update data.OPENCODE_MINIMUM")


if __name__ == "__main__":
    unittest.main()
