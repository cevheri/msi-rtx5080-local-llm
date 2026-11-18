"""The runtime matrix page's renderer: the Markdown subset of the coverage notes, repository links at the build's
commit (design D12), and the README recipe table (D-README)."""

import ast
import base64
import contextlib
import hashlib
import io
import re
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from launcher import clients
from launcher.runtimes import ADAPTERS
from launcher.site import data, render
from launcher.tests.fakes import REPO
from launcher.tests.site_fixture import (RECIPES, TOP10_IDS, TOP10_ROW, DEFAULT_RUN, SiteFixture, default_rows,
                                         parse, prefix, row, version_record, well_formed_problems)

BUILD = data.Build("https://github.com/o/r", "a" * 40, 0)
BLOB = "https://github.com/o/r/blob/" + "a" * 40
TREE = "https://github.com/o/r/tree/" + "a" * 40


class SiteRenderTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.fixture = SiteFixture(Path(self._tmp.name))
        self.links = render.Links(self.fixture.root, BUILD)

    def test_inline_renders_the_supported_subset(self):
        links = self.links
        text = ('Runs: `IQ2_XXS` & "x" Q8_0, see [recipe](../llamacpp/models.md#2-qwen36-35b-a3b), '
                '[cov](#runtime-coverage) and [hf](https://huggingface.co/a/b?c=1&d=2).')
        self.assertEqual(render.inline(text, "docs/model-selection.md", links),
                         'Runs: <code>IQ2_XXS</code> &amp; &quot;x&quot; Q8_0, see <a href="' + BLOB
                         + '/llamacpp/models.md#2-qwen36-35b-a3b">recipe</a>, <a href="' + BLOB
                         + '/docs/model-selection.md#runtime-coverage">cov</a> and '
                         '<a href="https://huggingface.co/a/b?c=1&amp;d=2">hf</a>.')
        self.assertEqual(render.inline('`a<b & "c"` and [x & "y"](#runtime-coverage)', "docs/model-selection.md", links),
                         '<code>a&lt;b &amp; &quot;c&quot;</code> and <a href="' + BLOB
                         + '/docs/model-selection.md#runtime-coverage">x &amp; &quot;y&quot;</a>')
        self.assertEqual(render.inline("`https://x.org` and `a@b.org` and `www.x`", "docs/model-selection.md", links),
                         "<code>https://x.org</code> and <code>a@b.org</code> and <code>www.x</code>")

    def test_inline_refuses_everything_else(self):
        links = self.links
        cases = {"a **b**": 3, "a *b*": 3, "a ~b": 3, "a <b>": 3, "a \\ b": 3, "a ![x](y.md)": 3, "a [b": 3,
                 "a ]b": 3, "a `b": 3, "_x_": 1, "[a](http://x.org)": 1, "[a](mailto:x@y.org)": 1,
                 "[a](/abs.md)": 1, "a &amp; b": 3, "a &#38; b": 3,                                 # ruling 9
                 "a [b](../missing.md)": 3, "a [b *c*](#runtime-coverage)": 6, "a > b": 3,
                 "a [b](https://x.org/q?a=1&amp;b=2)": 3, "a [b](https://x.org/&#x41;)": 3,
                 "a [b](https://x.org/a\\_b)": 3, "a `` b": 3, "``x``": 1, "a `x``y` b": 5,
                 "see https://example.com/x now": 5, "see http://x.org": 5, "see www.x.org": 5,
                 "mail x@y.org": 6, "`a`_b": 4}
        for text, column in cases.items():
            with self.subTest(text=text), self.assertRaisesRegex(
                    data.SiteError, rf"^docs/model-selection\.md: row m, vLLM: column {column}: "):
                render.inline(text, "docs/model-selection.md", links, "row m, vLLM: ")
        messages = {"a > b": "a > outside a code span", "a [b](https://x.org/&#x41;)": "an HTML entity in a link target",
                    "a [b](https://x.org/a\\_b)": "a backslash in a link target",
                    "a `` b": "a run of two or more backticks", "see www.x.org": "a bare URL or email address",
                    "mail x@y.org": "a bare URL or email address"}
        for text, message in messages.items():
            with self.subTest(text=text), self.assertRaisesRegex(data.SiteError, re.escape(message) + "$"):
                render.inline(text, "docs/model-selection.md", links)

    def test_links_point_at_the_commit(self):
        links = self.links
        self.assertEqual(links.repo_link("docs/model-selection.md", "runtime-coverage"),
                         BLOB + "/docs/model-selection.md#runtime-coverage")
        self.assertEqual(links.repo_link("research/2026-11-17/fixture-failure"),
                         TREE + "/research/2026-11-17/fixture-failure")
        self.assertEqual(links.repo_link(""), "https://github.com/o/r/tree/" + "a" * 40)
        self.assertEqual(links.repo_link("research/2026-11-17/fixture-failure/"),
                         TREE + "/research/2026-11-17/fixture-failure")
        for path, message in (("docs/model-selection.md/", "docs/model-selection.md/: a trailing / on a file"),
                              (".", ".: not a normalized repository path"),
                              ("research//2026-11-17", "research//2026-11-17: not a normalized repository path"),
                              ("LICENSE", "LICENSE: an anchor (#x) on a link to a file that is not Markdown"),
                              ("../outside.md", "../outside.md: not a path inside the repository")):
            with self.subTest(path=path), self.assertRaisesRegex(data.SiteError, "^" + re.escape(message) + "$"):
                links.repo_link(path, "x" if path == "LICENSE" else None)
        with self.assertRaisesRegex(data.SiteError,
                                    "^docs/model-selection.md: the link ../../outside.md leaves the repository$"):
            links.resolve("docs/model-selection.md", "../../outside.md")
        (self.fixture.root / "docs/latin1.md").write_bytes(b"# caf\xe9\n")
        with self.assertRaisesRegex(data.SiteError, "^docs/latin1.md: not valid UTF-8: "):
            links.repo_link("docs/latin1.md", "x")
        folder = self.fixture.root / "research/2026-11-18/a b#c"
        folder.mkdir(parents=True)
        (folder / "README.md").write_text("# R\n", encoding="utf-8")
        self.assertEqual(links.resolve("docs/model-selection.md", "../research/2026-11-18/a%20b%23c/README.md"),
                         BLOB + "/research/2026-11-18/a%20b%23c/README.md")
        for target in ("../../outside.md", "../missing.md", "../llamacpp/models.md#no-such-anchor",
                       "../recipes/llamacpp/qwen3.6-35b-a3b.toml#L1"):
            with self.subTest(target=target), self.assertRaises(data.SiteError):
                links.resolve("docs/model-selection.md", target)
        for path, anchor in (("docs/model-selection.md", "no-such-anchor"), ("docs/missing.md", None),
                             ("LICENSE", "x"), ("../outside.md", None)):
            with self.subTest(path=path), self.assertRaisesRegex(data.SiteError, "^" + re.escape(path)):
                links.repo_link(path, anchor)                                                 # ruling 2

    def test_readme_table_renders_from_the_catalog(self):
        fixture = self.fixture
        fixture.set_value("llamacpp", "k2-horizon-7b", "decode_tok_s", None)
        fixture.set_value("llamacpp", "qwen3-coder-next", "decode_tok_s", "66.0")
        fixture.set_value("llamacpp", "qwen3.6-35b-a3b", "aliases", '["q36l", "qwen3.6"]')
        fixture.set_value("sglang", "qwen3.6-35b-a3b", "aliases", '["qwen3.6", "q36"]')
        fixture.set_value("llamacpp", "mellum2.1-12b", "aliases", None)
        catalog = data.shipped_catalog(fixture.root)
        table = render.readme_table(catalog)
        self.assertTrue(table.endswith("\n"))
        lines = table.splitlines()
        self.assertEqual(lines[0], "| Model ID | Aliases | llama.cpp | Context | Decode tok/s | vLLM | Context "
                                   "| Decode tok/s | SGLang | Context | Decode tok/s |")
        self.assertEqual(lines[1], "|---" * 11 + "|")
        rows = {line.split(" | ")[0].removeprefix("| `").removesuffix("`"): line for line in lines[2:]}
        self.assertEqual(list(rows), sorted({recipe_id for _, recipe_id in RECIPES}))
        k2 = catalog.runtimes["llamacpp"].recipes["k2-horizon-7b"]
        self.assertEqual(rows["k2-horizon-7b"], "| `k2-horizon-7b` | " + ", ".join(f"`{a}`" for a in k2.aliases)
                         + f" | Yes | {k2.context // 1024}k | - | No | - | - | No | - | - |")
        self.assertTrue(rows["qwen3-coder-next"].endswith("| Yes | 128k | 66 | No | - | - | No | - | - |"))
        self.assertEqual(rows["qwen3.6-35b-a3b"],
                         "| `qwen3.6-35b-a3b` | `q36l`, `qwen3.6`, `q36` | Yes | 128k | 122.5 | No | - | - "
                         "| Yes | 128k | 151.3 |")
        self.assertEqual(rows["mellum2.1-12b"].split(" | ")[1], "-")
        fixture.set_value("llamacpp", "k2-horizon-7b", "context", "100000")
        with self.assertRaisesRegex(data.SiteError, r"^recipes/llamacpp/k2-horizon-7b\.toml: "):
            render.readme_table(data.shipped_catalog(fixture.root))

    def test_rewrite_readme_touches_only_the_marked_block(self):
        start, end = render.README_START, render.README_END
        before = "# T\n\nintro\n\n## Quick start\n\nRun it.\n\n"
        after = "\n\nMore.\n\n## License\n\nMIT\n"
        text = before + start + "\n\nold\n\n" + end + after
        rewritten = render.rewrite_readme(text, "T\n")
        self.assertEqual(rewritten, before + start + "\n\nT\n\n" + end + after)
        self.assertEqual(render.readme_block(rewritten), "\nT\n\n")
        self.assertEqual(render.readme_block(text), "\nold\n\n")
        quick = "## Quick start\n\n"
        broken = {"missing start": quick + "x\n" + end + "\n",
                  "missing end": quick + start + "\nx\n",
                  "doubled start": quick + start + "\n" + start + "\nx\n" + end + "\n",
                  "doubled end": quick + start + "\nx\n" + end + "\n" + end + "\n",
                  "reversed": quick + end + "\nx\n" + start + "\n",
                  "before the section": start + "\n" + quick + "x\n" + end + "\n",
                  "after the section": quick + start + "\nx\n## License\n\n" + end + "\n",
                  "no section": start + "\nx\n" + end + "\n"}
        for name, broken_text in broken.items():
            with self.subTest(name=name):
                with self.assertRaisesRegex(data.SiteError, "^README.md: "):
                    render.readme_block(broken_text)
                with self.assertRaisesRegex(data.SiteError, "^README.md: "):
                    render.rewrite_readme(broken_text, "T\n")
        with self.assertRaisesRegex(data.SiteError, "^README.md: the recipe table's end marker comes before its start "
                                                    "marker$"):
            render.readme_block(broken["reversed"])

    def test_cli_readme_rewrites_then_reports_current(self):
        from launcher.site import cli
        root = self.fixture.root
        stamps = []
        for expected in ("README.md: recipe table rewritten\n", "README.md: recipe table is current\n"):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(cli.main(["readme"], root=root), 0)
            self.assertEqual(out.getvalue(), expected)
            stat = (root / "README.md").stat()
            stamps.append((stat.st_ino, stat.st_mtime_ns))
        self.assertEqual(stamps[0], stamps[1], "a current README.md must not be written again")
        text = (root / "README.md").read_text(encoding="utf-8")
        table = render.readme_table(data.shipped_catalog(root))
        self.assertEqual(render.readme_block(text), "\n" + table + "\n")
        self.assertEqual(table.splitlines()[0].count(" | Context | "), len(ADAPTERS))
        (root / "README.md").write_text(text.replace(render.README_END, ""), encoding="utf-8")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(cli.main(["readme"], root=root), 1)
        self.assertTrue(err.getvalue().startswith("site: README.md: "), err.getvalue())
        for content, message in ((b"# caf\xe9\n", "site: README.md: not valid UTF-8: "),
                                 (None, "site: README.md: missing\n")):
            readme = root / "README.md"
            if content is None:
                readme.unlink()
            else:
                readme.write_bytes(content)
            err = io.StringIO()
            with self.subTest(message=message), contextlib.redirect_stderr(err):
                self.assertEqual(cli.main(["readme"], root=root), 1)
                self.assertTrue(err.getvalue().startswith(message), err.getvalue())

    def page(self) -> str:
        return render.page(self.fixture.load(), BUILD, self.fixture.root)

    def build(self, out: Path, commit: str = "a" * 40, time: str = "0") -> tuple[int, str, str]:
        from launcher.site import cli
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = cli.main(["build", "--out", str(out), "--commit", commit, "--commit-time", time],
                            root=self.fixture.root)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_page_is_well_formed(self):
        self.assertEqual(well_formed_problems(self.page()), [])
        problems = well_formed_problems('<p><b>x</p></b><i id="x"></i><i id="x"></i><a href="#y"></a><br/>')
        self.assertEqual(problems, ["</p> closes <b> out of order", "<br ... /> is written as a self-closing tag",
                                    "<p> is never closed", "the ID x is used 2 times",
                                    "0 h1 elements", "href=#y names no element"])

    def test_page_reads_without_javascript(self):
        matrix = self.fixture.load()
        tree = parse(render.page(matrix, BUILD, self.fixture.root))
        words = {"runs": "Runs", "failed": "Failed", "does-not-fit": "Does not fit"}
        table = tree.by_id("matrix")
        for model in matrix.models:
            for cell in model.cells:
                with self.subTest(model=model.id, runtime=cell.runtime):
                    td = next(node for node in tree.by_id("m-" + model.id).find_all("td")
                              if node.attrs["data-engine"] == cell.runtime)
                    self.assertIn(words[cell.status], td.find_all("span", "status")[0].text())
                    section = tree.by_id(f"d-{cell.runtime}-{model.id}")
                    self.assertIsNotNone(section)
                    self.assertIn("details", section.attrs["class"].split())
                    self.assertEqual((section.attrs["data-model"], section.attrs["data-engine"]),
                                     (model.id, cell.runtime))
        squash = lambda node: " ".join(node.text().split())
        self.assertEqual(squash(tree.by_id("answer")), data.view(matrix, "", data.Filters(), None).answer)
        labels = [clients.LABELS[name] for name in clients.NAMES]

        def check_uses(matrix, tree):
            """The static use-case lists (ruling 10) and the phrase template (rulings 5 and 6) of one page."""
            engine_labels = {runtime.id: runtime.label for runtime in matrix.runtimes}
            models = {model.id: model for model in matrix.models}
            uses = {node.attrs["data-use"]: node for node in tree.find_all("section", "use")}
            self.assertEqual(list(uses), [use.id for use in data.USE_CASES[1:]])
            for use in data.USE_CASES[1:]:
                with self.subTest(use=use.id):
                    section, found = uses[use.id], data.view(matrix, use.id, data.Filters(), None)
                    self.assertEqual(squash(section.find_all("h3")[0]), use.chip)
                    self.assertEqual(squash(section.find_all("p", "rule")[0]), data.rule_line(use, labels))
                    self.assertEqual(squash(section.find_all("p", "answer")[0]), found.answer)
                    expected = []
                    for model_id in found.rows:
                        cells = {cell.runtime: cell for cell in models[model_id].cells}
                        runtimes = [runtime for cell_model, runtime in found.cells if cell_model == model_id]
                        items = []
                        for runtime in runtimes:
                            facts = cells[runtime].recipe
                            speed = "speed not recorded" if facts.decode is None else f"{facts.decode:g} tok/s"
                            items.append(f"{engine_labels[runtime]} {speed}, {facts.context // 1024}k")
                        expected.append((f"{model_id}: " + "; ".join(items),
                                         [f"#d-{runtime}-{model_id}" for runtime in runtimes]))
                    self.assertEqual([(squash(li), [a.attrs["href"] for a in li.find_all("a")])
                                      for li in section.find_all("li")], expected)
                    self.assertEqual(len(section.find_all("ol")), 1 if found.rows else 0)
            spans = tree.by_id("phrases").find_all("span")
            expected = [(phrase.kind, phrase.target or "",
                         phrase.text + (" Clear filters" if phrase.kind == "empty" else ""),
                         ["#" + phrase.target] if phrase.link else []) for phrase in data.phrases(matrix)]
            expected += [("rule", use.id, data.rule_line(use, labels), []) for use in data.USE_CASES]
            self.assertEqual([(span.attrs["data-kind"], span.attrs["data-for"], squash(span),
                               [a.attrs["href"] for a in span.find_all("a")]) for span in spans], expected)
            empty = next(span for span in spans if span.attrs["data-kind"] == "empty")
            self.assertEqual([(node.attrs, node.text()) for node in empty.find_all("button")],
                             [({"type": "button", "class": "clear"}, "Clear filters")])

        check_uses(matrix, tree)
        self.assertEqual(squash(tree.by_id("caption")), "Each model on each engine, sorted by: Quality rank, then speed.")
        self.assertEqual(squash(tree.by_id("print-view")), "View: All results; sorted by: Quality rank, then speed.")
        self.assertEqual(squash(tree.by_id("use-rule")), data.rule_line(data.USE_CASES[0], labels))
        fieldset = tree.by_id("uses")
        self.assertEqual([(node.attrs["value"], node.attrs["data-sort"], "checked" in node.attrs)
                          for node in fieldset.find_all("input")],
                         [(use.id, use.sort, not use.id) for use in data.USE_CASES])
        self.assertEqual([squash(node) for node in fieldset.find_all("label")], [use.chip for use in data.USE_CASES])
        self.assertEqual(squash(tree.by_id("filters-label")), "Filters")
        live = tree.by_id("live")
        self.assertEqual((live.tag, live.attrs, live.text()), ("p", {"id": "live", "class": "vh", "aria-live": "polite"}, ""))
        runs = [cell for model in matrix.models for cell in model.cells if cell.status == "runs"]

        def boxes(name):
            return [(node.attrs["value"], "checked" in node.attrs) for node in tree.find_all("input")
                    if node.attrs.get("name") == name]

        def options(name):
            return [(node.attrs["value"], squash(node), "selected" in node.attrs)
                    for node in tree.by_id(name).find_all("option")]

        self.assertEqual(boxes("engine"), [(runtime.id, True) for runtime in matrix.runtimes])
        self.assertEqual(boxes("status"), [(status, True) for _, status in data.STATUS_WORDS])
        self.assertEqual(boxes("think"), [(kind, True) for kind in data.THINKING
                                          if kind in {cell.recipe.thinking for cell in runs}])
        self.assertEqual(options("pass"), [("", "Any client", True), *((name, clients.LABELS[name], False)
                                                                         for name in clients.NAMES),
                                           ("every", "Every client", False)])
        self.assertEqual(options("ctx"), [("", "Any", True), *((str(k), f"{k}k or more", False)
                                                                for k in sorted({cell.recipe.context // 1024
                                                                                 for cell in runs}))])
        self.assertEqual([value for value, _, _ in options("tools")], ["", "yes", "no"])
        self.assertEqual(options("ram"), [("", "Any", True), *((str(cap), f"{cap} GiB", False)
                                                                for cap in sorted({cell.recipe.memory_cap
                                                                                   for cell in runs}))])
        self.assertEqual(options("sort"), [(key, label, key == "rank") for key, label in data.SORTS])
        legend = tree.by_id("legend")
        self.assertEqual([squash(li) for li in legend.find_all("ul", "legend-classes")[0].find_all("li")],
                         [f"{word.capitalize()}: {render.CLASS_MEANINGS[cls]}." for cls, word in render.CLASS_WORDS.items()])
        self.assertEqual([squash(li) for li in legend.find_all("ul", "legend-statuses")[0].find_all("li")],
                         [*(f"{label}: {render.STATUS_MEANINGS[status]}." for status, label in render.STATUS_LABELS.items()),
                          "* Fastest by n%: the fastest engine of a model that runs on two or more, ahead of the next "
                          "fastest by that much decode speed.",
                          "Setup check failed: a setup step of the recipe failed in an e2e run of its newest date; the "
                          "details list it."])
        self.assertEqual(table.attrs["role"], "table")
        self.assertEqual([node.attrs.get("role") for node in table.find_all("thead") + table.find_all("tbody")],
                         ["rowgroup", "rowgroup"])
        self.assertEqual({node.attrs.get("role") for node in table.find_all("tr")}, {"row"})
        self.assertEqual({(node.attrs["scope"], node.attrs.get("role")) for node in table.find_all("th")},
                         {("col", "columnheader"), ("row", "rowheader")})
        self.assertEqual([node.attrs["aria-labelledby"] for node in tree.find_all("section")
                          if "aria-labelledby" in node.attrs], [])
        page = render.page(matrix, BUILD, self.fixture.root)
        self.assertIn("Filters and use cases need JavaScript; the full matrix and each use case as a list are below.",
                      page)
        self.assertEqual(squash(tree.by_id("legend").find_all("summary")[0]),
                         "Legend: O OpenCode, H Hermes, P Pi, D dsh, K Kimi")
        heads = [node for node in table.find_all("thead")[0].find_all("th") if node.attrs.get("data-engine")]
        for head, start, count in zip(heads, ("llama.cpp b12407", "vLLM v0.33.0", "SGLang v0.5.23"),
                                      ("6 recipes", "2 recipes", "2 recipes")):
            self.assertTrue(squash(head).startswith(start), squash(head))
            self.assertEqual(squash(head.find_all("span", "count")[0]), count)
        self.assertEqual(len(heads), len(ADAPTERS))
        self.fixture.run("2026-11-19", "llmctl-pi-e2e", [line for runtime, recipe_id in RECIPES
                                                         for line in (row(prefix(runtime, recipe_id), "up"),
                                                                      row(prefix(runtime, recipe_id), "pi", "no answer"))])
        matrix = self.fixture.load()
        tree = parse(render.page(matrix, BUILD, self.fixture.root))
        self.assertEqual(data.view(matrix, "every-client", data.Filters(), None).answer, "No recipe matches this view.")
        check_uses(matrix, tree)

    def test_page_loads_nothing_from_other_sites(self):
        page = self.page()
        tree = parse(page)
        self.assertNotIn("src=", page)
        self.assertEqual(re.findall(r"<link\b[^>]*>", page), ['<link rel="icon" href="data:,">'])
        self.assertNotIn("@import", page)
        self.assertEqual(re.findall(r"url\((?!data:)", page), [])
        for word in ("fetch(", "XMLHttpRequest", "sendBeacon", "import("):
            self.assertNotIn(word, page)
        for node in tree.iter():
            with self.subTest(tag=node.tag):
                self.assertNotIn("style", node.attrs)
                self.assertEqual([name for name in node.attrs if name.startswith("on")], [])
        self.assertEqual(tree.find_all("form"), [])

        def sha(text):
            return "'sha256-" + base64.b64encode(hashlib.sha256(text.encode("utf-8")).digest()).decode("ascii") + "'"

        head_js, page_js = [n.text() for n in tree.find_all("script")]
        self.assertEqual(head_js, render.HEAD_SCRIPT)
        style = tree.find_all("style")[0].text()
        policy = next(node.attrs["content"] for node in tree.find_all("meta")
                      if node.attrs.get("http-equiv") == "Content-Security-Policy")
        self.assertEqual(policy, f"default-src 'none'; script-src {sha(head_js)} {sha(page_js)}; style-src {sha(style)}; "
                                 "img-src data:; base-uri 'none'; form-action 'none'")

    def test_page_follows_the_copy_rules(self):
        page = self.page()
        for text in ("\u2013", "\u2014", "N/A"):
            self.assertNotIn(text, page)
        self.assertIsNone(re.search("[\u2600-\u27bf\u2b00-\u2bff\U0001f000-\U0001faff\ufe0f]", page))
        self.assertIsNone(re.search(r"\bNone\b", page))

    def test_page_holds_no_local_path_or_host_name(self):
        page = self.page()
        for text in (str(self.fixture.root), str(Path.home()), socket.gethostname()):
            self.assertNotIn(text, page)

    def test_license_is_quoted_from_the_readme(self):
        from launcher.tests.site_fixture import LICENSE_LINES
        page = self.page()
        license_node = parse(page).by_id("license")
        self.assertEqual([node.tag for node in license_node.children if not isinstance(node, str)],
                         ["p"] * len(LICENSE_LINES))
        for line in LICENSE_LINES:
            self.assertIn("<p>" + render.inline(line, "README.md", self.links) + "</p>", page)
        self.assertEqual([node.attrs["href"] for node in license_node.find_all("a")],
                         [BLOB + "/LICENSE", BLOB + "/LICENSE-CODE"])

    def test_cells_carry_the_view_attributes(self):
        matrix = self.fixture.load()
        tree = parse(render.page(matrix, BUILD, self.fixture.root))
        labels = {runtime.id: runtime.label for runtime in matrix.runtimes}
        for model in matrix.models:
            tr = tree.by_id("m-" + model.id)
            positions = dict(model.positions)
            self.assertEqual((tr.attrs["data-model"], tr.attrs["data-o-name"], tr.attrs["data-o-gap"]),
                             (model.id, str(positions["name"]), str(positions["gap"])))
            self.assertEqual(tr.attrs.get("data-rank"), None if model.rank is None else str(model.rank))
            tds = {td.attrs["data-engine"]: td for td in tr.find_all("td")}
            self.assertEqual(list(tds), list(ADAPTERS))
            for cell in model.cells:
                with self.subTest(model=model.id, runtime=cell.runtime):
                    attrs = tds[cell.runtime].attrs
                    self.assertEqual((attrs["data-status"], attrs["data-search"]),
                                     (cell.status, data.search_text(model, cell, labels[cell.runtime])))
                    tile = tds[cell.runtime].find_all("a", "tile")[0]
                    self.assertEqual(tile.attrs["href"], f"#d-{cell.runtime}-{model.id}")
                    out = tds[cell.runtime].find_all("span", "out")[0]
                    self.assertEqual((out.text(), "hidden" in out.attrs), ("Not in this view", True))
                    if cell.status != "runs":
                        self.assertEqual(sorted(attrs), ["data-engine", "data-search", "data-status", "role"])
                        continue
                    facts = cell.recipe
                    expected = {"data-ctx": str(facts.context // 1024), "data-cap": str(facts.memory_cap),
                                "data-think": facts.thinking, "data-tools": "no" if facts.chat_only else "yes",
                                "data-use": " ".join(cell.uses)}
                    expected.update({f"data-c-{result.client}": result.cls for result in cell.clients})
                    expected.update({f"data-o-{key}": str(place) for key, place in cell.positions})
                    if facts.decode is not None:
                        expected["data-speed"] = f"{facts.decode:g}"
                    self.assertEqual({name: attrs[name] for name in expected if name in attrs}, expected)
                    self.assertEqual(sorted(attrs), sorted([*expected, "data-engine", "data-status", "data-search",
                                                            "role"]))
        rows = [node.attrs["id"][2:] for node in tree.by_id("matrix").find_all("tbody")[0].find_all("tr")]
        self.assertEqual(rows, list(data.view(matrix, "", data.Filters(), None).rows))
        tile = self.tile(tree, "qwen3.6-35b-a3b", "sglang")
        tag = tile.find_all("span", "tag")[0]
        self.assertEqual(" ".join(tag.text().split()), "* Fastest by 24%")
        self.assertEqual([(node.attrs, node.text()) for node in tag.find_all("span")], [({"aria-hidden": "true"}, "*")])
        words = " ".join(node.text() for node in tile.find_all("span", "vh"))
        self.assertTrue(words.startswith("OpenCode passed,"), words)
        letters = {client.id: client.letter for client in matrix.clients}
        self.assertEqual(tile.find_all("span", "lamps")[0].text(),
                         " ".join(f"{letters[name]}{clients.LABELS[name]} passed," for name in clients.NAMES))
        lamps = tile.find_all("span", "lamps")[0].find_all("span", "lamp")
        self.assertEqual([node.attrs["class"] for node in lamps], ["lamp passed"] * len(clients.NAMES))
        self.assertEqual([node.find_all("span")[0].attrs for node in lamps], [{"aria-hidden": "true"}] * 5)
        self.assertEqual(tile.find_all("span", "setup"), [])
        self.fixture.set_value("vllm", "gpt-oss-20b", "decode_tok_s", None)
        cell = prefix("llamacpp", "qwen3.6-35b-a3b")
        self.fixture.run("2026-11-19", "llmctl-next-e2e", [row(cell, "up"), row(cell, "status", "no answer"),
                                                           row(cell, "opencode")])
        tree = parse(self.page())
        tile = self.tile(tree, "gpt-oss-20b", "vllm")
        self.assertEqual(" ".join(tile.find_all("span", "speed")[0].text().split()), "speed not recorded")
        self.assertNotIn("data-speed", tile_cell(tree, "gpt-oss-20b", "vllm").attrs)
        setups = [(model_id, td.attrs["data-engine"]) for model_id in sorted({r for _, r in RECIPES})
                  for td in tree.by_id("m-" + model_id).find_all("td") if td.find_all("span", "setup")]
        self.assertEqual(setups, [("qwen3.6-35b-a3b", "llamacpp")])
        self.assertEqual(self.tile(tree, "qwen3.6-35b-a3b", "llamacpp").find_all("span", "setup")[0].text(),
                         "Setup check failed")

    def tile(self, tree, model_id, runtime):
        return tile_cell(tree, model_id, runtime).find_all("a", "tile")[0]

    def test_repository_text_is_escaped(self):
        self.fixture.set_value("llamacpp", "mellum2.1-12b", "name", '"Mellum \\"Q\\" & <B>"')
        page = self.page()
        self.assertEqual(well_formed_problems(page), [])
        self.assertIn("mellum &quot;q&quot; &amp; &lt;b&gt;", page)
        self.assertNotIn("& <b>", page)
        self.assertNotIn("& <B>", page)
        self.assertIn('mellum "q" & <b>', tile_cell(parse(page), "mellum2.1-12b", "llamacpp").attrs["data-search"])

    def test_header_and_footer_facts(self):
        matrix = self.fixture.load()
        facts = [cell.recipe for model in matrix.models for cell in model.cells if cell.status == "runs"]
        hardware = "; ".join(sorted({recipe.hardware for recipe in facts}))
        oldest, newest = min(recipe.date for recipe in facts), max(recipe.date for recipe in facts)
        tree = parse(render.page(matrix, BUILD, self.fixture.root))
        squash = lambda node: " ".join(node.text().split())
        self.assertEqual(squash(tree.find_all("p", "facts")[0]),
                         f"Measured on {hardware} from {oldest} to {newest}; clients last tested 2026-11-18. "
                         "Built from commit aaaaaaa, committed 1970-01-01 00:00 UTC.")
        self.assertEqual([node.attrs["href"] for node in tree.find_all("p", "facts")[0].find_all("a")], [TREE])
        images = ", ".join(f"{runtime.label} {runtime.image}" for runtime in matrix.runtimes)
        versions = tree.by_id("versions")
        self.assertIn("open", versions.attrs)
        clients_line = ("Clients, as the e2e of 2026-11-18 recorded them: OpenCode opencode v2.3.0, Hermes Hermes Agent "
                        "v0.23.1, Pi 1.5.2, dsh 0.2.2-rc.3, Kimi 2.5.3.")
        self.assertEqual(squash(versions.find_all("p")[0]), f"Images llmctl starts: {images}. {clients_line}")
        n = len(default_rows())
        line = (f"2026-11-18, llmctl-fixture-e2e: {n} of {n} rows passed over 10 recipes; clients OpenCode, Hermes, Pi, "
                "dsh, Kimi; versions OpenCode opencode v2.3.0, Hermes Hermes Agent v0.23.1, Pi 1.5.2, dsh 0.2.2-rc.3, "
                "Kimi 2.5.3.")
        self.assertEqual([squash(li) for li in tree.by_id("runs").find_all("li")], [line])
        self.assertIsNone(tree.by_id("ranked-without-recipe"))
        description = next(node.attrs["content"] for node in tree.find_all("meta") if node.attrs.get("name") == "description")
        self.assertEqual(description, "Which open-weight LLMs run on one laptop with llama.cpp, vLLM and SGLang, how "
                                      f"fast, and with which agent clients; measured on {hardware}.")
        catalog = data.shipped_catalog(self.fixture.root)
        rows = [TOP10_ROW.format(rank=rank, id=recipe_id, repo=catalog.runtimes["llamacpp"].recipes[recipe_id].model.repo)
                for rank, recipe_id in enumerate(TOP10_IDS, start=1)]
        self.fixture.top10([*rows, "| 4 | [Unshipped](https://huggingface.co/org/unshipped) | llama.cpp |"])
        self.fixture.run("2026-11-17", "llmctl-old-e2e", default_rows() + [row("03-removed-model-llamacpp", "up"),
                                                                            row("04-qwen3.6-35b-a3b-oldengine", "up")],
                         readme=None, records=None)
        (self.fixture.root / "research" / DEFAULT_RUN[0] / DEFAULT_RUN[1] / "records" / "005-kimi-version").write_text(
            version_record("", 127), encoding="utf-8")
        tree = parse(self.page())
        ranked = tree.by_id("ranked-without-recipe")
        self.assertEqual(squash(ranked), "Ranked in the Top 10, without a shipped recipe yet: rank 4, Unshipped.")
        self.assertEqual([node.attrs["href"] for node in ranked.find_all("a")], ["https://huggingface.co/org/unshipped"])
        self.assertTrue(squash(tree.by_id("versions").find_all("p")[0]).endswith("dsh 0.2.2-rc.3, Kimi not recorded."))
        self.assertEqual([squash(li) for li in tree.by_id("runs").find_all("li")],
                         [f"2026-11-17, llmctl-old-e2e: {n + 2} of {n + 2} rows passed over 12 recipes, 2 rows of recipes "
                          "or engines that no longer ship; clients OpenCode, Hermes, Pi, dsh, Kimi; versions not recorded.",
                          line.replace("Kimi 2.5.3.", "Kimi not recorded.")])
        first = next(cell for model in matrix.models for cell in model.cells if cell.status == "runs")
        rig = "Test rig, RTX 5080 Laptop GPU 16 GB"
        self.fixture.set_value(first.runtime, first.model, "hardware", f'"{rig}"')
        hardware = "; ".join(sorted({hardware, rig}))
        self.assertNotEqual(hardware.split("; ")[0], rig)
        qwen = prefix("llamacpp", "qwen3.6-35b-a3b")
        gpt = prefix("llamacpp", "gpt-oss-20b")
        self.fixture.run("2026-11-16", "llmctl-kimi-e2e", [row(gpt, "up"), row(gpt, "kimi")],
                         records={"001-kimi-version": version_record("2.4.0")})
        self.fixture.run("2026-11-18", "llmctl-z-e2e", [row(qwen, "up"), row(qwen, "hermes-z"),
                                                        row(qwen, "hermes-chat", "no answer"),
                                                        row("03-removed-model-llamacpp", "up")],
                         records={"001-hermes-version": version_record("Hermes Agent v0.24.0")})
        tree = parse(self.page())
        self.assertEqual(squash(tree.find_all("p", "facts")[0]),
                         f"Measured on {hardware} from {oldest} to {newest}; clients last tested 2026-11-18. "
                         "Built from commit aaaaaaa, committed 1970-01-01 00:00 UTC.")
        self.assertEqual(squash(tree.by_id("versions").find_all("p")[0]),
                         f"Images llmctl starts: {images}. Clients, as the e2e of 2026-11-18 recorded them: OpenCode "
                         "opencode v2.3.0, Hermes Hermes Agent v0.23.1 (llmctl-fixture-e2e), Hermes Hermes Agent v0.24.0 "
                         "(llmctl-z-e2e), Pi 1.5.2, dsh 0.2.2-rc.3, Kimi 2.4.0 (2026-11-16).")
        runs = [squash(li) for li in tree.by_id("runs").find_all("li")]
        self.assertEqual((runs[0], runs[3]),
                         ("2026-11-16, llmctl-kimi-e2e: 2 of 2 rows passed over 1 recipe; clients Kimi; versions Kimi "
                          "2.4.0.",
                          "2026-11-18, llmctl-z-e2e: 3 of 4 rows passed over 2 recipes, 1 row of a recipe or engine that "
                          "no longer ships; clients Hermes; versions Hermes Hermes Agent v0.24.0."))
        speeds = ("Speeds and contexts: " + ", ".join(f"(measured on {hardware}, {runtime.heading})"
                                                      for runtime in matrix.runtimes[:-1])
                  + f" and (measured on {hardware}, {matrix.runtimes[-1].heading}); Does not fit: (estimate).")
        self.assertIn(speeds, [squash(node) for node in tree.by_id("legend").find_all("p")])
        footer = [squash(node) for node in tree.find_all("footer")[0].find_all("p")]
        self.assertIn(speeds, footer)
        self.assertEqual(footer[-1], "This page sets no cookies, has no analytics and loads nothing from other sites.")
        for record in sorted(self.fixture.root.glob("research/*/*/records/*-version")):
            record.unlink()
        tree = parse(self.page())
        self.assertEqual(squash(tree.by_id("versions").find_all("p")[0]),
                         f"Images llmctl starts: {images}. Clients: versions not recorded.")
        self.fixture.top10([*rows, "| 5 | *Unshipped* | llama.cpp |"])
        with self.assertRaises(data.SiteError) as raised:
            self.page()
        self.assertEqual(str(raised.exception), "docs/model-selection.md: row 5, Model: column 1: emphasis")

    def test_assets_are_checked_before_inlining(self):
        self.assertEqual((render.switch_width(3), render.switch_width(4)), ("58rem", "72.25rem"))
        for css, js in (("a {}", ""), ("@SWITCH@ @SWITCH@", ""), ("@SWITCH@ </style>", ""), ("@SWITCH@", "a</script>")):
            with self.subTest(css=css, js=js), self.assertRaises(data.SiteError):
                render.inline_assets(css, js, 3)
        self.assertEqual(render.inline_assets("@media (min-width: @SWITCH@) {}", '"use strict";\n', 3),
                         ("@media (min-width: 58rem) {}", '"use strict";\n'))

    def test_build_is_byte_identical_across_hash_seeds(self):
        code = ("import sys\nfrom pathlib import Path\nfrom launcher.site import cli\n"
                "sys.exit(cli.main(['build', '--out', sys.argv[1], '--commit', 'a' * 40, '--commit-time', '0'], "
                "root=Path(sys.argv[2])))")
        pages = []
        for seed in ("1", "2"):
            out = Path(self._tmp.name) / f"out-{seed}"
            done = subprocess.run([sys.executable, "-B", "-c", code, str(out), str(self.fixture.root)], cwd=REPO,
                                  env={"PYTHONHASHSEED": seed, "PYTHONDONTWRITEBYTECODE": "1"}, capture_output=True,
                                  text=True, timeout=120)
            self.assertEqual(done.returncode, 0, done.stderr)
            pages.append((out / "index.html").read_bytes())
        self.assertEqual(pages[0], pages[1])

    def test_only_the_build_facts_change_between_builds(self):
        pages = []
        for name, commit, time in (("a", "a" * 40, "0"), ("b", "b" * 40, "86400")):
            out = Path(self._tmp.name) / f"out-{name}"
            self.assertEqual(self.build(out, commit, time)[0], 0)
            page = (out / "index.html").read_text(encoding="utf-8")
            self.assertIn(commit, page)
            page = page.replace(commit, "COMMIT").replace(commit[:7], "SHORT")
            pages.append(re.sub(r"1970-01-0[12] 00:00", "TIME", page))
        self.assertEqual(pages[0], pages[1])

    def test_cli_checks_the_build_facts(self):
        from launcher.site import cli
        out = str(Path(self._tmp.name) / "out")
        good = {"--commit": "a" * 40, "--commit-time": "0"}
        for flag, value in (("--commit", "abc123"), ("--commit", "A" * 40), ("--commit-time", "-5"),
                            ("--commit-time", "abc"), ("--commit-time", "253402300800"),
                            ("--commit-time", "9" * 20), ("--repository", "https://gitlab.com/o/r"),
                            ("--repository", "http://github.com/o/r")):
            args = {**good, flag: value}
            argv = ["build", "--out", out, *(part for key, item in args.items() for part in (f"{key}={item}",))]
            stderr = io.StringIO()
            with self.subTest(flag=flag, value=value), contextlib.redirect_stderr(stderr):
                self.assertEqual(cli.main(argv, root=self.fixture.root), 1)
                self.assertTrue(stderr.getvalue().startswith(f"site: {flag} {value}: "), stderr.getvalue())
        self.assertFalse(Path(out, "index.html").exists())
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            cli.main(["build", "--out", out, "--commit-time", "0"], root=self.fixture.root)
        self.assertEqual(raised.exception.code, 2)

    def test_cli_refuses_an_out_folder_with_other_files(self):
        out = Path(self._tmp.name) / "out"
        out.mkdir()
        (out / "notes.txt").write_text("keep\n", encoding="utf-8")
        code, stdout, stderr = self.build(out)
        self.assertEqual((code, stdout), (1, ""))
        self.assertTrue(stderr.startswith("site: "), stderr)
        self.assertFalse((out / "index.html").exists())
        self.assertEqual((out / "notes.txt").read_text(encoding="utf-8"), "keep\n")
        (out / "notes.txt").unlink()
        (out / "index.html").write_text("old\n", encoding="utf-8")
        code, stdout, stderr = self.build(out)
        self.assertEqual((code, stderr), (0, ""))
        size = (out / "index.html").stat().st_size
        self.assertGreater(size, 4)
        self.assertEqual(stdout, f"wrote {out / 'index.html'}: {size} bytes, 6 models, {len(ADAPTERS)} engines, "
                                 f"{len(clients.NAMES)} clients, 1 e2e runs\n")
        from launcher.site import cli
        self.assertEqual((out / "index.html").read_text(encoding="utf-8"),
                         render.page(self.fixture.load(), data.Build(cli.REPOSITORY, "a" * 40, 0), self.fixture.root))
        fresh = Path(self._tmp.name) / "fresh" / "out"
        self.assertEqual(self.build(fresh)[0], 0)
        self.assertEqual((fresh / "index.html").stat().st_mode & 0o7777, 0o644)
        (out / "index.html").unlink()
        (out / "index.html").mkdir()
        target = Path(self._tmp.name) / "elsewhere.html"
        target.write_text("keep\n", encoding="utf-8")
        for name, refused in (("a folder", out), ("a file", target)):
            code, stdout, stderr = self.build(refused)
            with self.subTest(out=name):
                self.assertEqual((code, stdout), (1, ""))
                self.assertEqual(stderr, f"site: --out {refused}: not a folder\n" if refused == target else
                                 f"site: --out {out}: {out / 'index.html'} is not a regular file\n")
        (out / "index.html").rmdir()
        (out / "index.html").symlink_to(target)
        code, stdout, stderr = self.build(out)
        self.assertEqual((code, stderr), (1, f"site: --out {out}: {out / 'index.html'} is not a regular file\n"))
        self.assertEqual(target.read_text(encoding="utf-8"), "keep\n")

    def test_site_reads_no_environment_and_runs_nothing(self):
        files = sorted((REPO / "launcher/site").glob("*.py")) + [REPO / "launcher/e2e/summary.py"]
        self.assertGreater(len(files), 4)
        banned = {"subprocess", "socket", "urllib.request", "http.client", "launcher.e2e.clients"}
        for path in files:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                with self.subTest(file=path.name, line=getattr(node, "lineno", 0)):
                    if isinstance(node, ast.Attribute):
                        self.assertNotIn(node.attr, ("environ", "getenv", "expanduser"))
                        if node.attr == "home":
                            self.assertFalse(isinstance(node.value, ast.Name) and node.value.id == "Path")
                    if isinstance(node, ast.Name):
                        self.assertNotIn(node.id, ("environ", "getenv", "expanduser"))
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            self.assertFalse(alias.name in banned or alias.name.split(".")[0] in banned, alias.name)
                    if isinstance(node, ast.ImportFrom):
                        module = node.module or ""
                        self.assertFalse(module in banned or module.split(".")[0] in banned, module)
                        names = {alias.name for alias in node.names}
                        self.assertFalse(module == "launcher.e2e" and "clients" in names, module)
                        self.assertFalse(module == "urllib" and "request" in names, module)
                        self.assertFalse(module == "http" and "client" in names, module)
                        self.assertFalse(module == "os" and names & {"environ", "getenv"}, module)


def tile_cell(tree, model_id, runtime):
    return next(td for td in tree.by_id("m-" + model_id).find_all("td") if td.attrs["data-engine"] == runtime)


if __name__ == "__main__":
    unittest.main()
