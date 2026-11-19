"""The runtime matrix page's renderer: the Markdown subset of the coverage notes, repository links at the build's
commit (design D12), and the README recipe table (D-README)."""

import ast
import base64
import contextlib
import dataclasses
import hashlib
import io
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

from launcher import clients
from launcher.runtimes import ADAPTERS
from launcher.site import data, render
from launcher.tests.fakes import REPO
from launcher.tests.site_fixture import (RECIPES, TOP10_IDS, TOP10_ROW, DEFAULT_RUN, DOES_NOT_FIT_CELL, SiteFixture, cell_of,
                                         config_section, default_rows, parse, prefix, row, version_record,
                                         well_formed_problems)

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
        words_engine = {runtime.id: runtime.label for runtime in matrix.runtimes}
        squash = lambda node: " ".join(node.text().split())
        table = tree.by_id("matrix")
        # Without JavaScript the controls stay hidden and the no-JS line shows; the script, the skip link and WCAG
        # need the rest: the document language, the status line, the descriptions, the clipped caption and the
        # matrix as a focus target.
        page_text = render.page(matrix, BUILD, self.fixture.root)
        self.assertIn('<html lang="en" class="no-js">', page_text)
        self.assertEqual((tree.by_id("notice").tag, "hidden" in tree.by_id("notice").attrs), ("div", True))
        self.assertEqual(render.NO_JS,
                         "Views and filters need JavaScript; each view as a list and every pair's details follow.")
        self.assertEqual([node.text() for node in tree.find_all("p", "only-no-js")], [render.NO_JS])
        element = lambda node: (node.tag, node.attrs.get("id"), node.attrs.get("class"))
        body = tree.find_all("body")[0]
        self.assertEqual([element(node) for node in body.children if not isinstance(node, str)],
                         [("a", None, "skip"), ("header", None, None), ("main", None, None), ("footer", None, None),
                          ("dialog", "sheet", None), ("dialog", "drawer", None), ("template", "phrases", None),
                          ("template", "cards", None), ("script", None, None)])
        main = tree.find_all("main")[0]
        self.assertEqual([element(node) for node in main.children if not isinstance(node, str)],
                         [("p", "print-view", "print-only"), ("section", "picks", None), ("p", "key", "key"),
                          ("h2", None, "vh"), ("div", None, "tbl"), ("p", None, "only-no-js"), ("section", "use-cases", None),
                          ("section", "details", None), ("p", "live", "vh")])
        self.assertEqual([squash(node) for node in main.children if not isinstance(node, str) and node.tag == "h2"],
                         ["Models and engines"])
        self.assertTrue("hidden" in tree.by_id("controls").attrs)
        self.assertEqual([squash(tree.by_id(i)) for i in ("open-filters", "show")], ["Filters", "Show 6 models"])
        self.assertEqual(squash(tree.by_id("open-filters")), data.filters_label(data.Filters()))
        with mock.patch.object(data, "filters_label", lambda filters: "Filters probe"):
            self.assertEqual(squash(parse(render.page(matrix, BUILD, self.fixture.root)).by_id("open-filters")),
                             "Filters probe")

        # No script runs to unhide what this page must show: none of it, nor any ancestor, carries hidden.
        def with_ancestors(node, chain=()):
            for child in node.children:
                if not isinstance(child, str):
                    yield child, chain
                    yield from with_ancestors(child, (*chain, child))

        shown = {"matrix", "pick-list", "key", "use-cases", "details"}
        shown_classes = {"pick", "facts", "only-no-js", "details"}
        checked = [(node, chain) for node, chain in with_ancestors(tree)
                   if node.attrs.get("id") in shown or _classes(node) & shown_classes]
        self.assertGreater(len(checked), len(shown) + len(matrix.models))
        for node, chain in checked:
            with self.subTest(shown=node.attrs.get("id") or node.attrs.get("class")):
                self.assertEqual([(n.tag, n.attrs.get("id")) for n in (*chain, node) if "hidden" in n.attrs], [])
        self.assertEqual(table.attrs.get("data-pairs"), "18")
        self.assertEqual((table.tag, table.attrs.get("tabindex")), ("table", "-1"))
        self.assertEqual(tree.find_all("a", "skip")[0].attrs["href"], "#matrix")
        self.assertEqual((tree.by_id("answer").tag, tree.by_id("answer").attrs.get("role")), ("span", "status"))
        self.assertEqual((tree.by_id("caption").tag, tree.by_id("caption").attrs.get("class")), ("caption", "vh"))
        for model in matrix.models:
            for cell in model.cells:
                with self.subTest(model=model.id, runtime=cell.runtime):
                    td = next(node for node in tree.by_id("m-" + model.id).find_all("td")
                              if node.attrs["data-engine"] == cell.runtime)
                    if cell.status == "runs":
                        self.assertTrue(visible(td).startswith(data.speed_text(cell.recipe.decode)), visible(td))
                    else:
                        self.assertIn(words[cell.status], visible(td))
                    section = tree.by_id(f"d-{cell.runtime}-{model.id}")
                    self.assertIsNotNone(section)
                    # page.js names the dialog by this h3 (aria-labelledby).
                    self.assertEqual([(node.attrs.get("id"), squash(node)) for node in section.find_all("h3")],
                                     [(f"t-{cell.runtime}-{model.id}", f"{model.id} on {words_engine[cell.runtime]}")])
                    self.assertIn("details", section.attrs["class"].split())
                    self.assertEqual((section.attrs["data-model"], section.attrs["data-engine"]),
                                     (model.id, cell.runtime))
        v = data.view(matrix, "", data.Filters(), None)
        self.assertEqual(squash(tree.by_id("answer")), v.count + " " + data.picks_line(v.picks, matrix))
        answer_picks = tree.by_id("answer-picks")
        self.assertEqual((answer_picks.attrs.get("class"), answer_picks.text()),
                         ("vh", " " + data.picks_line(v.picks, matrix)))

        def check_lists(matrix, tree):
            """The static view lists (2026-11-19 plan ruling 11) and the phrase template of one page."""
            engine_labels = {runtime.id: runtime.label for runtime in matrix.runtimes}
            models = {model.id: model for model in matrix.models}
            rule_text = lambda use: data.rule_line(use, matrix) + (" What the rank means" if use.id == "quality" else "")
            lists = tree.by_id("use-cases")
            self.assertEqual(squash(lists.find_all("h2")[0]), "Each view as a list")
            uses = {node.attrs["data-use"]: node for node in lists.find_all("section", "use")}
            self.assertEqual(list(uses), [use.id for use in data.USE_CASES[1:]])
            for use in data.USE_CASES[1:]:
                with self.subTest(use=use.id):
                    section, found = uses[use.id], data.view(matrix, use.id, data.Filters(), None)
                    self.assertEqual(squash(section.find_all("h3")[0]), data.view_name(use, matrix))
                    paragraphs = [(node.attrs.get("class"), squash(node)) for node in section.find_all("p")]
                    # The Top 10 rule ends with the link to the footer's rank line (2026-11-19 plan ruling 11).
                    self.assertEqual([a.attrs["href"] for a in section.find_all("p", "rule")[0].find_all("a")],
                                     ["#top10"] if use.id == "quality" else [])
                    if not found.rows:
                        self.assertEqual(paragraphs, [("rule", rule_text(use)), (None, "No model matches this view.")])
                        self.assertEqual((section.find_all("ol"), section.find_all("button")), ([], []))
                        continue
                    self.assertEqual(paragraphs, [("rule", rule_text(use)),
                                                  ("count", found.count + " " + data.picks_line(found.picks, matrix))])
                    self.assertEqual([a.attrs["href"] for a in section.find_all("p", "count")[0].find_all("a")],
                                     [f"#d-{p.runtime}-{p.model}" for p in found.picks])
                    expected = []
                    for model_id in found.rows:
                        cells = {cell.runtime: cell for cell in models[model_id].cells}
                        runtimes = [runtime for cell_model, runtime in found.cells if cell_model == model_id]
                        items, links = [], []
                        for runtime in runtimes:
                            facts = cells[runtime].recipe
                            speed = ("speed not recorded" if facts.decode is None
                                     else f"{data.speed_text(facts.decode)} tok/s")
                            default = (", default in the OpenCode config"
                                       if use.id == "opencode" and (model_id, runtime) == matrix.default else "")
                            words = f"{engine_labels[runtime]} {speed}, {facts.context // 1024}k"
                            items.append(words + default)
                            links.append((words, f"#d-{runtime}-{model_id}"))
                        expected.append((f"{model_id}: " + "; ".join(items), links))
                    self.assertEqual([(squash(li), [(squash(a), a.attrs["href"]) for a in li.find_all("a")])
                                      for li in section.find_all("li")], expected)
                    self.assertEqual(len(section.find_all("ol")), 1)
            spans = tree.by_id("phrases").find_all("span")
            roles = [*data.ROLES, *(f"engine-{runtime}" for runtime in ADAPTERS)]
            expected = [("rule", use.id, rule_text(use), None) for use in data.USE_CASES]
            expected.append(("empty", "", "No model matches this view. Clear filters", None))
            expected += [("role", key, data.role_words(key, matrix),
                          data.role_words(key, matrix)[:1].lower() + data.role_words(key, matrix)[1:]) for key in roles]
            self.assertEqual([(span.attrs["data-kind"], span.attrs["data-for"], squash(span), span.attrs.get("data-mid"))
                              for span in spans], expected)
            self.assertEqual([(span.attrs["data-for"], [a.attrs["href"] for a in span.find_all("a")]) for span in spans
                              if span.find_all("a")], [("quality", ["#top10"])])
            empty = next(span for span in spans if span.attrs["data-kind"] == "empty")
            self.assertEqual([(node.attrs, node.text()) for node in empty.find_all("button")],
                             [({"type": "button", "class": "clear"}, "Clear filters")])

        check_lists(matrix, tree)
        self.assertEqual(squash(tree.by_id("caption")), "Decode speed, context and client results of each model on each engine. "
                         "View: All models; sorted by: Top 3 rank, then speed.")
        self.assertEqual(squash(tree.by_id("caption")), data.caption_line(data.USE_CASES[0], "rank", matrix))
        self.assertEqual(squash(tree.by_id("print-view")), "View: All models; sorted by: Top 3 rank, then speed.")
        self.assertEqual(squash(tree.by_id("print-view")),
                         data.print_line(data.USE_CASES[0], "rank", data.Filters(), matrix))
        live = tree.by_id("live")
        self.assertEqual((live.tag, live.attrs, live.text()), ("p", {"id": "live", "class": "vh", "aria-live": "polite"}, ""))
        self.assertEqual(table.attrs["role"], "table")
        self.assertEqual([node.attrs.get("role") for node in table.find_all("thead") + table.find_all("tbody")],
                         ["rowgroup", "rowgroup"])
        self.assertEqual({node.attrs.get("role") for node in table.find_all("tr")}, {"row"})
        self.assertEqual({(node.attrs["scope"], node.attrs.get("role")) for node in table.find_all("th")},
                         {("col", "columnheader"), ("row", "rowheader"), ("rowgroup", "rowheader")})
        self.assertEqual([(node.attrs.get("id"), node.attrs["aria-labelledby"]) for node in tree.find_all("section")
                          if "aria-labelledby" in node.attrs], [("picks", "picks-title")])
        # The lead's autofocus ruling: the markup carries no autofocus (Chrome logs a line for it on every load with
        # a hash), and the script sets it before its first showModal(); Chrome 157 then focuses button#close.
        drawer = tree.by_id("drawer")
        self.assertEqual((drawer.tag, drawer.attrs), ("dialog", {"id": "drawer"}))
        self.assertEqual([(node.tag, node.attrs) for node in drawer.children if not isinstance(node, str)],
                         [("div", {"id": "drawer-bar"}), ("div", {"id": "drawer-body"}),
                          ("p", {"id": "drawer-live", "class": "vh", "aria-live": "polite"})])
        bar = tree.by_id("drawer-bar")
        self.assertEqual([(node.tag, node.attrs, squash(node)) for node in bar.children if not isinstance(node, str)],
                         [("button", {"type": "button", "id": "close"}, "Close")])
        self.assertEqual([node for node in bar.children if isinstance(node, str) and node.strip()], [])
        script = (REPO / "launcher/site/page.js").read_text(encoding="utf-8")
        self.assertRegex(script, r"drawer\.autofocus = true;[^\0]*?drawer\.showModal\(\)")
        # The modal dialog makes #live inert, so a copy inside it is announced in the dialog's own live region.
        self.assertRegex(script, r"\(drawer\.open \? drawerLive : sheet\.open \? sheetLive : live\)\.textContent = text;")
        self.assertEqual([squash(node) for node in table.find_all("th", "mh")], ["Top 3 rank and model"])
        heads = [node for node in table.find_all("thead")[0].find_all("th") if node.attrs.get("data-engine")]
        self.assertEqual([tuple(squash(head.find_all("span", cls)[0]) for cls in ("en", "ver", "n")) for head in heads],
                         [("llama.cpp", "b12407", ", 6 of 6 run"), ("vLLM", "v0.33.0", ", 2 of 6 run"),
                          ("SGLang", "v0.5.23", ", 2 of 6 run")])
        self.fixture.run("2026-11-19", "llmctl-pi-e2e", [line for runtime, recipe_id in RECIPES
                                                         for line in (row(prefix(runtime, recipe_id), "up"),
                                                                      row(prefix(runtime, recipe_id), "pi", "no answer"))])
        matrix = self.fixture.load()
        tree = parse(render.page(matrix, BUILD, self.fixture.root))
        self.assertEqual(data.view(matrix, "every-client", data.Filters(), None).rows, ())
        check_lists(matrix, tree)
        # A head counts its engine's Runs cells, not its recipes: an estimate recipe ships in vLLM's Does not fit cell.
        target = self.fixture.root / "recipes/vllm/k2-horizon-7b.toml"
        shutil.copyfile(REPO / "recipes/vllm/k2-horizon-7b.toml", target)
        doc = tomllib.loads(target.read_text(encoding="utf-8"))["doc"]
        target.write_text("".join(line + "\n" for line in target.read_text(encoding="utf-8").splitlines()
                                  if not line.startswith("decode_tok_s = "))
                          .replace('kind = "measured"', f'kind = "estimate"\ndoc = "{doc}"'), encoding="utf-8")
        path, anchor = doc.split("#", 1)
        with (self.fixture.root / path).open("a", encoding="utf-8") as models:
            models.write(f"## {anchor}\n\n")
        # A heading that is the engine label alone has no span.ver (2026-11-19 plan ruling 13).
        self.fixture.coverage_table([self.fixture._coverage_lines()[0].replace("llama.cpp b12407", "llama.cpp"),
                                     *self.fixture._coverage_lines()[1:]])
        matrix = self.fixture.load()
        self.assertEqual(matrix.runtimes[1].recipes, 3)
        heads = [node for node in parse(render.page(matrix, BUILD, self.fixture.root)).by_id("matrix")
                 .find_all("thead")[0].find_all("th") if node.attrs.get("data-engine")]
        self.assertEqual([[(node.attrs["class"], squash(node)) for node in head.children if not isinstance(node, str)]
                          for head in heads],
                         [[("en", "llama.cpp"), ("n", ", 6 of 6 run")],
                          [("en", "vLLM"), ("ver", "v0.33.0"), ("n", ", 2 of 6 run")],
                          [("en", "SGLang"), ("ver", "v0.5.23"), ("n", ", 2 of 6 run")]])
        self.assertEqual(spoken(heads[0]), "llama.cpp, 6 of 6 run")

    def card_texts(self, card):
        """The class and squashed text of each p of a card, which the built and the template copy share."""
        return [(node.attrs.get("class"), squash(node)) for node in card.find_all("p")]

    def test_pick_cards_of_the_default_view(self):
        matrix = self.fixture.load()
        tree = parse(render.page(matrix, BUILD, self.fixture.root))
        facts = cell_of(matrix, "qwen3.6-35b-a3b", "sglang").recipe
        section = tree.by_id("picks")
        self.assertEqual((section.tag, section.attrs.get("aria-labelledby")), ("section", "picks-title"))
        title = tree.by_id("picks-title")
        self.assertEqual((title.tag, title.attrs.get("class"), squash(title)), ("h2", "vh", "Picks for this view"))
        listing = tree.by_id("pick-list")
        self.assertEqual((listing.tag, listing.attrs.get("class"), listing.attrs.get("data-long")), ("ol", "picks", "128"))
        cards = [node for node in listing.children if not isinstance(node, str)]
        self.assertEqual([(node.tag, node.attrs.get("class"), node.attrs.get("data-for")) for node in cards],
                         [("li", "pick", "d-sglang-qwen3.6-35b-a3b"), ("li", "pick", "d-llamacpp-mellum2.1-12b"),
                          ("li", "pick", "d-llamacpp-gpt-oss-20b")])
        first, mellum, gpt = cards
        h3 = first.find_all("h3")[0]
        self.assertEqual((squash(h3), squash(h3.find_all("span", "role")[0])), ("Highest ranked: qwen3.6-35b-a3b on SGLang", "Highest ranked"))
        self.assertEqual([(squash(n), n.attrs.get("class")) for n in h3.find_all("span", "vh")], [(":", "vh")])
        self.assertEqual([(a.attrs["href"], squash(a), squash(a.find_all("span", "pid")[0])) for a in h3.find_all("a")],
                         [("#d-sglang-qwen3.6-35b-a3b", "qwen3.6-35b-a3b on SGLang", "qwen3.6-35b-a3b")])
        pn = first.find_all("p", "pn")[0]
        self.assertEqual((squash(pn), squash(pn.find_all("span", "ps")[0]), pn.find_all("i")[0].attrs["class"]),
                         ("151.3 tok/s, 128k context", "151.3", "w-80"))
        self.assertEqual(pn.find_all("span", "bar")[0].attrs.get("aria-hidden"), "true")
        pg = first.find_all("p", "pg")[0]
        self.assertEqual(("hidden" in pg.attrs, squash(pg)), (True, "SGLang 24% ahead of llama.cpp"))
        self.assertEqual(squash(first.find_all("p", "pcl")[0]), "All 5 clients passed.")
        pq = first.find_all("p", "pq")[0]
        self.assertEqual(squash(pq), f"Needs GPU {facts.gpu}, RAM {facts.memory_cap} GiB; download {facts.size_bytes / 1e9:.1f} GB.")
        # Each number keeps its unit on one line (2026-11-19 plan ruling 15).
        self.assertEqual([n.text() for n in pq.find_all("span", "nw")],
                         [facts.gpu, f"{facts.memory_cap} GiB", f"{facts.size_bytes / 1e9:.1f} GB"])
        pst = first.find_all("details", "pst")[0]
        self.assertEqual(("open" in pst.attrs, squash(pst.find_all("summary")[0])), (False, "Start commands"))
        self.assertEqual([(squash(li.find_all("p", "step")[0]), li.find_all("code", "cmd")[0].text(),
                           li.find_all("button", "copy")[0].attrs.get("data-copy"), "hidden" in li.find_all("button", "copy")[0].attrs)
                          for li in pst.find_all("ol", "steps")[0].find_all("li")],
                         [("Download once", facts.pull, facts.pull, True), ("Start", facts.up, facts.up, True)])
        dialog_pre = tree.by_id("d-sglang-qwen3.6-35b-a3b").find_all("p", "pre")[0]
        card_pre = pst.find_all("p", "pre")
        self.assertEqual([(squash(n), [(a.attrs["href"], squash(a)) for a in n.find_all("a")]) for n in card_pre],
                         [(squash(dialog_pre), [(a.attrs["href"], squash(a)) for a in dialog_pre.find_all("a")])])
        self.assertEqual(len(dialog_pre.find_all("a")), 2)
        self.assertEqual(squash(mellum.find_all("h3")[0]), "Fastest: mellum2.1-12b on llama.cpp")
        self.assertEqual(mellum.find_all("p", "pg"), [])
        self.assertEqual(squash(gpt.find_all("h3")[0]), "Fastest in the Top 3: gpt-oss-20b on llama.cpp")
        self.assertEqual([("hidden" in n.attrs, squash(n)) for n in gpt.find_all("p", "pg")],
                         [(True, "llama.cpp 99% ahead of vLLM")])
        no_picks = tree.by_id("no-picks")
        self.assertEqual((no_picks.tag, "hidden" in no_picks.attrs, squash(no_picks)), ("p", True, data.NO_PICKS))

    def test_card_templates_cover_every_runs_pair(self):
        matrix = self.fixture.load()
        tree = parse(render.page(matrix, BUILD, self.fixture.root))
        template = tree.by_id("cards")
        self.assertEqual(template.tag, "template")
        cards = [node for node in template.children if not isinstance(node, str)]
        self.assertEqual([(node.tag, node.attrs.get("class")) for node in cards], [("li", "pick")] * len(cards))
        self.assertEqual([node.attrs["data-for"] for node in cards],
                         [f"d-{c.runtime}-{m.id}" for m in matrix.models for c in m.cells if c.status == "runs"])
        self.assertEqual({node.text() for card in cards for node in card.find_all("span", "role")}, {""})
        self.assertEqual([len(card.find_all("span", "role")) for card in cards], [1] * len(cards))
        self.assertEqual([node.attrs["id"] for node in template.iter() if "id" in node.attrs], [])
        buttons = template.find_all("button", "copy")
        self.assertEqual((len(buttons), [b for b in buttons if "hidden" in b.attrs]), (2 * len(cards), []))
        labels = {runtime.id: runtime.label for runtime in matrix.runtimes}
        gaps = {f"d-{c.runtime}-{m.id}": data.gap_text(m, labels) for m in matrix.models for c in m.cells
                if c.status == "runs" and data.gap_text(m, labels) is not None}
        self.assertEqual(sorted(gaps), sorted(f"d-{runtime}-{model_id}" for model_id, runtimes in (
            ("deepseek-r1-0528-qwen3-8b", ("llamacpp", "vllm")), ("gpt-oss-20b", ("llamacpp", "vllm", "sglang")),
            ("qwen3.6-35b-a3b", ("llamacpp", "sglang"))) for runtime in runtimes))
        self.assertEqual({card.attrs["data-for"]: [("hidden" in n.attrs, squash(n)) for n in card.find_all("p", "pg")]
                          for card in cards if card.find_all("p", "pg")},
                         {key: [(True, words)] for key, words in gaps.items()})
        by_for = {card.attrs["data-for"]: card for card in cards}
        self.assertEqual(squash(by_for["d-vllm-deepseek-r1-0528-qwen3-8b"].find_all("p", "pcl")[0]),
                         "4 of 5 clients; OpenCode: not supported.")
        # A GPU need in percent stands alone; the RAM and the download keep their units (2026-11-19 plan ruling 15).
        facts = cell_of(matrix, "deepseek-r1-0528-qwen3-8b", "vllm").recipe
        self.assertTrue(facts.gpu.endswith("%"), facts.gpu)
        self.assertEqual([n.text() for n in by_for["d-vllm-deepseek-r1-0528-qwen3-8b"].find_all("p", "pq")[0]
                          .find_all("span", "nw")], [f"{facts.memory_cap} GiB", f"{facts.size_bytes / 1e9:.1f} GB"])
        for card in tree.by_id("pick-list").find_all("li", "pick"):
            with self.subTest(card=card.attrs["data-for"]):
                self.assertEqual(self.card_texts(card), self.card_texts(by_for[card.attrs["data-for"]]))
        # A pair without a speed: no speed span, its words in its place.
        self.fixture.set_value("vllm", "gpt-oss-20b", "decode_tok_s", None)
        tree = parse(render.page(self.fixture.load(), BUILD, self.fixture.root))
        card = next(node for node in tree.by_id("cards").find_all("li", "pick")
                    if node.attrs["data-for"] == "d-vllm-gpt-oss-20b")
        pn = card.find_all("p", "pn")[0]
        self.assertEqual((squash(pn), pn.find_all("i")[0].attrs["class"]), ("Speed not recorded, 64k context", "w-0"))

    def test_key_names_only_the_marks_present(self):
        tree = parse(self.page())
        key = tree.by_id("key")
        parts = [n for n in key.children if not isinstance(n, str)]
        self.assertEqual(([p.attrs["data-k"] for p in parts], key.attrs.get("class")),
                         (["counts", "not-supported", "letters", "fastest", "bars"], "key"))
        self.assertEqual(visible(key), "4/5: clients whose latest session passed; boxed: O not supported (O OpenCode, H Hermes, "
                                       "P Pi, D dsh, K Kimi). * fastest engine and its lead over the next. Bars: 0 to 188.2 tok/s.")
        seps = [(n.attrs["class"], squash(n), "hidden" in n.attrs) for p in parts for n in p.find_all("span")
                if n.attrs.get("class", "").startswith("sep")]
        self.assertEqual(seps, [("sep-more", "; boxed:", False), ("sep-end", ".", True), ("sep", ",", True)])
        self.assertEqual([squash(n) for n in parts[0].find_all("b")], ["4/5"])
        # A letter never ends a line apart from its client, nor the top speed apart from its unit.
        self.assertEqual([n.text() for n in parts[2].find_all("span", "nw")],
                         ["O OpenCode", "H Hermes", "P Pi", "D dsh", "K Kimi"])
        self.assertEqual([n.text() for n in parts[4].find_all("span", "nw")], ["188.2 tok/s"])
        marked = parts[1]
        self.assertEqual(("data-cls" in marked.attrs, [(n.attrs["class"], n.text(), n.attrs.get("aria-hidden"))
                                                       for n in marked.find_all("span", "lamp")], spoken(marked)),
                         (True, [("lamp not-supported", "O", "true")], "a letter in a box with its corner struck: not supported"))
        self.assertEqual(spoken(parts[3]), "An asterisk: fastest engine and its lead over the next.")
        # Every class, a failed setup step: the default run without K2's kimi row, and a 2026-11-19 run of the
        # qwen3.6 llama.cpp pair whose status, hermes-z and pi fail, the note naming hermes-z.
        k2_kimi = row(prefix("llamacpp", "k2-horizon-7b"), "kimi")
        self.fixture.run(*DEFAULT_RUN, [line for line in default_rows() if line != k2_kimi],
                         readme="# e2e\n\n## The No rows\n\nNone.\n")
        qwen = prefix("llamacpp", "qwen3.6-35b-a3b")
        # The sample is the client with the most results of the class, ties to the earlier client (plan ruling 9):
        # known issue OpenCode 1, Hermes 2 gives H; failed Pi 1, Kimi 1 gives P.
        gpt, gpt_vllm, gpt_sglang = (prefix(runtime, "gpt-oss-20b") for runtime in ("llamacpp", "vllm", "sglang"))
        self.fixture.run("2026-11-19", "llmctl-next-e2e", [row(qwen, "up"), row(qwen, "status", "no answer"),
                                                           row(qwen, "hermes-z", "no answer"), row(qwen, "pi", "no answer"),
                                                           row(gpt, "up"), row(gpt, "opencode", "no answer"),
                                                           row(gpt_vllm, "up"), row(gpt_vllm, "hermes-z", "no answer"),
                                                           row(gpt_sglang, "up"), row(gpt_sglang, "kimi", "no answer")])
        self.fixture.coverage({cell: self.fixture._default_cell(*cell)
                               + f"; e2e 2026-11-19: `{step}` wrote hello.py elsewhere"
                               for cell, step in ((("qwen3.6-35b-a3b", "llamacpp"), "hermes-z"),
                                                  (("gpt-oss-20b", "llamacpp"), "opencode"),
                                                  (("gpt-oss-20b", "vllm"), "hermes-z"))})
        key = parse(self.page()).by_id("key")
        parts = [n for n in key.children if not isinstance(n, str)]
        self.assertEqual([p.attrs["data-k"] for p in parts], list(data.KEY_PARTS))
        classes = [p for p in parts if "data-cls" in p.attrs]
        self.assertEqual([[n.text() for n in p.find_all("span", "lamp")] for p in classes], [["H"], ["P"], ["K"], ["O"]])
        self.assertEqual([[("hidden" in n.attrs) for n in p.find_all("span", "sep")] for p in classes],
                         [[False], [False], [False], [True]])
        self.assertEqual(visible(key), "4/5: clients whose latest session passed; boxed: H known issue, P failed, no note yet, "
                                       "K not tested, O not supported (O OpenCode, H Hermes, P Pi, D dsh, K Kimi). Setup check "
                                       "failed: a setup step failed in the pair's newest e2e run; the details list it. * fastest "
                                       "engine and its lead over the next. Bars: 0 to 188.2 tok/s.")
        self.assertEqual([squash(n) for n in parts[6].find_all("span", "su")], ["Setup check failed"])
        # A part the key does not know stops the build instead of printing the bars sentence.
        real_view = data.view
        with mock.patch.object(data, "view", lambda *args: dataclasses.replace(real_view(*args), key=("counts", "size"))):
            with self.assertRaisesRegex(ValueError, "^unknown key part 'size'$"):
                self.page()
        # No exception class in the default view, and a whole-number top speed: a fresh fixture without the DeepSeek
        # vLLM pair, the only one OpenCode does not support, and mellum2.1-12b at 200.0.
        plain = tempfile.TemporaryDirectory()
        self.addCleanup(plain.cleanup)
        fixture = SiteFixture(Path(plain.name))
        (fixture.root / "recipes/vllm/deepseek-r1-0528-qwen3-8b.toml").unlink()
        fixture.coverage({("deepseek-r1-0528-qwen3-8b", "vllm"): DOES_NOT_FIT_CELL})
        fixture.set_value("llamacpp", "mellum2.1-12b", "decode_tok_s", "200.0")
        tree = parse(render.page(fixture.load(), BUILD, fixture.root))
        key = tree.by_id("key")
        parts = [n for n in key.children if not isinstance(n, str)]
        self.assertEqual([p.attrs["data-k"] for p in parts], ["counts", "fastest", "bars"])
        self.assertEqual(visible(key), "4/5: clients whose latest session passed. * fastest engine and its lead over the "
                                       "next. Bars: 0 to 200.0 tok/s.")
        self.assertEqual([(n.attrs["class"], "hidden" in n.attrs) for n in key.find_all("span")
                          if n.attrs.get("class", "").startswith("sep")], [("sep-more", True), ("sep-end", False)])
        self.assertEqual([squash(n) for n in parts[0].find_all("b")], ["4/5"])
        mellum = next(card for card in tree.by_id("pick-list").find_all("li", "pick")
                      if card.attrs["data-for"] == "d-llamacpp-mellum2.1-12b")
        self.assertEqual(squash(mellum.find_all("p", "pn")[0]), "200.0 tok/s, 128k context")

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
        # A policy in a meta element governs only what is parsed after it, so it comes before any script, style or link.
        policy_at = page.index('http-equiv="Content-Security-Policy"')
        self.assertEqual([tag for tag in ("<script", "<style", "<link") if page.index(tag) < policy_at], [])

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
        table = tree.by_id("matrix")
        self.assertEqual({name: table.attrs.get(name) for name in ("role", "tabindex", "data-pairs", "data-use", "data-sort")},
                         {"role": "table", "tabindex": "-1", "data-pairs": "18", "data-use": "", "data-sort": "rank"})
        self.assertEqual([node.find_all("table")[0].attrs.get("id") for node in tree.find_all("div", "tbl")], ["matrix"])
        self.assertEqual(table.find_all("span", "engine"), [])
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
                    self.assertEqual(len(tds[cell.runtime].find_all("a")), 1)
                    out = tds[cell.runtime].find_all("span", "out")
                    self.assertEqual([(squash(node), "hidden" in node.attrs) for node in out], [("Not in this view", True)])
                    if cell.status != "runs":
                        self.assertEqual(tile.attrs["class"], "tile x")
                        self.assertEqual(sorted(attrs), ["data-engine", "data-search", "data-status", "role"])
                        continue
                    self.assertEqual(tile.attrs["class"], "tile")
                    facts = cell.recipe
                    expected = {"data-ctx": str(facts.context // 1024), "data-cap": str(facts.memory_cap),
                                "data-think": facts.thinking, "data-tools": "no" if facts.chat_only else "yes",
                                "data-use": " ".join(cell.uses)}
                    expected.update({f"data-c-{result.client}": result.cls for result in cell.clients})
                    expected.update({f"data-o-{key}": str(place) for key, place in cell.positions})
                    if facts.decode is not None:
                        expected["data-speed"] = f"{facts.decode:g}"
                    if cell.fastest is not None:
                        expected["data-fast"] = None
                    if (model.id, cell.runtime) == matrix.default:
                        expected["data-default"] = None
                    if cell.setup:
                        expected["data-setup"] = None
                    self.assertEqual({name: attrs[name] for name in expected if name in attrs}, expected)
                    self.assertEqual(sorted(attrs), sorted([*expected, "data-engine", "data-status", "data-search",
                                                            "role"]))
        body = [node.attrs["id"] for node in table.find_all("tbody")[0].find_all("tr")]
        self.assertEqual([row_id[2:] for row_id in body if row_id.startswith("m-")],
                         list(data.view(matrix, "", data.Filters(), None).rows))
        self.assertEqual(body[body.index("grp") + 1], "m-mellum2.1-12b")
        # The row headers: the rank, the whole ID and the Chat only tag with its spoken words.
        deepseek = tree.by_id("m-deepseek-r1-0528-qwen3-8b").find_all("th")[0]
        self.assertEqual([(node.attrs["class"], squash(node)) for node in deepseek.children if not isinstance(node, str)
                          and node.attrs["class"] != "gv"],
                         [("rk", "3"), ("id", "deepseek-r1-0528-qwen3-8b"), ("tag", "Chat only, no tool calling")])
        tag = deepseek.find_all("span", "tag")[0]
        self.assertEqual((visible(tag), spoken(tag)), ("Chat only", "Chat only, no tool calling"))
        mellum = tree.by_id("m-mellum2.1-12b").find_all("th")[0]
        self.assertEqual((squash(mellum.find_all("span", "rk")[0]), mellum.find_all("span", "tag")), ("", []))
        size = lambda model_id, runtime: f"{cell_of(matrix, model_id, runtime).recipe.size_bytes / 1e9:.1f} GB"
        parts = lambda model_id, runtime: [(node.attrs["class"], visible(node)) for node in
                                           self.tile(tree, model_id, runtime).find_all("span", "g")[0].children
                                           if not isinstance(node, str)]
        self.assertEqual(parts("qwen3.6-35b-a3b", "sglang"),
                         [("sp", "151.3"), ("mk", "* 24%"), ("cx", "128k"), ("sz", size("qwen3.6-35b-a3b", "sglang")),
                          ("pc", "5/5"), ("bar", ""), ("vh", "")])
        self.assertEqual(parts("qwen3.6-35b-a3b", "llamacpp"),
                         [("sp", "122.5"), ("dm", "OpenCode default"), ("cx", "128k"),
                          ("sz", size("qwen3.6-35b-a3b", "llamacpp")), ("pc", "5/5"), ("bar", ""), ("vh", "")])
        self.assertEqual(tile_name(self.tile(tree, "qwen3.6-35b-a3b", "sglang")),
                         "151.3 tok/s, fastest, 24% ahead of llama.cpp, 128k context, SGLang: All 5 clients passed.")
        self.assertEqual(tile_name(self.tile(tree, "qwen3.6-35b-a3b", "llamacpp"), drop=("sz",)),
                         "122.5 tok/s, default in the OpenCode config, 128k context, llama.cpp: All 5 clients passed.")
        deepseek = self.tile(tree, "deepseek-r1-0528-qwen3-8b", "vllm")
        self.assertEqual(tile_name(deepseek), "72.5 tok/s, fastest, 3% ahead of llama.cpp, 64k context, "
                                              "vLLM: 4 of 5 clients; OpenCode: not supported.")
        pc = deepseek.find_all("span", "pc")[0]
        self.assertEqual((pc.attrs.get("aria-hidden"), squash(pc.find_all("span", "pcn")[0]),
                          [(n.attrs["class"], n.text()) for n in pc.find_all("span", "lamp")]),
                         ("true", "4/5", [("lamp not-supported", "O")]))
        # Run counts follow Known issue and Failed only: a client that was not tested has none.
        self.assertEqual(render.client_sentence((data.ClientResult("kimi", "not-tested", 0, 0, None, None),),
                                                {"kimi": "Kimi"}), "0 of 1 clients; Kimi: not tested.")
        widths = {("deepseek-r1-0528-qwen3-8b", "llamacpp"): "w-38", ("deepseek-r1-0528-qwen3-8b", "vllm"): "w-39",
                  ("gpt-oss-20b", "llamacpp"): "w-88", ("gpt-oss-20b", "vllm"): "w-44", ("gpt-oss-20b", "sglang"): "w-66",
                  ("k2-horizon-7b", "llamacpp"): "w-36", ("mellum2.1-12b", "llamacpp"): "w-100",
                  ("qwen3-coder-next", "llamacpp"): "w-38", ("qwen3.6-35b-a3b", "llamacpp"): "w-65",
                  ("qwen3.6-35b-a3b", "sglang"): "w-80"}
        self.assertEqual({key: self.tile(tree, *key).find_all("i")[0].attrs["class"] for key in widths}, widths)
        self.assertEqual({self.tile(tree, *key).find_all("span", "bar")[0].attrs.get("aria-hidden") for key in widths},
                         {"true"})
        self.assertEqual([visible(self.tile(tree, *key)) for key in (("deepseek-r1-0528-qwen3-8b", "sglang"),
                                                                    ("k2-horizon-7b", "vllm"))],
                         ["Failed 2026-11-17", "Does not fit (estimate)"])
        # The stylesheet's hooks: the failed word in b, the date in span.dt (visually hidden on phones), the evidence
        # label kept whole.
        hooks = lambda key: [(node.tag, node.attrs.get("class"), squash(node)) for node in
                             self.tile(tree, *key).find_all("span", "miss")[0].children if not isinstance(node, str)]
        self.assertEqual(hooks(("deepseek-r1-0528-qwen3-8b", "sglang")), [("b", None, "Failed"), ("span", "dt", "2026-11-17")])
        self.assertEqual(hooks(("k2-horizon-7b", "vllm")), [("span", "nw", "(estimate)")])
        # A failed setup step, a known issue and a failure, and a Runs cell without a speed.
        cell = ("qwen3.6-35b-a3b", "llamacpp")
        self.fixture.coverage({cell: self.fixture._default_cell(*cell)
                               + "; e2e 2026-11-19: `hermes-z` wrote hello.py elsewhere"})
        qwen = prefix("llamacpp", "qwen3.6-35b-a3b")
        self.fixture.run("2026-11-19", "llmctl-next-e2e", [row(qwen, "up"), row(qwen, "status", "no answer"),
                                                           row(qwen, "hermes-z", "no answer"), row(qwen, "pi", "no answer")])
        self.fixture.set_value("vllm", "gpt-oss-20b", "decode_tok_s", None)
        tree = parse(self.page())
        tile = self.tile(tree, "gpt-oss-20b", "vllm")
        self.assertEqual(parts("gpt-oss-20b", "vllm")[0], ("sp", "not recorded"))
        self.assertEqual(tile_name(tile), "speed not recorded, 64k context, vLLM: All 5 clients passed.")
        self.assertNotIn("data-speed", tile_cell(tree, "gpt-oss-20b", "vllm").attrs)
        self.assertEqual(tile.find_all("i")[0].attrs["class"], "w-0")
        setups = [(model_id, td.attrs["data-engine"]) for model_id in sorted({r for _, r in RECIPES})
                  for td in tree.by_id("m-" + model_id).find_all("td") if "data-setup" in td.attrs]
        self.assertEqual(setups, [("qwen3.6-35b-a3b", "llamacpp")])
        qwen_tile = self.tile(tree, "qwen3.6-35b-a3b", "llamacpp")
        su = qwen_tile.find_all("span", "su")
        self.assertEqual([(visible(node), spoken(node)) for node in su], [("Setup check failed", "Setup check failed,")])
        self.assertEqual([node.attrs["class"] for node in qwen_tile.find_all("span", "g")[0].children
                          if not isinstance(node, str)], ["sp", "dm", "su", "cx", "sz", "pc", "bar", "vh"])
        self.assertTrue(tile_name(qwen_tile, drop=("sz",)).startswith(
            "122.5 tok/s, default in the OpenCode config, Setup check failed, 128k context, llama.cpp: "),
            tile_name(qwen_tile, drop=("sz",)))
        pc = qwen_tile.find_all("span", "pc")[0]
        self.assertEqual((squash(pc.find_all("span", "pcn")[0]),
                          [(n.attrs["class"], n.text()) for n in pc.find_all("span", "lamp")]),
                         ("3/5", [("lamp known-issue", "H"), ("lamp failed", "P")]))
        self.assertEqual(squash(qwen_tile.find_all("span", "g")[0].find_all("span", "vh")[-1]),
                         "llama.cpp: 3 of 5 clients; Hermes: known issue, 1 of 2 runs passed; Pi: failed, 1 of 2 runs "
                         "passed.")
        # data-search holds the model ID, the aliases, the recipe name and the engine label, lower-cased (Cells).
        self.fixture.set_value("llamacpp", "k2-horizon-7b", "aliases", '["k2-7b", "horizon"]')
        self.fixture.set_value("llamacpp", "k2-horizon-7b", "name", '"K2 Horizon Q4"')
        tree = parse(self.page())
        self.assertEqual(tile_cell(tree, "k2-horizon-7b", "llamacpp").attrs["data-search"],
                         "k2-horizon-7b k2-7b horizon k2 horizon q4 llama.cpp")
        self.assertEqual(tile_cell(tree, "k2-horizon-7b", "vllm").attrs["data-search"],
                         "k2-horizon-7b k2-7b horizon vllm")
        # The tied and the under 1% forms of the fastest marker: the visible words hidden from readers, the spoken
        # words visually hidden.
        for runtime in ("llamacpp", "sglang"):                                   # vLLM has no speed since above
            self.fixture.set_value(runtime, "gpt-oss-20b", "decode_tok_s", "100.0")
        self.fixture.set_value("llamacpp", "deepseek-r1-0528-qwen3-8b", "decode_tok_s", "100.1")
        self.fixture.set_value("vllm", "deepseek-r1-0528-qwen3-8b", "decode_tok_s", "100.0")
        tree = parse(self.page())
        marks = lambda key: [(node.attrs.get("class"), node.attrs.get("aria-hidden"), node.text())
                             for node in self.tile(tree, *key).find_all("span", "mk")[0].children if not isinstance(node, str)]
        tied = self.tile(tree, "gpt-oss-20b", "llamacpp")
        self.assertEqual(marks(("gpt-oss-20b", "llamacpp")),
                         [(None, "true", "* tied"), ("vh", None, "tied fastest with SGLang,")])
        self.assertEqual((visible(tied.find_all("span", "mk")[0]), tile_name(tied)),
                         ("* tied", "100.0 tok/s, tied fastest with SGLang, 128k context, llama.cpp: All 5 clients passed."))
        under = self.tile(tree, "deepseek-r1-0528-qwen3-8b", "llamacpp")
        self.assertEqual(visible(under.find_all("span", "mk")[0]), "* under 1%")
        self.assertEqual(marks(("deepseek-r1-0528-qwen3-8b", "llamacpp"))[0], (None, "true", "* "))
        self.assertTrue(tile_name(under).startswith("100.1 tok/s, fastest, under 1% ahead of vLLM, 64k context, "),
                        tile_name(under))

    def test_sort_values_are_in_the_markup(self):
        matrix = self.fixture.load()
        tree = parse(render.page(matrix, BUILD, self.fixture.root))
        for model in matrix.models:
            for cell in model.cells:
                if cell.status != "runs":
                    continue
                with self.subTest(model=model.id, runtime=cell.runtime):
                    size = f"{cell.recipe.size_bytes / 1e9:.1f} GB"
                    sz = self.tile(tree, model.id, cell.runtime).find_all("span", "sz")
                    self.assertEqual([(visible(node), spoken(node)) for node in sz], [(size, size + " download,")])
        gaps = {}
        for node in tree.by_id("matrix").find_all("tbody")[0].find_all("tr"):
            for gv in node.find_all("span", "gv"):
                self.assertTrue(squash(gv).startswith("; "), squash(gv))
                gaps[node.attrs["id"][2:]] = visible(gv)
        self.assertEqual(gaps, {"qwen3.6-35b-a3b": "SGLang 24% ahead of llama.cpp",
                                "gpt-oss-20b": "llama.cpp 99% ahead of vLLM",
                                "deepseek-r1-0528-qwen3-8b": "vLLM 3% ahead of llama.cpp"})

    def test_divider_row_before_the_first_unranked_model(self):
        def body(tree):
            return tree.by_id("matrix").find_all("tbody")[0].find_all("tr")

        tree = parse(self.page())
        self.assertEqual([node.attrs["id"] for node in body(tree)],
                         ["m-qwen3.6-35b-a3b", "m-gpt-oss-20b", "m-deepseek-r1-0528-qwen3-8b", "grp", "m-mellum2.1-12b",
                          "m-qwen3-coder-next", "m-k2-horizon-7b"])
        grp = tree.by_id("grp")
        self.assertEqual((grp.attrs.get("class"), grp.attrs.get("role"), "hidden" in grp.attrs), ("grp", "row", False))
        self.assertEqual([(th.attrs.get("colspan"), th.attrs.get("scope"), th.attrs.get("role"), squash(th))
                          for th in grp.find_all("th")], [("4", "rowgroup", "rowheader", "Not in the Top 3")])
        ids = ("qwen3.6-35b-a3b", "gpt-oss-20b", "deepseek-r1-0528-qwen3-8b", "mellum2.1-12b", "qwen3-coder-next",
               "k2-horizon-7b")
        self.fixture.top10([TOP10_ROW.format(rank=rank, id=model_id,
                                             repo=self.fixture._read("llamacpp", model_id)["model"]["repo"])
                            for rank, model_id in enumerate(ids, start=1)])
        tree = parse(self.page())
        rows = body(tree)
        self.assertEqual([node.attrs["id"] for node in rows], [*(f"m-{model_id}" for model_id in ids), "grp"])
        self.assertEqual(("hidden" in rows[-1].attrs, squash(rows[-1])), (True, "Not in the Top 6"))
        # The divider follows View.divider: a ranked model without a Runs cell sorts last, so no unranked row
        # follows a ranked one and the divider stays hidden at the end (R-D16).
        target = self.fixture.root / "recipes/llamacpp/k2-horizon-7b.toml"
        text = target.read_text(encoding="utf-8")
        target.write_text("".join(line + "\n" for line in text.splitlines() if not line.startswith("decode_tok_s = "))
                          .replace('kind = "measured"', f'kind = "estimate"\ndoc = "{tomllib.loads(text)["doc"]}"'),
                          encoding="utf-8")
        self.fixture.coverage({("k2-horizon-7b", "llamacpp"): "Failed: out of memory, measured 2026-11-17 "
                               "([record](../research/2026-11-17/fixture-failure/README.md))"})
        ranked = ("qwen3.6-35b-a3b", "gpt-oss-20b", "deepseek-r1-0528-qwen3-8b", "k2-horizon-7b")
        self.fixture.top10([TOP10_ROW.format(rank=rank, id=model_id,
                                             repo=self.fixture._read("llamacpp", model_id)["model"]["repo"])
                            for rank, model_id in enumerate(ranked, start=1)])
        rows = body(parse(self.page()))
        self.assertEqual([(node.attrs["id"], "hidden" in node.attrs) for node in rows],
                         [("m-qwen3.6-35b-a3b", False), ("m-gpt-oss-20b", False), ("m-deepseek-r1-0528-qwen3-8b", False),
                          ("m-mellum2.1-12b", False), ("m-qwen3-coder-next", False), ("m-k2-horizon-7b", False),
                          ("grp", True)])

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
        # Free repository text in four more places: a short and a long reason, a client version and an evidence
        # summary; each must reach the page escaped once, never as markup.
        hostile = '<i>"x"</i> &'
        self.fixture.set_value("llamacpp", "mellum2.1-12b", "summary", '"' + hostile.replace('"', '\\"') + '"')
        qwen = prefix("llamacpp", "qwen3.6-35b-a3b")
        self.fixture.run("2026-11-19", "llmctl-hostile-e2e",
                         [row(qwen, "up"), row(qwen, "pi", "y" * 121 + hostile), row(qwen, "kimi", hostile)],
                         records={"005-kimi-version": version_record(hostile)})
        page = self.page()
        self.assertEqual(well_formed_problems(page), [])
        self.assertNotIn("<i>", page)
        escaped = "&lt;i&gt;&quot;x&quot;&lt;/i&gt; &amp;"
        tree = parse(page)
        self.assertIn(f"<p>{escaped}</p>", page)                                             # the evidence summary
        self.assertIn(f"<code>{escaped}</code>", page)                                       # the client versions
        self.assertEqual(page.count(f"Kimi <code>{escaped}</code>"), 2)                      # footer and run line
        self.assertIn(f"{'y' * 121}{escaped}</p></details>", page)                           # the long reason
        self.assertIn(f"<code>kimi</code> failed: {escaped} (no record named after this step)", page)
        self.assertIn(hostile, text(tree, "d-llamacpp-mellum2.1-12b"))

    def test_header_and_footer_facts(self):
        matrix = self.fixture.load()
        facts = [cell.recipe for model in matrix.models for cell in model.cells if cell.status == "runs"]
        hardware = "; ".join(sorted({recipe.hardware for recipe in facts}))
        oldest, newest = min(recipe.date for recipe in facts), max(recipe.date for recipe in facts)
        tree = parse(render.page(matrix, BUILD, self.fixture.root))
        squash = lambda node: " ".join(node.text().split())
        elements = lambda node: [child for child in node.children if not isinstance(child, str)]
        picks = ("Picks: Highest ranked: qwen3.6-35b-a3b on SGLang; Fastest: mellum2.1-12b on llama.cpp; Fastest in the "
                 "Top 3: gpt-oss-20b on llama.cpp.")
        self.assertEqual(squash(tree.find_all("p", "facts")[0]),
                         f"6 models; 10 of 18 pairs run. {picks} 3 engines and 5 agent clients. Measured on {hardware}, "
                         f"{oldest} to {newest}. About the data")
        self.assertEqual([(squash(node), node.attrs["href"]) for node in tree.find_all("p", "facts")[0].find_all("a")],
                         [("About the data", "#about")])
        self.assertEqual([squash(node) for node in tree.find_all("p", "facts")[0].find_all("span", "nw")], [oldest, newest])
        self.assertEqual((tree.by_id("about").tag, squash(tree.by_id("about"))), ("h2", "About the data"))
        images = data._joined([f"{runtime.label} {runtime.image}" for runtime in matrix.runtimes])
        speeds_of = lambda hardware: ("Speeds and contexts: " + ", ".join(f"(measured on {hardware}, {runtime.heading})"
                                                                          for runtime in matrix.runtimes[:-1])
                                      + f" and (measured on {hardware}, {matrix.runtimes[-1].heading}); Does not fit: "
                                        "(estimate).")
        speeds = speeds_of(hardware)
        n = len(default_rows())
        line = (f"2026-11-18, llmctl-fixture-e2e: {n} of {n} rows passed over 10 recipes; clients OpenCode, Hermes, Pi, "
                "dsh, Kimi; versions OpenCode opencode v2.3.0, Hermes Hermes Agent v0.23.1, Pi 1.5.2, dsh 0.2.2-rc.3, "
                "Kimi 2.5.3.")
        self.assertEqual([squash(li) for li in tree.by_id("runs").find_all("li")], [line])
        self.assertIsNone(tree.by_id("ranked-without-recipe"))
        # The header's and footer's fixed words and links (Header and controls, Footer, Copy), typed out rather than
        # taken from render.
        links = lambda node: [(squash(a), a.attrs["href"]) for a in node.find_all("a")]
        header = tree.find_all("header")[0]
        self.assertEqual((squash(tree.find_all("title")[0]), squash(header.find_all("h1")[0])),
                         ("Local LLMs on one laptop", "Local LLMs on one laptop"))
        self.assertEqual([node_id for node_id in ("versions", "legend", "uses", "toolbar", "filters", "filters-label")
                          if tree.by_id(node_id) is not None], [])
        self.assertEqual((header.find_all("p", "lead"), header.find_all("p", "links")), ([], []))
        # Order and the first screen: div.top holds the title and the control row, then the facts, the rule and the
        # notice.
        self.assertEqual([(node.tag, node.attrs.get("id") or node.attrs.get("class")) for node in elements(header)],
                         [("div", "top"), ("p", "facts"), ("p", "use-rule"), ("div", "notice")])
        self.assertEqual([(node.tag, node.attrs.get("id")) for node in elements(header.find_all("div", "top")[0])],
                         [("h1", None), ("div", "controls")])
        self.assertEqual([tree.by_id(node_id).attrs.get("class") for node_id in ("ctl", "usec", "sortc")],
                         ["ctl", "pair", "pair"])
        controls = tree.by_id("controls")
        self.assertEqual((controls.tag, "hidden" in controls.attrs, [(node.tag, node.attrs.get("id"))
                                                                     for node in elements(controls)]),
                         ("div", True, [("div", "ctl")]))
        self.assertEqual([(node.tag, node.attrs.get("id") or node.attrs.get("class")) for node in elements(tree.by_id("ctl"))],
                         [("span", "usec"), ("span", "q"), ("button", "open-filters"), ("span", "sortc"),
                          ("button", "copy-link")])
        use = tree.by_id("use")
        self.assertEqual((use.tag, use.attrs.get("aria-describedby")), ("select", "use-rule"))
        self.assertEqual([(node.attrs["value"], node.attrs["data-sort"], node.attrs["data-roles"], squash(node),
                           "selected" in node.attrs) for node in use.find_all("option")],
                         [(u.id, u.sort, " ".join(u.roles), data.view_name(u, matrix), not u.id) for u in data.USE_CASES])
        labels = {node.attrs["for"]: node for node in tree.find_all("label") if node.attrs.get("for")}
        self.assertEqual([(squash(labels[name]), labels[name].attrs.get("class")) for name in ("use", "q", "sort")],
                         [("View", None), ("Search models", "vh"), ("Sort", None)])
        self.assertEqual({name: tree.by_id("q").attrs.get(name) for name in ("type", "placeholder", "data-short")},
                         {"type": "search", "placeholder": "Search models", "data-short": "Search"})
        self.assertEqual([(node.tag, squash(node), node.attrs.get("aria-haspopup"))
                          for node in (tree.by_id("open-filters"), tree.by_id("copy-link"), tree.by_id("dismiss"))],
                         [("button", "Filters", "dialog"), ("button", "Copy link", None), ("button", "Dismiss", None)])
        self.assertEqual([(node.attrs["value"], squash(node), "selected" in node.attrs)
                          for node in tree.by_id("sort").find_all("option")],
                         [(key, data.sort_label(key, matrix), key == "rank") for key, _ in data.SORTS])
        rule = tree.by_id("use-rule")
        self.assertEqual((rule.tag, rule.attrs.get("class"), "hidden" in rule.attrs, rule.text()), ("p", "rule", True, ""))
        self.assertEqual([(squash(node), node.attrs["href"]) for node in tree.find_all("a", "skip")],
                         [("Skip to the table", "#matrix")])
        self.assertEqual(links(tree.by_id("runs")),
                         [("llmctl-fixture-e2e", BLOB + "/research/2026-11-18/llmctl-fixture-e2e/README.md#the-no-rows")])
        description = next(node.attrs["content"] for node in tree.find_all("meta") if node.attrs.get("name") == "description")
        self.assertEqual(description, "Which open-weight LLMs run on one laptop with llama.cpp, vLLM and SGLang, how "
                                      f"fast, and with which agent clients; measured on {hardware}.")
        footer = tree.find_all("footer")[0]
        self.assertEqual((footer.find_all("h2")[0].attrs.get("id"), squash(footer.find_all("h2")[0])), ("about", "About the data"))
        self.assertEqual([squash(n) for n in footer.children if not isinstance(n, str) and n.tag in ("p", "h3")], [
            "Built from the repository at commit " + "a" * 40 + ", committed 1970-01-01 00:00 UTC.",
            "A pair is one model on one engine: one cell of the table.",
            f"Images llmctl starts: {images}.",
            "Client versions in the e2e of 2026-11-18: OpenCode opencode v2.3.0, Hermes Hermes Agent v0.23.1, Pi 1.5.2, "
            "dsh 0.2.2-rc.3 and Kimi 2.5.3.",
            speeds,
            "How to read the client results",
            "An e2e (end-to-end) run starts each recipe with llmctl and runs every agent client's session against it.",
            "Run counts in the client lines count every e2e run of the same model on the same engine, so they can span "
            "changes to the pair's settings.",
            "The 5 agent clients (OpenCode, Hermes, Pi, dsh and Kimi) are the coding agents llmctl points at the running "
            "model; docs/clients.md and docs/opencode.md describe each one.",
            "Default in the OpenCode config: llamacpp-docker/qwen3.6-35b-a3b, the top-level model of the validated config; "
            "the Coding in OpenCode view marks it.",
            "What the rank means",
            "Rank is the model's place in the Top 3 of docs/model-selection.md, whose decision summary says how the ranking "
            "is made.",
            "Sources",
            "Data: the recipes in recipes/, the runtime coverage and Top 3 of docs/model-selection.md, the llmctl clients e2e "
            "summaries in research/, and the validated config of docs/opencode.md.",
            "The e2e runs:", "License, as README.md states it:",
            "This page sets no cookies, has no analytics and loads nothing from other sites."])
        self.assertEqual([(node.attrs.get("id"), squash(node)) for node in footer.find_all("h3")],
                         [("read-clients", "How to read the client results"), ("top10", "What the rank means"),
                          (None, "Sources")])
        self.assertEqual([" ".join(_text_without_lamps(li).split()) for li in footer.find_all("ul", "classes")[0].find_all("li")], [
            "Passed: the latest e2e session passed.",
            "Known issue: the latest session failed, and the runtime coverage table describes the failure.",
            "Failed: the latest session failed, and the runtime coverage table does not describe it yet.",
            "Not tested: no e2e run has covered this pair with this client yet.",
            "Not supported: llmctl writes no entry for this client on this pair."])
        # Each class item starts with a sample mark of its class: the first client in NAMES with that class on a Runs
        # cell, else the first client (2026-11-19 plan ruling 9); on the fixture that is OpenCode for every class.
        self.assertEqual([(elements(li)[0].tag, elements(li)[0].attrs, squash(elements(li)[0]))
                          for li in footer.find_all("ul", "classes")[0].find_all("li")],
                         [("span", {"class": f"lamp {cls}", "aria-hidden": "true"}, "O") for cls in data.CLASSES])
        self.assertEqual([[squash(node) for node in li.find_all("b")]
                          for li in footer.find_all("ul", "classes")[0].find_all("li")],
                         [["Passed"], ["Known issue"], ["Failed"], ["Not tested"], ["Not supported"]])
        paragraphs = {" ".join(squash(node).split()[:2]): node for node in elements(footer) if node.tag == "p"}
        self.assertEqual(links(paragraphs["Built from"]), [("a" * 40, TREE)])
        self.assertEqual(links(paragraphs["The 5"]), [("docs/clients.md", BLOB + "/docs/clients.md"),
                                                  ("docs/opencode.md", BLOB + "/docs/opencode.md")])
        self.assertEqual((links(paragraphs["Default in"]), [squash(node) for node in paragraphs["Default in"].find_all("code")]),
                         ([("validated config", BLOB + "/docs/opencode.md#the-validated-config")],
                          ["llamacpp-docker/qwen3.6-35b-a3b"]))
        self.assertEqual(links(paragraphs["Rank is"]), [("docs/model-selection.md", BLOB + "/docs/model-selection.md#top-10"),
                                                   ("decision summary", BLOB + "/docs/model-selection.md#decision-summary")])
        self.assertEqual(links(paragraphs["Data: the"]), [("recipes/", TREE + "/recipes"),
                                                    ("runtime coverage", BLOB + "/docs/model-selection.md#runtime-coverage"),
                                                    ("Top 3", BLOB + "/docs/model-selection.md#top-10"),
                                                    ("research/", TREE + "/research"),
                                                    ("validated config", BLOB + "/docs/opencode.md#the-validated-config")])
        versions = lambda tree: next(squash(node) for node in tree.find_all("footer")[0].find_all("p")
                                     if squash(node).startswith("Client versions"))
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
        self.assertEqual(squash(ranked), "Ranked in the Top 4, without a shipped recipe yet: rank 4, Unshipped.")
        self.assertEqual([node.attrs["href"] for node in ranked.find_all("a")], ["https://huggingface.co/org/unshipped"])
        self.assertTrue(versions(tree).endswith("dsh 0.2.2-rc.3 and Kimi not recorded."), versions(tree))
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
                         f"6 models; 10 of 18 pairs run. {picks.replace('Top 3', 'Top 4')} 3 engines and 5 agent clients. "
                         f"Measured on {hardware}, {oldest} to {newest}. About the data")
        self.assertEqual(versions(tree),
                         "Client versions in the e2e of 2026-11-18: OpenCode opencode v2.3.0, Hermes Hermes Agent v0.23.1 "
                         "(llmctl-fixture-e2e), Hermes Hermes Agent v0.24.0 (llmctl-z-e2e), Pi 1.5.2, dsh 0.2.2-rc.3 and "
                         "Kimi 2.4.0 (2026-11-16).")
        runs = [squash(li) for li in tree.by_id("runs").find_all("li")]
        self.assertEqual((runs[0], runs[3]),
                         ("2026-11-16, llmctl-kimi-e2e: 2 of 2 rows passed over 1 recipe; clients Kimi; versions Kimi "
                          "2.4.0.",
                          "2026-11-18, llmctl-z-e2e: 3 of 4 rows passed over 2 recipes, 1 row of a recipe or engine that "
                          "no longer ships; clients Hermes; versions Hermes Hermes Agent v0.24.0."))
        # The z run fails qwen3.6-35b-a3b's hermes-chat row, so Hermes is the first client with a Failed result and
        # gives that class its sample (2026-11-19 plan ruling 9).
        self.assertEqual([squash(li.find_all("span", "lamp")[0])
                          for li in tree.find_all("footer")[0].find_all("ul", "classes")[0].find_all("li")],
                         ["O", "O", "H", "O", "O"])
        speeds = speeds_of(hardware)
        self.assertEqual([squash(node) for node in tree.find_all("p") if squash(node).startswith("Speeds and contexts")],
                         [speeds])
        self.assertIn(speeds, [squash(node) for node in elements(tree.find_all("footer")[0]) if node.tag == "p"])
        footer = [squash(node) for node in tree.find_all("footer")[0].find_all("p")]
        self.assertEqual(footer[-1], "This page sets no cookies, has no analytics and loads nothing from other sites.")
        for record in sorted(self.fixture.root.glob("research/*/*/records/*-version")):
            record.unlink()
        tree = parse(self.page())
        self.assertEqual(versions(tree), "Client versions: not recorded.")
        self.fixture.top10([*rows, "| 4 | *Unshipped* | llama.cpp |"])
        with self.assertRaises(data.SiteError) as raised:
            self.page()
        self.assertEqual(str(raised.exception), "docs/model-selection.md: row 4, Model: column 1: emphasis")

    def test_filters_sheet_markup(self):
        matrix = self.fixture.load()
        tree = parse(render.page(matrix, BUILD, self.fixture.root))
        elements = lambda node: [child for child in node.children if not isinstance(child, str)]
        order = [(node.tag, node.attrs.get("id")) for node in elements(tree.find_all("body")[0])]
        at = order.index(("footer", None))
        self.assertEqual(order[at:at + 3], [("footer", None), ("dialog", "sheet"), ("dialog", "drawer")])
        sheet = tree.by_id("sheet")
        self.assertEqual(sheet.attrs, {"id": "sheet", "aria-labelledby": "sheet-title"})
        self.assertEqual([(node.tag, squash(node)) for node in (sheet.by_id("sheet-title"), sheet.by_id("sheet-close"))],
                         [("h2", "Filters"), ("button", "Close")])
        controls = [node for node in sheet.iter() if (node.tag, _classes(node)) in (("fieldset", {"group"}),
                                                                                     ("div", {"control"}))]
        self.assertEqual([squash((node.find_all("legend") or node.find_all("label"))[0]) for node in controls],
                         ["Engines", "Status", "Must pass in", "Context", "Thinking", "Tools", "Free RAM"])
        runs = [cell for model in matrix.models for cell in model.cells if cell.status == "runs"]

        def boxes(name):
            return [(node.attrs["value"], "checked" in node.attrs) for node in sheet.find_all("input")
                    if node.attrs.get("name") == name]

        def options(name):
            return [(node.attrs["value"], squash(node), "selected" in node.attrs)
                    for node in sheet.by_id(name).find_all("option")]

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
        self.assertEqual(sheet.by_id("ram").attrs.get("aria-describedby"), "ram-help")
        self.assertEqual(squash(sheet.by_id("ram-help")), "./llmctl up refuses to start a recipe when less RAM than its "
                                                           "container's memory cap is available.")
        slot = sheet.by_id("sheet-slot")
        self.assertEqual((slot.tag, slot.children), ("div", []))
        holder = next(node for node in sheet.iter() if any(child is slot for child in node.children))
        self.assertIs(elements(holder)[elements(holder).index(slot) - 1], controls[-1])
        self.assertEqual([(node.tag, node.attrs.get("id")) for node in elements(sheet.find_all("div", "sheet-foot")[0])],
                         [("button", "show"), ("button", "clear")])
        self.assertEqual([(node.tag, node.attrs.get("type"), squash(node)) for node in (sheet.by_id("show"),
                                                                                        sheet.by_id("clear"))],
                         [("button", "button", "Show 6 models"), ("button", "button", "Clear filters")])
        live = sheet.by_id("sheet-live")
        self.assertEqual((live.tag, live.attrs, live.text()),
                         ("p", {"id": "sheet-live", "class": "vh", "aria-live": "polite"}, ""))

    def test_opencode_default_mark(self):
        def check(model_id, runtime, model_string, words):
            tree = parse(self.page())
            table = tree.by_id("matrix")
            pairs = [(tr.attrs["data-model"], td) for tr in table.find_all("tr") if "data-model" in tr.attrs
                     for td in tr.find_all("td")]
            self.assertEqual([(model, td.attrs["data-engine"]) for model, td in pairs if "data-default" in td.attrs],
                             [(model_id, runtime)])
            self.assertEqual([(model, td.attrs["data-engine"]) for model, td in pairs if td.find_all("span", "dm")],
                             [(model_id, runtime)])
            mark = self.tile(tree, model_id, runtime).find_all("span", "dm")[0]
            self.assertEqual([(node.attrs, squash(node)) for node in mark.find_all("span", "tag")],
                             [({"class": "tag", "aria-hidden": "true"}, "OpenCode default")])
            self.assertEqual(spoken(mark), "default in the OpenCode config,")
            listed = next(node for node in tree.by_id("use-cases").find_all("section", "use")
                          if node.attrs["data-use"] == "opencode")
            self.assertEqual([squash(li) for li in listed.find_all("li") if "default in the" in squash(li)], [words])
            self.assertEqual([[squash(code) for code in node.find_all("code")] for node in tree.find_all("footer")[0].find_all("p")
                              if squash(node).startswith("Default in the OpenCode config: ")], [[model_string]])

        check("qwen3.6-35b-a3b", "llamacpp", "llamacpp-docker/qwen3.6-35b-a3b",
              "qwen3.6-35b-a3b: SGLang 151.3 tok/s, 128k; llama.cpp 122.5 tok/s, 128k, default in the OpenCode config")
        self.fixture.opencode(config_section("sglang-docker/gpt-oss-20b"))
        check("gpt-oss-20b", "sglang", "sglang-docker/gpt-oss-20b",
              "gpt-oss-20b: llama.cpp 166.2 tok/s, 128k; SGLang 124.6 tok/s, 128k, default in the OpenCode config; "
              "vLLM 83.5 tok/s, 64k")
        # A default pair that no longer meets the Coding in OpenCode rule leaves that view, and the footer says so.
        self.fixture.opencode(config_section("llamacpp-docker/qwen3.6-35b-a3b"))
        qwen = prefix("llamacpp", "qwen3.6-35b-a3b")
        self.fixture.run("2026-11-19", "llmctl-y-e2e", [row(qwen, "up"), row(qwen, "opencode", "it failed")])
        tree = parse(self.page())
        listed = next(node for node in tree.by_id("use-cases").find_all("section", "use")
                      if node.attrs["data-use"] == "opencode")
        self.assertEqual([squash(li) for li in listed.find_all("li") if "default in the" in squash(li)], [])
        self.assertEqual([squash(node) for node in tree.find_all("footer")[0].find_all("p")
                          if squash(node).startswith("Default in the OpenCode config: ")],
                         ["Default in the OpenCode config: llamacpp-docker/qwen3.6-35b-a3b, the top-level model of the "
                          "validated config; it does not meet the Coding in OpenCode rule today, so that view does not "
                          "show it."])

    def test_assets_are_checked_before_inlining(self):
        self.assertFalse(hasattr(render, "switch_width"))
        for css, js in (("a {}", ""), ("@ENGINES@ @ENGINES@", ""), ("@ENGINES@ </style>", ""), ("@ENGINES@", "a</script>")):
            with self.subTest(css=css, js=js), self.assertRaises(data.SiteError):
                render.inline_assets(css, js, 3, [0])
        self.assertEqual(render.inline_assets("a { b: repeat(@ENGINES@, 1fr); }\n", '"use strict";\n', 3, [65, 0, 100, 65]),
                         ("a { b: repeat(3, 1fr); }\n\n" + render.WIDTHS_NOTE + "\n.w-0 { width: 0%; }\n.w-65 { width: 65%; }\n"
                          ".w-100 { width: 100%; }\n", '"use strict";\n'))
        self.assertEqual(render.inline_assets("@ENGINES@", "", 4, []), ("4\n" + render.WIDTHS_NOTE + "\n", ""))
        # The page appends one rule per bar class it uses, ascending, so every bar with a speed has its width.
        tree = parse(self.page())
        used = sorted({int(i.attrs["class"][2:]) for bar in tree.find_all("span", "bar") for i in bar.find_all("i")})
        self.assertGreater(len(used), 2)
        style = tree.find_all("style")[0].text()
        self.assertTrue(style.endswith("\n" + render.WIDTHS_NOTE + "\n"
                                       + "".join(f".w-{n} {{ width: {n}%; }}\n" for n in used)), style[-300:])

    def test_stylesheet_keeps_hidden_elements_hidden(self):
        css = re.sub(r"/\*.*?\*/", "", (REPO / "launcher/site/page.css").read_text(encoding="utf-8"), flags=re.S)
        self.assertEqual(" ".join(css.split("}", 1)[0].split()) + " }", "[hidden] { display: none !important; }")
        header = [body for selector, body in re.findall(r"([^{}]*)\{([^{}]*)\}", css) if "thead" in selector]
        self.assertTrue(header)
        self.assertFalse(any(re.search(r"display\s*:\s*none", body) for body in header))
        # Phone layout: the model head is visually hidden by clipping, never removed.
        self.assertTrue(any(re.search(r"clip-path\s*:\s*inset\(50%\)", body) for selector, body in
                            re.findall(r"([^{}]*)\{([^{}]*)\}", css) if "thead" in selector and "th.mh" in selector))
        # Print keeps the table with its repeating head: the phone lines (max-width: 47.99rem) apply to screens only,
        # since a printed A4 or Letter page is narrower than 48rem.
        phones = re.findall(r"@media ([^{]*max-width: 47\.99rem[^{]*)\{", css)
        self.assertTrue(phones)
        self.assertEqual([query for query in phones if not query.startswith("screen and ")], [])
        lines = css.index("#matrix, #matrix caption, #matrix thead, #matrix tbody, #matrix tr, #matrix th, #matrix td")
        self.assertTrue(re.findall(r"@media ([^{]*)\{", css[:lines])[-1].startswith("screen and (max-width: 47.99rem)"))
        # R-D12: the phone line's engine columns come from the one engine-count token.
        self.assertEqual(css.count("@ENGINES@"), 1)
        self.assertIn("repeat(@ENGINES@, minmax(0, 1fr))", css)
        rules = [(" ".join(selector.split()), " ".join(body.split()))
                 for selector, body in re.findall(r"([^{}]*)\{([^{}]*)\}", css)]
        # R-D13: tiles and cards choose their layout by their own content box.
        self.assertTrue(any(selector in ("a.tile", ".tile") and "container: tile / inline-size" in body
                            for selector, body in rules))
        self.assertTrue(any(selector in ("li.pick", ".pick") and "container: card / inline-size" in body
                            for selector, body in rules))
        flat = " ".join(css.split())
        for query in ("@container tile (min-width: 16.5rem)", "@container tile (min-width: 18.5rem)",
                      "@container card (min-width: 23rem)"):
            self.assertIn(query, flat)
        self.assertTrue(any(selector == "#drawer-bar" and "position: sticky" in body for selector, body in rules))
        for token in ("--on-runs", "--failed-tint", "--nofit", "--hatch-a", "--hatch-b"):
            self.assertNotIn(token, css)
        # A cut reason keeps the native disclosure marker of its summary.
        self.assertFalse([body for selector, body in rules if "details.reason" in selector and "display" in body])
        # The key's setup words take the key's type, not the tile's smaller setup mark, and an engine head's run
        # count never leaves one word on a line of its own.
        self.assertIn(".key .su", [selector for selector, body in rules if "font-size: inherit;" in body])
        self.assertIn("th[data-engine] .n", [selector for selector, body in rules if "white-space: nowrap;" in body])
        # A Failed tile's date never breaks at its hyphens where it shows (from 48rem and in print).
        self.assertIn((".miss .dt", "white-space: nowrap;"), rules)
        # A tile shows the OpenCode default tag in Coding in OpenCode only (R-D11), its size under the size sort only
        # and its gap words under the gap sort only; hidden parts follow these rules, so the tile's name does too.
        for selector in ('#matrix:not([data-use="opencode"]) .dm', '#matrix:not([data-sort="size"]) .sz',
                         '#matrix:not([data-sort="gap"]) .gv'):
            self.assertIn((selector, "display: none;"), rules)
        # The engine name of the dialog's Engines list never breaks inside the word, and the phone grid gives it the
        # desktop label width.
        self.assertIn((".eg .en", "font-weight: 650; white-space: nowrap; overflow-wrap: normal;"), rules)
        phone_rules = [(" ".join(sel.split()), " ".join(body.split())) for query in
                       re.findall(r"@media screen and \(max-width: 47\.99rem\) \{(.*?)\n\}", css, flags=re.S)
                       for sel, body in re.findall(r"([^{}]*)\{([^{}]*)\}", query)]
        self.assertIn((".eg", "grid-template-columns: 5.25rem minmax(0, 1fr) auto; column-gap: 0.5rem;"), phone_rules)
        # The search placeholder is the field's only visible label, so it takes the muted text colour (4.5:1 and up).
        self.assertIn(("#q::placeholder", "color: var(--muted); opacity: 1;"), rules)
        # The card's Start commands summary stays inside the 9rem the role and the name keep free, and wraps under
        # larger text spacing (WCAG 1.4.12) instead of covering them.
        summary = " ".join(body for selector, body in rules if selector == ".pst > summary")
        self.assertIn("max-width: 9rem;", summary)
        self.assertIn("white-space: normal;", summary)
        self.assertNotIn("white-space: nowrap;", summary)
        # The title wraps rather than push the page sideways at 320 px with larger text.
        self.assertIn((".top h1", "flex: 0 1 auto; min-width: 0;"), rules)
        # Forced colors draw the Not supported box's border in the system text colour too.
        forced = re.search(r"@media \(forced-colors: active\) \{(.*?)\n\}", css, flags=re.S).group(1)
        self.assertIn("--edge: CanvasText;", forced)
        # A section in the dialog sits right under the zero-height sticky Close bar, so a load with its hash scrolls
        # its sticky title to the dialog's top with a zero margin, and the :target outline is for the jump without
        # JavaScript only.
        drawn = " ".join(body for selector, body in rules if selector == "#drawer-body section.details")
        self.assertRegex(drawn, r"scroll-margin-top: [0-9.]+rem;")
        self.assertIn(("#drawer-body section.details:target", "outline: none;"), rules)
        # Print: a details section may break across pages (each is taller than the space left on one), and a
        # heading never ends a page.
        printed = re.search(r"@media print \{(.*)\}\s*$", css, flags=re.S).group(1)
        self.assertNotIn("avoid-page", " ".join(body for selector, body in rules if "section.details" in selector))
        self.assertIn(("h2, h3, h4", "break-after: avoid-page;"),
                      [(" ".join(sel.split()), " ".join(body.split()))
                       for sel, body in re.findall(r"([^{}]*)\{([^{}]*)\}", printed)])

    def test_bars_without_any_speed(self):
        # R-D6: without any speed every bar is w-0, the build appends only that rule, and the key's bars part and the
        # dialog's scale line are left out.
        for runtime, recipe_id in RECIPES:
            self.fixture.set_value(runtime, recipe_id, "decode_tok_s", None)
        tree = parse(self.page())
        bars = [i for bar in tree.find_all("span", "bar") for i in bar.find_all("i")]
        self.assertTrue(bars)
        self.assertEqual({i.attrs.get("class") for i in bars}, {"w-0"})
        style = tree.find_all("style")[0].text()
        self.assertTrue(style.endswith("\n" + render.WIDTHS_NOTE + "\n.w-0 { width: 0%; }\n"), style[-200:])
        self.assertEqual(re.findall(r"\.w-\d+ \{", style), [".w-0 {"])
        self.assertEqual([node.attrs["data-k"] for node in tree.by_id("key").children
                          if not isinstance(node, str) and "data-k" in node.attrs],
                         ["counts", "not-supported", "letters"])
        self.assertEqual(tree.find_all("p", "scale"), [])
        self.assertEqual(text(tree, "d-llamacpp-qwen3.6-35b-a3b", "status-line"), "Speed not recorded; 128k context.")
        first = tree.by_id("pick-list").find_all("li", "pick")[0]
        self.assertEqual(squash(first.find_all("p", "pn")[0]), "Speed not recorded, 128k context")

    def test_script_keeps_no_state_outside_the_url(self):
        js = (REPO / "launcher/site/page.js").read_text(encoding="utf-8")
        for word in ("localStorage", "sessionStorage", "indexedDB", "document.cookie", ".style."):
            self.assertNotIn(word, js)
        # It never sets a style: no style property, no style attribute, no cssText.
        self.assertIsNone(re.search(r"\.style\b|[\"']style[\"']|cssText", js))

    def test_script_uses_only_safe_sinks(self):
        js = (REPO / "launcher/site/page.js").read_text(encoding="utf-8")
        for word in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write"):
            self.assertNotIn(word, js)
        self.assertIsNone(re.search(r"\beval\s*\(|\bFunction\s*\(", js))
        calls = re.findall(r"querySelector(?:All)?\(", js)
        literal = re.findall(r"querySelector(?:All)?\(\s*(?:\"[^\"\\]*\"|'[^'\\]*')\s*\)", js)
        self.assertEqual(len(calls), len(literal))
        self.assertIsNone(re.search(r"set(?:Timeout|Interval)\(\s*[\"'`]", js))

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
        # An --out the build cannot create ends in one line, as a source problem does (D13), not in a traceback.
        code, stdout, stderr = self.build(target / "out")
        self.assertEqual((code, stdout, stderr), (1, "", f"site: --out {target / 'out'}: Not a directory\n"))
        if os.geteuid() != 0:                                                        # root reads any file
            readme = self.fixture.root / "README.md"
            readme.chmod(0)
            self.addCleanup(readme.chmod, 0o644)
            code, stdout, stderr = self.build(Path(self._tmp.name) / "unread")
            self.assertEqual((code, stdout, stderr), (1, "", "site: README.md: Permission denied\n"))
        (out / "index.html").rmdir()
        (out / "index.html").symlink_to(target)
        code, stdout, stderr = self.build(out)
        self.assertEqual((code, stderr), (1, f"site: --out {out}: {out / 'index.html'} is not a regular file\n"))
        self.assertEqual(target.read_text(encoding="utf-8"), "keep\n")

    def test_site_reads_no_environment_and_runs_nothing(self):
        files = sorted((REPO / "launcher/site").glob("*.py")) + [REPO / "launcher/e2e/summary.py", REPO / "launcher/markdown.py"]
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

    def test_details_texts(self):
        tree = parse(render.page(self.fixture.load(), BUILD, self.fixture.root))
        status = lambda i: text(tree, i, "status-line")
        self.assertEqual([status(i) for i in ("d-sglang-qwen3.6-35b-a3b", "d-llamacpp-qwen3.6-35b-a3b", "d-llamacpp-k2-horizon-7b",
                                              "d-llamacpp-gpt-oss-20b")],
                         ["151.3 tok/s with 128k context. Fastest of 2 engines, 24% ahead of llama.cpp (122.5 tok/s).",
                          "122.5 tok/s with 128k context. SGLang is faster: 151.3 tok/s.",
                          "66.9 tok/s with 80k context. Runs only on llama.cpp.",
                          "166.2 tok/s with 128k context. Fastest of 3 engines, 33% ahead of SGLang (124.6 tok/s)."])
        engines = tree.by_id("d-llamacpp-deepseek-r1-0528-qwen3-8b").find_all("ul", "engines")[0]
        self.assertEqual([(squash(li), li.attrs, [a.attrs["href"] for a in li.find_all("a")])
                          for li in engines.children if not isinstance(li, str)],
                         [("llama.cpp 70.6 tok/s 64k", {"class": "cur", "aria-current": "true"}, []),
                          ("vLLM 72.5 tok/s 64k Fastest, 3% ahead", {}, ["#d-vllm-deepseek-r1-0528-qwen3-8b"]),
                          ("SGLang Failed 2026-11-17", {}, ["#d-sglang-deepseek-r1-0528-qwen3-8b"])])
        self.assertEqual([i.attrs["class"] for i in engines.find_all("i")], ["w-38", "w-39"])
        self.assertEqual([a.attrs["class"] for a in engines.find_all("a")], ["eg", "eg"])
        self.assertEqual(text(tree, "d-llamacpp-deepseek-r1-0528-qwen3-8b", "scale"),
                         "Decode speed on the page's scale, 0 to 188.2 tok/s.")
        scale = tree.by_id("d-llamacpp-deepseek-r1-0528-qwen3-8b").find_all("p", "scale")[0]
        self.assertEqual([node.text() for node in scale.find_all("span", "nw")], ["188.2 tok/s"])
        self.assertEqual([node.text() for node in engines.find_all("span", "nw")], ["2026-11-17"])
        thinking =lines(tree, "d-vllm-deepseek-r1-0528-qwen3-8b", "thinking")
        self.assertEqual(thinking[0], "Always on")
        self.assertEqual([line.split(":")[0] for line in thinking[1:]], ["Hermes", "Pi", "dsh", "Kimi"])  # after the type line
        self.assertEqual(lines(tree, "d-sglang-qwen3.6-35b-a3b", "thinking")[0], "Toggle, off by default")
        self.assertEqual(lines(tree, "d-vllm-gpt-oss-20b", "thinking")[0], "Effort levels: low, medium, high")
        self.assertEqual(lines(tree, "d-llamacpp-mellum2.1-12b", "thinking")[0], "Toggle, on by default")
        self.assertEqual(lines(tree, "d-llamacpp-qwen3-coder-next", "thinking")[0], "No thinking")
        for section_id in ("d-vllm-deepseek-r1-0528-qwen3-8b", "d-llamacpp-qwen3-coder-next"):  # inside Recipe settings
            settings = next(node for node in tree.by_id(section_id).find_all("details", "more")
                            if squash(node.find_all("summary")[0]) == "Recipe settings")
            self.assertEqual(len(settings.find_all("ul", "thinking")), 1)
        self.assertIn("OpenCode: not supported; llmctl writes no OpenCode entry for this pair.",
                      lines(tree, "d-vllm-deepseek-r1-0528-qwen3-8b", "clients"))
        self.assertIn("Hermes: passed, 1 of 1 runs (latest 2026-11-18).",
                      lines(tree, "d-vllm-deepseek-r1-0528-qwen3-8b", "clients"))
        for runtime, speed in (("llamacpp", "100.0"), ("vllm", "50.0"), ("sglang", "100.0")):
            self.fixture.set_value(runtime, "gpt-oss-20b", "decode_tok_s", speed)
        self.fixture.set_value("llamacpp", "deepseek-r1-0528-qwen3-8b", "decode_tok_s", "100.1")
        self.fixture.set_value("vllm", "deepseek-r1-0528-qwen3-8b", "decode_tok_s", "100.0")
        tree = parse(render.page(self.fixture.load(), BUILD, self.fixture.root))
        self.assertEqual(text(tree, "d-llamacpp-gpt-oss-20b", "status-line"),
                         "100.0 tok/s with 128k context. Tied fastest with SGLang at 100.0 tok/s.")
        self.assertEqual(text(tree, "d-vllm-gpt-oss-20b", "status-line"),
                         "50.0 tok/s with 64k context. llama.cpp and SGLang are faster: 100.0 tok/s.")
        self.assertIn("llama.cpp 100.0 tok/s 128k Tied fastest", lines(tree, "d-vllm-gpt-oss-20b", "engines"))
        decode = next(div for div in tree.by_id("d-llamacpp-gpt-oss-20b").find_all("dl", "kv")[0].find_all("div")
                      if squash(div.find_all("dt")[0]) == "Decode speed")
        self.assertEqual(squash(decode.find_all("dd")[0]), "100.0 tok/s")
        self.assertEqual(text(tree, "d-llamacpp-deepseek-r1-0528-qwen3-8b", "status-line"),
                         "100.1 tok/s with 64k context. Fastest of 2 engines, under 1% ahead of vLLM (100.0 tok/s).")
        self.fixture.set_value("sglang", "gpt-oss-20b", "decode_tok_s", "50.0")                   # tied at the second speed
        tree = parse(render.page(self.fixture.load(), BUILD, self.fixture.root))
        self.assertEqual(text(tree, "d-llamacpp-gpt-oss-20b", "status-line"),
                         "100.0 tok/s with 128k context. Fastest of 3 engines, 100% ahead of vLLM and SGLang (50.0 tok/s).")
        self.fixture.set_value("vllm", "gpt-oss-20b", "decode_tok_s", None)
        tree = parse(render.page(self.fixture.load(), BUILD, self.fixture.root))
        self.assertEqual(text(tree, "d-vllm-gpt-oss-20b", "status-line"), "Speed not recorded; 64k context.")
        self.assertIn("Decode speed not recorded", text(tree, "d-vllm-gpt-oss-20b"))
        self.assertEqual(text(tree, "d-llamacpp-gpt-oss-20b", "status-line"),
                         "100.0 tok/s with 128k context. Fastest of 2 engines, 100% ahead of SGLang (50.0 tok/s).")
        engines = tree.by_id("d-llamacpp-gpt-oss-20b").find_all("ul", "engines")[0]
        item = next(li for li in engines.find_all("li") if squash(li).startswith("vLLM"))
        self.assertEqual((squash(item), [i.attrs["class"] for i in item.find_all("i")]),
                         ("vLLM speed not recorded 64k", ["w-0"]))
        self.fixture.set_value("sglang", "gpt-oss-20b", "decode_tok_s", None)
        tree = parse(render.page(self.fixture.load(), BUILD, self.fixture.root))
        self.assertEqual(text(tree, "d-llamacpp-gpt-oss-20b", "status-line"), "100.0 tok/s with 128k context.")
        self.fixture.set_value("llamacpp", "mellum2.1-12b", "decode_tok_s", "200.0")          # the page's top
        tree = parse(render.page(self.fixture.load(), BUILD, self.fixture.root))
        self.assertEqual(text(tree, "d-llamacpp-gpt-oss-20b", "scale"), "Decode speed on the page's scale, 0 to 200.0 tok/s.")

    def rewrite_default_run(self, rows):
        self.fixture.run(*DEFAULT_RUN, rows, readme="# e2e\n\n## The No rows\n\nNone.\n")

    def test_client_lines(self):        # a 2026-11-19 run: default rows, except as the comments say
        # hermes-z and pi fail for prefix("llamacpp", "qwen3.6-35b-a3b"); up fails for prefix("sglang", "gpt-oss-20b")
        # with no row after it; the kimi row of prefix("llamacpp", "k2-horizon-7b") is left out; the qwen3.6 llama.cpp
        # note ends "; e2e 2026-11-19: `hermes-z` wrote hello.py elsewhere"
        k2_kimi = row(prefix("llamacpp", "k2-horizon-7b"), "kimi")
        base = [line for line in default_rows() if line != k2_kimi]
        self.rewrite_default_run(base)
        qwen, gpt = prefix("llamacpp", "qwen3.6-35b-a3b"), prefix("sglang", "gpt-oss-20b")
        rows = []
        for line in base:
            if line.startswith(f"| {gpt} | "):
                if line == row(gpt, "up"):
                    rows.append(row(gpt, "up", "the server did not start"))
                continue
            if line in (row(qwen, "hermes-z"), row(qwen, "pi")):
                line = row(qwen, line.split(" | ")[1], "no answer")
            rows.append(line)
        self.fixture.run("2026-11-19", "llmctl-next-e2e", rows)
        self.fixture.run("2026-11-17", "llmctl-old-e2e", [row(qwen, "up"), row(qwen, "dsh-headless", "no answer")])
        cell = ("qwen3.6-35b-a3b", "llamacpp")
        self.fixture.coverage({cell: self.fixture._default_cell(*cell)
                               + "; e2e 2026-11-19: `hermes-z` wrote hello.py elsewhere"})
        tree = parse(render.page(self.fixture.load(), BUILD, self.fixture.root))
        qwen_lines = lines(tree, "d-llamacpp-qwen3.6-35b-a3b", "clients")
        for line in ("Hermes: known issue, 1 of 2 runs passed (latest 2026-11-19); the coverage note below describes it.",
                     "Pi: failed, 1 of 2 runs passed (latest 2026-11-19); the runtime coverage table does not describe it "
                     "yet.", "dsh: passed, 2 of 3 runs passed (latest 2026-11-19).",
                     "Kimi: passed, 2 of 2 runs (latest 2026-11-19)."):
            with self.subTest(line=line):
                self.assertIn(line, qwen_lines)
        self.assertIn("OpenCode: failed, 1 of 2 runs passed (latest 2026-11-19); the runtime coverage table does not "
                      "describe it yet. Not reached: up failed.", lines(tree, "d-sglang-gpt-oss-20b", "clients"))
        self.assertEqual([squash(h4) for h4 in tree.by_id("d-sglang-gpt-oss-20b").find_all("h4")],
                         ["Start it", "Engines", "Clients", "Setup checks", "Coverage note", "e2e history"])
        self.assertIn("Kimi: not tested yet.", lines(tree, "d-llamacpp-k2-horizon-7b", "clients"))
        self.assertIn("OpenCode: not supported; llmctl writes no OpenCode entry for this pair.",
                      lines(tree, "d-vllm-deepseek-r1-0528-qwen3-8b", "clients"))
        self.assertIn("Setup checks in the e2e of 2026-11-19:", text(tree, "d-sglang-gpt-oss-20b"))
        # Dates never break at their hyphens on phones: each is one span.nw.
        for li in tree.by_id("d-llamacpp-qwen3.6-35b-a3b").find_all("ul", "clients")[0].find_all("li"):
            with self.subTest(line=squash(li)):
                self.assertEqual([node.text() for node in li.find_all("span", "nw")],
                                 re.findall(r"\d{4}-\d\d-\d\d", _text_without_lamps(li)))
        setup_line = next(p for p in tree.by_id("d-sglang-gpt-oss-20b").find_all("p")
                          if squash(p).startswith("Setup checks in the e2e of"))
        self.assertEqual([node.text() for node in setup_line.find_all("span", "nw")], ["2026-11-19"])
        section = tree.by_id("d-sglang-gpt-oss-20b")
        self.assertEqual([squash(li) for li in section.find_all("ul", "setup-checks")[0].find_all("li")],
                         ["up failed: the server did not start (no record named after this step)"])
        reached = next(li for li in section.find_all("ul", "clients")[0].find_all("li") if "Not reached" in li.text())
        self.assertEqual([node.text() for node in reached.find_all("code")], ["up"])
        self.assertEqual([(node.tag, node.attrs.get("class")) for node in reached.children if not isinstance(node, str)],
                         [("span", "lamp failed"), ("span", None)])          # one flex item for the sentence, as in v2
        for name, cls in (("hermes", "known-issue"), ("pi", "failed")):
            item = next(li for li in tree.by_id("d-llamacpp-qwen3.6-35b-a3b").find_all("ul", "clients")[0].find_all("li")
                        if _text_without_lamps(li).strip().startswith(clients.LABELS[name] + ":"))
            lamp = item.find_all("span", "lamp")[0]
            self.assertEqual((lamp.attrs["class"], lamp.text(), lamp.attrs["aria-hidden"]),
                             (f"lamp {cls}", "H" if name == "hermes" else "P", "true"))
        self.assertEqual(tree.by_id("d-llamacpp-qwen3.6-35b-a3b").find_all("ul", "setup-checks"), [])
        cell = ("gpt-oss-20b", "sglang")                                       # the note names the setup step: Known issue
        self.fixture.coverage({cell: self.fixture._default_cell(*cell) + "; e2e 2026-11-19: `up` failed on a busy GPU"})
        tree = parse(render.page(self.fixture.load(), BUILD, self.fixture.root))
        self.assertIn("OpenCode: known issue, 1 of 2 runs passed (latest 2026-11-19); the coverage note below describes "
                      "it. Not reached: up failed.", lines(tree, "d-sglang-gpt-oss-20b", "clients"))

    def test_details_of_failed_and_does_not_fit_cells(self):
        matrix = self.fixture.load()
        tree = parse(render.page(matrix, BUILD, self.fixture.root))
        hardware = "; ".join(sorted({cell.recipe.hardware for model in matrix.models for cell in model.cells
                                     if cell.status == "runs"}))
        coverage = BLOB + "/docs/model-selection.md#runtime-coverage"
        for section_id, title, status, links, current in (
                ("d-sglang-deepseek-r1-0528-qwen3-8b", "deepseek-r1-0528-qwen3-8b on SGLang",
                 f"Failed on 2026-11-17 (measured on {hardware}, SGLang v0.5.23).",
                 ["#d-llamacpp-deepseek-r1-0528-qwen3-8b", "#d-vllm-deepseek-r1-0528-qwen3-8b"], "SGLang Failed 2026-11-17"),
                ("d-vllm-k2-horizon-7b", "k2-horizon-7b on vLLM", "Does not fit (estimate).",
                 ["#d-llamacpp-k2-horizon-7b", "#d-sglang-k2-horizon-7b"], "vLLM Does not fit (estimate)")):
            with self.subTest(section=section_id):
                section = tree.by_id(section_id)
                model_id, runtime = section.attrs["data-model"], section.attrs["data-engine"]
                cell = next(c for c in next(m for m in matrix.models if m.id == model_id).cells if c.runtime == runtime)
                self.assertEqual(squash(section.find_all("h3")[0]), title)
                self.assertEqual(text(tree, section_id, "status-line"), status)
                self.assertEqual([squash(h4) for h4 in section.find_all("h4")], ["Why it does not run", "Engines"])
                self.assertEqual(len(section.find_all("p", "scale")), 1)
                note = section.find_all("p", "note")[0]
                html_note = render.inline(cell.note, "docs/model-selection.md", self.links)
                self.assertEqual(squash(note), squash(parse(html_note)))
                engines = section.find_all("ul", "engines")[0]
                self.assertEqual([a.attrs["href"] for a in engines.find_all("a")], links)
                self.assertEqual([squash(li) for li in engines.find_all("li", "cur")], [current])
                self.assertIn(coverage, [a.attrs["href"] for a in section.find_all("a")])
                self.assertEqual(section.find_all("button", "copy"), [])
                self.assertEqual(section.find_all("code", "cmd"), [])
                self.assertEqual(section.find_all("p", "recipe-name"), [])
        self.assertIn(BLOB + "/research/2026-11-17/fixture-failure/README.md",
                      [a.attrs["href"] for a in tree.by_id("d-sglang-deepseek-r1-0528-qwen3-8b").find_all("a")])
        # k2-horizon-7b with an estimate recipe on llama.cpp: no engine runs it (the kind line also adds the doc
        # that an estimate needs, since set_value writes one line per key)
        self.fixture.set_value("llamacpp", "k2-horizon-7b", "decode_tok_s", None)
        self.fixture.set_value("llamacpp", "k2-horizon-7b", "kind", '"estimate"\ndoc = "llamacpp/models.md"')
        k2 = ("k2-horizon-7b", "llamacpp")
        self.fixture.coverage({k2: "Does not fit: too large for the budget (estimate)"})
        tree = parse(render.page(self.fixture.load(), BUILD, self.fixture.root))
        self.assertIn("No engine runs this model on this laptop yet. Runtime coverage table", text(tree, "d-vllm-k2-horizon-7b"))
        qwen = ("qwen3.6-35b-a3b", "llamacpp")                     # a note outside the subset stops the build (D13)
        self.fixture.coverage({qwen: self.fixture._default_cell(*qwen) + "; a **bold** word"})
        column = cell_of(self.fixture.load(), *qwen).note.index("*") + 1
        with self.assertRaises(data.SiteError) as caught:
            render.page(self.fixture.load(), BUILD, self.fixture.root)
        self.assertEqual(str(caught.exception),
                         f"docs/model-selection.md: row qwen3.6-35b-a3b, llama.cpp, note: column {column}: emphasis")

    def test_commands_wrap_only_at_spaces_and_copy_the_plain_command(self):
        matrix = self.fixture.load()
        tree = parse(render.page(matrix, BUILD, self.fixture.root))
        facts = cell_of(matrix, "qwen3.6-35b-a3b", "sglang").recipe
        section = tree.by_id("d-sglang-qwen3.6-35b-a3b")
        self.assertEqual(squash(section.find_all("p", "recipe-name")[0]), facts.name)
        self.assertEqual([squash(node) for node in section.find_all("p", "step")], ["Download once", "Start"])
        self.assertEqual([squash(node) for node in section.find_all("ol", "steps")[0].find_all("p", "step")],
                         ["Download once", "Start"])
        commands = section.find_all("code", "cmd")
        buttons = section.find_all("button", "copy")
        self.assertEqual((len(commands), len(buttons)), (2, 2))
        for command, button, expected in zip(commands, buttons, (facts.pull, facts.up)):
            with self.subTest(command=expected):
                self.assertEqual([node.text() for node in command.find_all("span")], expected.split(" "))
                self.assertEqual(command.text(), expected)
                self.assertEqual((button.attrs["data-copy"], "hidden" in button.attrs, button.text()),
                                 (expected, True, "Copy"))
        self.assertEqual([line.find_all("code", "cmd")[0].text() for line in section.find_all("span", "cmdline")],
                         [facts.pull, facts.up])
        for line in section.find_all("span", "cmdline"):
            self.assertEqual([node.tag for node in line.children], ["code", "button"])
        body = text(tree, section.attrs["id"])
        for words in ("It asks before it downloads the model files and the image.",
                      "Add --replace to stop a running server first; ./llmctl up refuses to start when less GPU memory or "
                      "RAM is free.", "Without --runtime, ./llmctl up qwen3.6-35b-a3b starts the llama.cpp recipe."):
            with self.subTest(words=words):
                self.assertIn(words, body)
        self.assertNotIn("Without --runtime", text(tree, "d-llamacpp-qwen3.6-35b-a3b"))
        plain = next(p for p in section.find_all("p") if squash(p).startswith("Without --runtime"))
        self.assertEqual([node.text() for node in plain.find_all("code")[1].find_all("span", "nw")], ["qwen3.6-35b-a3b"])
        pre = section.find_all("p", "pre")[0]
        self.assertEqual(squash(pre), "Run both in a clone of this repository; llmctl needs Docker and the NVIDIA "
                                      "Container Toolkit: How to start a model, Host setup.")
        self.assertEqual([(squash(a), a.attrs["href"]) for a in pre.find_all("a")],
                         [("How to start a model", BLOB + "/README.md#quick-start"),
                          ("Host setup", BLOB + "/docs/host-setup.md")])

    def test_long_reasons_are_cut_before_escaping(self):
        qwen = prefix("llamacpp", "qwen3.6-35b-a3b")
        long_reason, exact = "x" * 118 + "<&" + "y" * 40, "z" * 119 + "&"
        self.fixture.run("2026-11-19", "llmctl-long-e2e", [row(qwen, "up"), row(qwen, "pi", long_reason),
                                                           row(qwen, "kimi", exact)])
        page = render.page(self.fixture.load(), BUILD, self.fixture.root)
        self.assertEqual(well_formed_problems(page), [])
        section = parse(page).by_id("d-llamacpp-qwen3.6-35b-a3b")
        reasons = section.find_all("details", "reason")
        self.assertEqual(len(reasons), 1)
        self.assertEqual(reasons[0].find_all("summary")[0].text(), "x" * 118 + "<&")
        self.assertIn("x" * 118 + "&lt;&amp;</summary>", page)
        self.assertEqual(reasons[0].find_all("p")[0].text(), long_reason)
        kimi = next(li for li in section.find_all("ul", "history")[0].find_all("li") if li.text().startswith("kimi failed"))
        self.assertEqual(squash(kimi), f"kimi failed: {exact} (no record named after this step)")
        self.assertEqual(kimi.find_all("details"), [])

    def test_e2e_history_lists_runs_by_date(self):
        qwen = prefix("llamacpp", "qwen3.6-35b-a3b")
        self.fixture.run("2026-11-17", "llmctl-a-e2e", default_rows(), readme=None, records=None)
        failing = {"pi": "pi went quiet", "probes": "probe A and B failed", "opencode": "opencode stopped"}
        rows = [row(qwen, line.split(" | ")[1], failing[line.split(" | ")[1]])
                if line.startswith(f"| {qwen} | ") and line.split(" | ")[1] in failing else line for line in default_rows()]
        names = ("012-01-qwen3.6-35b-a3b-llamacpp-pi", "020-01-qwen3.6-35b-a3b-llamacpp-probes-A",
                 "021-01-qwen3.6-35b-a3b-llamacpp-probes-B")
        self.fixture.run("2026-11-17", "llmctl-b-e2e", rows, readme="# e2e\n\n## The No rows\n\nThree.\n",
                         records={name: "{}\n" for name in names})
        steps = sum(1 for line in default_rows() if line.startswith(f"| {qwen} | ") and " | gateway | " not in line)
        tree = parse(render.page(self.fixture.load(), BUILD, self.fixture.root))
        history = tree.by_id("d-llamacpp-qwen3.6-35b-a3b").find_all("ul", "history")[0]
        days = [node for node in history.children if not isinstance(node, str)]
        self.assertEqual([node.find_all("span", "day")[0].text() for node in days], ["2026-11-18", "2026-11-17"])
        order = "Runs of one date are listed by folder name; the summaries do not record which ran first."
        self.assertEqual([order in squash(node) for node in days], [False, True])
        runs = [node for node in days[1].find_all("ul")[0].children if not isinstance(node, str)]
        self.assertEqual([squash(node).split(":")[0] for node in runs], ["llmctl-a-e2e", "llmctl-b-e2e"])
        base = "/research/2026-11-17/"
        a_links = [a.attrs["href"] for a in runs[0].find_all("a")]
        b_links = [a.attrs["href"] for a in runs[1].find_all("a")]
        self.assertTrue(squash(runs[0]).startswith(f"llmctl-a-e2e: {steps} of {steps} steps passed"), squash(runs[0]))
        self.assertTrue(squash(runs[1]).startswith(f"llmctl-b-e2e: {steps - 3} of {steps} steps passed, records"),
                        squash(runs[1]))
        self.assertEqual(a_links, [BLOB + base + "llmctl-a-e2e/summary.md"])
        self.assertEqual(b_links[:2], [BLOB + base + "llmctl-b-e2e/README.md#the-no-rows", TREE + base + "llmctl-b-e2e/records"])
        failures = runs[1].find_all("ul")[0].find_all("li")
        self.assertEqual([squash(li) for li in failures],
                         ["opencode failed: opencode stopped (no record named after this step)",
                          "pi failed: pi went quiet (record 012)",
                          "probes failed: probe A and B failed (records 020, 021)"])
        self.assertEqual([[code.text() for code in li.find_all("code")] for li in failures],
                         [["opencode"], ["pi"], ["probes"]])
        self.assertEqual([[(squash(a), a.attrs["href"]) for a in li.find_all("a")] for li in failures],
                         [[], [("012", BLOB + base + "llmctl-b-e2e/records/" + names[0])],
                          [("020", BLOB + base + "llmctl-b-e2e/records/" + names[1]),
                           ("021", BLOB + base + "llmctl-b-e2e/records/" + names[2])]])
        self.assertEqual(runs[0].find_all("ul"), [])

    def test_runs_details_show_absent_facts_as_not_recorded(self):
        for runtime in ("llamacpp", "vllm"):
            self.fixture.set_value(runtime, "gpt-oss-20b", "ram_gib", None)            # absent, pinned
            self.fixture.set_value(runtime, "gpt-oss-20b", "output", "16384")
        self.fixture.set_value("llamacpp", "qwen3.6-35b-a3b", "ram_gib", "14.0")
        matrix = self.fixture.load()
        tree = parse(render.page(matrix, BUILD, self.fixture.root))

        def kv(section_id):
            found = tree.by_id(section_id).find_all("dl", "kv")[0]
            return {squash(div.find_all("dt")[0]): squash(div.find_all("dd")[0]) for div in found.find_all("div")}

        def settings(section_id):
            found = tree.by_id(section_id).find_all("dl", "settings")[0]
            return dict(zip([squash(dt) for dt in found.find_all("dt")], [squash(dd) for dd in found.find_all("dd")]))

        for runtime, context in (("llamacpp", "128k 131,072 tokens"), ("vllm", "64k 65,536 tokens")):
            with self.subTest(runtime=runtime):
                facts, found = cell_of(matrix, "gpt-oss-20b", runtime).recipe, kv(f"d-{runtime}-gpt-oss-20b")
                self.assertEqual(found["Free RAM to start"], f"{facts.memory_cap} GiB peak not recorded")
                self.assertEqual(found["Context"], context)
                self.assertEqual(settings(f"d-{runtime}-gpt-oss-20b")["Output cap"], "16,384 tokens")
        gpu = cell_of(matrix, "gpt-oss-20b", "vllm").recipe.gpu
        self.assertTrue(gpu.endswith("%"))
        self.assertEqual(kv("d-vllm-gpt-oss-20b")["GPU memory needed"], f"{gpu} of what CUDA reports")
        self.assertEqual(kv("d-llamacpp-gpt-oss-20b")["GPU memory needed"], cell_of(matrix, "gpt-oss-20b", "llamacpp").recipe.gpu)
        facts = cell_of(matrix, "qwen3.6-35b-a3b", "llamacpp").recipe
        found = kv("d-llamacpp-qwen3.6-35b-a3b")
        self.assertEqual(list(found), ["Decode speed", "Context", "GPU memory needed", "Free RAM to start"])
        self.assertEqual((found["Free RAM to start"], found["Decode speed"]), ("20 GiB measured peak 14 GiB", "122.5 tok/s"))
        small = [squash(node) for node in tree.by_id("d-llamacpp-qwen3.6-35b-a3b").find_all("dl", "kv")[0].find_all("small")]
        self.assertEqual(small, ["131,072 tokens"] + (["of what CUDA reports"] if facts.gpu.endswith("%") else [])
                         + ["measured peak 14 GiB"])
        self.assertEqual(text(tree, "d-llamacpp-qwen3.6-35b-a3b", "ev"),
                         f"(measured on {facts.hardware}, llama.cpp b12407, {facts.date})")
        evidence = tree.by_id("d-llamacpp-qwen3.6-35b-a3b").find_all("p", "ev")[0]
        self.assertEqual([squash(node) for node in evidence.find_all("span", "nw")], [facts.date])
        self.assertEqual({key: value for key, value in settings("d-llamacpp-qwen3.6-35b-a3b").items() if key != "Thinking"},
                         {"Model files": f"{facts.repo} at {facts.revision[:12]}, {facts.size_bytes / 1e9:.1f} GB",
                          "Aliases": "qwen3.6", "Tools": "Yes", "Output cap": f"{facts.output:,} tokens",
                          "OpenCode model": facts.opencode, "Served name": facts.served})
        self.assertEqual(list(settings("d-llamacpp-qwen3.6-35b-a3b")),
                         ["Model files", "Aliases", "Thinking", "Tools", "Output cap", "OpenCode model", "Served name"])
        deepseek = settings("d-vllm-deepseek-r1-0528-qwen3-8b")
        self.assertEqual((deepseek["Tools"], deepseek["OpenCode model"]), ("No, chat only", "No OpenCode entry"))
        section = tree.by_id("d-llamacpp-qwen3.6-35b-a3b")
        self.assertEqual([squash(node.find_all("summary")[0]) for node in section.find_all("details", "more")],
                         ["Recipe settings", "Evidence and links"])
        self.assertEqual(["open" in node.attrs for node in section.find_all("details", "more")], [False, False])
        evidence = section.find_all("details", "more")[1]
        self.assertIn(facts.summary, squash(evidence))
        links = {squash(a): a.attrs["href"] for a in evidence.find_all("ul", "links")[0].find_all("a")}
        doc, anchor = facts.doc.split("#")
        self.assertEqual(links, {f"Recipe section in {doc}": f"{BLOB}/{facts.doc}", "Recipe file": f"{BLOB}/{facts.file}",
                                 "Record folder": f"{TREE}/{facts.record.rstrip('/')}",
                                 "Runtime coverage table": BLOB + "/docs/model-selection.md#runtime-coverage"})
        files = next(a for a in section.find_all("dl", "settings")[0].find_all("a")
                     if a.attrs["href"].startswith("https://huggingface.co/"))
        self.assertEqual(files.attrs["href"], f"https://huggingface.co/{facts.repo}/tree/{facts.revision}")
        self.assertEqual([squash(node) for node in section.find_all("h4")],
                         ["Start it", "Engines", "Clients", "Coverage note", "e2e history"])
        self.assertEqual(squash(section.find_all("a")[-1]), "Back to the table")
        k2 = prefix("llamacpp", "k2-horizon-7b")                                  # no e2e run covered the pair
        self.rewrite_default_run([line for line in default_rows() if not line.startswith(f"| {k2} | ")])
        self.fixture.set_value("llamacpp", "k2-horizon-7b", "aliases", None)
        tree = parse(render.page(self.fixture.load(), BUILD, self.fixture.root))
        self.assertEqual(settings("d-llamacpp-k2-horizon-7b")["Aliases"], "no alias")
        self.assertIn("e2e history No e2e run has covered this pair yet.", text(tree, "d-llamacpp-k2-horizon-7b"))
        self.assertEqual(tree.by_id("d-llamacpp-k2-horizon-7b").find_all("ul", "history"), [])
        self.fixture.set_value("llamacpp", "k2-horizon-7b", "doc", '"llamacpp/models.md"')      # a doc without #anchor
        with self.assertRaises(data.SiteError) as caught:
            render.page(self.fixture.load(), BUILD, self.fixture.root)
        self.assertEqual(str(caught.exception),
                         "recipes/llamacpp/k2-horizon-7b.toml: doc llamacpp/models.md names no section anchor")


def squash(node) -> str:
    return " ".join(node.text().split())


def _kept(node, keep) -> str:
    """The text of the node and its descendants that `keep` admits; a node it refuses drops its whole subtree."""
    if not keep(node):
        return ""
    return "".join(child if isinstance(child, str) else _kept(child, keep) for child in node.children)


def _classes(node) -> set[str]:
    return set((node.attrs.get("class") or "").split())


def visible(node) -> str:
    """The squashed text a sighted reader sees: without `.vh` and `[hidden]` elements."""
    return " ".join(_kept(node, lambda n: "vh" not in _classes(n) and "hidden" not in n.attrs).split())


def spoken(node, drop=()) -> str:
    """The squashed text a screen reader reads: without aria-hidden subtrees, `[hidden]` elements and the classes in
    `drop`."""
    return " ".join(_kept(node, lambda n: n.attrs.get("aria-hidden") != "true" and "hidden" not in n.attrs
                          and not _classes(n) & set(drop)).split())


def tile_name(tile, drop=("sz", "dm")) -> str:
    """A tile's accessible name: the spoken text of each child of its span.g, joined with spaces and squashed."""
    g = tile.find_all("span", "g")[0]
    return " ".join(" ".join(spoken(child, drop) for child in g.children if not isinstance(child, str)).split())


def _text_without_lamps(node) -> str:
    return "".join(child if isinstance(child, str) else
                   "" if "lamp" in (child.attrs.get("class") or "").split() else _text_without_lamps(child)
                   for child in node.children)


def text(tree, section_id: str, css: str | None = None) -> str:
    """The squashed text of the first element with class `css` inside the section, or of the whole section."""
    section = tree.by_id(section_id)
    node = section if css is None else section.find_all(cls=css)[0]
    return " ".join(node.text().split())


def lines(tree, section_id: str, css: str) -> list[str]:
    """The squashed li texts of the first list with class `css` inside the section, without the lamps' letters."""
    found = tree.by_id(section_id).find_all(cls=css)[0]
    return [" ".join(_text_without_lamps(li).split()) for li in found.find_all("li")]


def tile_cell(tree, model_id, runtime):
    return next(td for td in tree.by_id("m-" + model_id).find_all("td") if td.attrs["data-engine"] == runtime)


if __name__ == "__main__":
    unittest.main()
