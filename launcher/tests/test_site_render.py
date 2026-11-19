"""The model guide's renderer: the Markdown subset of the coverage notes, repository links at the build's commit
(design D12), the README recipe table (D-README), the shell every page shares and the CLI's site tree (spec 8 and 10)."""

import ast
import base64
import contextlib
import hashlib
import io
import os
import posixpath
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from launcher.runtimes import ADAPTERS
from launcher.site import data, draw, fit, render, words
from launcher.tests.fakes import REPO
from launcher.tests.site_fixture import (DEFAULT_RUN, RECIPES, Node, SiteFixture, cell_of, default_rows, parse,
                                         prefix, py311_fraction_format, row, well_formed_problems)

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

    def pages(self, matrix=None) -> dict[str, str]:
        return render.site(matrix or self.fixture.load(), BUILD, self.fixture.root)

    def build(self, out: Path, commit: str = "a" * 40, time: str = "0") -> tuple[int, str, str]:
        from launcher.site import cli
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = cli.main(["build", "--out", str(out), "--commit", commit, "--commit-time", time],
                            root=self.fixture.root)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_every_page_is_well_formed(self):
        for path, page in self.pages().items():
            with self.subTest(page=path):
                self.assertEqual(well_formed_problems(page), [])

    def test_pages_load_nothing_from_other_sites(self):
        for path, page in self.pages().items():
            tree = parse(page)
            with self.subTest(page=path):
                self.assertNotIn("src=", page)
                self.assertEqual(len(re.findall(r"<link\b[^>]*>", page)), 1)
                self.assertRegex(re.search(r"<link\b[^>]*>", page).group(0), r'^<link rel="icon" href="data:image/svg\+xml,')
                self.assertNotIn("@import", page)
                self.assertEqual([u for u in re.findall(r"url\(([^)]*)\)", page) if not u.startswith(("#", "data:"))], [])
                for word in ("fetch(", "XMLHttpRequest", "sendBeacon", "import("):
                    self.assertNotIn(word, page)
                for node in tree.iter():
                    self.assertNotIn("style", node.attrs)
                    self.assertEqual([name for name in node.attrs if name.startswith("on")], [])
                self.assertEqual(tree.find_all("form"), [])
                policy_at = page.index('http-equiv="Content-Security-Policy"')
                self.assertEqual([t for t in ("<script", "<style", "<link") if t in page and page.index(t) < policy_at], [])
                policy = next(n.attrs["content"] for n in tree.find_all("meta") if n.attrs.get("http-equiv") == "Content-Security-Policy")
                scripts = [n.text() for n in tree.find_all("script") if n.attrs.get("type") != "application/json"]
                self.assertEqual(len(scripts), 2)              # the head script and page.js (spec 8.1)
                style = tree.find_all("style")[0].text()
                head = f"script-src {' '.join(sha(s) for s in scripts)}; " if scripts else ""
                self.assertEqual(policy, f"default-src 'none'; {head}style-src {sha(style)}; img-src data:; base-uri 'none'; "
                                         "form-action 'none'")

    def test_pages_follow_the_copy_rules(self):
        for path, page in self.pages().items():
            visible = visible_text(page)
            with self.subTest(page=path):
                for text in ("\u2013", "\u2014", "\u00b7", "N/A", "Copy link"):
                    self.assertNotIn(text, page)
                self.assertIsNone(re.search("[\u2190-\u21ff\u2600-\u27bf\u2b00-\u2bff\U0001f000-\U0001faff\ufe0f]", page))
                self.assertIsNone(re.search(r"\bNone\b(?! yet\b)", visible))     # "None yet" is the agent label (H3)
                self.assertIsNone(re.search(r"\b[0-9a-f]{40}\b", visible))
                for text in (str(self.fixture.root), str(Path.home()), socket.gethostname()):
                    self.assertNotIn(text, page)

    def test_footer(self):
        footer = parse(self.pages()["index.html"]).find_all("footer")[0]
        self.assertEqual([squash_text(p) for p in footer.find_all("p")],
                         ["llmctl and this page come from r on GitHub. Updated 1 January 1970.",
                          "Docs and measurements: CC BY 4.0. Code: MIT.",
                          "This page sets no cookies, has no analytics and loads nothing from other sites."])
        self.assertEqual([a.attrs["href"] for a in footer.find_all("a")], ["https://github.com/o/r", "https://github.com/o/r/commit/" + "a" * 40])

    def test_home_blocks_in_order(self):
        main = parse(self.pages()["index.html"]).by_id("main")
        self.assertEqual([n.attrs.get("id") or n.attrs.get("class") for n in main.children if isinstance(n, Node) and n.tag == "section"],
                         ["fh", "models", "more-models", "get-started", "compare", "how-it-works", "questions", "how-we-tested"])

    def test_home_skip_link_nav_and_script_only_controls(self):
        tree = parse(self.pages()["index.html"])
        skip = tree.find_all("a")[0]
        self.assertEqual((skip.attrs.get("class"), skip.attrs["href"], squash_text(skip)),
                         ("skip", "#models", "Skip to the models"))
        self.assertEqual([(squash_text(a), a.attrs["href"]) for a in tree.find_all("header")[0].find_all("li")
                          for a in a.find_all("a")],
                         [("Models", "#models"), ("Get started", "#get-started"), ("How we tested", "#how-we-tested"),
                          ("GitHub", "https://github.com/o/r")])
        self.assertEqual([li.attrs.get("class") for li in tree.find_all("header")[0].find_all("li")],
                         [None, "wide", "wide", None])        # spec 3.2: the middle two hide on phones
        sortbox = tree.by_id("models").find_all("div", "sortbox")[0]
        self.assertIn("hidden", sortbox.attrs)                   # spec 3.4: the sort is the script's
        self.assertIn("data-js-only", sortbox.attrs)
        css = (REPO / "launcher/site/page.css").read_text(encoding="utf-8")
        self.assertRegex(css, r"(?m)^html:not\(\.js\) \.inputs\b")  # spec 3.4: no inputs without script

    def test_hero_shows_the_laptop_answer(self):
        tree = parse(self.pages()["index.html"])
        answer = tree.by_id("answer")
        self.assertEqual(squash_text(answer.find_all("p", "fh-verdict")[0]), "On our test laptop: all 6 models run as we measured them.")
        self.assertEqual(squash_text(answer.find_all("h2")[0]), "Start with Qwen3.6-35B-A3B")
        self.assertEqual(squash_text(answer.find_all("p", "reason")[0]), "Our default, and the model our OpenCode setup starts with.")
        self.assertEqual(squash_text(answer.find_all("p", "hagents")[0]),
                         "Passed the agent test in OpenCode (1 of 1 run). Also works in Hermes, Pi, dsh and Kimi.")
        notes = [[squash_text(s) for s in p.find_all("span") if "ch-r" in s.attrs.get("class", "") or
                  s.attrs.get("class") in ("gl", "rl")] for p in answer.find_all("p", "ch-notes")]
        self.assertEqual(notes, [["14.7 GB used of our 16 GB", "about 1 GB for the desktop (estimate)"],
                                 ["14.9 GB used at peak. llmctl asks for 20 GB free to start it.",
                                  "about 22 GB in use by the desktop and apps (estimate)"]])
        self.assertEqual(len(answer.find_all("i", "hb")), 16)
        self.assertEqual(squash_text(answer.find_all("summary")[0]), "Why does a 20.5 GB download fit in 14.7 GB of GPU memory?")
        self.assertEqual(answer.find_all("a", "btn")[0].attrs["href"], "#get-started")
        self.assertIsNone(re.search(r"\byour\b", squash_text(answer), re.I))
        self.assertIsNone(re.search(r"\bsetups\b", squash_text(tree.find_all("section", "fh")[0])))

    def test_memory_map_keeps_its_controls_out_of_the_image(self):
        # the lead's ruling on Task 5's deferred minor: role=img makes its children presentational, so the
        # mixture-of-experts question and the key sit outside the element that carries the M7 name
        matrix = self.fixture.load()
        b = fit.bench(matrix)
        pick = fit.hero_pick(b, b.home, fit.DEFAULT_AGENT)
        alt = words.map_text(words.labels(matrix), pick.setup, pick.verdict, b.home, True, False).alt
        figure = parse(self.pages(matrix)["index.html"]).by_id("answer").find_all("figure", "hmap")[0]
        self.assertNotIn("role", figure.attrs)
        self.assertNotIn("aria-label", figure.attrs)
        images = [n for n in figure.iter() if n.attrs.get("role") == "img"]
        self.assertEqual(len(images), 1)
        image = images[0]
        self.assertEqual(image.attrs["aria-label"], alt)
        self.assertEqual([n.attrs.get("class") for n in image.children if isinstance(n, Node)], ["ch", "ch"])
        self.assertEqual([n.tag for n in image.iter() if n.tag in ("a", "summary", "button", "input", "select", "details")
                          or "tabindex" in n.attrs], [])
        self.assertEqual(len(figure.find_all("summary")), 1)
        self.assertEqual(len(figure.find_all("figcaption", "hkey")), 1)

    def test_compare_engines_lists_every_combination(self):
        tree = parse(self.pages()["index.html"])
        table = tree.by_id("compare")
        self.assertEqual(squash_text(table.find_all("p", "sec-lead")[0]), "Every model on every engine we tried. 10 of the 18 "
                         "combinations run on our 16 GB laptop GPU; the others don't fit (our arithmetic) or didn't start (we tried).")
        self.assertEqual(len(table.find_all("td")), 18)
        links = [a.attrs["href"] for a in table.find_all("a", "cell")]
        self.assertEqual(len(links), 10)
        self.assertIn("models/qwen3.6-35b-a3b/#engine-sglang", links)

        def cell(model, i):
            row = next(r for r in table.find_all("tr") if r.find_all("th") and r.find_all("th")[0].find_all("a")
                       and r.find_all("th")[0].find_all("a")[0].attrs["href"] == f"models/{model}/")
            return row.find_all("td")[i]
        failed = cell("deepseek-r1-0528-qwen3-8b", 2)
        self.assertIn("Didn't start", squash_text(failed))
        self.assertEqual(squash_text(failed.find_all("span", "t2")[0]), "tried 17 Nov 2026")
        fast = cell("qwen3.6-35b-a3b", 2)
        self.assertEqual([squash_text(fast.find_all("span", c)[0]) for c in ("sp", "ags", "lead2")],
                         ["151.3 tok/s, 128k", "Passed the agent test in OpenCode (1 of 1 run).",
                          "Fastest for this model, 24% ahead"])
        self.assertEqual([squash_text(th) for th in table.find_all("thead")[0].find_all("th")],
                         ["Model", "llama.cpp b12407, the default engine", "vLLM 0.33.0", "SGLang 0.5.23"])
        self.assertEqual(len(table.find_all("span", "strip")), 10)
        # C3: a cell that doesn't run reads "<engine>: <what> <why>" with an outline or a hatched swatch
        self.assertEqual(squash_text(failed), "SGLang: Didn't start tried 17 Nov 2026")
        self.assertEqual(failed.find_all("i", "sw")[0].attrs["class"], "sw o")
        too_big = cell("qwen3.6-35b-a3b", 1)
        self.assertEqual(squash_text(too_big), "vLLM: Doesn't fit on 16 GB (our arithmetic)")
        self.assertEqual(too_big.find_all("i", "sw")[0].attrs["class"], "sw h")
        # the ranked rows with their rank and Chat only tag, the group row, then the unranked rows
        self.assertEqual([squash_text(r.find_all("th")[0]) for r in table.find_all("tbody")[0].find_all("tr")], [
            "Qwen3.6-35B-A3BRanked 1", "gpt-oss-20bRanked 2", "DeepSeek-R1-0528-Qwen3-8B Chat onlyRanked 3",
            "Not in our Top 3", "Qwen3-Coder-Next", "K2 Horizon 7B", "Mellum2.1 12B-A2.5B Thinking"])
        self.assertEqual([squash_text(s) for s in table.find_all("span", "tag")], ["Chat only"])
        self.assertEqual([squash_text(s) for s in table.find_all("span", "rk")], ["Ranked 1", "Ranked 2", "Ranked 3"])
        self.assertEqual(table.find_all("tr", "grp")[0].find_all("th")[0].attrs["scope"], "rowgroup")
        # a Runs cell draws its strip on our laptop, at the bench's scale
        matrix = self.fixture.load()
        b = fit.bench(matrix)
        setup = next(s for s in b.setups if s.key == ("qwen3.6-35b-a3b", "sglang"))
        page = self.pages(matrix)["index.html"]
        start = page.index('href="models/qwen3.6-35b-a3b/#engine-sglang"')
        self.assertIn(draw.strip_html(draw.strip(setup, b.home, b.test_gpu, draw.scale(b, b.home))),
                      page[start:page.index("</a>", start)])

    def test_how_it_works_questions_and_how_we_tested(self):
        tree = parse(self.pages()["index.html"])
        self.assertEqual([squash_text(s) for s in tree.by_id("questions").find_all("summary")], [
            "Does it work on Windows or a Mac?", "Will I get the same speed?", "What does \u201cpassed the agent test\u201d mean?",
            "How were the models ranked?", "Why not just use Ollama?", "What do tok/s, context and thinking mean?"])
        tested = [squash_text(p) for p in tree.by_id("how-we-tested").find_all("p")]
        self.assertTrue(tested[0].startswith("Every number on this page comes from one laptop: an MSI Raider 18 Max HX A2WI "
                                             "with an RTX 5080 Laptop GPU (16 GB) and 61 GB of RAM, measured between "))
        self.assertTrue(tested[0].endswith("It ran Ubuntu 26.04.1, with the GPU at its 140 W power limit."))
        self.assertIn("the page assumes about 1 GB of GPU memory goes to your desktop and that the same share of your RAM is "
                      "free as on our laptop (40 of 61 GB).", tested[3])
        self.assertEqual(tested[4], "Engines: llama.cpp b12407, vLLM 0.33.0 and SGLang 0.5.23. Agents: OpenCode 2.3.0, "
                         "Hermes 0.23.1, Pi 1.5.2, dsh 0.2.2-rc.3 and Kimi 2.5.3.")
        self.assertEqual(len(tree.by_id("how-it-works").find_all("svg", "ram-bar")), 1)

    def test_how_it_works_and_the_answers_follow_words(self):
        matrix = self.fixture.load()
        b, labels = fit.bench(matrix), words.labels(matrix)
        page = self.pages(matrix)["index.html"]
        tree = parse(page)
        flat = lambda rich: " ".join(words.plain(rich).split())
        # spec 7.9: the Ollama answer's contexts are the smallest and the largest of every Runs cell
        contexts = [cell.recipe.context for model in matrix.models for cell in model.cells if cell.status == "runs"]
        self.assertEqual((min(contexts), max(contexts)), (64 * 1024, 128 * 1024))
        answers = [squash_text(d.find_all("div")[0]) for d in tree.by_id("questions").find_all("details")]
        self.assertEqual(answers, [flat(answer) for _, answer in words.questions(labels, contexts)])
        self.assertIn("with 64k to 128k of context", answers[4])
        # spec 7.8: the prose, the five rules and the guide's link
        how = tree.by_id("how-it-works")
        self.assertEqual([squash_text(p) for p in how.find_all("p")],
                         [flat(words.HOW_PROSE), words.RULES_INTRO])
        self.assertEqual([squash_text(li) for li in how.find_all("ul", "rules")[0].find_all("li")], list(words.RULES))
        self.assertEqual(len(words.RULES), 5)
        self.assertEqual([(squash_text(a), a.attrs["href"]) for a in how.find_all("ul", "links")[0].find_all("a")],
                         [("Everything llmctl does, command by command", BLOB + "/docs/llmctl.md")])
        # the mini map is our default on our laptop, with the RAM bar at the diagram's 10 px
        default = next(setup for setup in b.setups if setup.key == b.default)
        mini = page[page.index('<span class="mini"'):]
        mini = mini[:mini.index("</svg>")]
        self.assertIn(draw.strip_html(draw.strip(default, b.home, b.test_gpu, draw.scale(b, b.home))), mini)
        self.assertIn(draw.ram_svg(draw.ram_bar(default, b.home), "mini"), mini + "</svg>")
        self.assertEqual(how.find_all("svg", "ram-bar")[0].attrs["height"], "10")
        # spec 7.10: the four links under How we tested
        self.assertEqual([(squash_text(a), a.attrs["href"]) for a in tree.by_id("how-we-tested").find_all("ul", "links")[0]
                          .find_all("a")], [("How we ranked the models", BLOB + "/docs/model-selection.md#decision-summary"),
                                            ("Measurement records", TREE + "/research"),
                                            ("llmctl's guide", BLOB + "/docs/llmctl.md"),
                                            ("The knowledge base on GitHub", "https://github.com/o/r")])

    def test_how_we_tested_names_the_latest_run(self):
        # spec 7.10: the latest run by date, then folder; the two older runs hold fewer steps than the latest
        self.fixture.run("2026-11-17", "llmctl-z-e2e", default_rows()[:3])
        self.fixture.run(DEFAULT_RUN[0], "llmctl-a-e2e", default_rows()[:2])
        matrix = self.fixture.load()
        self.assertEqual([run.folder for run in matrix.runs], ["llmctl-z-e2e", "llmctl-a-e2e", DEFAULT_RUN[1]])
        tested = [squash_text(p) for p in parse(self.pages(matrix)["index.html"]).by_id("how-we-tested").find_all("p")]
        rows = len(default_rows())
        self.assertIn(f"In the latest run, on 18 November 2026, {rows} of {rows} steps passed across 10 setups", tested[2])

    def test_shelf_rows_follow_our_ranking(self):
        tree = parse(self.pages()["index.html"])
        rows = lambda sid: [li.attrs["data-model"] for li in tree.by_id(sid).find_all("li", "row")]
        self.assertEqual(rows("models"), ["qwen3.6-35b-a3b", "gpt-oss-20b", "deepseek-r1-0528-qwen3-8b"])
        self.assertEqual(rows("more-models"), ["qwen3-coder-next", "k2-horizon-7b", "mellum2.1-12b"])
        row = tree.by_id("models").find_all("li", "row")[0]
        self.assertEqual(row.find_all("a")[0].attrs["href"], "models/qwen3.6-35b-a3b/")
        self.assertEqual(squash_text(row.find_all("h3")[0]), "Qwen3.6-35B-A3B")
        self.assertEqual([squash_text(t) for t in row.find_all("span", "tag")], ["Our default", "Start here"])
        self.assertEqual([squash_text(b) for b in row.find_all("b")], ["122.5", "128k", "20.5"])
        self.assertEqual([squash_text(row.find_all("span", c)[0]) for c in ("galt", "ag", "vd")],
                         ["Uses 14.7 GB of GPU memory.", "Passed (1 of 1 run)", "Runs as measured"])
        self.assertEqual([squash_text(v) for v in row.find_all("span", "vh") if squash_text(v).endswith(":")],
                         ["In OpenCode:", "On our laptop:"])        # S4: the hidden heads before the agents cell and verdict
        self.assertEqual(len(row.find_all("i", "b")), 16)

    def test_get_started_shows_the_picks_commands(self):
        tree = parse(self.pages()["index.html"])
        self.assertEqual(len(tree.find_all("div", "term")), 1)
        self.assertEqual(terminal_lines(tree.by_id("get-started")), [
            "# Get llmctl (once)", "$ git clone https://github.com/o/r", "$ cd r",
            "# Download 20.5 GB of model files; the llama.cpp engine (7.05 GB on disk) comes too, the first time",
            "$ ./llmctl pull qwen3.6-35b-a3b",
            "# Start it: llmctl checks your machine first, then points your agents at it", "$ ./llmctl up qwen3.6-35b-a3b",
            "# Open OpenCode and pick llamacpp-docker/qwen3.6-35b-a3b with /models", "$ opencode",
            "# Check: a reply, a tool call and your speed (ours: 122.5 tok/s)", "$ ./llmctl test",
            "# Stop it when you're done, so your GPU can rest", "$ ./llmctl down"])
        self.assertEqual(squash_text(tree.by_id("get-started").find_all("p", "ep")[0]),
                         "Other tools: http://127.0.0.1:8080/v1 with the model name local.")

    def test_site_writes_a_page_per_model(self):
        matrix = self.fixture.load()
        self.assertEqual(list(render.site(matrix, BUILD, self.fixture.root)),
                         ["index.html", *(f"models/{m.id}/index.html" for m in matrix.models)])

    def test_every_internal_link_resolves(self):
        pages = self.pages()
        ids = {path: {n.attrs["id"] for n in parse(page).iter() if "id" in n.attrs} for path, page in pages.items()}
        for path, page in pages.items():
            for node in parse(page).find_all("a"):
                href = node.attrs.get("href", "")
                if href.startswith("https://"):
                    continue
                part, _, anchor = href.partition("#")
                with self.subTest(page=path, href=href):
                    if part:
                        self.assertTrue(part.endswith("/"), href)
                        target = posixpath.normpath(posixpath.join(posixpath.dirname(path), part))
                        file = "index.html" if target == "." else f"{target}/index.html"
                    else:
                        file = path
                    self.assertIn(file, pages)
                    if anchor:
                        self.assertIn(anchor, ids[file])

    def test_cli_refuses_a_stale_model_page(self):
        out = Path(self._tmp.name) / "out"
        (out / "models" / "gone").mkdir(parents=True)
        (out / "models" / "gone" / "index.html").write_text("old\n", encoding="utf-8")
        self.assertEqual(self.build(out), (1, "", f"site: --out {out}: the folder holds files the build does not write: "
                                                  "models/gone\n"))
        shutil.rmtree(out / "models" / "gone")
        elsewhere = Path(self._tmp.name) / "elsewhere"
        elsewhere.mkdir()
        (out / "models").rmdir()
        for kind, make in (("a file", lambda p: p.write_text("x\n", encoding="utf-8")), ("a link", lambda p: p.symlink_to(elsewhere))):
            with self.subTest(models=kind):
                make(out / "models")
                self.assertEqual(self.build(out), (1, "", f"site: --out {out}: {out / 'models'} is not a folder\n"))
                self.assertEqual(list(elsewhere.iterdir()), [])            # nothing written through the link
                (out / "models").unlink()
        code, stdout, _ = self.build(out)
        self.assertEqual(code, 0)
        self.assertRegex(stdout, rf"^wrote {re.escape(str(out))}: 7 pages, ")
        self.assertTrue((out / "models" / "qwen3.6-35b-a3b" / "index.html").is_file())
        self.assertEqual((out / "models" / "qwen3.6-35b-a3b" / "index.html").stat().st_mode & 0o7777, 0o644)

    def test_every_page_has_one_terminal(self):
        for path, page in self.pages().items():
            with self.subTest(page=path):
                self.assertEqual(len(parse(page).find_all("div", "term")), 1)

    def test_in_page_links_of_a_model_page_keep_the_address(self):
        for path, page in self.pages().items():
            if path.startswith("models/"):
                jumps = [a for a in parse(page).find_all("a") if a.attrs.get("href", "").startswith("#")]
                with self.subTest(page=path):
                    self.assertEqual(sorted(a.attrs["href"] for a in jumps), ["#model", "#run"])
                    self.assertTrue(all("data-scroll" in a.attrs for a in jumps))

    def test_model_page_texts(self):
        tree = parse(self.pages()["models/qwen3.6-35b-a3b/index.html"])
        self.assertEqual(tree.find_all("title")[0].text(), "Qwen3.6-35B-A3B, llmctl")
        self.assertEqual(squash_text(tree.by_id("model")), "Qwen3.6-35B-A3B")
        self.assertEqual(squash_text(tree.find_all("p", "role")[0]), "Ranked 1 of 3 in our review, and our default.")
        self.assertEqual(squash_text(tree.find_all("p", "maker")[0]), "Vendor 1, Apache-2.0, 35B / 3B active MoE.")
        self.assertTrue(squash_text(tree.find_all("p", "summary")[0]).startswith(
            "A mixture-of-experts model: 35B parameters, 3B of them active for each token. On our 16 GB laptop GPU it writes "
            "122.5 tokens a second and keeps 128k of context in view. It uses 14.7 GB of GPU memory and keeps part of itself "
            "in RAM, 14.9 GB at peak."))
        panel = tree.by_id("your-machine")
        self.assertEqual(squash_text(panel.find_all("h2")[0]), "Our test laptop")
        verdict = panel.find_all("p", "verdict")[0]
        self.assertEqual([squash_text(verdict.find_all("span", c)[0]) for c in ("on", "vt")],
                         ["Qwen3.6-35B-A3B on llama.cpp, on our test laptop:", "Runs as measured."])
        engines = tree.by_id("engines")
        self.assertEqual([n.attrs["id"] for n in engines.iter() if n.attrs.get("id", "").startswith("engine-")],
                         ["engine-llamacpp", "engine-vllm", "engine-sglang"])
        self.assertIn("Doesn't fit on 16 GB with vLLM (our arithmetic, not a test run).", squash_text(engines))
        why = engines.find_all("details", "why")[0]
        self.assertEqual(squash_text(why.find_all("p")[0]), "Too large for the budget (estimate). Engine notes")
        self.assertEqual(why.find_all("a")[0].attrs["href"], "https://github.com/o/r/blob/" + "a" * 40 + "/vllm/models.md")
        good = squash_text(tree.by_id("good-to-know"))
        self.assertIn("24% faster on SGLang (151.3 tok/s), but llmctl asks for 27 GB of free RAM and SGLang's engine is a "
                      "49.7 GB image.", good)
        self.assertIn("Other tools can use it like OpenAI's API: http://127.0.0.1:8080/v1 with the model name local.", good)
        self.assertIn("Other tools can use it like OpenAI's API: http://127.0.0.1:30000/v1 with the model name "
                      "qwen3.6-35b-a3b.", good)                          # SGLang serves the recipe's served_name
        agents = squash_text(tree.by_id("agents"))
        for command in ("pi --model llmctl/local", "kimi -m llmctl/local", "pi --model llmctl/qwen3.6-35b-a3b",
                        "kimi -m llmctl/qwen3.6-35b-a3b", "hermes -p llmctl"):
            self.assertIn(command, agents)                               # M16, from RecipeFacts.launch
        self.assertEqual(len(tree.find_all("div", "term")), 1)
        self.assertEqual([squash_text(h) for h in tree.find_all("p", "per-h")][:2], ["On llama.cpp", "On SGLang"])
        pager = tree.find_all("nav", "pager")[0]
        self.assertEqual([a.attrs["href"] for a in pager.find_all("a")], ["../gpt-oss-20b/"])

    def test_model_page_holds_no_your_before_input(self):
        for path, page in self.pages().items():
            if path.startswith("models/"):
                panel = parse(page).by_id("your-machine")
                said = " ".join(squash_text(n) for n in (*panel.find_all("h2"), *panel.find_all("p"), *panel.find_all("figure")))
                with self.subTest(page=path):
                    self.assertIsNone(re.search(r"\byour\b", said, re.I))

    def test_model_page_marks_the_chosen_engine_and_hides_the_controls(self):
        tree = parse(self.pages()["models/qwen3.6-35b-a3b/index.html"])
        classes = lambda node: node.attrs.get("class", "").split()
        self.assertEqual([n.attrs["id"] for n in tree.find_all("div", "er") if "cur" in classes(n)], ["engine-llamacpp"])
        self.assertEqual({n.attrs["data-rt"] for n in tree.find_all("div", "per") if "cur" in classes(n)}, {"llamacpp"})
        self.assertEqual(len([n for n in tree.find_all("div", "per") if "cur" not in classes(n)]), 3)   # SGLang's blocks
        controls = [*tree.find_all("div", "mchips"), *tree.find_all("div", "runsel"), *tree.find_all("label", "usebtn")]
        self.assertEqual(len(controls), 4)                                # the chips, the select, two radios
        for node in controls:
            with self.subTest(control=node.attrs["class"]):
                self.assertIn("hidden", node.attrs)
                self.assertIn("data-js-only", node.attrs)
        self.assertEqual(squash_text(tree.find_all("p", "moe")[0]),      # M9
                         "Why a 20.5 GB download fits in 14.7 GB of GPU memory: it's a mixture-of-experts model, so most "
                         "of its experts wait in RAM and the busiest stay on the GPU. How that works")
        self.assertEqual(tree.find_all("p", "agent-note")[0].attrs.get("hidden", "shown"), None)  # OpenCode passed
        evidence = [a.attrs["href"] for a in tree.by_id("evidence").find_all("a")]
        self.assertEqual(evidence[2], BLOB + "/research/2026-11-18/llmctl-fixture-e2e/summary.md")   # M18
        self.assertEqual([n.attrs["class"] for n in tree.by_id("engines").find_all("p", "vw")], ["vw ok", "vw ok"])

    def test_engines_list_is_no_radio_group_without_script(self):
        # the lead's ruling on Task 7's deferred minor: the radios are hidden without script, so the static list has
        # no radio group role; page.js sets it when it reveals them (test_site_state)
        for path, page in self.pages().items():
            if path.startswith("models/"):
                engs = parse(page).find_all("div", "engs")[0]
                with self.subTest(page=path):
                    self.assertNotIn("role", engs.attrs)
                    self.assertNotIn("aria-labelledby", engs.attrs)

    def test_model_page_chooses_the_row_setup(self):
        # spec 4.4: the chosen engine is the model's row setup on the laptop, not its default engine
        self.fixture.set_value("llamacpp", "gpt-oss-20b", "memory_gib", "50")
        tree = parse(self.pages()["models/gpt-oss-20b/index.html"])
        classes = lambda node: node.attrs.get("class", "").split()
        self.assertEqual(squash_text(tree.find_all("p", "fe")[0]), "On SGLang")
        self.assertEqual([n.attrs["id"] for n in tree.find_all("div", "er") if "cur" in classes(n)], ["engine-sglang"])
        self.assertEqual({n.attrs["data-rt"] for n in tree.find_all("div", "per") if "cur" in classes(n)}, {"sglang"})
        verdict = tree.find_all("p", "verdict")[0]
        self.assertEqual([squash_text(verdict.find_all("span", c)[0]) for c in ("on", "vt")],
                         ["gpt-oss-20b on SGLang, on our test laptop:", "Runs as measured."])
        self.assertTrue(squash_text(tree.find_all("div", "term")[0]).startswith("Commands for gpt-oss-20b on SGLang"))

    def test_model_page_names_the_agent_to_try_after_a_failed_test(self):
        # M13: the agent note shows without script when our default agent failed the latest test
        k2 = prefix("llamacpp", "k2-horizon-7b")
        self.fixture.run("2026-11-19", "llmctl-k2-e2e",
                         [row(k2, "up"), row(k2, "status"), row(k2, "config"), row(k2, "opencode", "no tool call")])
        note = parse(self.pages()["models/k2-horizon-7b/index.html"]).find_all("p", "agent-note")[0]
        self.assertNotIn("hidden", note.attrs)
        self.assertEqual(squash_text(note), "Failed the latest test in OpenCode. In OpenCode, Qwen3.6-35B-A3B passed the "
                                            "agent test and runs on our laptop as measured.")
        self.assertEqual([a.attrs["href"] for a in note.find_all("a")], ["../qwen3.6-35b-a3b/"])

    def test_evidence_links_the_latest_run(self):
        # M18: the latest run by date, then folder, as How we tested names it (spec 7.10)
        self.fixture.run(DEFAULT_RUN[0], "llmctl-a-e2e", default_rows()[:2])
        tree = parse(self.pages()["models/qwen3.6-35b-a3b/index.html"])
        summaries = [a.attrs["href"] for a in tree.by_id("evidence").find_all("a") if a.attrs["href"].endswith("summary.md")]
        self.assertEqual(summaries, [BLOB + f"/research/{DEFAULT_RUN[0]}/{DEFAULT_RUN[1]}/summary.md"] * 2)

    def test_repository_text_is_escaped(self):
        self.fixture.set_value("llamacpp", "k2-horizon-7b", "name", '"K2 <b>&\\" Horizon Q8_0"')
        home = self.pages()["index.html"]
        self.assertIn("K2 &lt;b&gt;&amp;&quot; Horizon", home)
        self.assertNotIn("<b>&", home)

    def test_every_coverage_note_is_checked(self):
        # the pages show only first clauses, so render.check_notes keeps the subset rule for every note (P20)
        qwen = ("qwen3.6-35b-a3b", "llamacpp")
        self.fixture.coverage({qwen: self.fixture._default_cell(*qwen) + "; a **bold** word"})
        column = cell_of(self.fixture.load(), *qwen).note.index("*") + 1
        with self.assertRaises(data.SiteError) as caught:
            self.pages()
        self.assertEqual(str(caught.exception),
                         f"docs/model-selection.md: row qwen3.6-35b-a3b, llama.cpp, note: column {column}: emphasis")

    def test_pages_render_with_python_3_11_fractions(self):
        with py311_fraction_format():                  # Python 3.11's Fraction refuses every format spec
            self.assertGreater(len(self.pages()), 0)

    def test_colour_tokens_meet_wcag_aa(self):
        light, dark = tokens((REPO / "launcher/site/page.css").read_text(encoding="utf-8"))
        text = [(f, b) for f in ("ink", "ink-2", "ink-3", "gpu-ink", "ram-ink") for b in ("paper", "surface", "code-bg", "hover")]
        text += [(f, "term-bg") for f in ("term-ink", "term-dim", "term-hi")]
        graphics = [(f, b) for f in ("gpu", "ram", "track") for b in ("paper", "surface")]
        for scheme, colours in (("light", light), ("dark", dark)):
            for pairs, floor in ((text, 4.5), (graphics, 3)):
                for f, b in pairs:
                    with self.subTest(scheme=scheme, pair=(f, b)):
                        self.assertGreaterEqual(contrast(colours[f], colours[b]), floor)

    def test_well_formed_problems_names_each_problem(self):
        problems = well_formed_problems('<p><b>x</p></b><i id="x"></i><i id="x"></i><a href="#y"></a><br/>')
        self.assertEqual(problems, ["</p> closes <b> out of order", "<br ... /> is written as a self-closing tag",
                                    "<p> is never closed", "the ID x is used 2 times",
                                    "0 h1 elements", "href=#y names no element"])

    def test_assets_may_not_end_their_element(self):
        with self.assertRaisesRegex(data.SiteError, r"^launcher/site/page\.css: it holds </style, which would end the inline style$"):
            render.check_asset("page.css", "a {} </style>", "style")
        with self.assertRaisesRegex(data.SiteError, r"^launcher/site/page\.js: it holds </script, which would end the inline script$"):
            render.check_asset("page.js", "a </SCRIPT>", "script")
        self.assertEqual(render.check_asset("page.js", '"use strict";', "script"), '"use strict";')

    def test_rich_text_resolves_its_targets_for_the_page(self):
        value = ("a < b ", ("b", "bold"), " ", ("code", "up & go"), " ", ("a", "Home", "/"), " ",
                 ("a", "Start", "/#get-started"), " ", ("a", "Gemma", "/models/gemma-4-12b/"), " ",
                 ("a", "Refusals", "docs/llmctl.md#when-up-refuses"), " ", ("a", "Site", "https://opencode.ai"))
        parts = ('a &lt; b <b>bold</b> <code>up &amp; go</code> <a href="{home}">Home</a> <a href="{start}">Start</a> '
                 '<a href="{gemma}">Gemma</a> <a href="' + BLOB + '/docs/llmctl.md#when-up-refuses">Refusals</a> '
                 '<a href="https://opencode.ai">Site</a>')
        self.assertEqual(render.rich_html(value, self.links, "index.html"),
                         parts.format(home="./", start="#get-started", gemma="models/gemma-4-12b/"))
        self.assertEqual(render.rich_html(value, self.links, "models/qwen3.6-35b-a3b/index.html"),
                         parts.format(home="../../", start="../../#get-started", gemma="../gemma-4-12b/"))
        own = (("a", "Qwen", "/models/qwen3.6-35b-a3b/"), " ", ("a", "Engine", "/models/qwen3.6-35b-a3b/#engine-sglang"))
        self.assertEqual(render.rich_html(own, self.links, "models/qwen3.6-35b-a3b/index.html"),
                         '<a href="../qwen3.6-35b-a3b/">Qwen</a> <a href="../qwen3.6-35b-a3b/#engine-sglang">Engine</a>')
        self.assertEqual(render.rich_html(own, self.links, "index.html"),
                         '<a href="models/qwen3.6-35b-a3b/">Qwen</a> <a href="models/qwen3.6-35b-a3b/#engine-sglang">Engine</a>')
        self.assertEqual(render.rich_html("Tom & <Jerry>", self.links, "index.html"), "Tom &amp; &lt;Jerry&gt;")
        with self.assertRaisesRegex(data.SiteError, "^docs/llmctl.md: no heading has the anchor #nowhere$"):
            render.rich_html((("a", "x", "docs/llmctl.md#nowhere"),), self.links, "index.html")
        with self.assertRaisesRegex(ValueError, r"^launcher/site/render\.py: not a rich text part: \('i', 'x'\)$"):
            render.rich_html((("i", "x"),), self.links, "index.html")

    def test_header_hides_the_middle_of_the_nav_on_phones(self):
        nav = (("Models", "/#models"), ("Get started", "/#get-started"), ("GitHub", "https://github.com/o/r"))
        home = parse(render.header_html("index.html", nav))
        self.assertEqual([(li.attrs.get("class"), li.find_all("a")[0].attrs["href"]) for li in home.find_all("li")],
                         [(None, "#models"), ("wide", "#get-started"), (None, "https://github.com/o/r")])
        self.assertEqual(home.find_all("a", "brand")[0].attrs["href"], "#top")
        model = parse(render.header_html("models/x/index.html", nav))
        self.assertEqual([a.attrs["href"] for a in model.find_all("a")],
                         ["../../", "../../#models", "../../#get-started", "https://github.com/o/r"])

    def test_stylesheet_keeps_hidden_elements_hidden(self):
        css = re.sub(r"/\*.*?\*/", "", (REPO / "launcher/site/page.css").read_text(encoding="utf-8"), flags=re.S)
        self.assertEqual(" ".join(css.split("}", 1)[0].split()) + " }", "[hidden] { display: none !important; }")

    def test_build_is_byte_identical_across_hash_seeds(self):
        code = ("import sys\nfrom pathlib import Path\nfrom launcher.site import cli\n"
                "sys.exit(cli.main(['build', '--out', sys.argv[1], '--commit', 'a' * 40, '--commit-time', '0'], "
                "root=Path(sys.argv[2])))")
        trees = []
        for seed in ("1", "2"):
            out = Path(self._tmp.name) / f"out-{seed}"
            done = subprocess.run([sys.executable, "-B", "-c", code, str(out), str(self.fixture.root)], cwd=REPO,
                                  env={"PYTHONHASHSEED": seed, "PYTHONDONTWRITEBYTECODE": "1"}, capture_output=True,
                                  text=True, timeout=120)
            self.assertEqual(done.returncode, 0, done.stderr)
            trees.append([(path.relative_to(out).as_posix(), path.read_bytes())
                          for path in sorted(out.rglob("*")) if path.is_file()])
        self.assertGreater(len(trees[0]), 0)
        self.assertEqual(trees[0], trees[1])

    def test_only_the_build_facts_change_between_builds(self):
        trees = []
        for name, commit, time in (("a", "a" * 40, "0"), ("b", "b" * 40, "86400")):
            out = Path(self._tmp.name) / f"out-{name}"
            self.assertEqual(self.build(out, commit, time)[0], 0)
            pages = {}
            for path in sorted(out.rglob("*")):
                if path.is_file():
                    page = path.read_text(encoding="utf-8")
                    self.assertIn(commit, page)
                    page = page.replace(commit, "COMMIT")
                    pages[path.relative_to(out).as_posix()] = re.sub(r"Updated [12] January 1970", "UPDATED", page)
            trees.append(pages)
        self.assertGreater(len(trees[0]), 0)
        self.assertEqual(trees[0], trees[1])

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
        (out / "extra").mkdir(parents=True)
        (out / "extra" / "old.html").write_text("old\n", encoding="utf-8")
        (out / "notes.txt").write_text("keep\n", encoding="utf-8")
        self.assertEqual(self.build(out), (1, "", f"site: --out {out}: the folder holds files the build does not write: "
                                                  "extra, notes.txt\n"))
        self.assertEqual(sorted(p.name for p in out.iterdir()), ["extra", "notes.txt"])
        shutil.rmtree(out / "extra")
        (out / "notes.txt").unlink()
        (out / "index.html").write_text("old\n", encoding="utf-8")
        code, stdout, stderr = self.build(out)
        self.assertEqual((code, stderr), (0, ""))
        self.assertRegex(stdout, rf"^wrote {re.escape(str(out))}: 7 pages, \d+ bytes, 6 models, 3 engines, 5 clients, 1 e2e runs\n$")
        self.assertNotEqual((out / "index.html").read_text(encoding="utf-8"), "old\n")
        fresh = Path(self._tmp.name) / "fresh" / "out"      # a new page is 0644 whatever the umask; a replaced one keeps
        self.assertEqual(self.build(fresh)[0], 0)           # its mode (files.write_atomic), so only a fresh folder says
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
        (out / "index.html").rmdir()
        (out / "index.html").symlink_to(target)
        code, stdout, stderr = self.build(out)
        self.assertEqual((code, stderr), (1, f"site: --out {out}: {out / 'index.html'} is not a regular file\n"))
        self.assertEqual(target.read_text(encoding="utf-8"), "keep\n")
        # The pages' paths come from the repository, so the build reads it before it checks the --out folder.
        if os.geteuid() != 0:                                                        # root reads any file
            readme = self.fixture.root / "README.md"
            readme.chmod(0)
            self.addCleanup(readme.chmod, 0o644)
            code, stdout, stderr = self.build(Path(self._tmp.name) / "unread")
            self.assertEqual((code, stdout, stderr), (1, "", "site: README.md: Permission denied\n"))

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


def sha(text: str) -> str:
    return "'sha256-" + base64.b64encode(hashlib.sha256(text.encode("utf-8")).digest()).decode("ascii") + "'"


def squash_text(node) -> str:
    """The node's text with whitespace collapsed."""
    return " ".join(node.text().split())


def terminal_lines(node) -> list[str]:
    """The lines of the div.term inside node, as "# ..." for a comment and "$ ..." for a command."""
    out = []
    for line in node.find_all("div", "term")[0].find_all("div", "tl")[0].children:
        if not isinstance(line, Node):
            continue
        if "c" in (line.attrs.get("class") or "").split():
            out.append(squash_text(line))
        else:
            out.append("$ " + " ".join("".join(part if isinstance(part, str) else "" if "p" in (
                part.attrs.get("class") or "").split() else part.text() for part in line.children).split()))
    return out


def visible_text(page: str) -> str:
    """The text of the parsed body without script and style contents."""
    def walk(node) -> str:
        if node.tag in ("script", "style"):
            return ""
        return "".join(child if isinstance(child, str) else walk(child) for child in node.children)
    return walk(parse(page).find_all("body")[0])


def tokens(css: str) -> tuple[dict[str, str], dict[str, str]]:
    """The --name: #rrggbb values of the stylesheet's first :root block, then those of the :root block under
    @media (prefers-color-scheme: dark) laid over them."""
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    values = lambda block: dict(re.findall(r"--([a-z0-9-]+)\s*:\s*(#[0-9A-Fa-f]{6})\s*;", block))
    light = values(re.search(r":root\s*\{([^}]*)\}", css).group(1))
    dark = re.search(r"@media\s*\(prefers-color-scheme:\s*dark\)\s*\{\s*:root\s*\{([^}]*)\}", css)
    return light, {**light, **values(dark.group(1))}


def _luminance(colour: str) -> float:
    channels = [int(colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast(a: str, b: str) -> float:
    """The WCAG 2.1 contrast ratio of two #rrggbb colours."""
    high, low = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


if __name__ == "__main__":
    unittest.main()
