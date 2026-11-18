"""The shared Markdown helpers: heading anchors, sections and table rows."""

import tempfile
import unittest
from pathlib import Path

from launcher import markdown


class MarkdownTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def write(self, name: str, text: str) -> Path:
        path = self.root / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_slug_follows_the_link_checker(self):
        self.assertEqual(markdown.slug("2. Qwen3.6-35B-A3B"), "2-qwen36-35b-a3b")
        self.assertEqual(markdown.slug("10. DeepSeek-R1-0528-Qwen3-8B (chat and reasoning only)"),
                         "10-deepseek-r1-0528-qwen3-8b-chat-and-reasoning-only")
        self.assertEqual(markdown.slug("Mellum2.1 `Q6_K` build"), "mellum21-q6_k-build")
        self.assertEqual(markdown.slug("See [the doc](x.md#y) here"), "see-the-doc-here")

    def test_section_stops_at_a_heading_of_the_same_level(self):
        self.write("f.md", "# T\n## A\nx\n### B\ny\n## C\nz\n")
        self.assertEqual(markdown.section(self.root, "f.md#a"), ["x", "### B", "y"])

    def test_section_raises_lookup_error_naming_the_anchor(self):
        self.write("f.md", "# T\n## A\nx\n### B\ny\n## C\nz\n")
        with self.assertRaisesRegex(LookupError, r"^no section #x in f\.md$"):
            markdown.section(self.root, "f.md#x")

    def test_heading_slugs_skip_fenced_blocks(self):
        path = self.write("g.md", "# Top\n```\n# not one\n```\n## Real one\n")
        self.assertEqual(markdown.heading_slugs(path), {"top", "real-one"})

    def test_table_reads_rows_until_the_first_other_line(self):
        lines = ["intro", "| A | B |", "|---|---|", "| 1 | `2` |", "| 3 | 4 |", "", "| 5 | 6 |"]
        self.assertEqual(markdown.table(lines, "A"), (["A", "B"], [["1", "`2`"], ["3", "4"]]))
        with self.assertRaisesRegex(ValueError, r"\| 1 \| 2 \| 3 \|"):
            markdown.table(["| A | B |", "|---|---|", "| 1 | 2 | 3 |"], "A")
        with self.assertRaises(ValueError):
            markdown.table(["| A | B |", "| 1 | 2 |"], "A")
        with self.assertRaises(LookupError):
            markdown.table(lines, "C")
        self.assertEqual(markdown.table(["| A | B |", "|---|---|", "| 1 | 2 |", "text", "| 3 | 4 |"], "A"),
                         (["A", "B"], [["1", "2"]]))
        with self.assertRaises(LookupError):
            markdown.table(["| AB | C |", "|---|---|"], "A")
        with self.assertRaisesRegex(ValueError, r"^table A: 1 cell where the header has 2: \| x \|$"):
            markdown.table(["| A | B |", "|---|---|", "| x |"], "A")
        with self.assertRaisesRegex(ValueError, r"^table A: expected a separator line after the header, found: a blank line$"):
            markdown.table(["| A | B |", "", "| 1 | 2 |"], "A")
        with self.assertRaisesRegex(ValueError, r"^table A: the separator line has 1 cell where the header has 2: \|---\|$"):
            markdown.table(["| A | B |", "|---|", "| 1 | 2 |"], "A")


if __name__ == "__main__":
    unittest.main()
