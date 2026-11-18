"""The runtime matrix page's renderer: the Markdown subset of the coverage notes with repository links at the build's
commit (design D12), and the README recipe table (D-README)."""

import html
import posixpath
import re
from pathlib import Path
from urllib.parse import quote, unquote

from launcher import markdown
from launcher.catalog import Catalog
from launcher.runtimes import ADAPTERS
from launcher.site.data import Build, SiteError

README = "README.md"
README_START = "<!-- recipe table: written by python3 -m launcher.site readme from recipes/; do not edit by hand -->"
README_END = "<!-- end of recipe table -->"
QUICK_START = "## Quick start"

_LINK = re.compile(r"\[([^\[\]`]*)\]\(([^()\s]*)\)")
_ENTITY = re.compile(r"&[A-Za-z#][A-Za-z0-9]*;")
_BARE = re.compile(r"https?://|www\.|[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z]")
_FORBIDDEN = {"*": "emphasis", "~": "strikethrough", "<": "an HTML tag", ">": "a > outside a code span",
              "\\": "a backslash escape", "[": "a stray [", "]": "a stray ]"}


class Links:
    """Links to repository paths at the build's commit; every Markdown file's headings are read once."""

    def __init__(self, root: Path, build: Build):
        self.root = root
        self.build = build
        self._slugs: dict[str, set[str]] = {}

    def _anchors(self, path: str) -> set[str]:
        if path not in self._slugs:
            try:
                self._slugs[path] = markdown.heading_slugs(self.root / path)
            except UnicodeDecodeError as exc:
                raise SiteError(f"{path}: not valid UTF-8: {exc}") from None
        return self._slugs[path]

    def repo_link(self, path: str, anchor: str | None = None) -> str:
        """The URL of a repository path ("" is the root): blob for a file, tree for a folder, each segment quoted; a
        folder may be written with a trailing /, which the URL leaves out."""
        written = path
        if path.endswith("/") and len(path) > 1:
            path = path[:-1]
        if path.startswith("/") or path.split("/")[0] == "..":
            raise SiteError(f"{written}: not a path inside the repository")
        if path and (posixpath.normpath(path) != path or path == "."):
            raise SiteError(f"{written}: not a normalized repository path")
        full = self.root / path
        if not full.exists():
            raise SiteError(f"{written}: no such file or folder in the repository")
        folder = full.is_dir()
        if written != path and not folder:
            raise SiteError(f"{written}: a trailing / on a file")
        if anchor is not None:
            if folder or not path.endswith(".md"):
                raise SiteError(f"{path}: an anchor (#{anchor}) on a link to a file that is not Markdown")
            if anchor not in self._anchors(path):
                raise SiteError(f"{path}: no heading has the anchor #{anchor}")
        url = f"{self.build.repository}/{'tree' if folder else 'blob'}/{self.build.commit}"
        if path:
            url += "/" + "/".join(quote(segment, safe="") for segment in path.split("/"))
        return url if anchor is None else f"{url}#{anchor}"

    def resolve(self, source: str, target: str) -> str:
        """The URL of a relative link target written in the repository file `source`."""
        part, hash_mark, anchor = target.partition("#")
        if not part:
            return self.repo_link(source, anchor)
        path = posixpath.normpath(posixpath.join(posixpath.dirname(source), unquote(part)))
        if path.startswith("/") or path.split("/")[0] == "..":
            raise SiteError(f"{source}: the link {target} leaves the repository")
        return self.repo_link("" if path == "." else path, anchor if hash_mark else None)


def _target(target: str, source: str, links: Links) -> str:
    if target.startswith("https://"):
        return target
    if not target:
        raise SiteError("an empty link target")
    if target.startswith("/") or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", target):
        raise SiteError(f"the link target {target} is neither https://, an anchor nor a relative path")
    return links.resolve(source, target)


def _plain(text: str, lo: int, hi: int, fail) -> str:
    """text[lo:hi] escaped, when it holds no Markdown and nothing GitHub autolinks; an underscore needs a letter or
    digit on both sides in text."""
    bare = _BARE.search(text, lo, hi)
    for n in range(lo, hi):
        char = text[n]
        if bare is not None and n == bare.start():
            fail(n, "a bare URL or email address")
        if char in _FORBIDDEN:
            fail(n, _FORBIDDEN[char])
        if char == "_" and not (0 < n < len(text) - 1 and text[n - 1].isalnum() and text[n + 1].isalnum()):
            fail(n, "an underscore outside a word")
        if char == "&" and _ENTITY.match(text, n):
            fail(n, "an HTML entity outside a code span")
    return html.escape(text[lo:hi], quote=True)


def inline(text: str, source: str, links: Links, where: str = "") -> str:
    """The Markdown subset of design D12 as HTML: text, code spans, and links to https://, an anchor of the source or a
    relative repository path; anything else raises SiteError naming the source, `where` and the 1-based column."""

    def fail(index: int, what: str):
        raise SiteError(f"{source}: {where}column {index + 1}: {what}")

    out, n, start = [], 0, 0
    while n < len(text):
        char = text[n]
        if char == "!" and text.startswith("[", n + 1):
            fail(n, "an image")
        if char not in "`[":
            n += 1
            continue
        out.append(_plain(text, start, n, fail))
        if char == "`":
            if text.startswith("``", n):
                fail(n, "a run of two or more backticks")
            close = text.find("`", n + 1)
            if close < 0:
                fail(n, "a code span without its closing backtick")
            if text.startswith("``", close):
                fail(close, "a run of two or more backticks")
            out.append(f"<code>{html.escape(text[n + 1:close], quote=True)}</code>")
            n = close + 1
        else:
            link = _LINK.match(text, n)
            if link is None:
                fail(n, "a stray [")
            label = _plain(text, link.start(1), link.end(1), fail)
            if "\\" in link.group(2):
                fail(n, "a backslash in a link target")
            if _ENTITY.search(link.group(2)):
                fail(n, "an HTML entity in a link target")
            problem = None
            try:
                href = _target(link.group(2), source, links)
            except SiteError as exc:
                problem = str(exc)
            if problem is not None:
                fail(n, problem)
            out.append(f'<a href="{html.escape(href, quote=True)}">{label}</a>')
            n = link.end()
        start = n
    out.append(_plain(text, start, len(text), fail))
    return "".join(out)


def readme_table(catalog: Catalog) -> str:
    """The README's recipe table: one row per shipped model ID, three columns per engine in ADAPTERS order."""
    header = "| Model ID | Aliases | " + " | ".join(f"{adapter.LABEL} | Context | Decode tok/s"
                                                 for adapter in ADAPTERS.values()) + " |"
    lines = [header, "|" + "---|" * (2 + 3 * len(ADAPTERS))]
    ids = sorted({recipe_id for name in ADAPTERS for recipe_id in catalog.runtimes[name].recipes})
    for recipe_id in ids:
        aliases: list[str] = []
        cells: list[str] = []
        for name in ADAPTERS:
            recipe = catalog.runtimes[name].recipes.get(recipe_id)
            if recipe is None:
                cells += ["No", "-", "-"]
                continue
            if recipe.context % 1024:
                raise SiteError(f"recipes/{name}/{recipe_id}.toml: context {recipe.context} is not a whole number "
                                "of k")
            speed = recipe.evidence.decode_tok_s
            cells += ["Yes", f"{recipe.context // 1024}k", "-" if speed is None else f"{speed:g}"]
            aliases += [alias for alias in recipe.aliases if alias not in aliases]
        alias_cell = ", ".join(f"`{alias}`" for alias in aliases) or "-"
        lines.append(f"| `{recipe_id}` | {alias_cell} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _markers(text: str) -> tuple[list[str], int, int]:
    lines = text.splitlines(keepends=True)
    bare = [line.rstrip("\n") for line in lines]
    found = {}
    for marker, name in ((README_START, "start"), (README_END, "end")):
        places = [n for n, line in enumerate(bare) if line == marker]
        if len(places) != 1:
            raise SiteError(f"{README}: the recipe table's {name} marker appears {len(places)} times, not once: "
                            f"{marker}")
        found[name] = places[0]
    start, end = found["start"], found["end"]
    if start > end:
        raise SiteError(f"{README}: the recipe table's end marker comes before its start marker")
    if QUICK_START not in bare:
        raise SiteError(f"{README}: no {QUICK_START} section")
    section = bare.index(QUICK_START)
    after = next((n for n in range(section + 1, len(bare)) if bare[n].startswith("## ")), len(bare))
    if not section < start < end < after:
        raise SiteError(f"{README}: the recipe table's markers must lie inside the {QUICK_START} section")
    return lines, start, end


def readme_block(text: str) -> str:
    """The exact text between the README's two marker lines."""
    lines, start, end = _markers(text)
    return "".join(lines[start + 1:end])


def rewrite_readme(text: str, table: str) -> str:
    """The README with a blank line, the table and a blank line between its two marker lines."""
    lines, start, end = _markers(text)
    return "".join(lines[:start + 1]) + "\n" + table + "\n" + "".join(lines[end:])
