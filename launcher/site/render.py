"""The model guide's renderer: the shell every page shares and the pages themselves (spec 8), the Markdown subset of
the coverage notes with repository links at the build's commit (design D12), and the README recipe table
(D-README)."""

import base64
import hashlib
import html
import posixpath
import re
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import quote, unquote

from launcher import markdown
from launcher.catalog import Catalog
from launcher.runtimes import ADAPTERS
from launcher.site import data, words
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


_ASSETS = Path(__file__).resolve().parent
HOME = "index.html"
HEAD_SCRIPT = 'document.documentElement.classList.add("js");'
# v1's brand mark (spec 8.1): three solid blocks and one hatched, in the GPU colour.
FAVICON = "data:image/svg+xml," + quote(
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 20 20'>"
    "<rect width='9' height='9' rx='2' fill='#0F8CA3'/><rect x='11' width='9' height='9' rx='2' fill='#0F8CA3'/>"
    "<rect y='11' width='9' height='9' rx='2' fill='#0F8CA3'/>"
    "<rect x='11.5' y='11.5' width='8' height='8' rx='1.5' fill='none' stroke='#0F8CA3'/>"
    "<path d='M12 17l5-5M12 14.5l2.5-2.5M14.5 19l4.5-4.5' stroke='#0F8CA3' stroke-width='1.2'/></svg>", safe=":/'=")
# The hatch pattern every estimate in a drawing fills with (spec 8.1, 8.3).
DEFS = ('<svg class="defs" width="0" height="0" aria-hidden="true" focusable="false"><defs>'
        '<pattern id="hatch-desk" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
        '<rect class="hatch-line" width="2" height="6"></rect></pattern></defs></svg>')
MARK = '<span class="mark" aria-hidden="true"><i></i><i></i><i></i><i class="h"></i></span>'


def _e(text: object) -> str:
    return html.escape(str(text), quote=True)


def _sha(text: str) -> str:
    return "'sha256-" + base64.b64encode(hashlib.sha256(text.encode("utf-8")).digest()).decode("ascii") + "'"


def check_asset(name: str, text: str, tag: str) -> str:
    """The text of launcher/site/<name>, which may not end the inline element it goes into (spec 8.1)."""
    if f"</{tag}" in text.lower():
        raise SiteError(f"launcher/site/{name}: it holds </{tag}, which would end the inline {tag}")
    return text


def check_notes(matrix: data.Matrix, links: Links) -> None:
    """Every coverage note whole through inline, so a note outside the Markdown subset stops the build although the
    pages show only first clauses (spec 8.2, P20)."""
    labels = {runtime.id: runtime.label for runtime in matrix.runtimes}
    for model in matrix.models:
        for cell in model.cells:
            inline(cell.note, data.SELECTION, links, f"row {model.id}, {labels[cell.runtime]}, note: ")


def _page_href(target: str, here: str) -> str:
    """An https:// target as it is, or a page of this site (/...) relative to the page at `here` (spec 4.3): "/#models"
    is "#models" on the home and "../../#models" on a model page, and "/models/<id>/" is "models/<id>/" on the home
    and "../<id>/" on a model page, its own model's included."""
    if target.startswith("https://"):
        return target
    if not target.startswith("/"):
        raise ValueError(f"launcher/site/render.py: {target!r} is neither https:// nor a page of this site")
    if here != HOME and target.startswith("/models/"):
        return "../" + target[len("/models/"):]
    return "../" * here.count("/") + target[1:] or "./"


def _href(target: str, links: Links, here: str) -> str:
    """A link target of words.py (spec 7.1): https://, a page of this site, or a repository path with an optional
    #anchor, which Links checks and points at the build's commit."""
    if target.startswith(("https://", "/")):
        return _page_href(target, here)
    path, mark, anchor = target.partition("#")
    return links.repo_link(path, anchor if mark else None)


def rich_html(value: str | words.Rich, links: Links, here: str) -> str:
    """A plain or rich value of words.py as HTML: text escaped, bold, code spans and links resolved for the page at
    `here`."""
    if isinstance(value, str):
        return _e(value)
    out = []
    for part in value:
        if isinstance(part, str):
            out.append(_e(part))
        elif len(part) == 2 and part[0] in ("b", "code"):
            out.append(f"<{part[0]}>{_e(part[1])}</{part[0]}>")
        elif len(part) == 3 and part[0] == "a":
            out.append(f'<a href="{_e(_href(part[2], links, here))}">{_e(part[1])}</a>')
        else:
            raise ValueError(f"launcher/site/render.py: not a rich text part: {part!r}")
    return "".join(out)


def header_html(here: str, nav: Sequence[tuple[str, str]]) -> str:
    """The site header: the brand (to the home's top) and the nav, (label, target) pairs whose middle items hide on
    phones (spec 3.2)."""
    brand = "#top" if here == HOME else _page_href("/", here)
    items = "".join(('<li class="wide">' if 0 < n < len(nav) - 1 else "<li>")
                    + f'<a href="{_e(_page_href(target, here))}">{_e(label)}</a></li>' for n, (label, target) in enumerate(nav))
    return (f'<header class="site" id="top"><div class="wrap"><a class="brand" href="{_e(brand)}" '
            f'aria-label="{_e(words.BRAND_LABEL)}">{MARK}{_e(words.BRAND)}</a>'
            f'<nav class="topnav" aria-label="{_e(words.NAV_LABEL)}"><ul>{items}</ul></nav></div></header>')


def footer_html(matrix: data.Matrix, build: Build, links: Links, here: str) -> str:
    """The footer's three paragraphs (spec 7.11)."""
    paragraphs = words.footer(build.repository, build.commit, build.commit_time, matrix.licenses)
    return ('<footer class="foot"><div class="wrap">' + "".join(f"<p>{rich_html(p, links, here)}</p>" for p in paragraphs)
            + "</div></footer>")


def document(*, title: str, description: str, body: str, style: str, scripts: tuple[str, ...] = (),
             state: str | None = None) -> str:
    """One self-contained page (spec 8.1): the policy first, the style and, with scripts, the head script that marks
    the page as scripted; `body` between the drawing defs and the live region, then the state JSON and the scripts.
    The policy names a script-src only when the page has scripts."""
    inline_scripts = (HEAD_SCRIPT, *scripts) if scripts else ()
    sources = f"script-src {' '.join(_sha(script) for script in inline_scripts)}; " if inline_scripts else ""
    policy = (f"default-src 'none'; {sources}style-src {_sha(style)}; img-src data:; base-uri 'none'; "
              "form-action 'none'")
    head = ["<head>", '<meta charset="utf-8">', '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f'<meta http-equiv="Content-Security-Policy" content="{_e(policy)}">',
            '<meta name="color-scheme" content="light dark">', f'<meta name="description" content="{_e(description)}">',
            f"<title>{_e(title)}</title>", f'<link rel="icon" href="{_e(FAVICON)}">', f"<style>{style}</style>",
            *([f"<script>{HEAD_SCRIPT}</script>"] if scripts else []), "</head>"]
    tail = ['<div id="live" class="vh" aria-live="polite"></div>',
            *([] if state is None else [f'<script type="application/json" id="state">{state}</script>']),
            *(f"<script>{script}</script>" for script in scripts)]
    return "\n".join(["<!doctype html>", '<html lang="en">', *head, "<body>", DEFS, body, *tail, "</body>",
                      "</html>"]) + "\n"


def _home_main() -> str:
    return (f'<main id="main" class="wrap"><section class="fh" aria-labelledby="q"><div class="ask">'
            f'<h1 id="q">{_e(words.H1)}</h1></div></section></main>')


def site(matrix: data.Matrix, build: Build, root: Path) -> dict[str, str]:
    """Every page of the site by its path under --out, the home first (spec 3.1 and 10)."""
    links = Links(root, build)
    check_notes(matrix, links)
    labels = words.labels(matrix)
    style = check_asset("page.css", (_ASSETS / "page.css").read_text(encoding="utf-8"), "style")
    nav = ((words.NAV["github"], build.repository),)
    body = "\n".join([header_html(HOME, nav), _home_main(), footer_html(matrix, build, links, HOME)])
    return {HOME: document(title=words.HOME_TITLE, description=words.home_description(labels), body=body, style=style)}
