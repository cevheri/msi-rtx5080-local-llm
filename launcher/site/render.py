"""The runtime matrix page's renderer: the page itself (design D3, The page), the Markdown subset of the coverage
notes with repository links at the build's commit (design D12), and the README recipe table (D-README)."""

import base64
import datetime
import hashlib
import html
import posixpath
import re
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from urllib.parse import quote, unquote

from launcher import markdown
from launcher.catalog import Catalog
from launcher.runtimes import ADAPTERS
from launcher.site import data
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


HEAD_SCRIPT = 'document.documentElement.className = "js";'
TITLE = "Local LLMs on one laptop"
NO_JS = "Filters and use cases need JavaScript; the full matrix and each use case as a list are below."
PRIVACY = "This page sets no cookies, has no analytics and loads nothing from other sites."
STATUS_LABELS = {"runs": "Runs", "failed": "Failed", "does-not-fit": "Does not fit"}
CLASS_WORDS = {"passed": "passed", "known-issue": "known issue", "failed": "failed", "not-tested": "not tested",
               "not-supported": "not supported"}
CLASS_MEANINGS = {"passed": "the latest e2e session passed",
                  "known-issue": "the latest session failed, and the runtime coverage table describes the failure",
                  "failed": "the latest session failed, and the runtime coverage table does not describe it yet",
                  "not-tested": "no e2e run has covered this pair with this client yet",
                  "not-supported": "llmctl writes no entry for this client on this pair"}
SETUP_MARK = "Setup check failed"
SEARCH_LABEL = "Search models"
SEARCH_PLACEHOLDER = "Search models"
SEARCH_SHORT = "Search"
RAM_HELP = "./llmctl up refuses to start a recipe when less RAM than its container's memory cap is available."
_ASSETS = Path(__file__).resolve().parent


def _e(text: object) -> str:
    return html.escape(str(text), quote=True)


def _start(tag: str, attrs: Sequence[tuple[str, str | None]] = ()) -> str:
    """An opening tag; an attribute whose value is None is written bare."""
    return "<" + " ".join([tag, *(name if value is None else f'{name}="{_e(value)}"' for name, value in attrs)]) + ">"


def _sha(text: str) -> str:
    return "'sha256-" + base64.b64encode(hashlib.sha256(text.encode("utf-8")).digest()).decode("ascii") + "'"


def switch_width(engines: int) -> str:
    """The width from which the matrix is a table: 12 + 14 x engines + 0.25 x (engines + 1) + 3 rem."""
    width = 12 + 14 * engines + Fraction(1, 4) * (engines + 1) + 3
    return f"{float(width):g}rem"


def inline_assets(css: str, js: str, engines: int) -> tuple[str, str]:
    """The stylesheet with its one @SWITCH@ token replaced, and the script; either may not end its own element."""
    count = css.count("@SWITCH@")
    if count != 1:
        raise SiteError(f"launcher/site/page.css: the token @SWITCH@ appears {count} times, not once")
    if "</style" in css.lower():
        raise SiteError("launcher/site/page.css: it holds </style, which would end the inline style")
    if "</script" in js.lower():
        raise SiteError("launcher/site/page.js: it holds </script, which would end the inline script")
    return css.replace("@SWITCH@", switch_width(engines)), js


def _count(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def client_sentence(results: Sequence[data.ClientResult], labels: Mapping[str, str]) -> str:
    """The client result in words (Copy, card clients): the count the cell shows, then each client that did not pass,
    with its run count for Known issue and Failed."""
    passed = sum(1 for result in results if result.cls == "passed")
    if passed == len(results):
        return f"All {len(results)} clients passed."
    rest = [f"{labels[result.client]}: {CLASS_WORDS[result.cls]}"
            + (f", {result.passed_runs} of {result.runs} runs passed" if result.cls in ("known-issue", "failed") else "")
            for result in results if result.cls != "passed"]
    return f"{passed} of {len(results)} clients; " + "; ".join(rest) + "."


def _lamp(cls: str, letter: str, hidden: bool = True) -> str:
    """A client mark: span.lamp.<class> holding the letter; hidden from readers unless its container is."""
    attrs = [("class", f"lamp {cls}")] + ([("aria-hidden", "true")] if hidden else [])
    return _start("span", attrs) + f"<span>{_e(letter)}</span></span>"


def _bar(width: int) -> str:
    """A speed bar on the page's one scale, its width a w-<n> class the build styles."""
    return f'<span class="bar" aria-hidden="true"><i class="w-{width}"></i></span>'


class _Page:
    """The parts of one page; every repository string is escaped here, every Markdown text goes through inline."""

    def __init__(self, matrix: data.Matrix, build: Build, root: Path):
        self.matrix = matrix
        self.build = build
        self.links = Links(root, build)
        self.labels = {runtime.id: runtime.label for runtime in matrix.runtimes}
        self.client_labels = {client.id: client.label for client in matrix.clients}
        self.top = data.top_speed(matrix)
        self.runs = [(model, cell) for model in matrix.models for cell in model.cells if cell.status == "runs"]
        if not self.runs:
            raise SiteError(f"{data.SELECTION}: runtime coverage: no Runs: cell, so the page has no measurement")
        self.hardware = "; ".join(sorted({cell.recipe.hardware for _, cell in self.runs}))
        self.committed = datetime.datetime.fromtimestamp(build.commit_time, datetime.timezone.utc).strftime(
            "%Y-%m-%d %H:%M")

    def speeds_line(self) -> str:
        labels = [f"(measured on {self.hardware}, {runtime.heading})" for runtime in self.matrix.runtimes]
        return _e(f"Speeds and contexts: {data._joined(labels)}; Does not fit: (estimate).")

    def head(self, style: str, script: str) -> str:
        labels = data._joined([runtime.label for runtime in self.matrix.runtimes])
        description = (f"Which open-weight LLMs run on one laptop with {labels}, how fast, and with which agent clients; "
                       f"measured on {self.hardware}.")
        policy = (f"default-src 'none'; script-src {_sha(HEAD_SCRIPT)} {_sha(script)}; style-src {_sha(style)}; "
                  "img-src data:; base-uri 'none'; form-action 'none'")
        return "\n".join([
            "<head>", '<meta charset="utf-8">', '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f'<meta http-equiv="Content-Security-Policy" content="{policy}">', f"<title>{_e(TITLE)}</title>",
            f'<meta name="description" content="{_e(description)}">', '<meta name="color-scheme" content="light dark">',
            '<link rel="icon" href="data:,">', f"<script>{HEAD_SCRIPT}</script>", f"<style>{style}</style>", "</head>"])

    def header(self) -> str:
        dates = [cell.recipe.date for _, cell in self.runs]
        found = data.view(self.matrix, "", data.Filters(), None)
        picks = data.picks_line(found.picks, self.matrix)
        facts = (f'<span id="answer" role="status">{_e(found.count)}<span id="answer-picks" class="vh">'
                 f'{_e(" " + picks if picks else "")}</span></span> '
                 f"{_count(len(self.matrix.runtimes), 'engine', 'engines')} and "
                 f"{_count(len(self.matrix.clients), 'agent client', 'agent clients')}. "
                 f"Measured on {_e(self.hardware)}, "
                 f'<span class="nw">{_e(min(dates))}</span> to <span class="nw">{_e(max(dates))}</span>. '
                 '<a href="#about">About the data</a>')
        return "\n".join([
            "<header>", '<div class="top">', f"<h1>{_e(TITLE)}</h1>", self.controls(), "</div>",
            f'<p class="facts">{facts}</p>', '<p id="use-rule" class="rule" hidden></p>',
            '<div id="notice" hidden>', '<p id="notice-text"></p>', '<button type="button" id="dismiss">Dismiss</button>',
            "</div>", "</header>"])

    def controls(self) -> str:
        """The control row (Header and controls): View, Search, Filters, Sort and Copy link, shown by the script."""
        views = "".join(_start("option", [("value", use.id), ("data-sort", use.sort), ("data-roles", " ".join(use.roles))]
                               + ([("selected", None)] if not use.id else []))
                        + f"{_e(data.view_name(use, self.matrix))}</option>" for use in data.USE_CASES)
        sorts = "".join(_start("option", [("value", key)] + ([("selected", None)] if key == data.USE_CASES[0].sort else []))
                        + f"{_e(data.sort_label(key, self.matrix))}</option>" for key, _ in data.SORTS)
        return "\n".join([
            '<div id="controls" hidden>', '<div id="ctl" class="ctl">',
            f'<span id="usec" class="pair"><label for="use">View</label><select id="use" aria-describedby="use-rule">{views}</select>'
            "</span>",
            f'<span class="q"><label for="q" class="vh">{_e(SEARCH_LABEL)}</label>'
            + _start("input", [("id", "q"), ("type", "search"), ("placeholder", SEARCH_PLACEHOLDER),
                               ("data-short", SEARCH_SHORT)]) + "</span>",
            '<button type="button" id="open-filters" aria-haspopup="dialog">Filters</button>',
            f'<span id="sortc" class="pair"><label for="sort">Sort</label><select id="sort">{sorts}</select></span>',
            '<button type="button" id="copy-link">Copy link</button>', "</div>", "</div>"])

    def _checkboxes(self, legend: str, name: str, values: Sequence[tuple[str, str]]) -> str:
        boxes = "\n".join(f"<label>{_start('input', [('type', 'checkbox'), ('name', name), ('value', value), ('checked', None)])} "
                          f"{_e(label)}</label>" for value, label in values)
        return f'<fieldset class="group">\n<legend>{_e(legend)}</legend>\n{boxes}\n</fieldset>'

    def _select(self, name: str, label: str, values: Sequence[tuple[str, str]], selected: str = "",
                help_text: str | None = None) -> str:
        attrs = [("id", name), ("name", name)] + ([("aria-describedby", f"{name}-help")] if help_text else [])
        options = "\n".join(_start("option", [("value", value)] + ([("selected", None)] if value == selected else []))
                            + f"{_e(text)}</option>" for value, text in values)
        help_line = f'\n<p id="{name}-help" class="help">{_e(help_text)}</p>' if help_text else ""
        return (f'<div class="control">\n<label for="{name}">{_e(label)}</label>\n{_start("select", attrs)}\n{options}\n'
                f"</select>{help_line}\n</div>")

    def sheet(self) -> str:
        """The Filters sheet (Filters sheet): the shipped controls in the shipped order, the slot Sort and Copy link move
        into on phones, and the Show and Clear buttons."""
        facts = [cell.recipe for _, cell in self.runs]
        contexts = sorted({recipe.context // 1024 for recipe in facts})
        caps = sorted({recipe.memory_cap for recipe in facts})
        present = {recipe.thinking for recipe in facts}
        found = data.view(self.matrix, "", data.Filters(), None)
        return "\n".join([
            '<dialog id="sheet" aria-labelledby="sheet-title">', '<div class="sheet-head">',
            '<h2 id="sheet-title">Filters</h2>', '<button type="button" id="sheet-close">Close</button>', "</div>",
            '<div class="sheet-body">',
            self._checkboxes("Engines", "engine", [(runtime.id, runtime.label) for runtime in self.matrix.runtimes]),
            self._checkboxes("Status", "status", list(STATUS_LABELS.items())),
            self._select("pass", "Must pass in", [("", "Any client"),
                                                  *((client.id, client.label) for client in self.matrix.clients),
                                                  ("every", "Every client")]),
            self._select("ctx", "Context", [("", "Any"), *((str(k), f"{k}k or more") for k in contexts)]),
            self._checkboxes("Thinking", "think", [(kind, label) for kind, label in data.THINKING.items()
                                                    if kind in present]),
            self._select("tools", "Tools", [("", "Any"), ("yes", "Tool calling"), ("no", "Chat only")]),
            self._select("ram", "Free RAM", [("", "Any"), *((str(cap), f"{cap} GiB") for cap in caps)],
                         help_text=RAM_HELP),
            '<div id="sheet-slot"></div>', "</div>", '<div class="sheet-foot">',
            f'<button type="button" id="show">{_e(data.show_label(len(found.rows)))}</button>',
            '<button type="button" id="clear">Clear filters</button>', "</div>",
            '<p id="sheet-live" class="vh" aria-live="polite"></p>', "</dialog>"])

    def _runs_tile(self, model: data.Model, cell: data.Cell) -> str:
        """A Runs tile: span.g with its parts in the tile table's order (Table and cells)."""
        facts = cell.recipe
        label = self.labels[cell.runtime]
        if facts.decode is None:
            parts = ['<span class="sp"><span class="vh">speed </span><span class="none">not recorded</span>'
                     '<span class="vh">,</span></span>']
        else:
            parts = [f'<span class="sp">{_e(data.speed_text(facts.decode))}<span class="vh"> tok/s,</span></span>']
        marker = data.lead(model, cell, self.labels)
        if marker is not None:
            shown, words = marker
            if shown == "tied":
                parts.append(f'<span class="mk"><span aria-hidden="true">* tied</span><span class="vh">{_e(words)},'
                             "</span></span>")
            else:
                first, _, rest = words.partition(shown)
                parts.append(f'<span class="mk"><span aria-hidden="true">* </span><span class="vh">{_e(first)}</span>'
                             f'{_e(shown)}<span class="vh">{_e(rest)},</span></span>')
        if (model.id, cell.runtime) == self.matrix.default:
            words = data.role_words("default", self.matrix)
            parts.append(f'<span class="dm"><span class="tag" aria-hidden="true">{_e(self.client_labels["opencode"])} '
                         f'default</span><span class="vh">{_e(words[:1].lower() + words[1:])},</span></span>')
        if cell.setup:
            parts.append(f'<span class="su">{_e(SETUP_MARK)}<span class="vh">,</span></span>')
        parts.append(f'<span class="cx">{facts.context // 1024}k<span class="vh"> context,</span></span>')
        parts.append(f'<span class="sz">{facts.size_bytes / 1e9:.1f} GB<span class="vh"> download,</span></span>')
        letters = {client.id: client.letter for client in self.matrix.clients}
        passed = sum(1 for result in cell.clients if result.cls == "passed")
        parts.append(f'<span class="pc" aria-hidden="true"><span class="pcn">{passed}/{len(cell.clients)}</span>'
                     + "".join(_lamp(result.cls, letters[result.client], hidden=False)
                               for result in cell.clients if result.cls != "passed") + "</span>")
        parts.append(_bar(data.bar_width(facts.decode, self.top)))
        parts.append(f'<span class="vh">{_e(label)}: {_e(client_sentence(cell.clients, self.client_labels))}</span>')
        return (f'<a class="tile" href="#d-{_e(cell.runtime)}-{_e(model.id)}"><span class="g">' + "".join(parts)
                + "</span></a>")

    @staticmethod
    def _other_tile(model: data.Model, cell: data.Cell) -> str:
        """A Failed or Does not fit tile: its status word and the date or the evidence label."""
        if cell.status == "failed":
            words = f'<span class="miss"><b>Failed</b> <span class="dt">{_e(cell.measured)}</span></span>'
        else:
            words = '<span class="miss">Does not fit <span class="nw">(estimate)</span></span>'
        return f'<a class="tile x" href="#d-{_e(cell.runtime)}-{_e(model.id)}">{words}</a>'

    def _cell(self, model: data.Model, cell: data.Cell) -> str:
        attrs: list[tuple[str, str | None]] = [("role", "cell"), ("data-engine", cell.runtime),
                                               ("data-status", cell.status)]
        if cell.status == "runs":
            facts = cell.recipe
            if facts.decode is not None:
                attrs.append(("data-speed", f"{facts.decode:g}"))
            attrs += [("data-ctx", str(facts.context // 1024)), ("data-cap", str(facts.memory_cap)),
                      ("data-think", facts.thinking), ("data-tools", "no" if facts.chat_only else "yes"),
                      ("data-use", " ".join(cell.uses))]
            if cell.fastest is not None:
                attrs.append(("data-fast", None))
            if (model.id, cell.runtime) == self.matrix.default:
                attrs.append(("data-default", None))
            if cell.setup:
                attrs.append(("data-setup", None))
            attrs += [(f"data-c-{result.client}", result.cls) for result in cell.clients]
            attrs += [(f"data-o-{key}", str(place)) for key, place in cell.positions]
        attrs.append(("data-search", data.search_text(model, cell, self.labels[cell.runtime])))
        tile = self._runs_tile(model, cell) if cell.status == "runs" else self._other_tile(model, cell)
        return _start("td", attrs) + tile + '<span class="out" hidden>Not in this view</span></td>'

    def _row(self, model: data.Model) -> str:
        positions = dict(model.positions)
        attrs = [("id", f"m-{model.id}"), ("role", "row"), ("data-model", model.id),
                 ("data-o-name", str(positions["name"])), ("data-o-gap", str(positions["gap"]))]
        if model.rank is not None:
            attrs.append(("data-rank", str(model.rank)))
        header = (f'<th scope="row" role="rowheader"><span class="rk">{"" if model.rank is None else model.rank}</span> '
                  f'<span class="id">{_e(model.id)}</span>')
        if model.chat_only:
            header += ' <span class="tag">Chat only<span class="vh">, no tool calling</span></span>'
        gap = data.gap_text(model, self.labels)
        if gap is not None:
            header += f'<span class="gv"><span class="vh">; </span>{_e(gap)}</span>'
        return (_start("tr", attrs) + header + "</th>" + "".join(self._cell(model, cell) for cell in model.cells)
                + "</tr>")

    def _head(self, runtime: data.Runtime) -> str:
        """An engine's column head: its label, the version its heading gives after the label, and its Runs count."""
        runs = sum(1 for model in self.matrix.models for cell in model.cells
                   if cell.runtime == runtime.id and cell.status == "runs")
        version = runtime.heading.removeprefix(runtime.label).strip()
        return (f'<th scope="col" role="columnheader" data-engine="{_e(runtime.id)}"><span class="en">{_e(runtime.label)}'
                "</span>" + (f' <span class="ver">{_e(version)}</span>' if version else "")
                + f'<span class="n"><span class="vh">, </span>{runs} of {len(self.matrix.models)} run</span></th>')

    def matrix_table(self) -> str:
        found = data.view(self.matrix, "", data.Filters(), None)
        models = {model.id: model for model in self.matrix.models}
        heads = [f'<th scope="col" role="columnheader" class="mh">Top {self.matrix.top} rank and model</th>',
                 *(self._head(runtime) for runtime in self.matrix.runtimes)]
        divider = (_start("tr", [("id", "grp"), ("class", "grp"), ("role", "row")]
                          + ([("hidden", None)] if found.divider is None else []))
                   + f'<th colspan="{len(self.matrix.runtimes) + 1}" scope="rowgroup" role="rowheader">'
                   + f"Not in the Top {self.matrix.top}</th></tr>")
        rows = []
        for model_id in found.rows:
            if model_id == found.divider:
                rows.append(divider)
            rows.append(self._row(models[model_id]))
        if found.divider is None:
            rows.append(divider)
        caption = data.caption_line(data.USE_CASES[0], data.USE_CASES[0].sort, self.matrix)
        pairs = len(self.matrix.models) * len(self.matrix.runtimes)
        return "\n".join([
            '<h2 class="vh">Models and engines</h2>',
            _start("div", [("class", "tbl")]) + _start("table", [
                ("id", "matrix"), ("role", "table"), ("tabindex", "-1"), ("data-pairs", str(pairs)), ("data-use", ""),
                ("data-sort", data.USE_CASES[0].sort)]),
            f'<caption id="caption" class="vh">{_e(caption)}</caption>',
            '<thead role="rowgroup"><tr role="row">' + "".join(heads) + "</tr></thead>", '<tbody role="rowgroup">',
            *rows, "</tbody>", "</table></div>"])

    def _rule(self, use: data.UseCase) -> str:
        """A view's rule line; the Top 10 rule ends with a link to the footer's rank line (2026-11-19 plan ruling 11)."""
        rule = _e(data.rule_line(use, self.matrix))
        return rule + ' <a href="#top10">What the rank means</a>' if use.id == "quality" else rule

    def _pair_link(self, model_id: str, runtime: str, words: str) -> str:
        return f'<a href="#d-{_e(runtime)}-{_e(model_id)}">{_e(words)}</a>'

    def use_lists(self) -> str:
        """Each view but All models as a list (2026-11-19 plan ruling 11): its rule, its count and picks lines, and its rows."""
        models = {model.id: model for model in self.matrix.models}
        default = f", default in the {self.client_labels['opencode']} config"
        parts = ['<section id="use-cases">', '<h2 id="use-cases-title">Each view as a list</h2>']
        for use in data.USE_CASES[1:]:
            found = data.view(self.matrix, use.id, data.Filters(), None)
            parts += [f'<section class="use" data-use="{_e(use.id)}">', f"<h3>{_e(data.view_name(use, self.matrix))}</h3>",
                      f'<p class="rule">{self._rule(use)}</p>']
            if not found.rows:
                parts += [f"<p>{_e(data.EMPTY)}</p>", "</section>"]
                continue
            picks = "; ".join(f"{_e(pick.label)}: " + self._pair_link(pick.model, pick.runtime,
                                                                      f"{pick.model} on {self.labels[pick.runtime]}")
                              for pick in found.picks)
            parts += [f'<p class="count">{_e(found.count)}' + (f" Picks: {picks}." if picks else "") + "</p>", "<ol>"]
            for model_id in found.rows:
                cells = {cell.runtime: cell for cell in models[model_id].cells}
                links = []
                for _, runtime in (key for key in found.cells if key[0] == model_id):
                    facts = cells[runtime].recipe
                    speed = "speed not recorded" if facts.decode is None else f"{data.speed_text(facts.decode)} tok/s"
                    words = f"{self.labels[runtime]} {speed}, {facts.context // 1024}k"
                    link = self._pair_link(model_id, runtime, words)
                    if use.id == "opencode" and (model_id, runtime) == self.matrix.default:
                        link += _e(default)
                    links.append(link)
                parts.append(f"<li>{_e(model_id)}: {'; '.join(links)}</li>")
            parts += ["</ol>", "</section>"]
        parts.append("</section>")
        return "\n".join(parts)

    def details(self) -> str:
        parts = ['<section id="details">', '<h2 id="details-title">Details</h2>']
        for model in self.matrix.models:
            for cell in model.cells:
                name = f"{cell.runtime}-{model.id}"
                body = self._runs_details(model, cell) if cell.status == "runs" else self._other_details(model, cell)
                parts += [_start("section", [("class", "details"), ("id", f"d-{name}"), ("data-model", model.id),
                                             ("data-engine", cell.runtime)]),
                          f'<h3 id="t-{_e(name)}">{_e(model.id)} on {_e(self.labels[cell.runtime])}</h3>', *body,
                          f'<p class="back"><a href="#m-{_e(model.id)}">Back to the table</a></p>', "</section>"]
        parts.append("</section>")
        return "\n".join(parts)

    def _heading(self, runtime: str) -> str:
        return next(item.heading for item in self.matrix.runtimes if item.id == runtime)

    def _engine_item(self, model: data.Model, cell: data.Cell, current: data.Cell) -> str:
        """One engine of the model on the page's one speed scale, linked to its details unless it is this one (2026-11-19
        plan ruling 12: the parts separated by single spaces)."""
        parts = [f'<span class="en">{_e(self.labels[cell.runtime])}</span>']
        if cell.status == "runs":
            facts = cell.recipe
            parts.append(_bar(data.bar_width(facts.decode, self.top)))
            parts.append('<span class="es none">speed not recorded</span>' if facts.decode is None
                         else f'<span class="es">{_e(data.speed_text(facts.decode))} tok/s</span>')
            parts.append(f'<span class="ec">{facts.context // 1024}k</span>')
            marker = data.lead(model, cell, self.labels)
            if marker is not None:
                words = "Tied fastest" if marker[0] == "tied" else f"Fastest, {marker[0]} ahead"
                parts.append(f'<span class="el">{_e(words)}</span>')
        elif cell.status == "failed":
            parts.append(f'<span class="miss"><b>Failed</b> {_e(cell.measured)}</span>')
        else:
            parts.append('<span class="miss">Does not fit (estimate)</span>')
        body = " ".join(parts)
        if cell is current:
            return f'<li class="cur" aria-current="true"><span class="eg">{body}</span></li>'
        return f'<li><a class="eg" href="#d-{_e(cell.runtime)}-{_e(model.id)}">{body}</a></li>'

    def _engines(self, model: data.Model, cell: data.Cell) -> list[str]:
        """The Engines heading, the scale line (left out when no Runs cell has a speed, as the key's bars part is) and the
        model's engines in ADAPTERS order."""
        scale = ([] if self.top is None else
                 [f"<p class=\"scale\">Decode speed on the page's scale, 0 to {_e(data.speed_text(self.top))} tok/s.</p>"])
        return ["<h4>Engines</h4>", *scale, '<ul class="engines">',
                *(self._engine_item(model, other, cell) for other in model.cells), "</ul>"]

    def _note(self, model: data.Model, cell: data.Cell) -> str:
        where = f"row {model.id}, {self.labels[cell.runtime]}, note: "
        return f'<p class="note">{inline(cell.note, data.SELECTION, self.links, where)}</p>'

    def _coverage_link(self) -> str:
        return f'<a href="{_e(self.links.repo_link(data.SELECTION, "runtime-coverage"))}">Runtime coverage table</a>'

    def _other_details(self, model: data.Model, cell: data.Cell) -> list[str]:
        """The details of a Failed or Does not fit cell: no commands, the note quoted whole (Details)."""
        if cell.status == "failed":
            status = (f"Failed on {cell.measured} (measured on {self.hardware}, {self._heading(cell.runtime)}).")
        else:
            status = "Does not fit (estimate)."
        engines = self._engines(model, cell)
        if not any(other.status == "runs" for other in model.cells):
            engines.append("<p>No engine runs this model on this laptop yet.</p>")
        return [f'<p class="status-line">{_e(status)}</p>', "<h4>Why it does not run</h4>", self._note(model, cell),
                *engines, f'<p class="links">{self._coverage_link()}</p>']

    def _status_line(self, model: data.Model, cell: data.Cell) -> str:
        """The Runs status line over the model's Runs cells with a speed (Details dialog)."""
        facts = cell.recipe
        context = f"{facts.context // 1024}k"
        if facts.decode is None:
            return f"Speed not recorded; {context} context."
        head = f"{data.speed_text(facts.decode)} tok/s with {context} context."
        runs = [other for other in model.cells if other.status == "runs"]
        if len(runs) == 1:
            return f"{head} Runs only on {self.labels[cell.runtime]}."
        speeds = [(Fraction(str(other.recipe.decode)), other.recipe.decode, other.runtime) for other in runs
                  if other.recipe.decode is not None]
        if len(speeds) < 2:
            return head
        top = max(exact for exact, _, _ in speeds)
        top_text = data.speed_text(next(value for exact, value, _ in speeds if exact == top))
        leaders = [self.labels[runtime] for exact, _, runtime in speeds if exact == top]
        marker = data.lead(model, cell, self.labels)
        if marker is None:
            verb = "is" if len(leaders) == 1 else "are"
            return f"{head} {data._joined(leaders)} {verb} faster: {top_text} tok/s."
        if marker[0] == "tied":
            others = [label for label in leaders if label != self.labels[cell.runtime]]
            return f"{head} Tied fastest with {data._joined(others)} at {top_text} tok/s."
        second = max(exact for exact, _, _ in speeds if exact != top)
        second_text = data.speed_text(next(value for exact, value, _ in speeds if exact == second))
        behind = [self.labels[runtime] for exact, _, runtime in speeds if exact == second]
        return (f"{head} Fastest of {len(speeds)} engines, {marker[0]} ahead of {data._joined(behind)} "
                f"({second_text} tok/s).")

    @staticmethod
    def _command(command: str, hidden: bool = True) -> str:
        """A command as one span per space-separated token, its copy button directly after it, which copies the plain
        command; the script shows the button."""
        tokens = " ".join(f"<span>{_e(token)}</span>" for token in command.split(" "))
        button = _start("button", [("type", "button"), ("class", "copy"), ("data-copy", command)]
                        + ([("hidden", None)] if hidden else []))
        return f'<span class="cmdline"><code class="cmd">{tokens}</code>{button}Copy</button></span>'

    def _prerequisites(self) -> str:
        """The prerequisites line under the commands, in the dialog and on the cards (Copy)."""
        return (f'<p class="pre">Run both in a clone of this repository; llmctl needs Docker and the NVIDIA Container '
                f'Toolkit: <a href="{_e(self.links.repo_link(README, "quick-start"))}">How to start a model</a>, '
                f'<a href="{_e(self.links.repo_link("docs/host-setup.md"))}">Host setup</a>.</p>')

    def _thinking(self, cell: data.Cell) -> str:
        facts = cell.recipe
        if facts.thinking == "toggle":
            kind = f"Toggle, {'on' if facts.default == 'on' else 'off'} by default"
        elif facts.thinking == "effort":
            kind = f"Effort levels: {', '.join(facts.efforts)}"
        else:
            kind = data.THINKING[facts.thinking]
        supported = {result.client for result in cell.clients if result.cls != "not-supported"}
        items = [f"<li>{_e(kind)}</li>"] + [f"<li>{_e(self.client_labels[client])}: {_e(line)}</li>"
                                              for client, line in facts.thinking_texts if client in supported]
        return '<ul class="thinking">\n' + "\n".join(items) + "\n</ul>"

    def _client_line(self, result: data.ClientResult) -> str:
        """A client's result in words (Copy, dialog client lines) in one span after its mark, as in v2."""
        letter = next(client.letter for client in self.matrix.clients if client.id == result.client)
        label = self.client_labels[result.client]
        latest = f"(latest {result.latest})"
        tally = f"{result.passed_runs} of {result.runs} runs passed {latest}"
        words = {"passed": f"{label}: passed, " + (f"{result.runs} of {result.runs} runs {latest}"
                                                    if result.passed_runs == result.runs else tally) + ".",
                 "known-issue": f"{label}: known issue, {tally}; the coverage note below describes it.",
                 "failed": f"{label}: failed, {tally}; the runtime coverage table does not describe it yet.",
                 "not-tested": f"{label}: not tested yet.",
                 "not-supported": f"{label}: not supported; llmctl writes no {label} entry for this pair."}[result.cls]
        line = _e(words)
        if result.not_reached is not None and result.cls in ("known-issue", "failed"):
            line += f" Not reached: <code>{_e(result.not_reached)}</code> failed."
        return f"<li>{_lamp(result.cls, letter)} <span>{line}</span></li>"

    @staticmethod
    def _reasons(reasons: str) -> str:
        """A reason as escaped text; over 120 characters, its first 120 in a summary and the whole in a details."""
        if len(reasons) <= 120:
            return _e(reasons)
        return (f'<details class="reason"><summary>{_e(reasons[:120])}</summary><p>{_e(reasons)}</p></details>')

    def _failure(self, failure: data.Failure) -> str:
        """`<step>` failed: <reasons>, then its records or the sentence that it has none."""
        if not failure.records:
            records = "(no record named after this step)"
        else:
            numbers = [f'<a href="{_e(self.links.repo_link(path))}">{_e(path.rsplit("/", 1)[1].split("-", 1)[0])}</a>'
                       for path in failure.records]
            records = f"({'record' if len(numbers) == 1 else 'records'} {', '.join(numbers)})"
        return f"<li><code>{_e(failure.step)}</code> failed: {self._reasons(failure.reasons)} {records}</li>"

    def _history(self, cell: data.Cell) -> str:
        """The e2e history: one item per date, newest first, holding its runs in folder order (Details)."""
        days: dict[str, list[data.CellRun]] = {}
        for cell_run in cell.history:
            days.setdefault(cell_run.run.date, []).append(cell_run)
        parts = ['<ul class="history">']
        for date, cell_runs in days.items():
            parts += [f'<li><span class="day">{_e(date)}</span>', "<ul>"]
            for cell_run in cell_runs:
                run = cell_run.run
                if run.readme is not None:
                    target = self.links.repo_link(run.readme, "the-no-rows" if run.no_rows else None)
                else:
                    target = self.links.repo_link(f"{run.path}/summary.md")
                passed = cell_run.steps - len(cell_run.failures)
                line = (f'<li><a href="{_e(target)}">{_e(run.folder)}</a>: {passed} of {cell_run.steps} steps passed')
                if run.records:
                    line += f', <a href="{_e(self.links.repo_link(run.path + "/records"))}">records</a>'
                if cell_run.failures:
                    line += '\n<ul>\n' + "\n".join(self._failure(failure) for failure in cell_run.failures) + "\n</ul>"
                parts.append(line + "</li>")
            parts.append("</ul>")
            if len(cell_runs) > 1:
                parts.append('<p class="order">Runs of one date are listed by folder name; the summaries do not record '
                             "which ran first.</p>")
            parts.append("</li>")
        parts.append("</ul>")
        return "\n".join(parts)

    def _runs_details(self, model: data.Model, cell: data.Cell) -> list[str]:
        """The details of a Runs cell, in the order of the spec's Details dialog (R-D10)."""
        facts = cell.recipe
        gpu = "of what CUDA reports" if facts.gpu.endswith("%") else ""
        peak = "peak not recorded" if facts.ram_peak is None else f"measured peak {facts.ram_peak:g} GiB"
        speed = "not recorded" if facts.decode is None else f"{data.speed_text(facts.decode)} tok/s"
        kv = [("Decode speed", speed, ""), ("Context", f"{facts.context // 1024}k", f"{facts.context:,} tokens"),
              ("GPU memory needed", facts.gpu, gpu), ("Free RAM to start", f"{facts.memory_cap} GiB", peak)]
        grid = "".join(f"<div><dt>{_e(name)}</dt> <dd>{_e(value)}" + (f" <small>{_e(small)}</small>" if small else "")
                       + "</dd></div>" for name, value, small in kv)
        start = ['<li><p class="step">Download once</p>', self._command(facts.pull),
                 "<p>It asks before it downloads the model files and the image.</p></li>",
                 '<li><p class="step">Start</p>', self._command(facts.up),
                 "<p>Add <code>--replace</code> to stop a running server first; <code>./llmctl up</code> refuses to "
                 "start when less GPU memory or RAM is free.</p>"]
        if facts.plain_runtime != cell.runtime:
            start.append(f"<p>Without <code>--runtime</code>, <code>./llmctl up {_e(model.id)}</code> starts the "
                         f"{_e(self.labels[facts.plain_runtime])} recipe.</p>")
        start.append("</li>")
        files = f"https://huggingface.co/{facts.repo}/tree/{facts.revision}"
        doc, _, anchor = facts.doc.partition("#")
        if not anchor:
            raise SiteError(f"{facts.file}: doc {facts.doc} names no section anchor")
        links = [(f"Recipe section in {doc}", self.links.repo_link(doc, anchor)),
                 ("Recipe file", self.links.repo_link(facts.file)), ("Record folder", self.links.repo_link(facts.record)),
                 ("Runtime coverage table", self.links.repo_link(data.SELECTION, "runtime-coverage"))]
        aliases = ", ".join(f"<code>{_e(alias)}</code>" for alias in model.aliases) or "no alias"
        settings = [("Model files", f'<a href="{_e(files)}">{_e(facts.repo)} at {_e(facts.revision[:12])}, '
                                    f"{facts.size_bytes / 1e9:.1f} GB</a>"),
                    ("Aliases", aliases), ("Thinking", self._thinking(cell)),
                    ("Tools", "No, chat only" if facts.chat_only else "Yes"), ("Output cap", f"{facts.output:,} tokens"),
                    ("OpenCode model", f"<code>{_e(facts.opencode)}</code>" if facts.opencode else "No OpenCode entry"),
                    ("Served name", f"<code>{_e(facts.served)}</code>")]
        parts = [f'<p class="recipe-name">{_e(facts.name)}</p>',
                 f'<p class="status-line">{_e(self._status_line(model, cell))}</p>',
                 f'<dl class="kv">{grid}</dl>',
                 f'<p class="ev">(measured on {_e(facts.hardware)}, {_e(self._heading(cell.runtime))}, {_e(facts.date)})</p>',
                 "<h4>Start it</h4>", '<ol class="steps">', *start, "</ol>", self._prerequisites(),
                 *self._engines(model, cell),
                 "<h4>Clients</h4>", '<ul class="clients">', *(self._client_line(result) for result in cell.clients), "</ul>"]
        if cell.setup:
            parts += ["<h4>Setup checks</h4>", f"<p>Setup checks in the e2e of {_e(cell.covered)}:</p>",
                      '<ul class="setup-checks">', *(self._failure(failure) for failure in cell.setup), "</ul>"]
        parts += ["<h4>Coverage note</h4>", self._note(model, cell),
                  '<details class="more"><summary>Recipe settings</summary><dl class="settings">',
                  *(f"<dt>{_e(name)}</dt> <dd>{value}</dd>" for name, value in settings), "</dl></details>",
                  '<details class="more"><summary>Evidence and links</summary>', f"<p>{_e(facts.summary)}</p>",
                  '<ul class="links">', *(f'<li><a href="{_e(url)}">{_e(words)}</a></li>' for words, url in links), "</ul>",
                  "</details>",
                  "<h4>e2e history</h4>",
                  self._history(cell) if cell.history else "<p>No e2e run has covered this pair yet.</p>"]
        return parts

    def _run_line(self, run: data.Run) -> str:
        if run.readme is not None:
            target = self.links.repo_link(run.readme, "the-no-rows" if run.no_rows else None)
        else:
            target = self.links.repo_link(f"{run.path}/summary.md")
        line = (f'{_e(run.date)}, <a href="{_e(target)}">{_e(run.folder)}</a>: {run.passed_rows} of {run.rows} rows '
                f"passed over {_count(run.recipes, 'recipe', 'recipes')}")
        if run.orphan_rows:
            line += (", 1 row of a recipe or engine that no longer ships" if run.orphan_rows == 1 else
                     f", {run.orphan_rows} rows of recipes or engines that no longer ship")
        names = ", ".join(_e(self.client_labels[client]) for client in run.clients) or "none"
        usable = {client: version for client, version in run.versions if version is not None}
        if any(client in usable for client in run.clients):
            versions = ", ".join(f"{_e(self.client_labels[client])} <code>{_e(usable[client])}</code>"
                                 if client in usable else f"{_e(self.client_labels[client])} not recorded"
                                 for client in run.clients)
        else:
            versions = "not recorded"
        return f"<li>{line}; clients {names}; versions {versions}.</li>"

    def versions(self) -> str:
        """The footer's client versions line (Footer, line 4)."""
        found = [path for client in self.matrix.clients for _, path in client.versions]
        if not found:
            return "<p>Client versions: not recorded.</p>"
        newest = max(path.split("/")[1] for path in found)
        items = []
        for client in self.matrix.clients:
            if not client.versions:
                items.append(f"{_e(client.label)} not recorded")
            for version, path in client.versions:
                _, date, folder = path.split("/")
                item = f"{_e(client.label)} <code>{_e(version)}</code>"
                if len(client.versions) > 1:
                    item += f" ({_e(folder)})"
                if date != newest:
                    item += f" ({_e(date)})"
                items.append(item)
        return f"<p>Client versions in the e2e of {_e(newest)}: {data._joined(items)}.</p>"

    def classes(self) -> str:
        """How to read the client results: per class a sample mark, its word and its meaning; the sample is the first
        client in NAMES with that class on a Runs cell, else the first client (2026-11-19 plan ruling 9)."""
        items = []
        for cls in data.CLASSES:
            client = next((client for client in self.matrix.clients
                           if any(result.client == client.id and result.cls == cls
                                  for _, cell in self.runs for result in cell.clients)), self.matrix.clients[0])
            items.append(f"<li>{_lamp(cls, client.letter)}<span><b>{_e(CLASS_WORDS[cls].capitalize())}</b>: "
                         f"{_e(CLASS_MEANINGS[cls])}.</span></li>")
        return "\n".join(['<ul class="classes">', *items, "</ul>"])

    def footer(self) -> str:
        links = self.links
        tree = links.repo_link("")
        top = self.matrix.top
        images = data._joined([f"{_e(runtime.label)} <code>{_e(runtime.image)}</code>" for runtime in self.matrix.runtimes])
        labels = data._joined([_e(client.label) for client in self.matrix.clients])
        config, anchor = data.OPENCODE_CONFIG.split("#")
        coding = data.view_name(next(use for use in data.USE_CASES if use.id == "opencode"), self.matrix)
        parts = [
            "<footer>", '<h2 id="about">About the data</h2>',
            f'<p>Built from the repository at commit <a href="{_e(tree)}"><code>{_e(self.build.commit)}</code></a>, '
            f"committed {_e(self.committed)} UTC.</p>",
            "<p>A pair is one model on one engine: one cell of the table.</p>",
            f"<p>Images llmctl starts: {images}.</p>", self.versions(), f"<p>{self.speeds_line()}</p>",
            '<h3 id="read-clients">How to read the client results</h3>', self.classes(),
            "<p>Run counts in the client lines count every e2e run of the same model on the same engine, so they can "
            "span changes to the pair's settings.</p>",
            f"<p>The {len(self.matrix.clients)} agent clients ({labels}) are the coding agents llmctl points at the "
            f'running model; <a href="{_e(links.repo_link("docs/clients.md"))}">docs/clients.md</a> and '
            f'<a href="{_e(links.repo_link(config))}">{_e(config)}</a> describe each one.</p>',
            f"<p>{_e(data.role_words('default', self.matrix))}: <code>{_e(self.matrix.opencode_default)}</code>, the "
            f'top-level model of the <a href="{_e(links.repo_link(config, anchor))}">validated config</a>; '
            + (f"the {_e(coding)} view marks it.</p>"
               if self.matrix.default in data.view(self.matrix, "opencode", data.Filters(), None).cells
               else f"it does not meet the {_e(coding)} rule today, so that view does not show it.</p>"),
            '<h3 id="top10">What the rank means</h3>',
            f"<p>Rank is the model's place in the Top {top} of "
            f'<a href="{_e(links.repo_link(data.SELECTION, "top-10"))}">{_e(data.SELECTION)}</a>, whose '
            f'<a href="{_e(links.repo_link(data.SELECTION, "decision-summary"))}">decision summary</a> says how the '
            "ranking is made.</p>",
            "<h3>Sources</h3>",
            f'<p>Data: the recipes in <a href="{_e(links.repo_link("recipes"))}">recipes/</a>, the '
            f'<a href="{_e(links.repo_link(data.SELECTION, "runtime-coverage"))}">runtime coverage</a> and '
            f'<a href="{_e(links.repo_link(data.SELECTION, "top-10"))}">Top {top}</a> of {_e(data.SELECTION)}, the '
            f'llmctl clients e2e summaries in <a href="{_e(links.repo_link("research"))}">research/</a>, and the '
            f'<a href="{_e(links.repo_link(config, anchor))}">validated config</a> of {_e(config)}.</p>',
            "<p>The e2e runs:</p>", '<ul id="runs">', *(self._run_line(run) for run in self.matrix.runs), "</ul>"]
        if self.matrix.ranked_without_recipe:
            ranked = "; ".join(f"rank {rank}, " + inline(cell, data.SELECTION, links, f"row {rank}, Model: ")
                               for rank, cell in self.matrix.ranked_without_recipe)
            parts.append(f'<p id="ranked-without-recipe">Ranked in the Top {top}, without a shipped recipe yet: '
                         f"{ranked}.</p>")
        parts += [f"<p>License, as {README} states it:</p>", '<div id="license">',
                  *(f"<p>{inline(line, README, links)}</p>" for line in self.matrix.license), "</div>",
                  f"<p>{_e(PRIVACY)}</p>", "</footer>"]
        return "\n".join(parts)

    def template(self) -> str:
        """The built text the script copies: a rule span per view, the empty line and a role span per role key."""
        spans = [_start("span", [("data-kind", "rule"), ("data-for", use.id)]) + self._rule(use) + "</span>"
                 for use in data.USE_CASES]
        spans.append(_start("span", [("data-kind", "empty"), ("data-for", "")]) + _e(data.EMPTY)
                     + ' <button type="button" class="clear">Clear filters</button></span>')
        for role in [*data.ROLES, *(f"engine-{runtime.id}" for runtime in self.matrix.runtimes)]:
            words = data.role_words(role, self.matrix)
            spans.append(_start("span", [("data-kind", "role"), ("data-for", role),
                                         ("data-mid", words[:1].lower() + words[1:])]) + _e(words) + "</span>")
        return '<template id="phrases">\n' + "\n".join(spans) + "\n</template>"

    def body(self, script: str) -> str:
        default = data.USE_CASES[0]
        print_line = data.print_line(default, default.sort, data.Filters(), self.matrix)
        return "\n".join([
            "<body>", '<a class="skip" href="#matrix">Skip to the table</a>', self.header(), "<main>",
            f'<p class="only-no-js">{_e(NO_JS)}</p>', f'<p id="print-view" class="print-only">{_e(print_line)}</p>',
            self.matrix_table(), self.use_lists(), self.details(), '<p id="live" class="vh" aria-live="polite"></p>',
            "</main>", self.footer(), self.sheet(), '<dialog id="drawer">',
            '<div id="drawer-bar"><button type="button" id="close">Close</button></div>', '<div id="drawer-body"></div>',
            '<p id="drawer-live" class="vh" aria-live="polite"></p>', "</dialog>", self.template(), f"<script>{script}</script>", "</body>"])


def page(matrix: data.Matrix, build: Build, root: Path) -> str:
    """The whole page: one self-contained document with its style and scripts inline under a hash-based policy."""
    style, script = inline_assets((_ASSETS / "page.css").read_text(encoding="utf-8"),
                                  (_ASSETS / "page.js").read_text(encoding="utf-8"), len(matrix.runtimes))
    parts = _Page(matrix, build, root)
    return "\n".join(["<!doctype html>", '<html lang="en" class="no-js">', parts.head(style, script),
                      parts.body(script), "</html>"]) + "\n"
