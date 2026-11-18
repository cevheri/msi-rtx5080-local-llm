"""The runtime matrix page's renderer: the page itself (design D3, The page), the Markdown subset of the coverage
notes with repository links at the build's commit (design D12), and the README recipe table (D-README)."""

import base64
import datetime
import hashlib
import html
import posixpath
import re
from collections.abc import Sequence
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
TITLE = "Local LLM runtime matrix"
LEAD = "Which open-weight models run on this laptop, on which engine, how fast, and with which agent clients."
NO_JS = "Filters and use cases need JavaScript; the full matrix and each use case as a list are below."
PRIVACY = "This page sets no cookies, has no analytics and loads nothing from other sites."
STATUS_LABELS = {"runs": "Runs", "failed": "Failed", "does-not-fit": "Does not fit"}
STATUS_MEANINGS = {"runs": "a measured recipe ships for this engine; the cell gives its decode speed and context",
                   "failed": "the model was tried on this engine and did not run; the details say why",
                   "does-not-fit": "an estimate says the model does not fit this engine on this laptop"}
CLASS_WORDS = {"passed": "passed", "known-issue": "known issue", "failed": "failed", "not-tested": "not tested",
               "not-supported": "not supported"}
CLASS_MEANINGS = {"passed": "the newest e2e session passed",
                  "known-issue": "the newest session failed, and the runtime coverage table describes the failure",
                  "failed": "the newest session failed, and the runtime coverage table does not describe it yet",
                  "not-tested": "no e2e run has covered this recipe with this client",
                  "not-supported": "llmctl writes no entry for this client and recipe"}
SETUP_MARK = "Setup check failed"
SEARCH_LABEL = "Search models"
SEARCH_PLACEHOLDER = "ID, alias or build"
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


def _phrase(phrase: data.Phrase) -> str:
    if not phrase.link:
        return _e(phrase.text)
    return f'{_e(phrase.before)}<a href="#{_e(phrase.target)}">{_e(phrase.link)}</a>{_e(phrase.after)}'


def _answer(found: data.View) -> str:
    return " ".join(part for part in (_e(found.head), *(_phrase(phrase) for phrase in found.phrases)) if part)


def _count(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


class _Page:
    """The parts of one page; every repository string is escaped here, every Markdown text goes through inline."""

    def __init__(self, matrix: data.Matrix, build: Build, root: Path):
        self.matrix = matrix
        self.build = build
        self.links = Links(root, build)
        self.labels = {runtime.id: runtime.label for runtime in matrix.runtimes}
        self.client_labels = {client.id: client.label for client in matrix.clients}
        self.runs = [(model, cell) for model in matrix.models for cell in model.cells if cell.status == "runs"]
        if not self.runs:
            raise SiteError(f"{data.SELECTION}: runtime coverage: no Runs: cell, so the page has no measurement")
        self.hardware = "; ".join(sorted({cell.recipe.hardware for _, cell in self.runs}))
        self.sort_labels = dict(data.SORTS)
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

    def versions(self) -> str:
        images = ", ".join(f"{_e(runtime.label)} <code>{_e(runtime.image)}</code>" for runtime in self.matrix.runtimes)
        found = [path for client in self.matrix.clients for _, path in client.versions]
        if not found:
            clients_text = "Clients: versions not recorded."
        else:
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
            clients_text = f"Clients, as the e2e of {_e(newest)} recorded them: {', '.join(items)}."
        return f"<p>Images llmctl starts: {images}. {clients_text}</p>"

    def header(self) -> str:
        dates = [cell.recipe.date for _, cell in self.runs]
        tree = self.links.repo_link("")
        commit = self.build.commit
        facts = (f"Measured on {_e(self.hardware)} from {_e(min(dates))} to {_e(max(dates))}; clients last tested "
                 f"{_e(max(run.date for run in self.matrix.runs))}. Built from commit "
                 f'<a href="{_e(tree)}">{_e(commit[:7])}</a>, committed {_e(self.committed)} UTC.')
        return "\n".join([
            "<header>", f"<h1>{_e(TITLE)}</h1>", f'<p class="lead">{_e(LEAD)}</p>', f'<p class="facts">{facts}</p>',
            '<details id="versions" open>', "<summary>Versions</summary>", self.versions(), "</details>",
            f'<p class="links"><a href="{_e(tree)}">Repository</a> '
            f'<a href="{_e(self.links.repo_link(README, "quick-start"))}">How to start a model</a></p>', "</header>"])

    def uses(self) -> str:
        labels = [client.label for client in self.matrix.clients]
        parts = ['<fieldset id="uses" hidden aria-describedby="use-rule">', "<legend>Use case</legend>"]
        for use in data.USE_CASES:
            attrs = [("type", "radio"), ("name", "use"), ("value", use.id), ("data-sort", use.sort)]
            parts.append(f"<label>{_start('input', attrs + ([('checked', None)] if not use.id else []))} "
                         f"{_e(use.chip)}</label>")
        parts += [f'<p id="use-rule">{_e(data.rule_line(data.USE_CASES[0], labels))}</p>', "</fieldset>"]
        return "\n".join(parts)

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

    def toolbar(self) -> str:
        facts = [cell.recipe for _, cell in self.runs]
        contexts = sorted({recipe.context // 1024 for recipe in facts})
        caps = sorted({recipe.memory_cap for recipe in facts})
        present = {recipe.thinking for recipe in facts}
        return "\n".join([
            '<div id="toolbar" hidden>', '<div class="control">', f'<label for="q">{_e(SEARCH_LABEL)}</label>',
            f'<input id="q" type="search" placeholder="{_e(SEARCH_PLACEHOLDER)}">', "</div>",
            '<details id="filters">', '<summary><span id="filters-label">Filters</span></summary>',
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
            self._select("sort", "Sort", data.SORTS, selected=data.USE_CASES[0].sort),
            '<button type="button" id="clear">Clear filters</button>',
            '<button type="button" id="copy-link">Copy link to this view</button>', "</details>", "</div>"])

    def legend(self) -> str:
        letters = ", ".join(f"{client.letter} {client.label}" for client in self.matrix.clients)
        classes = "\n".join(f'<li><span class="lamp {cls}" aria-hidden="true"></span> {_e(word.capitalize())}: '
                            f"{_e(CLASS_MEANINGS[cls])}.</li>" for cls, word in CLASS_WORDS.items())
        statuses = "\n".join(f'<li><span class="status">{_e(label)}</span>: {_e(STATUS_MEANINGS[status])}.</li>'
                             for status, label in STATUS_LABELS.items())
        return "\n".join([
            '<details id="legend" open>', f"<summary>{_e('Legend: ' + letters)}</summary>",
            '<ul class="legend-classes">', classes, "</ul>", '<ul class="legend-statuses">', statuses,
            '<li><span class="tag"><span aria-hidden="true">*</span> Fastest by n%</span>: the fastest engine of a '
            "model that runs on two or more, ahead of the next fastest by that much decode speed.</li>",
            f'<li><span class="setup">{_e(SETUP_MARK)}</span>: a setup step of the recipe failed in an e2e run of '
            "its newest date; the details list it.</li>", "</ul>", f"<p>{self.speeds_line()}</p>",
            "<p>Client results count every e2e run of the same model ID on the same engine, so they can span changes "
            "to a recipe.</p>", "</details>"])

    def _tile(self, model: data.Model, cell: data.Cell) -> str:
        label = self.labels[cell.runtime]
        parts = [f'<span class="engine">{_e(label)}</span>',
                 f'<span class="status">{_e(STATUS_LABELS[cell.status])}</span>']
        if cell.status == "failed":
            parts.append(f'<span class="date">on {_e(cell.measured)}</span>')
        elif cell.status == "does-not-fit":
            parts.append('<span class="label">(estimate)</span>')
        else:
            facts = cell.recipe
            if cell.fastest is not None:
                parts.append(f'<span class="tag"><span aria-hidden="true">*</span> {_e(cell.fastest)}</span>')
            if facts.decode is None:
                parts.append('<span class="speed">speed not recorded</span>')
            else:
                parts.append(f'<span class="speed">{facts.decode:g}</span> <span class="unit">tok/s</span>')
            parts.append(f'<span class="pill">{facts.context // 1024}k</span>')
            letters = {client.id: client.letter for client in self.matrix.clients}
            lamps = " ".join(f'<span class="lamp {_e(result.cls)}"><span aria-hidden="true">{_e(letters[result.client])}'
                            f'</span><span class="vh">{_e(self.client_labels[result.client])} '
                            f"{_e(CLASS_WORDS[result.cls])},</span></span>" for result in cell.clients)
            parts.append(f'<span class="lamps">{lamps}</span>')
            if cell.setup:
                parts.append(f'<span class="setup">{_e(SETUP_MARK)}</span>')
        return f'<a class="tile" href="#d-{_e(cell.runtime)}-{_e(model.id)}">' + " ".join(parts) + "</a>"

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
            attrs += [(f"data-c-{result.client}", result.cls) for result in cell.clients]
            attrs += [(f"data-o-{key}", str(place)) for key, place in cell.positions]
        label = self.labels[cell.runtime]
        attrs.append(("data-search", data.search_text(model, cell, label)))
        return (_start("td", attrs) + self._tile(model, cell)
                + f' <span class="out" hidden><span class="engine" aria-hidden="true">{_e(label)}</span> '
                + "Not in this view</span></td>")

    def matrix_table(self) -> str:
        found = data.view(self.matrix, "", data.Filters(), None)
        models = {model.id: model for model in self.matrix.models}
        heads = ['<th scope="col" role="columnheader">Model</th>'] + [
            f'<th scope="col" role="columnheader" data-engine="{_e(runtime.id)}">{_e(runtime.heading)} '
            f'<span class="count">{_count(runtime.recipes, "recipe", "recipes")}</span></th>'
            for runtime in self.matrix.runtimes]
        rows = []
        for model_id in found.rows:
            model = models[model_id]
            positions = dict(model.positions)
            attrs = [("id", f"m-{model.id}"), ("role", "row"), ("data-model", model.id),
                     ("data-o-name", str(positions["name"])), ("data-o-gap", str(positions["gap"]))]
            if model.rank is not None:
                attrs.append(("data-rank", str(model.rank)))
            meta = ([f"Rank {model.rank}"] if model.rank is not None else []) + (["Chat only"] if model.chat_only else [])
            if model.aliases:
                meta.append(("alias " if len(model.aliases) == 1 else "aliases ") + ", ".join(model.aliases))
            header = (f'<th scope="row" role="rowheader"><span class="id">{_e(model.id)}</span>'
                      + "".join(f' <span class="meta">{_e(line)}</span>' for line in meta) + "</th>")
            rows.append(_start("tr", attrs) + header + "".join(self._cell(model, cell) for cell in model.cells) + "</tr>")
        caption = f"Each model on each engine, sorted by: {self.sort_labels[data.USE_CASES[0].sort]}."
        return "\n".join([
            '<h2 class="vh">Models and engines</h2>', '<table id="matrix" tabindex="-1" role="table">',
            f'<caption id="caption" class="vh">{_e(caption)}</caption>',
            '<thead role="rowgroup"><tr role="row">' + "".join(heads) + "</tr></thead>", '<tbody role="rowgroup">',
            *rows, "</tbody>", "</table>"])

    def use_lists(self) -> str:
        labels = [client.label for client in self.matrix.clients]
        models = {model.id: model for model in self.matrix.models}
        parts = ['<section id="use-cases">', '<h2 id="use-cases-title">Use cases</h2>']
        for use in data.USE_CASES[1:]:
            found = data.view(self.matrix, use.id, data.Filters(), None)
            parts += [f'<section class="use" data-use="{_e(use.id)}">', f"<h3>{_e(use.chip)}</h3>",
                      f'<p class="rule">{_e(data.rule_line(use, labels))}</p>', f'<p class="answer">{_answer(found)}</p>']
            if found.rows:
                parts.append("<ol>")
                for model_id in found.rows:
                    cells = {cell.runtime: cell for cell in models[model_id].cells}
                    links = []
                    for _, runtime in (key for key in found.cells if key[0] == model_id):
                        facts = cells[runtime].recipe
                        links.append(f'<a href="#d-{_e(runtime)}-{_e(model_id)}">{_e(self.labels[runtime])} '
                                     f"{_e(data._speed_text(facts.decode))}, "
                                     f"{facts.context // 1024}k</a>")
                    parts.append(f"<li>{_e(model_id)}: {'; '.join(links)}</li>")
                parts.append("</ol>")
            parts.append("</section>")
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
                          f'<p class="back"><a href="#m-{_e(model.id)}">Back to the matrix</a></p>', "</section>"]
        parts.append("</section>")
        return "\n".join(parts)

    def _heading(self, runtime: str) -> str:
        return next(item.heading for item in self.matrix.runtimes if item.id == runtime)

    def _sibling(self, model: data.Model, cell: data.Cell) -> str:
        """One line of Other engines or Runs elsewhere, linked to that cell's details."""
        label = self.labels[cell.runtime]
        if cell.status == "runs":
            words = f"{label}: Runs, {data._speed_text(cell.recipe.decode)}, {cell.recipe.context // 1024}k"
        elif cell.status == "failed":
            words = f"{label}: Failed on {cell.measured}"
        else:
            words = f"{label}: Does not fit (estimate)"
        return f'<li><a href="#d-{_e(cell.runtime)}-{_e(model.id)}">{_e(words)}</a></li>'

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
        runs = [other for other in model.cells if other.status == "runs"]
        elsewhere = (["<ul class=\"others\">", *(self._sibling(model, other) for other in runs), "</ul>"] if runs
                     else ["<p>No engine runs this model on this laptop yet.</p>"])
        return [f'<p class="status-line">{_e(status)}</p>', "<h4>Why it does not run</h4>", self._note(model, cell),
                "<h4>Runs elsewhere</h4>", *elsewhere, f'<p class="links">{self._coverage_link()}</p>']

    def _status_line(self, model: data.Model, cell: data.Cell) -> str:
        """The Runs status line over the model's Runs cells with a speed (Details)."""
        speeds = {other.runtime: other.recipe.decode for other in model.cells
                  if other.status == "runs" and other.recipe.decode is not None}
        own = cell.recipe.decode
        if own is None or len(speeds) < 2:
            return "Runs."
        top = max(speeds.values())
        leaders = [self.labels[runtime] for runtime, speed in speeds.items() if speed == top]
        n = len(speeds)
        if own != top:
            verb = "is" if len(leaders) == 1 else "are"
            return f"Runs. {data._joined(leaders)} {verb} faster: {top:g} tok/s."
        if len(leaders) > 1:
            others = [label for label in leaders if label != self.labels[cell.runtime]]
            return f"Runs. Tied fastest of {n} engines with {data._joined(others)}, at {top:g} tok/s."
        second = max(speed for speed in speeds.values() if speed != top)
        behind = [self.labels[runtime] for runtime, speed in speeds.items() if speed == second]
        ahead = cell.fastest.removeprefix("Fastest by ")
        return f"Runs. Fastest of {n} engines, {ahead} ahead of {data._joined(behind)} ({second:g} tok/s)."

    @staticmethod
    def _command(command: str) -> str:
        """A command as one span per space-separated token, and its copy button, which copies the plain command."""
        tokens = " ".join(f"<span>{_e(token)}</span>" for token in command.split(" "))
        return (f'<p><code class="cmd">{tokens}</code> '
                f'{_start("button", [("type", "button"), ("class", "copy"), ("data-copy", command), ("hidden", None)])}'
                "Copy command</button></p>")

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

    def _client_line(self, cell: data.Cell, result: data.ClientResult) -> str:
        letter = next(client.letter for client in self.matrix.clients if client.id == result.client)
        label = self.client_labels[result.client]
        counts = f"{result.passed_runs} of {result.runs} runs passed (latest {result.latest})"
        words = {"passed": f"{label}: Passed, {counts}.",
                 "known-issue": f"{label}: Known issue, {counts}; the coverage note below describes it.",
                 "failed": f"{label}: Failed, {counts}; the runtime coverage table does not describe this failure yet.",
                 "not-supported": f"{label}: Not supported: llmctl writes no {label} entry for this recipe.",
                 "not-tested": f"{label}: Not tested: no e2e run has covered this recipe with {label} yet."}[result.cls]
        line = _e(words)
        if result.not_reached is not None and result.cls in ("known-issue", "failed"):
            line += f" Not reached: <code>{_e(result.not_reached)}</code> failed."
        lamp = f'<span class="lamp {_e(result.cls)}"><span aria-hidden="true">{_e(letter)}</span></span>'
        return f"<li>{lamp} {line}</li>"

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
        """The details of a Runs cell, in the order of the spec's Details table."""
        facts = cell.recipe
        label = self.labels[cell.runtime]
        start = [self._command(facts.up), "<p>Add <code>--replace</code> to stop a running server first.</p>"]
        if facts.plain_runtime != cell.runtime:
            start.append(f"<p>Without <code>--runtime</code>, <code>./llmctl up {_e(model.id)}</code> starts the "
                         f"{_e(self.labels[facts.plain_runtime])} recipe.</p>")
        if facts.decode is None:
            speed = "not recorded"
        else:
            speed = f"{facts.decode:g} tok/s (measured on {facts.hardware}, {self._heading(cell.runtime)}, {facts.date})"
        gpu = facts.gpu + (" of the GPU memory CUDA reports" if facts.gpu.endswith("%") else "")
        peak = "not recorded" if facts.ram_peak is None else f"{facts.ram_peak:g} GiB"
        files = f"https://huggingface.co/{facts.repo}/tree/{facts.revision}"
        doc, _, anchor = facts.doc.partition("#")
        if not anchor:
            raise SiteError(f"{facts.file}: doc {facts.doc} names no section anchor")
        links = [(f"Recipe section in {doc}", self.links.repo_link(doc, anchor)),
                 ("Recipe file", self.links.repo_link(facts.file)), ("Record folder", self.links.repo_link(facts.record)),
                 ("Runtime coverage table", self.links.repo_link(data.SELECTION, "runtime-coverage"))]
        parts = [f'<p class="recipe-name">{_e(facts.name)}</p>',
                 f'<p class="status-line">{_e(self._status_line(model, cell))}</p>',
                 "<h4>Start it</h4>", *start,
                 "<h4>First time on this machine</h4>", self._command(facts.pull),
                 "<p>It asks before it downloads the model files and the image.</p>",
                 "<h4>Decode speed</h4>", f"<p>{_e(speed)}</p>",
                 "<h4>Context</h4>", f"<p>{facts.context // 1024}k ({facts.context:,} tokens)</p>",
                 "<h4>GPU need</h4>", f"<p>{_e(gpu)}; ./llmctl up refuses to start when less is free.</p>",
                 "<h4>RAM</h4>", f"<p>Container memory cap: {facts.memory_cap} GiB; ./llmctl up refuses to start when "
                                 f"less RAM is available. Measured RAM peak: {_e(peak)}.</p>",
                 "<h4>Model files</h4>", f'<p><a href="{_e(files)}">{_e(facts.repo)} at {_e(facts.revision[:12])}, '
                                         f"{facts.size_bytes / 1e9:.1f} GB</a></p>",
                 "<h4>Thinking</h4>", self._thinking(cell),
                 "<h4>Tools</h4>", f"<p>{'No, chat only' if facts.chat_only else 'Yes'}</p>",
                 "<h4>Output cap</h4>", f"<p>{facts.output:,} tokens</p>",
                 "<h4>OpenCode model</h4>",
                 f"<p><code>{_e(facts.opencode)}</code></p>" if facts.opencode else "<p>No OpenCode entry</p>",
                 "<h4>Served name</h4>", f"<p><code>{_e(facts.served)}</code></p>",
                 "<h4>Evidence</h4>", f"<p>{_e(facts.summary)}</p>",
                 "<h4>Links</h4>", '<ul class="links">',
                 *(f'<li><a href="{_e(url)}">{_e(words)}</a></li>' for words, url in links), "</ul>",
                 "<h4>Clients</h4>", '<ul class="clients">',
                 *(self._client_line(cell, result) for result in cell.clients), "</ul>"]
        if cell.setup:
            parts += ["<h4>Setup checks</h4>", f"<p>Setup checks in the e2e of {_e(cell.covered)}:</p>",
                      '<ul class="setup-checks">', *(self._failure(failure) for failure in cell.setup), "</ul>"]
        parts += ["<h4>e2e history</h4>", self._history(cell) if cell.history else "<p>No e2e run has covered this "
                  "recipe yet.</p>",
                  "<h4>Coverage note</h4>", self._note(model, cell),
                  "<h4>Other engines</h4>", '<ul class="others">',
                  *(self._sibling(model, other) for other in model.cells if other.runtime != cell.runtime), "</ul>"]
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

    def footer(self) -> str:
        links = self.links
        tree = links.repo_link("")
        parts = [
            "<footer>", "<h2>About the data</h2>",
            f'<p>Built from the repository at commit <a href="{_e(tree)}"><code>{_e(self.build.commit)}</code></a>, '
            f"committed {_e(self.committed)} UTC.</p>",
            f'<p>Data: the recipes in <a href="{_e(links.repo_link("recipes"))}">recipes/</a>, the '
            f'<a href="{_e(links.repo_link(data.SELECTION, "runtime-coverage"))}">runtime coverage</a> and '
            f'<a href="{_e(links.repo_link(data.SELECTION, "top-10"))}">Top 10</a> of {_e(data.SELECTION)}, and the '
            f'llmctl clients e2e summaries in <a href="{_e(links.repo_link("research"))}">research/</a>.</p>',
            f"<p>{self.speeds_line()}</p>", "<p>The e2e runs:</p>", '<ul id="runs">',
            *(self._run_line(run) for run in self.matrix.runs), "</ul>"]
        if self.matrix.ranked_without_recipe:
            ranked = "; ".join(f"rank {rank}, " + inline(cell, data.SELECTION, links, f"row {rank}, Model: ")
                               for rank, cell in self.matrix.ranked_without_recipe)
            parts.append(f'<p id="ranked-without-recipe">Ranked in the Top 10, without a shipped recipe yet: {ranked}.</p>')
        parts += [f"<p>License, as {README} states it:</p>", '<div id="license">',
                  *(f"<p>{inline(line, README, links)}</p>" for line in self.matrix.license), "</div>",
                  f"<p>{_e(PRIVACY)}</p>", "</footer>"]
        return "\n".join(parts)

    def template(self) -> str:
        labels = [client.label for client in self.matrix.clients]
        spans = []
        for phrase in data.phrases(self.matrix):
            content = _phrase(phrase)
            if phrase.kind == "empty":
                content += ' <button type="button" class="clear">Clear filters</button>'
            spans.append(_start("span", [("data-kind", phrase.kind), ("data-for", phrase.target or "")]) + content
                         + "</span>")
        spans += [_start("span", [("data-kind", "rule"), ("data-for", use.id)]) + _e(data.rule_line(use, labels))
                  + "</span>" for use in data.USE_CASES]
        return '<template id="phrases">\n' + "\n".join(spans) + "\n</template>"

    def body(self, script: str) -> str:
        default = data.USE_CASES[0]
        print_line = f"View: {default.chip}; sorted by: {self.sort_labels[default.sort]}."
        return "\n".join([
            "<body>", '<a class="skip" href="#matrix">Skip to the matrix</a>', self.header(), "<main>",
            f'<p class="only-no-js">{_e(NO_JS)}</p>', f'<p id="print-view" class="print-only">{_e(print_line)}</p>',
            self.uses(), f'<p id="answer" role="status">{_answer(data.view(self.matrix, "", data.Filters(), None))}</p>',
            self.toolbar(), '<div id="notice" hidden>', '<p id="notice-text"></p>',
            '<button type="button" id="dismiss">Dismiss</button>', "</div>", self.legend(), self.matrix_table(),
            self.use_lists(), self.details(), '<p id="live" class="vh" aria-live="polite"></p>', "</main>",
            self.footer(), '<dialog id="drawer">', '<button type="button" id="close">Close</button>',
            '<div id="drawer-body"></div>',
            '<p id="drawer-live" class="vh" aria-live="polite"></p>', "</dialog>", self.template(), f"<script>{script}</script>", "</body>"])


def page(matrix: data.Matrix, build: Build, root: Path) -> str:
    """The whole page: one self-contained document with its style and scripts inline under a hash-based policy."""
    style, script = inline_assets((_ASSETS / "page.css").read_text(encoding="utf-8"),
                                  (_ASSETS / "page.js").read_text(encoding="utf-8"), len(matrix.runtimes))
    parts = _Page(matrix, build, root)
    return "\n".join(["<!doctype html>", '<html lang="en" class="no-js">', parts.head(style, script),
                      parts.body(script), "</html>"]) + "\n"
