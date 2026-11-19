"""The model guide's renderer: the shell every page shares and the pages themselves (spec 8), the Markdown subset of
the coverage notes with repository links at the build's commit (design D12), and the README recipe table
(D-README)."""

import base64
import hashlib
import html
import posixpath
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, unquote

from launcher import markdown
from launcher.catalog import Catalog
from launcher.runtimes import ADAPTERS
from launcher.site import data, draw, fit, words
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


def _cell(matrix: data.Matrix, key: tuple[str, str]) -> data.Cell:
    """The Runs cell of a setup."""
    return next(cell for model in matrix.models if model.id == key[0] for cell in model.cells if cell.runtime == key[1])


def _runtime(matrix: data.Matrix, runtime: str) -> data.Runtime:
    return next(item for item in matrix.runtimes if item.id == runtime)


def chips_html(name: str, values: Sequence, labels: Mapping, selected, legend: str, *, swatch: str | None = None,
               help_html: str = "", select: str | None = None, cls: str = "field") -> str:
    """A fieldset of radio chips named `name` (one per value, labelled from `labels`, `selected` checked) under its
    legend, with an optional colour swatch, help line and, for the agent, the select that replaces the chips on
    phones."""
    row = "".join(f'<label class="chip"><input type="radio" name="{_e(name)}" value="{_e(value)}" data-k="{_e(name)}"'
                  f'{" checked" if value == selected else ""}><span>{_e(labels[value])}</span></label>'
                  for value in values)
    chips = f'<div class="chips">{row}</div>'
    if select is not None:
        chips = f'<div class="agent-chips">{chips}</div>{select}'
    mark = f'<i class="sw {_e(swatch)}" aria-hidden="true"></i>' if swatch else ""
    return f'<fieldset class="{_e(cls)}"><legend>{mark}{_e(legend)}</legend>{help_html}{chips}</fieldset>'


def map_html(text: words.MapText, track: str, bar: tuple[tuple[str, str, str | None], ...],
             moe: tuple[str, words.Rich] | None, links: Links, here: str) -> str:
    """The memory map (spec 7.4 and 8.3): the GPU track and the RAM bar with their notes, the mixture-of-experts
    question when there is one (hidden otherwise, for the script) and the key. The two drawings are one image named by
    the map's sentence; the question and the key stay outside it, so assistive technology still reaches them."""
    def notes(left: words.Rich, right: str, short: bool, side: str) -> str:
        return (f'<p class="ch-notes"><span class="{side}l">{rich_html(left, links, here)}</span>'
                f'<span class="ch-r {side}r{" short" if short else ""}">{_e(right)}</span></p>')
    if moe is None:
        question = '<details class="hmoe" hidden><summary></summary><p></p></details>'
    else:
        question = (f'<details class="hmoe"><summary>{_e(moe[0])}</summary><p>{rich_html(moe[1], links, here)}</p>'
                    "</details>")
    key = " ".join(f'<span><i class="k k-{kind}"></i>{_e(caption)}</span>'
                   for kind, caption in zip(("solid", "line", "hatch"), words.CAPTION))
    return (f'<figure class="hmap"><div class="hdraw" role="img" aria-label="{_e(text.alt)}">'
            f'<div class="ch"><p class="ch-label" aria-hidden="true"><i class="sw gpu"></i>{_e(words.GPU_TRACK)}</p>'
            f'<div class="ch-body"><div class="htrack" aria-hidden="true">{draw.track_html(track)}</div>'
            f'{notes(text.gpu_left, text.gpu_right, text.gpu_short, "g")}</div></div>'
            f'<div class="ch"><p class="ch-label" aria-hidden="true"><i class="sw ram"></i>{_e(words.RAM_TRACK)}</p>'
            f'<div class="ch-body">{draw.ram_svg(bar, "map")}'
            f'{notes(text.ram_left, text.ram_right, text.ram_short, "r")}</div></div></div>'
            f'{question}<figcaption class="hkey">{key}</figcaption></figure>')


def terminal_html(title: str, label: str, lines: Sequence[str]) -> str:
    """A terminal drawn without script (spec 7.6): comment lines and command lines, the prompt hidden from assistive
    technology; page.js turns the command lines into copy buttons."""
    body = []
    for line in lines:
        if line.startswith("# "):
            body.append(f'<div class="c">{_e(line)}</div>')
        elif line.startswith("$ "):
            body.append(f'<div class="x"><span class="p" aria-hidden="true">$</span>{_e(line[2:])}</div>')
        else:
            raise ValueError(f"launcher/site/render.py: a terminal line is neither a comment nor a command: {line!r}")
    hints = "".join(f'<span class="{kind}">{_e(hint)}</span>' for kind, hint in zip(("click", "tap"), words.COPY_HINTS))
    return (f'<div class="term"><div class="th"><span class="tt">{_e(title)}</span>'
            f'<span class="hint-copy">{hints}</span></div>'
            f'<div class="tb" tabindex="0" role="region" aria-label="{_e(label)}"><div class="tl">{"".join(body)}'
            "</div></div></div>")


@dataclass(frozen=True)
class _Home:
    """What the home's blocks share: the repository, the rules' bench, the names, the laptop and our pick there."""
    matrix: data.Matrix
    build: Build
    links: Links
    bench: fit.Bench
    labels: words.Labels
    pick: fit.Pick


def _speed_unit(decode: float | None) -> str:
    """The unit after a speed, hidden when it is not measured (H9, P17); the script shows or hides it per setup."""
    hidden = " hidden" if decode is None else ""
    return f'<span class="spu"{hidden}> <span class="unit">{_e(words.SPEED_UNIT)}</span></span>'


def hero_html(home: _Home) -> str:
    """The hero (spec 7.3): the question, the controls (script only) and the laptop's answer."""
    bench, labels, machine, agent = home.bench, home.labels, home.bench.home, fit.DEFAULT_AGENT
    agent_ids = (*bench.agents, fit.NONE)
    select = (f'<select class="agent-select" data-k="agent" aria-label="{_e(words.AGENT_LEGEND)}">'
              + "".join(f'<option value="{_e(a)}"{" selected" if a == agent else ""}>{_e(labels.agents[a])}</option>'
                        for a in agent_ids) + "</select>")
    inputs = (
        '<div class="inputs" id="inputs">'
        + chips_html("gpu", fit.GPUS, {g: words.gpu_chip(g) for g in fit.GPUS}, machine.gpu, words.GPU_LEGEND,
                     swatch="gpu", help_html=f'<p class="help">{rich_html(words.GPU_HELP, home.links, HOME)}</p>')
        + chips_html("ram", fit.RAMS, {r: words.ram_chip(r) for r in fit.RAMS}, machine.ram, words.RAM_LEGEND,
                     swatch="ram", help_html=f'<p class="help">{rich_html(words.RAM_HELP, home.links, HOME)}</p>')
        + chips_html("agent", agent_ids, labels.agents, agent, words.AGENT_LEGEND, cls="field field-agent",
                     select=select, help_html=f'<p class="help help-wide">{_e(words.agent_help(len(bench.agents)))}</p>')
        + f'<p class="fh-state" id="fh-state"><span class="st">{_e(words.STATE_UNTOUCHED)}</span> '
        f'<button type="button" class="linkish" id="reset" hidden>{_e(words.RESET)}</button></p></div>')
    ask = (f'<div class="ask"><h1 id="q">{_e(words.H1)}</h1><p class="fh-lead">{_e(words.LEAD)}</p>{inputs}'
           f'<p class="nojs-note">{_e(words.NOJS_NOTE)}</p>'
           f'<details class="os"><summary>{_e(words.OS_SUMMARY)}</summary>'
           f'<p>{rich_html(words.OS_BODY, home.links, HOME)}</p></details></div>')
    pick = home.pick
    setup, verdict = pick.setup, pick.verdict
    facts = _cell(home.matrix, setup.key).recipe
    names = [labels.models[model] for model in fit.models_as_measured(bench, machine)]
    reason = words.reason(labels, pick, fit.default_verdict(bench, machine), fit.change(setup, verdict), names,
                          fit.more_memory(bench, machine), agent)
    question = words.moe_question(setup)
    moe = None if question is None else (question, words.moe_answer(setup.runtime))
    text = words.map_text(labels, setup, verdict, machine, True, False)
    speed, context, download = words.FACT_LABELS
    agents = words.agents_line(_cell(home.matrix, setup.key).clients, labels, setup.chat_only, setup.runtime, agent)
    answer = (
        f'<div class="answer" id="answer" role="region" aria-label="{_e(words.ANSWER_LABEL)}" tabindex="-1">'
        f'<p class="fh-verdict" aria-live="polite">{_e(words.verdict_line(True, len(names), labels.total))}</p>'
        f'<h2 class="pick-h" tabindex="-1">'
        f'{_e(words.pick_heading(words.title(labels, setup.model, setup.runtime, setup.default_engine)))}</h2>'
        f'<p class="reason">{_e(reason)}</p><dl class="hfacts">'
        f'<div><dt>{_e(speed)}</dt><dd><span class="f-sp">{_e(words.speed(facts.decode))}</span>'
        f'{_speed_unit(facts.decode)}</dd></div>'
        f'<div><dt>{_e(context)}</dt><dd class="f-cx">{_e(words.context_k(facts.context))}</dd></div>'
        f'<div><dt>{_e(download)}</dt><dd><span class="f-dl">{_e(words.gb_bytes(facts.size_bytes))}</span> '
        f'{_e(words.GB_UNIT)}</dd></div></dl>'
        + map_html(text, draw.track(setup, machine, bench.test_gpu), draw.ram_bar(setup, machine), moe, home.links,
                   HOME)
        + f'<p class="hagents">{_e(agents)}</p>'
        f'<p class="cta"><a class="btn" href="#get-started">{_e(words.CTA)}</a></p></div>')
    return f'<section class="fh" aria-labelledby="q">{ask}{answer}</section>'


def _row_html(home: _Home, model: data.Model, scale: int) -> str:
    """One row of the model list (spec 7.5, S4) for the laptop and our default agent."""
    bench, labels, machine, agent = home.bench, home.labels, home.bench.home, fit.DEFAULT_AGENT
    setup = fit.row_setup(bench, model.id, machine)
    verdict = fit.verdict(setup, machine, bench.test_gpu)
    cell = _cell(home.matrix, setup.key)
    facts = cell.recipe
    pick = home.pick.setup.model == model.id
    default = next(item for item in bench.setups if item.model == model.id and item.default_engine)
    tags = []
    if default.key == bench.default:
        tags.append(f'<span class="tag">{_e(words.TAGS["default"])}</span>')
    if model.chat_only:
        tags.append(f'<span class="tag">{_e(words.TAGS["chat"])}</span>')
    tags.append(f'<span class="tag pick-tag"{"" if pick else " hidden"}>{_e(words.TAGS["pick"])}</span>')
    hidden = words.ROW_HIDDEN
    rank = "" if model.rank is None else f'<span class="vh">{_e(hidden["rank"])}</span>{model.rank}'
    engine = words.row_engine(labels, setup) or ""
    strip = draw.strip(setup, machine, bench.test_gpu, scale)
    return (
        f'<li class="row{" pick" if pick else ""}" data-model="{_e(model.id)}"><span class="rank">{rank}</span>'
        f'<div class="name"><h3><a href="{_e(_page_href(f"/models/{model.id}/", HOME))}">{_e(model.name)}</a></h3>'
        f'<span class="eng">{_e(engine)}</span>{"".join(tags)}</div>'
        f'<div class="stripcell"><span class="stripbox" aria-hidden="true">{draw.strip_html(strip)}</span>'
        f'<span class="vh galt">{_e(words.uses(labels, setup))}</span></div>'
        f'<div class="facts"><span class="num n sp"><b>{_e(words.speed(facts.decode))}</b>{_speed_unit(facts.decode)}'
        f'</span><span class="sep">, </span><span class="num n cx"><b>{_e(words.context_k(facts.context))}</b>'
        f'<span class="vh">{_e(hidden["context"])}</span></span><span class="sep">, </span>'
        f'<span class="num n dl"><b>{_e(words.gb_bytes(facts.size_bytes))}</b> <span class="unit">{_e(words.GB_UNIT)}'
        f'</span><span class="vh">{_e(hidden["download"])}</span></span></div>'
        f'<div class="more"><span class="vh"><span class="agp">{_e(words.agents_head(agent, labels))}</span>: </span>'
        f'<span class="ag">{_e(words.agents_cell(cell.clients, labels, setup.chat_only, agent))}</span>'
        f'<span class="dot">. </span><span class="vh"><span class="vdp">{_e(words.machine_head(True))}</span>: </span>'
        f'<span class="vd{" ok" if verdict.key == "ok" else ""}">{_e(words.verdict_word(verdict.key, labels))}</span>'
        '<span class="dot">.</span></div></li>')


def shelf_html(home: _Home) -> str:
    """Our Top {top} and More models we measured (spec 7.5): the key, the sort (script only), the column heads and
    the rows in our ranking."""
    bench, labels, machine = home.bench, home.labels, home.bench.home
    heads = words.COLUMN_HEADS
    head = ('<div class="colhead" aria-hidden="true">' + f"<span>{_e(heads[0])}</span><span>{_e(heads[1])}</span>"
            f'<span class="strip-head">{_e(words.strip_head(machine, labels, True))}</span>'
            + "".join(f'<span class="n">{_e(text)}</span>' for text in heads[2:])
            + f'<span class="ag ag-head">{_e(words.agents_head(fit.DEFAULT_AGENT, labels))}</span>'
            f'<span class="vd vd-head">{_e(words.machine_head(True))}</span></div>')
    scale = draw.scale(bench, machine)
    models = {model.id: model for model in home.matrix.models}
    order = [models[model] for model in fit.model_order(bench)]
    rows = lambda ranked: "".join(_row_html(home, model, scale) for model in order
                                  if (model.rank is not None) == ranked)
    sort = ('<div class="sortbox" data-js-only hidden><label class="sel inline">'
            f'<span class="lbl">{_e(words.SORT_LABEL)}</span><select name="sort" data-k="sort">'
            + "".join(f'<option value="{_e(value)}"{" selected" if n == 0 else ""}>{_e(text)}</option>'
                      for n, (value, text) in enumerate(words.SORTS)) + "</select></label></div>")
    key = (f'<p class="key strip-key"><span>{_e(words.KEY[0])}</span><span><i class="sg"></i>{_e(words.KEY[1])}</span>'
           f'<span><i class="h"></i>{_e(words.KEY[2])}</span></p>')
    return (f'<section class="shelf-sec" id="models" aria-labelledby="h-top"><div class="shelf-top">'
            f'<h2 id="h-top" tabindex="-1">{_e(words.top_heading(labels))}</h2>'
            f'<div class="shelf-tools">{key}{sort}</div></div>{head}<ul class="shelf" role="list">{rows(True)}</ul>'
            "</section>"
            f'<section class="shelf-sec" id="more-models" aria-labelledby="h-more">'
            f'<h2 id="h-more">{_e(words.MORE_HEADING)}</h2>{head}<ul class="shelf" role="list">{rows(False)}</ul>'
            "</section>")


def get_started_html(home: _Home) -> str:
    """Get started (spec 7.6) for our pick on the laptop: the checks, the terminal, its notes, the endpoint and the
    refusals."""
    links, labels, agent = home.links, home.labels, fit.DEFAULT_AGENT
    setup, verdict = home.pick.setup, home.pick.verdict
    facts = _cell(home.matrix, setup.key).recipe
    runtime = _runtime(home.matrix, setup.runtime)
    engine = labels.engines[setup.runtime]
    checks = (*words.CHECKS, words.disk_check(facts.size_bytes, runtime.image_gb, engine))
    items = "".join(f'<li><span class="what{" disk" if n == len(checks) - 1 else ""}">{_e(what)}</span>'
                    f'<span class="how">{rich_html(how, links, HOME)}</span></li>'
                    for n, (what, how) in enumerate(checks))
    lines = [*words.head_lines(home.build.repository), *words.pull_lines(facts, engine, runtime.image_gb),
             *words.override_lines(setup, verdict, fit.change(setup, verdict), labels.first_engine),
             *words.up_lines(facts), *words.agent_lines(facts, labels, setup.runtime, agent), *words.tail_lines(facts)]
    title = words.terminal_title(labels.models[setup.model], engine)
    notes = "".join(f"<li>{rich_html(note, links, HOME)}</li>" for note in words.terminal_notes(facts, labels, agent))
    refusals = "".join(f"<div><dt><code>{_e(message)}</code></dt><dd>{_e(fix)}</dd></div>"
                       for message, fix in words.REFUSALS)
    more = "".join(f"<li>{rich_html(item, links, HOME)}</li>" for item in words.REFUSE_LINKS)
    return (
        f'<section class="sec" id="get-started" aria-labelledby="h-gs">'
        f'<h2 id="h-gs" tabindex="-1">{_e(words.GS_HEADING)}</h2><p class="sec-lead">{_e(words.GS_LEAD)}</p>'
        f'<div class="sub" id="before-you-start"><h3>{_e(words.BEFORE_HEADING)}</h3>'
        f'<p class="prose">{_e(words.BEFORE_LEAD)}</p><ol class="checks">{items}</ol>'
        f'<p class="sudo">{rich_html(words.SUDO_LINE, links, HOME)}</p></div>'
        f'<div class="sub"><h3 id="the-commands" tabindex="-1">{_e(words.COMMANDS_HEADING)}</h3>'
        f'{terminal_html(title, words.COMMANDS_LABEL, lines)}<ul class="tnotes">{notes}</ul>'
        f'<p class="ep">{rich_html(words.endpoint(runtime.port, facts.served), links, HOME)}</p></div>'
        f'<div class="sub" id="if-llmctl-refuses"><h3>{_e(words.REFUSE_HEADING)}</h3>'
        f'<p class="prose">{_e(words.REFUSE_LEAD)}</p><dl class="refuse">{refusals}</dl><ul class="links">{more}</ul>'
        "</div></section>")


def compare_html(home: _Home) -> str:
    """Compare engines (spec 7.7): every model on every engine, a Runs cell linking to its engine on the model's page
    with its speed, agents, lead and strip on the laptop, the others saying why they don't run."""
    matrix, bench, labels, machine, agent = home.matrix, home.bench, home.labels, home.bench.home, fit.DEFAULT_AGENT
    setups = {setup.key: setup for setup in bench.setups}
    setup_index = {setup.key: n for n, setup in enumerate(bench.setups)}      # the state's setup order
    scale = draw.scale(bench, machine)
    heads = (words.engine_head(runtime, n == 0) for n, runtime in enumerate(matrix.runtimes))
    head = "".join(f'<th scope="col"><b>{_e(label)}</b> {_e(detail)}</th>' for label, detail in heads)

    def cell_html(model: data.Model, runtime: data.Runtime) -> str:
        cell = next(item for item in model.cells if item.runtime == runtime.id)
        label = f'<span class="el">{_e(words.engine_cell_label(labels.engines[runtime.id]))}</span>'
        if cell.status != "runs":
            swatch, (what, why) = (("h", words.does_not_fit(labels)) if cell.status == "does-not-fit"
                                   else ("o", words.did_not_start(cell.measured)))
            return (f'<td><span class="nr"><i class="sw {swatch}" aria-hidden="true"></i><span>{label}{_e(what)} '
                    f'<span class="t2">{_e(why)}</span></span></span></td>')
        setup, facts = setups[model.id, runtime.id], cell.recipe
        lead = words.lead_words(model, cell, labels)
        strip = draw.strip(setup, machine, bench.test_gpu, scale)
        href = _page_href(f"/models/{model.id}/#engine-{runtime.id}", HOME)
        return (f'<td><a class="cell" href="{_e(href)}" data-s="{setup_index[setup.key]}">{label}<span class="sp">{_e(words.speed(facts.decode))}'
                f'{_speed_unit(facts.decode)}<span class="cx2">, {_e(words.context_k(facts.context))}</span></span>'
                f'<span class="sub2 ags">'
                f'{_e(words.compare_agents(cell.clients, labels, setup.chat_only, runtime.id, agent))}</span>'
                + ("" if lead is None else f'<span class="lead2">{_e(lead)}</span>')
                + f'<span class="stripbox" aria-hidden="true">{draw.strip_html(strip)}</span></a></td>')

    def row_html(model: data.Model) -> str:
        chat = f' <span class="tag">{_e(words.TAGS["chat"])}</span>' if model.chat_only else ""
        rank = "" if model.rank is None else f'<span class="rk">{_e(words.ranked(model.rank))}</span>'
        href = _page_href(f"/models/{model.id}/", HOME)
        return (f'<tr><th scope="row"><a href="{_e(href)}">{_e(model.name)}</a>{chat}{rank}</th>'
                + "".join(cell_html(model, runtime) for runtime in matrix.runtimes) + "</tr>")

    models = {model.id: model for model in matrix.models}
    order = [models[model] for model in fit.model_order(bench)]
    body = "".join(row_html(model) for model in order if model.rank is not None)
    body += (f'<tr class="grp"><th colspan="{len(matrix.runtimes) + 1}" scope="rowgroup">{_e(words.not_top(labels))}'
             "</th></tr>" + "".join(row_html(model) for model in order if model.rank is None))
    runs = len(bench.setups)
    lead = words.compare_lead(labels, runs, len(matrix.models) * len(matrix.runtimes))
    return (f'<section class="sec" id="compare" aria-labelledby="h-cmp">'
            f'<h2 id="h-cmp" tabindex="-1">{_e(words.COMPARE_HEADING)}</h2><p class="sec-lead">{_e(lead)}</p>'
            f'<table class="cmp"><thead><tr><th scope="col">{_e(words.MODEL_HEAD)}</th>{head}</tr></thead>'
            f"<tbody>{body}</tbody></table></section>")


def how_it_works_html(home: _Home) -> str:
    """What llmctl does on your machine (spec 7.8): the diagram with the mini map of our default on the laptop, what
    up and down do, the rules and the guide's link."""
    bench, machine = home.bench, home.bench.home
    default = next(setup for setup in bench.setups if setup.key == bench.default)
    mini = (f'<span class="mini" aria-hidden="true">'
            f'{draw.strip_html(draw.strip(default, machine, bench.test_gpu, draw.scale(bench, machine)))}'
            f'{draw.ram_svg(draw.ram_bar(default, machine), "mini")}</span>')
    nodes = words.flow(home.labels)
    flow = '<li class="ln" aria-hidden="true"></li>'.join(
        f'<li class="nd"><b>{_e(name)}</b>' + (f"<span>{_e(detail)}</span>" if detail else mini) + "</li>"
        for name, detail in nodes)
    rules = "".join(f"<li>{_e(rule)}</li>" for rule in words.RULES)
    return (f'<section class="sec" id="how-it-works" aria-labelledby="h-how">'
            f'<h2 id="h-how" tabindex="-1">{_e(words.HOW_HEADING)}</h2>'
            f'<ol class="flow" aria-label="{_e(words.FLOW_LABEL)}">{flow}</ol>'
            f'<p class="prose">{rich_html(words.HOW_PROSE, home.links, HOME)}</p>'
            f'<p class="prose">{_e(words.RULES_INTRO)}</p><ul class="rules">{rules}</ul>'
            f'<ul class="links"><li>{rich_html(words.HOW_LINK, home.links, HOME)}</li></ul></section>')


def questions_html(home: _Home) -> str:
    """Questions (spec 7.9): six closed details."""
    contexts = [_cell(home.matrix, setup.key).recipe.context for setup in home.bench.setups]
    items = "".join(f"<details><summary>{_e(question)}</summary><div><p>{rich_html(answer, home.links, HOME)}</p>"
                    "</div></details>" for question, answer in words.questions(home.labels, contexts))
    return (f'<section class="sec" id="questions" aria-labelledby="h-qa">'
            f'<h2 id="h-qa" tabindex="-1">{_e(words.QUESTIONS_HEADING)}</h2><div class="qa">{items}</div></section>')


def tested_html(home: _Home) -> str:
    """How we tested (spec 7.10): five paragraphs on the latest run, then the links."""
    run = max(home.matrix.runs, key=lambda item: (item.date, item.folder))
    paragraphs = "".join(f"<p>{rich_html(p, home.links, HOME)}</p>" for p in words.tested(home.matrix, run))
    items = "".join(f"<li>{rich_html(item, home.links, HOME)}</li>"
                    for item in words.tested_links(home.build.repository))
    return (f'<section class="sec" id="how-we-tested" aria-labelledby="h-tested">'
            f'<h2 id="h-tested" tabindex="-1">{_e(words.TESTED_HEADING)}</h2><div class="prose">{paragraphs}</div>'
            f'<ul class="links">{items}</ul></section>')


def _home_main(home: _Home) -> str:
    return (f'<main id="main" class="wrap">{hero_html(home)}{shelf_html(home)}{get_started_html(home)}'
            f"{compare_html(home)}{how_it_works_html(home)}{questions_html(home)}{tested_html(home)}</main>")


def _nav(build: Build) -> tuple[tuple[str, str], ...]:
    return ((words.NAV["models"], "/#models"), (words.NAV["get-started"], "/#get-started"),
            (words.NAV["how-we-tested"], "/#how-we-tested"), (words.NAV["github"], build.repository))


def _model_path(model: str) -> str:
    """The path of a model's page under --out (spec 3.1)."""
    return f"models/{model}/index.html"


@dataclass(frozen=True)
class _ModelPage:
    """What the blocks of one model's page share: the repository, the rules' bench, the names, the model, its page's
    path, the laptop and the setup chosen there (the model's row, spec 4.4)."""
    matrix: data.Matrix
    links: Links
    bench: fit.Bench
    labels: words.Labels
    model: data.Model
    here: str
    chosen: fit.Setup

    @property
    def runs(self) -> tuple[data.Cell, ...]:
        """The model's Runs cells, in engine order."""
        return tuple(cell for cell in self.model.cells if cell.status == "runs")

    def setup(self, runtime: str) -> fit.Setup:
        return next(setup for setup in self.bench.setups if setup.key == (self.model.id, runtime))


def _per_html(page: _ModelPage, cell: data.Cell, inner: str) -> str:
    """One running engine's block of a section: shown for the chosen engine with script, every one under its
    sub-heading without (spec 3.3 and 3.4)."""
    cur = " cur" if cell.runtime == page.chosen.runtime else ""
    return (f'<div class="per{cur}" data-rt="{_e(cell.runtime)}">'
            f'<p class="per-h">{_e(words.on_engine(page.labels.engines[cell.runtime]))}</p>{inner}</div>')


def _model_top_html(page: _ModelPage) -> str:
    """The page's top (spec 3.3, M10 to M12): the name, role, maker, summary, figures and the button on the left; the
    laptop's panel with its chips (script only), verdict, memory map and mixture-of-experts paragraph on the right."""
    model, labels, links, here, setup = page.model, page.labels, page.links, page.here, page.chosen
    machine = page.bench.home
    facts = _cell(page.matrix, setup.key).recipe
    maker = words.maker_line(model)
    summary = words.summary(model, data.default_cell(model.id, model.cells), labels)
    speed, context, download = words.FACT_LABELS
    figures = (f'<div class="figs" aria-label="{_e(words.FIGURES_ARIA)}">'
               f'<p class="fe">{_e(words.figures_label(labels, setup.runtime, setup.default_engine))}</p><dl>'
               f'<div><dt>{_e(speed)}</dt><dd><b class="f-sp">{_e(words.speed(facts.decode))}</b>'
               f'{_speed_unit(facts.decode)}</dd></div>'
               f'<div><dt>{_e(context)}</dt><dd><b class="f-cx">{_e(words.context_k(facts.context))}</b></dd></div>'
               f'<div><dt>{_e(download)}</dt><dd><b class="f-dl">{_e(words.gb_bytes(facts.size_bytes))}</b> '
               f'{_e(words.GB_UNIT)}</dd></div></dl></div>')
    left = (f'<div><h1 id="model" tabindex="-1">{_e(model.name)}</h1>'
            f'<p class="role">{_e(words.role_line(model, labels))}</p>'
            + ("" if maker is None else f'<p class="maker">{_e(maker)}</p>')
            + f'<p class="summary">{_e(summary)}</p>{figures}'
            f'<p class="cta"><a class="btn" href="#run" data-scroll>{_e(words.CTA)}</a></p></div>')
    verdict = fit.verdict(setup, machine, page.bench.test_gpu)
    chips = ('<div class="mchips" data-js-only hidden>'
             + chips_html("gpu", fit.GPUS, {g: words.gpu_chip(g) for g in fit.GPUS}, machine.gpu,
                          words.GPU_CHIPS_LEGEND, cls="chipset")
             + chips_html("ram", fit.RAMS, {r: words.ram_chip(r) for r in fit.RAMS}, machine.ram,
                          words.RAM_CHIPS_LEGEND, cls="chipset") + "</div>")
    title = words.title(labels, model.id, setup.runtime, False)            # M12 names the engine, default or not
    moe = words.moe_paragraph(setup)
    panel = (f'<section class="m-mach" id="your-machine" aria-labelledby="h-machine">'
             f'<h2 id="h-machine">{_e(words.panel_heading(True))}</h2>{chips}'
             f'<p class="verdict" aria-live="polite"><span class="on">{_e(words.panel_label(title, True))}</span>'
             f'<span class="vt">{_e(words.verdict_sentence(verdict.key, labels))}</span></p>'
             + map_html(words.map_text(labels, setup, verdict, machine, True, True),
                        draw.track(setup, machine, page.bench.test_gpu), draw.ram_bar(setup, machine), None, links, here)
             + ('<p class="moe" hidden></p>' if moe is None else f'<p class="moe">{rich_html(moe, links, here)}</p>')
             + "</section>")
    return f'<div class="m-top">{left}{panel}</div>'


def _run_html(page: _ModelPage) -> str:
    """Run it (spec 3.3, M13): the agent select (script only), the fit and agent notes, the terminal and its notes for
    the chosen setup on the laptop and our default agent."""
    labels, links, here, setup, agent = page.labels, page.links, page.here, page.chosen, fit.DEFAULT_AGENT
    machine = page.bench.home
    cell = _cell(page.matrix, setup.key)
    facts = cell.recipe
    runtime = _runtime(page.matrix, setup.runtime)
    engine = labels.engines[setup.runtime]
    verdict = fit.verdict(setup, machine, page.bench.test_gpu)
    options = "".join(f'<option value="{_e(a)}"{" selected" if a == agent else ""}>{_e(labels.agents[a])}</option>'
                      for a in (*page.bench.agents, fit.NONE))
    select = (f'<div class="runsel" data-js-only hidden><label class="sel"><span class="lbl">{_e(words.AGENT_SELECT)}'
              f'</span><select name="agent" data-k="agent">{options}</select></label></div>')
    note = words.fit_note(labels, setup, verdict, setup.runtime)
    result = next(item for item in cell.clients if item.client == agent)
    pick = fit.recommend(page.bench, machine, agent)
    agent_note = words.agent_note(result, labels, setup.chat_only, setup.runtime,
                                  None if pick is None else words.agent_pick(labels, pick, agent, True))

    def pre(value: words.Rich | None, cls: str) -> str:
        if value is None:
            return f'<p class="{cls}" hidden></p>'
        return f'<p class="{cls}">{rich_html(value, links, here)}</p>'
    lines = [*words.head_lines(links.build.repository), *words.pull_lines(facts, engine, runtime.image_gb),
             *words.override_lines(setup, verdict, fit.change(setup, verdict), labels.first_engine),
             *words.up_lines(facts), *words.agent_lines(facts, labels, setup.runtime, agent), *words.tail_lines(facts)]
    name = labels.models[setup.model]
    notes = "".join(f"<li>{rich_html(item, links, here)}</li>" for item in words.terminal_notes(facts, labels, agent))
    return (f'<section class="m-sec" aria-labelledby="run"><div class="hd">'
            f'<h2 id="run" tabindex="-1">{_e(words.RUN_HEADING)}</h2>{select}</div>'
            f'{pre(note, "pre-note")}{pre(agent_note, "pre-note agent-note")}'
            f'{terminal_html(words.terminal_title(name, engine), words.commands_label(name), lines)}'
            f'<ul class="tnotes">{notes}</ul></section>')


def _fact_html(fact: words.Rich) -> str:
    """One fact of an engine row: its numbers bold, its words as units (M15)."""
    return "<span>" + "".join(f"<b>{_e(part[1])}</b>" if isinstance(part, tuple) else f'<span class="unit">{_e(part)}'
                              "</span>" for part in fact) + "</span>"


def _why_html(page: _ModelPage, cell: data.Cell) -> str:
    """M15's Why: the note's first clause through render.inline and the note's first link (D12, P19)."""
    where = f"row {page.model.id}, {page.labels.engines[cell.runtime]}, note: "
    found = _LINK.search(cell.note)
    if found is None:
        raise SiteError(f"{data.SELECTION}: {where}a {cell.status} note needs a link to its source")
    url = _target(found.group(2), data.SELECTION, page.links)
    clause = inline(words.why_clause(cell.note), data.SELECTION, page.links, where)
    return (f'<details class="why"><summary>{_e(words.WHY)}</summary><p>{clause} '
            f'<a href="{_e(url)}">{_e(words.why_label(url))}</a></p></details>')


def _engines_html(page: _ModelPage) -> str:
    """Engines (spec 3.3, M15): one row per engine, the chosen one marked, the ones that don't run saying why. The
    list is no radio group without script, whose radios are hidden; page.js makes it one when it shows them."""
    model, labels, machine = page.model, page.labels, page.bench.home
    scale = draw.scale(page.bench, machine)
    rows = []
    for cell in model.cells:
        runtime = _runtime(page.matrix, cell.runtime)
        label, detail = words.engine_head(runtime, runtime.id == labels.first_engine)
        head = f'<div class="en"><b>{_e(label)}</b><span>{_e(detail)}</span></div>'
        anchor = f"engine-{runtime.id}"
        if cell.status != "runs":
            rows.append(f'<div class="er no" id="{_e(anchor)}">{head}<div>'
                        f'<p class="nrtext">{_e(words.not_run_sentence(cell, runtime, labels))}</p>'
                        f"{_why_html(page, cell)}</div><div></div></div>")
            continue
        setup = page.setup(cell.runtime)
        verdict = fit.verdict(setup, machine, page.bench.test_gpu)
        cur = setup.key == page.chosen.key
        lead = words.lead_words(model, cell, labels)
        facts = "".join(_fact_html(fact) for fact in words.engine_facts(labels, setup, cell.recipe.context))
        strip = draw.strip(setup, machine, page.bench.test_gpu, scale)
        rows.append(
            f'<div class="er{" cur" if cur else ""}" id="{_e(anchor)}" data-rt="{_e(runtime.id)}">{head}'
            f'<div><div class="fx">{facts}</div>' + ("" if lead is None else f'<p class="ld">{_e(lead)}</p>')
            + f'<span class="stripbox" aria-hidden="true">{draw.strip_html(strip)}</span></div>'
            f'<div class="use"><label class="usebtn" data-js-only hidden><input type="radio" name="engine" '
            f'value="{_e(runtime.id)}" data-k="engine" aria-label="{_e(words.use_engine_label(label))}"'
            f'{" checked" if cur else ""}><span>{_e(words.USE_ENGINE)}</span></label>'
            f'<p class="vw{" ok" if verdict.key == "ok" else ""}">{_e(words.verdict_word(verdict.key, labels))}</p>'
            "</div></div>")
    return (f'<section class="m-sec" id="engines" aria-labelledby="h-engines">'
            f'<h2 id="h-engines">{_e(words.ENGINES_HEADING)}</h2>'
            f'<div class="engs">{"".join(rows)}</div></section>')


def _agents_html(page: _ModelPage) -> str:
    """In your coding agent (spec 3.3, M16): per running engine, one row per agent with its result, its command and
    its thinking line."""
    labels, links, here = page.labels, page.links, page.here
    blocks = []
    for cell in page.runs:
        facts = cell.recipe
        texts = dict(facts.thinking_texts)
        rows = []
        for agent in page.bench.agents:
            result = next(item for item in cell.clients if item.client == agent)
            command = words.agent_command(facts, agent)
            rows.append(f'<div class="ar" data-a="{_e(agent)}"><h3>{_e(labels.agents[agent])}'
                        f'<span class="mine-tag">{_e(words.YOUR_AGENT)}</span></h3><div>'
                        f'<p class="res">{_e(words.result_sentence(result, labels, facts.chat_only, cell.runtime))}</p>'
                        + ("" if command is None else f'<p class="ln">{rich_html(command, links, here)}</p>')
                        + f'<p class="th2">{_e(words.thinking_line(texts[agent]))}</p></div></div>')
        blocks.append(_per_html(page, cell, f'<div class="agl">{"".join(rows)}</div>'))
    return (f'<section class="m-sec" id="agents" aria-labelledby="h-agents">'
            f'<h2 id="h-agents">{_e(words.AGENTS_HEADING)}</h2>{"".join(blocks)}</section>')


def _good_html(page: _ModelPage) -> str:
    """Good to know (spec 3.3, M17), per running engine."""
    image_gb = {runtime.id: runtime.image_gb for runtime in page.matrix.runtimes}
    blocks = []
    for cell in page.runs:
        faster = words.faster_sentence(page.model, cell, page.labels, image_gb)
        items = words.good_to_know(page.model, cell, page.labels, _runtime(page.matrix, cell.runtime).port, faster)
        blocks.append(_per_html(page, cell, '<ul class="facts-list">' + "".join(
            f"<li>{rich_html(item, page.links, page.here)}</li>" for item in items) + "</ul>"))
    return (f'<section class="m-sec" id="good-to-know" aria-labelledby="h-good">'
            f'<h2 id="h-good">{_e(words.GOOD_HEADING)}</h2>{"".join(blocks)}</section>')


def _evidence_html(page: _ModelPage) -> str:
    """Evidence (spec 3.3, M18), per running engine: the setup notes, the measurement record and the latest run's
    summary, the latest by date, then folder, as How we tested names it (spec 7.10)."""
    blocks = []
    for cell in page.runs:
        latest = max(cell.history, key=lambda item: (item.run.date, item.run.folder), default=None)
        summary = None if latest is None else f"{latest.run.path}/summary.md"
        items = words.evidence_links(cell.recipe, summary)
        blocks.append(_per_html(page, cell, '<ul class="links">' + "".join(
            f"<li>{rich_html(item, page.links, page.here)}</li>" for item in items) + "</ul>"))
    return (f'<section class="m-sec ev" id="evidence" aria-labelledby="h-evidence">'
            f'<h2 id="h-evidence">{_e(words.EVIDENCE_HEADING)}</h2>{"".join(blocks)}</section>')


def _pager_html(page: _ModelPage) -> str:
    """Previous and Next in our ranking (spec 3.3, M19), only the neighbours that exist."""
    order = fit.model_order(page.bench)
    n = order.index(page.model.id)
    links = []
    for cls, key, at in (("pv", "previous", n - 1), ("nx", "next", n + 1)):
        if 0 <= at < len(order):
            href = _page_href(f"/models/{order[at]}/", page.here)
            links.append(f'<a class="{cls}" href="{_e(href)}"><span>{_e(words.PAGER[key])}</span>'
                         f"<b>{_e(page.labels.models[order[at]])}</b></a>")
    return f'<nav class="pager" aria-label="{_e(words.PAGER["label"])}">{"".join(links)}</nav>'


def model_html(matrix: data.Matrix, bench: fit.Bench, labels: words.Labels, model: data.Model, links: Links) -> str:
    """The body of a model's page (spec 3.3): the skip link, the header, the blocks of the laptop state with every
    running engine's blocks, and the footer."""
    here = _model_path(model.id)
    page = _ModelPage(matrix, links, bench, labels, model, here, fit.row_setup(bench, model.id, bench.home))
    main = (f'<main id="main" class="wrap m-page"><p class="back">'
            f'<a href="{_e(_page_href("/#models", here))}">{_e(words.BACK)}</a></p>'
            f"{_model_top_html(page)}{_run_html(page)}{_engines_html(page)}{_agents_html(page)}{_good_html(page)}"
            f"{_evidence_html(page)}{_pager_html(page)}</main>")
    skip = f'<a class="skip" href="#model" data-scroll>{_e(words.SKIP_TO_MODEL)}</a>'
    return "\n".join([skip, header_html(here, _nav(links.build)), main, footer_html(matrix, links.build, links, here)])


def site(matrix: data.Matrix, build: Build, root: Path) -> dict[str, str]:
    """Every page of the site by its path under --out: the home, then one page per model (spec 3.1 and 10)."""
    # state imports this module for Links and _href, so it is imported here: a module-level import both ways stops
    # the import on Python 3.11 to 3.13, which evaluate state's annotations while render is half initialized
    from launcher.site import state

    links = Links(root, build)
    check_notes(matrix, links)
    labels = words.labels(matrix)
    style = check_asset("page.css", (_ASSETS / "page.css").read_text(encoding="utf-8"), "style")
    script = check_asset("page.js", (_ASSETS / "page.js").read_text(encoding="utf-8"), "script")
    bench = fit.bench(matrix)
    home = _Home(matrix, build, links, bench, labels, fit.hero_pick(bench, bench.home, fit.DEFAULT_AGENT))
    skip = f'<a class="skip" href="#models">{_e(words.SKIP_TO_MODELS)}</a>'
    body = "\n".join([skip, header_html(HOME, _nav(build)), _home_main(home), footer_html(matrix, build, links, HOME)])
    pages = {HOME: document(title=words.HOME_TITLE, description=words.home_description(labels), body=body, style=style,
                            scripts=(script,), state=state.dumps(state.home_state(matrix, bench, labels, links)))}
    for model in matrix.models:
        pages[_model_path(model.id)] = document(title=words.model_title(model.name),
                                               description=words.model_description(model.name, labels),
                                               body=model_html(matrix, bench, labels, model, links), style=style,
                                               scripts=(script,),
                                               state=state.dumps(state.model_state(matrix, bench, labels, model, links)))
    return pages
