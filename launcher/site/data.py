"""The runtime matrix page's data model: every source of design D4 read into frozen dataclasses, and every rule that
decides what the page shows computed here, so that render.py reads nothing else.

A missing or malformed source, or any disagreement design D5 and D13 list, raises SiteError naming the source.
"""

import json
import math
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path

from launcher import clients, markdown, ops, preflight
from launcher.catalog import Catalog, load_catalog, resolve
from launcher.e2e.summary import SUMMARY_HEAD, SummaryRow, is_summary, read_summary, recipe_key, step_client
from launcher.errors import LlmctlError
from launcher.model import Recipe, RuntimeConfig
from launcher.runtimes import ADAPTERS

SELECTION = "docs/model-selection.md"
THINKING = {"toggle": "Toggle", "effort": "Effort levels", "always": "Always on", "none": "No thinking"}
LETTERS: dict[str, str] = {}
STATUS_WORDS = (("Runs:", "runs"), ("Failed:", "failed"), ("Does not fit:", "does-not-fit"))
# The clients llmctl writes an entry for only on some recipes; every other client is supported for every recipe.
SUPPORTED: dict[str, Callable[[Recipe], bool]] = {"opencode": lambda recipe: recipe.opencode}
# The driver's stand-in for output that named a private value (launcher/e2e/clients.py WITHHELD); the page may not
# import the driver, and test_versions_are_read_from_client_records_only holds the two equal.
WITHHELD = "[withheld: the output named a private value after redaction]"
CLASSES = ("passed", "known-issue", "failed", "not-tested", "not-supported")
# The setup steps of a recipe whose failure marks its cell (design D6); gateway, check, restore and down are not.
CELL_SETUP_STEPS = ("up", "status", "long-grammar", "config", "probes")
# The setup steps after whose failure the driver runs no more steps of the recipe.
STOPPING_STEPS = ("up", "long-grammar")

_MEASURED = re.compile(r"measured (\d{4}-\d{2}-\d{2})")
_HF_LINK = re.compile(r"\]\(https://huggingface\.co/([^)\s]+)\)")
_ID_CELL = re.compile(r"`([^`]+)`")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_E2E_DATE = re.compile(r"e2e \d{4}-\d{2}-\d{2}:")
_CODE_SPAN = re.compile(r"`([^`]*)`")
_VERSION_RECORD = re.compile(r"\d{3,}-(.+)-version")


class SiteError(Exception):
    """A page source is missing, malformed or disagrees with another; the message starts with the source."""


@dataclass(frozen=True)
class Build:
    repository: str
    commit: str
    commit_time: int


@dataclass(frozen=True)
class Runtime:
    id: str
    label: str
    heading: str
    image: str
    recipes: int


@dataclass(frozen=True)
class Client:
    id: str
    label: str
    letter: str
    versions: tuple[tuple[str, str], ...]  # (version, run path)


@dataclass(frozen=True)
class Run:
    path: str
    date: str
    folder: str
    rows: int
    passed_rows: int
    recipes: int
    orphan_rows: int
    readme: str | None
    no_rows: bool
    records: bool
    clients: tuple[str, ...]
    versions: tuple[tuple[str, str | None], ...]  # (client, version, or None for "not recorded")


@dataclass(frozen=True)
class Failure:
    step: str
    reasons: str
    records: tuple[str, ...]


@dataclass(frozen=True)
class CellRun:
    run: Run
    steps: int
    failures: tuple[Failure, ...]


@dataclass(frozen=True)
class ClientResult:
    client: str
    cls: str
    passed_runs: int
    runs: int
    latest: str | None
    not_reached: str | None


@dataclass(frozen=True)
class ClientRun:
    """One run that counts in a client's history for one recipe (design D6)."""
    date: str
    folder: str
    passed: bool
    failed: tuple[str, ...]  # the client's failed steps, or the setup step that stopped the recipe
    not_reached: str | None  # that setup step when the client had no row because the recipe stopped before it


@dataclass(frozen=True)
class RecipeFacts:
    name: str
    file: str
    doc: str
    context: int
    decode: float | None
    date: str
    hardware: str
    record: str
    summary: str
    gpu: str
    ram_peak: float | None
    memory_cap: int
    repo: str
    revision: str
    size_bytes: int
    thinking: str
    default: str | None
    efforts: tuple[str, ...]
    output: int
    chat_only: bool
    opencode: str | None
    served: str
    up: str
    pull: str
    plain_runtime: str
    thinking_texts: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class Cell:
    model: str
    runtime: str
    status: str
    note: str
    measured: str | None
    recipe: RecipeFacts | None
    fastest: str | None
    clients: tuple[ClientResult, ...]
    history: tuple[CellRun, ...]
    covered: str | None
    setup: tuple[Failure, ...]
    uses: tuple[str, ...]
    positions: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class Model:
    id: str
    aliases: tuple[str, ...]
    rank: int | None
    chat_only: bool
    cells: tuple[Cell, ...]
    positions: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class Matrix:
    runtimes: tuple[Runtime, ...]
    clients: tuple[Client, ...]
    models: tuple[Model, ...]
    runs: tuple[Run, ...]
    ranked_without_recipe: tuple[tuple[int, str], ...]
    license: tuple[str, ...]


@dataclass(frozen=True)
class UseCase:
    id: str
    chip: str
    sort: str
    rule: str


@dataclass(frozen=True)
class Filters:
    """The filters of one view; None means no filter, context is in k and must_pass a client ID or "every"."""
    search: str = ""
    engines: tuple[str, ...] | None = None
    statuses: tuple[str, ...] | None = None
    must_pass: str | None = None
    context: int | None = None
    thinking: tuple[str, ...] | None = None
    tools: str | None = None
    ram: int | None = None


@dataclass(frozen=True)
class Phrase:
    """One part of an answer line; render wraps `link` in a link to `target`."""
    kind: str
    target: str | None
    before: str
    link: str
    after: str

    @property
    def text(self) -> str:
        return self.before + self.link + self.after


@dataclass(frozen=True)
class View:
    rows: tuple[str, ...]
    cells: tuple[tuple[str, str], ...]
    head: str
    phrases: tuple[Phrase, ...]

    @property
    def answer(self) -> str:
        return " ".join(part for part in (self.head, *(phrase.text for phrase in self.phrases)) if part)


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def shipped_catalog(root: Path) -> Catalog:
    """The catalog of root/recipes without the override folder; any problem stops the build."""
    catalog = load_catalog(root / "recipes", None)
    for name, entry in catalog.runtimes.items():
        for problem in entry.problems:
            source = _relative(root, problem.path)
            message = problem.message.replace(f"{root}/", "")
            raise SiteError(message if message.startswith(source) else f"{source}: {message}")
        if entry.runtime is None:
            raise SiteError(f"recipes/{name}: the runtime file did not load")
    return catalog


def _section(root: Path, doc: str) -> list[str]:
    path, anchor = doc.split("#", 1)
    try:
        return markdown.section(root, doc)
    except FileNotFoundError:
        raise SiteError(f"{path}: missing") from None
    except UnicodeDecodeError as exc:
        raise SiteError(f"{path}: not valid UTF-8: {exc}") from None
    except LookupError:
        raise SiteError(f"{path}: no section #{anchor}") from None


def _table(lines: list[str], first_cell: str, source: str) -> tuple[list[str], list[list[str]]]:
    try:
        return markdown.table(lines, first_cell)
    except (LookupError, ValueError) as exc:
        raise SiteError(f"{source}: {exc}") from None


def client_letters(names: Sequence[str], labels: Mapping[str, str], overrides: Mapping[str, str]) -> dict[str, str]:
    """Each client's lamp letter: its override, else the first letter of its label; a shared letter stops the build."""
    letters: dict[str, str] = {}
    for name in names:
        letter = overrides.get(name, labels[name][0].upper())
        other = next((known for known, used in letters.items() if used == letter), None)
        if other is not None:
            raise SiteError(f"launcher/site/data.py: clients {other} and {name} share the letter {letter}; "
                            f"give one of them its own letter in LETTERS")
        letters[name] = letter
    return letters


def rank_of(lines: list[str], repos: Mapping[str, tuple[str, ...]]) -> tuple[dict[str, int], tuple[tuple[int, str], ...]]:
    """Ranks by model ID from the Top 10 table, joined by the Hugging Face repositories each row links (design D4),
    and the (rank, Model cell) of each row that links no shipped recipe's repository."""
    header, rows = _table(lines, "Rank", SELECTION)
    if "Model" not in header:
        raise SiteError(f"{SELECTION}: the Top 10 table has no Model column")
    model_column = header.index("Model")
    ranks: dict[str, int] = {}
    unmatched: list[tuple[int, str]] = []
    seen: set[int] = set()
    for cells in rows:
        if not cells[0]:
            raise SiteError(f"{SELECTION}: a row with an empty Rank cell: the rank must be a positive integer")
        where = f"{SELECTION}: row {cells[0]}"
        if not re.fullmatch(r"[1-9][0-9]*", cells[0]):
            raise SiteError(f"{where}, column Rank: the rank must be a positive integer")
        rank = int(cells[0])
        if rank in seen:
            raise SiteError(f"{where}, column Rank: the rank is used twice")
        seen.add(rank)
        linked = [repo for cell in cells for repo in _HF_LINK.findall(cell)]
        ids = sorted({model_id for repo in linked for model_id in repos.get(repo, ())})
        if len(ids) > 1:
            raise SiteError(f"{where}, column Model: the row links the repositories of several models: {', '.join(ids)}")
        if not ids:
            unmatched.append((rank, cells[model_column]))
            continue
        if ids[0] in ranks:
            raise SiteError(f"{where}, column Model: {ids[0]} is ranked twice, also at rank {ranks[ids[0]]}")
        ranks[ids[0]] = rank
    return ranks, tuple(unmatched)


def _runtimes(header: list[str], catalog: Catalog) -> tuple[Runtime, ...]:
    if len(header) != 1 + len(ADAPTERS):
        raise SiteError(f"{SELECTION}: runtime coverage: the header has {len(header)} columns; it needs Model ID and "
                        f"one column per engine: {', '.join(adapter.LABEL for adapter in ADAPTERS.values())}")
    runtimes = []
    for (name, adapter), heading in zip(ADAPTERS.items(), header[1:]):
        if heading != adapter.LABEL and not heading.startswith(adapter.LABEL + " "):
            raise SiteError(f"{SELECTION}: runtime coverage: the column heading {heading!r} must be {adapter.LABEL} "
                            f"or start with {adapter.LABEL + ' '!r}")
        entry = catalog.runtimes[name]
        runtimes.append(Runtime(name, adapter.LABEL, heading, entry.runtime.image, len(entry.recipes)))
    return tuple(runtimes)


def _coverage_rows(rows: list[list[str]], shipped: list[str]) -> dict[str, list[str]]:
    by_id: dict[str, list[str]] = {}
    for cells in rows:
        match = _ID_CELL.fullmatch(cells[0])
        if not cells[0]:
            raise SiteError(f"{SELECTION}: runtime coverage: a row with an empty Model ID cell: the Model ID must be one "
                            f"code span")
        if match is None:
            raise SiteError(f"{SELECTION}: runtime coverage: row {cells[0]}: the Model ID must be one code span")
        model_id = match.group(1)
        if model_id in by_id:
            raise SiteError(f"{SELECTION}: runtime coverage: row {model_id}: the model has more than one row")
        if model_id not in shipped:
            raise SiteError(f"{SELECTION}: runtime coverage: row {model_id}: no shipped recipe has this ID")
        by_id[model_id] = cells[1:]
    missing = [model_id for model_id in shipped if model_id not in by_id]
    if missing:
        raise SiteError(f"{SELECTION}: runtime coverage: no row for {', '.join(missing)}")
    return by_id


def _recipe_facts(root: Path, catalog: Catalog, recipe: Recipe, runtime: RuntimeConfig) -> RecipeFacts:
    file = _relative(root, recipe.source.path)
    if recipe.context % 1024:
        raise SiteError(f"{file}: context {recipe.context} is not a multiple of 1,024")
    meta = recipe.clients
    if meta.thinking not in THINKING:
        raise SiteError(f"{file}: unknown thinking type {meta.thinking}")
    target = ops.client_target(recipe, runtime)
    try:
        texts = tuple(zip(clients.NAMES, clients.thinking_texts(target), strict=True))
    except ValueError as exc:
        raise SiteError(f"{file}: the clients' thinking texts do not match clients.NAMES: {exc}") from None
    try:
        plain_runtime = resolve(catalog, recipe.id, None)[0].runtime
    except LlmctlError as exc:
        raise SiteError(f"{file}: {exc}") from None
    evidence = recipe.evidence
    return RecipeFacts(
        name=recipe.name, file=file, doc=recipe.doc, context=recipe.context, decode=evidence.decode_tok_s,
        date=evidence.date, hardware=evidence.hardware, record=evidence.record, summary=evidence.summary,
        gpu=ADAPTERS[recipe.runtime].gpu_need_text(recipe, runtime), ram_peak=recipe.needs_ram_gib,
        memory_cap=recipe.memory_gib, repo=recipe.model.repo, revision=recipe.model.revision,
        size_bytes=recipe.model.size_bytes, thinking=meta.thinking, default=meta.default, efforts=meta.efforts,
        output=meta.output, chat_only=recipe.chat_only,
        opencode=f"{runtime.opencode_provider}/{recipe.id}" if recipe.opencode else None, served=target.served,
        up=ops.up_command(recipe), pull=preflight.pull_hint(recipe), plain_runtime=plain_runtime,
        thinking_texts=texts)


def _cell(root: Path, catalog: Catalog, model_id: str, runtime: Runtime, text: str) -> Cell:
    where = f"{SELECTION}: row {model_id}, column {runtime.label}"
    starts = [(word, status) for word, status in STATUS_WORDS if text.startswith(word)]
    if len(starts) != 1:
        words = ", ".join(word for word, _ in STATUS_WORDS)
        raise SiteError(f"{where}: the cell must start with exactly one of {words}")
    word, status = starts[0]
    note = text[len(word):].strip()
    entry = catalog.runtimes[runtime.id]
    recipe = entry.recipes.get(model_id)
    measured_recipe = recipe is not None and recipe.evidence.kind == "measured"
    if status == "runs" and not measured_recipe:
        raise SiteError(f"{where}: a Runs: cell needs a shipped recipe with measured evidence")
    if status != "runs" and measured_recipe:
        raise SiteError(f"{where}: a measured recipe ships, so the cell must start with Runs:")
    measured = None
    if status == "failed":
        date = _MEASURED.search(note)
        if date is None:
            raise SiteError(f"{where}: a Failed: cell needs the date of its measurement as measured YYYY-MM-DD")
        measured = date.group(1)
    if status == "does-not-fit" and "(estimate)" not in note:
        raise SiteError(f"{where}: a Does not fit: cell needs the label (estimate)")
    facts = _recipe_facts(root, catalog, recipe, entry.runtime) if status == "runs" else None
    return Cell(model=model_id, runtime=runtime.id, status=status, note=note, measured=measured, recipe=facts,
                fastest=None, clients=(), history=(), covered=None, setup=(), uses=(), positions=())


def known_steps(note: str, date: str) -> frozenset[str]:
    """The code spans of the note text after each `e2e <date>:` up to the next `e2e YYYY-MM-DD:` or the end
    (plan ruling 3); empty when the note names no such date."""
    marker = f"e2e {date}:"
    spans: set[str] = set()
    start = note.find(marker)
    while start != -1:
        rest = note[start + len(marker):]
        following = _E2E_DATE.search(rest)
        spans.update(_CODE_SPAN.findall(rest if following is None else rest[:following.start()]))
        start = note.find(marker, start + len(marker))
    return frozenset(spans)


def classify(client: str, supported: bool, history: Sequence[ClientRun], note: str) -> ClientResult:
    """The class of one client on one cell from its history and the cell's note (design D6)."""
    if not supported:
        return ClientResult(client, "not-supported", 0, 0, None, None)
    if not history:
        return ClientResult(client, "not-tested", 0, 0, None, None)
    passed_runs = sum(1 for run in history if run.passed)
    latest = max(run.date for run in history)
    failing = [run for run in history if run.date == latest and not run.passed]
    if not failing:
        return ClientResult(client, "passed", passed_runs, len(history), latest, None)
    failed = {step for run in failing for step in run.failed}
    known = f"e2e {latest}:" in note and failed <= known_steps(note, latest)
    not_reached = next((run.not_reached for run in failing if run.not_reached is not None), None)
    return ClientResult(client, "known-issue" if known else "failed", passed_runs, len(history), latest, not_reached)


@dataclass(frozen=True)
class _RunData:
    run: Run
    rows: tuple[tuple[tuple[str, str] | None, str | None, SummaryRow], ...]  # (key, client, row) per row
    record_names: tuple[str, ...]


def _e2e_summaries(root: Path) -> list[tuple[str, str, str, str]]:
    """(summary path, date, folder, text) of each e2e summary under research/, sorted by path."""
    def unlisted(exc: OSError) -> None:
        raise SiteError(f"{_relative(root, Path(exc.filename))}: {exc.strerror}")

    # os.walk with onerror, not Path.glob, which skips a folder it cannot list without a word.
    paths = [Path(folder) / "summary.md" for folder, _, names in os.walk(root / "research", onerror=unlisted)
             if "summary.md" in names] if (root / "research").is_dir() else []
    found = []
    for path in sorted(paths):
        source = _relative(root, path)
        if not is_summary(path.read_bytes()):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise SiteError(f"{source}: not valid UTF-8: {exc}") from None
        parts = source.split("/")
        if len(parts) != 4 or not _DATE.fullmatch(parts[1]):
            raise SiteError(f"{source}: an e2e summary must lie at research/<YYYY-MM-DD>/<folder>/summary.md")
        found.append((source, parts[1], parts[2], text))
    if not found:
        raise SiteError(f"research: no e2e summary (a research/<YYYY-MM-DD>/<folder>/summary.md whose first line is "
                        f"{SUMMARY_HEAD[0]})")
    return sorted(found, key=lambda item: (item[1], item[2]))


def _versions(root: Path, path: str, names: Sequence[str]) -> tuple[tuple[str, str | None], ...]:
    """(client, version or None) for each client version record of the run, in clients.NAMES order."""
    by_client: dict[str, tuple[str, str | None]] = {}
    for name in names:
        match = _VERSION_RECORD.fullmatch(name)
        if match is None or match.group(1) not in clients.NAMES:
            continue
        client, source = match.group(1), f"{path}/records/{name}"
        if client in by_client:
            raise SiteError(f"{source}: a second version record of {client}, after {by_client[client][0]}")
        try:
            record = json.loads((root / source).read_text(encoding="utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise SiteError(f"{source}: not a JSON record: {exc}") from None
        if not isinstance(record, dict) or "exit" not in record or not isinstance(record.get("stdout"), str):
            raise SiteError(f"{source}: a version record must be a JSON object with exit and stdout")
        stdout = record["stdout"]
        line = stdout.split("\n", 1)[0].strip()
        usable = type(record["exit"]) is int and record["exit"] == 0 and line and stdout != WITHHELD
        by_client[client] = (source, line if usable else None)
    return tuple((client, by_client[client][1]) for client in clients.NAMES if client in by_client)


def _runs(root: Path, catalog: Catalog) -> list[_RunData]:
    runs = []
    for source, date, folder, text in _e2e_summaries(root):
        try:
            rows = read_summary(text)
        except ValueError as exc:
            raise SiteError(f"{source}: {exc}") from None
        keyed = []
        for n, row in enumerate(rows, start=len(SUMMARY_HEAD) + 1):
            try:
                keyed.append((recipe_key(row.recipe), step_client(row.step, clients.NAMES), row))
            except ValueError as exc:
                raise SiteError(f"{source}: line {n}: {exc}") from None
        path = source.rsplit("/", 1)[0]
        keys = {key for key, _, _ in keyed if key is not None}
        orphans = sum(1 for key, _, _ in keyed if key is not None and (
            key[1] not in catalog.runtimes or key[0] not in catalog.runtimes[key[1]].recipes))
        readme = root / path / "README.md"
        try:
            no_rows = readme.is_file() and "the-no-rows" in markdown.heading_slugs(readme)
        except UnicodeDecodeError as exc:
            raise SiteError(f"{path}/README.md: not valid UTF-8: {exc}") from None
        records = root / path / "records"
        names = tuple(sorted(entry.name for entry in records.iterdir())) if records.is_dir() else ()
        used = {client for _, client, _ in keyed if client is not None}
        run = Run(path=path, date=date, folder=folder, rows=len(rows), passed_rows=sum(1 for r in rows if r.passed),
                  recipes=len(keys), orphan_rows=orphans,
                  readme=f"{path}/README.md" if readme.is_file() else None,
                  no_rows=no_rows,
                  records=records.is_dir(), clients=tuple(name for name in clients.NAMES if name in used),
                  versions=_versions(root, path, names))
        runs.append(_RunData(run, tuple(keyed), names))
    return runs


def _record_slug(name: str) -> str:
    """The slug of a record name, as the driver's _Run.record writes it (launcher/e2e/clients.py)."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-")


def _records(data: _RunData, cell_prefix: str, step: str) -> tuple[str, ...]:
    """The records of a failed step: the one named after it exactly, else those whose name starts with it, but not
    those of a longer step of the same cell in the run (a failed kimi does not cite kimi-config's record)."""
    if not data.run.records:
        return ()
    slug = _record_slug(f"{cell_prefix} {step}")
    longer = [_record_slug(f"{cell_prefix} {row.step}") for _, _, row in data.rows
              if row.recipe == cell_prefix and row.step.startswith(step + "-")]
    exact = [name for name in data.record_names if re.fullmatch(rf"\d{{3,}}-{re.escape(slug)}", name)]
    if len(exact) > 1:
        raise SiteError(f"{data.run.path}/records: two records named after the failed step {step} of {cell_prefix}: "
                        f"{', '.join(exact)}")
    names = exact or [name for name in data.record_names if re.match(rf"\d{{3,}}-{re.escape(slug)}-", name)
                      and not any(re.fullmatch(rf"\d{{3,}}-{re.escape(other)}(-.*)?", name) for other in longer)]
    return tuple(f"{data.run.path}/records/{name}" for name in names)


def _cell_results(cell: Cell, recipe: Recipe, runs: list[_RunData]) -> Cell:
    """The cell with its history, client classes, covered date and setup checks (design D6)."""
    key = (cell.model, cell.runtime)
    history: list[CellRun] = []
    client_runs: dict[str, list[ClientRun]] = {name: [] for name in clients.NAMES}
    setup_by_date: dict[str, list[Failure]] = {}
    for data in runs:
        rows = [(client, row) for row_key, client, row in data.rows if row_key == key and row.step != "gateway"]
        if not rows:
            continue
        failures = tuple(Failure(row.step, row.reasons, _records(data, row.recipe, row.step))
                         for _, row in rows if not row.passed)
        history.append(CellRun(data.run, len(rows), failures))
        setup_by_date.setdefault(data.run.date, []).extend(
            failure for failure in failures if failure.step in CELL_SETUP_STEPS)
        stopped = next((step for step in STOPPING_STEPS
                        if any(row.step == step and not row.passed for _, row in rows)), None)
        for name in clients.NAMES:
            own = [row for client, row in rows if client == name]
            if own:
                failed = tuple(row.step for row in own if not row.passed)
                client_runs[name].append(ClientRun(data.run.date, data.run.folder, not failed, failed, None))
            elif stopped is not None and name in data.run.clients:
                client_runs[name].append(ClientRun(data.run.date, data.run.folder, False, (stopped,), stopped))
    results = tuple(classify(name, SUPPORTED.get(name, lambda _: True)(recipe), client_runs[name], cell.note)
                    for name in clients.NAMES)
    if not history:
        return replace(cell, clients=results)
    covered = max(cell_run.run.date for cell_run in history)
    ordered = sorted(history, key=lambda cell_run: (cell_run.run.date, cell_run.run.folder))
    ordered.sort(key=lambda cell_run: cell_run.run.date, reverse=True)
    return replace(cell, clients=results, history=tuple(ordered), covered=covered,
                   setup=tuple(setup_by_date.get(covered, ())))


def _client_versions(runs: Sequence[Run]) -> dict[str, tuple[tuple[str, str], ...]]:
    """Each client's distinct versions on the newest date that has one, in folder order, with the run's path."""
    versions: dict[str, tuple[tuple[str, str], ...]] = {}
    for name in clients.NAMES:
        found = [(run, version) for run in runs for client, version in run.versions
                 if client == name and version is not None]
        if not found:
            versions[name] = ()
            continue
        newest = max(run.date for run, _ in found)
        listed: list[tuple[str, str]] = []
        for run, version in found:
            if run.date == newest and version not in [known for known, _ in listed]:
                listed.append((version, run.path))
        versions[name] = tuple(listed)
    return versions


OPENCODE_MINIMUM = 65_536  # docs/opencode.md#context-budget: "The practical minimum for OpenCode is 64k tokens"
LONG_CONTEXT = 131_072
ASCII_SPACE = " \t\n\r\f\v"  # the search trims these only, as page.js does (str.strip and String.trim differ)
USE_CASES = (
    UseCase("", "All results", "rank", "Every model on every engine; best quality first, then fastest."),
    UseCase("opencode", "Coding in OpenCode", "rank",
            "Recipes with tool calling and at least 64k of context, OpenCode's practical minimum, whose latest OpenCode "
            "session passed; best quality first, then fastest."),
    UseCase("quality", "Best quality", "rank", "The ranked picks that run here; rank 1 first, then fastest."),
    UseCase("fastest", "Fastest", "speed", "Every recipe that runs, fastest decode first."),
    UseCase("long-context", "Long context", "rank", "Recipes that hold 128k tokens; best quality first, then fastest."),
    UseCase("every-client", "Every client", "rank",
            "Recipes whose latest session passed in every client: {clients}; best quality first, then fastest."),
    UseCase("compare", "Compare engines", "gap",
            "Models that run on two or more engines; biggest speed difference first."),
)
SORTS = (("rank", "Quality rank, then speed"), ("speed", "Fastest first"), ("context", "Longest context first"),
         ("size", "Smallest download first"), ("gap", "Biggest engine difference first"), ("name", "Model ID, A to Z"))
CELL_KEYS = ("rank", "speed", "context", "size")
ROW_KEYS = ("gap", "name")
NO_BEST = Phrase("no-best", None, "Best quality: no ranked model is in this view.", "", "")
EMPTY = Phrase("empty", None, "No recipe matches this view.", "", "")


def fastest_tags(speeds: Mapping[str, float | None]) -> dict[str, str]:
    """The fastest tag of a model's Runs cells by runtime, without the `*` (design D7); cells without a speed take no
    part, and a model with fewer than two speeds gets none."""
    known = {runtime: Fraction(str(speed)) for runtime, speed in speeds.items() if speed is not None}
    if len(known) < 2:
        return {}
    top = max(known.values())
    leaders = [runtime for runtime, speed in known.items() if speed == top]
    if len(leaders) > 1:
        return {runtime: "Tied fastest" for runtime in leaders}
    second = max(speed for speed in known.values() if speed != top)
    n = math.floor(100 * (top / second - 1) + Fraction(1, 2))
    return {leaders[0]: f"Fastest by {n}%" if n else "Fastest by under 1%"}


def _joined(items: Sequence[str]) -> str:
    if not items:
        raise ValueError("nothing to join")
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def rule_line(use: UseCase, labels: Sequence[str]) -> str:
    """The rule line of a use case, with the client labels joined into the Every client rule."""
    return use.rule.replace("{clients}", _joined(list(labels))) if "{clients}" in use.rule else use.rule


def _classes(cell: Cell) -> dict[str, str]:
    return {result.client: result.cls for result in cell.clients}


def use_cases_of(model: Model, cell: Cell) -> tuple[str, ...]:
    """The IDs of the use cases whose rule keeps the cell, in USE_CASES order; only Runs cells match any."""
    if cell.status != "runs" or cell.recipe is None:
        return ()
    recipe, classes = cell.recipe, _classes(cell)
    matches = {
        "opencode": (not recipe.chat_only and classes.get("opencode") == "passed"
                     and recipe.context >= OPENCODE_MINIMUM),
        "quality": model.rank is not None,
        "fastest": True,
        "long-context": recipe.context >= LONG_CONTEXT,
        "every-client": bool(cell.clients) and all(cls == "passed" for cls in classes.values()),
        "compare": sum(1 for other in model.cells if other.status == "runs") >= 2,
    }
    return tuple(use.id for use in USE_CASES if use.id and matches[use.id])


def _speed_order(cell: Cell) -> tuple[bool, float]:
    decode = cell.recipe.decode
    return decode is None, -(decode or 0.0)


def cell_positions(models: Sequence[Model], engines: Sequence[str]) -> dict[tuple[str, str], tuple[tuple[str, int], ...]]:
    """Each Runs cell's place, from 0, in the rank, speed, context and size cell orders (Filters and sort); every order
    ends with the model ID and then the engine order."""
    order = list(engines)
    runs = [(model, cell) for model in models for cell in model.cells if cell.status == "runs"]

    def tail(model: Model, cell: Cell) -> tuple[str, int]:
        return model.id, order.index(cell.runtime)

    keys: dict[str, Callable[[Model, Cell], tuple]] = {
        "rank": lambda model, cell: (model.rank is None, model.rank or 0, *_speed_order(cell), *tail(model, cell)),
        "speed": lambda model, cell: (*_speed_order(cell), *tail(model, cell)),
        "context": lambda model, cell: (-cell.recipe.context, *_speed_order(cell), *tail(model, cell)),
        "size": lambda model, cell: (cell.recipe.size_bytes, *tail(model, cell)),
    }
    places: dict[tuple[str, str], list[tuple[str, int]]] = {(model.id, cell.runtime): [] for model, cell in runs}
    for key in CELL_KEYS:
        ranked = sorted(runs, key=lambda pair: keys[key](*pair))
        for place, (model, cell) in enumerate(ranked):
            places[model.id, cell.runtime].append((key, place))
    return {key: tuple(value) for key, value in places.items()}


def _gap(model: Model) -> Fraction | None:
    """The ratio of the model's fastest to its slowest speed over its Runs cells with a speed; None below two."""
    speeds = [Fraction(str(cell.recipe.decode)) for cell in model.cells
              if cell.status == "runs" and cell.recipe.decode is not None]
    return max(speeds) / min(speeds) if len(speeds) >= 2 else None


def row_positions(models: Sequence[Model]) -> dict[str, tuple[tuple[str, int], ...]]:
    """Each row's place, from 0, in the gap order (biggest ratio first, rows without one after, then the model ID) and
    the name order."""
    gaps = {model.id: _gap(model) for model in models}
    by_gap = sorted(gaps, key=lambda model_id: (gaps[model_id] is None, -(gaps[model_id] or 0), model_id))
    by_name = sorted(gaps)
    return {model_id: (("gap", by_gap.index(model_id)), ("name", by_name.index(model_id))) for model_id in by_name}


def search_text(model: Model, cell: Cell, label: str) -> str:
    """The lower-case text the search filter looks in: model ID, aliases, recipe name and engine label."""
    return " ".join([model.id, *model.aliases, *([cell.recipe.name] if cell.recipe else []), label]).lower()


def _keeps(model: Model, cell: Cell, label: str, filters: Filters) -> bool:
    """Whether every active filter keeps the cell; a filter that needs a running cell drops every other cell."""
    search = filters.search.strip(ASCII_SPACE).lower()
    if search and search not in search_text(model, cell, label):
        return False
    if filters.engines is not None and cell.runtime not in filters.engines:
        return False
    if filters.statuses is not None and cell.status not in filters.statuses:
        return False
    running = (filters.must_pass, filters.context, filters.thinking, filters.tools, filters.ram)
    if all(value is None for value in running):
        return True
    recipe = cell.recipe
    if cell.status != "runs" or recipe is None:
        return False
    classes = _classes(cell)
    if filters.must_pass == "every":
        if not all(cls == "passed" for cls in classes.values()):
            return False
    elif filters.must_pass is not None:
        if filters.must_pass not in classes:
            raise ValueError(f"unknown client {filters.must_pass!r}")
        if classes[filters.must_pass] != "passed":
            return False
    if filters.context is not None and recipe.context < filters.context * 1024:
        return False
    if filters.thinking is not None and recipe.thinking not in filters.thinking:
        return False
    if filters.tools is not None:
        if filters.tools not in ("yes", "no"):
            raise ValueError(f"unknown tools value {filters.tools!r}")
        if recipe.chat_only != (filters.tools == "no"):
            return False
    return filters.ram is None or recipe.memory_cap <= filters.ram


def _speed_text(decode: float | None) -> str:
    return "speed not recorded" if decode is None else f"{decode:g} tok/s"


def _best(model: Model, cell: Cell, label: str) -> Phrase:
    return Phrase("best", f"d-{cell.runtime}-{model.id}", "Best quality: ", f"{model.id} on {label}",
                  f" (rank {model.rank}), {_speed_text(cell.recipe.decode)}, {cell.recipe.context // 1024}k.")


def _fastest(model: Model, cell: Cell, label: str) -> Phrase:
    return Phrase("fastest", f"d-{cell.runtime}-{model.id}", "Fastest: ", f"{model.id} on {label}",
                  f", {_speed_text(cell.recipe.decode)}, {cell.recipe.context // 1024}k.")


def _gap_phrase(model: Model, labels: Mapping[str, str]) -> Phrase:
    """The Biggest difference phrase of a row with two speeds: the fastest engine first in engine order on a tie, the
    slowest last."""
    speeds = [(cell.recipe.decode, cell.runtime) for cell in model.cells
              if cell.status == "runs" and cell.recipe.decode is not None]
    fast = max(speeds, key=lambda pair: pair[0])
    slow = min(reversed(speeds), key=lambda pair: pair[0])
    return Phrase("gap", f"m-{model.id}", "Biggest difference: ", model.id,
                  f", {fast[0]:g} tok/s on {labels[fast[1]]} against {slow[0]:g} on {labels[slow[1]]}.")


def phrases(matrix: Matrix) -> tuple[Phrase, ...]:
    """The answer-line phrases the script copies: best per Runs cell of a ranked model, then fastest per Runs cell with
    a speed (models in order, engines in ADAPTERS order), then gap per row with two speeds, then no-best and empty."""
    labels = {runtime.id: runtime.label for runtime in matrix.runtimes}
    runs = [(model, cell) for model in matrix.models for cell in model.cells if cell.status == "runs"]
    return (*(_best(model, cell, labels[cell.runtime]) for model, cell in runs if model.rank is not None),
            *(_fastest(model, cell, labels[cell.runtime]) for model, cell in runs if cell.recipe.decode is not None),
            *(_gap_phrase(model, labels) for model in matrix.models if _gap(model) is not None),
            NO_BEST, EMPTY)


def _counted(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def view(matrix: Matrix, use: str, filters: Filters, sort: str | None) -> View:
    """The rows, cells and answer line of one view, as the page script computes them (design D9); sort None means
    the use case's own."""
    cases = {case.id: case for case in USE_CASES}
    if use not in cases:
        raise ValueError(f"unknown use case {use!r}")
    case = cases[use]
    key = case.sort if sort is None else sort
    if key not in dict(SORTS):
        raise ValueError(f"unknown sort {key!r}")
    labels = {runtime.id: runtime.label for runtime in matrix.runtimes}
    shown: list[tuple[Model, list[Cell]]] = []
    for model in matrix.models:
        cells = [cell for cell in model.cells
                 if (not use or use in cell.uses) and _keeps(model, cell, labels[cell.runtime], filters)]
        if cells:
            shown.append((model, cells))
    if not shown:
        return View((), (), "", (EMPTY,))

    def place(item: Model | Cell, name: str) -> int:
        return dict(item.positions)[name]

    if key in CELL_KEYS:
        def row_order(entry: tuple[Model, list[Cell]]) -> tuple[int, int]:
            found = [place(cell, key) for cell in entry[1] if cell.positions]
            return (0, min(found)) if found else (1, place(entry[0], "name"))

        shown.sort(key=row_order)
        shown = [(model, sorted(cells, key=lambda cell: (0, place(cell, key)) if cell.positions else (1, 0)))
                 for model, cells in shown]
    else:
        shown.sort(key=lambda entry: place(entry[0], key))
    runs = [(model, cell) for model, cells in shown for cell in cells if cell.status == "runs"]
    rows = tuple(model.id for model, _ in shown)
    cells = tuple((model.id, cell.runtime) for model, cells in shown for cell in cells)
    if case.sort == "gap":
        head = f"{case.chip}: {_counted(len(rows), 'model', 'models')}."
        gapped = sorted((model for model, _ in shown if _gap(model) is not None), key=lambda model: place(model, "gap"))
        return View(rows, cells, head, (_gap_phrase(gapped[0], labels),) if gapped else ())
    head = (f"{case.chip}: {_counted(len(rows), 'model', 'models')}, "
            f"{_counted(len(runs), 'recipe that runs', 'recipes that run')}.")
    if not runs:
        return View(rows, cells, head, (NO_BEST,))
    best = min(runs, key=lambda pair: place(pair[1], "rank"))
    fastest = min(runs, key=lambda pair: place(pair[1], "speed"))
    found = [_best(best[0], best[1], labels[best[1].runtime]) if best[0].rank is not None else NO_BEST]
    if fastest[1].recipe.decode is not None:
        found.append(_fastest(fastest[0], fastest[1], labels[fastest[1].runtime]))
    return View(rows, cells, head, tuple(found))


def load(root: Path) -> Matrix:
    """The page's data model from the repository at `root`."""
    catalog = shipped_catalog(root)
    shipped = sorted({recipe_id for entry in catalog.runtimes.values() for recipe_id in entry.recipes})
    header, rows = _table(_section(root, f"{SELECTION}#runtime-coverage"), "Model ID", SELECTION)
    runtimes = _runtimes(header, catalog)
    coverage = _coverage_rows(rows, shipped)
    repos: dict[str, list[str]] = {}
    for entry in catalog.runtimes.values():
        for recipe in entry.recipes.values():
            ids = repos.setdefault(recipe.model.repo, [])
            if recipe.id not in ids:
                ids.append(recipe.id)
    ranks, ranked_without_recipe = rank_of(_section(root, f"{SELECTION}#top-10"),
                                           {repo: tuple(sorted(ids)) for repo, ids in sorted(repos.items())})
    letters = client_letters(clients.NAMES, clients.LABELS, LETTERS)
    runs = _runs(root, catalog)
    models = []
    for model_id in shipped:
        recipes = [catalog.runtimes[name].recipes[model_id] for name in ADAPTERS
                   if model_id in catalog.runtimes[name].recipes]
        aliases: list[str] = []
        for recipe in recipes:
            aliases += [alias for alias in recipe.aliases if alias not in aliases]
        cells = tuple(_cell(root, catalog, model_id, runtime, text) for runtime, text in zip(runtimes, coverage[model_id]))
        cells = tuple(_cell_results(cell, catalog.runtimes[cell.runtime].recipes[model_id], runs)
                      if cell.status == "runs" else cell for cell in cells)
        tags = fastest_tags({cell.runtime: cell.recipe.decode for cell in cells if cell.status == "runs"})
        cells = tuple(replace(cell, fastest=tags.get(cell.runtime)) for cell in cells)
        model = Model(id=model_id, aliases=tuple(aliases), rank=ranks.get(model_id),
                      chat_only=any(recipe.chat_only for recipe in recipes), cells=cells, positions=())
        models.append(replace(model, cells=tuple(replace(cell, uses=use_cases_of(model, cell)) for cell in cells)))
    places = cell_positions(models, list(ADAPTERS))
    row_places = row_positions(models)
    models = [replace(model, positions=row_places[model.id],
                      cells=tuple(replace(cell, positions=places.get((model.id, cell.runtime), ()))
                                  for cell in model.cells))
              for model in models]
    license_lines = tuple(line for line in _section(root, "README.md#license") if line.strip())
    if not license_lines:
        raise SiteError("README.md: the License section has no text")
    run_list = tuple(data.run for data in runs)
    versions = _client_versions(run_list)
    return Matrix(runtimes=runtimes,
                  clients=tuple(Client(name, clients.LABELS[name], letters[name], versions[name])
                                for name in clients.NAMES),
                  models=tuple(models), runs=run_list, ranked_without_recipe=ranked_without_recipe,
                  license=license_lines)
