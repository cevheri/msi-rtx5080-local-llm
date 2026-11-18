"""The runtime matrix page's data model: every source of design D4 read into frozen dataclasses, and every rule that
decides what the page shows computed here, so that render.py reads nothing else.

A missing or malformed source, or any disagreement design D5 and D13 list, raises SiteError naming the source.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from launcher import clients, markdown, ops, preflight
from launcher.catalog import Catalog, load_catalog, resolve
from launcher.errors import LlmctlError
from launcher.model import Recipe, RuntimeConfig
from launcher.runtimes import ADAPTERS

SELECTION = "docs/model-selection.md"
THINKING = {"toggle": "Toggle", "effort": "Effort levels", "always": "Always on", "none": "No thinking"}
LETTERS: dict[str, str] = {}
STATUS_WORDS = (("Runs:", "runs"), ("Failed:", "failed"), ("Does not fit:", "does-not-fit"))

_MEASURED = re.compile(r"measured (\d{4}-\d{2}-\d{2})")
_HF_LINK = re.compile(r"\]\(https://huggingface\.co/([^)\s]+)\)")
_ID_CELL = re.compile(r"`([^`]+)`")


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
    models = []
    for model_id in shipped:
        recipes = [catalog.runtimes[name].recipes[model_id] for name in ADAPTERS
                   if model_id in catalog.runtimes[name].recipes]
        aliases: list[str] = []
        for recipe in recipes:
            aliases += [alias for alias in recipe.aliases if alias not in aliases]
        cells = tuple(_cell(root, catalog, model_id, runtime, text) for runtime, text in zip(runtimes, coverage[model_id]))
        models.append(Model(id=model_id, aliases=tuple(aliases), rank=ranks.get(model_id),
                            chat_only=any(recipe.chat_only for recipe in recipes), cells=cells, positions=()))
    license_lines = tuple(line for line in _section(root, "README.md#license") if line.strip())
    if not license_lines:
        raise SiteError("README.md: the License section has no text")
    return Matrix(runtimes=runtimes,
                  clients=tuple(Client(name, clients.LABELS[name], letters[name], ()) for name in clients.NAMES),
                  models=tuple(models), runs=(), ranked_without_recipe=ranked_without_recipe, license=license_lines)
