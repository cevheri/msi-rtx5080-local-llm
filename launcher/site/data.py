"""The model guide's data model: every source of design D4 and spec 6 read into frozen dataclasses; the machine rules
live in fit.py and the sentences in words.py.

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

import launcher
from launcher import (clients, dsh_config, hermes_config, kimi_config, markdown, ops, pi_config, preflight,
                      validate)
from launcher.catalog import Catalog, load_catalog, resolve
from launcher.e2e.summary import SUMMARY_HEAD, SummaryRow, is_summary, read_summary, recipe_key, step_client
from launcher.errors import LlmctlError
from launcher.model import Recipe, RuntimeConfig, Target
from launcher.runtimes import ADAPTERS

SELECTION = "docs/model-selection.md"
OPENCODE_CONFIG = "docs/opencode.md#the-validated-config"
THINKING = {"toggle": "Toggle", "effort": "Effort levels", "always": "Always on", "none": "No thinking"}
STATUS_WORDS = (("Runs:", "runs"), ("Failed:", "failed"), ("Does not fit:", "does-not-fit"))
# The clients llmctl writes an entry for only on some recipes; every other client is supported for every recipe.
SUPPORTED: dict[str, Callable[[Recipe], bool]] = {"opencode": lambda recipe: recipe.opencode}
# The driver's stand-in for output that named a private value (launcher/e2e/clients.py WITHHELD); the page may not
# import the driver, and test_versions_are_read_from_client_records_only holds the two equal.
WITHHELD = "[withheld: the output named a private value after redaction]"
# The setup steps of a recipe whose failure marks its cell (design D6); gateway, check, restore and down are not.
CELL_SETUP_STEPS = ("up", "status", "long-grammar", "config", "probes")
# The setup steps after whose failure the driver runs no more steps of the recipe.
STOPPING_STEPS = ("up", "long-grammar")

_MEASURED = re.compile(r"measured (\d{4}-\d{2}-\d{2})")
_HF_LINK = re.compile(r"\]\(https://huggingface\.co/([^)\s]+)\)")
_ID_CELL = re.compile(r"`([^`]+)`")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_SOURCE_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
_E2E_DATE = re.compile(r"e2e \d{4}-\d{2}-\d{2}:")
_CODE_SPAN = re.compile(r"`([^`]*)`")
_VERSION_RECORD = re.compile(r"\d{3,}-(.+)-version")
OFFLOAD_FLAGS = ("--n-cpu-moe", "--moe-cache-mib", "--enable-paged-experts")
# The first word of a recipe name that names the build, not the model (spec 6, D1): a quantization, a checkpoint size or
# a packaging word.
BUILD_TOKEN = re.compile(r"^(?:UD-)?(?:I?Q\d|MXFP|NVFP|AWQ|W\d+A\d+|FP\d|QAT$|IT$|GGUF$|\d+GB$)")
LICENSES = ("CC BY 4.0", "MIT")


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
    port: int
    image_gb: str  # as docs/host-setup.md#disk-budget writes it, such as "7.05"


@dataclass(frozen=True)
class Launch:
    """The names each agent uses for one recipe, as llmctl writes them (spec 6, D4); None where the agent has none."""
    opencode: str | None
    hermes: str
    pi: str
    dsh: str
    kimi: str | None
    kimi_file: str


@dataclass(frozen=True)
class Maker:
    """A ranked model's Vendor, Params and License cells in the Top 10 (spec 6, D8)."""
    vendor: str
    params: str
    license: str


@dataclass(frozen=True)
class Laptop:
    """The one test machine every Runs cell was measured on, and the dates of the first and last measurement (D10)."""
    name: str
    gpu_name: str
    gpu_gb: int
    ram_gib: int
    ram_free_gib: int
    power_w: int
    os: str
    first: str
    last: str


@dataclass(frozen=True)
class Client:
    id: str
    label: str
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
    offload: bool
    launch: Launch


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


@dataclass(frozen=True)
class Model:
    id: str
    aliases: tuple[str, ...]
    rank: int | None
    chat_only: bool
    cells: tuple[Cell, ...]
    name: str
    honorable: bool
    maker: Maker | None


@dataclass(frozen=True)
class Matrix:
    runtimes: tuple[Runtime, ...]
    clients: tuple[Client, ...]
    models: tuple[Model, ...]
    runs: tuple[Run, ...]
    license: tuple[str, ...]
    top: int
    opencode_default: str
    default: tuple[str, str]
    laptop: Laptop
    licenses: tuple[str, str]


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


def opencode_default(root: Path) -> str:
    """The top-level "model" of the first json block in the validated OpenCode config section."""
    lines = _section(root, OPENCODE_CONFIG)
    start = next((n for n, line in enumerate(lines) if line == "```json"), None)
    end = None if start is None else next((n for n in range(start + 1, len(lines)) if lines[n].startswith("```")), None)
    if start is None or end is None:
        raise SiteError(f"{OPENCODE_CONFIG}: the section has no complete json block")
    try:
        config = json.loads("\n".join(lines[start + 1:end]))
    except json.JSONDecodeError as exc:
        raise SiteError(f"{OPENCODE_CONFIG}: the validated config is not JSON: {exc}") from None
    model = config.get("model") if isinstance(config, dict) else None
    if not isinstance(model, str) or not model:
        raise SiteError(f'{OPENCODE_CONFIG}: the validated config has no top-level "model" string')
    return model


def _table(lines: list[str], first_cell: str, source: str) -> tuple[list[str], list[list[str]]]:
    try:
        return markdown.table(lines, first_cell)
    except (LookupError, ValueError) as exc:
        raise SiteError(f"{source}: {exc}") from None


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
    if sorted(seen) != list(range(1, len(seen) + 1)):
        raise SiteError(f"{SELECTION}: the Top 10 table's ranks must run from 1 to {len(seen)} without a gap")
    return ranks, tuple(unmatched)


def makers(lines: list[str]) -> dict[int, Maker]:
    """Each Top 10 rank's Vendor, Params and License cells (spec 6, D8); rank_of checks the ranks themselves."""
    source = f"{SELECTION}#top-10"
    header, rows = _table(lines, "Rank", source)
    for column in ("Vendor", "Params", "License"):
        if column not in header:
            raise SiteError(f"{source}: no {column} column")
    vendor, params, license_ = (header.index(column) for column in ("Vendor", "Params", "License"))
    for cells in rows:
        for column, index in (("Vendor", vendor), ("Params", params), ("License", license_)):
            if not cells[index]:
                raise SiteError(f"{source}: row {cells[0]}: the {column} cell is empty")
    return {int(cells[0]): Maker(cells[vendor], cells[params], cells[license_]) for cells in rows}


def honorable_names(lines: list[str]) -> tuple[str, ...]:
    """The link text of each Model cell in the honorable mentions table (spec 6, D7)."""
    source = f"{SELECTION}#honorable-mentions"
    _, rows = _table(lines, "Model", source)
    names = []
    for cells in rows:
        match = re.match(r"^\[([^\]]+)\]", cells[0])
        if match is None:
            raise SiteError(f"{source}: row {cells[0]!r}: the Model cell must start with a link")
        names.append(match.group(1))
    return tuple(names)


def display_name(name: str, source: str) -> str:
    """The model's name on the page from a recipe name (spec 6, D1): one trailing parenthesis dropped, then the words
    before the first build word."""
    words = re.sub(r"\s*\([^()]*\)$", "", name).split()
    kept = []
    for word in words:
        if BUILD_TOKEN.match(word):
            break
        kept.append(word)
    if not kept:
        raise SiteError(f"{source}: no display name in the recipe name {name!r}")
    return " ".join(kept)


def default_cell(model_id: str, cells: Sequence[Cell]) -> Cell:
    """The model's one Runs cell on its default engine, the recipe's plain_runtime (spec 6, D1)."""
    found = [cell for cell in cells if cell.status == "runs" and cell.runtime == cell.recipe.plain_runtime]
    if len(found) != 1:
        raise SiteError(f"{SELECTION}: row {model_id}: no setup runs on its default engine, so the page has no name "
                        "for it")
    return found[0]


def _image_sizes(root: Path, labels: Mapping[str, str], images: Mapping[str, str]) -> dict[str, str]:
    """Each runtime's image size in GB from docs/host-setup.md#disk-budget (spec 6, D5): the row whose Item cell holds
    the image in a code span."""
    source = "docs/host-setup.md#disk-budget"
    _, rows = _table(_section(root, source), "Item", source)
    sizes: dict[str, str] = {}
    for name, image in images.items():
        for cells in rows:
            size = re.match(r"^(\d+(?:\.\d+)?) GB on disk", cells[1]) if len(cells) > 1 else None
            if image in _CODE_SPAN.findall(cells[0]) and size is not None:
                sizes[name] = size.group(1)
                break
    missing = [labels[name] for name in images if name not in sizes]
    if missing:
        raise SiteError(f"{source}: no measured image size for {_joined(missing)}")
    return sizes


def _runtimes(root: Path, header: list[str], catalog: Catalog) -> tuple[Runtime, ...]:
    if len(header) != 1 + len(ADAPTERS):
        raise SiteError(f"{SELECTION}: runtime coverage: the header has {len(header)} columns; it needs Model ID and "
                        f"one column per engine: {', '.join(adapter.LABEL for adapter in ADAPTERS.values())}")
    for (name, adapter), heading in zip(ADAPTERS.items(), header[1:]):
        if heading != adapter.LABEL and not heading.startswith(adapter.LABEL + " "):
            raise SiteError(f"{SELECTION}: runtime coverage: the column heading {heading!r} must be {adapter.LABEL} "
                            f"or start with {adapter.LABEL + ' '!r}")
    sizes = _image_sizes(root, {name: adapter.LABEL for name, adapter in ADAPTERS.items()},
                         {name: catalog.runtimes[name].runtime.image for name in ADAPTERS})
    return tuple(Runtime(name, adapter.LABEL, heading, catalog.runtimes[name].runtime.image,
                         len(catalog.runtimes[name].recipes), catalog.runtimes[name].runtime.host_port, sizes[name])
                 for (name, adapter), heading in zip(ADAPTERS.items(), header[1:]))


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
    opencode = f"{runtime.opencode_provider}/{recipe.id}" if recipe.opencode else None
    up = ops.up_command(recipe)
    return RecipeFacts(
        name=recipe.name, file=file, doc=recipe.doc, context=recipe.context, decode=evidence.decode_tok_s,
        date=evidence.date, hardware=evidence.hardware, record=evidence.record, summary=evidence.summary,
        gpu=ADAPTERS[recipe.runtime].gpu_need_text(recipe, runtime), ram_peak=recipe.needs_ram_gib,
        memory_cap=recipe.memory_gib, repo=recipe.model.repo, revision=recipe.model.revision,
        size_bytes=recipe.model.size_bytes, thinking=meta.thinking, default=meta.default, efforts=meta.efforts,
        output=meta.output, chat_only=recipe.chat_only,
        opencode=opencode, served=target.served, up=up, pull=preflight.pull_hint(recipe), plain_runtime=plain_runtime,
        thinking_texts=texts, offload=any(validate.flag_present(recipe.args, flag) for flag in OFFLOAD_FLAGS),
        launch=_launch(file, recipe, target, opencode, up))


def _launch(file: str, recipe: Recipe, target: Target, opencode: str | None, up: str) -> Launch:
    """The agents' names for the recipe (spec 6, D4), read from the lines up prints, so the page never drifts."""
    lines = dsh_config.usage_lines(target, held=False, up_command=up)
    line = lines[1] if len(lines) > 1 else ""
    dsh = re.fullmatch(r"\s+dsh web, then pick (.+) in its model picker", line)
    if dsh is None:
        raise SiteError(f"{file}: dsh's up line changed: {line!r}")
    kimi = None
    if not recipe.chat_only:
        lines = kimi_config.usage_lines(target, is_default=False, up_command=up)
        line = lines[1] if len(lines) > 1 else ""
        match = re.match(r"\s+kimi -m (\S+),", line)
        if match is None:
            raise SiteError(f"{file}: Kimi's up line changed: {line!r}")
        kimi = match.group(1)
    kimi_file = kimi_config.AGENT_FILE.resolve().relative_to(Path(launcher.__file__).resolve().parent.parent)
    return Launch(opencode=opencode, hermes=hermes_config.PROFILE, pi=f"{pi_config.PROVIDER}/{target.served}",
                  dsh=dsh.group(1), kimi=kimi, kimi_file=kimi_file.as_posix())


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
    if status != "runs" and _SOURCE_LINK.search(note) is None:          # the source a Why shows (spec 6, D12)
        raise SiteError(f"{where}: a {word} cell needs a link to its source")
    facts = _recipe_facts(root, catalog, recipe, entry.runtime) if status == "runs" else None
    return Cell(model=model_id, runtime=runtime.id, status=status, note=note, measured=measured, recipe=facts,
                fastest=None, clients=(), history=(), covered=None, setup=())


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


def speed_text(decode: float) -> str:
    """A speed as the page writes it: `:g`, with ".0" added when that has no point, so at least one decimal (R-D6)."""
    text = f"{decode:g}"
    return text if "." in text else text + ".0"


def _speeds(model: Model) -> list[tuple[Fraction, str]]:
    """The model's Runs cells with a speed, as (speed, runtime) in the order of its cells."""
    return [(Fraction(str(cell.recipe.decode)), cell.runtime) for cell in model.cells
            if cell.status == "runs" and cell.recipe.decode is not None]


def lead(model: Model, cell: Cell, labels: Mapping[str, str]) -> tuple[str, str] | None:
    """The visible lead of a cell with the fastest tag and its spoken words, from D7's tag: ("24%", "fastest, 24% ahead
    of llama.cpp") or ("tied", "tied fastest with SGLang"), the engines in the order of the model's cells, which is
    ADAPTERS order; None without the tag."""
    if cell.fastest is None:
        return None
    speeds = _speeds(model)
    top = max(speed for speed, _ in speeds)
    if cell.fastest == "Tied fastest":
        others = [labels[runtime] for speed, runtime in speeds if speed == top and runtime != cell.runtime]
        return "tied", f"tied fastest with {_joined(others)}"
    if not cell.fastest.startswith("Fastest by "):
        raise ValueError(f"unknown fastest tag {cell.fastest!r}")
    visible = cell.fastest.removeprefix("Fastest by ")
    second = max(speed for speed, _ in speeds if speed != top)
    behind = [labels[runtime] for speed, runtime in speeds if speed == second]
    return visible, f"fastest, {visible} ahead of {_joined(behind)}"


def _joined(items: Sequence[str]) -> str:
    if not items:
        raise ValueError("nothing to join")
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def licenses(lines: Sequence[str]) -> tuple[str, str]:
    """The two licenses README.md#license names (spec 6, D11)."""
    text = "\n".join(lines)
    if not all(re.search(rf"(?<![\w.]){re.escape(name)}(?!\w|\.\w)", text) for name in LICENSES):
        raise SiteError(f"README.md#license: expected {' and '.join(LICENSES)}")
    return LICENSES


def laptop(root: Path, models: Sequence[Model]) -> Laptop:
    """The one test machine of every Runs cell, with README.md#machine's power limit, free RAM and OS (spec 6, D10)."""
    facts = [cell.recipe for model in models for cell in model.cells if cell.status == "runs"]
    machines = sorted({recipe.hardware for recipe in facts})
    if len(machines) != 1:
        raise SiteError(f"recipes name {len(machines)} different test machines; the page assumes one: "
                        f"{'; '.join(machines)}")
    parts = machines[0].split(", ")
    gpu = next((match for part in parts if (match := re.fullmatch(r"(.+ GPU) (\d+) GB", part))), None)
    ram = next((match for part in parts if (match := re.fullmatch(r"(\d+) GiB RAM", part))), None)
    if len(parts) != 3 or gpu is None or ram is None:
        first = next(recipe.file for recipe in facts)
        raise SiteError(f"{first}: the hardware {machines[0]!r} must read '<laptop>, <GPU name> <N> GB, <M> GiB RAM'")
    source = "README.md#machine"
    _, rows = _table(_section(root, source), "Item", source)
    values = {cells[0]: cells[1] for cells in rows if len(cells) > 1}
    free = re.search(r"(\d+) GiB usable, about (\d+) GiB free", values.get("RAM", ""))
    if free is None or int(free.group(1)) != int(ram.group(1)):
        raise SiteError(f"{source}: the RAM row must read '<N> GiB usable, about <M> GiB free' with N as in the "
                        "recipes")
    power = re.search(r"(\d+) W power limit", values.get("GPU", ""))
    system = values.get("Software", "").split(",")[0].strip()
    if power is None or not system:
        raise SiteError(f"{source}: the GPU row needs '<N> W power limit' and the Software row an OS first")
    dates = sorted(recipe.date for recipe in facts)
    return Laptop(name=parts[0], gpu_name=gpu.group(1), gpu_gb=int(gpu.group(2)), ram_gib=int(ram.group(1)),
                  ram_free_gib=int(free.group(2)), power_w=int(power.group(1)), os=system, first=dates[0],
                  last=dates[-1])


def load(root: Path) -> Matrix:
    """The page's data model from the repository at `root`."""
    catalog = shipped_catalog(root)
    shipped = sorted({recipe_id for entry in catalog.runtimes.values() for recipe_id in entry.recipes})
    header, rows = _table(_section(root, f"{SELECTION}#runtime-coverage"), "Model ID", SELECTION)
    runtimes = _runtimes(root, header, catalog)
    coverage = _coverage_rows(rows, shipped)
    repos: dict[str, list[str]] = {}
    for entry in catalog.runtimes.values():
        for recipe in entry.recipes.values():
            ids = repos.setdefault(recipe.model.repo, [])
            if recipe.id not in ids:
                ids.append(recipe.id)
    top_lines = _section(root, f"{SELECTION}#top-10")
    ranks, unshipped = rank_of(top_lines, {repo: tuple(sorted(ids)) for repo, ids in sorted(repos.items())})
    if unshipped:
        rank, cell = unshipped[0]
        raise SiteError(f"{SELECTION}#top-10: rank {rank} ({cell}) has no shipped recipe; the page lists only models "
                        "llmctl can start")
    made = makers(top_lines)
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
        default = default_cell(model_id, cells).recipe
        name = display_name(default.name, default.file)
        rank = ranks.get(model_id)
        model = Model(id=model_id, aliases=tuple(aliases), rank=rank,
                      chat_only=any(recipe.chat_only for recipe in recipes), cells=cells, name=name,
                      honorable=False, maker=None if rank is None else made[rank])
        models.append(model)
    honorable = honorable_names(_section(root, f"{SELECTION}#honorable-mentions"))
    models = [replace(model, honorable=any(text == model.name or text.startswith(model.name + " ")
                                           for text in honorable))
              for model in models]
    license_lines = tuple(line for line in _section(root, "README.md#license") if line.strip())
    if not license_lines:
        raise SiteError("README.md: the License section has no text")
    named = licenses(license_lines)
    default_model = opencode_default(root)
    pairs = [(model.id, cell.runtime) for model in models for cell in model.cells
             if cell.status == "runs" and cell.recipe.opencode == default_model]
    if len(pairs) != 1:
        raise SiteError(f"{OPENCODE_CONFIG}: the default model {default_model} names {len(pairs)} OpenCode entries of "
                        "pairs that run; it must name exactly one")
    run_list = tuple(data.run for data in runs)
    versions = _client_versions(run_list)
    return Matrix(runtimes=runtimes,
                  clients=tuple(Client(name, clients.LABELS[name], versions[name])
                                for name in clients.NAMES),
                  models=tuple(models), runs=run_list,
                  license=license_lines, top=len(ranks),
                  opencode_default=default_model, default=pairs[0], laptop=laptop(root, models), licenses=named)
