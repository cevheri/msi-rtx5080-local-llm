"""A small repository for the runtime matrix page's tests: real recipe files with pinned values, and the documents the
page reads, each written so that a test can change one fact and load the page model again."""

import html.parser
import json
import re
import shutil
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path

from launcher.e2e.summary import SUMMARY_HEAD
from launcher.runtimes import ADAPTERS
from launcher.site.render import README_END, README_START
from launcher.tests.fakes import REPO

RECIPES = (("llamacpp", "qwen3.6-35b-a3b"), ("sglang", "qwen3.6-35b-a3b"), ("llamacpp", "gpt-oss-20b"),
           ("vllm", "gpt-oss-20b"), ("sglang", "gpt-oss-20b"), ("llamacpp", "deepseek-r1-0528-qwen3-8b"),
           ("vllm", "deepseek-r1-0528-qwen3-8b"), ("llamacpp", "k2-horizon-7b"), ("llamacpp", "qwen3-coder-next"),
           ("llamacpp", "mellum2.1-12b"))
SPEEDS = dict(zip(RECIPES, ("122.5", "151.3", "166.2", "83.5", "124.6", "70.6", "72.5", "66.9", "71.5", "188.2")))
CONTEXTS = dict(zip(RECIPES, ("131072", "131072", "131072", "65536", "131072", "65536", "65536", "81920", "131072",
                              "131072")))
COVERAGE_HEADER = "| Model ID | llama.cpp b12407 | vLLM v0.33.0 | SGLang v0.5.23 |"
RUNS_CELL = "Runs: {k}k, {speed} tok/s, measured {date} ([recipe](../{doc}))"
FAILED_CELL = ("Failed: out of memory, measured 2026-11-17 "
               "([record](../research/2026-11-17/fixture-failure/README.md))")
FAILED_AT = ("deepseek-r1-0528-qwen3-8b", "sglang")
DOES_NOT_FIT_CELL = "Does not fit: too large for the budget (estimate)"
TOP10_HEADER = "| Rank | Model | Runtime |"
TOP10_ROW = "| {rank} | [{id}](https://huggingface.co/{repo}) | llama.cpp |"
TOP10_IDS = ("qwen3.6-35b-a3b", "gpt-oss-20b", "deepseek-r1-0528-qwen3-8b")
LICENSE_LINES = ("The documentation is licensed under CC BY 4.0 ([LICENSE](LICENSE)); the code under MIT "
                 "([LICENSE-CODE](LICENSE-CODE)).", "Third-party material keeps its own license.")


DEFAULT_RUN = ("2026-11-18", "llmctl-fixture-e2e")
DEFAULT_VERSIONS = (("001-hermes-version", "Hermes Agent v0.23.1"), ("002-pi-version", "1.5.2"),
                    ("003-opencode-version", "opencode v2.3.0"), ("004-dsh-version", "0.2.2-rc.3"),
                    ("005-kimi-version", "2.5.3"), ("006-nvidia-smi-version", "[N/A]"))
CLIENT_STEPS = ("opencode", "hermes-z", "hermes-chat", "pi", "dsh-headless", "kimi")


def row(recipe: str, step: str, reasons: str = "", seconds: int = 1) -> str:
    """One summary row as the driver writes it; a row with reasons failed."""
    return f"| {recipe} | {step} | {'No' if reasons else 'Yes'} | {reasons} | {seconds} |"


def prefix(runtime: str, recipe_id: str) -> str:
    """The Recipe cell of a fixture recipe: its 1-based place in RECIPES, its ID and its runtime."""
    return f"{RECIPES.index((runtime, recipe_id)) + 1:02d}-{recipe_id}-{runtime}"


def version_record(stdout: str, exit: int | None = 0) -> str:
    """A `<client> --version` record as the driver writes it."""
    record = {"command": "client --version", "cwd": "$HOME/projects/llm", "exit": exit, "seconds": 1.0,
              "stdout": stdout, "stderr": ""}
    return json.dumps(record, indent=2, ensure_ascii=False) + "\n"


def default_rows() -> list[str]:
    """Every recipe of RECIPES passing every step, then the round trip and the final down."""
    rows = []
    for runtime, recipe_id in RECIPES:
        cell = prefix(runtime, recipe_id)
        steps = ["up", "status", *(["long-grammar"] if runtime == "sglang" else []), "config",
                 *(step for step in CLIENT_STEPS
                   if not (step == "opencode" and (runtime, recipe_id) == ("vllm", "deepseek-r1-0528-qwen3-8b"))),
                 "probes"]
        for step in steps:
            rows.append(row(cell, step))
            if step == "up" and len(rows) == 1:
                rows.append(row(cell, "gateway"))
    rows += [row("round-trip", "up"), row("round-trip", "opencode"), row("round-trip", "restore"), row("-", "down")]
    return rows


def _separator(header: str) -> str:
    return "|" + "---|" * (header.count("|") - 1)


class SiteFixture:
    """The default repository of the plan's Task 3, written under `root`."""

    def __init__(self, root: Path):
        self.root = root
        for runtime in ADAPTERS:
            (root / "recipes" / runtime).mkdir(parents=True)
            shutil.copyfile(REPO / "recipes" / runtime / "_runtime.toml", root / "recipes" / runtime / "_runtime.toml")
        for runtime, recipe_id in RECIPES:
            shutil.copyfile(REPO / "recipes" / runtime / f"{recipe_id}.toml", self._recipe(runtime, recipe_id))
            self.set_value(runtime, recipe_id, "decode_tok_s", SPEEDS[runtime, recipe_id])
            self.set_value(runtime, recipe_id, "context", CONTEXTS[runtime, recipe_id])
        self._cells: dict[tuple[str, str], str] = {}
        self._table: list[str] | None = None
        self._top10 = [TOP10_ROW.format(rank=rank, id=recipe_id, repo=self._read("llamacpp", recipe_id)["model"]["repo"])
                       for rank, recipe_id in enumerate(TOP10_IDS, start=1)]
        anchors: dict[str, list[str]] = {}
        records = {"research/2026-11-17/fixture-failure/"}
        for runtime, recipe_id in RECIPES:
            data = self._read(runtime, recipe_id)
            path, anchor = data["doc"].split("#", 1)
            anchors.setdefault(path, []).append(anchor)
            records.add(data["evidence"]["record"])
        for path, names in sorted(anchors.items()):
            self._write(path, "# Models\n\n" + "".join(f"## {name}\n\n" for name in names))
        for record in sorted(records):
            self._write(f"{record.rstrip('/')}/README.md", "# Record\n")
        self._write("README.md", "# Fixture\n\n## Quick start\n\nRun a model with llmctl.\n\n" + README_START + "\n\n"
                    + README_END + "\n\n## License\n\n" + "\n".join(LICENSE_LINES) + "\n")
        self._write("LICENSE", "CC BY 4.0\n")
        self._write("LICENSE-CODE", "MIT\n")
        self._write_selection()
        self.run(*DEFAULT_RUN, default_rows(), readme="# e2e\n\n## The No rows\n\nNone.\n",
                 records={name: version_record(stdout) for name, stdout in DEFAULT_VERSIONS})

    def run(self, date: str, folder: str, rows: Sequence[str], *, readme: str | None = "# e2e\n",
            records: Mapping[str, str] | None = None) -> str:
        """Write research/<date>/<folder>/summary.md with the rows, its README and its records; the run's path."""
        path = f"research/{date}/{folder}"
        self._write(f"{path}/summary.md", "\n".join([*SUMMARY_HEAD, *rows]) + "\n")
        if readme is not None:
            self._write(f"{path}/README.md", readme)
        if records is not None:
            (self.root / path / "records").mkdir(parents=True, exist_ok=True)
            for name, text in records.items():
                self._write(f"{path}/records/{name}", text)
        return path

    def _recipe(self, runtime: str, recipe_id: str) -> Path:
        return self.root / "recipes" / runtime / f"{recipe_id}.toml"

    def _read(self, runtime: str, recipe_id: str) -> dict:
        return tomllib.loads(self._recipe(runtime, recipe_id).read_text(encoding="utf-8"))

    def _write(self, relative: str, text: str) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def set_value(self, runtime: str, recipe_id: str, key: str, value: str | None) -> None:
        """Replace the recipe file's first `<key> = ` line with `<key> = <value>`, or remove it for None."""
        path = self._recipe(runtime, recipe_id)
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        index = next((n for n, line in enumerate(lines) if re.match(rf"^{re.escape(key)} = ", line)), None)
        if index is None:
            if value is None:
                return
            raise LookupError(f"{path.name} has no {key} line")
        if value is None:
            del lines[index]
        else:
            lines[index] = f"{key} = {value}\n"
        path.write_text("".join(lines), encoding="utf-8")

    def _default_cell(self, model_id: str, runtime: str) -> str:
        if (runtime, model_id) in SPEEDS:
            data = self._read(runtime, model_id)
            evidence = data["evidence"]
            return RUNS_CELL.format(k=data["server"]["context"] // 1024, speed=f"{evidence['decode_tok_s']:g}",
                                    date=evidence["date"], doc=data["doc"])
        return FAILED_CELL if (model_id, runtime) == FAILED_AT else DOES_NOT_FIT_CELL

    def _coverage_lines(self) -> list[str]:
        if self._table is not None:
            return self._table
        rows = []
        for model_id in sorted({recipe_id for _, recipe_id in RECIPES}):
            cells = [self._cells[model_id, runtime] if (model_id, runtime) in self._cells
                     else self._default_cell(model_id, runtime) for runtime in ADAPTERS]
            rows.append(f"| `{model_id}` | " + " | ".join(cells) + " |")
        return [COVERAGE_HEADER, _separator(COVERAGE_HEADER), *rows]

    def _write_selection(self) -> None:
        lines = ["# Model selection", "", "## Runtime coverage", "", "Every model has one cell per engine.", "",
                 *self._coverage_lines(), "", "## Top 10", "", TOP10_HEADER, _separator(TOP10_HEADER), *self._top10]
        self._write("docs/model-selection.md", "\n".join(lines) + "\n")

    def coverage(self, cells: Mapping[tuple[str, str], str]) -> None:
        """Set whole cell texts, keyed by model ID and runtime; the other cells keep their defaults.

        A model ID outside RECIPES or a runtime outside ADAPTERS raises LookupError, because the table has no such cell.
        """
        models = {recipe_id for _, recipe_id in RECIPES}
        for model_id, runtime in cells:
            if model_id not in models or runtime not in ADAPTERS:
                raise LookupError(f"the coverage table has no cell for {model_id} on {runtime}")
        self._cells.update(cells)
        self._write_selection()

    def coverage_table(self, lines: list[str]) -> None:
        """Replace the coverage table's header, separator and rows."""
        self._table = list(lines)
        self._write_selection()

    def top10(self, rows: list[str]) -> None:
        """Replace the Top 10 rows under the default header."""
        self._top10 = list(rows)
        self._write_selection()

    def load(self):
        from launcher.site import data
        return data.load(self.root)


def cell_of(matrix, model_id: str, runtime: str):
    return next(cell for cell in model_of(matrix, model_id).cells if cell.runtime == runtime)


def model_of(matrix, model_id: str):
    return next(model for model in matrix.models if model.id == model_id)


VOID = frozenset({"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"})


class Node:
    """One element of a parsed page: its tag, its attributes and its children (elements and text)."""

    def __init__(self, tag: str, attrs: dict[str, str | None]):
        self.tag = tag
        self.attrs = attrs
        self.children: list["Node | str"] = []

    def text(self) -> str:
        return "".join(child if isinstance(child, str) else child.text() for child in self.children)

    def iter(self):
        """The descendant elements in document order."""
        for child in self.children:
            if isinstance(child, Node):
                yield child
                yield from child.iter()

    def by_id(self, element_id: str) -> "Node | None":
        return next((node for node in self.iter() if node.attrs.get("id") == element_id), None)

    def find_all(self, tag: str | None = None, cls: str | None = None) -> list["Node"]:
        return [node for node in self.iter() if (tag is None or node.tag == tag)
                and (cls is None or cls in (node.attrs.get("class") or "").split())]


class _Tree(html.parser.HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = Node("#document", {})
        self.stack = [self.root]
        self.problems: list[str] = []
        self.ids: dict[str, int] = {}
        self.hrefs: list[str] = []

    def handle_starttag(self, tag, attrs):
        node = Node(tag, dict(attrs))
        self.stack[-1].children.append(node)
        element_id = node.attrs.get("id")
        if element_id is not None:
            self.ids[element_id] = self.ids.get(element_id, 0) + 1
        href = node.attrs.get("href")
        if href is not None and href.startswith("#"):
            self.hrefs.append(href[1:])
        if tag not in VOID:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.problems.append(f"<{tag} ... /> is written as a self-closing tag")
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag):
        if tag in VOID:
            self.problems.append(f"</{tag}> closes a void element")
            return
        if len(self.stack) == 1 or self.stack[-1].tag != tag:
            self.problems.append(f"</{tag}> closes <{self.stack[-1].tag}> out of order")
            return
        self.stack.pop()

    def handle_data(self, data):
        self.stack[-1].children.append(data)


def _tree(page: str) -> _Tree:
    tree = _Tree()
    tree.feed(page)
    tree.close()
    return tree


def parse(page: str) -> Node:
    """The page as a tree of Node elements."""
    return _tree(page).root


def well_formed_problems(page: str) -> list[str]:
    """What keeps the page from being well formed: an element closed out of order or left open, a duplicate ID, a
    count of h1 other than one, and an href="#x" without an element whose ID is x."""
    tree = _tree(page)
    problems = list(tree.problems)
    problems += [f"<{node.tag}> is never closed" for node in tree.stack[1:]]
    problems += [f"the ID {element_id} is used {n} times" for element_id, n in sorted(tree.ids.items()) if n > 1]
    h1 = len(tree.root.find_all("h1"))
    if h1 != 1:
        problems.append(f"{h1} h1 elements")
    problems += [f"href=#{target} names no element" for target in sorted(set(tree.hrefs)) if target not in tree.ids]
    return problems
