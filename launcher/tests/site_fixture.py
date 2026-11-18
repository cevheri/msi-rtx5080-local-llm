"""A small repository for the runtime matrix page's tests: real recipe files with pinned values, and the documents the
page reads, each written so that a test can change one fact and load the page model again."""

import re
import shutil
import tomllib
from collections.abc import Mapping
from pathlib import Path

from launcher.runtimes import ADAPTERS
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
        self._write("README.md", "# Fixture\n\n## Quick start\n\nRun a model with llmctl.\n\n## License\n\n"
                    + "\n".join(LICENSE_LINES) + "\n")
        self._write("LICENSE", "CC BY 4.0\n")
        self._write("LICENSE-CODE", "MIT\n")
        self._write_selection()

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
