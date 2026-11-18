"""The e2e summary format and step vocabulary, shared by the driver's writer and the page's reader.

Standard library only: the page generator imports this module without the driver's process and socket code.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

SUMMARY_HEAD = ("# llmctl clients e2e", "", "| Recipe | Step | Pass | Reasons | Seconds |", "|---|---|---|---|---|")
# Every step the driver writes that belongs to no client.
SETUP_STEPS = ("up", "status", "long-grammar", "config", "probes", "gateway", "check", "restore", "down")
_SPECIAL_RECIPES = ("-", "round-trip")
_RECIPE = re.compile(r"\d{2,}-.+-[a-z0-9_]+")
_SECONDS = re.compile(r"[0-9]+")


@dataclass(frozen=True)
class SummaryRow:
    recipe: str
    step: str
    passed: bool
    reasons: str
    seconds: int


def is_summary(raw: bytes) -> bool:
    """Whether a summary.md is an e2e summary: its first line, after a UTF-8 byte order mark, is the head's first line.

    It reads bytes, so a research note in another encoding is not one; a summary with a byte order mark is one, and
    read_summary then stops at its line 1.
    """
    return (raw.removeprefix(b"\xef\xbb\xbf").splitlines() or [b""])[0] == SUMMARY_HEAD[0].encode("utf-8")


def read_summary(text: str) -> list[SummaryRow]:
    """Read a summary that _summary() wrote; any other text raises ValueError naming its 1-based line."""
    lines = text.split("\n")
    for n, expected in enumerate(SUMMARY_HEAD, start=1):
        if n > len(lines) or lines[n - 1] != expected:
            wanted = "a blank line" if expected == "" else f"the summary head line {expected!r}"
            raise ValueError(f"line {n}: expected {wanted}")
    if lines[-1] != "":
        raise ValueError(f"line {len(lines)}: the summary does not end with a newline")
    rows = []
    for n, line in enumerate(lines[len(SUMMARY_HEAD):-1], start=len(SUMMARY_HEAD) + 1):
        if not (line.startswith("| ") and line.endswith(" |")):
            raise ValueError(f"line {n}: not a table row: {line!r}")
        cells = line[2:-2].split(" | ")
        if len(cells) != 5:
            raise ValueError(f"line {n}: {len(cells)} {'cell' if len(cells) == 1 else 'cells'} instead of 5")
        recipe, step, passed, reasons, seconds = cells
        if passed not in ("Yes", "No"):
            raise ValueError(f"line {n}: Pass is {passed!r}, not Yes or No")
        if (passed == "Yes") != (reasons == ""):
            raise ValueError(f"line {n}: Pass {passed} with {'no reasons' if reasons == '' else 'reasons'}")
        if not _SECONDS.fullmatch(seconds):
            raise ValueError(f"line {n}: Seconds {seconds!r} is not a whole number")
        if recipe not in _SPECIAL_RECIPES and not _RECIPE.fullmatch(recipe):
            raise ValueError(f"line {n}: Recipe {recipe!r} is not '-', 'round-trip' or '<nn>-<id>-<engine>'")
        rows.append(SummaryRow(recipe, step, passed == "Yes", reasons, int(seconds)))
    return rows


def step_client(step: str, names: Sequence[str]) -> str | None:
    """The client of a step: the longest name that is the step or its prefix before a -; None for a setup step."""
    matches = [name for name in names if step == name or step.startswith(name + "-")]
    if matches:
        return max(matches, key=len)
    if step in SETUP_STEPS:
        return None
    raise ValueError(f"step {step!r} is neither a client's nor a setup step")


def recipe_key(cell: str) -> tuple[str, str] | None:
    """(id, engine) of a Recipe cell <nn>-<id>-<engine>, None for - and round-trip."""
    if cell in _SPECIAL_RECIPES:
        return None
    if not _RECIPE.fullmatch(cell):
        raise ValueError(f"Recipe {cell!r} is not '-', 'round-trip' or '<nn>-<id>-<engine>'")
    recipe_id, engine = cell.split("-", 1)[1].rsplit("-", 1)
    return recipe_id, engine
