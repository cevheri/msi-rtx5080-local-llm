"""Markdown helpers shared by the docs tests and the runtime matrix page: heading anchors, sections and table rows."""

import re
from pathlib import Path

_SEPARATOR = re.compile(r"^\|(\s*:?-+:?\s*\|)+$")


def slug(heading: str) -> str:
    """GitHub's anchor rule, as the knowledge base's link checker applies it."""
    text = re.sub(r"`|\*\*|\*", "", heading)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    return re.sub(r"[^\w\- ]", "", text.lower()).replace(" ", "-")


def section(root: Path, doc: str) -> list[str]:
    """The lines under the heading that `doc` ("<repository path>#<anchor>") names.

    The section ends at the next heading of level 2 up to its own level, even one inside a fenced block, because
    fences are not tracked; a level 1 heading does not end it. This is the docs tests' rule, moved unchanged.
    """
    path, anchor = doc.split("#", 1)
    lines, found, depth = [], False, 0
    for line in (root / path).read_text(encoding="utf-8").splitlines():
        heading = re.match(r"^(#{2,6})\s+(.*?)\s*$", line)
        if heading:
            if found and len(heading.group(1)) <= depth:
                break
            if slug(heading.group(2)) == anchor:
                found, depth = True, len(heading.group(1))
                continue
        if found:
            lines.append(line)
    if not found:
        raise LookupError(f"no section #{anchor} in {path}")
    return lines


def heading_slugs(path: Path) -> set[str]:
    """The anchors of every Markdown heading in a file, outside fenced blocks."""
    slugs, inside = set(), False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("```"):
            inside = not inside
            continue
        heading = None if inside else re.match(r"^#{1,6}\s+(.*?)\s*$", line)
        if heading:
            slugs.add(slug(heading.group(1)))
    return slugs


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _count(n: int) -> str:
    return "1 cell" if n == 1 else f"{n} cells"


def table(lines: list[str], first_cell: str) -> tuple[list[str], list[list[str]]]:
    """The header and rows of the first table whose header starts with `first_cell`.

    Rows are read until the first line that does not start with "|". The separator line must have as many cells as the
    header, because GitHub shows a table whose separator differs as plain text.
    """
    start = next((n for n, line in enumerate(lines) if line.startswith("| " + first_cell + " |")), None)
    if start is None:
        raise LookupError(f"no table with first column {first_cell}")
    header = _cells(lines[start])
    if start + 1 >= len(lines) or not _SEPARATOR.match(lines[start + 1].strip()):
        if start + 1 >= len(lines):
            found = "the end of the text"
        elif not lines[start + 1].strip():
            found = "a blank line"
        else:
            found = lines[start + 1]
        raise ValueError(f"table {first_cell}: expected a separator line after the header, found: {found}")
    separator = _cells(lines[start + 1])
    if len(separator) != len(header):
        raise ValueError(f"table {first_cell}: the separator line has {_count(len(separator))} where the header has "
                         f"{len(header)}: {lines[start + 1]}")
    rows = []
    for line in lines[start + 2:]:
        if not line.startswith("|"):
            break
        cells = _cells(line)
        if len(cells) != len(header):
            raise ValueError(f"table {first_cell}: {_count(len(cells))} where the header has {len(header)}: {line}")
        rows.append(cells)
    return header, rows
