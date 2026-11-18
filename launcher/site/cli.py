"""Command line: python3 -m launcher.site <command>. A page source problem prints one line, site: <message>, and
returns 1; a usage error is argparse's own (exit 2)."""

import argparse
import sys
from pathlib import Path

from launcher import files
from launcher.site import data, render

REPO = Path(__file__).resolve().parent.parent.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python3 -m launcher.site",
                                     description="Build the runtime matrix page from the repository's own data.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("readme", help="rewrite the recipe table of README.md#quick-start from recipes/")
    return parser


def _readme(root: Path) -> None:
    catalog = data.shipped_catalog(root)
    path = root / render.README
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise data.SiteError(f"{render.README}: missing") from None
    except UnicodeDecodeError as exc:
        raise data.SiteError(f"{render.README}: not valid UTF-8: {exc}") from None
    rewritten = render.rewrite_readme(text, render.readme_table(catalog))
    if rewritten == text:
        print(f"{render.README}: recipe table is current")
        return
    files.write_atomic(path, rewritten)
    print(f"{render.README}: recipe table rewritten")


def main(argv: list[str] | None = None, *, root: Path = REPO) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "readme":
            _readme(root)
        return 0
    except data.SiteError as exc:
        print(f"site: {exc}", file=sys.stderr)
        return 1
