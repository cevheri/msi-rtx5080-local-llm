"""Command line: python3 -m launcher.site <command>. A page source problem prints one line, site: <message>, and
returns 1; a usage error is argparse's own (exit 2)."""

import argparse
import datetime
import re
import stat
import sys
from pathlib import Path

from launcher import files
from launcher.site import data, render

REPO = Path(__file__).resolve().parent.parent.parent
REPOSITORY = "https://github.com/cevheri/msi-rtx5080-local-llm"
INDEX = "index.html"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python3 -m launcher.site",
                                     description="Build the runtime matrix page from the repository's own data.")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="write the runtime matrix page as <out>/index.html")
    build.add_argument("--out", required=True, help="the folder to write index.html into; it may hold nothing else")
    build.add_argument("--commit", required=True, help="the commit the page is built from, 40 hex digits")
    build.add_argument("--commit-time", required=True, help="the commit's time in Unix seconds (git log -1 --format=%%ct)")
    build.add_argument("--repository", default=REPOSITORY, help=f"the GitHub repository (default {REPOSITORY})")
    sub.add_parser("readme", help="rewrite the recipe table of README.md#quick-start from recipes/")
    return parser


def _build_facts(args: argparse.Namespace) -> data.Build:
    if not re.fullmatch(r"[0-9a-f]{40}", args.commit):
        raise data.SiteError(f"--commit {args.commit}: a commit is 40 lower-case hex digits")
    if not re.fullmatch(r"[0-9]+", args.commit_time):
        raise data.SiteError(f"--commit-time {args.commit_time}: a commit time is Unix seconds, a whole number of at "
                             "least 0")
    try:
        datetime.datetime.fromtimestamp(int(args.commit_time), datetime.timezone.utc)
    except (ValueError, OverflowError, OSError) as exc:
        raise data.SiteError(f"--commit-time {args.commit_time}: not a time this page can print: {exc}") from None
    if not re.fullmatch(r"https://github\.com/[A-Za-z0-9-]+/[A-Za-z0-9._-]+", args.repository):
        raise data.SiteError(f"--repository {args.repository}: a repository is https://github.com/<owner>/<name>")
    return data.Build(args.repository, args.commit, int(args.commit_time))


def _build(args: argparse.Namespace, root: Path) -> None:
    facts = _build_facts(args)
    out = Path(args.out)
    if out.exists() and not out.is_dir():
        raise data.SiteError(f"--out {out}: not a folder")
    if out.is_dir():
        others = sorted(entry.name for entry in out.iterdir() if entry.name != INDEX)
        if others:
            raise data.SiteError(f"--out {out}: the folder holds more than {INDEX}: {', '.join(others)}")
        index = out / INDEX
        if index.is_symlink() or (index.exists() and not stat.S_ISREG(index.stat().st_mode)):
            raise data.SiteError(f"--out {out}: {index} is not a regular file")
    matrix = data.load(root)
    text = render.page(matrix, facts, root)
    out.mkdir(parents=True, exist_ok=True)
    path = out / INDEX
    files.write_atomic(path, text, new_mode=0o644)
    print(f"wrote {path}: {len(text.encode('utf-8'))} bytes, {len(matrix.models)} models, {len(matrix.runtimes)} engines, "
          f"{len(matrix.clients)} clients, {len(matrix.runs)} e2e runs")


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
        if args.command == "build":
            _build(args, root)
        elif args.command == "readme":
            _readme(root)
        return 0
    except data.SiteError as exc:
        print(f"site: {exc}", file=sys.stderr)
        return 1
