"""Command line: ./llmctl <command>. Errors print one line and return their exit code."""

import argparse
import re
import shlex
from pathlib import Path

from launcher import docker_state, ops
from launcher.catalog import load_catalog, override_dir, resolve
from launcher.command import LABEL_PREFIX, docker_run_argv
from launcher.errors import LlmctlError, PreflightError, UsageError
from launcher.hfcache import hf_home
from launcher.host import RealHost
from launcher.model import HostEnv
from launcher.runtimes import ADAPTERS
from launcher.smoke import smoke

REPO = Path(__file__).resolve().parent.parent


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise UsageError(f"{message}\n{self.format_usage().strip()}")


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="llmctl", description="Start a measured local LLM recipe by name (docs/llmctl.md).")
    sub = parser.add_subparsers(dest="command", required=True, parser_class=_Parser)
    listing = sub.add_parser("list", help="every recipe, its evidence, and what is downloaded")
    listing.add_argument("--runtime")
    show = sub.add_parser("show", help="the effective recipe and the docker command it runs")
    show.add_argument("model")
    show.add_argument("--runtime")
    show.add_argument("--as-override", action="store_true", help="print a complete recipe to save as an override")
    pull = sub.add_parser("pull", help="download the model files and the image, after asking")
    pull.add_argument("model")
    pull.add_argument("--runtime")
    pull.add_argument("--yes", action="store_true", help="do not ask before downloading")
    pull.add_argument("--hf-env-file", type=Path, help="a file with an HF_TOKEN= line, for gated models")
    up = sub.add_parser("up", help="check the machine, start the model, wait until it answers")
    up.add_argument("model")
    up.add_argument("--runtime")
    up.add_argument("--replace", action="store_true", help="stop the model server that is running first")
    up.add_argument("--dry-run", action="store_true", help="print the docker command and start nothing")
    sub.add_parser("status", help="the model server llmctl started, and its health")
    logs = sub.add_parser("logs", help="the model server's log")
    logs.add_argument("-f", "--follow", action="store_true")
    logs.add_argument("--tail", type=int, default=100)
    sub.add_parser("test", help="a reply, a tool call and a decode speed against the running server")
    down = sub.add_parser("down", help="stop the model server llmctl started")
    down.add_argument("--remove", action="store_true", help="also remove the stopped container")
    sub.add_parser("check", help="validate every shipped recipe and override")
    return parser


def _table(rows: list[tuple[str, ...]]) -> str:
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip() for row in rows) + "\n"


def _list(host, catalog, runtime_name) -> int:
    if runtime_name is not None and runtime_name not in ADAPTERS:
        raise UsageError(f"unknown runtime '{runtime_name}'; choose {' or '.join(ADAPTERS)}")
    home = hf_home(host.env)
    rows = [("ID", "RUNTIME", "ALIASES", "EVIDENCE", "CONTEXT", "GPU", "RAM GiB", "SIZE GB", "FILES", "IMAGE", "SOURCE")]
    problems = []
    for name, entry in catalog.runtimes.items():
        if runtime_name and name != runtime_name:
            continue
        problems += entry.problems
        adapter = ADAPTERS[name]
        image = "yes" if entry.runtime and docker_state.image_present(host, entry.runtime.image) else "no"
        for recipe_id in sorted(entry.recipes):
            recipe = entry.recipes[recipe_id]
            gpu = adapter.gpu_need_text(recipe, entry.runtime) if entry.runtime else "?"
            ram = f"{recipe.needs_ram_gib:g}" if recipe.needs_ram_gib else "-"
            files = "no" if adapter.file_problems(recipe, home) else "yes"
            rows.append((recipe.id, name, ",".join(recipe.aliases) or "-", recipe.evidence.kind, str(recipe.context),
                         gpu, ram, f"{recipe.model.size_bytes / 1e9:.1f}", files, image, recipe.source.kind))
    host.out.write(_table(rows))
    for problem in problems:
        host.out.write(f"problem: {problem.message}\n")
    host.out.write(f"model cache: {home}\noverrides: {override_dir(host.env)}\n")
    return 0


def _as_override(recipe, directory: Path) -> str:
    text = recipe.source.path.read_text()
    match = re.search(r"(?m)^\[evidence\]\s*$", text)
    following = re.search(r"(?m)^\[", text[match.end():])
    end = match.end() + following.start() if following else len(text)
    evidence = (f'[evidence]\nkind = "estimate"\nsummary = "Copied from the {recipe.source.kind} recipe {recipe.id}; '
                f'not measured on this machine"\ndoc = "{recipe.doc}"\n')
    header = f"# Save as {directory / recipe.runtime / (recipe.id + '.toml')} and change what your machine needs.\n"
    rest = text[end:]
    return header + text[:match.start()] + evidence + (("\n" + rest) if rest.strip() else "")


def _show(host, catalog, model, runtime_name, as_override) -> int:
    recipe, runtime = resolve(catalog, model, runtime_name)
    if as_override:
        host.out.write(_as_override(recipe, override_dir(host.env)))
        return 0
    adapter = ADAPTERS[recipe.runtime]
    uid, gid = host.uid_gid()
    argv = docker_run_argv(recipe, runtime, adapter.container_spec(recipe, runtime, HostEnv(hf_home(host.env), uid, gid)))
    model_info = recipe.model
    lines = [
        f"{recipe.id} ({recipe.name}) on {recipe.runtime}",
        f"aliases: {', '.join(recipe.aliases) or '-'}",
        f"recipe file: {recipe.source.path} ({recipe.source.kind}), sha256 {recipe.source.sha256}",
        f"runtime file: {runtime.source.path} ({runtime.source.kind}), image {runtime.image}",
        f"evidence: {recipe.evidence.kind}: {recipe.evidence.summary}",
        f"model: {model_info.repo} at {model_info.revision}, {model_info.size_bytes / 1e9:.1f} GB",
        f"container memory cap: {recipe.memory_gib} GiB, context {recipe.context}, GPU need {adapter.gpu_need_text(recipe, runtime)}",
        f"OpenCode: {runtime.opencode_provider}/{recipe.id}" if recipe.opencode else "OpenCode: no entry",
        f"documented in: {recipe.doc}",
        "command:",
        shlex.join(argv),
    ]
    host.out.write("\n".join(lines) + "\n")
    return 0


def _test(host, catalog) -> int:
    running = [c for c in docker_state.managed(host) if c.state == "running"]
    if not running:
        raise PreflightError("no model server started by llmctl is running; start one with ./llmctl up <model>")
    labels = running[0].labels
    recipe, runtime = resolve(catalog, labels.get(LABEL_PREFIX + "recipe", ""), labels.get(LABEL_PREFIX + "runtime"))
    return smoke(host, recipe, runtime)


def _check(host, catalog) -> int:
    problems = [problem for entry in catalog.runtimes.values() for problem in entry.problems]
    for problem in problems:
        host.out.write(f"problem: {problem.message}\n")
    if problems:
        return 2
    counts = ", ".join(f"{name} {len(entry.recipes)}" for name, entry in catalog.runtimes.items())
    total = sum(len(entry.recipes) for entry in catalog.runtimes.values())
    host.out.write(f"ok: {total} recipes ({counts}); overrides: {override_dir(host.env)}\n")
    return 0


def main(argv: list[str], host=None) -> int:
    host = host or RealHost()
    try:
        args = build_parser().parse_args(argv)
        catalog = load_catalog(REPO / "recipes", override_dir(host.env))
        command = args.command
        if command == "list":
            return _list(host, catalog, args.runtime)
        if command == "show":
            return _show(host, catalog, args.model, args.runtime, args.as_override)
        if command == "pull":
            return ops.pull(host, catalog, args.model, args.runtime, yes=args.yes, env_file=args.hf_env_file)
        if command == "up":
            return ops.up(host, catalog, args.model, args.runtime, replace=args.replace, dry_run=args.dry_run)
        if command == "status":
            return ops.status(host, catalog)
        if command == "logs":
            return ops.logs(host, follow=args.follow, tail=args.tail)
        if command == "test":
            return _test(host, catalog)
        if command == "down":
            return ops.down(host, remove=args.remove)
        return _check(host, catalog)
    except LlmctlError as exc:
        host.err.write(f"error: {exc}\n")
        return exc.exit_code
    except KeyboardInterrupt:
        host.err.write("interrupted\n")
        return 130
