"""Command line: ./llmctl <command>. Errors print one line and return their exit code."""

import argparse
import re
import shlex
from pathlib import Path

from launcher import clients, docker_state, ops
from launcher.catalog import cache_dir, load_catalog, override_dir, resolve, runtime_choices
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
    show.add_argument("--as-override", action="store_true", help="print a copy of the shipped recipe to save as an override")
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
    up.add_argument("--set-default", action="store_true",
                    help="once the server is ready, make it the default of every installed client: "
                         "OpenCode, Hermes, Pi, dsh and Kimi (or set LLMCTL_SET_DEFAULT=1)")
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


def _status(host, recipe, entry, running) -> str:
    """'-', 'running', 'starting', 'running (older file)', or '?' when Docker did not answer."""
    if running is None:
        return "?"
    container = running.get((recipe.runtime, recipe.id))
    if container is None:
        return "-"
    if container.labels.get(LABEL_PREFIX + "sha256") != recipe.source.sha256:
        return "running (older file)"
    return "running" if ADAPTERS[recipe.runtime].ready(host, entry.runtime, recipe) else "starting"


def _list(host, catalog, runtime_name) -> int:
    if runtime_name is not None and runtime_name not in ADAPTERS:
        raise UsageError(f"unknown runtime '{runtime_name}'; choose {runtime_choices()}")
    home = hf_home(host.env)
    try:
        containers = [c for c in docker_state.managed(host) if c.state == "running"]
        running = {(c.labels.get(LABEL_PREFIX + "runtime"), c.labels.get(LABEL_PREFIX + "recipe")): c for c in containers}
        docker_problem = None
    except PreflightError as exc:
        containers, running, docker_problem = [], None, str(exc)
    rows = [("ID", "RUNTIME", "STATUS", "ALIASES", "EVIDENCE", "TOK/S", "CONTEXT", "GPU", "RAM GiB", "SIZE GB",
             "FILES", "IMAGE", "SOURCE")]
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
            speed = f"{recipe.evidence.decode_tok_s:g}" if recipe.evidence.decode_tok_s else "-"
            rows.append((recipe.id, name, _status(host, recipe, entry, running), ",".join(recipe.aliases) or "-",
                         recipe.evidence.kind, speed, str(recipe.context), gpu, ram,
                         f"{recipe.model.size_bytes / 1e9:.1f}", files, image, recipe.source.kind))
    host.out.write(_table(rows))
    for problem in problems:
        host.out.write(f"problem: {problem.message}\n")
    if docker_problem:
        host.out.write(f"Docker did not answer, so STATUS is unknown: {docker_problem}\n")
    for container in containers:
        runtime_label = container.labels.get(LABEL_PREFIX + "runtime", "?")
        entry = catalog.runtimes.get(runtime_label)
        address = f", http://127.0.0.1:{entry.runtime.host_port}/v1" if entry and entry.runtime else ""
        host.out.write(f"running: {container.labels.get(LABEL_PREFIX + 'recipe', '?')} on {runtime_label}{address}\n")
    target = ops.target_from(catalog, containers)
    # A running container whose recipe the catalog lacks is still a running model, so no "no model is running".
    running_known = docker_problem is None and (target is not None or not containers)
    for line in clients.report(host, catalog, target, with_paths=False, running_known=running_known):
        host.out.write(line + "\n")
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


def _shipped_for_override(catalog, model, runtime_name):
    """The shipped recipe, read without the override folder, so that a redirect into the override path works."""
    shipped = load_catalog(REPO / "recipes", None)
    try:
        recipe, _ = resolve(shipped, model, runtime_name)
        return recipe
    except UsageError:
        pass
    # Not shipped: the full catalog either reports why the name fails or finds it in the override folder.
    recipe, _ = resolve(catalog, model, runtime_name)
    # An override may give a shipped model an extra alias; its shipped ID still copies.
    shipped_id = f", or name the shipped ID '{recipe.id}'" if recipe.id in shipped.runtimes[recipe.runtime].recipes else ""
    raise UsageError(f"'{model}' exists only as the override {recipe.source.path}; show --as-override copies shipped "
                     f"recipes, so copy that file instead{shipped_id}")


def _show(host, catalog, model, runtime_name, as_override) -> int:
    if as_override:
        host.out.write(_as_override(_shipped_for_override(catalog, model, runtime_name), override_dir(host.env)))
        return 0
    recipe, runtime = resolve(catalog, model, runtime_name)
    adapter = ADAPTERS[recipe.runtime]
    uid, gid = host.uid_gid()
    env = HostEnv(hf_home(host.env), uid, gid, cache_dir(host.env))
    argv = docker_run_argv(recipe, runtime, adapter.container_spec(recipe, runtime, env))
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


def set_default_from_env(env) -> bool:
    """LLMCTL_SET_DEFAULT: unset, empty or 0 is off, 1 is on, anything else is a usage error."""
    value = env.get("LLMCTL_SET_DEFAULT", "")
    if value in ("", "0"):
        return False
    if value == "1":
        return True
    raise UsageError(f"LLMCTL_SET_DEFAULT must be 1 or 0, found '{value}'")


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
            from_env = set_default_from_env(host.env)
            return ops.up(host, catalog, args.model, args.runtime, replace=args.replace, dry_run=args.dry_run,
                          set_default=args.set_default or from_env)
        if command == "status":
            return ops.status(host, catalog)
        if command == "logs":
            return ops.logs(host, follow=args.follow, tail=args.tail)
        if command == "test":
            return _test(host, catalog)
        if command == "down":
            return ops.down(host, catalog, remove=args.remove)
        return _check(host, catalog)
    except LlmctlError as exc:
        host.err.write(f"error: {exc}\n")
        return exc.exit_code
    except KeyboardInterrupt:
        host.err.write("interrupted\n")
        return 130
