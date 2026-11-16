"""The opt-in e2e of llmctl's clients (spec D10, L3, L9): real servers, real clients, real config files.

Only the lead runs it, alone and through heavy, after the full unit suite passes:
heavy python3 -m launcher.e2e.clients --out research/<date>/llmctl-clients-e2e [--matrix default|extended]
[--keep-running] [--set-default-round-trip]
--out holds only the records; the clients run in fresh work folders under one private temporary folder per run
(llmctl-e2e-*), outside the repository, which the run leaves in place for inspection.
Importing this module has no side effects; everything runs inside main and the functions below.
"""

import argparse
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from launcher import docker_state, files, hermes_config, opencode_config, pi_config
from launcher.catalog import load_catalog, override_dir
from launcher.command import LABEL_PREFIX
from launcher.errors import LlmctlError, UsageError
from launcher.host import RealHost
from launcher.model import Target
from launcher.ops import client_target
from launcher.runtimes import ADAPTERS

REPO = Path(__file__).resolve().parents[2]
TASK = ("Create a file named hello.py containing a function add(a, b) that returns a + b, and a main block that "
        "prints add(2, 3). Then read the file back and tell me how many lines it has. Do not run any shell commands.")
CHAT = "Say hello in one sentence."
PERMISSIONS = {name: "deny" for name in ("bash", "external_directory", "webfetch", "websearch", "task", "question")}
STEP_TIMEOUT_S = 900
GATEWAY_WAIT_S = 35
DEFAULT_MATRIX = (("qwen3.8-27b", "llamacpp"), ("k2-horizon-7b", "llamacpp"), ("gpt-oss-20b", "vllm"),
                  ("qwen3.6-35b-a3b", "llamacpp"))
EXTENDED_MATRIX = DEFAULT_MATRIX[:-1] + tuple((name, "llamacpp") for name in (
    "nemotron-3.5-bolt", "qwen3-coder-next", "deepseek-r1-0528-qwen3-8b", "gemma-4-26b-a4b",
    "nemotron-3.5-lightning", "groot-19b-a3b", "k2-horizon-mova-36b", "gpt-oss-20b")) + DEFAULT_MATRIX[-1:]
MATRICES = {"default": DEFAULT_MATRIX, "extended": EXTENDED_MATRIX}
ROUND_TRIP_MODEL = "qwen3.6-35b-a3b"
PROBE_C = ("k2-horizon-7b", "k2-horizon-mova-36b", "qwen3.6-35b-a3b")
WORK = "{work}"  # replaced by the step's work folder when the step runs
ROUND_TRIP = "round-trip"
IN_PROCESS = ()  # the argv of a step the driver runs itself: the config check and the probes
_TITLE_PROMPT = "Write a short title for a chat about adding two numbers in Python."
_OK_PROMPT = "Reply with exactly the word OK."
_REASONING = "I will write the file."
# Streaming events the records leave out: they split the answer into tokens, so redact() never sees a path whole,
# and the complete text is also in Hermes' result event and Pi's message_end event
DROPPED_EVENTS = {"hermes-chat": ("text",), "pi": ("message_update",)}
WITHHELD = "[withheld: the output named a private value after redaction]"
WITHHELD_REASON = "record withheld: the output named a private value after redaction"


@dataclass(frozen=True)
class Outcome:
    passed: bool
    reasons: tuple[str, ...]
    models: tuple[str, ...]


@dataclass(frozen=True)
class Step:
    name: str
    argv: tuple[str, ...]
    cwd: str  # "." for the repository, else a fresh work folder under the run's private work root
    timeout_s: int


@dataclass(frozen=True)
class Defaults:
    opencode_model: str
    pi_provider: str | None
    pi_model: str | None
    hermes_profile: str


# Parsers of the clients' output


def _events(lines: Iterable[str]) -> tuple[list[dict], list[str]]:
    """The JSON object lines, and one reason per line that is not one."""
    events, reasons = [], []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except ValueError:
            reasons.append(f"line {number} is not JSON: {line.strip()[:120]}")
            continue
        if isinstance(event, dict):
            events.append(event)
        else:
            reasons.append(f"line {number} is not JSON object")
    return events, reasons


def _dict(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _add_model(models: list[str], provider: object, model: object) -> None:
    if isinstance(provider, str) and isinstance(model, str) and provider and model:
        name = f"{provider}/{model}"
        if name not in models:
            models.append(name)


def _outcome(reasons: list[str], models: list[str]) -> Outcome:
    return Outcome(not reasons, tuple(reasons), tuple(models))


def opencode_outcome(lines: Iterable[str]) -> Outcome:
    """`opencode run --format json`: a completed write and read tool_use, and no error event (spec D10 step 3)."""
    events, reasons = _events(lines)
    done, models = set(), []
    for event in events:
        part = _dict(event.get("part"))
        for holder in (event, part, _dict(part.get("info"))):
            _add_model(models, holder.get("providerID"), holder.get("modelID"))
        if event.get("type") == "tool_use" and _dict(part.get("state")).get("status") == "completed":
            done.add(part.get("tool"))
        if event.get("type") == "error":
            reasons.append(f"error event: {json.dumps(event)[:300]}")
    for tool in ("write", "read"):
        if tool not in done:
            reasons.append(f"no completed {tool} tool_use")
    return _outcome(reasons, models)


def hermes_stream_outcome(lines: Iterable[str], *, tools: bool = True) -> Outcome:
    """`hermes chat --format stream-json`: a last result event with exit_code 0 and clean file tools (step 5)."""
    events, reasons = _events(lines)
    models: list[str] = []
    names = set()
    for event in events:
        kind = event.get("type")
        if kind == "system" and isinstance(event.get("model"), str) and event["model"] and event["model"] not in models:
            models.append(event["model"])
        if kind in ("tool_use", "tool_result"):
            names.add(event.get("name"))
        if kind == "tool_result" and event.get("is_error") is True:
            reasons.append(f"tool_result {event.get('name')} has is_error true")
    last = events[-1] if events else {}
    if last.get("type") != "result" or last.get("exit_code") != 0:
        reasons.append(f"the last event is not a result with exit_code 0: {json.dumps(last)[:200]}")
    if tools:
        for tool in ("write_file", "read_file"):
            if tool not in names:
                reasons.append(f"no {tool} tool event")
    return _outcome(reasons, models)


def hermes_usage_outcome(text: str) -> Outcome:
    """`hermes -z --usage-file`: completed true; the model as <provider>/<model> (spec D10 step 4)."""
    try:
        usage = json.loads(text)
    except ValueError as exc:
        return Outcome(False, (f"the usage file is not JSON: {exc}",), ())
    if not isinstance(usage, dict):
        return Outcome(False, ("the usage file is not a JSON object",), ())
    models: list[str] = []
    _add_model(models, usage.get("provider"), usage.get("model"))
    reasons = [] if usage.get("completed") is True else [f"completed is {json.dumps(usage.get('completed'))}"]
    return _outcome(reasons, models)


def pi_outcome(lines: Iterable[str], *, tools: bool = True) -> Outcome:
    """`pi --mode json`: no assistant message_end that stopped on error, clean write and read (step 6)."""
    events, reasons = _events(lines)
    models: list[str] = []
    clean = set()
    for event in events:
        message = _dict(event.get("message"))
        if message.get("role") == "assistant":
            _add_model(models, message.get("provider"), message.get("model"))
            if event.get("type") == "message_end" and message.get("stopReason") == "error":
                reasons.append("an assistant message_end has stopReason error")
        if event.get("type") == "tool_execution_end":
            if event.get("isError") is False:
                clean.add(event.get("toolName"))
            else:
                reasons.append(f"tool_execution_end {event.get('toolName')} has isError {event.get('isError')}")
    if tools:
        for tool in ("write", "read"):
            if tool not in clean:
                reasons.append(f"no tool_execution_end for {tool} with isError false")
    return _outcome(reasons, models)


def hello_valid(path: Path) -> bool:
    """The task's file exists and holds def add and add(2, 3)."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return False
    return "def add" in text and "add(2, 3)" in text


def gateway_serves_llmctl(text: str) -> bool:
    """True when gateway_state.json says a running gateway serves the profile llmctl (spec L3)."""
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("not a JSON object")
    served = data.get("served_profiles")
    return data.get("gateway_state") == "running" and isinstance(served, list) and hermes_config.PROFILE in served


GATEWAY_WARNING = "warning: Hermes' gateway serves the profile llmctl"


def status_reasons(text: str, *, opencode: bool) -> list[str]:
    """One reason per client line of ./llmctl status that does not say the running model (spec D10 step 1), and
    the gateway warning (spec L3)."""
    lines = {}
    if "clients:\n" in text:
        for line in text.split("clients:\n", 1)[1].splitlines():
            if line.startswith("  ") and ": " in line:
                label, rest = line.strip().split(": ", 1)
                lines[label] = rest
    reasons = []
    for label in (("OpenCode",) if opencode else ()) + ("Hermes", "Pi"):
        if label not in lines:
            reasons.append(f"no {label} line in status")
        elif "the running model" not in lines[label]:
            reasons.append(f"{label}: {lines[label]}")
    reasons += [line for line in text.splitlines() if line.startswith(GATEWAY_WARNING)]
    return reasons


# The plan


def _target(catalog, recipe_id: str, runtime: str) -> Target:
    entry = catalog.runtimes.get(runtime)
    recipe = entry.recipes.get(recipe_id) if entry else None
    if recipe is None or entry.runtime is None:
        raise UsageError(f"the e2e matrix names {recipe_id} on {runtime}, which the catalog lacks")
    return client_target(recipe, entry.runtime)


def _session_argv(target: Target, prompt: str, tools: bool, *, pinned: bool) -> dict[str, tuple[str, ...]]:
    """The client sessions of D10 steps 3 to 6; pinned False drops -p and --model (the L9 round trip)."""
    recipe = target.recipe
    profile = ("-p", hermes_config.PROFILE) if pinned else ()
    file_tools = ("-t", "file") if tools else ()
    sessions = {}
    if tools and recipe.opencode:
        model = ("--model", f"{target.runtime.opencode_provider}/{recipe.id}") if pinned else ()
        sessions["opencode"] = ("opencode", "run", "--standalone", "--archived", "--auto", "--format", "json",
                                *model, prompt)
    sessions["hermes-z"] = ("hermes", *profile, "-z", prompt, *file_tools, "--usage-file", f"{WORK}/usage.json")
    if pinned:
        sessions["hermes-chat"] = ("hermes", *profile, "chat", "-q", prompt, *file_tools, "--format", "stream-json",
                                   "--source", "tool", "--max-turns", "10")
    pi_tools = ("--tools", "read,write") if tools else ("--no-tools",)
    model = ("--model", f"{pi_config.PROVIDER}/{target.served}") if pinned else ()
    sessions["pi"] = ("pi", "--mode", "json", "--no-session", "--no-approve", "--no-context-files", *pi_tools,
                      *model, prompt)
    return sessions


def up_argv(target: Target, *extra: str) -> tuple[str, ...]:
    runtime = target.recipe.runtime
    flag = () if runtime == next(iter(ADAPTERS)) else ("--runtime", runtime)
    return ("./llmctl", "up", target.recipe.id, *flag, *extra)


def plan_steps(catalog, matrix: Sequence[tuple[str, str]], *, round_trip: bool, keep_running: bool) -> list[Step]:
    """Every step of a run, in order (spec D10, L3, L9); argv () marks a step the driver runs in-process."""
    steps = []

    def add(name: str, argv: tuple[str, ...], cwd: str = ".") -> None:
        steps.append(Step(name, argv, cwd, STEP_TIMEOUT_S))

    for number, (recipe_id, runtime) in enumerate(matrix, 1):
        target = _target(catalog, recipe_id, runtime)
        prefix = f"{number:02d}-{recipe_id}-{runtime}"
        tools = not target.recipe.chat_only
        add(f"{prefix} up", up_argv(target, "--replace"))
        add(f"{prefix} status", ("./llmctl", "status"))
        add(f"{prefix} config", IN_PROCESS)
        for key in hermes_config.block_values(target):
            get = ("hermes", "-p", hermes_config.PROFILE, "config", "get", key, "--json")
            add(f"{prefix} hermes-config-{key}", get, f"{prefix}-hermes-config-{key}")
        for kind, argv in _session_argv(target, TASK if tools else CHAT, tools, pinned=True).items():
            add(f"{prefix} {kind}", argv, f"{prefix}-{kind}")
        add(f"{prefix} probes", IN_PROCESS)
    if round_trip:
        target = _target(catalog, ROUND_TRIP_MODEL, "llamacpp")
        add(f"{ROUND_TRIP} up", up_argv(target, "--set-default"))
        for kind, argv in _session_argv(target, TASK, True, pinned=False).items():
            add(f"{ROUND_TRIP} {kind}", argv, f"{ROUND_TRIP}-{kind}")
    if not keep_running:
        add("down", ("./llmctl", "down"))
    return steps


# Probes (spec D10 step 7)


def _write_tool() -> dict:
    return {"type": "function", "function": {
        "name": "write",
        "description": "Write content to a file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                       "required": ["path", "content"]},
    }}


def probe_bodies(target: Target) -> dict[str, dict]:
    served, recipe = target.served, target.recipe
    title_schema = {"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"],
                    "additionalProperties": False}
    bodies = {
        "A": {"model": served, "messages": [{"role": "user", "content": _TITLE_PROMPT}], "reasoning_effort": "none",
              "response_format": {"type": "json_schema", "json_schema": {
                  "name": "session_title", "strict": True, "schema": title_schema}},
              "max_tokens": recipe.clients.output, "stream": False},
        "B": {"model": served, "messages": [
            {"role": "user", "content": TASK},
            {"role": "assistant", "content": "", "reasoning": _REASONING, "reasoning_content": _REASONING,
             "tool_calls": [{"id": "call_1", "type": "function", "function": {
                 "name": "write", "arguments": json.dumps({"path": "hello.py", "content": "print(1)\n"})}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
        ], "tools": [_write_tool()], "max_tokens": 256},
    }
    plain = {"model": served, "messages": [{"role": "user", "content": _OK_PROMPT}], "max_tokens": 256}
    if recipe.id in PROBE_C:
        bodies["C"] = {"model": served, "messages": plain["messages"], "reasoning_effort": "none", "max_tokens": 256}
    if recipe.runtime == "llamacpp":
        bodies["D"] = dict(plain, model=recipe.id)
        if recipe.id == "k2-horizon-7b":
            bodies["E"] = dict(plain, chat_template_kwargs={"tool_call_format": "xml"})
    return bodies


def _non_empty(value: object) -> bool:
    return isinstance(value, str) and value != ""


def probe_record(name: str, status: int, body: bytes) -> dict:
    """The recorded result of one probe: status, content and reasoning presence, and for A whether the title parses."""
    try:
        message = _dict(_dict((json.loads(body).get("choices") or [{}])[0]).get("message"))
    except (ValueError, AttributeError, IndexError):
        message = {}
    content = message.get("content")
    record = {"probe": name, "status": status, "content": _non_empty(content),
              "reasoning": _non_empty(message.get("reasoning")) or _non_empty(message.get("reasoning_content"))}
    if name == "A":
        try:
            title = json.loads(content).get("title") if _non_empty(content) else None
        except (ValueError, AttributeError):
            title = None
        record["title_ok"] = _non_empty(title.strip() if isinstance(title, str) else None)
    return record


# Environment and records


def _contains(value: object, wanted: object) -> bool:
    """Whether value holds wanted: every key of a wanted dict, recursively; an empty dict or anything else equal,
    with JSON types."""
    if isinstance(wanted, dict) and wanted:
        return isinstance(value, dict) and all(key in value and _contains(value[key], item)
                                               for key, item in wanted.items())
    return type(value) is type(wanted) and value == wanted


def driver_env(environ: Mapping[str, str]) -> dict[str, str]:
    """The environment of every command the driver runs: LLMCTL_SET_DEFAULT removed (spec D10)."""
    return {name: value for name, value in environ.items() if name != "LLMCTL_SET_DEFAULT"}


def _replace_path(text: str, path: str, name: str) -> str:
    if not path:
        return text
    return re.sub(re.escape(path.rstrip("/")) + r"(?![A-Za-z0-9_.-])", lambda _: name, text)


def redact(text: str, home: str, work: str = "") -> str:
    """The run's work root becomes $WORK and the home path $HOME, but not inside a longer name such as /home/xy."""
    return _replace_path(_replace_path(text, work, "$WORK"), home, "$HOME")


def reduce_events(text: str, types: Sequence[str]) -> tuple[str, dict[str, int]]:
    """text without its JSON object lines of the given types, and how many of each it dropped."""
    dropped = {kind: 0 for kind in types}
    kept = []
    for line in text.splitlines(keepends=True):
        try:
            event = json.loads(line)
        except ValueError:
            event = None
        if isinstance(event, dict) and event.get("type") in dropped:
            dropped[event["type"]] += 1
        else:
            kept.append(line)
    return "".join(kept), dropped


def _strings(value: object, path: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], str]]:
    """Every string in value with its key path; list indices do not count, so a path names one kind of field."""
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _strings(item, (*path, str(key)))
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item, path)


def joined_strings(text: str) -> list[str]:
    """Every JSON string value of the lines that parse and every other line, in order and without separators, so a
    value split over several events reads whole again: once all together, and once per key path, because fields
    such as type would otherwise sit between the pieces."""
    every, by_path, lines = [], {}, []
    for line in text.splitlines():
        try:
            found = list(_strings(json.loads(line)))
        except ValueError:
            found = [((), line)]
            lines.append(line)
        for path, value in found:
            every.append(value)
            by_path.setdefault(path, []).append(value)
    return ["".join(every), "".join(lines), *("".join(values) for values in by_path.values())]


def private_values(env: Mapping[str, str], work: str) -> tuple[str, ...]:
    """The values a record must not name: the home path, the user name, the host name and the work root."""
    home = env.get("HOME", "").rstrip("/")
    values = (home, Path(home).name if home else "", env.get("USER", ""), env.get("LOGNAME", ""),
              socket.gethostname(), work.rstrip("/"))
    return tuple(dict.fromkeys(value for value in values if value))


def names_private(text: str, values: Sequence[str]) -> bool:
    texts = [text, *joined_strings(text)]
    return any(value in joined for value in values for joined in texts)


# Defaults, their restore and the gateway check (spec L3, L9)


def snapshot_defaults(env: Mapping[str, str]) -> Defaults:
    path, model, problem = opencode_config.current_default(env)
    if not isinstance(model, str):
        detail = problem or f"its model is {json.dumps(model)}"
        raise UsageError(f"the round trip needs OpenCode's top-level model in {path} ({detail}), because llmctl's "
                         "line edit can set that value but not remove it; set it, "
                         "or run without --set-default-round-trip")
    provider, pi_model = pi_config.pi_default(env)
    return Defaults(model, provider, pi_model, hermes_config.sticky_profile(env))


def _write(change: files.FileChange | None) -> None:
    if change is None:
        return
    files.make_dirs(change.dirs)
    if change.backup and change.old_text is not None:
        files.backup_once(change.path, change.old_text)
    files.write_atomic(change.path, change.new_text, new_mode=change.mode)


def _shown(value: object) -> str:
    return "unset" if value is None else str(value)


def _restore_pi(host, snapshot: Defaults) -> list[str]:
    path = pi_config.settings_path(host.env)
    with pi_config.settings_lock(host, path):
        _write(pi_config.plan_defaults(host.env, snapshot.pi_provider, snapshot.pi_model))
    found = pi_config.pi_default(host.env)
    return [f"Pi: {name} reads {_shown(value)}, expected {_shown(wanted)}"
            for name, value, wanted in zip(("defaultProvider", "defaultModel"), found,
                                           (snapshot.pi_provider, snapshot.pi_model)) if value != wanted]


def _restore_hermes(host, snapshot: Defaults) -> list[str]:
    wanted = snapshot.hermes_profile
    if hermes_config.sticky_profile(host.env) == wanted:
        return []
    command = ["hermes", "profile", "use", wanted]
    result = host.run(command)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        return [f"Hermes: {shlex.join(command)} exited {result.returncode}: {detail}"]
    found = hermes_config.sticky_profile(host.env)
    return [] if found == wanted else [f"Hermes: active profile reads {found}, expected {wanted}"]


def _restore_opencode(host, snapshot: Defaults) -> list[str]:
    _write(opencode_config.plan_set_model(host.env, snapshot.opencode_model))
    _, found, problem = opencode_config.current_default(host.env)
    if found == snapshot.opencode_model:
        return []
    return [f"OpenCode: model reads {problem or _shown(found)}, expected {snapshot.opencode_model}"]


def restore_defaults(host, snapshot: Defaults) -> list[str]:
    """Put each client's default back on its own, Pi, Hermes, OpenCode; one line per value not restored (L9)."""
    problems = []
    for label, restore in (("Pi", _restore_pi), ("Hermes", _restore_hermes), ("OpenCode", _restore_opencode)):
        try:
            problems += restore(host, snapshot)
        except (LlmctlError, OSError) as exc:
            problems.append(f"{label}: {exc}")
    return problems


def round_trip(host, check: Callable[[], None], report: Callable[[list[str]], None] | None = None) -> list[str]:
    """Snapshot the defaults, run check, and restore them even when check raises (spec L9).

    report, when given, gets the restore's problem lines on both paths, so a caller can record them although check
    raised."""
    snapshot = snapshot_defaults(host.env)
    try:
        check()
    except BaseException:
        problems = restore_defaults(host, snapshot)
        for line in problems:
            host.err.write(f"not restored: {line}\n")
        if report is not None:
            report(problems)
        raise
    problems = restore_defaults(host, snapshot)
    if report is not None:
        report(problems)
    return problems


def gateway_check(host, *, created: bool) -> str | None:
    """The failure text when Hermes' live gateway serves the profile llmctl (spec L3), else None."""
    if created:
        host.sleep(GATEWAY_WAIT_S)
    path = hermes_config.hermes_root(host.env) / "gateway_state.json"
    try:
        serves = gateway_serves_llmctl(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        return f"cannot read {path}: {exc}"
    return f"Hermes' gateway serves the profile llmctl ({path})" if serves else None


# The run


@dataclass
class _Row:
    recipe: str
    step: str
    passed: bool
    reasons: list[str]
    seconds: float


class _Run:
    """Runs the planned steps against the real clients and writes every record under --out."""

    def __init__(self, out: Path, catalog, host: RealHost, env: dict[str, str], work_root: Path):
        self.out = out
        self.work_root = work_root
        self.catalog = catalog
        self.host = host
        self.env = env
        self.home = env.get("HOME", "")
        self.private = private_values(env, str(work_root))
        self.rows: list[_Row] = []
        self.count = 0

    def record(self, name: str, text: str) -> None:
        self.count += 1
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-")
        (self.out / "records" / f"{self.count:03d}-{slug}").write_text(self.redact(text), encoding="utf-8")

    def redact(self, text: str) -> str:
        return redact(text, self.home, str(self.work_root))

    def command(self, name: str, argv: Sequence[str], cwd: Path, timeout_s: int, *, with_stderr: bool = False,
                drop: Sequence[str] = ()) -> tuple[int | None, str, float, bool]:
        """Run one command; (exit status or None on timeout, stdout, seconds, withheld), all of it recorded.

        with_stderr returns stdout followed by stderr instead; the record keeps the two apart.
        drop names the event types the recorded stdout leaves out; the returned stdout keeps them.
        withheld is True when the redacted output still named a private value, so the record holds neither stream.
        """
        start = time.monotonic()
        try:
            # OpenCode takes its working directory from PWD, so PWD has to name the folder the command runs in
            env = dict(self.env, PWD=str(Path(cwd).resolve()))
            done = subprocess.run(list(argv), cwd=cwd, env=env, capture_output=True, text=True,
                                  timeout=timeout_s)
            code, stdout, stderr = done.returncode, done.stdout, done.stderr
        except subprocess.TimeoutExpired as exc:
            code = None
            stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        except OSError as exc:
            code, stdout, stderr = 127, "", f"{argv[0]}: {exc}"
        seconds = time.monotonic() - start
        recorded, dropped = reduce_events(stdout, drop)
        recorded, recorded_err = self.redact(recorded), self.redact(stderr)
        withheld = names_private(recorded, self.private) or names_private(recorded_err, self.private)
        if withheld:
            recorded = recorded_err = WITHHELD
        record = {"command": shlex.join(argv), "cwd": str(cwd), "exit": code, "seconds": round(seconds, 1),
                  "stdout": recorded, "stderr": recorded_err}
        if drop:
            record["dropped_events"] = dropped
        self.record(name, json.dumps(record, indent=2, ensure_ascii=False) + "\n")
        return code, stdout + stderr if with_stderr else stdout, seconds, withheld

    def add(self, recipe: str, step: str, reasons: list[str], seconds: float) -> None:
        self.rows.append(_Row(recipe, step, not reasons, reasons, seconds))

    def work(self, step: Step) -> Path:
        if step.cwd == ".":
            return REPO
        folder = self.work_root / step.cwd
        folder.mkdir(parents=True)
        if step.argv[0] == "opencode":
            config = {"$schema": "https://opencode.ai/config.json", "permission": PERMISSIONS}
            (folder / "opencode.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        return folder

    def run_step(self, step: Step, target: Target) -> tuple[list[str], float]:
        """Run one planned step; (its failure reasons, seconds)."""
        kind = step.name.split(" ", 1)[1]
        start = time.monotonic()
        if kind == "config":
            return self.config_check(target), time.monotonic() - start
        if kind == "probes":
            return self.probes(step, target), time.monotonic() - start
        folder = self.work(step)
        argv = [part.replace(WORK, str(folder)) for part in step.argv]
        code, stdout, seconds, withheld = self.command(step.name, argv, folder, step.timeout_s,
                                                       drop=DROPPED_EVENTS.get(kind, ()))
        tools = not target.recipe.chat_only
        reasons = [] if code == 0 or kind == "pi" or kind == "hermes-chat" else [f"exit status {code}"]
        if withheld:  # the other reasons may quote the output that the record withheld
            return reasons + [WITHHELD_REASON], seconds
        if kind == "status":
            reasons += status_reasons(stdout, opencode=target.recipe.opencode)
        elif kind.startswith("hermes-config-"):
            reasons += self.config_value(kind[len("hermes-config-"):], stdout, target)
        elif kind in ("opencode", "hermes-z", "hermes-chat", "pi"):
            reasons += self.session(kind, stdout, folder, tools, step.name.startswith(ROUND_TRIP), target)
        return reasons, seconds

    def config_value(self, key: str, stdout: str, target: Target) -> list[str]:
        try:
            value = json.loads(stdout)
        except ValueError as exc:
            return [f"hermes config get {key} printed no JSON: {exc}"]
        wanted = hermes_config.block_values(target)[key]
        if _contains(value, wanted):
            return []
        return [f"{key} reads {json.dumps(value)}, expected it to contain {json.dumps(wanted)}"]

    def session(self, kind: str, stdout: str, folder: Path, tools: bool, round_trip_step: bool,
                target: Target) -> list[str]:
        lines = stdout.splitlines()
        if kind == "opencode":
            outcome = opencode_outcome(lines)
        elif kind == "hermes-chat":
            outcome = hermes_stream_outcome(lines, tools=tools)
        elif kind == "pi":
            outcome = pi_outcome(lines, tools=tools)
        else:
            usage = folder / "usage.json"
            outcome = (hermes_usage_outcome(usage.read_text(encoding="utf-8")) if usage.is_file()
                       else Outcome(False, (f"no usage file {usage}",), ()))
        reasons = list(outcome.reasons)
        if tools and not hello_valid(folder / "hello.py"):
            reasons.append("hello.py is missing or lacks def add and add(2, 3)")
        if round_trip_step and kind == "pi" and f"{pi_config.PROVIDER}/{target.served}" not in outcome.models:
            reasons.append(f"Pi answered from {', '.join(outcome.models) or 'no model'}, "
                           f"not {pi_config.PROVIDER}/{target.served}")
        if round_trip_step and kind == "hermes-z" and outcome.models != (f"custom/{target.served}",):
            reasons.append(f"Hermes answered from {', '.join(outcome.models) or 'no model'}, "
                           f"not custom/{target.served}")
        return reasons

    def config_check(self, target: Target) -> list[str]:
        """The in-process half of D10 step 2: the block, Pi's entry and Pi's level as llmctl wrote them."""
        env, reasons = self.env, []
        checks = (
            ("the Hermes block", lambda: hermes_config.profile_block(env), hermes_config.render_block(target)),
            ("Pi's providers.llmctl", lambda: pi_config.installed_entry(env), pi_config.provider_entry(target)),
            ("Pi's thinking level", lambda: pi_config.stored_level(env, target.served),
             pi_config.default_level(target.recipe.clients)),
        )
        for name, read, wanted in checks:
            try:
                found = read()
            except LlmctlError as exc:
                reasons.append(f"{name}: {exc}")
                continue
            if found != wanted:
                reasons.append(f"{name} differs from the generator")
                self.record(f"config-{name}", json.dumps({"found": found, "wanted": wanted}, indent=2) + "\n")
        return reasons

    def probes(self, step: Step, target: Target) -> list[str]:
        url = f"http://127.0.0.1:{target.runtime.host_port}/v1/chat/completions"
        reasons = []
        for name, body in probe_bodies(target).items():
            try:
                response = self.host.http("POST", url, headers={"Content-Type": "application/json"},
                                          body=json.dumps(body).encode("utf-8"), timeout=step.timeout_s)
                status, reply = response.status, response.body
            except (OSError, LlmctlError) as exc:
                status, reply = 0, str(exc).encode("utf-8")
            record = probe_record(name, status, reply)
            self.record(f"{step.name} {name}", json.dumps({"request": body, "status": status,
                                                            "reply": reply.decode("utf-8", errors="replace"),
                                                            "record": record}, indent=2, ensure_ascii=False) + "\n")
            if name == "A" and not (status == 200 and record["title_ok"]):
                reasons.append(f"probe A: status {status}, title_ok {record['title_ok']}")
            if name == "B" and status != 200:
                reasons.append(f"probe B: status {status}")
        return reasons

    def logs_since(self, since: str, target: Target) -> list[str]:
        # docker logs replays the container's stderr on stderr, and llama.cpp writes its log there
        code, text, _, withheld = self.command(f"{ROUND_TRIP} docker-logs",
                                               ["docker", "logs", "--since", since, target.runtime.container_name],
                                               REPO, STEP_TIMEOUT_S, with_stderr=True)
        if code != 0:
            return [f"docker logs exited {code}"]
        if withheld:
            return [f"docker logs: {WITHHELD_REASON}"]
        if "/v1/chat/completions" in text or "print_timing" in text:
            return []
        return ["the llama.cpp log shows no request since the OpenCode run started"]


def _steps_by_prefix(steps: list[Step]) -> dict[str, list[Step]]:
    groups: dict[str, list[Step]] = {}
    for step in steps:
        groups.setdefault(step.name.split(" ", 1)[0], []).append(step)
    return groups


def _summary(rows: list[_Row]) -> str:
    lines = ["# llmctl clients e2e", "", "| Recipe | Step | Pass | Reasons | Seconds |", "|---|---|---|---|---|"]
    for row in rows:
        reasons = "; ".join(row.reasons).replace("|", "/").replace("\n", " ")
        lines.append(f"| {row.recipe} | {row.step} | {'Yes' if row.passed else 'No'} | {reasons} | {row.seconds:.0f} |")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python3 -m launcher.e2e.clients",
                                     description="Opt-in e2e of llmctl's clients; run it only through heavy, alone.")
    parser.add_argument("--out", required=True, help="a new folder for the records, such as "
                        "research/<date>/llmctl-clients-e2e")
    parser.add_argument("--matrix", choices=sorted(MATRICES), default="default", help="the recipes to run")
    parser.add_argument("--keep-running", action="store_true", help="leave the last model running")
    parser.add_argument("--set-default-round-trip", action="store_true",
                        help="after the matrix, run up --set-default, check every client, then restore the defaults")
    return parser


def _run_matrix(run: _Run, steps: list[Step], catalog, matrix) -> None:
    groups = _steps_by_prefix(steps)
    profile = hermes_config.profile_config(run.env)
    existed_before_run = profile.exists()
    gateway_pending = True  # until an up that passed gives the gateway a profile llmctl to check (spec L3)
    for number, (recipe_id, runtime) in enumerate(matrix, 1):
        target = _target(catalog, recipe_id, runtime)
        prefix = f"{number:02d}-{recipe_id}-{runtime}"
        existed = profile.exists()
        for step in groups[prefix]:
            reasons, seconds = run.run_step(step, target)
            kind = step.name.split(" ", 1)[1]
            run.add(prefix, kind, reasons, seconds)
            if kind == "up" and gateway_pending and not reasons and (existed_before_run or profile.exists()):
                gateway_pending = False
                found = gateway_check(run.host, created=not existed)
                run.add(prefix, "gateway", [found] if found else [], 0.0)
            if kind == "up" and reasons:
                break  # no server, so the client steps would only wait for their timeouts
    if gateway_pending:
        run.add("-", "gateway", [f"no up created the profile llmctl ({profile}), so the gateway check did not run"],
                0.0)


def _run_round_trip(run: _Run, steps: list[Step], catalog) -> None:
    target = _target(catalog, ROUND_TRIP_MODEL, "llamacpp")

    def check() -> None:
        for step in steps:
            kind = step.name.split(" ", 1)[1]
            since = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            reasons, seconds = run.run_step(step, target)
            if kind == "opencode":
                reasons += run.logs_since(since, target)
            run.add(ROUND_TRIP, kind, reasons, seconds)
            if kind == "up" and reasons:
                return

    restored: list[list[str]] = []
    try:
        round_trip(run.host, check, report=restored.append)
    except (LlmctlError, OSError) as exc:
        run.add(ROUND_TRIP, "check", [f"{type(exc).__name__}: {exc}"], 0.0)
    run.add(ROUND_TRIP, "restore", restored[0] if restored else ["the restore did not run"], 0.0)


def running_servers(host) -> str | None:
    """`<recipe> on <runtime>` of each llmctl container that runs, None when none runs."""
    running = [container for container in docker_state.managed(host) if container.state == "running"]
    if not running:
        return None
    return ", ".join(f"{c.labels.get(LABEL_PREFIX + 'recipe', '?')} on {c.labels.get(LABEL_PREFIX + 'runtime', '?')}"
                     for c in running)


def round_trip_refusal(catalog, env: Mapping[str, str]) -> str | None:
    """Why an unpinned session of the round trip would go to another model than llmctl's, else None (spec L9)."""
    note = hermes_config.hermes_home_note(env)
    if note is not None:
        name = Path(env["HERMES_HOME"]).name
        return (f"the round trip runs hermes without -p, but HERMES_HOME names the profile {name} in this shell, so "
                "that session would use another model; unset HERMES_HOME, or run without --set-default-round-trip")
    served = _target(catalog, ROUND_TRIP_MODEL, "llamacpp").served
    _, settings = files.load_json_object(pi_config.settings_path(env), "Pi")
    first = pi_config.enabled_models_skip(settings, served)
    if first is not None:
        own = f"{pi_config.PROVIDER}/{served}"
        return (f"the round trip runs pi without --model, but Pi's enabledModels would start it with {first} instead "
                f"of {own}; add {own} to enabledModels or clear it, or run without --set-default-round-trip")
    return None


def main(argv: list[str] | None = None, *, make_host: Callable[[dict[str, str]], RealHost] = RealHost) -> int:
    args = build_parser().parse_args(argv)
    try:
        if "LLMCTL_CLIENTS" in os.environ:
            raise UsageError("unset LLMCTL_CLIENTS: the e2e checks all three clients")
        env = driver_env(os.environ)
        os.environ.pop("LLMCTL_SET_DEFAULT", None)  # RealHost.run passes os.environ to hermes profile use
        host = make_host(env)
        catalog = load_catalog(REPO / "recipes", override_dir(env))
        matrix = MATRICES[args.matrix]
        steps = plan_steps(catalog, matrix, round_trip=args.set_default_round_trip, keep_running=args.keep_running)
        running = running_servers(host)
        if running is not None:  # the first up --replace would stop the owner's server
            raise UsageError(f"an llmctl model server is running ({running}); stop it with ./llmctl down, "
                             "then run the driver again")
        if args.set_default_round_trip:
            snapshot_defaults(env)  # refuse before the first server starts when OpenCode has no model to restore
            refusal = round_trip_refusal(catalog, env)
            if refusal is not None:
                raise UsageError(refusal)
        out = Path(args.out).resolve()  # the clients run in other folders, so a relative path would move
        out.mkdir(parents=True)
        (out / "records").mkdir()
        work_root = Path(tempfile.mkdtemp(prefix="llmctl-e2e-"))
        os.chmod(work_root, 0o700)
    except (LlmctlError, OSError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1
    sys.stdout.write(f"records in {out}; work folders in {work_root}, left in place after the run\n")
    sys.stdout.flush()
    run = _Run(out, catalog, host, env, work_root)
    for argv_ in (["hermes", "--version"], ["pi", "--version"], ["opencode", "--version"],
                  ["nvidia-smi", "--query-gpu=power.limit", "--format=csv,noheader"]):
        if run.command(argv_[0] + " version", argv_, REPO, 60)[3]:
            sys.stderr.write(f"{argv_[0]} version: {WITHHELD_REASON}\n")
    try:
        _run_matrix(run, [s for s in steps if s.name.split(" ", 1)[0] not in (ROUND_TRIP, "down")], catalog, matrix)
        if args.set_default_round_trip:
            _run_round_trip(run, [s for s in steps if s.name.startswith(ROUND_TRIP + " ")], catalog)
    finally:
        for step in (s for s in steps if s.name == "down"):
            code, _, seconds, withheld = run.command(step.name, step.argv, REPO, step.timeout_s)
            reasons = ([] if code == 0 else [f"exit status {code}"]) + ([WITHHELD_REASON] if withheld else [])
            run.add("-", "down", reasons, seconds)
        (out / "summary.md").write_text(run.redact(_summary(run.rows)), encoding="utf-8")
    return 0 if all(row.passed for row in run.rows) else 1


if __name__ == "__main__":
    sys.exit(main())
