"""The opt-in e2e of llmctl's clients (spec D10, L3, L9, dsh design D9, Kimi design D11): real servers, real clients,
real config files.

Only the lead runs it, alone and through heavy, after the full unit suite passes:
heavy python3 -m launcher.e2e.clients --out research/<date>/llmctl-clients-e2e [--matrix default|extended|all]
[--keep-running] [--set-default-round-trip] [--skip-dsh] [--skip-kimi]
It checks five clients, OpenCode, Hermes, Pi, dsh and Kimi; --skip-dsh and --skip-kimi leave one out each, and with
either LLMCTL_CLIENTS names the clients that stay for every command.
Without --skip-kimi it puts <HOME>/.kimi-code/bin first on PATH for itself and every command.
Without --skip-dsh it refuses before anything starts when dsh is not installed, a dsh profile patch sets the
llm-pi-ai row, DEEPSEEK_API_KEY is set, 127.0.0.1:3182 is in use, or the dsh home patch is unreadable or has damaged
markers; it opens no .env or credential file.
Without --skip-kimi it refuses when Kimi is not installed, a KIMI_MODEL_ variable is set, rg or both fd and fdfind
are missing from PATH, Kimi's config.toml is unreadable or not valid TOML, or its default_model is in a form that
llmctl's default edit cannot change, and with --set-default-round-trip when its default_model is not a string; it
reads only config.toml in the Kimi home.
--out holds only the records; the clients run in fresh work folders under one private temporary folder per run
(llmctl-e2e-*), outside the repository, which the run leaves in place for inspection.
Each recipe's dsh steps run with telemetry off, TMPDIR under that folder and a private dsh home there that holds a
byte copy of the real home patch; the round trip's dsh session runs in the real dsh home.
Each Kimi command runs with a private Kimi home under that folder, an empty HOME, TMPDIR there, one Kimi cache per
run and the three switches; a recipe's session gets a copy of the real config.toml with llmctl's default edit when
its default_model does not name the recipe, and the round trip's session a pure byte copy.
Probe G, the OK request at reasoning_effort low, runs for every recipe and is recorded without judging.
Every SGLang recipe gets the long-grammar step right after its status step: a long prompt, then grammar requests,
then the readiness check (SGLang design D15); a server that stopped answering skips the recipe's remaining steps.
Importing this module has no side effects; everything runs inside main and the functions below.
"""

import argparse
import json
import math
import os
import re
import secrets
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from launcher import (
    clients,
    docker_state,
    dsh_config,
    files,
    gpu,
    hermes_config,
    kimi_config,
    opencode_config,
    pi_config,
)
from launcher.catalog import load_catalog, override_dir
from launcher.command import LABEL_PREFIX
from launcher.e2e.summary import SUMMARY_HEAD
from launcher.errors import ClientError, LlmctlError, UsageError
from launcher.host import RealHost
from launcher.model import Target
from launcher.ops import client_target
from launcher.runtimes import ADAPTERS
from launcher.smoke import WEATHER_TOOL

REPO = Path(__file__).resolve().parents[2]
TASK = ("Create a file named hello.py containing a function add(a, b) that returns a + b, and a main block that "
        "prints add(2, 3). Then read the file back and tell me how many lines it has. Do not run any shell commands.")
CHAT = "Say hello in one sentence."
PERMISSIONS = {name: "deny" for name in ("bash", "external_directory", "webfetch", "websearch", "task", "question")}
STEP_TIMEOUT_S = 900
GATEWAY_WAIT_S = 35
DEFAULT_MATRIX = (("qwen3.8-27b", "llamacpp"), ("k2-horizon-7b", "llamacpp"), ("gpt-oss-20b", "vllm"),
                  ("qwen3.5-9b", "sglang"), ("qwen3.6-35b-a3b", "llamacpp"))
EXTENDED_MATRIX = DEFAULT_MATRIX[:-1] + tuple((name, "llamacpp") for name in (
    "nemotron-3.5-bolt", "qwen3-coder-next", "deepseek-r1-0528-qwen3-8b", "gemma-4-26b-a4b",
    "nemotron-3.5-lightning", "groot-19b-a3b", "k2-horizon-mova-36b", "gpt-oss-20b")) + DEFAULT_MATRIX[-1:]
# L8 item 5: every shipped recipe, the llama.cpp ones first, then vLLM, then SGLang, ending on the default model as
# extended does
ALL_MATRIX = tuple((name, "llamacpp") for name in (
    "qwen3.8-27b", "k2-horizon-7b", "nemotron-3.5-bolt", "qwen3-coder-next", "deepseek-r1-0528-qwen3-8b",
    "gemma-4-26b-a4b", "nemotron-3.5-lightning", "groot-19b-a3b", "k2-horizon-mova-36b", "gpt-oss-20b", "qwen3.5-9b",
    "gemma-4-12b", "gemma-4-e4b", "mellum2.1-12b")) + tuple(
    (name, "vllm") for name in ("gpt-oss-20b", "mellum2.1-12b", "k2-horizon-7b", "qwen3.5-9b", "gemma-4-12b",
                                "gemma-4-e4b", "groot-19b-a3b", "deepseek-r1-0528-qwen3-8b")) + tuple(
    (name, "sglang") for name in ("qwen3.5-9b", "k2-horizon-7b", "gpt-oss-20b", "gemma-4-12b", "gemma-4-e4b",
                                  "qwen3.6-35b-a3b")) + (
    ("qwen3.6-35b-a3b", "llamacpp"),)
MATRICES = {"default": DEFAULT_MATRIX, "extended": EXTENDED_MATRIX, "all": ALL_MATRIX}
ROUND_TRIP_MODEL = "qwen3.6-35b-a3b"
PROBE_C = ("k2-horizon-7b", "k2-horizon-mova-36b", "qwen3.6-35b-a3b")
PROBE_F = "nemotron-3.5-bolt"  # dsh's Medium level on Bolt's real template (dsh design D9 step 5)
DSH_PORT = 3182  # dsh web's port in the e2e; 3080 is the owner's
WEB_READY_S = 30
WEB_TAIL_S = 3
WEB_KILL_S = 15
BASE_CLIENTS = "opencode,hermes,pi"  # the clients every run checks; dsh and Kimi follow unless skipped
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
SERVER_STOPPED = "the server stopped answering after the long-prompt-then-grammar requests"
# The fixed code paragraph that the long-grammar step repeats to fill the context (SGLang design D15)
LONG_PARAGRAPH = ("def merge_sorted(left, right):\n"
                  "    # merge two sorted lists into one sorted list\n"
                  "    result, i, j = [], 0, 0\n"
                  "    while i < len(left) and j < len(right):\n"
                  "        if left[i] <= right[j]:\n"
                  "            result.append(left[i])\n"
                  "            i += 1\n"
                  "        else:\n"
                  "            result.append(right[j])\n"
                  "            j += 1\n"
                  "    return result + left[i:] + right[j:]\n"
                  "\n")
_FORCED_TOOL_PROMPT = "What is the weather in Paris? Use the get_weather tool."


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
    dsh_default: bool | None = None  # None when the round trip leaves dsh out
    kimi_default: str | None = None  # None when Kimi has no default_model, or the round trip leaves Kimi out
    kimi_read: bool = False  # whether the snapshot read Kimi's default_model


@dataclass(frozen=True)
class WebBoot:
    line: str | None  # the `dsh web:` line, or None when none came within WEB_READY_S
    stdout: str
    stderr: str
    code: int | None


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


def opencode_outcome(lines: Iterable[str], *, tools: bool = True) -> Outcome:
    """`opencode run --format json`: a completed write and read tool_use, and no error event (spec D10 step 3);
    without tools a non-empty text answer and no tool_use event (L8 item 5)."""
    events, reasons = _events(lines)
    done, models = set(), []
    used = answered = False
    for event in events:
        part = _dict(event.get("part"))
        for holder in (event, part, _dict(part.get("info"))):
            _add_model(models, holder.get("providerID"), holder.get("modelID"))
        if event.get("type") == "tool_use":
            used = True
            if _dict(part.get("state")).get("status") == "completed":
                done.add(part.get("tool"))
        if event.get("type") == "text" and _non_empty(part.get("text")):
            answered = True
        if event.get("type") == "error":
            reasons.append(f"error event: {json.dumps(event)[:300]}")
    if tools:
        for tool in ("write", "read"):
            if tool not in done:
                reasons.append(f"no completed {tool} tool_use")
    else:
        if not answered:
            reasons.append("no text answer")
        if used:
            reasons.append("a tool_use event in a chat-only session")
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


def dsh_outcome(lines: Iterable[str], *, tools: bool = True) -> Outcome:
    """`dsh --profile headless ... --json`: no error event, every tool_result completed, a final event that
    completed, and with tools a completed write and read (dsh design D9, L6); dsh's events do not name the model."""
    events, reasons = _events(lines)
    errors, results = [], []
    called: dict[object, object] = {}
    done, used, final = set(), False, None
    for event in events:
        kind = event.get("type")
        if kind == "error":
            errors.append(f"error event: {json.dumps(event)[:300]}")
        elif kind == "tool_call":
            used = True
            called[event.get("callId")] = event.get("tool")
        elif kind == "tool_result":
            used = True
            tool, status = called.get(event.get("callId"), "?"), event.get("status")
            if status == "completed":
                done.add(tool)
            else:
                code = _dict(event.get("error")).get("code") or "?"
                results.append(f"tool_result {tool} has status {status} ({code})")
        elif kind == "final":
            final = event
    reasons += errors + results
    if final is None:
        reasons.append("no final event")
    else:
        reason = _dict(final.get("reason"))
        if reason.get("kind") != "completed":
            error = _dict(reason.get("error"))
            reasons.append(f"final reason {reason.get('kind')} ({error.get('code') or '?'}: "
                           f"{error.get('message') or '?'})")
    if tools:
        if not done & {"write", "apply_patch", "edit"}:
            reasons.append("no completed write, apply_patch or edit tool_result")
        if "read" not in done:
            reasons.append("no completed read tool_result")
    else:
        if used:
            reasons.append("a tool event in a chat-only session")
        if final is not None and not _non_empty(final.get("text")):
            reasons.append("the final text is empty")
    return _outcome(reasons, [])


def kimi_outcome(lines: Iterable[str], *, tools: bool = True) -> Outcome:
    """`kimi -p ... --output-format stream-json` (Kimi design D11): no retried request, with tools a Write and a Read
    call, each answered by a tool line, and no Bash call, without tools no tool call, and a last assistant message
    with text; Kimi's lines do not name the model, and a tool line carries no error flag."""
    events, reasons = _events(lines)
    retried = None
    calls: list[tuple[object, object]] = []
    answered = set()
    assistants = []
    for event in events:
        role = event.get("role")
        if role == "meta" and event.get("type") == "turn.step.retrying":
            retried = event
        elif role == "assistant":
            assistants.append(event)
            tool_calls = event.get("tool_calls")
            for call in tool_calls if isinstance(tool_calls, list) else ():
                call = _dict(call)
                calls.append((_dict(call.get("function")).get("name"), call.get("id")))
        elif role == "tool":
            answered.add(event.get("tool_call_id"))
    if retried is not None:
        reasons.append(f"Kimi retried a request {retried.get('failed_attempt', '?')} times "
                       f"({retried.get('error_name', '?')}: {retried.get('error_message', '?')})")
    names = [name for name, _ in calls]
    if tools:
        if "Bash" in names:
            reasons.append("a Bash tool call, which the task forbids")
        reasons += [f"tool call {name} ({call_id}) has no tool line" for name, call_id in calls
                    if call_id not in answered]
        for tool in ("Write", "Read"):
            if tool not in names:
                reasons.append(f"no {tool} tool call")
    elif any(event.get("tool_calls") for event in assistants):
        reasons.append("a tool call in a chat-only session")
    if not assistants:
        reasons.append("no assistant message")
    elif not _non_empty(assistants[-1].get("content")):
        reasons.append("the last assistant message has no text")
    return _outcome(reasons, [])


WEB_LINE = "dsh web:"
WEB_SHOWN = "dsh web: <address withheld>"


def web_line(line: str) -> tuple[str, str | None]:
    """The `dsh web:` line as a record may show it, and the launch token of its URL, or None without one."""
    if not line.startswith(WEB_LINE):
        raise ValueError(f"not a {WEB_LINE!r} line")
    url = line[len(WEB_LINE):].strip()
    tokens = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query).get("token")
    return WEB_SHOWN, tokens[0] if tokens else None


def web_outcome(boot: WebBoot) -> Outcome:
    """`dsh web`: a ready line within WEB_READY_S, and no stderr line saying that an entry did not activate (D9)."""
    reasons = [] if boot.line is not None else [f"no {WEB_LINE!r} line within {WEB_READY_S} s"]
    reasons += [line.strip() for line in boot.stderr.splitlines() if "did not activate" in line]
    return _outcome(reasons, [])


def _signal(killpg: Callable[[int, int], None], pid: int, sig: int) -> None:
    try:
        killpg(pid, sig)
    except ProcessLookupError:
        pass  # the group is already gone


def _stop(proc, killpg: Callable[[int, int], None], kill_s: float) -> int | None:
    """SIGTERM to the process group unless the process exited; SIGKILL when it lingers kill_s later."""
    if proc.poll() is None:
        _signal(killpg, proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=kill_s)
        except subprocess.TimeoutExpired:
            _signal(killpg, proc.pid, signal.SIGKILL)
            proc.wait()
    return proc.returncode


def web_boot(argv: Sequence[str], *, cwd: Path, env: Mapping[str, str], popen=subprocess.Popen, killpg=os.killpg,
             sleep=time.sleep, timer=threading.Timer, ready_s: float = WEB_READY_S, tail_s: float = WEB_TAIL_S,
             kill_s: float = WEB_KILL_S) -> WebBoot:
    """Start `dsh web` in its own process group, wait up to ready_s for its `dsh web:` line, read stderr for tail_s
    more, and stop the group, also on an interrupt (dsh design D9 step 4); without the line by ready_s the group gets
    SIGTERM, and SIGKILL kill_s later when it is still alive; only a failure to start is exit 127, and an error after
    the start propagates."""
    try:
        proc = popen(list(argv), cwd=cwd, env=dict(env), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE, text=True, start_new_session=True)
    except OSError as exc:
        return WebBoot(None, "", f"{argv[0]}: {exc}", 127)
    err: list[str] = []
    reader = threading.Thread(target=lambda: err.extend(iter(proc.stderr.readline, "")), daemon=True)
    reader.start()
    escalations: list = []

    def kill_if_alive() -> None:
        if proc.poll() is None:
            _signal(killpg, proc.pid, signal.SIGKILL)

    def escalate() -> None:
        # Ends a stdout that never says ready: SIGTERM, then SIGKILL kill_s later for a group that ignores it, so the
        # readline below returns even then.
        _signal(killpg, proc.pid, signal.SIGTERM)
        escalation = timer(kill_s, kill_if_alive)
        escalations.append(escalation)
        escalation.start()

    alarm = timer(ready_s, escalate)
    alarm.start()
    out: list[str] = []
    line = None
    try:
        for text in iter(proc.stdout.readline, ""):
            out.append(text)
            if text.startswith(WEB_LINE):
                line = text.rstrip("\n")
                break
        alarm.cancel()
        if line is not None:
            sleep(tail_s)  # stderr keeps arriving in the reader
    finally:
        alarm.cancel()
        for escalation in escalations:
            escalation.cancel()
        code = _stop(proc, killpg, kill_s)
        reader.join(kill_s)
    out.append(proc.stdout.read())
    return WebBoot(line, "".join(out), "".join(err), code)


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


def status_reasons(text: str, *, opencode: bool, dsh: bool = False, kimi: bool = False) -> list[str]:
    """One reason per client line of ./llmctl status that does not say the running model (spec D10 step 1), and
    the gateway warning (spec L3)."""
    lines = {}
    if "clients:\n" in text:
        for line in text.split("clients:\n", 1)[1].splitlines():
            if line.startswith("  ") and ": " in line:
                label, rest = line.strip().split(": ", 1)
                lines[label] = rest
    reasons = []
    for label in (("OpenCode",) if opencode else ()) + ("Hermes", "Pi") + (("dsh",) if dsh else ()) + (
            ("Kimi",) if kimi else ()):
        if label not in lines:
            reasons.append(f"no {label} line in status")
        elif "the running model" not in lines[label] or "not the running model" in lines[label]:
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
    if recipe.opencode:
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


def _dsh_headless(prompt: str, chat_only: bool, *, pinned: bool) -> tuple[str, ...]:
    """dsh's headless session; its launcher flags come before --json, which dsh hands to the app (plan, Spec)."""
    patches = ()
    if pinned:
        patches = ("--patch", f"{WORK}/llmctl-default.patch.yml")
        if chat_only:
            patches += ("--patch", f"{WORK}/llmctl-no-tools.patch.yml")
    return ("dsh", "--profile", "headless", *patches, "--json", prompt)


def _kimi_session(target: Target, prompt: str, chat_only: bool, *, pinned: bool) -> tuple[str, ...]:
    """Kimi's print-mode session in E3's order: -m when pinned, then --agent-file for a chat-only recipe (D11, L-k)."""
    model = ("-m", kimi_config.alias(target)) if pinned else ()
    agent = ("--agent-file", str(kimi_config.AGENT_FILE)) if chat_only else ()
    return ("kimi", "-p", prompt, "--output-format", "stream-json", *model, *agent)


def plan_steps(catalog, matrix: Sequence[tuple[str, str]], *, round_trip: bool, keep_running: bool,
               dsh: bool = False, kimi: bool = False) -> list[Step]:
    """Every step of a run, in order (spec D10, L3, L9, dsh design D9, Kimi design D11); argv () marks a step the
    driver runs in-process."""
    steps = []

    def add(name: str, argv: tuple[str, ...], cwd: str = ".", timeout_s: int = STEP_TIMEOUT_S) -> None:
        steps.append(Step(name, argv, cwd, timeout_s))

    def up_timeout(target: Target) -> int:
        # llmctl's own start timeout, which prints the log tail, fires before the driver's (SGLang design D15)
        return max(STEP_TIMEOUT_S, target.runtime.start_timeout_s + 120)

    for number, (recipe_id, runtime) in enumerate(matrix, 1):
        target = _target(catalog, recipe_id, runtime)
        prefix = f"{number:02d}-{recipe_id}-{runtime}"
        tools = not target.recipe.chat_only
        add(f"{prefix} up", up_argv(target, "--replace"), timeout_s=up_timeout(target))
        add(f"{prefix} status", ("./llmctl", "status"))
        if target.recipe.runtime == "sglang":
            add(f"{prefix} long-grammar", IN_PROCESS)
        add(f"{prefix} config", IN_PROCESS)
        for key in hermes_config.block_values(target):
            get = ("hermes", "-p", hermes_config.PROFILE, "config", "get", key, "--json")
            add(f"{prefix} hermes-config-{key}", get, f"{prefix}-hermes-config-{key}")
        for kind, argv in _session_argv(target, TASK if tools else CHAT, tools, pinned=True).items():
            add(f"{prefix} {kind}", argv, f"{prefix}-{kind}")
        if dsh:
            add(f"{prefix} dsh-config", IN_PROCESS)
            add(f"{prefix} dsh-headless", _dsh_headless(TASK if tools else CHAT, not tools, pinned=True),
                f"{prefix}-dsh-headless")
            add(f"{prefix} dsh-web", ("dsh", "web", "--port", str(DSH_PORT), "--no-open"), f"{prefix}-dsh-web")
        if kimi:
            add(f"{prefix} kimi-config", IN_PROCESS)
            add(f"{prefix} kimi", _kimi_session(target, TASK if tools else CHAT, not tools, pinned=True),
                f"{prefix}-kimi")
        add(f"{prefix} probes", IN_PROCESS)
    if round_trip:
        target = _target(catalog, ROUND_TRIP_MODEL, "llamacpp")
        add(f"{ROUND_TRIP} up", up_argv(target, "--set-default"), timeout_s=up_timeout(target))
        for kind, argv in _session_argv(target, TASK, True, pinned=False).items():
            add(f"{ROUND_TRIP} {kind}", argv, f"{ROUND_TRIP}-{kind}")
        if dsh:
            add(f"{ROUND_TRIP} dsh-headless", _dsh_headless(TASK, False, pinned=False), f"{ROUND_TRIP}-dsh-headless")
        if kimi:
            add(f"{ROUND_TRIP} kimi", _kimi_session(target, TASK, False, pinned=False), f"{ROUND_TRIP}-kimi")
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
    """The probe requests for one recipe by name: A and B always, except that a chat-only recipe gets no B, then C,
    D, E and F where they apply, and G, the OK request at Kimi's effort low, always (Kimi design LR3)."""
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
    # no e2e session sends a chat-only recipe tools; vLLM answers them with HTTP 400, and llama.cpp and SGLang
    # leave a call as text
    if recipe.chat_only:
        del bodies["B"]
    plain = {"model": served, "messages": [{"role": "user", "content": _OK_PROMPT}], "max_tokens": 256}
    if recipe.id in PROBE_C:
        bodies["C"] = {"model": served, "messages": plain["messages"], "reasoning_effort": "none", "max_tokens": 256}
    if recipe.runtime in ("llamacpp", "sglang"):
        bodies["D"] = dict(plain, model=recipe.id)
    if recipe.runtime == "llamacpp" and recipe.id == "k2-horizon-7b":
        bodies["E"] = dict(plain, chat_template_kwargs={"tool_call_format": "xml"})
    if recipe.id == PROBE_F:
        bodies["F"] = dict(plain, chat_template_kwargs={"enable_thinking": True, "medium_effort": 1})
    bodies["G"] = dict(plain, reasoning_effort="low")  # recorded, never judged (Kimi plan ruling 1)
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


def long_grammar_copies(context: int, calibration_tokens: int) -> int:
    """The paragraph copies of the long request: 90% of the context at the calibration's tokens per copy."""
    return math.floor(0.9 * context / (calibration_tokens / 20))


def long_prompt(line: str, copies: int) -> str:
    """A session line, a newline, then copies of the fixed code paragraph."""
    return line + "\n" + LONG_PARAGRAPH * copies


# Environment and records


def _contains(value: object, wanted: object) -> bool:
    """Whether value holds wanted: every key of a wanted dict, recursively; an empty dict or anything else equal,
    with JSON types."""
    if isinstance(wanted, dict) and wanted:
        return isinstance(value, dict) and all(key in value and _contains(value[key], item)
                                               for key, item in wanted.items())
    return type(value) is type(wanted) and value == wanted


def driver_env(environ: Mapping[str, str], *, skip_dsh: bool = False, skip_kimi: bool = False) -> dict[str, str]:
    """The environment of every command the driver runs: LLMCTL_SET_DEFAULT removed (spec D10), unless skip_kimi
    <HOME>/.kimi-code/bin first on PATH, and with skip_dsh or skip_kimi LLMCTL_CLIENTS naming the clients that stay
    (dsh design D9, Kimi design D11)."""
    env = {name: value for name, value in environ.items() if name != "LLMCTL_SET_DEFAULT"}
    if not skip_kimi:
        kimi_bin = str(files.home_dir(environ) / ".kimi-code" / "bin")
        env["PATH"] = kimi_bin + os.pathsep + env["PATH"] if "PATH" in env else kimi_bin
    if skip_dsh or skip_kimi:
        env["LLMCTL_CLIENTS"] = ",".join([BASE_CLIENTS] + ([] if skip_dsh else ["dsh"]) +
                                         ([] if skip_kimi else ["kimi"]))
    return env


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


def snapshot_defaults(env: Mapping[str, str], *, dsh: bool = False, kimi: bool = False) -> Defaults:
    path, model, problem = opencode_config.current_default(env)
    if not isinstance(model, str):
        detail = problem or f"its model is {json.dumps(model)}"
        raise UsageError(f"the round trip needs OpenCode's top-level model in {path} ({detail}), because llmctl's "
                         "line edit can set that value but not remove it; set it, "
                         "or run without --set-default-round-trip")
    provider, pi_model = pi_config.pi_default(env)
    kimi_default = None
    if kimi:
        state = kimi_config.read_state(env)
        value = kimi_config.MISSING if state is None else state.default
        if value is not kimi_config.MISSING and not isinstance(value, str):
            raise UsageError(f"Kimi's default_model in {kimi_config.config_path(env)} is "
                             f"{json.dumps(value, default=str)}, not a string, so the round trip could not write it "
                             "back; fix it, or run without --set-default-round-trip")
        kimi_default = None if value is kimi_config.MISSING else value
    return Defaults(model, provider, pi_model, hermes_config.sticky_profile(env),
                    dsh_config.holds_default(env) if dsh else None, kimi_default, kimi)


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


def _restore_dsh(host, snapshot: Defaults, target: Target | None) -> list[str]:
    """Release dsh's default when llmctl's block did not hold it before, or check that it still does (D9)."""
    if snapshot.dsh_default:
        held = dsh_config.holds_default(host.env)
        return [] if held else ["dsh: the llmctl block no longer holds dsh's default"]
    if target is None:
        return ["dsh: no target to release the default"]
    _write(dsh_config.plan_release_default(host.env, target))
    return ["dsh: the llmctl block still holds dsh's default"] if dsh_config.holds_default(host.env) else []


def _kimi_shown(value: object) -> str:
    return "unset" if value is kimi_config.MISSING else str(value)


def _kimi_recorded(value: object) -> object:
    """A default_model value as a record holds it: null when the file has none."""
    return None if value is kimi_config.MISSING else value


def _same_default(found: object, wanted: object) -> bool:
    if found is kimi_config.MISSING or wanted is kimi_config.MISSING:
        return found is wanted
    return type(found) is type(wanted) and found == wanted


def _withhold_key(table: object) -> object:
    """A table as a record may show it: an api_key other than llmctl's own is written <withheld>."""
    if isinstance(table, dict) and "api_key" in table and table["api_key"] != kimi_config.API_KEY:
        return dict(table, api_key="<withheld>")
    return table


def _restore_kimi(host, snapshot: Defaults) -> list[str]:
    """Put Kimi's default_model back, or remove it when the owner had none, then read it back (Kimi design D11)."""
    change = kimi_config.plan_default(host.env, snapshot.kimi_default)
    _write(change)
    if change is not None:
        kimi_config.read_back(change)
    state = kimi_config.read_state(host.env)
    found = kimi_config.MISSING if state is None else state.default
    wanted = kimi_config.MISSING if snapshot.kimi_default is None else snapshot.kimi_default
    if found is wanted or (found is not kimi_config.MISSING and wanted is not kimi_config.MISSING
                           and found == wanted):
        return []
    return [f"Kimi: default_model reads {_kimi_shown(found)}, expected {_kimi_shown(wanted)}"]


def restore_defaults(host, snapshot: Defaults, *, target: Target | None = None) -> list[str]:
    """Put each client's default back on its own, Pi, Hermes, OpenCode, then dsh and Kimi when the snapshot read
    them; one line per value not restored (L9, dsh design D9, Kimi design D11)."""
    problems = []
    steps = [("Pi", _restore_pi), ("Hermes", _restore_hermes), ("OpenCode", _restore_opencode)]
    if snapshot.dsh_default is not None:
        steps.append(("dsh", lambda host_, snapshot_: _restore_dsh(host_, snapshot_, target)))
    if snapshot.kimi_read:
        steps.append(("Kimi", _restore_kimi))
    for label, restore in steps:
        try:
            problems += restore(host, snapshot)
        except (LlmctlError, OSError) as exc:
            problems.append(f"{label}: {exc}")
    return problems


def round_trip(host, check: Callable[[], None], report: Callable[[list[str]], None] | None = None, *,
               dsh: bool = False, kimi: bool = False, target: Target | None = None) -> list[str]:
    """Snapshot the defaults, run check, and restore them even when check raises (spec L9).

    report, when given, gets the restore's problem lines on both paths, so a caller can record them although check
    raised."""
    snapshot = snapshot_defaults(host.env, dsh=dsh, kimi=kimi)
    try:
        check()
    except BaseException:
        problems = restore_defaults(host, snapshot, target=target)
        for line in problems:
            host.err.write(f"not restored: {line}\n")
        if report is not None:
            report(problems)
        raise
    problems = restore_defaults(host, snapshot, target=target)
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

    def __init__(self, out: Path, catalog, host: RealHost, env: dict[str, str], work_root: Path, *,
                 dsh_snapshot: tuple[str | None, bool] | None = None, kimi_snapshot: "KimiSnapshot | None" = None):
        self.out = out
        self.work_root = work_root
        self.catalog = catalog
        self.host = host
        self.env = env
        self.home = env.get("HOME", "")
        self.private = private_values(env, str(work_root))
        self.rows: list[_Row] = []
        self.count = 0
        self.dsh_snapshot = dsh_snapshot  # the home patch text (None when missing) and holds_default at the start
        self.dsh = dsh_snapshot is not None
        self.kimi_snapshot = kimi_snapshot  # Kimi's config.toml at the start (Kimi design D11)
        self.kimi = kimi_snapshot is not None

    def record(self, name: str, text: str) -> None:
        self.count += 1
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-")
        (self.out / "records" / f"{self.count:03d}-{slug}").write_text(self.redact(text), encoding="utf-8")

    def redact(self, text: str) -> str:
        return redact(text, self.home, str(self.work_root))

    def command(self, name: str, argv: Sequence[str], cwd: Path, timeout_s: int, *, with_stderr: bool = False,
                drop: Sequence[str] = (), extra_env: Mapping[str, str] | None = None
                ) -> tuple[int | None, str, float, bool]:
        """Run one command; (exit status or None on timeout, stdout, seconds, withheld), all of it recorded.

        with_stderr returns stdout followed by stderr instead; the record keeps the two apart.
        drop names the event types the recorded stdout leaves out; the returned stdout keeps them.
        withheld is True when the redacted output still named a private value, so the record holds neither stream.
        extra_env adds to the driver's environment for this command only.
        """
        start = time.monotonic()
        try:
            env = self.step_env(cwd, extra_env)
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

    def step_env(self, cwd: Path, extra_env: Mapping[str, str] | None = None) -> dict[str, str]:
        # OpenCode and dsh take their working directory from PWD, so PWD has to name the folder the command runs in
        return dict(self.env, **(extra_env or {}), PWD=str(Path(cwd).resolve()))

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

    def dsh_env(self, step: Step, folder: Path, target: Target) -> dict[str, str]:
        """The extra environment of a dsh step (dsh design D9): telemetry off and TMPDIR under the work root; a matrix
        step also gets a private DSH_HOME holding a byte copy of the real home patch and an empty DSH_AGENTS_HOME,
        and its headless session the overlay files its argv names."""
        tmp = self.work_root / f"{step.cwd}.tmp"
        tmp.mkdir()
        extra = {"DSH_TELEMETRY_DISABLED": "1", "TMPDIR": str(tmp)}
        if step.name.startswith(ROUND_TRIP + " "):
            return extra  # the round trip's session runs in the real dsh home
        home, agents = self.work_root / f"{step.cwd}.home", self.work_root / f"{step.cwd}.agents"
        home.mkdir(mode=0o700)
        os.chmod(home, 0o700)
        agents.mkdir()
        (home / "cordis.patch.yml").write_bytes(dsh_config.home_patch(self.env).read_bytes())
        extra.update(DSH_HOME=str(home), DSH_AGENTS_HOME=str(agents))
        if step.name.endswith(" dsh-headless"):
            (folder / "llmctl-default.patch.yml").write_text(dsh_config.default_overlay(target), encoding="utf-8")
            if target.recipe.chat_only:
                (folder / "llmctl-no-tools.patch.yml").write_text(dsh_config.no_tools_patch(), encoding="utf-8")
        return extra

    def kimi_env(self, step: Step, folder: Path, target: Target) -> dict[str, str]:
        """The extra environment of a Kimi session (Kimi design D11, O6): a private Kimi home holding a copy of the
        real config.toml with mode 0600, which a matrix step gets with llmctl's default edit when its default_model
        does not name the recipe, and the round trip as a pure byte copy."""
        extra = _kimi_extra(self.work_root, step.cwd)
        with open(kimi_config.config_path(self.env), encoding="utf-8", newline="") as handle:
            text = handle.read()
        copy = Path(extra["KIMI_CODE_HOME"]) / "config.toml"
        if not step.name.startswith(ROUND_TRIP + " "):
            if kimi_config.parse_state(copy, text).default != kimi_config.alias(target):
                text = kimi_config.config_text(copy, text, target, set_default=True)
        handle = os.open(copy, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
        return extra

    def run_step(self, step: Step, target: Target) -> tuple[list[str], float]:
        """Run one planned step; (its failure reasons, seconds)."""
        kind = step.name.split(" ", 1)[1]
        start = time.monotonic()
        if kind == "config":
            return self.config_check(target), time.monotonic() - start
        if kind == "probes":
            return self.probes(step, target), time.monotonic() - start
        if kind == "long-grammar":
            return self.long_grammar(step, target), time.monotonic() - start
        if kind == "dsh-config":
            return self.dsh_config_check(step, target), time.monotonic() - start
        if kind == "kimi-config":
            return self.kimi_config_check(step, target), time.monotonic() - start
        folder = self.work(step)
        argv = [part.replace(WORK, str(folder)) for part in step.argv]
        tools = not target.recipe.chat_only
        extra_env = None
        if kind.startswith("dsh-"):
            try:
                extra_env = self.dsh_env(step, folder, target)
            except OSError as exc:
                return [f"cannot prepare the dsh folders: {exc}"], time.monotonic() - start
        if kind == "dsh-web":
            return self.web(step, argv, folder, extra_env), time.monotonic() - start
        private, kept = None, None
        if kind == "kimi":
            try:
                extra_env = self.kimi_env(step, folder, target)
                private = Path(extra_env["KIMI_CODE_HOME"]) / "config.toml"
                kept = private.read_bytes()
            except (OSError, UnicodeDecodeError, LlmctlError) as exc:
                return [f"cannot prepare the Kimi folders: {exc}"], time.monotonic() - start
        since = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        code, stdout, seconds, withheld = self.command(step.name, argv, folder, step.timeout_s,
                                                       drop=DROPPED_EVENTS.get(kind, ()), extra_env=extra_env)
        reasons = [] if code == 0 or kind == "pi" or kind == "hermes-chat" else [f"exit status {code}"]
        if withheld:  # the other reasons may quote the output that the record withheld
            return reasons + [WITHHELD_REASON], seconds
        if kind == "status":
            reasons += status_reasons(stdout, opencode=target.recipe.opencode, dsh=self.dsh, kimi=self.kimi)
        elif kind == "kimi":
            reasons += list(kimi_outcome(stdout.splitlines(), tools=tools).reasons)
            if tools and not hello_valid(folder / "hello.py"):
                reasons.append("hello.py is missing or lacks def add and add(2, 3)")
            try:
                changed = private.read_bytes() != kept
            except OSError as exc:
                reasons.append(f"cannot read the private config.toml after the Kimi session: {exc}")
            else:
                if changed:
                    reasons.append("the private config.toml changed during the Kimi session")
            reasons += self.logs_since(since, target, record=f"{step.name} docker-logs", session="the Kimi session")
        elif kind == "dsh-headless":
            reasons += list(dsh_outcome(stdout.splitlines(), tools=tools).reasons)
            if tools and not hello_valid(folder / "hello.py"):
                reasons.append("hello.py is missing or lacks def add and add(2, 3)")
            reasons += self.logs_since(since, target, record=f"{step.name} docker-logs", session="the dsh session")
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
            outcome = opencode_outcome(lines, tools=tools)
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

    def guarded(self, *texts: str, extra: Sequence[str] = ()) -> tuple[list[str], bool]:
        """The texts redacted, or all of them WITHHELD when one still names a private value or one of extra."""
        redacted = [self.redact(text) for text in texts]
        values = (*self.private, *extra)
        if any(names_private(text, values) for text in redacted):
            return [WITHHELD] * len(texts), True
        return redacted, False

    def dsh_config_check(self, step: Step, target: Target) -> list[str]:
        """dsh design D9 step 1: the home patch is what llmctl writes from the text it held at the start, keeping
        the default it held then, and no profile patch sets the llm-pi-ai row."""
        old, held = self.dsh_snapshot
        path = dsh_config.home_patch(self.env)
        reasons = []
        try:
            wanted = dsh_config.home_text(path, old, target, default=held)
            with open(path, encoding="utf-8", newline="") as handle:
                found = handle.read()
        except (LlmctlError, OSError, UnicodeDecodeError) as exc:
            reasons.append(f"the dsh home patch: {exc}")
        else:
            if found != wanted:
                reasons.append(f"the dsh home patch is not what llmctl writes for {target.recipe.id} from the text "
                               "it held when the run started")
                (found_shown, wanted_shown), _ = self.guarded(found, wanted)
                self.record(step.name, json.dumps({"found": found_shown, "wanted": wanted_shown}, indent=2,
                                                  ensure_ascii=False) + "\n")
        try:
            findings = dsh_config.profile_findings(self.env)
        except LlmctlError as exc:
            return reasons + [f"the dsh profile patches: {exc}"]
        reasons += [f"{finding.path} line {finding.line} sets the llm-pi-ai row" for finding in findings
                    if finding.row_id == "llm-pi-ai"]
        return reasons

    def kimi_config_check(self, step: Step, target: Target) -> list[str]:
        """Kimi design D11 step 1: llmctl's tables in the real config.toml equal the generator's by value, its
        default_model is what D5 gives from the snapshot without --set-default, and the lines outside llmctl's tables
        are the snapshot's; a mismatch records only llmctl's tables and default_model, never the rest of the file."""
        snapshot = self.kimi_snapshot
        try:
            state = kimi_config.read_state(self.env)
            foreign = () if state is None else tuple(kimi_config.foreign_lines(state.path, state.text))
        except LlmctlError as exc:
            return [f"the Kimi config.toml: {exc}"]
        provider = None if state is None else state.provider
        aliases = () if state is None else state.aliases
        default = kimi_config.MISSING if state is None else state.default
        wanted_provider = kimi_config.provider_table(target)
        wanted_aliases = ((kimi_config.alias(target), kimi_config.model_table(target)),)
        held = snapshot.default
        wanted_default = (kimi_config.alias(target) if isinstance(held, str)
                          and held.startswith(kimi_config.ALIAS_PREFIX) else held)
        reasons = []
        if provider != wanted_provider:
            reasons.append("llmctl's provider table differs from the generator")
        if aliases != wanted_aliases:
            reasons.append("llmctl's model table differs from the generator")
        if not _same_default(default, wanted_default):
            reasons.append(f"default_model reads {_kimi_shown(default)}, expected {_kimi_shown(wanted_default)}")
        if foreign != snapshot.foreign:
            reasons.append("the lines outside llmctl's tables changed since the run started")
        if reasons:
            shown = {"provider": {"found": _withhold_key(provider), "wanted": wanted_provider},
                     "models": {"found": [[name, _withhold_key(table)] for name, table in aliases],
                                "wanted": [[name, table] for name, table in wanted_aliases]},
                     "default_model": {"found": _kimi_recorded(default), "wanted": _kimi_recorded(wanted_default)},
                     "failed": reasons}
            (text,), _ = self.guarded(json.dumps(shown, indent=2, ensure_ascii=False, default=str) + "\n")
            self.record(step.name, text)
        return reasons

    def web(self, step: Step, argv: list[str], folder: Path, extra_env: Mapping[str, str]) -> list[str]:
        """dsh design D9 step 4: boot dsh web, record its output without the launch token, and check it."""
        start = time.monotonic()
        boot = web_boot(argv, cwd=folder, env=self.step_env(folder, extra_env))
        seconds = time.monotonic() - start
        stdout, tokens = [], []
        for line in boot.stdout.splitlines(keepends=True):
            if line.startswith(WEB_LINE):
                shown, token = web_line(line.rstrip("\n"))
                line = shown + "\n" if line.endswith("\n") else shown
                tokens += [token] if token else []
            stdout.append(line)
        if boot.line is not None:
            token = web_line(boot.line)[1]
            tokens += [token] if token else []
        (recorded, recorded_err), withheld = self.guarded("".join(stdout), boot.stderr, extra=tokens)
        self.record(step.name, json.dumps({"command": shlex.join(argv), "cwd": str(folder), "exit": boot.code,
                                           "seconds": round(seconds, 1), "stdout": recorded,
                                           "stderr": recorded_err}, indent=2, ensure_ascii=False) + "\n")
        if withheld:  # web_outcome's reasons quote stderr, which the record withheld
            return list(web_outcome(replace(boot, stderr="")).reasons) + [WITHHELD_REASON]
        return list(web_outcome(boot).reasons)

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
            if name in ("B", "F") and status != 200:
                reasons.append(f"probe {name}: status {status}")
        return reasons

    def free_mib(self) -> tuple[int | None, str | None]:
        """nvidia-smi's free MiB, or None with the error text; a failure fails nothing."""
        try:
            return gpu.gpu_info(self.host).free_mib, None
        except (OSError, LlmctlError) as exc:
            return None, str(exc)

    def grammar_request(self, step: Step, name: str, method: str, url: str, body: dict | None,
                        timeout_s: float) -> tuple[int, dict]:
        """One request of the long-grammar step, recorded without its prompt text; (status, parsed reply)."""
        before, before_error = self.free_mib()
        start = time.monotonic()
        error = None
        try:
            response = self.host.http(method, url, headers={"Content-Type": "application/json"} if body else None,
                                      body=json.dumps(body).encode("utf-8") if body else None, timeout=timeout_s)
            status, reply = response.status, response.body
        except (OSError, LlmctlError) as exc:
            status, reply, error = 0, b"", str(exc)
        seconds = time.monotonic() - start
        after, after_error = self.free_mib()
        try:
            parsed = _dict(json.loads(reply))
        except ValueError:
            parsed = {}
        tokens = _dict(parsed.get("usage")).get("prompt_tokens")
        record = {"status": status, "prompt_tokens": tokens if type(tokens) is int else None,
                  "seconds": round(seconds, 1), "free_mib_before": before, "free_mib_after": after}
        for key, text in (("free_mib_before_error", before_error), ("free_mib_after_error", after_error),
                          ("error", error)):
            if text is not None:
                record[key] = text
        self.record(f"{step.name} {name}", json.dumps(record, indent=2, ensure_ascii=False) + "\n")
        return status, parsed

    def long_grammar(self, step: Step, target: Target) -> list[str]:
        """SGLang design D15: a calibration, a long prompt near the context, then the grammar requests (the title and,
        for a recipe with tools, the forced tool call), then the readiness check; a forced call answered as text
        passes, because gpt-oss does that."""
        recipe, served = target.recipe, target.served
        base = f"http://127.0.0.1:{target.runtime.host_port}"
        url = f"{base}/v1/chat/completions"
        reasons = []

        def chat(name: str, body: dict) -> tuple[int, dict]:
            status, parsed = self.grammar_request(step, name, "POST", url, body, step.timeout_s)
            if status != 200:
                reasons.append(f"long-grammar {name}: status {status}")
            return status, parsed

        def user(content: str) -> list[dict]:
            return [{"role": "user", "content": content}]

        status, parsed = chat("calibration", {"model": served, "messages": user(long_prompt(secrets.token_hex(16), 20)),
                                              "max_tokens": 1})
        tokens = _dict(parsed.get("usage")).get("prompt_tokens")
        if type(tokens) is not int or tokens <= 0:
            if status == 200:
                reasons.append("long-grammar calibration: no usage.prompt_tokens")
        else:
            copies = long_grammar_copies(recipe.context, tokens)
            chat("long", {"model": served, "messages": user(long_prompt(secrets.token_hex(16), copies)),
                          "max_tokens": 32})
            status, parsed = chat("title", probe_bodies(target)["A"])
            if status == 200 and not probe_record("A", status, json.dumps(parsed).encode("utf-8"))["title_ok"]:
                reasons.append("long-grammar title: title_ok False")
            if not recipe.chat_only:
                chat("forced-tool", {"model": served, "messages": user(_FORCED_TOOL_PROMPT), "tools": [WEATHER_TOOL],
                                     "tool_choice": "required", "max_tokens": 256})
        status, _ = self.grammar_request(step, "ready", "GET", f"{base}{target.runtime.ready_path}", None, 2.0)
        if status != 200:
            reasons.append(SERVER_STOPPED)
        return reasons

    def logs_since(self, since: str, target: Target, *, record: str = f"{ROUND_TRIP} docker-logs",
                   session: str = "the OpenCode run") -> list[str]:
        # docker logs replays the container's stderr on stderr, and llama.cpp writes its log there
        code, text, _, withheld = self.command(record,
                                               ["docker", "logs", "--since", since, target.runtime.container_name],
                                               REPO, STEP_TIMEOUT_S, with_stderr=True)
        if code != 0:
            return [f"docker logs exited {code}"]
        if withheld:
            return [f"docker logs: {WITHHELD_REASON}"]
        if "/v1/chat/completions" in text or "print_timing" in text:
            return []
        return [f"the {target.label} log shows no request since {session} started"]


def _steps_by_prefix(steps: list[Step]) -> dict[str, list[Step]]:
    groups: dict[str, list[Step]] = {}
    for step in steps:
        groups.setdefault(step.name.split(" ", 1)[0], []).append(step)
    return groups


def _summary(rows: list[_Row]) -> str:
    lines = list(SUMMARY_HEAD)
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
    parser.add_argument("--skip-dsh", action="store_true",
                        help="leave dsh out: no dsh steps, and LLMCTL_CLIENTS without dsh for every command")
    parser.add_argument("--skip-kimi", action="store_true",
                        help="leave Kimi out: no Kimi steps, and LLMCTL_CLIENTS without kimi for every command")
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
            if kind == "long-grammar" and SERVER_STOPPED in reasons:
                break  # the server stopped after the grammar requests, so the client steps would only time out
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
        round_trip(run.host, check, report=restored.append, dsh=run.dsh, kimi=run.kimi, target=target)
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


def dsh_refusal(host, env: Mapping[str, str]) -> str | None:
    """Why the dsh steps cannot run here, else None (dsh design D9); it opens no .env or credential file."""
    state = next(state for state in clients.client_states(host) if state.name == "dsh")
    if not state.active:
        return f"dsh is not installed here ({state.reason}); install it, or run with --skip-dsh"
    rows = [f"{finding.path} line {finding.line}" for finding in dsh_config.profile_findings(env)
            if finding.row_id == "llm-pi-ai"]
    if rows:
        return (f"a dsh profile patch sets the llm-pi-ai row ({'; '.join(rows)}); move it as "
                "docs/clients.md#moving-a-hand-made-setup says, or run with --skip-dsh")
    if env.get("DEEPSEEK_API_KEY"):
        return ("DEEPSEEK_API_KEY is set, so dsh's web_search could send model-written queries to DeepSeek; unset it "
                "for the e2e, or run with --skip-dsh")
    if host.port_open(DSH_PORT):
        return (f"127.0.0.1:{DSH_PORT} is in use, and the dsh web step needs it; stop what listens there, or run with "
                "--skip-dsh")
    return None


def _home_snapshot(env: Mapping[str, str]) -> tuple[str | None, bool]:
    """The dsh home patch's text as it is (None when missing) and whether llmctl's block holds dsh's default."""
    path = dsh_config.home_patch(env)
    try:
        with open(path, encoding="utf-8", newline="") as handle:
            text = handle.read()
    except FileNotFoundError:
        text = None
    except (OSError, UnicodeDecodeError) as exc:
        raise UsageError(f"cannot read {path}: {exc}") from exc
    return text, dsh_config.holds_default(env)


@dataclass(frozen=True)
class KimiSnapshot:
    text: str | None  # Kimi's config.toml at the start, None when it is missing
    foreign: tuple[str, ...]  # its lines outside llmctl's tables
    default: object  # its default_model, kimi_config.MISSING when it has none


def _kimi_snapshot(env: Mapping[str, str]) -> KimiSnapshot:
    """Kimi's config.toml at the start (Kimi design D11); an unreadable or invalid file raises ClientError."""
    state = kimi_config.read_state(env)
    if state is None:
        return KimiSnapshot(None, (), kimi_config.MISSING)
    return KimiSnapshot(state.text, tuple(kimi_config.foreign_lines(state.path, state.text)), state.default)


def kimi_refusal(host, env: Mapping[str, str]) -> str | None:
    """Why the Kimi steps cannot run here, else None (Kimi design D11 refusals 1 to 3); it runs no command and
    opens no credential file."""
    state = next(state for state in clients.client_states(host) if state.name == "kimi")
    if not state.active:
        return f"Kimi is not installed here ({state.reason}); install it, or run with --skip-kimi"
    names = sorted(name for name in env if name.startswith("KIMI_MODEL_"))
    if names:
        return (f"KIMI_MODEL_ variables are set in this shell ({', '.join(names)}), and they change which model Kimi "
                "uses and what it sends; unset them for the e2e, or run with --skip-kimi")
    if host.which("rg") is None:
        return ("no rg on PATH, so Kimi's Grep tool would download ripgrep from Kimi's CDN into the private home; "
                "install ripgrep (sudo apt install ripgrep), or run with --skip-kimi")
    if host.which("fd") is None and host.which("fdfind") is None:
        return ("neither fd nor fdfind is on PATH, so Kimi's Glob tool would download fd from Kimi's CDN into the "
                "private home; install fd-find (sudo apt install fd-find), or run with --skip-kimi")
    return None


def kimi_default_refusal(env: Mapping[str, str], snapshot: KimiSnapshot) -> str | None:
    """Why llmctl's default edit, which every matrix Kimi session applies to its private copy, cannot change the
    file's default_model, else None (Kimi design D11)."""
    if snapshot.text is None:
        return None
    path = kimi_config.config_path(env)
    try:
        kimi_config.default_text(path, snapshot.text, "llmctl/local")
    except ClientError:
        return (f"Kimi's default_model in {path} is written in a form that llmctl's default edit cannot change, and "
                "every Kimi session of the e2e needs that edit; fix default_model, or run with --skip-kimi")
    return None


def _kimi_extra(work_root: Path, name: str) -> dict[str, str]:
    """The private folders of one Kimi command (Kimi design D11, O2): its Kimi home (mode 0700), TMPDIR and an
    empty HOME under the work root, the run's one Kimi cache (mode 0700), and the three switches."""
    home, tmp, user_home = (work_root / f"{name}{suffix}" for suffix in (".kimi", ".tmp", ".home"))
    home.mkdir(mode=0o700)
    os.chmod(home, 0o700)
    tmp.mkdir()
    user_home.mkdir()
    cache = work_root / "kimi-cache"
    if not cache.is_dir():
        cache.mkdir(mode=0o700)
        os.chmod(cache, 0o700)
    return {"KIMI_CODE_HOME": str(home), "KIMI_CODE_CACHE_DIR": str(cache), "TMPDIR": str(tmp),
            "HOME": str(user_home), **dict(kimi_config.SWITCHES)}


def main(argv: list[str] | None = None, *, make_host: Callable[[dict[str, str]], RealHost] = RealHost) -> int:
    args = build_parser().parse_args(argv)
    dsh_snapshot = kimi_snapshot = None
    try:
        if "LLMCTL_CLIENTS" in os.environ:
            raise UsageError("unset LLMCTL_CLIENTS: the e2e checks all five clients, or fewer with --skip-dsh and "
                             "--skip-kimi")
        env = driver_env(os.environ, skip_dsh=args.skip_dsh, skip_kimi=args.skip_kimi)
        os.environ.pop("LLMCTL_SET_DEFAULT", None)  # RealHost.run passes os.environ to hermes profile use
        host = make_host(env)
        catalog = load_catalog(REPO / "recipes", override_dir(env))
        matrix = MATRICES[args.matrix]
        steps = plan_steps(catalog, matrix, round_trip=args.set_default_round_trip, keep_running=args.keep_running,
                           dsh=not args.skip_dsh, kimi=not args.skip_kimi)
        running = running_servers(host)
        if running is not None:  # the first up --replace would stop the owner's server
            raise UsageError(f"an llmctl model server is running ({running}); stop it with ./llmctl down, "
                             "then run the driver again")
        if not args.skip_dsh:
            refusal = dsh_refusal(host, env)
            if refusal is not None:
                raise UsageError(refusal)
            dsh_snapshot = _home_snapshot(env)  # damaged markers or an unreadable file refuse here, too
        if not args.skip_kimi:
            refusal = kimi_refusal(host, env)
            if refusal is not None:
                raise UsageError(refusal)
            kimi_snapshot = _kimi_snapshot(env)  # an unreadable or invalid config.toml refuses here (D11 item 4)
        if args.set_default_round_trip:
            # refuse before the first server starts when OpenCode has no model to restore, or Kimi's default_model
            # is not a string (D11 item 5)
            snapshot_defaults(env, kimi=not args.skip_kimi)
            refusal = round_trip_refusal(catalog, env)
            if refusal is not None:
                raise UsageError(refusal)
        if kimi_snapshot is not None:
            refusal = kimi_default_refusal(env, kimi_snapshot)
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
    run = _Run(out, catalog, host, env, work_root, dsh_snapshot=dsh_snapshot, kimi_snapshot=kimi_snapshot)
    versions = [["hermes", "--version"], ["pi", "--version"], ["opencode", "--version"]]
    versions += [["dsh", "--version"]] if run.dsh else []
    versions += [["kimi", "--version"]] if run.kimi else []
    for argv_ in versions + [["nvidia-smi", "--query-gpu=power.limit", "--format=csv,noheader"]]:
        extra_env, cwd = None, REPO
        if argv_[0] == "dsh":  # dsh design D9: every dsh command of the e2e runs in a fresh work folder, with
            cwd = work_root / "dsh-version"  # telemetry off and TMPDIR under the work root
            tmp = work_root / "dsh-version.tmp"
            cwd.mkdir()
            tmp.mkdir()
            extra_env = {"DSH_TELEMETRY_DISABLED": "1", "TMPDIR": str(tmp)}
        if argv_[0] == "kimi":  # Kimi design D11: in a fresh work folder with its own private Kimi home
            cwd = work_root / "kimi-version"
            cwd.mkdir()
            extra_env = _kimi_extra(work_root, "kimi-version")
        if run.command(argv_[0] + " version", argv_, cwd, 60, extra_env=extra_env)[3]:
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
