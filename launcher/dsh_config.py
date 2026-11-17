"""DeepSeek Harness (dsh): the block llmctl owns in dsh's home patch <dsh home>/cordis.patch.yml.

Every row in the block is one JSON flow line, `- {...}`, which dsh's js-yaml reads as intended, so llmctl needs no
YAML library; see docs/superpowers/specs/2026-11-16-llmctl-dsh-design.md, sections D2 (the file and rows llmctl
owns) and D3 (rows from a recipe, per thinking type).
"""

import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from launcher import files
from launcher.errors import ClientError, UsageError
from launcher.files import FileChange, home_dir
from launcher.host import Host
from launcher.model import ClientMeta, Target

PROVIDER = "llmctl"
VERIFIED_VERSION = "0.2.2-rc.3"
OWNED_ROWS = ("llm-pi-ai", "agent-default-model", "acp")
STREAM_IDLE_TIMEOUT_MS = 900000  # an estimate (design D3, Open point 3)
RETRYABLE_CODES = ("EMPTY_RESPONSE", "RATE_LIMIT", "SERVER", "TRANSPORT")  # dsh's default list without TIMEOUT
COMPACTION_HEADROOM = 65536
NO_TOOLS_ROWS = ("tool-bash", "tool-goal", "tool-fs", "tool-fs-search", "tool-jobs", "tool-subagent-control",
                 "tool-skill", "tool-subagent", "tool-subagent-fork", "tool-todo", "tool-web", "tool-workflow",
                 "tool-working-directory", "tool-agent-team")
OUT = "leave dsh out with LLMCTL_CLIENTS"
HEADER = ("# dsh rows for the model llmctl runs (docs/llmctl.md#clients).\n"
          "# llmctl rewrites the lines between the two markers on every ./llmctl up and keeps everything else in this file.\n"
          "# While the block holds the agent-default-model line, the running model is dsh's default in every profile;\n"
          "# delete that line and the acp line to give the default back to dsh's model picker.\n")
BEGIN = "# >>> llmctl managed block: rewritten by llmctl on every start; edits here are lost >>>"
END = "# <<< llmctl managed block <<<"
# A line that targets one of llmctl's rows (design D6): block style, JSON flow, Web UI multi-line flow, insert rows.
TARGET = re.compile(r"""(?:^|[\s{,])["']?id["']?[ \t]*:[ \t]*["']?"""
                    r"""(llm-pi-ai|agent-default-model|acp)["']?[ \t]*(?:[,}#]|$)""")
_EMPTY_LIST = re.compile(r"\[[ \t]*\][ \t]*(?:#.*)?")
_JS_YAML_BREAK = re.compile(r"\r\n|\r|\n")
_BOM = "\ufeff"


def dsh_home(env: Mapping[str, str]) -> Path:
    """dsh's home folder, as resolveDshHome and expandHomePath in dsh-home-paths/src/index.ts:87-110 (design D1).

    `$NAME` is never expanded, because dsh does not expand it; a relative value is refused, because dsh resolves it
    against each command's working folder.
    """
    value = env.get("DSH_HOME", "")
    if not value.strip():
        home = str(home_dir(env) / ".dsh")
    elif value == "~":
        home = str(home_dir(env))
    elif value.startswith(("~/", "~\\")):
        home = str(home_dir(env)) + "/" + value[2:]
    elif value.startswith("/"):
        home = value
    else:
        raise UsageError(f"DSH_HOME is relative ({value}), and dsh resolves it against each command's working "
                         "folder; set it to an absolute path, ~ or ~/...")
    home = os.path.normpath(home)
    if home.startswith("//"):
        home = home[1:]
    return Path(home)


def home_patch(env: Mapping[str, str]) -> Path:
    return dsh_home(env) / "cordis.patch.yml"


def profiles_dir(env: Mapping[str, str]) -> Path:
    return dsh_home(env) / "profiles"


def _thinking_fields(target: Target) -> tuple[dict, dict]:
    """(model fields, route fields) for the recipe's thinking type, keys in design D3 order."""
    recipe = target.recipe
    meta = recipe.clients
    if meta.thinking == "toggle":
        efforts = {"off": None, "medium": "medium"}
        kwargs = {"enable_thinking": {"$var": "thinking.enabled"}}
        route = {"reasoning": "medium" if meta.default == "on" else "off"}
        if meta.medium_kwargs:
            for key, value in meta.medium_kwargs:
                if value is not True:
                    raise ClientError(f"dsh cannot send {key} = {json.dumps(value)} of {recipe.id} per level, because "
                                      f"llmctl maps a middle level only to true; {OUT} for this recipe")
                kwargs[key] = {"$var": "thinking.budget"}
            efforts["high"] = "high"
            route["thinkingBudgets"] = {"medium": 1, "high": 0}
        return ({"reasoningEfforts": efforts,
                 "compat": {"thinkingFormat": "chat-template", "chatTemplateKwargs": kwargs}}, route)
    if meta.thinking == "effort":
        efforts = {("off" if effort == "none" else effort): effort for effort in meta.efforts}
        return ({"reasoningEfforts": efforts, "compat": {"thinkingFormat": "openai", "supportsReasoningEffort": True}},
                {"reasoning": "medium"})
    if meta.thinking == "always":
        return ({"reasoningEfforts": {"high": "high"}, "compat": {"supportsReasoningEffort": False}},
                {"reasoning": "high"})
    if meta.thinking == "none":
        return {"reasoningEfforts": False}, {}
    raise ValueError(f"unknown thinking type {meta.thinking!r}")


def route(target: Target) -> dict:
    """The provider `llmctl` of the llm-pi-ai row (design D3)."""
    recipe = target.recipe
    model_fields, route_fields = _thinking_fields(target)
    return {
        "displayName": PROVIDER,
        "api": "openai-completions",
        "baseURL": f"http://127.0.0.1:{target.runtime.host_port}/v1",
        "headers": {"Authorization": "Bearer no-key-required"},
        "compat": {"supportsStore": False, "supportsDeveloperRole": False, "maxTokensField": "max_tokens"},
        **route_fields,
        "streamIdleTimeoutMs": STREAM_IDLE_TIMEOUT_MS,
        "retryPolicy": {"mode": "normal", "maxRetries": 5, "retryableCodes": list(RETRYABLE_CODES)},
        "models": [{"id": target.served, "name": f"{recipe.id} on {target.label} (llmctl)",
                    "contextWindow": recipe.context, "maxTokens": recipe.clients.output, "input": ["text"],
                    **model_fields}],
    }


def rows(target: Target, *, default: bool) -> list[dict]:
    """The block's rows: llm-pi-ai, and with default also agent-default-model and acp (design D2)."""
    found = [{"id": "llm-pi-ai", "config": {"providers": {PROVIDER: route(target)}}}]
    if default:
        found.append({"id": "agent-default-model", "config": {"provider": PROVIDER, "model": target.served}})
        found.append({"id": "acp", "config": {"provider": PROVIDER, "model": target.served}})
    return found


def row_line(row: dict) -> str:
    """One row as a YAML sequence item holding one JSON flow mapping, ending in a newline."""
    return "- " + json.dumps(row, ensure_ascii=True, separators=(", ", ": ")) + "\n"


def render_block(target: Target, *, default: bool) -> str:
    """The lines between the markers."""
    return "".join(row_line(row) for row in rows(target, default=default))


def new_home_text(target: Target, *, default: bool) -> str:
    """The text of a new home patch, or of the header and block appended to one without markers."""
    return HEADER + BEGIN + "\n" + render_block(target, default=default) + END + "\n"


def default_overlay(target: Target) -> str:
    """The agent-default-model line alone, for a --patch overlay (headless warns about an acp row, design A4)."""
    return row_line(rows(target, default=True)[1])


def no_tools_patch() -> str:
    """The rows that disable every dsh tool, for a single prompt to a chat-only recipe (design D3, B4-2)."""
    return "".join(row_line({"id": row, "disabled": True}) for row in NO_TOOLS_ROWS)


@dataclass(frozen=True)
class Block:
    """The rows of llmctl's block; description and served come from the llm-pi-ai row's first model."""

    rows: tuple[dict, ...]
    holds_default: bool
    description: str | None
    served: str | None


@dataclass(frozen=True)
class Finding:
    """A profile patch line that targets one of llmctl's rows; profile is the profile folder's name."""

    path: Path
    line: int
    row_id: str
    profile: str


def _read_text(path: Path) -> str | None:
    """The file's text with its line endings as they are; None when it does not exist."""
    try:
        with open(path, encoding="utf-8", newline="") as handle:
            return handle.read()
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError) as exc:
        raise ClientError(f"cannot read {path}: {exc}") from exc


def _row(rows: tuple[dict, ...] | list[dict], row_id: str) -> dict | None:
    return next((row for row in rows if row["id"] == row_id), None)


def _first_model(rows: tuple[dict, ...]) -> tuple[str | None, str | None]:
    """(description, served) from the llm-pi-ai row's first model; (None, None) when it is not readable."""
    row = _row(rows, "llm-pi-ai")
    try:
        model = row["config"]["providers"][PROVIDER]["models"][0]
        name, served = model["name"], model["id"]
    except (KeyError, IndexError, TypeError):
        return None, None
    if not isinstance(name, str) or not isinstance(served, str):
        return None, None
    return name.removesuffix(" (llmctl)"), served


def _parse_lines(path: Path, lines: list[str], found: tuple[int, int]) -> Block:
    """The block between the markers (design D2 "Reading the block", D6 item 5)."""
    parsed: list[dict] = []
    last = -1
    for index in range(found[0] + 1, found[1]):
        line = files.bare(lines[index])
        if "\r" not in line and (not line.strip() or line.lstrip(" \t").startswith("#")):
            continue  # a lone CR is a line break to js-yaml, so even a comment that holds one is refused below
        row = None
        if line.startswith("- ") and "\r" not in line:
            try:
                row = files.parse_json(line[2:])
                json.dumps(row, ensure_ascii=False).encode("utf-8")  # a lone surrogate (clients ledger, review 6-9)
            except ValueError:  # UnicodeEncodeError is a ValueError
                row = None
        order = OWNED_ROWS.index(row["id"]) if isinstance(row, dict) and row.get("id") in OWNED_ROWS else -1
        if order <= last:  # not an owned row, a repeat, or out of the table's order
            raise ClientError(f"{path} line {index + 1}: the llmctl block holds a line llmctl did not write; delete "
                              "it, or move your row after the end marker")
        last = order
        parsed.append(row)
    found_rows = tuple(parsed)
    description, served = _first_model(found_rows)
    return Block(found_rows, _row(found_rows, "agent-default-model") is not None, description, served)


def parse_block(path: Path, text: str) -> Block | None:
    """llmctl's block in the text; None when it has no markers."""
    lines = files.split_lines(text)
    found = files.find_markers(path, lines, begin=BEGIN, end=END)
    if found is None:
        return None
    return _parse_lines(path, lines, found)


def read_block(env: Mapping[str, str]) -> Block | None:
    """llmctl's block in the home patch; None when the file is missing or has no markers."""
    path = home_patch(env)
    text = _read_text(path)
    return None if text is None else parse_block(path, text)


def holds_default(env: Mapping[str, str]) -> bool:
    block = read_block(env)
    return block is not None and block.holds_default


def profile_findings(env: Mapping[str, str]) -> list[Finding]:
    """Every line of a profile patch that targets an owned row, by profile, then line (design D6 item 8, L3)."""
    found = []
    for path in sorted(profiles_dir(env).glob("*/cordis.patch.yml")):
        text = _read_text(path)
        if text is None:  # removed since the listing
            continue
        if text.startswith(_BOM):
            text = text[1:]
        for index, line in enumerate(_JS_YAML_BREAK.split(text)):
            if line.lstrip(" \t").startswith("#"):
                continue
            for match in TARGET.finditer(line):
                found.append(Finding(path, index + 1, match.group(1), path.parent.name))
    return found


def home_text(path: Path, old: str | None, target: Target, *, default: bool) -> str:
    """The home patch with llmctl's block: a new file, the header and block appended, or the block replaced."""
    block = render_block(target, default=default)
    if old is None or old == "":
        return new_home_text(target, default=default)
    lines = files.split_lines(old)
    found = files.find_markers(path, lines, begin=BEGIN, end=END)
    if found is None:  # first adoption: append, after a newline when missing (S4b)
        return old + ("" if old.endswith("\n") else "\n") + HEADER + BEGIN + "\n" + block + END + "\n"
    begin, end = found
    return "".join(lines[:begin + 1]) + block + "".join(lines[end:])


def _scan_outside(path: Path, lines: list[str], found: tuple[int, int] | None, *, default: bool) -> None:
    """D6 items 6 and 7, with js-yaml's line breaks and byte-order mark (L3)."""
    for index, line in enumerate(lines):
        if found is not None and found[0] <= index <= found[1]:
            continue
        text = files.bare(line)
        if index == 0 and text.startswith(_BOM):
            text = text[1:]  # js-yaml skips a byte-order mark; the bytes stay in the file
        where = f"{path} line {index + 1}"
        if "\r" in text:  # a lone CR is a line break to js-yaml
            raise ClientError(f"{where}: llmctl cannot check this top-level line; write each row as a '- ' list item")
        stripped = text.lstrip(" \t")
        if not stripped or stripped.startswith("#"):
            continue
        if text[0] not in " \t":  # a line at column 0
            if _EMPTY_LIST.fullmatch(text):
                raise ClientError(f"{where}: dsh reads this file as one YAML list, so llmctl cannot add its rows "
                                  "after '[]'; delete that line")
            if text != "-" and not text.startswith("- "):
                raise ClientError(f"{where}: llmctl cannot check this top-level line; write each row as a '- ' list "
                                  "item")
        match = TARGET.search(text)
        if match and match.group(1) == "llm-pi-ai":
            raise ClientError(f"{where}: the llm-pi-ai row is set outside the llmctl block, and dsh applies only one "
                              "of the two; delete it (docs/clients.md#the-home-patch)")
        if match and default:
            raise ClientError(f"{where}: the {match.group(1)} row is set outside the llmctl block while llmctl holds "
                              "or sets dsh's default; delete that row, or delete llmctl's agent-default-model and "
                              "acp lines and run up without --set-default")


def default_name(row: dict | None) -> str:
    """<provider>/<model> of an agent-default-model row, with ? for a value that is not a string."""
    config = row.get("config") if row is not None else None
    values = [config.get(key) if isinstance(config, dict) else None for key in ("provider", "model")]
    return "/".join(value if isinstance(value, str) else "?" for value in values)


def _summary(old: str | None, block: Block | None, target: Target, new_rows: list[dict], *, default: bool) -> str:
    """The line after `would write <path>: ` and `wrote <path>: ` (design D7)."""
    recipe = f"{target.recipe.id} on {target.label}"
    held = f"dsh's default llmctl/{target.served} in every profile"
    if not old:
        part = f"{held} (was not held by llmctl)" if default else "dsh's default stays with its model picker"
        return f"created it with the llmctl rows for {recipe}; {part}"
    if block is None:
        part = f"{held} (was not held by llmctl)" if default else "dsh's default stays with its model picker"
        return f"added the llmctl rows for {recipe} after your rows; {part}"
    head = f"llmctl rows for {recipe}"
    if block.description is not None and block.description != recipe:
        head += f" (was {block.description})"
    elif block.description is not None and _row(block.rows, "llm-pi-ai") != _row(new_rows, "llm-pi-ai"):
        head += " (same recipe, rows rewritten)"
    elif block.description is not None:
        head += " (same recipe)"
    old_default, old_acp = _row(block.rows, "agent-default-model"), _row(block.rows, "acp")
    new_default, new_acp = _row(new_rows, "agent-default-model"), _row(new_rows, "acp")
    old_name = default_name(old_default)
    parts = []
    if block.holds_default and default and old_name != f"llmctl/{target.served}":
        parts.append(f"{held} (was {old_name})")
    if default and not block.holds_default:
        parts.append(f"{held} (was not held by llmctl)")
    if default and block.holds_default and (old_acp is None or (old_acp != new_acp and old_default == new_default)):
        parts.append("restored the acp row")
    if old_acp is not None and not block.holds_default and not default:
        parts.append("removed the acp row, because the block no longer holds agent-default-model")
    if block.holds_default and not default:
        parts.append(f"dsh's default goes back to its model picker (was {old_name})")
    return "; ".join([head, *parts])


def _plan(env: Mapping[str, str], target: Target, *, set_default: bool, release: bool) -> FileChange | None:
    """The one plan of plan_home and plan_release_default (design D5, D6); every refusal is a ClientError."""
    route(target)  # D6 item 9
    path = home_patch(env)
    old = _read_text(path)  # D6 item 3
    lines = files.split_lines(old) if old is not None else []
    found = files.find_markers(path, lines, begin=BEGIN, end=END)  # D6 item 4
    block = _parse_lines(path, lines, found) if found is not None else None  # D6 item 5
    if release and block is None:
        return None  # no llmctl block to release; the restore never adds rows
    default = False if release else set_default or (block is not None and block.holds_default)
    _scan_outside(path, lines, found, default=default)  # D6 items 6 and 7
    shadowed = [finding for finding in profile_findings(env) if finding.row_id == "llm-pi-ai"]
    if shadowed:  # D6 item 8
        where = "; ".join(f"{finding.path} line {finding.line}" for finding in shadowed)
        raise ClientError(f"dsh's profile patches set the llm-pi-ai row ({where}), which llmctl's row in {path} would "
                          f"hide without a message; delete those rows (docs/clients.md#moving-a-hand-made-setup), or "
                          f"{OUT}")
    new = home_text(path, old, target, default=default)
    new_rows = rows(target, default=default)
    if not old:
        prefix, suffix = "", ""
    elif found is None:
        prefix, suffix = old + ("" if old.endswith("\n") else "\n"), ""
    else:
        prefix, suffix = "".join(lines[:found[0]]), "".join(lines[found[1] + 1:])
    try:
        checked = parse_block(path, new)
    except ClientError:
        checked = None
    if (not new.startswith(prefix) or not new.endswith(suffix) or checked is None
            or list(checked.rows) != new_rows):  # D5 self-check, D6 item 10
        raise ClientError(f"{path}: llmctl's own check of the new home patch failed; nothing was written")
    if new == old:
        return None
    if not old:
        backup = False
    elif found is None:
        backup = True
    else:
        outside = "".join(lines[:found[0] + 1]) + "".join(lines[found[1]:])
        backup = outside != HEADER + BEGIN + "\n" + END + "\n"
    summary = _summary(old, block, target, new_rows, default=default)
    return FileChange(path, old, new, summary, f"would write {path}: {summary}", mode=0o600, backup=backup)


def plan_home(env: Mapping[str, str], target: Target, *, set_default: bool) -> FileChange | None:
    """The home patch change for up; None when the file already holds the rows."""
    return _plan(env, target, set_default=set_default, release=False)


def plan_release_default(env: Mapping[str, str], target: Target) -> FileChange | None:
    """The e2e restore: the same plan with the default lines forced out; None without an llmctl block."""
    return _plan(env, target, set_default=False, release=True)


def installed_version(host: Host) -> str:
    """The first non-empty line of `dsh --version` (S6)."""
    result = host.run(["dsh", "--version"])
    if result.returncode != 0:
        detail = next((line.strip() for line in result.stderr.splitlines() if line.strip()),
                      f"exit {result.returncode}")
        raise ClientError(f"dsh --version failed: {detail}; check that dsh runs in this shell, or {OUT}")
    version = next((line.strip() for line in result.stdout.splitlines() if line.strip()), None)
    if version is None:
        raise ClientError(f"dsh --version failed: it printed no version; check that dsh runs in this shell, or {OUT}")
    return version


def version_warning(host: Host) -> str | None:
    """The D7 warning when the installed dsh is not the version llmctl's rows were checked with."""
    version = installed_version(host)
    if version == VERIFIED_VERSION:
        return None
    return (f"warning: dsh {version} is not {VERIFIED_VERSION}, the version llmctl's dsh rows were checked with; "
            "run the dsh steps of the e2e before relying on them (docs/clients.md#maintenance)")


def _applied(dump: str, patch: str) -> set[str]:
    """The row ids that the dump lists under a group header naming `patch` as one of its patches (S1, S2)."""
    item = re.compile(rf"(?:^|, ){re.escape(patch)}(?:, |$)")
    applied = set()
    patched = False
    for line in dump.split("\n"):
        if line.startswith("# == "):
            _, sep, after = line.partition(", patched by ")
            patched = bool(sep) and item.search(after) is not None
            continue
        match = re.fullmatch(r"- id: (\S+)", line)
        if match and patched:
            applied.add(match.group(1))
    return applied


def check_home(host: Host, text: str, target: Target, *, path: Path, written: tuple[str, ...]) -> None:
    """Let the installed dsh read the candidate home patch in a private home (design D5); refuse what it rejects.

    `path` is the real home patch, named in the messages; `written` the ids of the rows the candidate's block holds.
    """
    try:
        tmp = tempfile.mkdtemp(prefix="llmctl-dsh-")
        try:
            os.chmod(tmp, 0o700)
            home = os.path.join(tmp, "home")
            os.mkdir(home, 0o700)
            patch = os.path.join(home, "cordis.patch.yml")
            fd = os.open(patch, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                handle.write(text)
            result = host.run(["dsh", "--profile", "acp", "--dump-config"],
                              env_extra={"DSH_HOME": home, "DSH_TELEMETRY_DISABLED": "1"})
        finally:
            shutil.rmtree(tmp)
    except (OSError, UnicodeError) as exc:
        raise ClientError(f"cannot check {path} with dsh: {exc}") from exc

    def real(line: str) -> str:
        return line.replace(patch, str(path)).replace(home, str(path.parent))

    if result.returncode != 0:
        for line in result.stderr.splitlines():
            if line.startswith("Error: "):
                raise ClientError(f"dsh rejects the home patch llmctl would write for {target.recipe.id}: "
                                  f"{real(line[len('Error: '):].strip())}")
        detail = next((real(line.strip()) for line in result.stderr.splitlines() if line.strip()),
                      f"exit {result.returncode}")
        raise ClientError(f"dsh --dump-config failed: {detail}; check that dsh runs in this shell, or {OUT}")
    ids = [row_id for row_id in OWNED_ROWS if row_id in written]
    for row_id in ids:
        if f'patch: entry "{row_id}" not found' in result.stderr:
            raise ClientError(f"dsh {installed_version(host)} has no {row_id} row, so llmctl's row would not apply; "
                              f"llmctl's dsh rows were checked with dsh {VERIFIED_VERSION}; {OUT} until llmctl "
                              "supports this version")
    applied = _applied(result.stdout, patch)
    for row_id in ids:
        if row_id not in applied:
            raise ClientError(f"dsh did not apply llmctl's {row_id} row from {path}; {OUT} and report it")


def _effort_levels(meta: ClientMeta) -> list[str]:
    """dsh's Effort levels of an effort recipe: the keys of the model's reasoningEfforts (design D3)."""
    return ["off" if effort == "none" else effort for effort in meta.efforts]


def thinking_text(meta: ClientMeta) -> str:
    """How dsh's model picker chooses this recipe's thinking (design D7, Thinking)."""
    if meta.thinking == "toggle" and meta.default == "on":
        return "on; choose Off under Effort to turn it off"
    if meta.thinking == "toggle" and meta.medium_kwargs:
        return "off; Medium under Effort for medium effort, High for full thinking"
    if meta.thinking == "toggle":
        return "off; choose Medium under Effort in dsh's model picker to turn it on"
    if meta.thinking == "effort":
        return f"medium effort; choose another under Effort ({', '.join(_effort_levels(meta))})"
    if meta.thinking == "always":
        return "always on, at the effort the server sets"
    if meta.thinking == "none":
        return "none"
    raise ValueError(f"unknown thinking type {meta.thinking!r}")


def usage_lines(target: Target, *, held: bool, up_command: str) -> list[str]:
    """The dsh section of up (design D7): the default, how to use it, thinking, chat only and compaction."""
    recipe = target.recipe
    model = f"{PROVIDER}/{target.served}"
    if held:
        lines = [f"dsh: {model}, the default in every dsh profile",
                 '  dsh web, or for a single prompt: dsh --profile headless "your prompt"']
    else:
        lines = [f"dsh: {model}, not the default ({up_command} --set-default makes it one)",
                 f"  dsh web, then pick {recipe.id} on {target.label} (llmctl) in its model picker"]
    lines.append(f"  thinking: {thinking_text(recipe.clients)}")
    if recipe.chat_only:
        rows_hint = "use dsh --profile headless --patch <file> with the rows in docs/clients.md#chat-only-recipes"
        if target.runtime.refuses_tools_without_parser:
            lines.append(f"  chat only: {target.label} rejects every dsh request that carries tools for this recipe "
                         f"(HTTP 400); {rows_hint}")
        else:
            lines.append(f"  chat only: dsh still sends its tools; for a prompt without them {rows_hint}")
    output = recipe.clients.output
    if recipe.context - output <= COMPACTION_HEADROOM:
        lines.append(f"  compaction: dsh compacts these sessions only after the server rejects an overlong prompt, "
                     f"because context {recipe.context} minus output {output} leaves no room for its 65,536-token "
                     "headroom (docs/clients.md#context-compaction-and-timeouts)")
    return lines


def notes(old: Block | None, target: Target, *, path: Path, held: bool, findings: Sequence[Finding],
          up_command: str) -> list[str]:
    """The notes after this up wrote the home patch (design D7); old is the block read just before the write."""
    served = target.served
    changed = old is not None and old.served is not None and old.served != served
    sessions = (f", and its open sessions on {PROVIDER}/{old.served} fail until you pick {PROVIDER}/{served}"
                if changed else "")
    lines = [f"note: dsh web uses the new rows within seconds{sessions}; dsh headless, acp and sdk processes that are "
             "already running keep the old rows until they restart"]
    if held and (old is None or not old.holds_default):
        lines.append("note: llmctl now holds dsh's default, so dsh's model picker changes only the open session and "
                     "cannot save another default (dsh refuses that save without a message); delete the "
                     f"agent-default-model and acp lines in {path} to give it back")
    if old is None or _row(old.rows, "llm-pi-ai") is None:
        lines.append(f"note: while llmctl's llm-pi-ai row is in {path}, dsh's Models page cannot add or edit "
                     'OpenAI-compatible providers (it says Configuration for "llm-pi-ai" is overridden by a home '
                     f"patch or command-line overlay); {OUT} if you need them (docs/clients.md#the-home-patch)")
    if not held and changed:
        for finding in findings:
            if finding.row_id != "agent-default-model":
                continue
            lines.append(f"note: {finding.path} line {finding.line} sets dsh's default model for the "
                         f"{finding.profile} profile, which llmctl does not change; if it names "
                         f"{PROVIDER}/{old.served}, dsh asks for a model again ({up_command} --set-default lets "
                         "llmctl hold dsh's default)")
    return lines


def switch_back(env: Mapping[str, str]) -> str:
    """The down hint that gives dsh's default back to its model picker."""
    return f"dsh: delete the agent-default-model and acp lines in {home_patch(env)}"
