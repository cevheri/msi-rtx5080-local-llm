"""Kimi Code CLI: the two tables llmctl owns in Kimi's <Kimi home>/config.toml.

llmctl writes its TOML lines itself, each value JSON-encoded, so it needs no TOML writer; see
docs/superpowers/specs/2026-11-17-llmctl-kimi-design.md, sections D1 (Kimi's home and file), D2 (the tables llmctl
owns and their key order), D3 (finding, replacing and inserting them), D4 (values as TOML), D5 (default_model), D6
(thinking per type), D7 (chat-only recipes and the agent file), D8 (the checks before writing), D9 (refusals and the
read-back) and D10 (the lines up, status, list and down print).
"""

import json
import os
import shutil
import tempfile
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from launcher.errors import ClientError, ClientWriteError, UsageError
from launcher.files import FileChange, bare, home_dir, split_lines
from launcher.host import Host
from launcher.model import ClientMeta, Target

PROVIDER = "llmctl"
ALIAS_PREFIX = "llmctl/"
VERIFIED_VERSION = "2.5.3"
API_KEY = "no-key-required"
SWITCHES = (("KIMI_DISABLE_TELEMETRY", "1"), ("KIMI_CODE_NO_AUTO_UPDATE", "1"), ("KIMI_CODE_REGION_MARKER", "off"))
AGENT_FILE = Path(__file__).resolve().parent / "kimi-chat-only.md"
OUT = "leave Kimi out with LLMCTL_CLIENTS"
RETRY_NOTE = "  Kimi retries a stopped server for about two and a half minutes before a prompt fails"
SUFFIX = " (llmctl)"
MISSING = object()


def kimi_home(env: Mapping[str, str]) -> Path:
    """Kimi's home folder, as Kimi 2.5.3 finds it (design D1).

    `~` and `$NAME` are never expanded, because Kimi does not expand them; an empty or relative value is refused,
    because Kimi then reads different files in different commands, or resolves it against each working folder.
    """
    if "KIMI_CODE_HOME" not in env:
        return home_dir(env) / ".kimi-code"
    value = env["KIMI_CODE_HOME"]
    if value == "":
        raise UsageError("KIMI_CODE_HOME is set but empty, and Kimi 2.5.3 then reads config.toml from the working "
                         "folder in some commands and from ~/.kimi-code in others; unset it, or set it to an "
                         "absolute path")
    if not value.startswith("/"):
        raise UsageError(f"KIMI_CODE_HOME is relative ({value}), and Kimi uses it as given, against each command's "
                         "working folder; set it to an absolute path, or unset it for ~/.kimi-code")
    home = os.path.normpath(value)
    if home.startswith("//"):
        home = home[1:]
    return Path(home)


def config_path(env: Mapping[str, str]) -> Path:
    return kimi_home(env) / "config.toml"


def alias(target: Target) -> str:
    return ALIAS_PREFIX + target.served


def provider_table(target: Target) -> dict:
    """The keys of `[providers.llmctl]`, in design D2 order."""
    return {"type": "openai", "base_url": f"http://127.0.0.1:{target.runtime.host_port}/v1", "api_key": API_KEY,
            "model_source": "static"}


def model_table(target: Target) -> dict:
    """The keys of `[models."llmctl/<served name>"]` for the recipe's thinking type, in design D2 order."""
    recipe = target.recipe
    clients = recipe.clients
    if recipe.chat_only:
        capabilities = []
    else:
        capabilities = ["tool_use"]
        if clients.thinking == "effort":
            capabilities.append("thinking")
    table = {"provider": PROVIDER, "model": target.served, "display_name": f"{recipe.id} on {target.label} (llmctl)",
             "max_context_size": recipe.context, "max_output_size": clients.output, "capabilities": capabilities}
    if clients.thinking == "effort":
        table["support_efforts"] = [effort for effort in clients.efforts if effort != "none"]
        table["default_effort"] = "medium"
        if "none" in clients.efforts:
            table["off_effort"] = "none"
    return table


def toml_value(value: str | int | list[str]) -> str:
    """A TOML value from a string, an integer or a list of strings; JSON's string escapes are valid TOML."""
    if isinstance(value, bool):
        raise ValueError(f"llmctl writes no TOML booleans: {value!r}")
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False).replace("\x7f", "\\u007f")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, list):
        for item in value:
            if not isinstance(item, str):
                raise ValueError(f"llmctl writes only lists of strings: {value!r}")
        return "[" + ", ".join(toml_value(item) for item in value) + "]"
    raise ValueError(f"llmctl writes only strings, integers and lists of strings: {value!r}")


def render_tables(target: Target) -> str:
    """The text of llmctl's two tables: the provider, an empty line, the model alias."""
    lines = ["[providers.llmctl]"]
    lines += [f"{key} = {toml_value(value)}" for key, value in provider_table(target).items()]
    lines += ["", "[models." + toml_value(alias(target)) + "]"]
    lines += [f"{key} = {toml_value(value)}" for key, value in model_table(target).items()]
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class State:
    """Kimi's config.toml as llmctl reads it; provider, aliases and thinking hold only tables."""

    path: Path
    text: str
    data: dict
    provider: dict | None
    aliases: tuple[tuple[str, dict], ...]
    default: object
    thinking: dict | None

    @property
    def description(self) -> str | None:
        """The first alias's display_name without its final " (llmctl)"; None when it is not one of llmctl's."""
        if not self.aliases:
            return None
        name = self.aliases[0][1].get("display_name")
        if isinstance(name, str) and name.endswith(SUFFIX) and len(name) > len(SUFFIX):
            return name[:-len(SUFFIX)]
        return None

    @property
    def served(self) -> str | None:
        if not self.aliases:
            return None
        model = self.aliases[0][1].get("model")
        return model if isinstance(model, str) else None


@dataclass(frozen=True)
class _Item:
    """A trivia line, a header or a statement over lines start to end (inclusive, from 0); path is a key path."""

    kind: str
    start: int
    end: int
    path: tuple
    array: bool


@dataclass(frozen=True)
class _Table:
    """A table from its header to the last line of its last statement (inclusive, from 0)."""

    path: tuple
    array: bool
    start: int
    end: int


def _header_path(body: str) -> tuple[tuple, bool]:
    """The key path of a header line and whether it opens an array of tables."""
    data = tomllib.loads(body)
    path, node = [], data
    while isinstance(node, dict) and len(node) == 1:
        key = next(iter(node))
        path.append(key)
        node = node[key]
        if node == {}:
            return tuple(path), False
        if node == [{}]:
            return tuple(path), True
    raise ValueError(f"not a table header: {body!r}")


def _layout(text: str) -> tuple[list[str], list[_Item], list[_Table]]:
    """The lines, items and tables of a text that tomllib already parsed (design D3 steps 2 to 5)."""
    lines = split_lines(text)
    items: list[_Item] = []
    i = 0
    while i < len(lines):
        body = bare(lines[i])
        stripped = body.strip()
        if not stripped or stripped.startswith("#"):
            items.append(_Item("trivia", i, i, (), False))
            i += 1
            continue
        if stripped.startswith("["):
            path, array = _header_path(body)
            items.append(_Item("header", i, i, path, array))
            i += 1
            continue
        for j in range(i, len(lines)):
            try:
                data = tomllib.loads("".join(lines[i:j + 1]))
            except tomllib.TOMLDecodeError:
                continue
            items.append(_Item("statement", i, j, (next(iter(data)),), False))
            i = j + 1
            break
        else:
            raise ValueError(f"line {i + 1}: no statement ends there")
    tables: list[_Table] = []
    for index, item in enumerate(items):
        if item.kind != "header":
            continue
        end = item.end
        for following in items[index + 1:]:
            if following.kind == "header":
                break
            if following.kind == "statement":
                end = following.end
        tables.append(_Table(item.path, item.array, item.start, end))
    return lines, items, tables


def _owned(path: tuple) -> bool:
    """True for llmctl's provider, its alias tables and every table below them (design D3 step 6)."""
    if len(path) < 2:
        return False
    if path[0] == "providers":
        return path[1] == PROVIDER
    return path[0] == "models" and isinstance(path[1], str) and path[1].startswith(ALIAS_PREFIX)


def _parse(path: Path, text: str) -> dict:
    if text.startswith("﻿"):
        raise ClientError(f"{path} is not valid TOML (it starts with a byte-order mark, which TOML does not allow); "
                          f"fix it, or {OUT}")
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ClientError(f"{path} is not valid TOML ({exc}); fix it, or {OUT}") from exc


def parse_state(path: Path, text: str) -> State:
    """The state of a config.toml text; refuses what llmctl cannot edit line by line."""
    data = _parse(path, text)
    if "providers" in data and not isinstance(data["providers"], dict):
        raise ClientError(f'{path}: "providers" is not a table; Kimi needs [providers.<name>] tables; fix it, or {OUT}')
    if "models" in data and not isinstance(data["models"], dict):
        raise ClientError(f'{path}: "models" is not a table; Kimi needs [models."<alias>"] tables; fix it, or {OUT}')
    providers = data.get("providers", {})
    models = data.get("models", {})
    headers = {table.path for table in _layout(text)[2] if not table.array}
    if PROVIDER in providers and ("providers", PROVIDER) not in headers:
        raise ClientError(f"{path}: llmctl's provider is written with dotted keys or as an inline table, so llmctl "
                          "cannot replace it line by line; write it as a [providers.llmctl] table, or delete it")
    for key in models:
        if key.startswith(ALIAS_PREFIX) and ("models", key) not in headers:
            raise ClientError(f"{path}: the model {key} is written with dotted keys or as an inline table, so llmctl "
                              f'cannot replace it line by line; write it as a [models."{key}"] table, or delete it')
    provider = providers.get(PROVIDER)
    thinking = data.get("thinking")
    return State(path=path, text=text, data=data, provider=provider if isinstance(provider, dict) else None,
                 aliases=tuple((key, value) for key, value in models.items()
                               if key.startswith(ALIAS_PREFIX) and isinstance(value, dict)),
                 default=data.get("default_model", MISSING),
                 thinking=thinking if isinstance(thinking, dict) else None)


def read_text(path: Path) -> str | None:
    """The file's text with its line endings kept; None when it does not exist."""
    try:
        with open(path, encoding="utf-8", newline="") as handle:
            return handle.read()
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError) as exc:
        raise ClientError(f"cannot read {path}: {exc}") from exc


def read_state(env: Mapping[str, str]) -> State | None:
    """The state of Kimi's config.toml, its line endings kept; None when the file does not exist."""
    path = config_path(env)
    text = read_text(path)
    return None if text is None else parse_state(path, text)


def _equal(found: object, wanted: object) -> bool:
    """TOML value equality in which a bool never equals an int."""
    if isinstance(found, list) and isinstance(wanted, list):
        return len(found) == len(wanted) and all(_equal(a, b) for a, b in zip(found, wanted))
    if isinstance(found, bool) or isinstance(wanted, bool):
        return type(found) is type(wanted) and found == wanted
    return found == wanted


def _holds(table: dict, wanted: dict) -> bool:
    return all(key in table and _equal(table[key], value) for key, value in wanted.items())


def entry_matches(state: State | None, target: Target) -> bool:
    """True when the file holds llmctl's tables for target by value, as its only llmctl alias."""
    if state is None or state.provider is None or not _holds(state.provider, provider_table(target)):
        return False
    if len(state.aliases) != 1 or state.aliases[0][0] != alias(target):
        return False
    return _holds(state.aliases[0][1], model_table(target))


def points_at_llmctl(state: State | None) -> bool:
    return state is not None and isinstance(state.default, str) and state.default.startswith(ALIAS_PREFIX)


def foreign_lines(path: Path, text: str) -> list[str]:
    """Every non-blank line outside llmctl's tables and the root default_model statement, as bare lines."""
    parse_state(path, text)
    lines, items, tables = _layout(text)
    mine: set[int] = set()
    for table in tables:
        if _owned(table.path):
            mine.update(range(table.start, table.end + 1))
    first_header = next((item.start for item in items if item.kind == "header"), len(lines))
    for item in items:
        if item.kind == "statement" and item.start < first_header and item.path == ("default_model",):
            mine.update(range(item.start, item.end + 1))
    return [bare(line) for index, line in enumerate(lines) if index not in mine and bare(line).strip()]


_PROVIDER_KEYS = frozenset(("type", "base_url", "api_key", "model_source"))
_MODEL_KEYS = frozenset(("provider", "model", "display_name", "max_context_size", "max_output_size", "capabilities",
                         "support_efforts", "default_effort", "off_effort"))


def _same(found: object, wanted: object) -> bool:
    """Parsed TOML values equal in type and value: a bool never equals an int, and nan equals nan."""
    if type(found) is not type(wanted):
        return False
    if isinstance(found, dict):
        return found.keys() == wanted.keys() and all(_same(found[key], wanted[key]) for key in found)
    if isinstance(found, list):
        return len(found) == len(wanted) and all(_same(a, b) for a, b in zip(found, wanted))
    if isinstance(found, float) and found != found:
        return wanted != wanted
    return found == wanted


def _strip(data: dict) -> dict:
    """The data without llmctl's provider, its aliases and default_model; an emptied providers or models goes too."""
    rest = dict(data)
    rest.pop("default_model", None)
    for key in ("providers", "models"):
        if isinstance(rest.get(key), dict):
            kept = {name: value for name, value in rest[key].items()
                    if not (key == "providers" and name == PROVIDER)
                    and not (key == "models" and name.startswith(ALIAS_PREFIX))}
            if kept:
                rest[key] = kept
            else:
                del rest[key]
    return rest


def _eol(lines: list[str]) -> str:
    """CRLF when the first line ends with CRLF, else LF, as Kimi's detectEol decides (design D3 step 6)."""
    return "\r\n" if lines and lines[0].endswith("\r\n") else "\n"


def _root_statements(items: list[_Item]) -> list[_Item]:
    first_header = next((item.start for item in items if item.kind == "header"), None)
    return [item for item in items
            if item.kind == "statement" and (first_header is None or item.start < first_header)]


def _default_edit(path: Path, lines: list[str], items: list[_Item], data: dict, value: str | None,
                  eol: str) -> tuple[dict[int, tuple[int, list[str]]], list[tuple[int, int, list[str]]]]:
    """The replacements and the insert that make the top-level default_model value; None removes it (design D5)."""
    root = _root_statements(items)
    held = [item for item in root if item.path == ("default_model",)]
    current = data.get("default_model", MISSING)
    if value is None:
        if current is MISSING:
            return {}, []
        if not held:
            raise ClientError(f"could not remove the top-level default_model in {path} safely; remove it yourself")
        return {item.start: (item.end, []) for item in held}, []
    if current is MISSING:
        position = root[-1].end + 1 if root else 0
        return {}, [(position, -1, [f"default_model = {toml_value(value)}{eol}"])]
    if _same(current, value):
        return {}, []
    refusal = ClientError(f'could not change the top-level default_model in {path} safely; set it to "{value}" '
                          "yourself")
    if len(held) != 1 or held[0].start != held[0].end:
        raise refusal
    line = lines[held[0].start]
    body = bare(line)
    equals = body.find("=")
    if equals < 0:
        raise refusal
    start = equals + 1
    while start < len(body) and body[start] in " \t":
        start += 1
    token = body[start:]
    if token.startswith('"') and not token.startswith('"""'):
        end = 1
        while end < len(token) and token[end] != '"':
            end += 2 if token[end] == "\\" else 1
    elif token.startswith("'") and not token.startswith("'''"):
        end = token.find("'", 1)
    else:
        raise refusal
    if not 0 < end < len(token):
        raise refusal
    new_body = body[:start] + toml_value(value) + token[end + 1:]
    try:
        parsed = tomllib.loads(new_body)
    except tomllib.TOMLDecodeError as exc:
        raise refusal from exc
    if not _same(parsed, {"default_model": value}):
        raise refusal
    return {held[0].start: (held[0].end, [new_body + line[len(body):]])}, []


def _apply(lines: list[str], replacements: dict[int, tuple[int, list[str]]],
           inserts: list[tuple[int, int, list[str]]], eol: str) -> str:
    """All edits in one pass over the old lines (design D3 step 7).

    At each old index the inserts come first, ordered by their order (the default -1, the provider 0, the model 1),
    then the line or the replacement of its span; an inserted table (order 0 or more) gets an empty line before it
    after a line that is not blank and one after it before such a line, and a last line without a newline gets eol
    when something lands after it.
    """
    source = list(lines)
    at: dict[int, list[tuple[int, list[str]]]] = {}
    for position, order, block in inserts:
        at.setdefault(position, []).append((order, block))
    if source and not source[-1].endswith("\n") and len(source) in at:
        source[-1] += eol
    out: list[str] = []
    index = 0
    while True:
        for order, block in sorted(at.get(index, []), key=lambda entry: entry[0]):
            block = list(block)
            if order >= 0:
                if out and bare(out[-1]).strip():
                    block = [eol] + block
                if index < len(source) and bare(source[index]).strip():
                    block = block + [eol]
            out += block
        if index >= len(source):
            return "".join(out)
        if index in replacements:
            end, block = replacements[index]
            out += block
            index = end + 1
            continue
        out.append(source[index])
        index += 1


def _parses(text: str) -> dict | None:
    try:
        text.encode("utf-8")
        return tomllib.loads(text)
    except (UnicodeEncodeError, tomllib.TOMLDecodeError):
        return None


def _checks(path: Path, old: State, new: str, target: Target, default: object) -> bool:
    """llmctl's own check of a planned config.toml (design D8 check 1)."""
    data = _parses(new)
    if data is None or not _same(_strip(data), _strip(old.data)):
        return False
    providers, models = data.get("providers"), data.get("models")
    if not isinstance(providers, dict) or not _same(providers.get(PROVIDER), provider_table(target)):
        return False
    name = alias(target)
    if not isinstance(models, dict) or [key for key in models if key.startswith(ALIAS_PREFIX)] != [name]:
        return False
    if not _same(models[name], model_table(target)) or not _same(data.get("default_model", MISSING), default):
        return False
    try:
        return foreign_lines(path, new) == foreign_lines(path, old.text)
    except ClientError:
        return False


def _own_check_failed(path: Path) -> ClientError:
    return ClientError(f"{path}: llmctl's own check of the new config.toml failed; nothing was written")


def _plan(path: Path, old: str | None, target: Target, *, set_default: bool) -> tuple[str, State, bool, bool]:
    """The planned text, the old state, whether llmctl's tables change and whether default_model changes."""
    state = parse_state(path, old or "")
    lines, items, tables = _layout(state.text)
    eol = _eol(lines)
    root = _root_statements(items)
    for item in root:
        if item.path == ("providers",):
            raise ClientError(f"{path} line {item.start + 1}: providers is written as an inline table or with dotted "
                              "keys, so llmctl cannot add its [providers.llmctl] table after it; write each provider "
                              "as a [providers.<name>] table")
        if item.path == ("models",):
            raise ClientError(f"{path} line {item.start + 1}: models is written as an inline table or with dotted "
                              f'keys, so llmctl cannot add its [models.{toml_value(alias(target))}] table after it; '
                              'write each model as a [models."<alias>"] table')
    name = alias(target)
    provider_lines = [f"[providers.{PROVIDER}]{eol}"]
    provider_lines += [f"{key} = {toml_value(value)}{eol}" for key, value in provider_table(target).items()]
    model_lines = [f"[models.{toml_value(name)}]{eol}"]
    model_lines += [f"{key} = {toml_value(value)}{eol}" for key, value in model_table(target).items()]
    providers = state.data.get("providers", {})
    models = state.data.get("models", {})
    replacements: dict[int, tuple[int, list[str]]] = {}
    inserts: list[tuple[int, int, list[str]]] = []
    keep_provider = _same(providers.get(PROVIDER), provider_table(target))
    if not keep_provider:
        owned = [table for table in tables if table.path[:2] == ("providers", PROVIDER)]
        main = [table for table in owned if len(table.path) == 2 and not table.array]
        if main:
            for table in owned:
                replacements[table.start] = (table.end, provider_lines if table is main[0] else [])
        else:
            last = [table for table in tables if table.path[0] == "providers"]
            inserts.append((last[-1].end + 1 if last else len(lines), 0, provider_lines))
    keep_model = ([key for key in models if key.startswith(ALIAS_PREFIX)] == [name]
                  and _same(models[name], model_table(target)))
    if not keep_model:
        owned = [table for table in tables if table.path[0] == "models" and _owned(table.path)]
        main = [table for table in owned if len(table.path) == 2 and not table.array]
        if main:
            for table in owned:
                replacements[table.start] = (table.end, model_lines if table is main[0] else [])
        else:
            last = [table for table in tables if table.path[0] == "models"]
            inserts.append((last[-1].end + 1 if last else len(lines), 1, model_lines))
    default = state.default
    if set_default or (isinstance(default, str) and default.startswith(ALIAS_PREFIX)):
        default_replacements, default_inserts = _default_edit(path, lines, items, state.data, name, eol)
        replacements.update(default_replacements)
        inserts += default_inserts
        default = name
    new = _apply(lines, replacements, inserts, eol)
    if not _checks(path, state, new, target, default):
        raise _own_check_failed(path)
    return new, state, not (keep_provider and keep_model), not _same(default, state.default)


def config_text(path: Path, old: str | None, target: Target, *, set_default: bool) -> str:
    """The config.toml text with llmctl's tables for target and default_model by design D5; refusals raise."""
    return _plan(path, old, target, set_default=set_default)[0]


def _default_spans(text: str) -> str:
    """The text without its root default_model statements."""
    lines, items, _ = _layout(text)
    held: set[int] = set()
    for item in _root_statements(items):
        if item.path == ("default_model",):
            held.update(range(item.start, item.end + 1))
    return "".join(line for index, line in enumerate(lines) if index not in held)


def default_text(path: Path, text: str, value: str | None) -> str:
    """The text with only the top-level default_model set to value, or removed for None (design D5)."""
    state = parse_state(path, text)
    lines, items, _ = _layout(text)
    eol = _eol(lines)
    replacements, inserts = _default_edit(path, lines, items, state.data, value, eol)
    if not replacements and not inserts:
        return text
    new = _apply(lines, replacements, inserts, eol)
    data = _parses(new)
    rest, old_rest = (None, None) if data is None else (_default_spans(new), _default_spans(text))
    if (data is None or not _same({k: v for k, v in data.items() if k != "default_model"},
                                  {k: v for k, v in state.data.items() if k != "default_model"})
            or not _same(data.get("default_model", MISSING), MISSING if value is None else value)
            or not (rest == old_rest or (not text.endswith("\n") and rest == old_rest + eol))):
        raise _own_check_failed(path)
    return new


def _writable(data: dict, table: _Table) -> bool:
    """True for an owned table that llmctl could have written (plan ruling 4); below-owned tables never are."""
    if table.array or len(table.path) != 2:
        return False
    if table.path == ("providers", PROVIDER):
        value = data["providers"][PROVIDER]
        return (value.keys() <= _PROVIDER_KEYS and _same(value.get("api_key"), API_KEY)
                and _same(value.get("model_source"), "static"))
    if table.path[0] == "models" and _owned(table.path):
        value = data["models"][table.path[1]]
        name = value.get("display_name")
        return (value.keys() <= _MODEL_KEYS and _same(value.get("provider"), PROVIDER) and isinstance(name, str)
                and name.endswith(SUFFIX))
    return False


def _needs_backup(state: State) -> bool:
    """True when the file holds a line that is not blank besides what llmctl could have written (design D9)."""
    lines, items, tables = _layout(state.text)
    mine: set[int] = set()
    for table in tables:
        if _writable(state.data, table):
            mine.update(range(table.start, table.end + 1))
    if points_at_llmctl(state):
        for item in _root_statements(items):
            if item.path == ("default_model",):
                mine.update(range(item.start, item.end + 1))
    return any(bare(line).strip() for index, line in enumerate(lines) if index not in mine)


def _summary(state: State, target: Target, created: bool, tables_changed: bool, default_changed: bool) -> str:
    """The D10 summary of a planned write (plan ruling 5)."""
    name = alias(target)
    recipe = f"{target.recipe.id} on {target.label}"
    if created:
        return (f"created it with provider {PROVIDER} and model {name} for {recipe}"
                + (f"; default_model {name}" if default_changed else ""))
    summary = f"provider {PROVIDER} and model {name} for {recipe}"
    if not state.aliases:
        summary += " (was absent)"
    elif state.description is None:
        summary += " (was an unknown model)"
    elif state.description != recipe:
        summary += f" (was {state.description})"
    elif tables_changed:
        summary += " (same recipe, tables rewritten)"
    else:
        summary += " (same recipe)"
    if default_changed:
        summary += f"; default_model {name} (was {'unset' if state.default is MISSING else state.default})"
    for key, _ in state.aliases[1:]:
        if key != name:
            summary += f"; removed the model {key}"
    return summary


def plan_config(env: Mapping[str, str], target: Target, *, set_default: bool) -> FileChange | None:
    """The change that puts llmctl's tables for target into Kimi's config.toml; None when it holds them already."""
    path = config_path(env)
    old = read_text(path)
    new, state, tables_changed, default_changed = _plan(path, old, target, set_default=set_default)
    if new == old:
        return None
    summary = _summary(state, target, not old, tables_changed, default_changed)
    return FileChange(path, old, new, summary, f"would write {path}: {summary}", mode=0o600,
                      backup=_needs_backup(state))


def plan_default(env: Mapping[str, str], value: str | None) -> FileChange | None:
    """The e2e restore: Kimi's top-level default_model set to value, or removed for None; None when unchanged."""
    path = config_path(env)
    old = read_text(path)
    if old is None:
        if value is None:
            return None
        raise ClientError(f"cannot read {path}: it does not exist")
    new = default_text(path, old, value)
    if new == old:
        return None
    state = parse_state(path, old)
    was = "unset" if state.default is MISSING else state.default
    summary = f"removed default_model (was {was})" if value is None else f"default_model {value} (was {was})"
    return FileChange(path, old, new, summary, f"would write {path}: {summary}", mode=0o600,
                      backup=_needs_backup(state))


def _first_line(text: str) -> str | None:
    return next((line.strip() for line in text.splitlines() if line.strip()), None)


def _camel(key: str) -> str:
    """Kimi's name for a TOML key: each underscore and the letter after it become that letter in upper case."""
    head, *rest = key.split("_")
    return head + "".join(part[:1].upper() + part[1:] for part in rest)


def _entry_problem(data: dict, target: Target) -> str | None:
    """The first difference between Kimi's reading and llmctl's tables, in design D2 order; no other value named."""
    name = alias(target)
    for kind, key, wanted, missing in (("providers", PROVIDER, provider_table(target), f"no provider {PROVIDER}"),
                                       ("models", name, model_table(target), f"no model {name}")):
        group = data.get(kind)
        entry = group.get(key) if isinstance(group, dict) else None
        if not isinstance(entry, dict):
            return missing
        for field, value in wanted.items():
            if _camel(field) not in entry:
                return f"{field} is missing"
            found = entry[_camel(field)]
            if not _same(found, value):
                if field == "api_key":
                    return "api_key differs from llmctl's"
                return f"{field} is {json.dumps(found)} instead of {json.dumps(value)}"
    return None


def check_config(host: Host, text: str, target: Target, *, path: Path) -> str | None:
    """Let the installed Kimi read the candidate in a private home (design D8 check 2); refuse what it misreads.

    `path` is the real config.toml, named in the messages. Returns the version warning, or None on Kimi 2.5.3.
    The list's stdout holds every provider's API key, so no message holds its text other than an llmctl value.
    """
    try:
        tmp = tempfile.mkdtemp(prefix="llmctl-kimi-")
        try:
            os.chmod(tmp, 0o700)
            home = os.path.join(tmp, "home")
            cache = os.path.join(tmp, "cache")
            os.mkdir(home, 0o700)
            os.mkdir(cache, 0o700)
            fd = os.open(os.path.join(home, "config.toml"), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                handle.write(text)
            env_extra = {"KIMI_CODE_HOME": home, "KIMI_CODE_CACHE_DIR": cache, **dict(SWITCHES)}
            result = host.run(["kimi", "--version"], env_extra=env_extra)
            if result.returncode != 0:
                detail = _first_line(result.stderr) or f"exit {result.returncode}"
                raise ClientError(f"kimi --version failed: {detail}; check that kimi runs in this shell, or {OUT}")
            version = _first_line(result.stdout)
            if version is None:
                raise ClientError(f"kimi --version failed: it printed no version; check that kimi runs in this "
                                  f"shell, or {OUT}")
            listed = host.run(["kimi", "provider", "list", "--json"], env_extra=env_extra)
        finally:
            shutil.rmtree(tmp)
    except (OSError, UnicodeError) as exc:
        raise ClientError(f"cannot check {path} with kimi: {exc}") from exc
    if listed.returncode != 0:
        detail = _first_line(listed.stderr) or f"exit {listed.returncode}"
        raise ClientError(f"kimi provider list --json failed: {detail}; check that kimi runs in this shell, or {OUT}")
    try:
        data = json.loads(listed.stdout)
    except ValueError:
        data = None
    if not isinstance(data, dict):
        raise ClientError("kimi provider list --json failed: its output is not a JSON object; check that kimi runs "
                          f"in this shell, or {OUT}")
    problem = _entry_problem(data, target)
    if problem is not None:
        message = (f"Kimi {version} does not read llmctl's entry for {target.recipe.id} from the new config.toml: "
                   f"{problem}")
        if version == VERIFIED_VERSION:
            message += f"; run kimi doctor config {path} to see whether Kimi reads the rest of the file"
        else:
            message += (f"; llmctl's entry was checked with Kimi {VERIFIED_VERSION}, so leave Kimi out with "
                        "LLMCTL_CLIENTS until llmctl supports this version")
        raise ClientError(message)
    if version != VERIFIED_VERSION:
        return (f"warning: Kimi {version} is not {VERIFIED_VERSION}, the version llmctl's Kimi entry was checked "
                "with; run the Kimi steps of the e2e before relying on it (docs/clients.md#maintenance)")
    return None


CHANGED = "the file changed while llmctl wrote it, probably through Kimi; check it, then run ./llmctl up again"


def unchanged_since_plan(change: FileChange) -> None:
    """Re-read the file right before the rename; a text other than the planned old one means Kimi saved since the
    plan, so llmctl does not write over it (plan ruling 9)."""
    if read_text(change.path) != change.old_text:
        raise ClientWriteError(f"{change.path}: {CHANGED}")


def read_back(change: FileChange) -> None:
    """Read the written file back; a text other than the planned one means Kimi (or another writer) saved meanwhile."""
    try:
        with open(change.path, encoding="utf-8", newline="") as handle:
            text = handle.read()
    except (OSError, UnicodeDecodeError) as exc:
        raise ClientWriteError(f"{change.path}: cannot read it back after writing: {exc}") from exc
    if text != change.new_text:
        raise ClientWriteError(f"{change.path}: {CHANGED}")


def thinking_text(meta: ClientMeta) -> str:
    """How Kimi chooses this recipe's thinking (design D6)."""
    if meta.thinking == "toggle":
        return f"{'on' if meta.default == 'on' else 'off'}, as the server sets it"
    if meta.thinking == "effort":
        levels = ", ".join("off" if effort == "none" else effort for effort in meta.efforts)
        return f"medium effort; /effort in kimi for another ({levels})"
    if meta.thinking == "always":
        return "always on, at the effort the server sets"
    if meta.thinking == "none":
        return "none"
    raise ValueError(f"unknown thinking type {meta.thinking!r}")


def usage_lines(target: Target, *, is_default: bool, up_command: str) -> list[str]:
    """The Kimi section of up (design D10): the default, how to use it, thinking and chat only."""
    recipe = target.recipe
    name = alias(target)
    if is_default:
        lines = [f"Kimi: {name}, the default"]
        usage = ['  kimi, or for a single prompt: kimi -p "your prompt"']
    else:
        lines = [f"Kimi: {name}, not the default ({up_command} --set-default makes it one)"]
        usage = [f"  kimi -m {name}, or pick {recipe.id} on {target.label} (llmctl) with /model in kimi",
                 f"  kimi -p needs Kimi's default even with -m, so for a single prompt run {up_command} --set-default "
                 "first"]
    if not recipe.chat_only:  # Kimi's TUI always sends its tools, so a chat-only recipe gets only print mode (D7)
        lines.extend(usage)
    lines.append(f"  thinking: {thinking_text(recipe.clients)}")
    if recipe.chat_only:
        if target.label == "vLLM":
            start = ("  chat only: vLLM rejects every Kimi request that carries tools for this recipe (HTTP 400), and "
                     "Kimi's TUI always sends them")
        else:
            start = "  chat only: Kimi sends its tools with every request, also in its TUI"
        prompt = f'kimi -p "your prompt" --agent-file {AGENT_FILE}'
        if is_default:
            lines.append(f"{start}; for a prompt without them: {prompt}")
        else:
            lines.append(f"{start}; for a prompt without them, make it the default ({up_command} --set-default), "
                         f"then {prompt}")
    return lines


def _thinking_line(state: State, target: Target) -> str | None:
    """The line for the owner's [thinking] table while target's entry is the file's (design D6, first row wins)."""
    meta = target.recipe.clients
    name = alias(target)
    table = state.thinking or {}
    effort = table.get("effort")
    if meta.thinking == "effort" and (table.get("enabled") is False or effort == "off"):
        what = "enabled = false" if table.get("enabled") is False else 'effort = "off"'
        if "none" not in meta.efforts:
            return (f"warning: [thinking] {what} in {state.path} makes Kimi refuse every prompt to {name} before "
                    "sending it, because the model reasons by default and declares no off effort; set enabled = true "
                    "there, or choose an effort with /effort in kimi")
        return (f"warning: [thinking] {what} in {state.path} makes Kimi send reasoning_effort none to {name}, so "
                f"{target.recipe.id} runs without thinking; set enabled = true there, or choose an effort with "
                "/effort in kimi")
    if table.get("enabled") is False or not isinstance(effort, str) or effort in ("on", "off"):
        return None
    if meta.thinking == "effort" and effort in meta.efforts:
        if effort == "medium":
            return None
        return (f'note: [thinking] effort = "{effort}" in {state.path} makes Kimi use effort {effort} for {name} '
                "instead of medium")
    if meta.thinking == "effort":
        which = "lists only " + ", ".join(level for level in meta.efforts if level != "none")
    else:
        which = f"sets its thinking on the server ({meta.thinking})"
    return (f'warning: [thinking] effort = "{effort}" in {state.path} makes Kimi send reasoning_effort {effort} to '
            f"{name}, which {which}; the server may refuse it or think differently; remove that effort line, or "
            "choose another with /effort in kimi")


def state_lines(state: State | None, target: Target | None, env: Mapping[str, str], *, up_command: str) -> list[str]:
    """The [thinking] line, the missing-alias warning and the KIMI_MODEL_NAME note (design D10)."""
    lines = []
    if state is not None:
        if target is not None and entry_matches(state, target):
            line = _thinking_line(state, target)
            if line is not None:
                lines.append(line)
        if isinstance(state.default, str) and state.default not in state.data.get("models", {}):
            lines.append(f"warning: Kimi's default_model {state.default} names no model in {state.path}, so every "
                         f'kimi -p fails (Model "{state.default}" is not configured in config.toml.); pick a model '
                         f"with /model in kimi, or run {up_command} --set-default")
    if env.get("KIMI_MODEL_NAME"):
        if target is not None:
            name = alias(target)
        elif state is not None and state.aliases:
            name = state.aliases[0][0]
        else:
            name = ALIAS_PREFIX + "<served name>"
        lines.append("note: KIMI_MODEL_NAME is set in this shell, so kimi uses that model instead of Kimi's "
                     f"default_model; unset it, or pass -m {name}")
    return lines


def session_note(old: State | None, target: Target) -> str | None:
    """The note for an open Kimi session after this up changed llmctl's tables (design D10, E8)."""
    if old is None or not old.aliases or entry_matches(old, target):
        return None
    name = alias(target)
    if old.aliases[0][0] != name:
        return (f"note: an open Kimi session on {old.aliases[0][0]} fails its next turn until you pick {name} with "
                "/model or start a new session")
    return (f"note: an open Kimi session on {name} sends its next turn to {target.recipe.id} on {target.label} but "
            "keeps its old thinking level; start a new session")


def switch_back(env: Mapping[str, str]) -> str:
    """The down hint that gives Kimi's default back to its own model picker."""
    return "Kimi /model"
