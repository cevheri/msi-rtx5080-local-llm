"""The Hermes profile llmctl owns: the managed block of <root>/profiles/llmctl/config.yaml.

Every value in the block is written as one JSON flow line, which Hermes reads as YAML 1.1, so llmctl needs no YAML
library; see docs/superpowers/specs/2026-11-16-llmctl-clients-design.md, section D1.
"""

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path

from launcher.errors import ClientError, ClientWriteError
from launcher.files import FileChange, bare, env_path, find_markers, home_dir, split_lines
from launcher.host import Host
from launcher.model import Target

HERMES_CONFIG_VERSION = 53  # DEFAULT_CONFIG["_config_version"] of Hermes Agent v0.23.1
MIN_CONTEXT = 64000
PROFILE = "llmctl"
OWNED_KEYS = ("model", "providers", "model_overrides", "gateway", "platform_toolsets")
HEADER = ("# Hermes profile for the model llmctl runs (docs/llmctl.md#clients).\n"
          "# llmctl rewrites the block below on every ./llmctl up and keeps everything outside it.\n"
          "# The keys model, providers, model_overrides, gateway and platform_toolsets belong to the block.\n")
BEGIN = "# >>> llmctl managed block: rewritten by llmctl on every start; edits here are lost >>>"
END = "# <<< llmctl managed block <<<"
IDENTITY_MARKERS = ("config.yaml", ".env", "SOUL.md", "profile.yaml", "auth.json", "state.db")
OUT = "leave Hermes out with LLMCTL_CLIENTS"
_OWNED_LINE = re.compile(r"^(model|providers|model_overrides|gateway|platform_toolsets)[ \t]*:")
_PLAIN_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*[ \t]*:([ \t]|$)")
_AGENT = re.compile(r"^agent[ \t]*:[ \t]*(#.*)?$")
_REASONING_EFFORT = re.compile(r"^reasoning_effort[ \t]*:(.*)$")
_YAML_BREAK = re.compile("\r\n|[\n\r\x85\u2028\u2029]")  # every line break of YAML 1.1
_LONE_BREAKS = ("\r", "\x85", "\u2028", "\u2029")


def block_values(target: Target) -> dict[str, object]:
    """The owned keys and their values, in block order; platform_toolsets only for a chat-only recipe."""
    recipe, served = target.recipe, target.served
    meta = recipe.clients
    values = {
        "model": {"provider": "custom:llmctl", "default": served},
        "providers": {"llmctl": {
            "name": f"llmctl: {recipe.id} on {target.label}",
            "api": f"http://127.0.0.1:{target.runtime.host_port}/v1",
            "transport": "chat_completions",
            "default_model": served,
            "discover_models": False,
            "extra_body": {"max_tokens": meta.output},
            "models": {served: {"context_length": recipe.context}},
        }},
        "model_overrides": {} if meta.thinking == "effort" else {"custom": {served: {"supports_reasoning": False}}},
        "gateway": {"standalone": True},
    }
    if recipe.chat_only:
        values["platform_toolsets"] = {"cli": []}
    return values


def render_block(target: Target) -> str:
    """The lines between the markers, each `<key>: <JSON value>` and ending in a newline."""
    return "".join(f"{key}: {json.dumps(value, ensure_ascii=True, separators=(', ', ': '))}\n"
                   for key, value in block_values(target).items())


def new_config_text(target: Target) -> str:
    """The profile config llmctl creates: header, block, and the version stamp after the end marker."""
    return HEADER + BEGIN + "\n" + render_block(target) + END + "\n" + f"_config_version: {HERMES_CONFIG_VERSION}\n"


def hermes_root(env: Mapping[str, str]) -> Path:
    """Hermes' root folder, as _hermes_root_of in Hermes' hermes_constants.py (spec D3)."""
    default = home_dir(env) / (".hermes" + env.get("HERMES_DATA_DIR_SUFFIX", ""))
    value = env.get("HERMES_HOME", "")
    if not value:
        return default
    home = env_path(env, value)
    real, real_default = os.path.realpath(home), os.path.realpath(default)
    if real == real_default or real.startswith(real_default.rstrip(os.sep) + os.sep):
        return default
    return home.parent.parent if home.parent.name == "profiles" else home


def profile_dir(env: Mapping[str, str]) -> Path:
    return hermes_root(env) / "profiles" / PROFILE


def profile_config(env: Mapping[str, str]) -> Path:
    return profile_dir(env) / "config.yaml"


def _read_text(path: Path) -> str | None:
    """The file's text with its line endings as they are; None when it does not exist."""
    try:
        with open(path, encoding="utf-8", newline="") as handle:
            return handle.read()
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError) as exc:
        raise ClientError(f"cannot read {path}: {exc}") from exc


def _markers(path: Path, lines: list[str]) -> tuple[int, int]:
    """The indices of the begin and end marker lines, or a refusal."""
    found = find_markers(path, lines, begin=BEGIN, end=END)
    if found is None:
        raise ClientError(f"{path} has no llmctl block, so the Hermes profile llmctl was not created by llmctl; "
                          f"rename it with hermes profile rename llmctl <name>, or {OUT}")
    return found


def _check_outside(path: Path, lines: list[str], begin: int, end: int) -> None:
    for index, line in enumerate(lines):
        if begin <= index <= end:
            continue
        text = bare(line)
        if any(char in text for char in _LONE_BREAKS):
            raise ClientError(f"{path} line {index + 1}: llmctl cannot check this top-level line; "
                              "write top-level keys as plain 'key:'")
        if not text.strip() or text.startswith(("#", " ", "\t")):
            continue
        owned = _OWNED_LINE.match(text)
        if owned:
            raise ClientError(f"{path} line {index + 1}: '{owned.group(1)}' is set outside the llmctl block; "
                              "llmctl owns it in this profile and Hermes stops on a duplicate key; delete that setting")
        if not _PLAIN_KEY.match(text):
            raise ClientError(f"{path} line {index + 1}: llmctl cannot check this top-level line; "
                              "write top-level keys as plain 'key:'")


def profile_block(env: Mapping[str, str]) -> str | None:
    """The lines between the markers of the existing profile, each ending in a newline; None when it is missing."""
    path = profile_config(env)
    text = _read_text(path)
    if text is None:
        return None
    lines = split_lines(text)
    begin, end = _markers(path, lines)
    return "".join(bare(line) + "\n" for line in lines[begin + 1:end])


def block_description(block: str) -> str | None:
    """`<id> on <label>` from the block's providers line; None when it does not parse."""
    for line in block.splitlines():
        key, sep, value = line.partition(": ")
        if key != "providers" or not sep:
            continue
        try:
            name = json.loads(value)["llmctl"]["name"]
        except (ValueError, KeyError, TypeError):
            return None
        if isinstance(name, str) and name.startswith("llmctl: "):
            return name[len("llmctl: "):]
        return None
    return None


def _block_reparses(block: str, target: Target) -> bool:
    expected = block_values(target)
    pairs = []
    for line in block.splitlines():
        key, sep, value = line.partition(": ")
        if not sep:
            return False
        try:
            pairs.append((key, json.loads(value)))
        except ValueError:
            return False
    return [key for key, _ in pairs] == list(expected) and dict(pairs) == expected


def plan_profile(env: Mapping[str, str], target: Target) -> FileChange | None:
    """The change that makes the llmctl profile describe target, None when it does already, or a refusal."""
    recipe = target.recipe
    description = f"{recipe.id} on {target.label}"
    if recipe.context < MIN_CONTEXT:
        raise ClientError(f"Hermes refuses a context below 64,000 tokens and {recipe.id} has {recipe.context}; "
                          "leave Hermes out with LLMCTL_CLIENTS=opencode,pi")
    root = hermes_root(env)
    tombstone = root / "profiles" / ".deleted" / PROFILE
    if os.path.lexists(tombstone):
        raise ClientError(f"Hermes marks the profile llmctl as deleted ({tombstone}); delete that file to let llmctl "
                          f"recreate the profile, or {OUT}")
    if not os.path.isdir(root):
        raise ClientError(f"{root} does not exist; llmctl creates a Hermes profile only inside an existing Hermes home")
    directory = profile_dir(env)
    path = directory / "config.yaml"
    if not os.path.lexists(path):
        for marker in IDENTITY_MARKERS[1:]:
            if os.path.lexists(directory / marker):
                raise ClientError(f"{directory} holds a Hermes profile that llmctl did not create ({marker} without "
                                  f"config.yaml); rename it with hermes profile rename llmctl <name>, or {OUT}")
        summary = (f"created the Hermes profile llmctl for {description} "
                   "(standalone, not served by Hermes' gateway)")
        return FileChange(path, None, new_config_text(target), summary, f"would write {path}: {summary}",
                          mode=0o600, backup=False, dirs=(root / "profiles", directory))
    old = _read_text(path)
    if old is None:
        raise ClientError(f"cannot read {path}: it disappeared while llmctl read it")
    lines = split_lines(old)
    begin, end = _markers(path, lines)
    _check_outside(path, lines, begin, end)
    before, after = "".join(lines[:begin + 1]), "".join(lines[end:])
    block = render_block(target)
    new = before + block + after
    if not (new.startswith("".join(lines[:begin])) and new.endswith("".join(lines[end + 1:]))
            and _block_reparses(new[len(before):len(new) - len(after)], target)):
        raise ClientError(f"{path}: llmctl's own check of the new profile failed; nothing was written")
    if new == old:
        return None
    summary = f"llmctl block for {description}"
    previous = block_description("".join(bare(line) + "\n" for line in lines[begin + 1:end]))
    if previous == description:
        summary += " (same recipe, block rewritten)"
    elif previous is not None:
        summary += f" (was {previous})"
    return FileChange(path, old, new, summary, f"would write {path}: {summary}", mode=0o600, backup=False)


def sticky_profile(env: Mapping[str, str]) -> str:
    """Hermes' sticky default profile: <root>/active_profile, `default` when missing or empty."""
    path = hermes_root(env) / "active_profile"
    try:
        value = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return "default"
    except (OSError, UnicodeDecodeError) as exc:
        raise ClientError(f"cannot read {path}: {exc}") from exc
    return value or "default"


def use_profile(host: Host) -> list[str]:
    """Make llmctl Hermes' sticky default profile through Hermes itself (spec D2, L6)."""
    command = ["hermes", "profile", "use", PROFILE]
    try:
        previous = sticky_profile(host.env)
    except ClientError as exc:
        raise ClientWriteError(str(exc)) from exc
    result = host.run(command)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ClientWriteError(f"hermes profile use llmctl exited {result.returncode}: {detail}")
    try:
        value = sticky_profile(host.env)
    except ClientError as exc:
        raise ClientWriteError(f"hermes profile use llmctl exited 0, but {exc}") from exc
    if value != PROFILE:
        raise ClientWriteError(f"hermes profile use llmctl exited 0, but {hermes_root(host.env)}/active_profile "
                               f"reads '{value}'")
    return [
        "ran hermes profile use llmctl: plain hermes commands, hermes gateway and hermes config included, "
        f"now use the profile llmctl; switch back with hermes profile use {previous}",
        f"Hermes default profile: {previous} -> llmctl",
    ]


def gateway_warning(env: Mapping[str, str]) -> str | None:
    """A warning when Hermes' running gateway serves the llmctl profile (spec D3)."""
    path = hermes_root(env) / "gateway_state.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        return f"note: cannot read {path}: {exc}"
    if not isinstance(data, dict):
        return f"note: cannot read {path}: not a JSON object"
    served = data.get("served_profiles")
    if data.get("gateway_state") == "running" and isinstance(served, list) and PROFILE in served:
        return (f"warning: Hermes' gateway serves the profile llmctl ({path}), so its sessions run inside your "
                "messaging gateway; this Hermes may no longer honour gateway.standalone (docs/clients.md#hermes-agent)")
    return None


def _scalar(value: str) -> str:
    """A plain YAML value without a trailing comment and surrounding quotes."""
    if value.startswith("#"):
        return ""
    comment = value.find(" #")
    if comment >= 0:
        value = value[:comment]
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        value = value[1:-1]
    return value


def reasoning_effort_warning(env: Mapping[str, str]) -> str | None:
    """A warning when the profile sets agent.reasoning_effort outside the block (spec L5)."""
    path = profile_config(env)
    text = _read_text(path)
    if text is None:
        return None
    lines = _YAML_BREAK.split(text)
    begin, end = _markers(path, lines)
    for index, line in enumerate(lines):
        if begin <= index <= end or not _AGENT.match(line):
            continue
        indent = None
        for child in lines[index + 1:]:
            stripped = child.lstrip(" \t")
            if not stripped or stripped.startswith("#"):
                continue
            if child == stripped:
                break
            prefix = child[:len(child) - len(stripped)]
            if indent is None:
                indent = prefix
            if prefix != indent:
                continue
            match = _REASONING_EFFORT.match(stripped)
            value = _scalar(match.group(1).strip()) if match else ""
            if value:
                return (f"warning: {path} sets agent.reasoning_effort {value} outside the llmctl block, so Hermes "
                        "sends it to every model instead of the server's default; clear it with: "
                        'hermes -p llmctl config set agent.reasoning_effort ""')
    return None


def hermes_home_note(env: Mapping[str, str]) -> str | None:
    """A note when HERMES_HOME names a profile folder, which makes Hermes ignore the sticky profile (spec D2)."""
    value = env.get("HERMES_HOME", "")
    if not value:
        return None
    home = env_path(env, value)
    if home.parent.name != "profiles":
        return None
    return (f"note: HERMES_HOME names the profile {home.name} in this shell, so plain hermes ignores the default "
            "profile; use hermes -p llmctl")
