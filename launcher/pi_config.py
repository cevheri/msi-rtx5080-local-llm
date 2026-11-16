"""Pi's llmctl provider: the entry in models.json and the default thinking level llmctl writes to settings.json.

See docs/superpowers/specs/2026-11-16-llmctl-clients-design.md, sections D4 to D6, "Generated client entries" and
"File safety per file".
"""

import fnmatch
import json
import os
import re
import shutil
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

from launcher.errors import ClientError, ClientWriteError
from launcher.files import FileChange, env_path, home_dir, load_json_object, parse_json
from launcher.host import Host
from launcher.model import ClientMeta, Target

PROVIDER = "llmctl"
PI_LEVELS = ("off", "minimal", "low", "medium", "high")
DEFAULT_LEVELS = {"effort": "medium", "always": "high", "none": "off"}  # toggle follows the server's default


def _thinking_fields(meta: ClientMeta) -> dict:
    """The model fields that tell Pi how this recipe thinks, in the order Pi's entry carries them."""
    if meta.thinking == "toggle":
        medium = dict(meta.medium_kwargs)
        level_map = {"minimal": None, "low": None} if medium else {"minimal": None, "low": None, "high": None}
        fields = {
            "thinkingLevelMap": level_map,
            "compat": {"thinkingFormat": "chat-template",
                       "chatTemplateKwargs": {"enable_thinking": {"$var": "thinking.enabled"}}},
        }
        if medium:
            fields["samplingParamsByThinkingLevel"] = {
                "medium": {"chat_template_kwargs": {"enable_thinking": True, **medium}}}
        return fields
    if meta.thinking == "effort":
        efforts = {level: "none" if level == "off" else level for level in PI_LEVELS}
        return {
            "thinkingLevelMap": {level: effort if effort in meta.efforts else None for level, effort in efforts.items()},
            "compat": {"thinkingFormat": "openai", "supportsReasoningEffort": True},
        }
    if meta.thinking == "always":
        return {
            "thinkingLevelMap": {"off": None, "minimal": None, "low": None, "medium": None},
            "compat": {"supportsReasoningEffort": False},
        }
    if meta.thinking == "none":
        return {}
    raise ValueError(f"unknown thinking type {meta.thinking!r}")


def provider_entry(target: Target) -> dict:
    recipe, meta = target.recipe, target.recipe.clients
    model = {
        "id": target.served,
        "name": f"{recipe.id} on {target.label} (llmctl)",
        "reasoning": meta.thinking != "none",
        "input": ["text"],
        "contextWindow": recipe.context,
        "maxTokens": meta.output,
        **_thinking_fields(meta),
    }
    return {
        "name": PROVIDER,
        "baseUrl": f"http://127.0.0.1:{target.runtime.host_port}/v1",
        "api": "openai-completions",
        "apiKey": "no-key-required",
        "compat": {"supportsStore": False, "supportsDeveloperRole": False, "maxTokensField": "max_tokens"},
        "models": [model],
    }


def default_level(meta: ClientMeta) -> str:
    """The level llmctl writes to modelThinkingLevels["llmctl/<served>"] (spec D4)."""
    if meta.thinking == "toggle":
        return "medium" if meta.default == "on" else "off"
    return DEFAULT_LEVELS[meta.thinking]


def models_text(doc: dict) -> str:
    return json.dumps(doc, indent=2, ensure_ascii=False) + "\n"


def new_models_text(target: Target) -> str:
    return models_text({"providers": {PROVIDER: provider_entry(target)}})


OUT = "leave Pi out with LLMCTL_CLIENTS"
LOCK_WAIT_S = 2.0
LOCK_STEP_S = 0.1
_DEFAULT_KEYS = ("defaultProvider", "defaultModel")
_LEVELS = "modelThinkingLevels"
_LLMCTL_PREFIX = PROVIDER + "/"
_ENTRY_SUFFIX = " (llmctl)"
_LOAD_ERRORS = "errors loading models.json"


def agent_dir(env: Mapping[str, str]) -> Path:
    """Pi's agent directory: PI_CODING_AGENT_DIR when set and non-empty, else ~/.pi/agent."""
    value = env.get("PI_CODING_AGENT_DIR", "")
    if value:
        return env_path(env, value)
    return home_dir(env) / ".pi" / "agent"


def models_path(env: Mapping[str, str]) -> Path:
    return agent_dir(env) / "models.json"


def settings_path(env: Mapping[str, str]) -> Path:
    return agent_dir(env) / "settings.json"


def _object_value(path: Path, doc: dict, key: str) -> dict | None:
    if key in doc and not isinstance(doc[key], dict):
        raise ClientError(f'{path}: "{key}" is not a JSON object; fix it, or {OUT}')
    return doc.get(key)


def _without_llmctl_provider(doc: dict) -> dict:
    rest = dict(doc)
    if isinstance(rest.get("providers"), dict):
        providers = {name: entry for name, entry in rest["providers"].items() if name != PROVIDER}
        if providers:
            rest["providers"] = providers
        else:
            del rest["providers"]
    return rest


def entry_description(entry: dict) -> str | None:
    """`<id> on <runtime>` from the entry's first model name, None when the entry does not carry one."""
    models = entry.get("models") if isinstance(entry, dict) else None
    if not isinstance(models, list) or not models or not isinstance(models[0], dict):
        return None
    name = models[0].get("name")
    if not isinstance(name, str) or not name.endswith(_ENTRY_SUFFIX) or name == _ENTRY_SUFFIX:
        return None
    return name[:-len(_ENTRY_SUFFIX)]


def installed_entry(env: Mapping[str, str]) -> dict | None:
    """providers.llmctl of Pi's models.json, None when the file or the entry is absent."""
    path = models_path(env)
    _, doc = load_json_object(path, "Pi")
    entry = (_object_value(path, doc, "providers") or {}).get(PROVIDER)
    return entry if isinstance(entry, dict) else None


def plan_models(env: Mapping[str, str], target: Target) -> FileChange | None:
    """The change that puts llmctl's provider entry for target into models.json, None when it is there already."""
    path = models_path(env)
    text, doc = load_json_object(path, "Pi")
    providers = _object_value(path, doc, "providers")
    entry = provider_entry(target)
    if providers is not None and PROVIDER in providers and providers[PROVIDER] == entry:
        return None
    new = dict(doc)
    new["providers"] = {**(providers or {}), PROVIDER: entry}
    new_text = models_text(new)
    rest = _without_llmctl_provider(doc)
    if _without_llmctl_provider(parse_json(new_text)) != rest:
        raise ClientError(f"{path}: llmctl's own check of the new models.json failed; nothing was written")
    if providers is None or PROVIDER not in providers:
        old = "absent"
    else:
        old = entry_description(providers[PROVIDER]) or "another llmctl entry"
    summary = f"provider llmctl for {target.recipe.id} on {target.label}, model {target.served} (was {old})"
    return FileChange(path, text, new_text, summary, f"would write {path}: {summary}", mode=0o600, backup=bool(rest))


def validate_models(host: Host, text: str, target: Target, path: Path) -> str | None:
    """Run pi --list-models over the candidate models.json in a private directory (spec D5, L4).

    Returns a warning when Pi lists the llmctl row but reports other problems in the file; refuses without the row.
    """
    try:
        tmp = tempfile.mkdtemp(prefix="llmctl-pi-")
    except OSError as exc:
        raise ClientError(f"cannot check {path} with pi: {exc}") from exc
    try:
        os.chmod(tmp, 0o700)
        fd = os.open(os.path.join(tmp, "models.json"), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        result = host.run(["pi", "--offline", "--list-models", PROVIDER],
                          env_extra={"PI_CODING_AGENT_DIR": tmp, "PI_OFFLINE": "1",
                                     "PI_SKIP_VERSION_CHECK": "1", "PI_TELEMETRY": "0"})
    except (OSError, UnicodeError) as exc:
        raise ClientError(f"cannot check {path} with pi: {exc}") from exc
    finally:
        shutil.rmtree(tmp)
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit {result.returncode}"
        raise ClientError(f"pi --list-models failed: {detail}; check that pi runs in this shell, or {OUT}")
    row = re.compile(rf"^{PROVIDER}\s+{re.escape(target.served)}(\s|$)")
    stderr_lines = [line.strip() for line in result.stderr.splitlines() if line.strip()]
    if not any(row.match(line) for line in result.stdout.splitlines()):
        detail = " / ".join(stderr_lines) or "no llmctl row in pi --list-models"
        raise ClientError(f"Pi rejects the llmctl entry for {target.recipe.id}: {detail}")
    if _LOAD_ERRORS in result.stderr:
        return f"warning: Pi reports problems in {path} outside the llmctl entry: {stderr_lines[0]}"
    return None


def _show(value: object) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _settings_rest(doc: dict) -> dict:
    """The settings document without the keys llmctl owns."""
    rest = {key: value for key, value in doc.items() if key not in _DEFAULT_KEYS}
    if isinstance(rest.get(_LEVELS), dict):
        levels = {key: value for key, value in rest[_LEVELS].items() if not key.startswith(_LLMCTL_PREFIX)}
        if levels:
            rest[_LEVELS] = levels
        else:
            del rest[_LEVELS]
    return rest


def _settings_summary(old: dict, new: dict) -> str:
    parts = []
    old_levels, new_levels = old.get(_LEVELS) or {}, new.get(_LEVELS) or {}
    for key, level in new_levels.items():
        if key.startswith(_LLMCTL_PREFIX) and (key not in old_levels or old_levels[key] != level):
            was = _show(old_levels[key]) if key in old_levels else "absent"
            parts.append(f"{_LEVELS} {key} {_show(level)} (was {was})")
    parts += [f"removed {_LEVELS} {key}" for key in old_levels if key not in new_levels]
    for key in _DEFAULT_KEYS:
        was = _show(old[key]) if key in old else "absent"
        if key in new and (key not in old or old[key] != new[key]):
            parts.append(f"{key} {_show(new[key])} (was {was})")
        elif key in old and key not in new:
            parts.append(f"removed {key} (was {was})")
    return "; ".join(parts)


def _settings_change(path: Path, text: str | None, old: dict, new: dict) -> FileChange | None:
    if new == old:
        return None
    new_text = json.dumps(new, indent=2, ensure_ascii=False)
    if text is None or text.endswith("\n"):
        new_text += "\n"
    if _settings_rest(parse_json(new_text)) != _settings_rest(old):
        raise ClientError(f"{path}: llmctl's own check of the new settings failed; nothing was written")
    summary = _settings_summary(old, new)
    backup = any(key not in (_LEVELS, *_DEFAULT_KEYS) for key in old)
    return FileChange(path, text, new_text, summary, f"would write {path}: {summary}", mode=0o600, backup=backup)


def _load_settings(env: Mapping[str, str]) -> tuple[Path, str | None, dict]:
    path = settings_path(env)
    text, doc = load_json_object(path, "Pi")
    _object_value(path, doc, _LEVELS)
    return path, text, doc


def plan_settings(env: Mapping[str, str], target: Target, *, set_default: bool) -> FileChange | None:
    """The change to settings.json: llmctl's thinking level for target and the defaults by spec D6."""
    path, text, doc = _load_settings(env)
    key = f"{_LLMCTL_PREFIX}{target.served}"
    levels = {name: value for name, value in (doc.get(_LEVELS) or {}).items()
              if not name.startswith(_LLMCTL_PREFIX) or name == key}
    levels[key] = default_level(target.recipe.clients)
    new = dict(doc)
    new[_LEVELS] = levels
    if set_default:
        new["defaultProvider"] = PROVIDER
        new["defaultModel"] = target.served
    elif doc.get("defaultProvider") == PROVIDER:
        new["defaultModel"] = target.served
    return _settings_change(path, text, doc, new)


def plan_defaults(env: Mapping[str, str], provider: str | None, model: str | None) -> FileChange | None:
    """Set Pi's defaultProvider and defaultModel, removing a key whose value is None (the e2e restore)."""
    path, text, doc = _load_settings(env)
    new = dict(doc)
    for key, value in zip(_DEFAULT_KEYS, (provider, model)):
        if value is None:
            new.pop(key, None)
        else:
            new[key] = value
    return _settings_change(path, text, doc, new)


@contextmanager
def settings_lock(host: Host, path: Path) -> Iterator[None]:
    """Hold Pi's settings lock directory (<real settings.json>.lock) while llmctl writes settings.json."""
    lock = Path(os.path.realpath(path) + ".lock")
    deadline = host.monotonic() + LOCK_WAIT_S
    while True:
        try:
            os.mkdir(lock, 0o700)
            break
        except FileExistsError:
            if host.monotonic() >= deadline:
                raise ClientWriteError(f"{path}: Pi holds its settings lock {lock}; if no Pi is running, "
                                       f"remove {lock}, then run ./llmctl up again") from None
            host.sleep(LOCK_STEP_S)
    inode = os.lstat(lock).st_ino
    try:
        yield
    finally:
        # Pi treats an old lock as stale and replaces it, so llmctl removes only the lock directory it created.
        try:
            own = os.lstat(lock).st_ino == inode
        except FileNotFoundError:
            own = False
        if own:
            os.rmdir(lock)
    if not own:
        raise ClientWriteError(f"{path}: Pi took over its settings lock while llmctl held it, so another writer may "
                               "have changed settings.json; check it")


def pi_default(env: Mapping[str, str]) -> tuple[str | None, str | None]:
    """Pi's defaultProvider and defaultModel; None for a key that is absent or not a string."""
    _, _, doc = _load_settings(env)
    provider, model = (doc.get(key) for key in _DEFAULT_KEYS)
    return (provider if isinstance(provider, str) else None, model if isinstance(model, str) else None)


def stored_level(env: Mapping[str, str], served: str) -> str | None:
    _, _, doc = _load_settings(env)
    level = (doc.get(_LEVELS) or {}).get(f"{_LLMCTL_PREFIX}{served}")
    return level if isinstance(level, str) else None


_GLOB = ("*", "?", "[")
_SUFFIX_LEVELS = (*PI_LEVELS, "xhigh")


def _enabled_matches(entry: object, served: str) -> bool:
    """True when one enabledModels entry selects llmctl/<served> by Pi 1.5.2's rules (model-resolver.js:250-290)."""
    if not isinstance(entry, str):
        return False
    text, names = entry.lower(), (f"{_LLMCTL_PREFIX}{served}".lower(), served.lower())
    base, _, suffix = text.rpartition(":")
    if any(char in text for char in _GLOB):
        pattern = base if base and suffix in _SUFFIX_LEVELS else text
        return any(fnmatch.fnmatchcase(name, pattern) for name in names)
    return text in names or (bool(base) and base in names)


def enabled_models_skip(settings: dict, served: str) -> str | None:
    """The first enabledModels entry when no entry selects llmctl/<served>, so a new Pi session without --model
    starts with that entry (dist/main.js:435-455); None when enabledModels is empty or an entry matches."""
    enabled = settings.get("enabledModels")
    if not isinstance(enabled, list) or not enabled:
        return None
    if any(_enabled_matches(entry, served) for entry in enabled):
        return None
    return _show(enabled[0])


def enabled_models_note(env: Mapping[str, str], target: Target) -> str | None:
    """The note when Pi's enabledModels makes a new session start with another model than its llmctl default."""
    _, _, doc = _load_settings(env)
    if doc.get("defaultProvider") != PROVIDER:
        return None
    first = enabled_models_skip(doc, target.served)
    if first is None:
        return None
    own = f"{_LLMCTL_PREFIX}{target.served}"
    return (f"note: a new Pi session without --model starts with the first of enabledModels ({first}) instead of "
            f"{own}; add {own} to enabledModels or clear enabledModels")
