"""OpenCode's default model line, the notes about its entry, and the expected entry of a recipe.

llmctl never edits OpenCode's provider catalog; it changes only the top-level "model" value of the global config,
and only when that config holds the recipe's entry (spec D6, L1): on every up it follows a default that already
names a model of one of llmctl's providers, and with --set-default (or LLMCTL_SET_DEFAULT=1) it sets the default
whatever it named before.
The result is parsed again and must equal the old config with that one value changed, or nothing is written.
Every path comes from the environment mapping a caller passes in.
"""

import json
from collections.abc import Collection, Mapping
from pathlib import Path

from launcher.errors import ClientError
from launcher.files import FileChange, env_path, home_dir, load_json_object, parse_json
from launcher.model import Recipe, Target

_MISSING = object()
_WHITESPACE = " \t\n\r"


def expected_entry(recipe: Recipe, served: str) -> dict:
    """The OpenCode model entry, without its display name, that the recipe's [clients] table implies."""
    meta = recipe.clients
    reasoning = {"supported": meta.thinking != "none"}
    if meta.thinking == "effort":
        reasoning["effort"] = list(meta.efforts)
    entry = {
        "modelID": served,
        "capabilities": {"tools": not recipe.chat_only, "input": ["text"], "output": ["text"], "reasoning": reasoning},
        "limit": {"context": recipe.context, "output": meta.output},
    }
    if meta.thinking == "toggle":
        variants = [{"id": "thinking", "body": {"chat_template_kwargs": {"enable_thinking": True}}}]
        if meta.medium_kwargs:
            variants.append({"id": "medium",
                             "body": {"chat_template_kwargs": {"enable_thinking": True, **dict(meta.medium_kwargs)}}})
        variants.append({"id": "none", "body": {"chat_template_kwargs": {"enable_thinking": False}}})
        entry["variants"] = variants
    return entry


def config_path(env: Mapping[str, str]) -> Path:
    xdg = env.get("XDG_CONFIG_HOME", "")
    base = env_path(env, xdg) if xdg else home_dir(env) / ".config"
    directory = base / "opencode"
    if not (directory / "opencode.json").exists() and (directory / "opencode.jsonc").is_file():
        return directory / "opencode.jsonc"
    return directory / "opencode.json"


def config_root(env: Mapping[str, str]) -> Path:
    return config_path(env).parent


def points_at_llmctl(model: object, providers: Collection[str]) -> bool:
    """True when the model names a model of one of llmctl's OpenCode providers."""
    if not isinstance(model, str) or "/" not in model:
        return False
    return model.partition("/")[0] in providers


def _skip_ws(text: str, i: int) -> int:
    while i < len(text) and text[i] in _WHITESPACE:
        i += 1
    return i


def _string_end(text: str, i: int) -> int:
    i += 1
    while text[i] != '"':
        i += 2 if text[i] == "\\" else 1
    return i + 1


def _value_end(text: str, i: int) -> int:
    if text[i] == '"':
        return _string_end(text, i)
    if text[i] in "{[":
        depth = 0
        while True:
            char = text[i]
            if char == '"':
                i = _string_end(text, i)
                continue
            if char in "{[":
                depth += 1
            elif char in "}]":
                depth -= 1
                if depth == 0:
                    return i + 1
            i += 1
    while i < len(text) and text[i] not in ",}]" + _WHITESPACE:
        i += 1
    return i


def set_top_level_model(text: str, model: str) -> str:
    """Replace the value of the top-level "model" member of a JSON object text, or insert it as the first member."""
    value = json.dumps(model)
    start = _skip_ws(text, 0)
    if start >= len(text) or text[start] != "{":
        raise ValueError("the text is not a JSON object")
    first = i = _skip_ws(text, start + 1)
    if text[i] == "}":
        return text[:start + 1] + f'"model": {value}' + text[start + 1:]
    while True:
        key_end = _string_end(text, i)
        key = json.loads(text[i:key_end])
        i = _skip_ws(text, key_end)
        if text[i] != ":":
            raise ValueError(f"expected ':' at offset {i}")
        value_start = _skip_ws(text, i + 1)
        value_end = _value_end(text, value_start)
        if key == "model":
            return text[:value_start] + value + text[value_end:]
        i = _skip_ws(text, value_end)
        if text[i] == "}":
            break
        if text[i] != ",":
            raise ValueError(f"expected ',' or '}}' at offset {i}")
        i = _skip_ws(text, i + 1)
    gap = text[start + 1:first]
    if "\n" not in gap:
        return text[:start + 1] + f'"model": {value}, ' + text[start + 1:]
    line_start = start + 1 + gap.rindex("\n") + 1
    newline = "\r\n" if text[line_start - 2:line_start] == "\r\n" else "\n"
    indent = text[line_start:first]
    return text[:line_start] + f'{indent}"model": {value},{newline}' + text[line_start:]


def _strict_equal(a: object, b: object) -> bool:
    """JSON equality: true and 1 differ, as do a list and a tuple of the same values."""
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_strict_equal(a[key], b[key]) for key in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_strict_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, (dict, list, tuple)) or isinstance(b, (dict, list, tuple)):
        return False
    return a == b


def _show(value: object) -> str:
    return "absent" if value is _MISSING else json.dumps(value, separators=(", ", ": "))


def entry_difference(installed: dict, expected: dict, _prefix: str = "") -> str | None:
    """The first path where the installed entry differs from the expected one; keys only installed are ignored."""
    for key, want in expected.items():
        have = installed.get(key, _MISSING) if isinstance(installed, dict) else _MISSING
        path = f"{_prefix}{key}"
        if isinstance(want, dict) and isinstance(have, dict):
            found = entry_difference(have, want, path + ".")
            if found is not None:
                return found
        elif have is _MISSING or not _strict_equal(have, want):
            return f"{path} {_show(have)} instead of {_show(want)}"
    return None


def _installed_entry(data: dict, provider: str, recipe_id: str) -> object:
    providers = data.get("providers")
    entry = providers.get(provider) if isinstance(providers, dict) else None
    models = entry.get("models") if isinstance(entry, dict) else None
    return models.get(recipe_id, _MISSING) if isinstance(models, dict) else _MISSING


def plan_set_model(env: Mapping[str, str], model: str) -> FileChange | None:
    """The change that sets OpenCode's top-level "model" to `model`, or None when it is set already."""
    path = config_path(env)
    text, data = load_json_object(path, "OpenCode")
    if text is None:
        raise ClientError(f"no OpenCode config at {path}")
    old = data.get("model")
    if old == model:
        return None
    refusal = ClientError(f'could not change the top-level "model" in {path} safely; set it to "{model}" yourself')
    try:
        new_text = set_top_level_model(text, model)
        safe = parse_json(new_text) == dict(data, model=model)
    except ValueError:
        raise refusal from None
    if not safe:
        raise refusal
    return FileChange(path=path, old_text=text, new_text=new_text,
                      summary=f"model {model} (was {old or 'unset'})",
                      preview=f"would set OpenCode's default model: {old or 'unset'} -> {model} ({path})",
                      backup=True)


def plan_model_line(env: Mapping[str, str], target: Target, providers: Collection[str], *,
                    set_default: bool) -> FileChange | None:
    """The model-line change of D6 rules 1 to 3: follow an llmctl default, or set it when asked."""
    if not target.recipe.opencode:
        return None
    try:
        text, data = load_json_object(config_path(env), "OpenCode")
    except ClientError:
        if set_default:
            raise
        return None
    if text is None:
        return None
    if _installed_entry(data, target.runtime.opencode_provider, target.recipe.id) is _MISSING:
        return None
    if not set_default and not points_at_llmctl(data.get("model"), providers):
        return None
    return plan_set_model(env, f"{target.runtime.opencode_provider}/{target.recipe.id}")


def _inspect(path: Path) -> tuple[dict | None, str | None]:
    """(data, problem note): data None with no note means the file is missing."""
    leave = "so llmctl leaves OpenCode's default as it is"
    try:
        with open(path, encoding="utf-8", newline="") as handle:
            text = handle.read()
    except FileNotFoundError:
        return None, None
    except (OSError, UnicodeDecodeError) as exc:
        return None, f"note: cannot read {path} ({exc}), {leave}"
    try:
        data = parse_json(text)
    except ValueError as exc:
        return None, f"note: {path} is not plain JSON ({exc}), {leave}"
    if not isinstance(data, dict):
        return None, f"note: {path} is not a JSON object, {leave}"
    return data, None


def has_entry(env: Mapping[str, str], target: Target) -> bool:
    """True when the user's OpenCode config is readable JSON and holds the target's provider entry."""
    data, _ = _inspect(config_path(env))
    return data is not None and _installed_entry(data, target.runtime.opencode_provider,
                                                 target.recipe.id) is not _MISSING


def opencode_notes(env: Mapping[str, str], target: Target) -> list[str]:
    """The lines up prints about OpenCode for this target, in order, only those that apply."""
    path = config_path(env)
    recipe = target.recipe
    data, problem = _inspect(path)
    if not recipe.opencode:
        if problem is not None:
            current = "unknown"
        else:
            model = (data or {}).get("model")
            current = "unset" if model is None else model if isinstance(model, str) else json.dumps(model)
        return [f"OpenCode: {recipe.id} has no OpenCode entry in docs/opencode.md; its default stays {current}"]
    if problem is not None:
        return [problem]
    model = f"{target.runtime.opencode_provider}/{recipe.id}"
    installed = _MISSING if data is None else _installed_entry(data, target.runtime.opencode_provider, recipe.id)
    if installed is _MISSING:
        return [f"OpenCode: no entry for {model} in {path}; add the provider config from docs/opencode.md"]
    difference = entry_difference(installed, expected_entry(recipe, target.served))
    if difference is None:
        return []
    return [f"warning: OpenCode's entry {model} in {path} differs from the recipe: {difference}; "
            "update it from docs/opencode.md"]


def current_default(env) -> tuple[Path, str | None, str | None]:
    """OpenCode's configured default model: (config path, model or None, problem or None)."""
    path = config_path(env)
    if not path.is_file():
        return path, None, f"no OpenCode config at {path}"
    try:
        return path, json.loads(path.read_text()).get("model"), None
    except ValueError:
        return path, None, f"{path} is not plain JSON"
