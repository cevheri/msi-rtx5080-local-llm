"""Pointing OpenCode's default model at the model llmctl started, when asked (up --set-opencode-default).

Only the top-level "model" line of the global config changes; the result is parsed again and must equal the
old config with that one value changed, or nothing is written.
"""

import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path

from launcher.errors import UsageError


@dataclass(frozen=True)
class Plan:
    path: Path
    model: str
    previous: str | None
    new_text: str | None  # None when the model is already the default


def config_path(env) -> Path:
    base = Path(env["XDG_CONFIG_HOME"]) if env.get("XDG_CONFIG_HOME") else Path(env.get("HOME", "~")).expanduser() / ".config"
    for name in ("opencode.json", "opencode.jsonc"):
        if (base / "opencode" / name).is_file():
            return base / "opencode" / name
    return base / "opencode" / "opencode.json"


def current_default(env) -> tuple[Path, str | None, str | None]:
    """OpenCode's configured default model: (config path, model or None, problem or None)."""
    path = config_path(env)
    if not path.is_file():
        return path, None, f"no OpenCode config at {path}"
    try:
        return path, json.loads(path.read_text()).get("model"), None
    except ValueError:
        return path, None, f"{path} is not plain JSON"


def plan_default(env, model: str) -> Plan:
    """Check that OpenCode's default can be set to `model`, without writing anything."""
    path = config_path(env)
    if not path.is_file():
        raise UsageError(f"no OpenCode config at {path}; install the provider config from docs/opencode.md first")
    text = path.read_text()
    try:
        data = json.loads(text)
    except ValueError:
        raise UsageError(f"{path} is not plain JSON (it may have comments); set \"model\": \"{model}\" in it yourself") from None
    provider, _, entry = model.partition("/")
    if entry not in data.get("providers", {}).get(provider, {}).get("models", {}):
        raise UsageError(f"{path} has no entry {model}; add the provider config from docs/opencode.md")
    previous = data.get("model")
    if previous == model:
        return Plan(path, model, previous, None)
    if "model" in data:
        new_text, count = re.subn(r'^(\s*"model"\s*:\s*)"[^"\n]*"', lambda m: m.group(1) + json.dumps(model),
                                  text, count=1, flags=re.M)
    else:
        new_text, count = re.subn(r"^\{[ \t]*\n", lambda m: m.group(0) + f'  "model": {json.dumps(model)},\n', text, count=1)
    expected = dict(data, model=model)
    try:
        safe = count == 1 and json.loads(new_text) == expected
    except ValueError:
        safe = False
    if not safe:
        raise UsageError(f"could not change the top-level \"model\" in {path} safely; set it to \"{model}\" yourself")
    return Plan(path, model, previous, new_text)


def apply_default(plan: Plan) -> None:
    if plan.new_text is None:
        return
    mode = stat.S_IMODE(plan.path.stat().st_mode)
    temporary = plan.path.with_name(plan.path.name + ".llmctl-tmp")
    temporary.write_text(plan.new_text)
    os.chmod(temporary, mode)
    os.replace(temporary, plan.path)
