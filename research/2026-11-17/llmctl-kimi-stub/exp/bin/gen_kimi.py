"""Throwaway prototype: the two Kimi Code tables llmctl would own, for every shipped recipe.

Report sections 5.1 (tables), 5.3 (thinking keys) and 5.5 (caps) of
docs/superpowers/research/2026-11-17-kimi-code/research-report.md.
Imports the repository's launcher package read-only (run with python3 -I -B).

usage: python3 -I -B gen_kimi.py <repo> <out dir>
Writes <out>/<runtime>/<id>.toml (the two tables) and <out>/index.json (one row per entry).
"""

import json
import sys
from pathlib import Path

REPO = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(REPO))

from launcher.catalog import load_catalog  # noqa: E402
from launcher.model import Target  # noqa: E402
from launcher.runtimes import ADAPTERS  # noqa: E402

PROVIDER = "llmctl"
API_KEY = "no-key-required"


def q(value) -> str:
    """A TOML value written as JSON: strings, integers and arrays of strings are valid TOML this way."""
    return json.dumps(value, ensure_ascii=True)


def model_fields(target: Target) -> dict:
    recipe, meta = target.recipe, target.recipe.clients
    caps = [] if recipe.chat_only else ["tool_use"]
    fields = {
        "provider": PROVIDER,
        "model": target.served,
        "display_name": f"{recipe.id} on {target.label} (llmctl)",
        "max_context_size": recipe.context,
        "max_output_size": meta.output,
    }
    if meta.thinking == "effort":
        caps.append("thinking")
        fields["capabilities"] = caps
        fields["support_efforts"] = [e for e in meta.efforts if e != "none"]
        fields["default_effort"] = "medium"
        if "none" in meta.efforts:
            fields["off_effort"] = "none"
    else:
        fields["capabilities"] = caps
    return fields


def provider_fields(target: Target) -> dict:
    return {
        "type": "openai",
        "base_url": f"http://127.0.0.1:{target.runtime.host_port}/v1",
        "api_key": API_KEY,
        "model_source": "static",
    }


def alias(target: Target) -> str:
    return f"{PROVIDER}/{target.served}"


def tables_text(target: Target) -> str:
    lines = [f"[providers.{PROVIDER}]"]
    lines += [f"{k} = {q(v)}" for k, v in provider_fields(target).items()]
    lines += ["", f"[models.{q(alias(target))}]"]
    lines += [f"{k} = {q(v)}" for k, v in model_fields(target).items()]
    return "\n".join(lines) + "\n"


def main() -> None:
    catalog = load_catalog(REPO / "recipes", None)
    index = []
    for runtime, entry in sorted(catalog.runtimes.items()):
        if entry.problems:
            raise SystemExit(f"{runtime}: recipe problems: {[p.message for p in entry.problems]}")
        adapter = ADAPTERS[runtime]
        for recipe_id in sorted(entry.recipes):
            recipe = entry.recipes[recipe_id]
            target = Target(recipe, entry.runtime, adapter.served_name(recipe, entry.runtime), adapter.LABEL)
            text = tables_text(target)
            path = OUT / runtime / f"{recipe_id}.toml"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            meta = recipe.clients
            index.append({
                "runtime": runtime, "label": target.label, "id": recipe_id, "served": target.served,
                "alias": alias(target), "port": target.runtime.host_port, "chat_only": recipe.chat_only,
                "thinking": meta.thinking, "default": meta.default, "efforts": list(meta.efforts),
                "context": recipe.context, "output": meta.output,
                "provider": provider_fields(target), "model": model_fields(target),
                "file": f"{runtime}/{recipe_id}.toml",
            })
    (OUT / "index.json").write_text(json.dumps(index, indent=1) + "\n", encoding="utf-8")
    kinds = {}
    for row in index:
        kinds[row["thinking"]] = kinds.get(row["thinking"], 0) + 1
    print(f"{len(index)} entries: {kinds}")


if __name__ == "__main__":
    main()
