#!/usr/bin/env python3
"""Reference generator for the dsh rows of the design (stdlib only; reads the shipped recipes through launcher/).

gen_dsh.py <out_dir>            writes <out_dir>/<runtime>/<id>.dsh.yml (block lines with the default rows) and
                                <out_dir>/<runtime>/<id>.dsh-nodefault.yml (the llm-pi-ai line only)
gen_dsh.py --home <runtime> <id> [--no-default] [--port N]
                                prints a complete new home patch (header, markers, block) for one recipe
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]  # the repository root, from research/2026-11-16/llmctl-dsh-stub/bin/spec
sys.path.insert(0, str(REPO))

from launcher.catalog import load_catalog  # noqa: E402
from launcher.ops import client_target  # noqa: E402

PROVIDER = "llmctl"
STREAM_IDLE_TIMEOUT_MS = 900000  # estimate (lane B, B3-4 and B3-8)
RETRYABLE_CODES = ["EMPTY_RESPONSE", "RATE_LIMIT", "SERVER", "TRANSPORT"]  # dsh's default list without TIMEOUT
HEADER = (
    "# dsh rows for the model llmctl runs (docs/llmctl.md#clients).\n"
    "# llmctl rewrites the lines between the two markers on every ./llmctl up and keeps everything else in this file.\n"
    "# While the block holds the agent-default-model line, the running model is dsh's default in every profile;\n"
    "# delete that line and the acp line to give the default back to dsh's model picker.\n"
)
BEGIN = "# >>> llmctl managed block: rewritten by llmctl on every start; edits here are lost >>>"
END = "# <<< llmctl managed block <<<"
NO_TOOLS_ROWS = ("tool-bash", "tool-goal", "tool-fs", "tool-fs-search", "tool-jobs", "tool-subagent-control",
                 "tool-skill", "tool-subagent", "tool-subagent-fork", "tool-todo", "tool-web", "tool-workflow",
                 "tool-working-directory", "tool-agent-team")


def thinking(meta):
    """(model fields, route fields) for one [clients] table."""
    if meta.thinking == "toggle":
        efforts = {"off": None, "medium": "medium"}
        kwargs = {"enable_thinking": {"$var": "thinking.enabled"}}
        route = {"reasoning": "medium" if meta.default == "on" else "off"}
        if meta.medium_kwargs:
            for key, value in meta.medium_kwargs:
                if value is not True:
                    raise SystemExit(f"medium_kwargs {key} = {value!r}: dsh can send only true")
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
        return ({"reasoningEfforts": False}, {})
    raise SystemExit(f"unknown thinking type {meta.thinking!r}")


def route(target, port=None):
    recipe, meta = target.recipe, target.recipe.clients
    model_fields, route_fields = thinking(meta)
    model = {"id": target.served, "name": f"{recipe.id} on {target.label} (llmctl)",
             "contextWindow": recipe.context, "maxTokens": meta.output, "input": ["text"], **model_fields}
    value = {
        "displayName": PROVIDER,
        "api": "openai-completions",
        "baseURL": f"http://127.0.0.1:{port or target.runtime.host_port}/v1",
        "headers": {"Authorization": "Bearer no-key-required"},
        "compat": {"supportsStore": False, "supportsDeveloperRole": False, "maxTokensField": "max_tokens"},
        **route_fields,
        "streamIdleTimeoutMs": STREAM_IDLE_TIMEOUT_MS,
        "retryPolicy": {"mode": "normal", "maxRetries": 5, "retryableCodes": RETRYABLE_CODES},
        "models": [model],
    }
    return value


def rows(target, *, default, port=None):
    found = [{"id": "llm-pi-ai", "config": {"providers": {PROVIDER: route(target, port)}}}]
    if default:
        found.append({"id": "agent-default-model", "config": {"provider": PROVIDER, "model": target.served}})
        found.append({"id": "acp", "config": {"provider": PROVIDER, "model": target.served}})
    return found


def line(row):
    return "- " + json.dumps(row, ensure_ascii=True, separators=(", ", ": ")) + "\n"


def block(target, *, default, port=None):
    return "".join(line(row) for row in rows(target, default=default, port=port))


def home_text(target, *, default, port=None):
    return HEADER + BEGIN + "\n" + block(target, default=default, port=port) + END + "\n"


def no_tools_patch():
    return "".join(line({"id": row, "disabled": True}) for row in NO_TOOLS_ROWS)


def targets():
    catalog = load_catalog(REPO / "recipes", Path("/nonexistent-llmctl-overrides"))
    for runtime, entry in catalog.runtimes.items():
        for recipe_id in sorted(entry.recipes):
            yield runtime, recipe_id, client_target(entry.recipes[recipe_id], entry.runtime)


def main(argv):
    if argv and argv[0] == "--home":
        runtime, recipe_id = argv[1], argv[2]
        default = "--no-default" not in argv
        port = int(argv[argv.index("--port") + 1]) if "--port" in argv else None
        for found_runtime, found_id, target in targets():
            if (found_runtime, found_id) == (runtime, recipe_id):
                sys.stdout.write(home_text(target, default=default, port=port))
                return 0
        raise SystemExit(f"no recipe {recipe_id} on {runtime}")
    if argv and argv[0] == "--no-tools":
        sys.stdout.write(no_tools_patch())
        return 0
    out = Path(argv[0])
    count = 0
    for runtime, recipe_id, target in targets():
        folder = out / runtime
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{recipe_id}.dsh.yml").write_text(block(target, default=True), encoding="utf-8")
        (folder / f"{recipe_id}.dsh-nodefault.yml").write_text(block(target, default=False), encoding="utf-8")
        count += 1
    print(f"{count} recipes")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
