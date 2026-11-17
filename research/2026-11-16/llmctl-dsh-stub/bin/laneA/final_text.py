#!/usr/bin/env python3
"""Render the home patch llmctl would write (recommended design) for one target; stdlib only."""
import json, sys

HEADER = [
    "# dsh rows for the model llmctl runs (docs/llmctl.md#clients).",
    "# llmctl rewrites the lines between the two markers on every ./llmctl up and keeps the rest of this file.",
    "# The agent-default-model and acp rows stay only while they are in the block: delete both lines to give",
    "# dsh's model picker its default back; ./llmctl up <model> --set-default writes them again.",
]
BEGIN = "# >>> llmctl managed block: rewritten by llmctl on every start; edits here are lost >>>"
END = "# <<< llmctl managed block <<<"

def flow(obj):
    return "- " + json.dumps(obj, ensure_ascii=True, separators=(", ", ": "))

TARGETS = {
    "gemma-4-26b-a4b": {
        "label": "llama.cpp", "port": 8080, "served": "local", "context": 131072, "output": 16384,
        "efforts": {"off": None, "medium": "medium"},
        "compat": {"thinkingFormat": "chat-template", "chatTemplateKwargs": {"enable_thinking": {"$var": "thinking.enabled"}}},
        "level": "off",
    },
    "gpt-oss-20b": {
        "label": "vLLM", "port": 8000, "served": "gpt-oss-20b", "context": 65536, "output": 16384,
        "efforts": {"low": "low", "medium": "medium", "high": "high"},
        "compat": {"thinkingFormat": "openai", "supportsReasoningEffort": True},
        "level": "medium",
    },
}

def block(recipe, with_default=True):
    t = TARGETS[recipe]
    model = {"id": t["served"], "name": f"{recipe} on {t['label']} (llmctl)", "contextWindow": t["context"],
             "maxTokens": t["output"], "input": ["text"], "reasoningEfforts": t["efforts"], "compat": t["compat"]}
    provider = {"displayName": "llmctl", "api": "openai-completions", "baseURL": f"http://127.0.0.1:{t['port']}/v1",
                "headers": {"Authorization": "Bearer no-key-required"},
                "compat": {"supportsStore": False, "supportsDeveloperRole": False, "maxTokensField": "max_tokens"},
                "models": [model]}
    rows = [flow({"id": "llm-pi-ai", "config": {"providers": {"llmctl": provider}}})]
    if with_default:
        rows.append(flow({"id": "agent-default-model", "config": {"provider": "llmctl", "model": t["served"], "reasoningEffort": t["level"]}}))
        rows.append(flow({"id": "acp", "config": {"provider": "llmctl", "model": t["served"]}}))
    return rows

if __name__ == "__main__":
    recipe = sys.argv[1]
    print("\n".join(HEADER + [BEGIN] + block(recipe) + [END]))
