#!/usr/bin/env python3
"""Print llmctl dsh rows as JSON flow lines (no YAML library), for lane A experiments."""
import json, sys

def provider(runtime):
    if runtime == "llamacpp-gemma":
        return {
            "displayName": "llmctl",
            "api": "openai-completions",
            "baseURL": "http://127.0.0.1:8080/v1",
            "headers": {"Authorization": "Bearer no-key-required"},
            "compat": {"supportsStore": False, "supportsDeveloperRole": False, "maxTokensField": "max_tokens"},
            "models": [{
                "id": "local",
                "name": "gemma-4-26b-a4b on llama.cpp (llmctl)",
                "contextWindow": 131072,
                "maxTokens": 16384,
                "input": ["text"],
                "reasoningEfforts": {"off": None, "medium": "medium"},
                "compat": {"thinkingFormat": "chat-template", "chatTemplateKwargs": {"enable_thinking": {"$var": "thinking.enabled"}}},
            }],
        }
    if runtime == "vllm-gptoss":
        return {
            "displayName": "llmctl",
            "api": "openai-completions",
            "baseURL": "http://127.0.0.1:8000/v1",
            "headers": {"Authorization": "Bearer no-key-required"},
            "compat": {"supportsStore": False, "supportsDeveloperRole": False, "maxTokensField": "max_tokens"},
            "models": [{
                "id": "gpt-oss-20b",
                "name": "gpt-oss-20b on vLLM (llmctl)",
                "contextWindow": 65536,
                "maxTokens": 16384,
                "input": ["text"],
                "reasoningEfforts": {"low": "low", "medium": "medium", "high": "high"},
                "compat": {"thinkingFormat": "openai", "supportsReasoningEffort": True},
            }],
        }
    if runtime == "gptoss-on-8080":
        p = provider("vllm-gptoss")
        p["baseURL"] = "http://127.0.0.1:8080/v1"
        return p
    raise SystemExit("unknown runtime " + runtime)

def line(obj):
    return "- " + json.dumps(obj, ensure_ascii=True, separators=(", ", ": "))

if __name__ == "__main__":
    kind, runtime = sys.argv[1], sys.argv[2]
    p = provider(runtime)
    served = p["models"][0]["id"]
    if kind == "row":            # base llm-pi-ai row, llmctl provider only
        print(line({"id": "llm-pi-ai", "config": {"providers": {"llmctl": p}}}))
    elif kind == "insert":       # second pi-ai instance with its own id
        print(line({"insert": [{"id": "llmctl", "name": "@deepseek-ai/dsh-llm-pi-ai", "config": {"providers": {"llmctl": p}}}]}))
    elif kind == "default":
        eff = sys.argv[3] if len(sys.argv) > 3 else None
        cfg = {"provider": "llmctl", "model": served}
        if eff: cfg["reasoningEffort"] = eff
        print(line({"id": "agent-default-model", "config": cfg}))
    elif kind == "acp":
        print(line({"id": "acp", "config": {"provider": "llmctl", "model": served}}))
    else:
        raise SystemExit("unknown kind " + kind)
