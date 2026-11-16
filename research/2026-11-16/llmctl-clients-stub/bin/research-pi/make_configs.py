#!/usr/bin/env python3
"""Write one models.json per scenario under configs/<name>/ (port placeholder __PORT__)."""
import copy
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_PROVIDER = {
    "name": "llmctl (local server)",
    "baseUrl": "http://127.0.0.1:__PORT__/v1",
    "api": "openai-completions",
    "apiKey": "local",
    "compat": {"supportsStore": False, "supportsDeveloperRole": False, "maxTokensField": "max_tokens"},
}
CHAT_TEMPLATE = {"thinkingFormat": "chat-template", "chatTemplateKwargs": {"enable_thinking": {"$var": "thinking.enabled"}}}

MODELS = {
    # toggle model (Qwen3.6, Gemma 4, Nemotron Lightning, Groot): enable_thinking true/false
    "toggle": {
        "id": "local", "name": "Qwen3.6-35B-A3B Q4_K_M (llama.cpp, llmctl)", "reasoning": True,
        "thinkingLevelMap": {"minimal": None, "low": None, "high": None},
        "input": ["text"], "contextWindow": 131072, "maxTokens": 16384,
        "compat": copy.deepcopy(CHAT_TEMPLATE),
    },
    # Nemotron 3.5 Bolt: toggle plus medium_effort at the medium level
    "bolt": {
        "id": "local", "name": "Nemotron 3.5 Bolt (llama.cpp, llmctl)", "reasoning": True,
        "thinkingLevelMap": {"minimal": None, "low": None},
        "input": ["text"], "contextWindow": 131072, "maxTokens": 16384,
        "compat": copy.deepcopy(CHAT_TEMPLATE),
        "samplingParamsByThinkingLevel": {
            "medium": {"chat_template_kwargs": {"enable_thinking": True, "medium_effort": True}}
        },
    },
    # Qwen3.8-27B: reasoning_effort none/low/medium
    "effort-none-low-medium": {
        "id": "local", "name": "Qwen3.8-27B UD-Q3_K_XL (llama.cpp, llmctl)", "reasoning": True,
        "thinkingLevelMap": {"off": "none", "minimal": None, "low": "low", "medium": "medium", "high": None},
        "input": ["text"], "contextWindow": 65536, "maxTokens": 16384,
        "compat": {"thinkingFormat": "openai", "supportsReasoningEffort": True},
    },
    # K2 Horizon MoVA: always thinks, server sets effort high, client must send nothing
    "always-on": {
        "id": "local", "name": "K2 Horizon MoVA 36B A4B (llama.cpp, llmctl)", "reasoning": True,
        "thinkingLevelMap": {"off": None, "minimal": None, "low": None, "medium": None},
        "input": ["text"], "contextWindow": 65536, "maxTokens": 16384,
        "compat": {"supportsReasoningEffort": False},
    },
    # K2 Horizon 7B variant of always-on with the 81920/32768 limits
    "always-on-7b": {
        "id": "local", "name": "K2 Horizon 7B Q8_0 (llama.cpp, llmctl)", "reasoning": True,
        "thinkingLevelMap": {"off": None, "minimal": None, "low": None, "medium": None},
        "input": ["text"], "contextWindow": 81920, "maxTokens": 32768,
        "compat": {"supportsReasoningEffort": False},
    },
    # gpt-oss-20b on vLLM: reasoning_effort low/medium/high, served name gpt-oss-20b
    "gpt-oss-vllm": {
        "id": "gpt-oss-20b", "name": "gpt-oss-20b (vLLM, llmctl)", "reasoning": True,
        "thinkingLevelMap": {"off": None, "minimal": None},
        "input": ["text"], "contextWindow": 65536, "maxTokens": 16384,
        "compat": {"thinkingFormat": "openai", "supportsReasoningEffort": True},
    },
    # Groot on vLLM: toggle model behind vLLM (reasoning field "reasoning")
    "toggle-vllm": {
        "id": "groot-19b-a3b", "name": "Groot-19B-A3B NVFP4 (vLLM, llmctl)", "reasoning": True,
        "thinkingLevelMap": {"minimal": None, "low": None, "high": None},
        "input": ["text"], "contextWindow": 131072, "maxTokens": 16384,
        "compat": copy.deepcopy(CHAT_TEMPLATE),
    },
    # Qwen3-Coder-Next: no thinking
    "no-thinking": {
        "id": "local", "name": "Qwen3-Coder-Next UD-IQ4_XS (llama.cpp, llmctl)", "reasoning": False,
        "input": ["text"], "contextWindow": 131072, "maxTokens": 16384,
    },
    # DeepSeek-R1-0528-Qwen3-8B: chat only (tools off is a CLI choice), always thinks
    "chat-only": {
        "id": "local", "name": "DeepSeek-R1-0528-Qwen3-8B Q8_0 (llama.cpp, llmctl, chat only)", "reasoning": True,
        "thinkingLevelMap": {"off": None, "minimal": None, "low": None, "medium": None},
        "input": ["text"], "contextWindow": 65536, "maxTokens": 16384,
        "compat": {"supportsReasoningEffort": False},
    },
}


def write(name, providers):
    directory = os.path.join(ROOT, "configs", name)
    os.makedirs(directory, exist_ok=True)
    doc = {"$schema": "https://pi.dev/schemas/models.schema.json", "providers": providers}
    with open(os.path.join(directory, "models.json"), "w") as fh:
        json.dump(doc, fh, indent=2)
        fh.write("\n")


for key, model in MODELS.items():
    provider = copy.deepcopy(BASE_PROVIDER)
    provider["models"] = [model]
    write(key, {"llmctl": provider})

# Baseline: no compat at all, reasoning model, to show Pi's auto-detected defaults for 127.0.0.1
baseline = {"name": "llmctl (no compat)", "baseUrl": "http://127.0.0.1:__PORT__/v1", "api": "openai-completions",
            "apiKey": "local", "models": [{"id": "local", "reasoning": True, "contextWindow": 131072, "maxTokens": 16384}]}
write("baseline-nocompat", {"llmctl": baseline})

# No apiKey at all
nokey = copy.deepcopy(BASE_PROVIDER)
del nokey["apiKey"]
nokey["models"] = [copy.deepcopy(MODELS["toggle"])]
write("no-apikey", {"llmctl": nokey})
print("written:", sorted(os.listdir(os.path.join(ROOT, "configs"))))
