#!/usr/bin/env python3
"""Write a dsh --patch overlay (JSON, which is valid YAML 1.2) for one candidate llmctl entry.

gen_patch.py <out.yml> <type> [key=value ...]
types: toggle, bolt (off/medium/high via thinking.budget), bolt2 (off/medium via omitWhenOff),
       effort-gptoss, effort-qwen38, always (high only, supportsReasoningEffort false),
       always-false (reasoningEfforts false), always-off (off and high), none
keys: level=<agent-default reasoningEffort, 'omit' to leave it out>  route_reasoning=<level>
      port=18080 model=local context=131072 max_tokens=16384 idle_ms=<n> retry=<json>
      title=off (disable the session-title-llm row)  compaction=<json config for compaction-basic>
      extra_route=<json merged into the route>  extra_model=<json merged into the model>
"""
import json
import sys

out, kind = sys.argv[1], sys.argv[2]
opts = dict(arg.split("=", 1) for arg in sys.argv[3:])
port = opts.get("port", "18080")
model_id = opts.get("model", "local")

thinking = {
    "toggle": {"reasoningEfforts": {"off": None, "medium": "medium"},
               "compat": {"thinkingFormat": "chat-template",
                          "chatTemplateKwargs": {"enable_thinking": {"$var": "thinking.enabled"}}}},
    "bolt": {"reasoningEfforts": {"off": None, "medium": "medium", "high": "high"},
             "compat": {"thinkingFormat": "chat-template",
                        "chatTemplateKwargs": {"enable_thinking": {"$var": "thinking.enabled"},
                                               "medium_effort": {"$var": "thinking.budget"}}}},
    "bolt2": {"reasoningEfforts": {"off": None, "medium": "medium"},
              "compat": {"thinkingFormat": "chat-template",
                         "chatTemplateKwargs": {"enable_thinking": {"$var": "thinking.enabled"},
                                                "medium_effort": {"$var": "thinking.enabled", "omitWhenOff": True}}}},
    "effort-gptoss": {"reasoningEfforts": {"low": "low", "medium": "medium", "high": "high"},
                      "compat": {"thinkingFormat": "openai", "supportsReasoningEffort": True}},
    "effort-qwen38": {"reasoningEfforts": {"off": "none", "low": "low", "medium": "medium"},
                      "compat": {"thinkingFormat": "openai", "supportsReasoningEffort": True}},
    "always": {"reasoningEfforts": {"high": "high"}, "compat": {"supportsReasoningEffort": False}},
    "always-false": {"reasoningEfforts": False},
    "always-off": {"reasoningEfforts": {"off": None, "high": "high"}, "compat": {"supportsReasoningEffort": False}},
    "none": {"reasoningEfforts": False},
}[kind]

model = {"id": model_id, "name": f"{kind} candidate (llmctl)",
         "contextWindow": int(opts.get("context", "131072")),
         "maxTokens": int(opts.get("max_tokens", "16384")), "input": ["text"], **thinking}
model.update(json.loads(opts.get("extra_model", "{}")))
route = {"displayName": "llmctl", "api": "openai-completions", "baseURL": f"http://127.0.0.1:{port}/v1",
         "headers": {"Authorization": "Bearer no-key-required"},
         "compat": {"supportsStore": False, "supportsDeveloperRole": False, "maxTokensField": "max_tokens"},
         "models": [model]}
if kind == "bolt":
    route["thinkingBudgets"] = {"medium": 1, "high": 0}
if "route_reasoning" in opts:
    route["reasoning"] = opts["route_reasoning"]
if "idle_ms" in opts:
    route["streamIdleTimeoutMs"] = int(opts["idle_ms"])
if "retry" in opts:
    route["retryPolicy"] = json.loads(opts["retry"])
route.update(json.loads(opts.get("extra_route", "{}")))
default = {"provider": "llmctl", "model": model_id}
level = opts.get("level", "omit")
if level != "omit":
    default["reasoningEffort"] = level
patch = [{"id": "llm-pi-ai", "config": {"providers": {"llmctl": route}}},
         {"id": "agent-default-model", "config": default}]
if opts.get("title") == "off":
    patch.append({"id": "session-title-llm", "disabled": True})
if "compaction" in opts:
    patch.append({"id": "compaction-basic", "config": json.loads(opts["compaction"])})
with open(out, "w", encoding="utf-8") as handle:
    json.dump(patch, handle, indent=1)
    handle.write("\n")
print(out)
