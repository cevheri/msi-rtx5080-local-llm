"""Validate docs/opencode.md's config with the proposed sglang-docker provider added, against a saved copy of
https://opencode.ai/config.json. Usage: python3 -I validate_opencode.py SCHEMA DOC OUT"""
import copy, json, re, sys
import jsonschema

schema_path, doc_path, out_path = sys.argv[1:4]
schema = json.load(open(schema_path))
text = open(doc_path).read()
block = re.search(r"```json\n(.*?)\n```", text[text.index("## The validated config"):], re.S).group(1)
config = json.loads(block)
toggle_variants = [
    {"id": "thinking", "body": {"chat_template_kwargs": {"enable_thinking": True}}},
    {"id": "none", "body": {"chat_template_kwargs": {"enable_thinking": False}}},
]
provider = {
    "name": "SGLang (Docker)",
    "package": "@opencode/ai/providers/openai-compatible",
    "settings": {"baseURL": "http://127.0.0.1:30000/v1", "headerTimeout": 600000},
    "compatibility": {"reasoningField": "reasoning_content"},
    "models": {
        "qwen3.5-9b": {
            "modelID": "qwen3.5-9b",
            "name": "Qwen3.5-9B NVFP4 (SGLang)",
            "capabilities": {"tools": True, "input": ["text"], "output": ["text"], "reasoning": {"supported": True}},
            "limit": {"context": 131072, "output": 16384},
            "variants": toggle_variants,
        },
        "k2-horizon-7b": {
            "modelID": "k2-horizon-7b",
            "name": "K2 Horizon 7B AWQ-INT4 (SGLang)",
            "capabilities": {"tools": True, "input": ["text"], "output": ["text"], "reasoning": {"supported": True}},
            "limit": {"context": 65536, "output": 16384},
        },
        "gpt-oss-20b": {
            "modelID": "gpt-oss-20b",
            "name": "gpt-oss-20b MXFP4, Paged Experts (SGLang)",
            "capabilities": {"tools": True, "input": ["text"], "output": ["text"],
                             "reasoning": {"supported": True, "effort": ["low", "medium", "high"]}},
            "limit": {"context": 131072, "output": 16384},
        },
    },
}
config["providers"]["sglang-docker"] = provider
validator = jsonschema.Draft202012Validator(schema)
errors = sorted(validator.iter_errors(config), key=lambda e: list(e.path))
bad_npm = copy.deepcopy(config); bad_npm["providers"]["sglang-docker"]["npm"] = "@ai-sdk/openai-compatible"
bad_tools = copy.deepcopy(config); bad_tools["providers"]["sglang-docker"]["models"]["qwen3.5-9b"]["capabilities"]["tools"] = "true"
result = {
    "models": sum(len(p["models"]) for p in config["providers"].values()),
    "errors": [f"{list(e.path)}: {e.message}" for e in errors],
    "negative_npm_rejected": any(True for _ in validator.iter_errors(bad_npm)),
    "negative_tools_string_rejected": any(True for _ in validator.iter_errors(bad_tools)),
}
json.dump(result, open(out_path, "w"), indent=1)
print(json.dumps(result, indent=1))
json.dump(provider, open(out_path.replace(".json", "-provider.json"), "w"), indent=2)
