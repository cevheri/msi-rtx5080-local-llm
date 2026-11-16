#!/usr/bin/env python3
"""Design prototype (not llmctl code): generate the client entries the design specifies.

Reads the shipped recipes, applies the proposed [clients] metadata, and writes for each recipe:
  <out>/<runtime>/<id>/hermes-config.yaml   a new llmctl profile config (header + managed block)
  <out>/<runtime>/<id>/hermes-block.yaml    the managed block alone
  <out>/<runtime>/<id>/pi-provider.json     providers["llmctl"] for models.json
  <out>/<runtime>/<id>/pi-level.txt         the modelThinkingLevels value
  <out>/<runtime>/<id>/opencode-expected.json  the OpenCode entry the metadata implies
It also compares the implied OpenCode entry with docs/opencode.md and the vllm/models.md snippets.
usage: gen.py <repo> <out>
"""
import json
import re
import sys
import tomllib
from pathlib import Path

BEGIN = "# >>> llmctl managed block: rewritten by llmctl on every start; edits here are lost >>>"
END = "# <<< llmctl managed block <<<"
HEADER = ("# Hermes profile for the model llmctl runs (docs/llmctl.md#clients).\n"
          "# llmctl rewrites the block below on every ./llmctl up and keeps everything outside it.\n"
          "# The keys model, providers, model_overrides, gateway and platform_toolsets belong to the block.\n")
TRAILER = "_config_version: 53\n"
PI_LEVELS = ("off", "minimal", "low", "medium", "high")

# Proposed [clients] metadata: (output, thinking, default, efforts, medium_kwargs)
META = {
    ("llamacpp", "qwen3.8-27b"): (16384, "effort", "medium", ["none", "low", "medium"], None),
    ("llamacpp", "qwen3.6-35b-a3b"): (16384, "toggle", "off", None, None),
    ("llamacpp", "gemma-4-26b-a4b"): (16384, "toggle", "off", None, None),
    ("llamacpp", "nemotron-3.5-lightning"): (16384, "toggle", "off", None, None),
    ("llamacpp", "groot-19b-a3b"): (16384, "toggle", "off", None, None),
    ("llamacpp", "nemotron-3.5-bolt"): (16384, "toggle", "off", None, {"medium_effort": True}),
    ("llamacpp", "k2-horizon-mova-36b"): (16384, "always", None, None, None),
    ("llamacpp", "k2-horizon-7b"): (32768, "always", None, None, None),
    ("llamacpp", "qwen3-coder-next"): (16384, "none", None, None, None),
    ("llamacpp", "gpt-oss-20b"): (16384, "effort", "medium", ["low", "medium", "high"], None),
    ("llamacpp", "deepseek-r1-0528-qwen3-8b"): (16384, "always", None, None, None),
    ("vllm", "gpt-oss-20b"): (16384, "effort", "medium", ["low", "medium", "high"], None),
    ("vllm", "groot-19b-a3b"): (16384, "toggle", "off", None, None),
    ("vllm", "k2-horizon-7b"): (16384, "always", None, None, None),
    ("vllm", "deepseek-r1-0528-qwen3-8b"): (16384, "always", None, None, None),
    ("vllm", "gemma-4-12b"): (16384, "toggle", "off", None, None),
    ("vllm", "gemma-4-e4b"): (16384, "toggle", "off", None, None),
    ("vllm", "qwen3.5-9b"): (16384, "toggle", "off", None, None),
    ("vllm", "mellum2.1-12b"): (16384, "toggle", "on", None, None),
}
RUNTIME_LABEL = {"llamacpp": "llama.cpp", "vllm": "vLLM"}


def flow(value) -> str:
    """JSON text, which YAML 1.1 reads as the same flow collection: ASCII only, no '\\/' escapes."""
    text = json.dumps(value, ensure_ascii=True, separators=(", ", ": "))
    assert "\\/" not in text and "\t" not in text
    return text


def default_level(thinking, default):
    return {"toggle": "off" if default == "off" else "medium", "effort": "medium",
            "always": "high", "none": "off"}[thinking]


def hermes_block(rid, runtime, recipe, served, port, meta, sha):
    output, thinking, default, efforts, medium = meta
    provider = {"name": f"llmctl: {rid} on {RUNTIME_LABEL[runtime]}",
                "api": f"http://127.0.0.1:{port}/v1",
                "transport": "chat_completions", "default_model": served, "discover_models": False,
                "extra_body": {"max_tokens": output},
                "models": {served: {"context_length": recipe["server"]["context"]}}}
    overrides = {} if thinking == "effort" else {"custom": {served: {"supports_reasoning": False}}}
    lines = [BEGIN,
             f"model: {flow({'provider': 'custom:llmctl', 'default': served})}",
             f"providers: {flow({'llmctl': provider})}",
             f"model_overrides: {flow(overrides)}",
             f"gateway: {flow({'standalone': True})}"]
    if recipe.get("chat_only"):
        lines.append(f"platform_toolsets: {flow({'cli': []})}")
    lines.append(END)
    return "\n".join(lines) + "\n"


def pi_provider(rid, runtime, recipe, served, port, meta):
    output, thinking, default, efforts, medium = meta
    model = {"id": served, "name": f"{rid} on {RUNTIME_LABEL[runtime]} (llmctl)",
             "reasoning": thinking != "none", "input": ["text"],
             "contextWindow": recipe["server"]["context"], "maxTokens": output}
    if thinking == "toggle":
        model["thinkingLevelMap"] = ({"minimal": None, "low": None} if medium
                                     else {"minimal": None, "low": None, "high": None})
        model["compat"] = {"thinkingFormat": "chat-template",
                           "chatTemplateKwargs": {"enable_thinking": {"$var": "thinking.enabled"}}}
        if medium:
            model["samplingParamsByThinkingLevel"] = {
                "medium": {"chat_template_kwargs": {"enable_thinking": True, **medium}}}
    elif thinking == "effort":
        level_map = {}
        for level in PI_LEVELS:
            value = "none" if level == "off" else level
            level_map[level] = value if value in efforts else None
        model["thinkingLevelMap"] = level_map
        model["compat"] = {"thinkingFormat": "openai", "supportsReasoningEffort": True}
    elif thinking == "always":
        model["thinkingLevelMap"] = {"off": None, "minimal": None, "low": None, "medium": None}
        model["compat"] = {"supportsReasoningEffort": False}
    return {"name": "llmctl", "baseUrl": f"http://127.0.0.1:{port}/v1", "api": "openai-completions",
            "apiKey": "no-key-required",
            "compat": {"supportsStore": False, "supportsDeveloperRole": False, "maxTokensField": "max_tokens"},
            "models": [model]}


def opencode_expected(recipe, served, meta):
    output, thinking, default, efforts, medium = meta
    reasoning = {"supported": thinking != "none"}
    if thinking == "effort":
        reasoning["effort"] = efforts
    entry = {"modelID": served,
             "capabilities": {"tools": not recipe.get("chat_only", False), "input": ["text"], "output": ["text"],
                              "reasoning": reasoning},
             "limit": {"context": recipe["server"]["context"], "output": output}}
    if thinking == "toggle":
        variants = [{"id": "thinking", "body": {"chat_template_kwargs": {"enable_thinking": True}}}]
        if medium:
            variants.append({"id": "medium", "body": {"chat_template_kwargs": {"enable_thinking": True, **medium}}})
        variants.append({"id": "none", "body": {"chat_template_kwargs": {"enable_thinking": False}}})
        entry["variants"] = variants
    return entry


def documented_entries(repo):
    """OpenCode entries from docs/opencode.md (by provider and key) and the vllm/models.md snippets (by modelID)."""
    text = (repo / "docs/opencode.md").read_text()
    block = re.search(r"```json\n(.*?)\n```", text[text.index("## The validated config"):], re.S).group(1)
    providers = json.loads(block)["providers"]
    snippets = {}
    for doc in ("vllm/models.md", "llamacpp/models.md"):
        for match in re.finditer(r"OpenCode entry[^\n]*\n\n```json\n(.*?)\n```", (repo / doc).read_text(), re.S):
            entry = json.loads("{" + match.group(1) + "}")
            (key, value), = entry.items()
            snippets[(doc.split("/")[0], value["modelID"])] = value
    return providers, snippets


def compare(expected, documented):
    doc = {k: v for k, v in documented.items() if k != "name"}
    return [] if doc == expected else [f"documented {json.dumps(doc, sort_keys=True)}",
                                        f"expected   {json.dumps(expected, sort_keys=True)}"]


def main():
    repo, out = Path(sys.argv[1]), Path(sys.argv[2])
    providers, snippets = documented_entries(repo)
    report = []
    for (runtime, rid), meta in META.items():
        path = repo / "recipes" / runtime / f"{rid}.toml"
        raw = path.read_bytes()
        recipe = tomllib.loads(raw.decode())
        import hashlib
        sha = hashlib.sha256(raw).hexdigest()
        runtime_file = tomllib.loads((repo / "recipes" / runtime / "_runtime.toml").read_text())
        served = runtime_file.get("served_name") or recipe["model"]["served_name"]
        port = runtime_file["host_port"]
        target = out / runtime / rid
        target.mkdir(parents=True, exist_ok=True)
        block = hermes_block(rid, runtime, recipe, served, port, meta, sha)
        (target / "hermes-block.yaml").write_text(block)
        (target / "hermes-config.yaml").write_text(HEADER + block + TRAILER)
        (target / "pi-provider.json").write_text(json.dumps(pi_provider(rid, runtime, recipe, served, port, meta), indent=2) + "\n")
        (target / "pi-level.txt").write_text(default_level(meta[1], meta[2]) + "\n")
        expected = opencode_expected(recipe, served, meta)
        (target / "opencode-expected.json").write_text(json.dumps(expected, indent=2) + "\n")
        sources = []
        provider = runtime_file["opencode_provider"]
        if rid in providers.get(provider, {}).get("models", {}):
            sources.append(("docs/opencode.md", providers[provider]["models"][rid]))
        if (runtime, served) in snippets and runtime == "vllm":
            sources.append(("vllm/models.md", snippets[(runtime, served)]))
        if not sources:
            report.append(f"{runtime}/{rid}: NO documented OpenCode entry (opencode={recipe.get('opencode', False)})")
        for name, documented in sources:
            problems = compare(expected, documented)
            report.append(f"{runtime}/{rid} vs {name}: {'match' if not problems else 'MISMATCH'}")
            report += ["    " + p for p in problems]
    (out / "metadata-vs-docs.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
