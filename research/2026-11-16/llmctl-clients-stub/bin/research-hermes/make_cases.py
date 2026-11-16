#!/usr/bin/env python3
"""Write one Hermes profile config.yaml per request scenario into configs/cases/."""
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "configs" / "cases"


def entry(api="http://127.0.0.1:8080/v1", model="local", ctx=131072, extra_body=None):
    lines = ["providers:", "  llmctl:", "    name: llmctl", f"    api: {api}", "    transport: chat_completions",
             f"    default_model: {model}", "    discover_models: false", "    models:", f"      {model}:",
             f"        context_length: {ctx}"]
    if extra_body:
        lines.append("    extra_body:")
        lines += ["      " + l for l in extra_body.strip("\n").splitlines()]
    return "\n".join(lines)


def cfg(name, *, model="local", api="http://127.0.0.1:8080/v1", ctx=131072, extra_body=None, tail="", model_extra=""):
    text = "\n".join([
        "model:", "  provider: custom:llmctl", f"  default: {model}", *([model_extra] if model_extra else []),
        entry(api, model, ctx, extra_body), "gateway:", "  standalone: true", tail.strip("\n"),
        "_config_version: 53", ""])
    (OUT / f"{name}.yaml").write_text(text.replace("\n\n", "\n"))


OUT.mkdir(parents=True, exist_ok=True)
NO_REASONING_FULL = "model_overrides:\n  custom:llmctl:\n    local:\n      supports_reasoning: false"
NO_REASONING_BARE = "model_overrides:\n  custom:\n    local:\n      supports_reasoning: false"

cfg("r02-mo-full", tail=NO_REASONING_FULL)
cfg("r03-mo-bare", tail=NO_REASONING_BARE)
cfg("r04-mo-bare-effort-medium", tail=NO_REASONING_BARE + "\nagent:\n  reasoning_effort: medium")
cfg("r05-ctk-on", extra_body="chat_template_kwargs:\n  enable_thinking: true", tail=NO_REASONING_BARE)
cfg("r06-ctk-off", extra_body="chat_template_kwargs:\n  enable_thinking: false", tail=NO_REASONING_BARE)
cfg("r07-bolt-medium", extra_body="chat_template_kwargs:\n  enable_thinking: true\n  medium_effort: true",
    tail=NO_REASONING_BARE)
cfg("r08-extra-maxtok", extra_body="max_tokens: 16384")
cfg("r09-model-maxtok", model_extra="  max_tokens: 16384")
cfg("r10-effort-none", tail="agent:\n  reasoning_effort: none")
cfg("r11-override-low", tail="agent:\n  reasoning_overrides:\n    local: low")
cfg("r12-baseline-for-flag")
cfg("r13-mo-bare-for-flag", tail=NO_REASONING_BARE)
cfg("r14-vllm-gptoss", model="gpt-oss-20b", api="http://127.0.0.1:8000/v1", ctx=65536)
cfg("r15-vllm-effort-none", model="gpt-oss-20b", api="http://127.0.0.1:8000/v1", ctx=65536,
    tail="agent:\n  reasoning_effort: none")
cfg("r16-supports-tools-false", tail="model_overrides:\n  custom:\n    local:\n      supports_tools: false\n      supports_reasoning: false")
cfg("r17-nostream", model_extra="  streaming: false")
cfg("r18-stale-base-url", model_extra="  base_url: http://127.0.0.1:9/v1")
cfg("r19-ctx-32k", ctx=32768)
cfg("r20-vllm-effort-high", model="gpt-oss-20b", api="http://127.0.0.1:8000/v1", ctx=65536,
    tail="agent:\n  reasoning_effort: high")
cfg("r21-ctk-on-keep-effort", extra_body="chat_template_kwargs:\n  enable_thinking: true")
cfg("r22-platform-toolsets-empty", tail="platform_toolsets:\n  cli: []")
for p in sorted(OUT.glob("r*.yaml")):
    print(p.name)
