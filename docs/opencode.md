# OpenCode with the local servers

Last verified: 2026-11-16.

Point OpenCode v2.3.0 at the local servers through custom `@opencode/ai/providers/openai-compatible` providers, one for llama.cpp on `127.0.0.1:8080` and one for vLLM on `127.0.0.1:8000`, with `capabilities.tools: true` and an explicit `limit.context` on every model.
Turn off the built-in vLLM discovery, because it registers every discovered model with tools disabled.
Keep thinking off by default on the server for toggle models and opt in per session with a `#thinking` variant; Qwen3.8-27B runs at medium effort with `none`, `low` and `medium` choices instead, and the K2 Horizon models and gpt-oss-20b always think.
Measured on 2026-11-16: the same OpenCode v2.3.0 write-and-read session passed against eleven local servers (gpt-oss-20b on llama.cpp and on vLLM, Qwen3.8-27B, Qwen3.6-35B-A3B, Gemma 4 26B-A4B, K2 Horizon 7B and MoVA, Qwen3-Coder-Next, Groot-19B-A3B, and Nemotron 3.5 Bolt and Lightning on llama.cpp), each with the entry's default variant ([records](../research/2026-11-16/README.md)).
It also passed against five small models on Ollama 0.40.16 ([Ollama notes](#ollama-notes)).
The `#thinking` variants and longer sessions are not measured yet; the rest of this page is schema-validated config and behaviour read from source.

## Config shape in v2.3.0

OpenCode v2.3.0 was released on 2026-11-13 from the [anomalyco/opencode repository](https://github.com/anomalyco/opencode/releases/tag/v2.3.0), and V2 changed the config shape ([migration guide](https://opencode.ai/docs/migrate-v1/)).
V1 keys are renamed: `provider` became `providers`, `npm` became `package`, `options` split into `settings`, `headers` and `body`, `tool_call` became `capabilities.tools`, and `small_model` became `agents.title.model` ([migration guide](https://opencode.ai/docs/migrate-v1/)).
The live schema at [https://opencode.ai/config.json](https://opencode.ai/config.json) rejects the V1 keys `npm`, `options` and `tool_call` inside a V2 provider, so old snippets from blog posts fail validation.

Config files load from `~/.config/opencode/opencode.json(c)` (global) and `opencode.json(c)` in the project, with the project file overriding the global one ([config docs](https://opencode.ai/docs/config/)).
After editing, `opencode reload` reloads the configuration and `opencode debug config` lists the configuration sources (`opencode --help` on v2.3.0, 2026-11-15).

| Key | Where | Meaning here |
|---|---|---|
| `providers.<id>` | root | Provider ID; it becomes the prefix in `provider/model` ([providers docs](https://opencode.ai/docs/providers/)) |
| `package` | provider | `@opencode/ai/providers/openai-compatible`; the old `@ai-sdk/openai-compatible` maps to the same native package ([aisdk-native.ts](https://github.com/anomalyco/opencode/blob/v2.3.0/packages/core/src/aisdk-native.ts#L127)) |
| `settings.baseURL`, `settings.apiKey`, `settings.headerTimeout` | provider | Endpoint, key (use `{env:VAR}`), and time to first response header; default 5 minutes ([providers docs](https://opencode.ai/docs/providers/)) |
| `compatibility.reasoningField` | provider or model | Name of the streamed reasoning field: `reasoning` or `reasoning_content` ([models docs](https://opencode.ai/docs/models/)) |
| `compatibility.maxTokensField` | provider or model | `max_completion_tokens` (default) or `max_tokens` ([schema](https://opencode.ai/config.json)) |
| `models.<key>.modelID` | model | The model name sent to the server; must match what the server serves |
| `capabilities.tools` | model | Must be `true` for an agent; OpenCode sends no tools otherwise |
| `capabilities.reasoning` | model | `{supported, effort[]}`; `effort` builds variants that send `reasoning_effort` ([variant.ts](https://github.com/anomalyco/opencode/blob/v2.3.0/packages/core/src/variant.ts)) |
| `limit.context`, `limit.output` | model | Window and output cap; without them a custom model is assumed to have 200k context and 32k output ([models docs](https://opencode.ai/docs/models/)) |
| `variants[]` | model | Array of `{id, settings, headers, body}`; select with `provider/model#variant` ([models docs](https://opencode.ai/docs/models/)) |
| `plugins` | root | `"-opencode.provider.vllm"` removes the built-in vLLM discovery ([models docs](https://opencode.ai/docs/models/)) |
| `model` | root | Default model as `provider/model`; the root default does not keep a `#variant` ([models docs](https://opencode.ai/docs/models/)) |

OpenCode merges `body` in the order provider, then model, then variant ([variant.ts](https://github.com/anomalyco/opencode/blob/v2.3.0/packages/core/src/variant.ts)).
A custom openai-compatible model with neither `variants` nor `capabilities.reasoning` set gets automatic low, medium and high `reasoning_effort` variants ([config/plugin/provider.ts](https://github.com/anomalyco/opencode/blob/v2.3.0/packages/core/src/config/plugin/provider.ts)), so set `reasoning: { "supported": false }` on a model that has no thinking mode.

## The validated config

This is the two-provider config for the top picks in [model-selection.md](model-selection.md): Qwen3.6-35B-A3B (the default), Qwen3.8-27B, K2 Horizon MoVA 36B A4B and K2 Horizon 7B on llama.cpp, and gpt-oss-20b and Groot-19B-A3B-261023 NVFP4 on vLLM.
It extends the research-phase config with a `qwen3.8-27b` entry that uses `reasoning_effort` levels, as the audit asked.
After the 2026-11-15 re-ranking, the default moved from Qwen3.6-35B-A3B to Qwen3.8-27B, the two K2 Horizon entries were added, and the Gemma 4 12B entry (now an honorable mention; its entry is in [vllm/models.md](../vllm/models.md)) was replaced by Groot, whose entry is copied from [vllm/models.md](../vllm/models.md).
On 2026-11-16 the user made Qwen3.6-35B-A3B the default again, after it measured 122 tok/s with 128k context against 36 tok/s with 64k for Qwen3.8-27B ([model-selection.md](model-selection.md#decision-summary)).
The K2 Horizon entries have no variants because both models always think and the server sets `reasoning_effort` high ([llamacpp/models.md](../llamacpp/models.md)).
I re-validated the result on 2026-11-16, after adding the Qwen3-Coder-Next entry, against a fresh copy of [https://opencode.ai/config.json](https://opencode.ai/config.json) (unchanged since 2026-11-15) with Python `jsonschema` 4.19.2 (Draft 2020-12): 0 errors, and negative tests with a V1 `npm` key and a string `tools` value are rejected.
On 2026-11-16 this config was installed as the global `~/.config/opencode/opencode.json` on this laptop, merged with the `mcp` block that file already held, and validated again with 0 errors against the live schema (unchanged).
Measured the same day: `opencode run` without `-m` reached the default `qwen3.6-35b-a3b` entry and passed the write-and-read task in 9 s ([records](../research/2026-11-16/opencode-global-config/)).

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "llamacpp-docker/qwen3.6-35b-a3b",
  "plugins": ["-opencode.provider.vllm"],
  "providers": {
    "llamacpp-docker": {
      "name": "llama.cpp (Docker)",
      "package": "@opencode/ai/providers/openai-compatible",
      "settings": { "baseURL": "http://127.0.0.1:8080/v1", "headerTimeout": 600000 },
      "compatibility": { "reasoningField": "reasoning_content" },
      "models": {
        "qwen3.8-27b": {
          "modelID": "local",
          "name": "Qwen3.8-27B UD-Q3_K_XL (llama.cpp)",
          "capabilities": { "tools": true, "input": ["text"], "output": ["text"], "reasoning": { "supported": true, "effort": ["none", "low", "medium"] } },
          "limit": { "context": 65536, "output": 16384 }
        },
        "qwen3.6-35b-a3b": {
          "modelID": "local",
          "name": "Qwen3.6-35B-A3B Q4_K_M (llama.cpp)",
          "capabilities": { "tools": true, "input": ["text"], "output": ["text"], "reasoning": { "supported": true } },
          "limit": { "context": 131072, "output": 16384 },
          "variants": [
            { "id": "thinking", "body": { "chat_template_kwargs": { "enable_thinking": true } } },
            { "id": "none", "body": { "chat_template_kwargs": { "enable_thinking": false } } }
          ]
        },
        "gemma-4-26b-a4b": {
          "modelID": "local",
          "name": "Gemma 4 26B-A4B QAT Q4_0 (llama.cpp)",
          "capabilities": { "tools": true, "input": ["text"], "output": ["text"], "reasoning": { "supported": true } },
          "limit": { "context": 131072, "output": 16384 },
          "variants": [
            { "id": "thinking", "body": { "chat_template_kwargs": { "enable_thinking": true } } },
            { "id": "none", "body": { "chat_template_kwargs": { "enable_thinking": false } } }
          ]
        },
        "nemotron-3.5-lightning": {
          "modelID": "local",
          "name": "Nemotron 3.5 Lightning 30B-A3B Q4_0 (llama.cpp)",
          "capabilities": { "tools": true, "input": ["text"], "output": ["text"], "reasoning": { "supported": true } },
          "limit": { "context": 131072, "output": 16384 },
          "variants": [
            { "id": "thinking", "body": { "chat_template_kwargs": { "enable_thinking": true } } },
            { "id": "none", "body": { "chat_template_kwargs": { "enable_thinking": false } } }
          ]
        },
        "groot-19b-a3b": {
          "modelID": "local",
          "name": "Groot-19B-A3B Q4_K_M (llama.cpp)",
          "capabilities": { "tools": true, "input": ["text"], "output": ["text"], "reasoning": { "supported": true } },
          "limit": { "context": 131072, "output": 16384 },
          "variants": [
            { "id": "thinking", "body": { "chat_template_kwargs": { "enable_thinking": true } } },
            { "id": "none", "body": { "chat_template_kwargs": { "enable_thinking": false } } }
          ]
        },
        "nemotron-3.5-bolt": {
          "modelID": "local",
          "name": "Nemotron 3.5 Bolt 74B-A9B 24GB GGUF (llama.cpp)",
          "capabilities": { "tools": true, "input": ["text"], "output": ["text"], "reasoning": { "supported": true } },
          "limit": { "context": 131072, "output": 16384 },
          "variants": [
            { "id": "thinking", "body": { "chat_template_kwargs": { "enable_thinking": true } } },
            { "id": "medium", "body": { "chat_template_kwargs": { "enable_thinking": true, "medium_effort": true } } },
            { "id": "none", "body": { "chat_template_kwargs": { "enable_thinking": false } } }
          ]
        },
        "k2-horizon-mova-36b": {
          "modelID": "local",
          "name": "K2 Horizon MoVA 36B A4B Q4_K_M (llama.cpp)",
          "capabilities": { "tools": true, "input": ["text"], "output": ["text"], "reasoning": { "supported": true } },
          "limit": { "context": 65536, "output": 16384 }
        },
        "k2-horizon-7b": {
          "modelID": "local",
          "name": "K2 Horizon 7B Q8_0 (llama.cpp)",
          "capabilities": { "tools": true, "input": ["text"], "output": ["text"], "reasoning": { "supported": true } },
          "limit": { "context": 81920, "output": 32768 }
        },
        "qwen3-coder-next": {
          "modelID": "local",
          "name": "Qwen3-Coder-Next UD-IQ4_XS (llama.cpp)",
          "capabilities": { "tools": true, "input": ["text"], "output": ["text"], "reasoning": { "supported": false } },
          "limit": { "context": 131072, "output": 16384 }
        }
      }
    },
    "vllm-docker": {
      "name": "vLLM (Docker)",
      "package": "@opencode/ai/providers/openai-compatible",
      "settings": { "baseURL": "http://127.0.0.1:8000/v1", "headerTimeout": 600000 },
      "compatibility": { "reasoningField": "reasoning" },
      "models": {
        "gpt-oss-20b": {
          "modelID": "gpt-oss-20b",
          "name": "gpt-oss-20b (vLLM)",
          "capabilities": { "tools": true, "input": ["text"], "output": ["text"], "reasoning": { "supported": true, "effort": ["low", "medium", "high"] } },
          "limit": { "context": 65536, "output": 16384 }
        },
        "groot-19b-a3b": {
          "modelID": "groot-19b-a3b",
          "name": "Groot-19B-A3B NVFP4 (vLLM)",
          "capabilities": { "tools": true, "input": ["text"], "output": ["text"], "reasoning": { "supported": true } },
          "limit": { "context": 131072, "output": 16384 },
          "variants": [
            { "id": "thinking", "body": { "chat_template_kwargs": { "enable_thinking": true } } },
            { "id": "none", "body": { "chat_template_kwargs": { "enable_thinking": false } } }
          ]
        }
      }
    }
  }
}
```

Only one model fits the 16 GB GPU at a time, so only one of these entries answers at any moment; pick the entry that matches the container you started.
The `headerTimeout` of 600000 ms (10 minutes) leaves room for a long first prefill when MoE experts run from system RAM; the default is 5 minutes ([providers docs](https://opencode.ai/docs/providers/)).
No `apiKey` is set because the servers listen on 127.0.0.1 only (see [host-setup.md](host-setup.md)); if you start vLLM with `--api-key`, add `"apiKey": "{env:VLLM_API_KEY}"` to `settings`.

To add another pick, copy the entry shape and take the served name, context and thinking mechanism from [vllm/models.md](../vllm/models.md) or [llamacpp/models.md](../llamacpp/models.md).
The Gemma 4 26B-A4B, Nemotron 3.5 Bolt and Lightning, llama.cpp Groot and Qwen3-Coder-Next entries above were added on 2026-11-16 after their measurements; Bolt's extra `medium` variant sends the template's `medium_effort` flag.
For a model without thinking, such as Qwen3-Coder-Next, set `"reasoning": { "supported": false }` and drop the variants, as in its entry above.
For DeepSeek-R1-0528-Qwen3-8B set `"tools": false`: it has no tool calling and is a chat-only pick ([model-selection.md](model-selection.md)).
Measured on 2026-11-16: with tools enabled for a test, it wrote the `write` tool's arguments into its answer as text and never called the tool ([llamacpp/models.md](../llamacpp/models.md#10-deepseek-r1-0528-qwen3-8b-chat-and-reasoning-only)).
Re-validate after any change; the Maintenance section has the command.

## Matching rules

OpenCode does not ask the server what it serves when discovery is off, so three values must match by hand.

| OpenCode field | vLLM flag | llama.cpp flag |
|---|---|---|
| `modelID` | `--served-model-name <name>` | `--alias <name>` (`-a`); without it the model `id` is the file path ([server README](https://github.com/ggml-org/llama.cpp/blob/b12418/tools/server/README.md)) |
| `limit.context` | `--max-model-len N` | `-c N` / `--ctx-size N` |
| `compatibility.reasoningField` | `reasoning` | `reasoning_content` ([server README](https://github.com/ggml-org/llama.cpp/blob/b12418/tools/server/README.md)) |

The config above assumes these server settings; the full commands live in [vllm/README.md](../vllm/README.md) and [llamacpp/README.md](../llamacpp/README.md).

| Entry | Server | Served name | Context |
|---|---|---|---|
| `llamacpp-docker/qwen3.8-27b` | llama.cpp :8080 | `--alias local` | `-c 65536` |
| `llamacpp-docker/qwen3.6-35b-a3b` | llama.cpp :8080 | `--alias local` | `-c 131072` |
| `llamacpp-docker/k2-horizon-mova-36b` | llama.cpp :8080 | `--alias local` | `-c 65536` |
| `llamacpp-docker/k2-horizon-7b` | llama.cpp :8080 | `--alias local` | `-c 65536` |
| `vllm-docker/groot-19b-a3b` | vLLM :8000 | `--served-model-name groot-19b-a3b` | `--max-model-len 131072` (FP8 KV) |
| `vllm-docker/gpt-oss-20b` | vLLM :8000 | `--served-model-name gpt-oss-20b` | `--max-model-len 65536` (needs `--gpu-memory-utilization 0.95`, FP8 KV and `TRITON_ATTN`) |

Serving every llama.cpp model as `local` means the entry names, not the server, carry the model identity; it also means a wrong entry still gets an answer from whatever model is loaded, so check the window and thinking settings when you switch.
If `limit.context` is larger than the server window, OpenCode never compacts in time and the server rejects the request; llama-server answers HTTP 400 `exceed_context_size_error` ([Unsloth Qwen3-Coder-Next page](https://unsloth.ai/docs/models/qwen3-coder-next)).
If it is smaller, you waste window that the KV budget already paid for.

On vLLM, tool calling also needs `--enable-auto-tool-choice` and the right `--tool-call-parser` for the model, and a thinking model also needs its `--reasoning-parser` ([vllm/README.md](../vllm/README.md)).
A wrong tool parser does not raise an error: the tool calls arrive as plain text and the agent stalls (lesson from the user's earlier DGX Spark runbook).

## Built-in vLLM discovery disables tools

OpenCode V2 discovers a vLLM server on `127.0.0.1:8000/v1` by itself ([models docs](https://opencode.ai/docs/models/)).
Discovery reads only the model id and `max_model_len`, and every discovered model gets `tools: spec.tools ?? false` ([vllm.ts](https://github.com/anomalyco/opencode/blob/v2.3.0/packages/core/src/plugin/provider/vllm.ts), [local.ts](https://github.com/anomalyco/opencode/blob/v2.3.0/packages/core/src/plugin/provider/local.ts)), because tool calling depends on server flags that discovery cannot see.
A discovered vLLM model therefore runs the agent without tools.

The config above avoids this in two ways: `"plugins": ["-opencode.provider.vllm"]` removes discovery, and the separate provider ID `vllm-docker` declares tools explicitly.
An alternative is to keep discovery and override `providers.vllm.models["<served name>"].capabilities.tools: true`; the config plugin runs after the discovery plugin ([internal.ts](https://github.com/anomalyco/opencode/blob/v2.3.0/packages/core/src/plugin/internal.ts)), so the override should win, but that is read from the code and not tested.
Ollama and LM Studio discovery do read tool support, so the problem is specific to vLLM ([local.ts](https://github.com/anomalyco/opencode/blob/v2.3.0/packages/core/src/plugin/provider/local.ts)).

## Thinking variants per runtime

| Model family | Runtime | OpenCode mechanism | Reasoning field |
|---|---|---|---|
| Qwen3.6, Groot, Gemma 4, Nemotron 3.5, GLM-4.7 | vLLM or llama.cpp | `variants` with `body.chat_template_kwargs.enable_thinking` true or false | vLLM `reasoning`, llama.cpp `reasoning_content` |
| Qwen3.8-27B | llama.cpp | `capabilities.reasoning.effort` `none`, `low`, `medium`; llama.cpp passes `reasoning_effort` to the template and `none` disables thinking ([server README](https://github.com/ggml-org/llama.cpp/blob/b12418/tools/server/README.md)) | `reasoning_content` |
| K2 Horizon MoVA and 7B | llama.cpp (7B also vLLM) | No variants: thinking is always on, and the server sets `reasoning_effort` high through `--chat-template-kwargs` (llama.cpp) or `--default-chat-template-kwargs` (vLLM), as in [llamacpp/models.md](../llamacpp/models.md) and [vllm/models.md](../vllm/models.md) | as the runtime |
| gpt-oss-20b | vLLM or llama.cpp | `capabilities.reasoning.effort` `low`, `medium`, `high`, sent as `reasoning_effort` ([variant.ts](https://github.com/anomalyco/opencode/blob/v2.3.0/packages/core/src/variant.ts)); it always thinks | as the runtime |
| Any | Ollama | `capabilities.reasoning.effort` with `none` to turn thinking off, because Ollama `/v1` maps `none` to `think: false` ([openai.go](https://github.com/ollama/ollama/blob/v0.40.16/openai/openai.go)); a model that cannot disable thinking, such as gpt-oss, uses its lowest level instead ([Ollama thinking docs](https://docs.ollama.com/capabilities/thinking)) | `reasoning` |

Select a variant with `provider/model#variant`, for example `opencode run --model llamacpp-docker/qwen3.6-35b-a3b#thinking` ([models docs](https://opencode.ai/docs/models/)).
OpenCode's own NVIDIA protocol uses the same `chat_template_kwargs.enable_thinking` pattern for Qwen, Gemma and Nemotron, which is a good sign the body passes through unchanged ([variant.ts](https://github.com/anomalyco/opencode/blob/v2.3.0/packages/core/src/variant.ts)).
Setting `reasoningField` also turns on `requireReasoning`, which re-sends earlier reasoning on assistant messages ([models docs](https://opencode.ai/docs/models/)); whether each chat template uses that field is unchecked, and `"requireReasoning": false` is the fallback if it causes errors.

For toggle models, keep thinking off by default on the server: vLLM `--default-chat-template-kwargs '{"enable_thinking":false}'`, llama.cpp `--reasoning off` or `--chat-template-kwargs` ([vllm/README.md](../vllm/README.md), [llamacpp/README.md](../llamacpp/README.md)).
A request without a variant then gets the server default, and `#thinking` opts in.
The lesson comes from the user's DGX Spark runbook: Qwen3.6 with thinking on spent all 1024 output tokens reasoning (`finish_reason: length`, no answer, 23.7 s), while thinking off answered correctly in 1.5 s (measured on a DGX Spark with vLLM, 2026-09-05; not measured on this laptop).
Thinking still helps agent quality: Qwen3.8-27B scores 49.1 on Terminal-Bench 2.1 without reasoning and 67.4 at low effort (measured by [Artificial Analysis](https://artificialanalysis.ai/models/qwen3-8-27b) at full precision through cloud APIs with the Terminus 2 harness, not a local quant or OpenCode; see [community-benchmarks.md](community-benchmarks.md)), so benchmark both settings instead of assuming one.
Qwen3.8's chat template defaults to `xhigh` effort and accepts only `xhigh`, `medium` and `low` ([chat_template.jinja](https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/chat_template.jinja)); llama.cpp handles `none` itself before the template ([server README](https://github.com/ggml-org/llama.cpp/blob/b12418/tools/server/README.md)).
Because `xhigh` is verbose, its entry offers only `none`, `low` and `medium`.
A desktop RTX 5080 user running OpenCode on llama.cpp reports a `--reasoning-budget 1024` cap to stop runaway thinking ([r/LocalLLaMA post](https://www.reddit.com/r/LocalLLaMA/comments/1s9mkm1/benchmarked_18_models_that_i_can_run_on_my_rtx/), anecdotal).

For title generation and compaction you can point `agents.title.model` and `agents.compaction.model` at a `#none` variant ([compaction docs](https://opencode.ai/docs/compaction/)), but only at the model that is actually being served.
A pin to a model that is not loaded fails on every title and compaction request, so either leave these unset (they fall back to the session model) or repoint them each time you switch containers.

## Context budget

The practical minimum for OpenCode is 64k tokens: Ollama's OpenCode guide says OpenCode needs at least 64k ([Ollama docs](https://docs.ollama.com/integrations/opencode)), and the arithmetic below agrees.
The default build agent starts with about 19k to 24k characters of fixed prompt, roughly 5.5k to 7k tokens at 3.5 characters per token (estimate from the v2.3.0 source: about 2k characters of system prompt, about 8.9k of tool descriptions for 13 tools plus about 3.7k of JSON schema, and 1.7k to 9.7k for the Code Mode `execute` tool and catalog ([tool plugins](https://github.com/anomalyco/opencode/tree/v2.3.0/packages/core/src/tool/plugin))).
`AGENTS.md`, skills (OpenCode also scans `.claude` directories) and MCP tools come on top of that and are not included in the estimate.
Measured on 2026-11-16: the first request of an OpenCode v2.3.0 `run` against llama.cpp with gpt-oss-20b was 3,648 prompt tokens, in a scratch project with no `AGENTS.md`, the MCP server disabled and the `bash`, `task`, web and skill tools denied ([record](../research/2026-11-16/llamacpp-gpt-oss-20b/opencode-e2e-events.jsonl), [test config](../research/2026-11-16/opencode-test-config.json)).
With the same config the first request was 4,180 tokens for Qwen3.8-27B, 4,186 for Qwen3.6-35B-A3B and Groot-19B-A3B, 3,824 for Gemma 4 26B-A4B, 3,482 and 3,489 for K2 Horizon 7B and MoVA, 4,370 and 4,372 for Nemotron 3.5 Lightning and Bolt, and 4,249 for Qwen3-Coder-Next, whose tokenizers and chat templates differ from gpt-oss ([Qwen3.6 record](../research/2026-11-16/llamacpp-qwen3.6-35b-a3b/opencode-e2e-events.jsonl)).
Denied tools are likely left out of the tool list, so a default session sends more; measure it again with the tools you actually allow (inferred).
Measured with the config installed globally on 2026-11-16: with only shell, task, web, question and external-directory tools denied, the first Qwen3.6 request was 6,588 tokens, and denying the skill tool as well brought it back to 4,192 ([records](../research/2026-11-16/opencode-global-config/)).
So the skill tool and its list of the skills OpenCode found on this machine cost about 2.4k tokens; the configured MCP server was not running, so it added no tools.
On later steps llama.cpp reused the cached prefix, and only 78 and 86 new tokens were processed.

Compaction keeps a buffer of 15% or at least 16k tokens free and keeps about 15k recent tokens ([compaction docs](https://opencode.ai/docs/compaction/)).
At 32k, compaction would start near 16k used and leave about 9k tokens of real conversation (estimate); at 64k the threshold is near 48k and leaves about 41k (estimate).
The docs warn that compaction cannot help when fixed instructions and tools fill most of the window ([compaction docs](https://opencode.ai/docs/compaction/)).
One desktop RTX 5080 user who runs OpenCode daily wants 80k to 120k at minimum and runs 131k ([r/LocalLLaMA post](https://www.reddit.com/r/LocalLLaMA/comments/1s9mkm1/benchmarked_18_models_that_i_can_run_on_my_rtx/), anecdotal), so treat 64k as the floor and prefer 128k where the KV budget allows it.
A 2B to 4B model is not a viable OpenCode backend even when its chat answers look good: in OpenCode-style long-context agent loops, small models fail through malformed or missing tool calls, worse as context grows ([r/LocalLLaMA thread](https://www.reddit.com/r/LocalLLaMA/comments/1s7r9wu/), one author's 25-task text-to-SQL agent benchmark, single runs mostly through OpenRouter; anecdotal).
That is a statement about long contexts, not about tool calling as such: 1.7B to 4B models (lfm3:1.7b, bonsai-2:2b, qwen3:4b, qwen3.5:4b) passed the user's shorter, guarded LibreDB Studio SQL tool loop 30/30 (measured on the M5 Max, libredb-studio; [agent-harness-evidence.md](agent-harness-evidence.md)).

## Ollama notes

Ollama is not the main runtime in this knowledge base, but OpenCode supports it and the traps are specific.

- Ollama defaults to a 4k context on GPUs under 24 GiB, so set `OLLAMA_CONTEXT_LENGTH=65536` or more ([Ollama context docs](https://docs.ollama.com/context-length)).
- OpenCode's Ollama discovery takes `limit.context` from the GGUF's trained maximum (`model_info.*.context_length`), not from `OLLAMA_CONTEXT_LENGTH` ([ollama.ts](https://github.com/anomalyco/opencode/blob/v2.3.0/packages/core/src/plugin/provider/ollama.ts)), so set `limit.context` by hand or OpenCode never compacts while Ollama truncates.
- Ollama v0.40.16 `/v1` reads only `max_tokens`, while OpenCode sends `max_completion_tokens` by default ([openai.go](https://github.com/ollama/ollama/blob/v0.40.16/openai/openai.go)): set `compatibility.maxTokensField` to `max_tokens`.
- Ollama `/v1` ignores `tool_choice` ([openai.go](https://github.com/ollama/ollama/blob/v0.40.16/openai/openai.go)).
- Pull explicit quant tags such as `gemma4:26b-a4b-it-qat` (Q4_0, 13.45 GiB plus a 1.11 GiB projector); the default `gemma4:26b` is Q4_K_M plus a draft layer and does not fit 16 GB ([Ollama gemma4 tags](https://ollama.com/library/gemma4/tags)).
- Measured on 2026-11-16 with `ollama/ollama:0.40.16`: qwen3.5:9b, gemma4:12b, ornith:9b, ministral-3:8b and qwen2.5:7b ran fully on the GPU and passed the OpenCode write-and-read session through a provider of the shape below, using the `#none` variant for the first two ([test config](../research/2026-11-16/opencode-test-config-v3.json), [records](../research/2026-11-16/README.md)).
- A fresh Ollama container compiles CUDA kernels for this GPU on first use: model load took 15-29 s and the first request 15-24 s more, and with the saved cache mounted the gemma4:12b load fell to 3.7 s and its first request to 0.5 s (measured, [record](../research/2026-11-16/ollama-gemma4-12b/jit-cache-test.txt)).
  Keep the cache in a named volume, for example `-v ollama-nv:/root/.nv`, or keep the container instead of recreating it.
- Ollama 0.40.16 runs these tags through a bundled `llama-server` with the CUDA 13 backend, started with `--no-jinja --chat-template chatml`, so Ollama renders the chat template and parses tool calls itself; for gemma4:12b it also turns on MTP drafting (`--spec-type draft-mtp`) (read from the server logs).
- Some models send `write` and `read` as parallel calls in one step, and OpenCode then runs the read before the file exists: qwen2.5:7b on Ollama and Nemotron 3.5 Bolt on llama.cpp both had to read the file again (measured).

This custom-provider version validates with 0 errors against the same schema (2026-11-15):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "ollama-local/gemma-4-26b",
  "plugins": ["-opencode.provider.ollama"],
  "providers": {
    "ollama-local": {
      "name": "Ollama (Docker)",
      "package": "@opencode/ai/providers/openai-compatible",
      "settings": { "baseURL": "http://127.0.0.1:11434/v1", "headerTimeout": 600000 },
      "compatibility": { "reasoningField": "reasoning", "maxTokensField": "max_tokens" },
      "models": {
        "gemma-4-26b": {
          "modelID": "gemma4:26b-a4b-it-qat",
          "capabilities": { "tools": true, "input": ["text"], "output": ["text"], "reasoning": { "supported": true, "effort": ["none", "high"] } },
          "limit": { "context": 65536, "output": 16384 }
        }
      }
    }
  }
}
```

If you ever point OpenCode at Unsloth Studio instead, start it with `--disable-tools`, or Studio runs its own server-side tools and swallows OpenCode's tool calls ([Unsloth OpenCode guide](https://unsloth.ai/docs/integrations/opencode)).

## Known OpenCode issues

All four issues were open on 2026-11-15, and both fix PRs were unmerged ([#60219](https://github.com/anomalyco/opencode/pull/60219), [#58053](https://github.com/anomalyco/opencode/pull/58053)).

| Issue | Effect | Affects this setup | Mitigation |
|---|---|---|---|
| [#55378](https://github.com/anomalyco/opencode/issues/55378) | A turn that ends with reasoning only (no text, no tool call, finish `stop`) is settled silently and the agent stops | Yes, mostly with thinking on; also reported with vLLM behind LiteLLM | Type "continue"; the `#none` variant reduces exposure (inferred) |
| [#50814](https://github.com/anomalyco/opencode/issues/50814) | After compaction, earlier tool calls are summarised as `[Assistant tool call]: name(input)` text and some models copy that format, so tools never run | Yes, on any small window | Start a new session; a larger window means fewer compactions |
| [#60202](https://github.com/anomalyco/opencode/issues/60202) | V2 omits `tool_choice` on normal steps | No for vLLM v0.33.0 used directly: it sets `tool_choice` to `auto` when tools are present ([protocol.py L988-991](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/entrypoints/openai/chat_completion/protocol.py#L988-L991)); gateways in front of vLLM may be affected | If needed, a model `body` with `"tool_choice": "auto"`, plus a separate entry without it for title and compaction, because vLLM rejects `tool_choice` without tools |
| [#58019](https://github.com/anomalyco/opencode/issues/58019) | The subagent tool schema pattern `^ses` gives HTTP 400 on llama.cpp | No for b12407 and later: llama.cpp [PR #26939](https://github.com/ggml-org/llama.cpp/pull/26939) (merged 2026-08-20) falls back on unsupported regex; not tested here | On an older build, deny the subagent tool for the build agent ([permissions docs](https://opencode.ai/docs/permissions/)) |

One untested risk: OpenCode can send `tools: []` when tool history exists but no tool is active, for example in the compaction fallback, and vLLM v0.33.0 rejects an empty tools array with HTTP 400 ([protocol.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/entrypoints/openai/chat_completion/protocol.py)).
Runtime-side issues that hit OpenCode sessions, such as llama.cpp prefix-cache misses on Qwen3.5+ hybrid models and Qwen3.6 tool calls leaking into content, are tracked in [llamacpp/README.md](../llamacpp/README.md).

## Trying a model remotely first with OpenRouter

Before downloading many gigabytes, you can run the same OpenCode agent flow against a hosted copy of a pick through OpenRouter.
OpenCode has a native OpenRouter provider: provide `OPENROUTER_API_KEY` and use the full OpenRouter model ID after the `openrouter/` prefix ([providers docs](https://opencode.ai/docs/providers/)).

| Pick | OpenRouter model ID | Hosted context | Notes (2026-11-15) |
|---|---|---|---|
| Qwen3.6-35B-A3B | `qwen/qwen3.6-35b-a3b` | 262144 | 12 endpoints; CoreWeave and SiliconFlow do not list tools |
| Qwen3.8-27B | `qwen/qwen3.8-27b` | listing says 1,024,000; 18 of 23 endpoints serve 262144, one serves 65536 | |
| Gemma 4 26B-A4B | `google/gemma-4-26b-a4b-it` | 262144 | Parasail, CoreWeave and SiliconFlow do not list tools |
| gpt-oss-20b | `openai/gpt-oss-20b` | 131072 | SiliconFlow fp8 does not list tools |
| Qwen3-Coder-Next | `qwen/qwen3-coder-next` | 262144 | One provider only |
| Nemotron 3.5 Lightning (honorable) | `nvidia/nemotron-3.5-lightning` | 262144 | |
| GLM-4.7-Flash (honorable) | `z-ai/glm-4.7-flash` | 200000 | Two endpoints |
| Qwen3.5-9B (honorable) | `qwen/qwen3.5-9b` | 262144 | |

Source for the table: the [OpenRouter models API](https://openrouter.ai/api/v1/models) and per-model endpoint listings such as [qwen3.6-35b-a3b endpoints](https://openrouter.ai/api/v1/models/qwen/qwen3.6-35b-a3b/endpoints), read and re-checked on 2026-11-15; prices are live and move, so check them there.
Ornith-1.5-9B, Gemma 4 12B, DeepSeek-R1-0528-Qwen3-8B, KAT-Coder-V2.5-Dev and Laguna XS 2.1 have no OpenRouter endpoint, so they can only be tried locally.

Treat a hosted run as an upper bound, not a preview.
OpenRouter endpoints serve BF16, FP8 or FP4 weights on datacenter GPUs and report their quantization per endpoint ([provider selection docs](https://openrouter.ai/docs/guides/routing/provider-selection)), so a local Q3 or Q4 GGUF can only match or trail the hosted quality.
Hosted speed says nothing about this laptop, and the hosted chat template and tool parser are not yours, so a good hosted run does not validate the local vLLM or llama.cpp config.

### Privacy trade-off

Sending a session to OpenRouter sends your code to OpenRouter and to a third-party inference provider.
OpenRouter does not store prompts or responses unless you opt in to logging or to its use of inputs and outputs, apart from CSAM review and legal preservation ([data collection docs](https://openrouter.ai/docs/guides/privacy/data-collection)).
It does keep request metadata such as token counts and latency ([data collection docs](https://openrouter.ai/docs/guides/privacy/data-collection)).
Each provider has its own retention and training policy; endpoints with Safety Retention may keep prompts and responses from any request for abuse prevention, without opt-in ([provider logging docs](https://openrouter.ai/docs/guides/privacy/provider-logging), [ZDR docs](https://openrouter.ai/docs/guides/features/zdr)).
Since 2026-10-18 Google requires Safety Retention for Google models on Vertex, which can retain flagged prompts for up to 90 days ([ZDR docs](https://openrouter.ai/docs/guides/features/zdr)).

Settings that limit this:

- In the account [privacy settings](https://openrouter.ai/settings/privacy), disallow providers that may train on your data; paid and free models have separate settings ([provider logging docs](https://openrouter.ai/docs/guides/privacy/provider-logging)).
- In the same settings, enforce ZDR and choose **ZDR only** to exclude Safety Retention endpoints; self-serve accounts allow Safety Retention unless you choose ([ZDR docs](https://openrouter.ai/docs/guides/features/zdr)).
- Per request, `provider.data_collection: "deny"` skips providers that may store data and `provider.zdr: true` routes only to ZDR endpoints ([provider selection docs](https://openrouter.ai/docs/guides/routing/provider-selection)).
- `provider.require_parameters: true` routes only to providers that support every parameter in the request, which keeps tool calls away from endpoints without tools ([provider selection docs](https://openrouter.ai/docs/guides/routing/provider-selection)).
- Avoid `:free` variants for code, and never send a confidential or proprietary repository; keep those on the local endpoints.

In OpenCode v2.3.0 the OpenRouter package treats every non-connection key in provider `settings` as a provider option and copies `provider` into the request body ([settings.ts](https://github.com/anomalyco/opencode/blob/v2.3.0/packages/ai/src/providers/settings.ts), [openrouter.ts](https://github.com/anomalyco/opencode/blob/v2.3.0/packages/ai/src/providers/openrouter.ts)).
This config validates with 0 errors against the schema (2026-11-15); that the routing object reaches OpenRouter is read from the source, not tested with a request:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "openrouter/qwen/qwen3.6-35b-a3b",
  "providers": {
    "openrouter": {
      "settings": {
        "provider": { "data_collection": "deny", "zdr": true, "require_parameters": true }
      }
    }
  }
}
```

Keep this in a separate config or project from the local one, so a remote default never picks up a private repository by accident.

## Sources

- OpenCode v2.3.0 release: https://github.com/anomalyco/opencode/releases/tag/v2.3.0
- OpenCode config schema: https://opencode.ai/config.json
- OpenCode docs: [config](https://opencode.ai/docs/config/), [providers](https://opencode.ai/docs/providers/), [models](https://opencode.ai/docs/models/), [migration from V1](https://opencode.ai/docs/migrate-v1/), [compaction](https://opencode.ai/docs/compaction/), [permissions](https://opencode.ai/docs/permissions/)
- OpenCode v2.3.0 source: [vllm.ts](https://github.com/anomalyco/opencode/blob/v2.3.0/packages/core/src/plugin/provider/vllm.ts), [local.ts](https://github.com/anomalyco/opencode/blob/v2.3.0/packages/core/src/plugin/provider/local.ts), [ollama.ts](https://github.com/anomalyco/opencode/blob/v2.3.0/packages/core/src/plugin/provider/ollama.ts), [variant.ts](https://github.com/anomalyco/opencode/blob/v2.3.0/packages/core/src/variant.ts), [openrouter.ts](https://github.com/anomalyco/opencode/blob/v2.3.0/packages/ai/src/providers/openrouter.ts)
- OpenCode issues: [#55378](https://github.com/anomalyco/opencode/issues/55378), [#50814](https://github.com/anomalyco/opencode/issues/50814), [#60202](https://github.com/anomalyco/opencode/issues/60202), [#58019](https://github.com/anomalyco/opencode/issues/58019)
- vLLM v0.33.0 request protocol: https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/entrypoints/openai/chat_completion/protocol.py
- llama.cpp server README at b12418: https://github.com/ggml-org/llama.cpp/blob/b12418/tools/server/README.md and [PR #26939](https://github.com/ggml-org/llama.cpp/pull/26939)
- Ollama: [OpenCode integration](https://docs.ollama.com/integrations/opencode), [context length](https://docs.ollama.com/context-length), [thinking](https://docs.ollama.com/capabilities/thinking), [openai.go v0.40.16](https://github.com/ollama/ollama/blob/v0.40.16/openai/openai.go)
- Unsloth: [OpenCode guide](https://unsloth.ai/docs/integrations/opencode), [Qwen3-Coder-Next page](https://unsloth.ai/docs/models/qwen3-coder-next)
- Qwen3.8-27B chat template: https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/chat_template.jinja
- Artificial Analysis Qwen3.8 27B page: https://artificialanalysis.ai/models/qwen3-8-27b
- OpenRouter: [models API](https://openrouter.ai/api/v1/models), [data collection](https://openrouter.ai/docs/guides/privacy/data-collection), [provider logging](https://openrouter.ai/docs/guides/privacy/provider-logging), [ZDR](https://openrouter.ai/docs/guides/features/zdr), [provider selection](https://openrouter.ai/docs/guides/routing/provider-selection)
- Source credibility and the user-provided source reviews: [sources.md](sources.md)

## Maintenance

Before relying on this page, re-check:

- The installed version with `opencode --version`; on any version other than v2.3.0, re-read the migration guide and re-validate the config.
- The config against the live schema, for example with `python3 -I -c 'import json,sys,jsonschema; jsonschema.Draft202012Validator(json.load(open(sys.argv[1]))).validate(json.load(open(sys.argv[2])))' schema.json opencode.json` after downloading [https://opencode.ai/config.json](https://opencode.ai/config.json) to `schema.json`.
- The state of [#55378](https://github.com/anomalyco/opencode/issues/55378), [#50814](https://github.com/anomalyco/opencode/issues/50814), [#60202](https://github.com/anomalyco/opencode/issues/60202) and [#58019](https://github.com/anomalyco/opencode/issues/58019), and whether `vllm.ts` still forces tools off.
- That every `modelID` and `limit.context` still matches the server commands in [vllm/README.md](../vllm/README.md) and [llamacpp/README.md](../llamacpp/README.md).
- OpenRouter availability, tool support per endpoint and the privacy settings, which change without notice.
- Re-measure the prompt overhead when the allowed tools, `AGENTS.md` or MCP servers change; the measured 3,482 to 4,372 tokens cover only the test config with shell, web, task and skill tools denied.
