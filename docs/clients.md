# Hermes Agent, Pi, DeepSeek Harness and Kimi Code CLI with the local servers

Last verified: 2026-11-18.

`./llmctl up <model>` writes one llmctl entry into Hermes Agent, one into Pi, one block of rows into DeepSeek Harness (dsh), and two tables into Kimi Code CLI's `config.toml`, `[providers.llmctl]` and `[models."llmctl/<served name>"]`, with Kimi's `default_model` when you ask for it, so each reaches the model llmctl runs with its context, output cap and default thinking, without a manual config edit.
This document shows those entries, how each client thinks with them, what llmctl checks, and what was measured; which files llmctl writes and when, `LLMCTL_CLIENTS`, `--set-default` and the refusals are in [llmctl.md](llmctl.md#clients), and OpenCode is in [opencode.md](opencode.md).

## Versions

- [Hermes Agent v0.23.1](https://github.com/NousResearch/hermes-agent/tree/v0.23.1): the tag `v0.23.1` is commit `3b421578588243f920886e6fe79a6faa87ea026e`, and every Hermes source file cited here is [at that commit](https://github.com/NousResearch/hermes-agent/tree/3b421578588243f920886e6fe79a6faa87ea026e).
- [Pi 1.5.2 on npm](https://www.npmjs.com/package/@earendil-works/pi-coding-agent/v/1.5.2) (`@earendil-works/pi-coding-agent`; that page answers scripts with HTTP 403, so a script checks the [registry record](https://registry.npmjs.org/@earendil-works/pi-coding-agent/1.5.2) instead) and [its source at v1.5.2](https://github.com/earendil-works/pi/tree/v1.5.2/packages/coding-agent), tag `v1.5.2` = commit `b4afdddf44e059c67ccf5097e5317d913521b470`; the `dist/` and `docs/` files cited here are in the 1.5.2 npm package as published, and the `packages/ai/src/` file is at that tag.
- [DeepSeek Harness 0.2.2-rc.3 on npm](https://www.npmjs.com/package/@deepseek-ai/dsh/v/0.2.2-rc.3) (`@deepseek-ai/dsh`, dist-tag `next`; a script checks [its registry record](https://registry.npmjs.org/@deepseek-ai%2fdsh/0.2.2-rc.3)) and [the repository at commit `38a6a5c965aa99ab525161efb7cc5cf1fe62fced`](https://github.com/deepseek-ai/deepseek-harness/tree/38a6a5c965aa99ab525161efb7cc5cf1fe62fced), tag `dsh-v0.2.2-rc.3`; it needs Node 24.9 or newer (24.22.1 here) and resolves `@earendil-works/pi-ai` 1.5.2 from `^1.3.0` without a shrinkwrap.
  Install it with `npm install -g @deepseek-ai/dsh@0.2.2-rc.3`, because the `latest` dist-tag is still 0.2.2-rc.1 and every release is a developer preview (`runtime-1`).
- Kimi Code CLI 2.5.3, the native executable `~/.kimi-code/bin/kimi`, a Node.js v24.15.0 single executable application (local-K01), published on npm as [`@moonshot-ai/kimi-code` 2.5.3](https://www.npmjs.com/package/@moonshot-ai/kimi-code/v/2.5.3) (a script checks [its registry record](https://registry.npmjs.org/@moonshot-ai%2fkimi-code/2.5.3)) from [the repository `MoonshotAI/kimi-code` at tag `@moonshot-ai/kimi-code@2.5.3`](https://github.com/MoonshotAI/kimi-code/tree/5af9942e9fb789b6a1530830575b3768499932e9).
  npm's `latest` is already 2.6.2, published on 2026-11-13 (docs-K007, docs-K008) (vendor-reported), and Kimi checks for a newer version at every interactive and `-p` start and installs it in the background (local-K41, docs-K161, docs-K164), so `export KIMI_CODE_NO_AUTO_UPDATE=1` in your shell profile to stay on 2.5.3; it stops the check itself, not only the install (docs-K163) (vendor-reported).
- OpenCode v2.3.0, as in [opencode.md](opencode.md).

These are the versions the entries were verified with on 2026-11-16, and for Kimi on 2026-11-17; any other version is unverified.
The request shapes below were recorded with the real clients against a recording stub server on 127.0.0.1, with no model behind it, and are labelled "(measured on the MSI Raider 18 Max HX against a recording stub, Hermes v0.23.1)" or "(measured on the MSI Raider 18 Max HX against a recording stub, Pi 1.5.2)".
The records, the scripts and the claim ids used below (E1 to E9, R02 and the others) are in [research/2026-11-16/llmctl-clients-stub/](../research/2026-11-16/llmctl-clients-stub/README.md); a request shape keeps the model, the top-level keys and the thinking fields of a request, never its messages.
Real sessions through real llama.cpp and vLLM servers, and the servers' replies to these request shapes, come from the opt-in e2e run of 2026-11-16 in [Verified on this laptop](#verified-on-this-laptop); [Maintenance](#maintenance) has its command.

## Hermes Agent

llmctl gives Hermes a profile of its own, `llmctl`, and never edits the default profile (`~/.hermes/config.yaml`) or any other Hermes file, apart from `active_profile` through `hermes profile use llmctl` when you ask for `--set-default`.
Use it with `hermes -p llmctl`, or make it the sticky default with `./llmctl up <model> --set-default`.

### The profile

Hermes' root is `~/.hermes` (with `HERMES_DATA_DIR_SUFFIX` appended when set) when `HERMES_HOME` is unset or lies inside that folder; otherwise it is `HERMES_HOME`, or two levels up when its parent folder is `profiles`, as Hermes' own `_hermes_root_of` decides ([hermes_constants.py:99-107](https://github.com/NousResearch/hermes-agent/blob/3b421578588243f920886e6fe79a6faa87ea026e/hermes_constants.py#L99-L107), the default at [57-64](https://github.com/NousResearch/hermes-agent/blob/3b421578588243f920886e6fe79a6faa87ea026e/hermes_constants.py#L57-L64)).
The profile is `<root>/profiles/llmctl/config.yaml`.
llmctl creates `profiles/` and `profiles/llmctl/` with mode 0700 when they are missing, writes the file as `config.yaml.llmctl-tmp` with mode 0600, and renames it into place.
Hermes treats a folder under `profiles/` as a profile only when it holds an identity marker such as `config.yaml`, `.env` or `SOUL.md` ([hermes_constants.py:399](https://github.com/NousResearch/hermes-agent/blob/3b421578588243f920886e6fe79a6faa87ea026e/hermes_constants.py#L399), [profiles.py:485-502](https://github.com/NousResearch/hermes-agent/blob/3b421578588243f920886e6fe79a6faa87ea026e/hermes_cli/profiles.py#L485-L502)), and the temporary name is none of them, so Hermes never sees the profile without `gateway.standalone`.

The file llmctl creates for qwen3.6-35b-a3b on llama.cpp:

```yaml
# Hermes profile for the model llmctl runs (docs/llmctl.md#clients).
# llmctl rewrites the block below on every ./llmctl up and keeps everything outside it.
# The keys model, providers, model_overrides, gateway and platform_toolsets belong to the block.
# >>> llmctl managed block: rewritten by llmctl on every start; edits here are lost >>>
model: {"provider": "custom:llmctl", "default": "local"}
providers: {"llmctl": {"name": "llmctl: qwen3.6-35b-a3b on llama.cpp", "api": "http://127.0.0.1:8080/v1", "transport": "chat_completions", "default_model": "local", "discover_models": false, "extra_body": {"max_tokens": 16384}, "models": {"local": {"context_length": 131072}}}}
model_overrides: {"custom": {"local": {"supports_reasoning": false}}}
gateway: {"standalone": true}
# <<< llmctl managed block <<<
_config_version: 53
```

On every later `up`, llmctl rewrites only the lines between the two marker lines and keeps every byte outside them, including keys that you or Hermes add.
The keys `model`, `providers`, `model_overrides`, `gateway` and `platform_toolsets` belong to the block for every recipe; `platform_toolsets: {"cli": []}` appears only for a chat-only recipe, so Hermes sends no tools (R22).
Each value is one JSON line, which Hermes reads as YAML: Hermes parses its config with ruamel in YAML 1.1 mode and rejects duplicate keys ([hermes_yaml.py:1-37](https://github.com/NousResearch/hermes-agent/blob/3b421578588243f920886e6fe79a6faa87ea026e/hermes_yaml.py#L1-L37)), and every value llmctl writes is a JSON string, number, boolean or object of them, kept ASCII.
All 19 generated configs loaded in Hermes' own loader to exactly the intended values, and `hermes -p llmctl config get <key> --json` returned the block values (E5).

`_config_version: 53` sits after the end marker for two reasons.
Without it, Hermes' boot maintenance reads the profile as unstamped, below its support floor of version 12, so it skips migrating the profile and logs that the config "predates version 12" and "can no longer be auto-migrated" ([post_update.py:81-93](https://github.com/NousResearch/hermes-agent/blob/3b421578588243f920886e6fe79a6faa87ea026e/hermes_cli/post_update.py#L81-L93), [config_migrations.py:23-28](https://github.com/NousResearch/hermes-agent/blob/3b421578588243f920886e6fe79a6faa87ea026e/hermes_cli/config_migrations.py#L23-L28) and [43-52](https://github.com/NousResearch/hermes-agent/blob/3b421578588243f920886e6fe79a6faa87ea026e/hermes_cli/config_migrations.py#L43-L52)); 53 is `DEFAULT_CONFIG["_config_version"]` at v0.23.1 ([config_defaults.py:2700](https://github.com/NousResearch/hermes-agent/blob/3b421578588243f920886e6fe79a6faa87ea026e/hermes_cli/config_defaults.py#L2700)), and a newer Hermes migrates it in place ([post_update.py:61-110](https://github.com/NousResearch/hermes-agent/blob/3b421578588243f920886e6fe79a6faa87ea026e/hermes_cli/post_update.py#L61-L110)), outside the block, where llmctl leaves it.
And when the block ended the file, `hermes config set` of a new key landed inside the block, where the next `up` would drop it; with the stamp last, new keys land after it (E4).

Generated block for qwen3.6-35b-a3b on llama.cpp (`launcher/tests/golden/clients/llamacpp/qwen3.6-35b-a3b.hermes.yaml`):

```yaml
model: {"provider": "custom:llmctl", "default": "local"}
providers: {"llmctl": {"name": "llmctl: qwen3.6-35b-a3b on llama.cpp", "api": "http://127.0.0.1:8080/v1", "transport": "chat_completions", "default_model": "local", "discover_models": false, "extra_body": {"max_tokens": 16384}, "models": {"local": {"context_length": 131072}}}}
model_overrides: {"custom": {"local": {"supports_reasoning": false}}}
gateway: {"standalone": true}
```

Generated block for gpt-oss-20b on vLLM (`launcher/tests/golden/clients/vllm/gpt-oss-20b.hermes.yaml`):

```yaml
model: {"provider": "custom:llmctl", "default": "gpt-oss-20b"}
providers: {"llmctl": {"name": "llmctl: gpt-oss-20b on vLLM", "api": "http://127.0.0.1:8000/v1", "transport": "chat_completions", "default_model": "gpt-oss-20b", "discover_models": false, "extra_body": {"max_tokens": 16384}, "models": {"gpt-oss-20b": {"context_length": 65536}}}}
model_overrides: {}
gateway: {"standalone": true}
```

Every llama.cpp recipe is served as `local`, so its block names the model `local`; a vLLM or SGLang recipe uses its served name.
`context_length` equals the recipe's context, and `extra_body.max_tokens` its `[clients] output`, so every request carries the output cap.
Hermes refuses a context below 64,000 tokens (R19), so llmctl refuses such a recipe for Hermes; every shipped recipe has 65,536 or more.
The golden files for all 29 recipes are under `launcher/tests/golden/clients/`, and the tests keep them equal to what llmctl writes.

Before it writes, llmctl refuses a profile it cannot change safely: one without the two marker lines (a profile llmctl did not create), damaged markers, an owned key set outside the block (Hermes' loader rejects a file with a duplicate key (E5), and a gateway then serves the profile on default settings (E3)), a top-level line that is not a plain `key:`, or Hermes' deleted-profile marker `<root>/profiles/.deleted/llmctl` ([hermes_constants.py:387-392](https://github.com/NousResearch/hermes-agent/blob/3b421578588243f920886e6fe79a6faa87ea026e/hermes_constants.py#L387-L392)).
It then checks its own result: the text before and after the block is unchanged byte for byte, and every block line parses back to the intended value.
No Hermes command runs during `up` to check the profile: `hermes -p llmctl config get` starts no daemon and makes no network call, but it writes into Hermes' tree (an install bootstrap record, a prune stamp, lock files and about 160 bytecode files) and takes about 1 s (E1), so the Hermes-side parse belongs to the e2e run.

### Thinking per type

The server sets each model's default thinking, and the block keeps Hermes from overriding it: for every type except `effort` it sets `supports_reasoning: false` under `model_overrides.custom.<model>`, and it never writes `agent.reasoning_effort` or `chat_template_kwargs`.
The override key must be `custom`, not `custom:llmctl`, which has no effect (R02, R03).
Hermes sends `chat_template_kwargs` through a provider entry's `extra_body`, which it merges into every request for that provider ([providers.md:1374-1402](https://github.com/NousResearch/hermes-agent/blob/3b421578588243f920886e6fe79a6faa87ea026e/website/docs/integrations/providers.md?plain=1#L1374-L1402)); llmctl's entry puts only `max_tokens` there, so a toggle model runs as the server starts it.

| Thinking type | Recipes | Main request | Your choice |
|---|---|---|---|
| toggle | Qwen3.6, Gemma 4, Groot, Nemotron 3.5, Mellum, Qwen3.5 | No `reasoning_effort` and no kwargs, so the server's default applies | None through llmctl's entry; use OpenCode or Pi to turn thinking on |
| effort | Qwen3.8-27B, gpt-oss-20b | `reasoning_effort: "medium"`, Hermes' default, which equals the server's | `hermes --reasoning <level>` sends that level (R12) |
| always | K2 Horizon, DeepSeek-R1 | No thinking field | Do not pass `--reasoning`: it still sends a level although the block sets `supports_reasoning: false` (R13), and these models think at the effort the server sets |
| none | Qwen3-Coder-Next | No thinking field | None |

Seven recipes wrote and read `hello.py` through these blocks with exactly these main requests, each with `max_tokens` equal to the recipe's output cap (E6, [shapes](../research/2026-11-16/llmctl-clients-stub/shapes/design/hermes-e6.jsonl), [runs](../research/2026-11-16/llmctl-clients-stub/hermes/design/hz/)) (measured on the MSI Raider 18 Max HX against a recording stub, Hermes v0.23.1).
An explicit `agent.reasoning_effort` in the profile, such as the `medium` of Hermes' default config, is sent to every model despite the override (R04).
So `up` and `status` warn when the profile sets it outside the block, and print the command that clears it: `hermes -p llmctl config set agent.reasoning_effort ""`; llmctl never edits it.

Hermes titles each session after its first exchange with a separate request that asks for a strict `json_schema` `response_format` with thinking off ([sessions.md:308-312](https://github.com/NousResearch/hermes-agent/blob/3b421578588243f920886e6fe79a6faa87ea026e/website/docs/user-guide/sessions.md?plain=1#L308-L312), [title_generator.py:119-123](https://github.com/NousResearch/hermes-agent/blob/3b421578588243f920886e6fe79a6faa87ea026e/agent/title_generator.py#L119-L123) and [514-531](https://github.com/NousResearch/hermes-agent/blob/3b421578588243f920886e6fe79a6faa87ea026e/agent/title_generator.py#L514-L531)).
In E6, `hermes chat -q` made one such request per run and `hermes -z` made none, and each one carried `reasoning_effort: "none"`, a `json_schema` `response_format` and the block's `max_tokens` (E6, [runs](../research/2026-11-16/llmctl-clients-stub/hermes/design/hq/)) (measured on the MSI Raider 18 Max HX against a recording stub, Hermes v0.23.1); interactive sessions were not recorded.
In the e2e, a request of that shape got HTTP 200 and a title that parses from llama.cpp with Qwen3.8-27B, K2 Horizon 7B and Qwen3.6-35B-A3B and from vLLM with gpt-oss-20b (probe A, [Verified on this laptop](#verified-on-this-laptop)); K2 Horizon 7B still reasoned, and vLLM's gpt-oss-20b recipe maps `none` to `low` through `--reasoning-effort-map` ([vllm/README.md](../vllm/README.md#flags-reference)).
In the first full e2e run of 2026-11-17, vLLM's K2 Horizon 7B recipe refused that request with HTTP 400 `Unsupported reasoning_effort 'none'. Supported values: high, medium, low.` ([record 214 of that run](../research/2026-11-17/llmctl-dsh-e2e-run1/records/214-13-k2-horizon-7b-vllm-probes-A)), so the recipe now carries the same `--reasoning-effort-map`; with it the title request got HTTP 200 and a title that parses ([check](../research/2026-11-17/vllm-k2-horizon-7b/effort-map-check.txt), the second run's probe A in [record 213](../research/2026-11-17/llmctl-dsh-e2e/records/213-13-k2-horizon-7b-vllm-probes-A)) (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17).
SGLang serves that title request through its grammar path, whose first request in v0.5.23 allocates NCCL memory and can crash a server that has too little GPU memory left; the SGLang runtime file's two NCCL variables move that allocation before ready, and `up`'s post-ready check refuses a start that leaves less free memory than the title request and a long prompt need ([sglang/README.md](../sglang/README.md#grammar-requests-and-nccl)).

Hermes and Pi sessions with K2 Horizon 7B on vLLM still fail whenever they replay an assistant tool turn that came back without reasoning.
The model's chat template requires one of `think`, `reasoning`, `reasoning_content`, `think_fast` or `think_faster` on every assistant message and otherwise raises `Assistant message is missing a thinking field` ([chat_template.jinja:958-959](https://huggingface.co/cyankiwi/K2-Horizon-7B-AWQ-INT4/blob/530d5ceccd8ea660ceaa2a9f3a4bcbb68ba44291/chat_template.jinja#L958-L959) of `cyankiwi/K2-Horizon-7B-AWQ-INT4` at the pinned revision), which vLLM returns as HTTP 400.
In the first full e2e run of 2026-11-17 that failed `hermes chat` and Pi (that run's [record 209](../research/2026-11-17/llmctl-dsh-e2e-run1/records/209-13-k2-horizon-7b-vllm-hermes-chat) and [record 210](../research/2026-11-17/llmctl-dsh-e2e-run1/records/210-13-k2-horizon-7b-vllm-pi)); in the second it failed `hermes -z` ([record 207](../research/2026-11-17/llmctl-dsh-e2e/records/207-13-k2-horizon-7b-vllm-hermes-z)) and Pi ([record 209](../research/2026-11-17/llmctl-dsh-e2e/records/209-13-k2-horizon-7b-vllm-pi)) while `hermes chat` passed ([record 208](../research/2026-11-17/llmctl-dsh-e2e/records/208-13-k2-horizon-7b-vllm-hermes-chat)), and OpenCode ([record 206](../research/2026-11-17/llmctl-dsh-e2e/records/206-13-k2-horizon-7b-vllm-opencode)) and dsh ([record 210](../research/2026-11-17/llmctl-dsh-e2e/records/210-13-k2-horizon-7b-vllm-dsh-headless)) passed in both runs (the first run's [record 207](../research/2026-11-17/llmctl-dsh-e2e-run1/records/207-13-k2-horizon-7b-vllm-opencode) and [record 211](../research/2026-11-17/llmctl-dsh-e2e-run1/records/211-13-k2-horizon-7b-vllm-dsh-headless)) (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, Hermes Agent v0.23.1, Pi 1.5.2, dsh 0.2.2-rc.3, 2026-11-17).
In the e2e of 2026-11-18 the same refusal failed `hermes chat` after a write and a read in the work folder on the vLLM recipe and on the new SGLang recipe ([record 321](../research/2026-11-18/llmctl-sglang-e2e/records/321-17-k2-horizon-7b-vllm-hermes-chat), [record 459](../research/2026-11-18/llmctl-sglang-e2e/records/459-24-k2-horizon-7b-sglang-hermes-chat)), and Pi and, for the first time, dsh on the vLLM recipe ([record 322](../research/2026-11-18/llmctl-sglang-e2e/records/322-17-k2-horizon-7b-vllm-pi), [record 323](../research/2026-11-18/llmctl-sglang-e2e/records/323-17-k2-horizon-7b-vllm-dsh-headless)), while `hermes -z` failed on both recipes because the model aimed `write_file` at the profile's `cache/scratch/hello.py`, which already existed, and Hermes refused to overwrite it ([record 320](../research/2026-11-18/llmctl-sglang-e2e/records/320-17-k2-horizon-7b-vllm-hermes-z), [record 458](../research/2026-11-18/llmctl-sglang-e2e/records/458-24-k2-horizon-7b-sglang-hermes-z)) (measured on the MSI Raider 18 Max HX, vLLM v0.33.0 and SGLang v0.5.23, Hermes Agent v0.23.1, Pi 1.5.2, dsh 0.2.2-rc.3, 2026-11-18).
So which Hermes session fails depends on whether the model's tool turn carried reasoning in that run; for Hermes and Pi use the llama.cpp K2 Horizon 7B recipe, whose `hermes -z`, `hermes chat` and Pi sessions passed in the second run and again on 2026-11-18 (records [029](../research/2026-11-17/llmctl-dsh-e2e/records/029-02-k2-horizon-7b-llamacpp-hermes-z), [030](../research/2026-11-17/llmctl-dsh-e2e/records/030-02-k2-horizon-7b-llamacpp-hermes-chat) and [031](../research/2026-11-17/llmctl-dsh-e2e/records/031-02-k2-horizon-7b-llamacpp-pi)), or see [vllm/models.md](../vllm/models.md#k2-horizon-7b-awq-int4).

### The gateway

`hermes chat` and the TUI run their sessions in the profile's gateway process and attach to it ([cli.md:175-176](https://github.com/NousResearch/hermes-agent/blob/3b421578588243f920886e6fe79a6faa87ea026e/website/docs/user-guide/cli.md?plain=1#L175-L176), [tui.md:298](https://github.com/NousResearch/hermes-agent/blob/3b421578588243f920886e6fe79a6faa87ea026e/website/docs/user-guide/tui.md?plain=1#L298)); a client that finds no gateway for its profile starts a detached one that exits when idle ([run_idle_exit.py:1-13](https://github.com/NousResearch/hermes-agent/blob/3b421578588243f920886e6fe79a6faa87ea026e/gateway/run_idle_exit.py#L1-L13)).
Without `gateway.standalone: true`, the host's multiplexing gateway, the one that serves your messaging platforms, serves every named profile that is not parked; with it, the host does not serve the profile, and the profile runs its own gateway ([multi-profile-gateways.md:133-136](https://github.com/NousResearch/hermes-agent/blob/3b421578588243f920886e6fe79a6faa87ea026e/website/docs/user-guide/multi-profile-gateways.md?plain=1#L133-L136), [189-190](https://github.com/NousResearch/hermes-agent/blob/3b421578588243f920886e6fe79a6faa87ea026e/website/docs/user-guide/multi-profile-gateways.md?plain=1#L189-L190)).
In a sandboxed multiplexing gateway, the profile as llmctl writes it was skipped and logged as standalone, while a profile with a duplicate key was served on default settings although its text said `standalone: true` (E3, [state](../research/2026-11-16/llmctl-clients-stub/hermes/design/mux/state.txt)).
Hermes calls `gateway.standalone` "a temporary compatibility shim" ([multi-profile-gateways.md:133-136](https://github.com/NousResearch/hermes-agent/blob/3b421578588243f920886e6fe79a6faa87ea026e/website/docs/user-guide/multi-profile-gateways.md?plain=1#L133-L136)) that will be removed once the gaps it covers are fixed, and advises against building new setups on it ([291-305](https://github.com/NousResearch/hermes-agent/blob/3b421578588243f920886e6fe79a6faa87ea026e/website/docs/user-guide/multi-profile-gateways.md?plain=1#L291-L305)), which is why the [Maintenance](#maintenance) list watches it; `hermes gateway migrate --multiplex` leaves standalone profiles alone ([gateway_migrate.py:127-129](https://github.com/NousResearch/hermes-agent/blob/3b421578588243f920886e6fe79a6faa87ea026e/hermes_cli/gateway_migrate.py#L127-L129)).

So `up`, `status` and `list` read `<root>/gateway_state.json`, where Hermes' gateway records its `gateway_state` and `served_profiles` ([status.py:41](https://github.com/NousResearch/hermes-agent/blob/3b421578588243f920886e6fe79a6faa87ea026e/gateway/status.py#L41), [1125-1130](https://github.com/NousResearch/hermes-agent/blob/3b421578588243f920886e6fe79a6faa87ea026e/gateway/status.py#L1125-L1130)), and when the gateway is `running` and its `served_profiles` lists `llmctl`, they print:

```text
warning: Hermes' gateway serves the profile llmctl (<root>/gateway_state.json), so its sessions run inside your messaging gateway; this Hermes may no longer honour gateway.standalone (docs/clients.md#hermes-agent)
```

When you see it, stop using `hermes -p llmctl` with that Hermes version until the profile is standalone again, and check the shim in the Hermes release notes; a missing state file prints nothing, and an unreadable one prints a note.
llmctl never starts, stops, parks or migrates a gateway; with `gateway.standalone: true`, the host never parks or serves the profile, and `hermes -p llmctl gateway stop` and `start` act on the profile's own gateway process ([multi-profile-gateways.md:236-251](https://github.com/NousResearch/hermes-agent/blob/3b421578588243f920886e6fe79a6faa87ea026e/website/docs/user-guide/multi-profile-gateways.md?plain=1#L236-L251)).
The profile's own daemon, when a client started it, exits once it has served no client and had nothing to do for 600 s, Hermes' default `gateway.unmanaged_idle_exit_seconds` ([config_defaults.py:2088-2089](https://github.com/NousResearch/hermes-agent/blob/3b421578588243f920886e6fe79a6faa87ea026e/hermes_cli/config_defaults.py#L2088-L2089), [run_idle_exit.py:1-13](https://github.com/NousResearch/hermes-agent/blob/3b421578588243f920886e6fe79a6faa87ea026e/gateway/run_idle_exit.py#L1-L13)); a gateway under a supervisor, or one started without `--idle-exit` such as `hermes gateway run` by hand, never arms that exit ([run_idle_exit.py:15-17](https://github.com/NousResearch/hermes-agent/blob/3b421578588243f920886e6fe79a6faa87ea026e/gateway/run_idle_exit.py#L15-L17)).
While the profile's own gateway runs, its cron ticker visits the cron jobs of its own profile, the default profile and every live named profile that is neither standalone nor parked, skipping each one that another live gateway owns ([run_bootstrap.py:782-787](https://github.com/NousResearch/hermes-agent/blob/3b421578588243f920886e6fe79a6faa87ea026e/gateway/run_bootstrap.py#L782-L787) and [808-809](https://github.com/NousResearch/hermes-agent/blob/3b421578588243f920886e6fe79a6faa87ea026e/gateway/run_bootstrap.py#L808-L809), [run.py:1649-1656](https://github.com/NousResearch/hermes-agent/blob/3b421578588243f920886e6fe79a6faa87ea026e/gateway/run.py#L1649-L1656) and [1666-1690](https://github.com/NousResearch/hermes-agent/blob/3b421578588243f920886e6fe79a6faa87ea026e/gateway/run.py#L1666-L1690), [profiles.py:1099-1105](https://github.com/NousResearch/hermes-agent/blob/3b421578588243f920886e6fe79a6faa87ea026e/hermes_cli/profiles.py#L1099-L1105)), and a runnable cron job in a home it ticks counts as work ([run_idle_exit.py:88-108](https://github.com/NousResearch/hermes-agent/blob/3b421578588243f920886e6fe79a6faa87ea026e/gateway/run_idle_exit.py#L88-L108)), so with Hermes cron jobs a daemon that a client started may not exit at all (read in the source, not measured).

### The sticky profile

`./llmctl up <model> --set-default` runs `hermes profile use llmctl` and then checks that `<root>/active_profile` reads `llmctl`.
That command writes `active_profile` through a temporary file and a rename ([profiles.py:1956-1968](https://github.com/NousResearch/hermes-agent/blob/3b421578588243f920886e6fe79a6faa87ea026e/hermes_cli/profiles.py#L1956-L1968)), touches a prune stamp and two lock files, starts no daemon and makes no network call, in about 0.7 s (E2).
From then on, every plain `hermes` command, `hermes gateway` and `hermes config` included, acts on the profile llmctl; `up` prints that, and `Hermes default profile: <previous> -> llmctl`.
Switch back with `hermes profile use default` (or the previous profile `up` named), which deletes `active_profile` (`stick`).
The runtime lease files the pruner may remove are skipped while a live process holds them ([runtime_state.py:247-270](https://github.com/NousResearch/hermes-agent/blob/3b421578588243f920886e6fe79a6faa87ea026e/hermes_cli/runtime_state.py#L247-L270)).
A gateway that runs as a supervised child (`HERMES_SUPERVISED_CHILD=1`) never reads `active_profile`, so a systemd-managed messaging gateway keeps its own profile ([main.py:451-478](https://github.com/NousResearch/hermes-agent/blob/3b421578588243f920886e6fe79a6faa87ea026e/hermes_cli/main.py#L451-L478), [main.py:508-537](https://github.com/NousResearch/hermes-agent/blob/3b421578588243f920886e6fe79a6faa87ea026e/hermes_cli/main.py#L508-L537), `stick`).
When `HERMES_HOME` names a profile folder in your shell, Hermes ignores the sticky profile there ([main.py:521-523](https://github.com/NousResearch/hermes-agent/blob/3b421578588243f920886e6fe79a6faa87ea026e/hermes_cli/main.py#L521-L523)); `up` and `status` then print a note to use `hermes -p llmctl`.

### Sessions and interactive paths

Hermes keeps a resumed session's model ([sessions.md:227](https://github.com/NousResearch/hermes-agent/blob/3b421578588243f920886e6fe79a6faa87ea026e/website/docs/user-guide/sessions.md?plain=1#L227)) and merges a provider's `extra_body`, the output cap included, into the agent when it builds it ([providers.md:1402](https://github.com/NousResearch/hermes-agent/blob/3b421578588243f920886e6fe79a6faa87ea026e/website/docs/integrations/providers.md?plain=1#L1402)), so a session that was open before an `up` may keep the old model, URL and output cap; start a new session after an `up` that changed the block (not tested).
The interactive `/reasoning` and `/model` commands inside an open session are not tested with these profiles; treat them as unverified.

### Where files land

Qwen3.6-35B-A3B, with thinking off as its recipe sets it, wrote `hello.py` outside the session's working folder, into your home folder or into the profile folder `~/.hermes/profiles/llmctl/`, in all three Hermes sessions of the recorded e2e run and of the run before it, and in one of three in an earlier run that day (measured on the MSI Raider 18 Max HX, llama.cpp b12407, Hermes v0.23.1, 2026-11-16; [Verified on this laptop](#verified-on-this-laptop)).
K2 Horizon 7B did the same once, writing into the profile folder in the recorded run's `hermes -z` session (record 025) after writing into the working folder in the two runs before; Qwen3.8-27B and gpt-oss-20b always wrote into the working folder in Hermes.
Hermes accepted those paths: Qwen3.6's `hermes chat` record shows `write_file` with the absolute path `$HOME/.hermes/profiles/llmctl/hello.py` (record 053), while the other three models passed the relative path `hello.py` there, which Hermes resolved into the working folder (records 013, 026 and 041).
The same Qwen3.6 server wrote into the working folder through OpenCode and Pi, so this is the model's behaviour in Hermes, not llmctl's entry.
On 2026-11-18 Qwen3.6-35B-A3B did the same on its new SGLang recipe, whose `hermes -z` wrote `/tmp/hello.py` ([record 555](../research/2026-11-18/llmctl-sglang-e2e/records/555-28-qwen3.6-35b-a3b-sglang-hermes-z)), and on llama.cpp again, where `hermes -z` wrote `$HOME/hello.py` and `hermes chat` aimed at that `/tmp/hello.py`, which Hermes refused to overwrite ([record 575](../research/2026-11-18/llmctl-sglang-e2e/records/575-29-qwen3.6-35b-a3b-llamacpp-hermes-z), [record 576](../research/2026-11-18/llmctl-sglang-e2e/records/576-29-qwen3.6-35b-a3b-llamacpp-hermes-chat)) (measured on the MSI Raider 18 Max HX, llama.cpp b12407 and SGLang v0.5.23, Hermes Agent v0.23.1, 2026-11-18); K2 Horizon 7B's `hermes -z` wrote into the profile's `cache/scratch` folder on vLLM and SGLang ([Thinking per type](#thinking-per-type)).
With Qwen3.6, and less often K2 Horizon 7B, in Hermes, check where files land, or name the target folder in the prompt (not tested).

## Pi

llmctl owns the provider `llmctl` in Pi's `models.json` and the `llmctl/` keys of `modelThinkingLevels` in `settings.json`, both in Pi's agent folder: `PI_CODING_AGENT_DIR` when set, else `~/.pi/agent` (`dist/config.js:510-516`).
Use it with `pi --model llmctl/<served name>`, which `up` prints (`llmctl/local` for every llama.cpp recipe), or make it Pi's default with `./llmctl up <model> --set-default`.
With K2 Horizon 7B, use the llama.cpp recipe in Pi: on vLLM the model's chat template rejects Pi's replayed tool turns that carry no reasoning field ([Thinking per type](#thinking-per-type)).

### The entry

llmctl replaces `providers.llmctl` in place, or appends it, and keeps every other provider, key and their order; the file is written with 2-space indentation, and a parse of the result without `llmctl` must equal the old file without it.
Pi itself accepts `//` comments and trailing commas in `models.json` (`dist/utils/json.js:1-6`, `dist/core/model-config.js:139`), but llmctl refuses them, with the advice to remove them or leave Pi out with `LLMCTL_CLIENTS`.

Generated `models.json` for qwen3.6-35b-a3b on llama.cpp (`launcher/tests/golden/clients/llamacpp/qwen3.6-35b-a3b.pi.json`):

```json
{
  "providers": {
    "llmctl": {
      "name": "llmctl",
      "baseUrl": "http://127.0.0.1:8080/v1",
      "api": "openai-completions",
      "apiKey": "no-key-required",
      "compat": {
        "supportsStore": false,
        "supportsDeveloperRole": false,
        "maxTokensField": "max_tokens"
      },
      "models": [
        {
          "id": "local",
          "name": "qwen3.6-35b-a3b on llama.cpp (llmctl)",
          "reasoning": true,
          "input": [
            "text"
          ],
          "contextWindow": 131072,
          "maxTokens": 16384,
          "thinkingLevelMap": {
            "minimal": null,
            "low": null,
            "high": null
          },
          "compat": {
            "thinkingFormat": "chat-template",
            "chatTemplateKwargs": {
              "enable_thinking": {
                "$var": "thinking.enabled"
              }
            }
          }
        }
      ]
    }
  }
}
```

Generated `models.json` for gpt-oss-20b on vLLM (`launcher/tests/golden/clients/vllm/gpt-oss-20b.pi.json`):

```json
{
  "providers": {
    "llmctl": {
      "name": "llmctl",
      "baseUrl": "http://127.0.0.1:8000/v1",
      "api": "openai-completions",
      "apiKey": "no-key-required",
      "compat": {
        "supportsStore": false,
        "supportsDeveloperRole": false,
        "maxTokensField": "max_tokens"
      },
      "models": [
        {
          "id": "gpt-oss-20b",
          "name": "gpt-oss-20b on vLLM (llmctl)",
          "reasoning": true,
          "input": [
            "text"
          ],
          "contextWindow": 65536,
          "maxTokens": 16384,
          "thinkingLevelMap": {
            "off": null,
            "minimal": null,
            "low": "low",
            "medium": "medium",
            "high": "high"
          },
          "compat": {
            "thinkingFormat": "openai",
            "supportsReasoningEffort": true
          }
        }
      ]
    }
  }
}
```

The provider `compat` is never left out: without it Pi sends `max_completion_tokens`, `store` and the `developer` role to the local server (item 8, [record](../research/2026-11-16/llmctl-clients-stub/pi/phase1/M-baseline-nocompat/)).
`apiKey` is the placeholder `no-key-required`, because the servers here take no key.
The entry uses only the keys of Pi 1.5.2's `ModelDefinitionSchema` (`dist/core/model-config.js:32-49`), so a stricter schema later does not reject it.

### Thinking levels

Pi 1.5.2's `models.json` has no per-model default level (`ModelDefinitionSchema` in `dist/core/model-config.js:32-49` has no such key), and virtual models need an extension (`docs/virtual-models.md` in the package), so llmctl writes the recipe's level to `settings.json` as `modelThinkingLevels["llmctl/<served name>"]` on every `up` where it differs, and removes the other `llmctl/` keys.
A level you set for `llmctl/local` in Pi's settings is reset on the next `up`, because every llama.cpp recipe shares that id; your `defaultThinkingLevel` and every other provider's key stay.
When a session starts, an explicit `--thinking` or `:level` suffix wins, then a resumed session's recorded level (an older session with messages but no recorded level takes `defaultThinkingLevel` or medium), then `modelThinkingLevels`, then `defaultThinkingLevel`, then medium (`dist/core/sdk.js:121-143`); a model switch inside a session takes an explicit level passed with it first, then the per-model level, then `defaultThinkingLevel`, then the session's current level (`dist/core/agent-session.js:2259-2271`).

| Thinking type | Level llmctl writes | Request at that level | Other levels, as the entry maps them |
|---|---|---|---|
| toggle, off by default | `off` | `chat_template_kwargs {"enable_thinking": false}` | `--thinking medium` turns it on (sends `true`); Nemotron 3.5 Bolt's entry adds `"medium_effort": true` at `medium` and keeps `high` as full thinking; the other levels are unsupported |
| toggle, on by default (Mellum) | `medium` | `{"enable_thinking": true}` (not recorded) | `off` turns it off; the other levels are unsupported |
| effort | `medium` | `reasoning_effort: "medium"` | Each level from `off` to `high` maps to the effort of the same name (`off` to `none`) when the recipe lists it, and is unsupported otherwise |
| always | `high` | No thinking field | Every other level is unsupported, so Pi clamps it to `high` |
| none | `off` | No thinking field | None |

The third column, with `max_tokens` equal to the output cap, came from the generated entries at the written levels for every type but Mellum's, and `--thinking medium` turned Qwen3.6's thinking on (E8, [shapes](../research/2026-11-16/llmctl-clients-stub/shapes/design/pi-e8.jsonl), [runs](../research/2026-11-16/llmctl-clients-stub/pi/design/)) (measured on the MSI Raider 18 Max HX against a recording stub, Pi 1.5.2).
Mellum's request and the rest of the fourth column are not recorded; they follow from the generated entries under `launcher/tests/golden/clients/` and Pi's documented rules: `null` in `thinkingLevelMap` marks a level unsupported, Pi clamps an unsupported level before it builds the request, `samplingParamsByThinkingLevel` adds fields at one level, and `$var` `thinking.enabled` fills in Pi's own on or off value (`docs/models-configuration.md:37` and `126`, `docs/models.md:116-138` in the package).
A chat-only recipe gets `pi --model llmctl/local --no-tools`, because Pi takes its tools from the command line and `settings.json`, not from a model entry (`dist/core/sdk.js:148`; `ModelDefinitionSchema` has no tools key).

### Validation

Before anything starts, and again once the server is ready but before llmctl takes Pi's settings lock, llmctl writes the candidate `models.json` (your file with llmctl's entry in place) into a new private temporary folder and runs `pi --offline --list-models llmctl` there with `PI_CODING_AGENT_DIR` set to it, `PI_OFFLINE=1`, `PI_SKIP_VERSION_CHECK=1` and `PI_TELEMETRY=0`; the folder is removed afterwards.
The entry passes when Pi lists the row `llmctl <served name>`; without it, `up` refuses and quotes Pi's stderr, because Pi drops a file that fails its schema as a whole (`dist/core/model-config.js:144-150`; Pi listed no models in E7 `neg-format`), and a default that names a model Pi no longer has falls back to another provider (S3).
When the row is there but Pi reports `errors loading models.json` (`dist/cli/list-models.js:25-28`), llmctl writes the file and warns that Pi reports problems outside the llmctl entry.
The exit status is not used: Pi exits 0 on schema errors and on a missing key, and a missing key prints nothing at all (E7, [records](../research/2026-11-16/llmctl-clients-stub/pi/design/pi-validate-all.txt)).
All 19 generated entries passed in about 0.2 s each, and with the `llmctl` filter Pi did not run another provider's `!command` key (E7, `cmdkey`).
llmctl runs `pi` from `PATH`, which works for the managed launcher and an npm install alike.
When `PI_MANAGED_INSTALL_ROOT` names the managed install that holds the running `pi`, every `pi` start, llmctl's check included, takes and releases `<root>/update.lock` and removes abandoned `staging/update-*` entries, and skips both while an update holds the lock; an npm-installed `pi` does neither (`dist/package-manager-cli.js:25-46` and `119-154`, `dist/main.js:505`).
`up --dry-run` runs no validation and says so.

`settings.json` is changed only while llmctl holds the lock folder `settings.json.lock` beside it, which it tries for 2 s.
Pi takes the same lock, a folder that the proper-lockfile code bundled into `pi` makes with `mkdir`, around each read and each write of `settings.json`, and tries it 10 times, 20 ms apart (`dist/core/settings-manager.js:126-178` and `262`; `getLockFile` and `acquireLock` in `dist/bundle/chunks/chunk-4KSTFUIK.js`).
Inside the lock, llmctl plans both Pi files again, runs Pi's check once more only when the `models.json` text differs from the one it just checked, and then writes `models.json` and `settings.json`.
When Pi holds the lock at that moment, or a crashed Pi left it behind, `up` exits with code 6, names the lock, and leaves both Pi files as they were, while OpenCode and Hermes are still written; llmctl never removes a lock it did not create.
That bundled code treats a lock older than 10 s as stale and replaces it (`stale:1e4` and `isLockStale` in the same file); when Pi does that while llmctl holds the lock, llmctl leaves Pi's lock alone, and after its writes `up` exits with code 6 with `<settings path>: Pi took over its settings lock while llmctl held it, so another writer may have changed settings.json; check it`, printing `Pi: not updated, see the error below` in place of Pi's lines.
Pi 1.5.2 also takes that lock when it reads `settings.json` at start, gives up after about 200 ms with only a warning (`Invalid settings file <path>: ...`), and starts with empty global settings and so no default provider, so a Pi that starts during llmctl's write of `settings.json` may start on another model (`dist/core/settings-manager.js:126-149`, `237-279` and `538-540`; `dist/core/settings-diagnostics.js:1-6`; `dist/main.js:584`).
When the `settings.json` write fails after `models.json` was written, llmctl puts `models.json` back as it was, or removes it when llmctl created it, so the two files keep naming the same model, and the exit 6 message says `<models path> was put back as it was`.
When `up` is interrupted there and that put-back fails too, it prints `error: putting <models path> back failed too (<models path>: <reason>), so models.json and settings.json now disagree about the llmctl model; fix them by hand or run up again` and exits with code 130.

### Defaults, sessions and timeouts

With `--set-default`, llmctl sets `defaultProvider` to `llmctl` and `defaultModel` to the served name; without it, when `defaultProvider` is already `llmctl`, it moves `defaultModel` to the running model.
That follow write matters: with `defaultModel` left at `local` after a vLLM `up`, Pi answered from another provider with exit 0 and an empty stderr (E8 `d6-stale`, S3), and the follow write fixed it (`d6-follow`).
A trusted project's `.pi/settings.json` overrides these keys inside that project (`dist/core/settings-manager.js:124` and `220`, `docs/settings.md:3` and `docs/configuration.md:3` in the package), and llmctl never reads or writes it.
When `enabledModels` is not empty and no entry selects `llmctl/<served name>`, a new Pi session started without `--model` uses the first entry instead of the default (`dist/main.js:435-455` and `715-718`).
While `defaultProvider` is `llmctl`, `up` and `status` then print `note: a new Pi session without --model starts with the first of enabledModels (<entry>) instead of llmctl/<served name>; add llmctl/<served name> to enabledModels or clear enabledModels`, and llmctl edits nothing.
llmctl counts an entry as selecting the model when it is `llmctl/<served name>` or the bare `<served name>`, either with an optional `:<level>` suffix, or a glob that matches either one, all without regard to case, as Pi's scope rules do (`dist/core/model-resolver.js:250-291`); Pi 1.5.2 also matches a plain entry as part of a model's id or name (`dist/core/model-resolver.js:184-187`), so the note can appear for an entry such as `loc` that Pi still resolves to `llmctl/local`.
A `--model llmctl/<id>` with an id the entry lacks only warns and sends that id (X2), so use the id `up` prints.

A Pi session that was open before an `up` keeps the entry it loaded, with its context, output cap and thinking fields, because Pi loads `models.json` once when it builds a session's runtime (`dist/core/model-config.js:116-155`, `dist/core/agent-session-services.js:53-62`).
In the TUI, opening `/model` reloads the file (`docs/models.md:67` in the package), and `/new` builds a new runtime that reads `models.json` and `settings.json` again (`dist/core/agent-session-runtime.js:147-173`, `dist/main.js:646-660`), as a restart does (read in the Pi source, not tested).
A session resumed with `--continue` after its model id disappeared falls back to the settings default silently (SESS2).
Pi's idle timeout for response headers and body is 300 s, and `"httpIdleTimeoutMs": 600000` in `settings.json` raises it to 600 s (`dist/core/http-dispatcher.js:3`, `docs/settings.md:126` in the package), which the long first prefill of a RAM-offloaded MoE model may need; llmctl does not write that setting.
Pi retries a failed request 3 times, 2 s, 4 s and 8 s apart (`dist/core/settings-defaults.js:17-20`, `dist/core/agent-session.js:3192-3226`, [retry.ts:149-153](https://github.com/earendil-works/pi/blob/v1.5.2/packages/ai/src/utils/retry.ts#L149-L153)), and it counts a refused connection as retryable ([retry.ts:67-88](https://github.com/earendil-works/pi/blob/v1.5.2/packages/ai/src/utils/retry.ts#L67-L88)), so a stopped server fails each prompt after about 14 s (estimate).

## DeepSeek Harness

llmctl points DeepSeek Harness (dsh) at the running model through one marked block of rows in dsh's home patch, and writes nothing else in dsh.
Use it with `dsh web` and pick `<id> on <runtime> (llmctl)` in the model picker, or make it dsh's default in every profile with `./llmctl up <model> --set-default`.
The evidence ids below (A1 to A4, B1-0 to B4-3, S1 to S11 and the others) are mapped to their files in [research/2026-11-16/llmctl-dsh-stub/](../research/2026-11-16/llmctl-dsh-stub/README.md), and the research ids (`configuration-4`, `runtime-17`, `extra-0` and the others) to the verdicts archived in [research-claims.json](../research/2026-11-16/llmctl-dsh-stub/research-claims.json), one line per id in that README's [Research claims](../research/2026-11-16/llmctl-dsh-stub/README.md#research-claims).
Lane B and S5 ran against a recording stub on 127.0.0.1 and are labelled "(measured on the MSI Raider 18 Max HX against a recording stub, dsh 0.2.2-rc.3)", except lane B's `real-*` cases, which went through a recording proxy to the llama.cpp server and are labelled "(measured on the MSI Raider 18 Max HX through a recording proxy to the llama.cpp server, dsh 0.2.2-rc.3)".
Lane A's rows pointed at the gemma-4-26b-a4b llama.cpp server running on 127.0.0.1:8080 ([rows](../research/2026-11-16/llmctl-dsh-stub/laneA/a2h-home-0.yml)), and its one vLLM session found no server on 127.0.0.1:8000 (`ECONNREFUSED`, [stderr](../research/2026-11-16/llmctl-dsh-stub/laneA/final-gptoss-headless.err)); lane A, S10 and S11 are labelled "(measured on the MSI Raider 18 Max HX, llama.cpp b12407, dsh 0.2.2-rc.3)".
S1 to S4d and S6 to S9 contacted no model server and are labelled "(measured on the MSI Raider 18 Max HX, dsh 0.2.2-rc.3, no model server)".
Each dsh source file cited below links to the repository at commit `38a6a5c965aa99ab525161efb7cc5cf1fe62fced` and is byte-identical to its copy in the 0.2.2-rc.3 npm packages installed here.
In those packages `packages/<group>/<name>` ships as `@deepseek-ai/dsh-<name>`, except that `client/ui-model-selection` and `api/session-controller` ship as `@deepseek-ai/dsh-client-ui-model-selection` and `@deepseek-ai/dsh-api-session-controller`; `apps/cli` is `@deepseek-ai/dsh` itself, and `vendor/include` is `@deepseek-ai/cordis-plugin-include` 1.1.0.

### The home patch

dsh composes, in order, its bundle patches, the profile's `profiles/<name>/cordis.patch.yml`, the home patch `$DSH_HOME/cordis.patch.yml` and any `--patch` overlays, and a later row replaces each key it sets, `config` included, whole ([app-boot/src/profile-context.ts:74-84](https://github.com/deepseek-ai/deepseek-harness/blob/38a6a5c965aa99ab525161efb7cc5cf1fe62fced/packages/boot/app-boot/src/profile-context.ts#L74-L84), `cordis-plugin-include` [src/patch.ts:91-105](https://github.com/deepseek-ai/deepseek-harness/blob/38a6a5c965aa99ab525161efb7cc5cf1fe62fced/vendor/include/src/patch.ts#L91-L105), read in the dsh source).
`$DSH_HOME` is `~/.dsh` when unset or blank; `~`, `~/` and `~\` are expanded with your `HOME`, and any other relative value is refused, because dsh resolves it against each command's working folder ([home-paths/src/index.ts:87-110](https://github.com/deepseek-ai/deepseek-harness/blob/38a6a5c965aa99ab525161efb7cc5cf1fe62fced/packages/util/home-paths/src/index.ts#L87-L110)).

Three places were tried.
A second pi-ai plugin instance for llmctl cannot mount, because exactly one pi-ai instance mounts per installation (A1, [stderr](../research/2026-11-16/llmctl-dsh-stub/laneA/a1-headless-run.err)).
A block in each profile patch does not survive: the Web UI owns that file and rewrites it on the first page load, on every picker change and on every Models page save, turns JSON lines into YAML and puts its own rows inside the block (A2-profile, [snapshots](../research/2026-11-16/llmctl-dsh-stub/laneA/a2p-3-after-add-provider.yml)).
The home patch works: dsh never writes it (A2-home, sha256 unchanged; `config-editor` [README.md:67](https://github.com/deepseek-ai/deepseek-harness/blob/38a6a5c965aa99ab525161efb7cc5cf1fe62fced/packages/boot/config-editor/README.md?plain=1#L67)), one file covers the web, headless, acp and sdk profiles (A4), and a running `dsh web` reloads it (A2-hmr) (measured on the MSI Raider 18 Max HX, llama.cpp b12407, dsh 0.2.2-rc.3).

A home patch that llmctl creates holds a header, the begin marker, one row per line and the end marker, with mode 0600:

```yaml
# dsh rows for the model llmctl runs (docs/llmctl.md#clients).
# llmctl rewrites the lines between the two markers on every ./llmctl up and keeps everything else in this file.
# While the block holds the agent-default-model line, the running model is dsh's default in every profile;
# delete that line and the acp line to give the default back to dsh's model picker.
# >>> llmctl managed block: rewritten by llmctl on every start; edits here are lost >>>
<rows>
# <<< llmctl managed block <<<
```

Each row is `- ` followed by one JSON object, which dsh's js-yaml 4 reads as YAML 1.2 flow, so `"off": null` stays a string key (A2-json).
The block holds up to three rows, in this order:

| Row | Content | When it is in the block |
|---|---|---|
| `llm-pi-ai` | `{"providers": {"llmctl": <route>}}` as its `config` ([Rows per thinking type](#rows-per-thinking-type)) | Always, while dsh is an active client |
| `agent-default-model` | `{"provider": "llmctl", "model": <served name>}`, never a `reasoningEffort` | While llmctl holds dsh's default ([The default model and the picker](#the-default-model-and-the-picker)) |
| `acp` | `{"provider": "llmctl", "model": <served name>}` | Exactly when `agent-default-model` is there, because the acp profile ignores `agent-default-model` and reads this row (A4) |

dsh's own `llm-pi-ai` row has no config ([base/cordis.patch.yml:135-136](https://github.com/deepseek-ai/deepseek-harness/blob/38a6a5c965aa99ab525161efb7cc5cf1fe62fced/packages/bundle/base/cordis.patch.yml#L135-L136)) and its `acp` row only `provider` and `model` ([acp-app/cordis.patch.yml:15-20](https://github.com/deepseek-ai/deepseek-harness/blob/38a6a5c965aa99ab525161efb7cc5cf1fe62fced/packages/bundle/acp-app/cordis.patch.yml#L15-L20)), so replacing either config loses nothing of dsh's; replacing `llm-pi-ai`'s config does replace every pi-ai provider configured in a lower layer, which is why llmctl refuses such rows instead of hiding them.

When the home patch exists without markers, as when it holds your own rows, llmctl backs it up once to `cordis.patch.yml.llmctl-backup` and appends the header and the block after your rows.
When the file does not end with a newline, llmctl adds one first: without it, dsh read the last value `disabled: true` as the string `true# dsh rows for the model llmctl runs (docs/llmctl.md#clients).`, which its dump printed unquoted as `disabled: true# dsh rows ...`, easy to misread as `true` (S4b, [candidate](../research/2026-11-16/llmctl-dsh-stub/spec/s4b-candidate.yml), [dump line 176](../research/2026-11-16/llmctl-dsh-stub/spec/s4b.stdout)) (measured on the MSI Raider 18 Max HX, dsh 0.2.2-rc.3, no model server).
On every later `up`, llmctl replaces only the lines between the markers and keeps every byte outside them, a byte-order mark, CRLF line endings and comments included; a byte-order mark directly before the begin marker on the first line is accepted and kept, as in Hermes' profile.
A backup is written once, never overwritten and never restored by llmctl; a symlinked home patch is written at its target and keeps its mode, and the home patch takes no lock, because dsh never writes it.

For example, a home patch that holds only a row disabling dsh's session telemetry looks like this after `./llmctl up gemma-4-26b-a4b --set-default`; the first two lines are yours and stay byte for byte, and S10 and S11 ran this file against the real server:

```yaml
- id: session-telemetry-otel
  disabled: true
# dsh rows for the model llmctl runs (docs/llmctl.md#clients).
# llmctl rewrites the lines between the two markers on every ./llmctl up and keeps everything else in this file.
# While the block holds the agent-default-model line, the running model is dsh's default in every profile;
# delete that line and the acp line to give the default back to dsh's model picker.
# >>> llmctl managed block: rewritten by llmctl on every start; edits here are lost >>>
- {"id": "llm-pi-ai", "config": {"providers": {"llmctl": {"displayName": "llmctl", "api": "openai-completions", "baseURL": "http://127.0.0.1:8080/v1", "headers": {"Authorization": "Bearer no-key-required"}, "compat": {"supportsStore": false, "supportsDeveloperRole": false, "maxTokensField": "max_tokens"}, "reasoning": "off", "streamIdleTimeoutMs": 900000, "retryPolicy": {"mode": "normal", "maxRetries": 5, "retryableCodes": ["EMPTY_RESPONSE", "RATE_LIMIT", "SERVER", "TRANSPORT"]}, "models": [{"id": "local", "name": "gemma-4-26b-a4b on llama.cpp (llmctl)", "contextWindow": 131072, "maxTokens": 16384, "input": ["text"], "reasoningEfforts": {"off": null, "medium": "medium"}, "compat": {"thinkingFormat": "chat-template", "chatTemplateKwargs": {"enable_thinking": {"$var": "thinking.enabled"}}}}]}}}}
- {"id": "agent-default-model", "config": {"provider": "llmctl", "model": "local"}}
- {"id": "acp", "config": {"provider": "llmctl", "model": "local"}}
# <<< llmctl managed block <<<
```

Before it writes, llmctl refuses a home patch it cannot change safely, and starts nothing:

- a file that is not UTF-8 text;
- damaged markers: not exactly one begin and one end marker, begin first;
- a line inside the block that is not blank, not a comment and not one of the three rows as llmctl writes them, because the block is llmctl's only state and a line it cannot read could hide whether it holds the default;
- outside the block, a line at column 0 that is not blank, not a comment, not exactly `-` and does not start with `- `; a `[]` line, as in dsh's template, gets its own message, because dsh then reads the file as one YAML list and the appended rows would break it (S3: exit 1 with a `YAMLException`);
- outside the block, a row that targets `llm-pi-ai`, or `agent-default-model` or `acp` while the block holds or would hold the default, because dsh applies only the later of two rows with one id without a word (S4d);
- any profile patch `$DSH_HOME/profiles/*/cordis.patch.yml` that targets `llm-pi-ai`, because llmctl's home row would hide it without a message (A2-home); [Moving a hand-made setup](#moving-a-hand-made-setup) covers this case.

"Targets" is a conservative scan of lines whose first non-space character is not `#` for an `id` key whose value is one of the three row ids; it matches block style (`- id: llm-pi-ai`), JSON flow lines, the Web UI's multi-line flow maps and `insert` rows, and not `name: '@deepseek-ai/dsh-llm-pi-ai'` or `acp-app-startup`.
It can miss exotic YAML such as anchors, `? id` keys or `!!js` ids; the dsh check under [Checks before writing](#checks-before-writing) catches the composition problems it can see.

The cost of this place: while llmctl's `llm-pi-ai` row is in the home patch, dsh's Models page cannot add or edit OpenAI-compatible providers, and says `Configuration for "llm-pi-ai" is overridden by a home patch or command-line overlay` (A2-home; `config-editor` [src/index.ts:136-140](https://github.com/deepseek-ai/deepseek-harness/blob/38a6a5c965aa99ab525161efb7cc5cf1fe62fced/packages/boot/config-editor/src/index.ts#L136-L140)).
`up` prints a note about it the first time it writes the block; leave dsh out with `LLMCTL_CLIENTS` if you need such providers beside llmctl's.

### Rows per thinking type

The `llm-pi-ai` row describes one provider, `llmctl`, with one model, built from the recipe's `[clients]` table:

| Key | Value | Why |
|---|---|---|
| `displayName` | `"llmctl"` | The provider label in the picker |
| `api` | `"openai-completions"` | The protocol of hand-declared OpenAI-compatible routes (`configuration-4`) |
| `baseURL` | `http://127.0.0.1:<port>/v1` | 8080 for llama.cpp, 8000 for vLLM, 30000 for SGLang |
| `headers` | `{"Authorization": "Bearer no-key-required"}` | pi-ai sends nothing without a key or an Authorization header (`configuration-7`); the servers here take no key |
| `compat` | `{"supportsStore": false, "supportsDeveloperRole": false, "maxTokensField": "max_tokens"}` | Without it pi-ai sends `store`, `max_completion_tokens` and the `developer` role to 127.0.0.1 (B1-0, [shapes](../research/2026-11-16/llmctl-dsh-stub/shapes/laneB/b1-nocompat-gptoss.jsonl)) |
| `reasoning` | The recipe's default level, absent for `none` | The session's level unless the session or `agent-default-model` picks one (B1-8) |
| `thinkingBudgets` | `{"medium": 1, "high": 0}`, Nemotron 3.5 Bolt only | Lets one key carry Bolt's medium level (B1-2) |
| `streamIdleTimeoutMs` | `900000` | [Context, compaction and timeouts](#context-compaction-and-timeouts) (estimate) |
| `retryPolicy` | `{"mode": "normal", "maxRetries": 5, "retryableCodes": ["EMPTY_RESPONSE", "RATE_LIMIT", "SERVER", "TRANSPORT"]}` | dsh's default list without `TIMEOUT` (estimate, same section) |
| `models` | One model: `id` the served name, `name` `<id> on <runtime> (llmctl)`, `contextWindow` the recipe's context, `maxTokens` its output cap, `input` `["text"]`, and the thinking keys below | Without `maxTokens`, pi-ai picks `max_tokens` itself: with `contextWindow` 65,536 set, as llmctl always sets it, it sent 54,737 (B3-1, `b3-cap-nomax`), and 128,000 only with no sizes at all (`b3-cap-nosizes`); vLLM refuses a value above `max_model_len` (B3-6) |

These are keys of the 0.2.2-rc.3 schema, the model keys at [llm-pi-ai/src/config.ts:326-350](https://github.com/deepseek-ai/deepseek-harness/blob/38a6a5c965aa99ab525161efb7cc5cf1fe62fced/packages/llm/llm-pi-ai/src/config.ts#L326-L350) and the route keys at [355-380](https://github.com/deepseek-ai/deepseek-harness/blob/38a6a5c965aa99ab525161efb7cc5cf1fe62fced/packages/llm/llm-pi-ai/src/config.ts#L355-L380); dsh drops an unknown model key without an error, as a `tools: false` key showed (B4-1), so llmctl writes no other key.

| Thinking type | Recipes | `reasoningEfforts` | Model `compat` | Route `reasoning` | Main request at that level | Title request |
|---|---|---|---|---|---|---|
| toggle, off by default | Qwen3.6, Gemma 4, Groot, Nemotron 3.5 Lightning, Qwen3.5 | `{"off": null, "medium": "medium"}` | `{"thinkingFormat": "chat-template", "chatTemplateKwargs": {"enable_thinking": {"$var": "thinking.enabled"}}}` | `"off"` | `chat_template_kwargs {"enable_thinking": false}` | `false` |
| toggle, on by default | Mellum | the same | the same | `"medium"` | `{"enable_thinking": true}` | `false` |
| toggle with `medium_kwargs` | Nemotron 3.5 Bolt | `{"off": null, "medium": "medium", "high": "high"}` | as toggle, plus `"medium_effort": {"$var": "thinking.budget"}` | `"off"`, plus `thinkingBudgets` | Off: `{"enable_thinking": false}`; Medium: `{"enable_thinking": true, "medium_effort": 1}`; High: `{"enable_thinking": true}` | `false` |
| effort | Qwen3.8-27B, gpt-oss-20b | each recipe effort under the dsh level of the same name, `none` under `"off"` | `{"thinkingFormat": "openai", "supportsReasoningEffort": true}` | `"medium"` | `reasoning_effort "medium"` | the least supported level: `none` for Qwen3.8-27B, `low` for gpt-oss-20b |
| always | K2 Horizon, DeepSeek-R1 | `{"high": "high"}` | `{"supportsReasoningEffort": false}` | `"high"` | No thinking field | No thinking field |
| none | Qwen3-Coder-Next | `false` | none | none | No thinking field | No thinking field |

All 19 generated home patches went through `dsh --profile headless --json` against the stub and exited 0 with `pong` in about 1.0 s, with exactly these main and title requests; every main request carried `max_tokens` equal to the recipe's output cap, no `store`, a `system` first message, 23 tools and `stream_options.include_usage` (S5, [summary](../research/2026-11-16/llmctl-dsh-stub/spec/s5-summary.jsonl), [shapes](../research/2026-11-16/llmctl-dsh-stub/shapes/spec/llamacpp-qwen3.6-35b-a3b.jsonl)) (measured on the MSI Raider 18 Max HX against a recording stub, dsh 0.2.2-rc.3).
A level a recipe does not declare fails with `UNSUPPORTED_REASONING_EFFORT` before any request (B1-1 to B1-7).

Generated block for qwen3.6-35b-a3b on llama.cpp, with the default lines (`launcher/tests/golden/clients/llamacpp/qwen3.6-35b-a3b.dsh.yml`):

```yaml
- {"id": "llm-pi-ai", "config": {"providers": {"llmctl": {"displayName": "llmctl", "api": "openai-completions", "baseURL": "http://127.0.0.1:8080/v1", "headers": {"Authorization": "Bearer no-key-required"}, "compat": {"supportsStore": false, "supportsDeveloperRole": false, "maxTokensField": "max_tokens"}, "reasoning": "off", "streamIdleTimeoutMs": 900000, "retryPolicy": {"mode": "normal", "maxRetries": 5, "retryableCodes": ["EMPTY_RESPONSE", "RATE_LIMIT", "SERVER", "TRANSPORT"]}, "models": [{"id": "local", "name": "qwen3.6-35b-a3b on llama.cpp (llmctl)", "contextWindow": 131072, "maxTokens": 16384, "input": ["text"], "reasoningEfforts": {"off": null, "medium": "medium"}, "compat": {"thinkingFormat": "chat-template", "chatTemplateKwargs": {"enable_thinking": {"$var": "thinking.enabled"}}}}]}}}}
- {"id": "agent-default-model", "config": {"provider": "llmctl", "model": "local"}}
- {"id": "acp", "config": {"provider": "llmctl", "model": "local"}}
```

Generated block for gpt-oss-20b on vLLM, with the default lines (`launcher/tests/golden/clients/vllm/gpt-oss-20b.dsh.yml`):

```yaml
- {"id": "llm-pi-ai", "config": {"providers": {"llmctl": {"displayName": "llmctl", "api": "openai-completions", "baseURL": "http://127.0.0.1:8000/v1", "headers": {"Authorization": "Bearer no-key-required"}, "compat": {"supportsStore": false, "supportsDeveloperRole": false, "maxTokensField": "max_tokens"}, "reasoning": "medium", "streamIdleTimeoutMs": 900000, "retryPolicy": {"mode": "normal", "maxRetries": 5, "retryableCodes": ["EMPTY_RESPONSE", "RATE_LIMIT", "SERVER", "TRANSPORT"]}, "models": [{"id": "gpt-oss-20b", "name": "gpt-oss-20b on vLLM (llmctl)", "contextWindow": 65536, "maxTokens": 16384, "input": ["text"], "reasoningEfforts": {"low": "low", "medium": "medium", "high": "high"}, "compat": {"thinkingFormat": "openai", "supportsReasoningEffort": true}}]}}}}
- {"id": "agent-default-model", "config": {"provider": "llmctl", "model": "gpt-oss-20b"}}
- {"id": "acp", "config": {"provider": "llmctl", "model": "gpt-oss-20b"}}
```

Without `--set-default`, and without default lines in the earlier block, the block holds only the first line.
Every llama.cpp recipe is served as `local`, so its rows name the model `local`; a vLLM or SGLang recipe uses its served name.
The golden files for all 29 recipes are under `launcher/tests/golden/clients/`, and the tests keep them equal to what llmctl writes.

Nemotron 3.5 Bolt's medium level relies on two things: pi-ai leaves out a thinking budget of 0, so High sends no `medium_effort` (`@earendil-works/pi-ai` 1.5.2 `dist/api/openai-completions.js:750-756`, read in the package), and Bolt's chat template tests `medium_effort` only for truth (B-T, read in the template, not rendered), so `1` works like `true` (B1-2, [shapes](../research/2026-11-16/llmctl-dsh-stub/shapes/laneB/b1-bolt-medium.jsonl)).
A `medium_kwargs` value other than `true` cannot be sent per level that way, so llmctl refuses such a recipe for dsh; no shipped recipe has one.

In dsh's model picker, the levels under Effort are the keys of `reasoningEfforts`, and the route's `reasoning` level is the one selected by default; the picker offers a `Default` entry, the provider's own default, only for a route without `reasoning`, such as lane A's rows in A3-follow ([ui-model-selection/src/client/locales.ts:54-55](https://github.com/deepseek-ai/deepseek-harness/blob/38a6a5c965aa99ab525161efb7cc5cf1fe62fced/packages/client/ui-model-selection/src/client/locales.ts#L54-L55), [ModelSelect.tsx:138-149](https://github.com/deepseek-ai/deepseek-harness/blob/38a6a5c965aa99ab525161efb7cc5cf1fe62fced/packages/client/ui-model-selection/src/client/ModelSelect.tsx#L138-L149), [llm-pi-ai/src/adapter.ts:221-235](https://github.com/deepseek-ai/deepseek-harness/blob/38a6a5c965aa99ab525161efb7cc5cf1fe62fced/packages/llm/llm-pi-ai/src/adapter.ts#L221-L235) and [379](https://github.com/deepseek-ai/deepseek-harness/blob/38a6a5c965aa99ab525161efb7cc5cf1fe62fced/packages/llm/llm-pi-ai/src/adapter.ts#L379)).
`up` prints the choice per type: for a toggle model `off; choose Medium under Effort in dsh's model picker to turn it on` (Bolt: `off; Medium under Effort for medium effort, High for full thinking`), for Mellum `on; choose Off under Effort to turn it off`, for an effort model `medium effort; choose another under Effort (<levels>)` with `off, low, medium` for Qwen3.8-27B and `low, medium, high` for gpt-oss-20b, for K2 Horizon and DeepSeek-R1 `always on, at the effort the server sets`, and `none` for Qwen3-Coder-Next.

### The default model and the picker

dsh's default for new sessions is the composed `agent-default-model` row, `deepseek-official/deepseek-flash` in the base bundle ([base/cordis.patch.yml:90-94](https://github.com/deepseek-ai/deepseek-harness/blob/38a6a5c965aa99ab525161efb7cc5cf1fe62fced/packages/bundle/base/cordis.patch.yml#L90-L94)), and the acp profile's default is the `acp` row (A4).
For dsh, "the default points at llmctl" means that llmctl's block holds the `agent-default-model` line, and the rules of [llmctl.md](llmctl.md#defaults-and-the-follow-rule) read:

1. With `--set-default` or `LLMCTL_SET_DEFAULT=1`, llmctl writes the `agent-default-model` and `acp` lines.
2. Without it, default lines that are already in the block follow the running model on every `up` (A3-follow), and llmctl adds none; a lone `acp` line is dropped, and a missing `acp` line next to `agent-default-model` is written back.
3. Every recipe gets a provider entry, so dsh can always point at the running model.
4. `down` changes nothing; dsh then fails each turn with `TRANSPORT` `ECONNREFUSED` after its retries, in about 16 s (B3-7, [run](../research/2026-11-16/llmctl-dsh-stub/laneB/runs/b3-refused/)).

The default row carries no `reasoningEffort`.
An explicit level there outranks the route's `reasoning` (B1-8), is recorded in each session, and fails every request of a model that lacks it, such as `off` on gpt-oss-20b, K2 Horizon, DeepSeek-R1 and Qwen3-Coder-Next (B1-6, B1-7, B1-9), so the route's `reasoning` gives each recipe its own default instead (measured on the MSI Raider 18 Max HX against a recording stub, dsh 0.2.2-rc.3).

While llmctl holds the default, the home row outranks every profile's `agent-default-model` row (S5, case `migration-profile-rows`), and dsh's model picker changes only the open session: dsh refuses to save another default and only logs a warning ([api/session-controller/src/commands.ts:180-184](https://github.com/deepseek-ai/deepseek-harness/blob/38a6a5c965aa99ab525161efb7cc5cf1fe62fced/packages/api/session-controller/src/commands.ts#L180-L184); A2-home).
`up` prints a note when it takes the default, and `status` says `held by llmctl`.
To give the default back to the picker, delete the `agent-default-model` and `acp` lines in the home patch; `--set-default` writes them again.

The picker saves its default into the active profile patch, which the Web UI owns ([agent-default-model/src/index.ts:82-94](https://github.com/deepseek-ai/deepseek-harness/blob/38a6a5c965aa99ab525161efb7cc5cf1fe62fced/packages/core/agent-default-model/src/index.ts#L82-L94)), and llmctl never edits that file.
So a picker default that names an earlier llmctl model goes stale when the served name changes, for example from `local` to `gpt-oss-20b`, and fails loudly: a blank web session shows `Select model`, and a send fails with `UNKNOWN_MODEL`, with no fallback (A3-stale, [snapshot](../research/2026-11-16/llmctl-dsh-stub/laneA/a3-home-1-model-rewritten.yml)).
When the served name changes while llmctl does not hold the default, `up` prints a note for each profile patch line that sets `agent-default-model`, and `--set-default` lets llmctl hold the default and follow every later `up`.

### Profiles, running processes and sessions

The web, headless, acp and sdk profiles read the home patch; `sdk-minimal` has no base bundle and so no `llm-pi-ai` row, and llmctl does not serve it (A4; [official-bundles/src/bundles.ts:10-26](https://github.com/deepseek-ai/deepseek-harness/blob/38a6a5c965aa99ab525161efb7cc5cf1fe62fced/packages/util/official-bundles/src/bundles.ts#L10-L26)).
The `acp` row makes `dsh --dump-config` of the web, headless and sdk profiles print `patch: entry "acp" not found`, a warning that changes nothing, because those profiles have no `acp` row (A4, [stderr](../research/2026-11-16/llmctl-dsh-stub/laneA/a4-web-dump.err)).

A running `dsh web` reloads the home patch after a rewrite: its picker showed the new rows within 16 s, and an open session used them on its next turn after 8 s (A2-hmr; [hmr/src/index.ts:384-407](https://github.com/deepseek-ai/deepseek-harness/blob/38a6a5c965aa99ab525161efb7cc5cf1fe62fced/packages/boot/hmr/src/index.ts#L384-L407)).
The headless and acp bundles disable that reload, and a running acp process kept the old rows, so headless, acp and sdk processes that are already running keep the old rows until they restart.
`up` prints that note whenever the rows changed, and adds that open web sessions on the old model fail until you pick the new one when the served name changed.

A resumed session keeps a level it took from the route across entry changes; a session with an explicit `off` fails on a model without that level, and a resume onto a model id with a 64k window failed with `MODEL_SWITCH_COMPACTION_FAILED` (B1-9, [stderr](../research/2026-11-16/llmctl-dsh-stub/laneB/runs/b1-resume/r3.err)) (measured on the MSI Raider 18 Max HX against a recording stub, dsh 0.2.2-rc.3).
Start a new session after an `up` that changed the rows.

### Checks before writing

Before anything starts, llmctl checks its own result: the new text starts with the old text before the begin marker (or with the whole old text and the added newline when it appends) and ends with the old text after the end marker, byte for byte, and every block line parses with `json.loads` to the intended row.

Then it lets the installed dsh read the candidate:

1. `dsh --version`; a failure refuses, and a version other than 0.2.2-rc.3 prints a warning after the clients line and goes on (S6: 0.07 s, no writes).
2. In a new private temporary folder (`llmctl-dsh-*`, mode 0700, removed afterwards), llmctl writes `home/cordis.patch.yml` with the candidate text, your rows included, and runs `dsh --profile acp --dump-config` with `DSH_HOME` set to that `home` and `DSH_TELEMETRY_DISABLED=1`.
3. The dump loads no `.env` file and boots no plugin (`apps/cli` [src/bin.ts:55-64](https://github.com/deepseek-ai/deepseek-harness/blob/38a6a5c965aa99ab525161efb7cc5cf1fe62fced/apps/cli/src/bin.ts#L55-L64), [src/dump-config.ts:32-42](https://github.com/deepseek-ai/deepseek-harness/blob/38a6a5c965aa99ab525161efb7cc5cf1fe62fced/apps/cli/src/dump-config.ts#L32-L42)); it took 0.07 to 0.08 s in four timed runs and wrote only inside the private home (S1, [strace](../research/2026-11-16/llmctl-dsh-stub/spec/s1.strace)) (measured on the MSI Raider 18 Max HX, dsh 0.2.2-rc.3, no model server).
4. The check passes when the exit status is 0, stderr has no `patch: entry "<id>" not found` line for a row llmctl writes (S2), and each row llmctl writes appears as `- id: <id>` in a `# == ..., patched by ...` group that names the private home patch (S1, [dump](../research/2026-11-16/llmctl-dsh-stub/spec/s1.stdout)).
5. A non-zero exit refuses with dsh's `Error: ` line, the private path replaced by yours (S3, S4, [stderr](../research/2026-11-16/llmctl-dsh-stub/spec/s4.stderr)).

The acp profile is used because it is the one shipped profile that has all three rows; warnings about your own rows do not fail the check.
After the server is ready, llmctl plans the file again and repeats the dsh check only when the candidate text changed.

What the check proves: the candidate parses as dsh's patch list, the installed dsh still has the three row ids, and the home layer reaches them, which closes the silent path of a renamed row id.
What it does not prove: the schema of llmctl's values, because the dump mounts no plugin.
A value the installed dsh rejects makes the `llm-pi-ai` row fail to activate, which dsh reports as `dsh: warning: 1 entry did not activate` (S8) and as `NO_ADAPTER` on the next request (A1); the version warning, the golden files, S5 and the e2e cover the verified version.
`up --dry-run` runs neither dsh command, creates no folder, and ends with `dry run: nothing was written, and dsh has not checked llmctl's rows`.

### Context, compaction and timeouts

dsh compacts a session before the window fills only above `min(0.8 x window, window - maxTokens - 65,536)` tokens, and never when that value is not positive, which dsh reports only once per model through its own logger and headless did not print at all (B3-2; [compaction-basic/src/config.ts:20](https://github.com/deepseek-ai/deepseek-harness/blob/38a6a5c965aa99ab525161efb7cc5cf1fe62fced/packages/compaction/compaction-basic/src/config.ts#L20), [75](https://github.com/deepseek-ai/deepseek-harness/blob/38a6a5c965aa99ab525161efb7cc5cf1fe62fced/packages/compaction/compaction-basic/src/config.ts#L75) and [172-194](https://github.com/deepseek-ai/deepseek-harness/blob/38a6a5c965aa99ab525161efb7cc5cf1fe62fced/packages/compaction/compaction-basic/src/config.ts#L172-L194), [src/index.ts:165-171](https://github.com/deepseek-ai/deepseek-harness/blob/38a6a5c965aa99ab525161efb7cc5cf1fe62fced/packages/compaction/compaction-basic/src/index.ts#L165-L171), [runs](../research/2026-11-16/llmctl-dsh-stub/laneB/runs/b3-comp-64k-60000/)) (measured on the MSI Raider 18 Max HX against a recording stub, dsh 0.2.2-rc.3).
The 128k recipes compact at 49,152 tokens.
10 of 29 recipes have a context minus output cap of 65,536 tokens or less and get no proactive compaction: on llama.cpp qwen3.8-27b, k2-horizon-7b, k2-horizon-mova-36b and deepseek-r1-0528-qwen3-8b, on vLLM gemma-4-e4b, gpt-oss-20b, k2-horizon-7b and deepseek-r1-0528-qwen3-8b, and on SGLang k2-horizon-7b and gemma-4-e4b.
For them dsh still recovers after the server rejects an overlong prompt (`configuration-23`, `extra-5`), and `up` prints a compaction line; llmctl writes no compaction row.

`streamIdleTimeoutMs` 900000 replaces dsh's default of 300,000 ([llm-pi-ai/src/config.ts:47](https://github.com/deepseek-ai/deepseek-harness/blob/38a6a5c965aa99ab525161efb7cc5cf1fe62fced/packages/llm/llm-pi-ai/src/config.ts#L47)), because time spent queued behind another request on a single-slot server counts as idle (B3-4), and 16,384 output tokens at Qwen3.8-27B's 36 tok/s take about 455 s (B3-8) (estimate).
The retry list is dsh's default without `TIMEOUT` ([llm/src/retry-policy.ts:15-25](https://github.com/deepseek-ai/deepseek-harness/blob/38a6a5c965aa99ab525161efb7cc5cf1fe62fced/packages/llm/llm/src/retry-policy.ts#L15-L25)), because a timed-out request was retried 5 times behind its own abandoned attempt on one slot, and a policy without `TIMEOUT` failed once (B3-5, [run](../research/2026-11-16/llmctl-dsh-stub/laneB/runs/b3-retry-policy/)); both values are estimates until an e2e run records a step that waited behind another request.
A stopped server fails each turn after about 16 s (B3-7).

### Title requests and single-slot servers

dsh titles a new session with a separate request on the same route, with `max_tokens` 4096 and the least supported thinking level, sent at the same time as the main request (B2-1 to B2-6, [shapes](../research/2026-11-16/llmctl-dsh-stub/shapes/laneB/b2-parallel-title.jsonl)).
On a single-slot llama.cpp server the title request was served first in 8 of 8 real runs, so the first answer of each new session waits for the title ([real runs](../research/2026-11-16/llmctl-dsh-stub/laneB/runs/real-title-thinking/)) (measured on the MSI Raider 18 Max HX through a recording proxy to the llama.cpp server, dsh 0.2.2-rc.3).
For K2 Horizon and DeepSeek-R1, which always think, that wait may reach the title's 60 s deadline (estimate).
Compaction requests use only the route's `reasoning`.
llmctl writes no title row, because a `session-title-llm` row would change titles for every provider.

### Chat-only recipes

dsh has no per-model tool switch and sends its tools with every request unless a `--patch` disables its tool rows (B4-1, `configuration-25`), so llmctl writes the `always` rows for both `deepseek-r1-0528-qwen3-8b` recipes anyway, to keep dsh's provider on the running model, and `up` prints a chat-only line.
The other clients leave tools out of a chat-only recipe themselves: OpenCode through the entry's `"tools": false`, Hermes through the block's `platform_toolsets: {"cli": []}`, and Pi when started with `--no-tools`.
On vLLM, whose recipe has no tool parser, every dsh request that carries tools fails with HTTP 400 (B4-3); on llama.cpp the template renders the tools (not run); SGLang ships no chat-only recipe, and an SGLang recipe without a tool parser would answer such a request with HTTP 200 and any call as text (inferred from serving_chat.py, [sglang/README.md](../sglang/README.md#tool-calling-and-reasoning)).
For a prompt without tools, save these rows to a file and pass it with `--patch`; they disable dsh's 14 tool rows, and dsh then sends no `tools` (B4-2, [shapes](../research/2026-11-16/llmctl-dsh-stub/shapes/laneB/b4-notools.jsonl)) (measured on the MSI Raider 18 Max HX against a recording stub, dsh 0.2.2-rc.3):

```yaml
- {"id": "tool-bash", "disabled": true}
- {"id": "tool-goal", "disabled": true}
- {"id": "tool-fs", "disabled": true}
- {"id": "tool-fs-search", "disabled": true}
- {"id": "tool-jobs", "disabled": true}
- {"id": "tool-subagent-control", "disabled": true}
- {"id": "tool-skill", "disabled": true}
- {"id": "tool-subagent", "disabled": true}
- {"id": "tool-subagent-fork", "disabled": true}
- {"id": "tool-todo", "disabled": true}
- {"id": "tool-web", "disabled": true}
- {"id": "tool-workflow", "disabled": true}
- {"id": "tool-working-directory", "disabled": true}
- {"id": "tool-agent-team", "disabled": true}
```

```bash
dsh --profile headless --patch <file> "your prompt"
```

dsh reads its own flags (`--profile`, `--patch`, `--dump-config`) only up to the first token it does not know and hands the rest to the booted app ([apps/cli/src/args.ts:5-11](https://github.com/deepseek-ai/deepseek-harness/blob/38a6a5c965aa99ab525161efb7cc5cf1fe62fced/apps/cli/src/args.ts#L5-L11)), so `--patch` goes before `--json` or any other headless option; `dsh --profile headless --json --patch <file> ...` exits 1 with `unknown option '--patch'`.

### Model errors in dsh sessions

Gemma 4 models sometimes add a `justification` argument to dsh's `write` tool, which dsh refuses with `SANDBOX_ESCALATION_INVALID_ARGUMENT` (`justification is only valid together with sandbox_permissions`); the model then writes without it, and the session ends with the file written, but the e2e fails a dsh session on any tool error.
The e2e saw it with Gemma 4 26B-A4B on llama.cpp and Gemma 4 E4B on vLLM on 2026-11-17 ([record 115](../research/2026-11-17/llmctl-kimi-e2e/records/115-06-gemma-4-26b-a4b-llamacpp-dsh-headless), [record 302](../research/2026-11-17/llmctl-kimi-e2e/records/302-16-gemma-4-e4b-vllm-dsh-headless)) and on 2026-11-18 ([record 115](../research/2026-11-18/llmctl-sglang-e2e/records/115-06-gemma-4-26b-a4b-llamacpp-dsh-headless), [record 378](../research/2026-11-18/llmctl-sglang-e2e/records/378-20-gemma-4-e4b-vllm-dsh-headless)), and with Gemma 4 E4B on its new SGLang recipe, three refused writes before one that succeeded ([record 534](../research/2026-11-18/llmctl-sglang-e2e/records/534-27-gemma-4-e4b-sglang-dsh-headless)) (measured on the MSI Raider 18 Max HX, llama.cpp b12407, vLLM v0.33.0 and SGLang v0.5.23, dsh 0.2.2-rc.3, 2026-11-17 and 2026-11-18).
Gemma 4 E4B on llama.cpp and Gemma 4 12B on llama.cpp and SGLang passed their dsh sessions in the run of 2026-11-18, so this is the model's choice of arguments, not llmctl's rows (inferred from the records).

### Moving a hand-made setup

If you set dsh up for an llmctl server by hand before, with `llm-pi-ai` and `agent-default-model` rows in the profile patches that the Web UI writes, move it this way; llmctl never edits those files, so nothing you or the Web UI added there can be lost by llmctl:

1. The first `./llmctl up` with dsh active refuses and names each profile patch line that sets `llm-pi-ai`.
2. Delete those `llm-pi-ai` rows; llmctl's row in the home patch serves the same purpose and would outrank them anyway (S5).
3. Delete your `agent-default-model` rows too, or keep them and run `--set-default`: while llmctl holds the default its home row outranks them (S5), but when you later give the default back, a `reasoningEffort` such as `'off'` in them fails every recipe without that level (B1-6, B1-7).
4. Run `./llmctl up <model> --set-default`: llmctl backs up the home patch once, keeps your rows in it byte for byte, and appends the header and the block.

Rows the Web UI wrote, such as `ui-settings-general`, stay in the profile patches; llmctl reads those files only for the refusal and the profile-default note.

### Privacy and side effects

dsh uploads a whole session, its history, tool arguments and results and the working folder's path included, to DeepSeek's collector `dsh-otel-collector.deepseeksvc.com` whenever you rate an answer, change or withdraw a rating, edit a note or send `/feedback`, for every provider including local ones; this is on by default in the web, headless, sdk and acp profiles (`runtime-17`, `runtime-18`; [base/cordis.patch.yml:218-232](https://github.com/deepseek-ai/deepseek-harness/blob/38a6a5c965aa99ab525161efb7cc5cf1fe62fced/packages/bundle/base/cordis.patch.yml#L218-L232), [session-telemetry-otel/README.md:14](https://github.com/deepseek-ai/deepseek-harness/blob/38a6a5c965aa99ab525161efb7cc5cf1fe62fced/packages/session/session-telemetry-otel/README.md?plain=1#L14) and [36](https://github.com/deepseek-ai/deepseek-harness/blob/38a6a5c965aa99ab525161efb7cc5cf1fe62fced/packages/session/session-telemetry-otel/README.md?plain=1#L36), [session-telemetry/src/coordinator.ts:298-299](https://github.com/deepseek-ai/deepseek-harness/blob/38a6a5c965aa99ab525161efb7cc5cf1fe62fced/packages/session/session-telemetry/src/coordinator.ts#L298-L299)) (read in the dsh source).
Export `DSH_TELEMETRY_DISABLED=1` in your shell to turn it off; any non-empty value disables it, and a `DSH_*` variable cannot come from a `.env` file (`runtime-19`; [app-boot/src/profile-context.ts:39-56](https://github.com/deepseek-ai/deepseek-harness/blob/38a6a5c965aa99ab525161efb7cc5cf1fe62fced/packages/boot/app-boot/src/profile-context.ts#L39-L56)).
llmctl writes no telemetry row; a `- id: session-telemetry-otel` row with `disabled: true` in the home patch, outside llmctl's block, works too.

`web_fetch` reaches 127.0.0.1 and private-network addresses without an approval, and file reads are not confined, so a model steered by a fetched page can read local files and send them out in a URL (`extra-0`; [web-fetch-http/README.md:12](https://github.com/deepseek-ai/deepseek-harness/blob/38a6a5c965aa99ab525161efb7cc5cf1fe62fced/packages/web/web-fetch-http/README.md?plain=1#L12)) (read in the dsh source).
`web_search` uses the DeepSeek Search API with the session's DeepSeek credentials, so with `DEEPSEEK_API_KEY` set, dsh sends model-written search queries to DeepSeek (`runtime-26`; [base/cordis.patch.yml:484-493](https://github.com/deepseek-ai/deepseek-harness/blob/38a6a5c965aa99ab525161efb7cc5cf1fe62fced/packages/bundle/base/cordis.patch.yml#L484-L493)).
The first page load of the Web UI creates `~/Documents/deepseek-harness/default-workspace`, outside `DSH_HOME` (SIDE, lane A's report); on 2026-11-17 it did so even with `DSH_HOME`, `DSH_AGENTS_HOME` and `PWD` in a private home (measured on the MSI Raider 18 Max HX, dsh 0.2.2-rc.3, 2026-11-17; [record](../research/2026-11-17/llmctl-dsh-web/README.md#cleanup)).

Every command dsh runs is placed in its own scope with `systemd-run --user --scope` and no `--slice` (`extra-8`; [subprocess-local/src/linux-scope.ts:542-560](https://github.com/deepseek-ai/deepseek-harness/blob/38a6a5c965aa99ab525161efb7cc5cf1fe62fced/packages/subprocess/subprocess-local/src/linux-scope.ts#L542-L560)).
On this laptop a `cat /proc/self/cgroup` that dsh's bash tool ran, in a headless session started under `heavy`, reported `/user.slice/user-1000.slice/user@1000.service/app.slice/dsh-subprocess-<pid>-<id>.scope`: the command ran in the user manager's `app.slice`, outside `heavy.slice`, so heavy's memory cap does not cover the commands dsh's tools run even when dsh itself runs under `heavy` (measured on the MSI Raider 18 Max HX, llama.cpp b12407, dsh 0.2.2-rc.3, 2026-11-17; [tool result](../research/2026-11-17/llmctl-dsh-e2e/lead-checks/cgroup-tool-result.jsonl)).
With telemetry disabled, no DeepSeek key and only llmctl's route, the research found no built-in path that sends prompts or code off the machine (`runtime-29`).
A headless `pong` session run under `strace -f -e trace=connect` without `DSH_TELEMETRY_DISABLED`, with the home patch's `session-telemetry-otel` row disabled and llmctl's route as the default, made two connect calls, both to 127.0.0.1 port 8080, and no DNS lookup (measured on the MSI Raider 18 Max HX, llama.cpp b12407, dsh 0.2.2-rc.3, 2026-11-17; [connect lines](../research/2026-11-17/llmctl-dsh-e2e/lead-checks/connect.strace.txt)).
That one short session used no tool, rating, note or web search, so it does not cover the paths above.

## Kimi Code CLI

llmctl points Kimi Code CLI at the running model through two tables in Kimi's own `config.toml`, `[providers.llmctl]` and one `[models."llmctl/<served name>"]`, and through Kimi's top-level `default_model` when you ask for it.
Use it with `kimi -m llmctl/<served name>` or pick `<id> on <runtime> (llmctl)` with `/model` in Kimi, or make it Kimi's default with `./llmctl up <model> --set-default`, after which plain `kimi` and `kimi -p` use it.
The evidence ids below (E1 to E8, P1 to P3, the prototype and the hands-on runs) are mapped to their files in [research/2026-11-17/llmctl-kimi-stub/](../research/2026-11-17/llmctl-kimi-stub/README.md#claims-and-files), and the research ids (`docs-K036` for the docs lane, `local-K25` for the binary lane) to the verdicts archived in [research-claims.json](../research/2026-11-17/llmctl-kimi-stub/research-claims.json), described in that README's [Research claims](../research/2026-11-17/llmctl-kimi-stub/README.md#research-claims).
E1 to E8 and P1 to P3 ran Kimi Code 2.5.3 in private Kimi homes inside an offline sandbox, against a recording stub on the entry's own port or with no server, and are labelled "(measured on the MSI Raider 18 Max HX, Kimi Code 2.5.3, recording stub)" or "(measured on the MSI Raider 18 Max HX, Kimi Code 2.5.3, no server)"; the hands-on runs used one hand-written entry through a recording proxy to the gemma-4-26b-a4b llama.cpp server and are labelled "(measured on the MSI Raider 18 Max HX, Kimi Code 2.5.3, llama.cpp b12407)".
A `main.cjs:<lines>` citation is lines of the bundle that [sea_extract.py](../research/2026-11-17/llmctl-kimi-stub/bin/sea_extract.py) extracts from the 2.5.3 executable ([The bundle](../research/2026-11-17/llmctl-kimi-stub/README.md#the-bundle)), read in the code and not run unless a record says so.

### The Kimi tables

The file is `$KIMI_CODE_HOME/config.toml`, or `~/.kimi-code/config.toml` when `KIMI_CODE_HOME` is unset; Kimi has no flag or variable for another path (local-K07, local-K09; docs-K019, docs-K021 (vendor-reported)).
`KIMI_CODE_HOME` moves everything Kimi keeps, sessions, logs and credentials included, not only the config (docs-K012) (vendor-reported).
Kimi 2.5.3 disagrees with itself about an empty `KIMI_CODE_HOME`: its `getDataDir` and its login code treat an empty value as unset and use `~/.kimi-code`, while the engine's `resolveKimiHome` keeps the empty value and so reads `config.toml` from the working folder (local-K09, main.cjs:61652-61657 against 322789-322793, 16023-16027 and 17479-17483) (inferred from main.cjs, not run).
So llmctl refuses an empty `KIMI_CODE_HOME`, and a relative one, which Kimi uses as given against each command's working folder; it expands neither `~` nor `$NAME`, because Kimi does not ([llmctl.md](llmctl.md#when-up-refuses)).
llmctl never creates the Kimi home; when the home exists without `config.toml`, the first write creates the file with mode 0600, the mode Kimi gives it (local-K13).

In that file llmctl owns, by name and wherever Kimi moved them:

- `[providers.llmctl]` and every table or array of tables below it;
- every `[models."<alias>"]` table whose alias starts with `llmctl/`, and every table below one; after each write exactly one exists, `llmctl/<served name>`;
- the top-level `default_model`, only by the rules of [The Kimi default model](#the-kimi-default-model).

Everything else is yours or Kimi's, and llmctl writes no comment and no marker.
A key or a table that you or Kimi add inside llmctl's tables is removed by the next write that changes them, as an edit inside the Hermes and dsh blocks is.
llmctl never writes `telemetry`, `[thinking]`, `[tools]`, `default_provider` or any other key, and no other file in the Kimi home: not `tui.toml`, `agents/`, `mcp.json`, `credentials/` or `device_id`.

The provider table, in this key order:

| Key | Value | Why |
|---|---|---|
| `type` | `"openai"` | The Chat Completions wire, documented for compatible servers (docs-K036, docs-K041) (vendor-reported) |
| `base_url` | `"http://127.0.0.1:<port>/v1"`, 8080 for llama.cpp, 8000 for vLLM and 30000 for SGLang | The type's default is the OpenAI endpoint (docs-K043), and without a `base_url` Kimi uses the shell's `OPENAI_BASE_URL` (local-K30) |
| `api_key` | `"no-key-required"` | Without a credential Kimi sends the shell's `OPENAI_API_KEY`, else `Bearer unused` (local-K30); E2 recorded `Authorization: Bearer no-key-required` |
| `model_source` | `"static"` | Keeps the provider out of every catalog refresh (local-K45); a refresh left llmctl's tables byte for byte (E4d) |

The model table, in this key order:

| Key | Value | Why |
|---|---|---|
| `provider` | `"llmctl"` | The provider table above |
| `model` | The served name | Sent as `model` by all 19 entries (E2) |
| `display_name` | `"<id> on <runtime> (llmctl)"` | The label in `/model` and in the TUI's footer (E8, P1) |
| `max_context_size` | The recipe's context | Required (docs-K065, local-K20) |
| `max_output_size` | The recipe's output cap | Sent as `max_tokens` by all 19 entries (E2), although the docs say only the anthropic type reads it (docs-K067) |
| `capabilities` | `["tool_use"]`, plus `"thinking"` for an effort recipe, and `[]` for a chat-only recipe | [Kimi thinking per type](#kimi-thinking-per-type), [Kimi chat-only recipes](#kimi-chat-only-recipes); never `image_in`, because every llmctl entry is text only |
| `support_efforts` | The recipe's efforts without `none` | Effort recipes only |
| `default_effort` | `"medium"` | Effort recipes only; the servers' default effort |
| `off_effort` | `"none"` | Only when the recipe's efforts include `none`, which today is Qwen3.8-27B |

Each value is written as JSON, which TOML reads as the same value, so llmctl needs no TOML writer, and the alias is always quoted, as Kimi asks for an alias with a dot (docs-K016) (vendor-reported).
All 19 generated table pairs were accepted by `kimi provider list --json` with every value, and each first request carried `max_tokens`, `messages`, `model` (the served name), `prompt_cache_key`, `stream`, `stream_options` and 24 `tools`, plus `reasoning_effort: "medium"` for the three effort entries only (E2, [summary](../research/2026-11-17/llmctl-kimi-stub/exp/runs/e2-summary-all.jsonl)) (measured on the MSI Raider 18 Max HX, Kimi Code 2.5.3, recording stub).

Generated tables for qwen3.6-35b-a3b on llama.cpp, toggle with the server's default off (`launcher/tests/golden/clients/llamacpp/qwen3.6-35b-a3b.kimi.toml`):

```toml
[providers.llmctl]
type = "openai"
base_url = "http://127.0.0.1:8080/v1"
api_key = "no-key-required"
model_source = "static"

[models."llmctl/local"]
provider = "llmctl"
model = "local"
display_name = "qwen3.6-35b-a3b on llama.cpp (llmctl)"
max_context_size = 131072
max_output_size = 16384
capabilities = ["tool_use"]
```

Generated tables for gpt-oss-20b on vLLM, effort without `none` (`launcher/tests/golden/clients/vllm/gpt-oss-20b.kimi.toml`):

```toml
[providers.llmctl]
type = "openai"
base_url = "http://127.0.0.1:8000/v1"
api_key = "no-key-required"
model_source = "static"

[models."llmctl/gpt-oss-20b"]
provider = "llmctl"
model = "gpt-oss-20b"
display_name = "gpt-oss-20b on vLLM (llmctl)"
max_context_size = 65536
max_output_size = 16384
capabilities = ["tool_use", "thinking"]
support_efforts = ["low", "medium", "high"]
default_effort = "medium"
```

Every llama.cpp recipe is served as `local`, so its alias is `llmctl/local`; a vLLM or SGLang recipe's alias is `llmctl/<served name>`, such as `llmctl/gpt-oss-20b`, and an SGLang recipe of a model that also has a vLLM recipe keeps that recipe's served name, so an open Kimi session keeps its alias across a switch between those two.
The golden files for all 29 recipes are under `launcher/tests/golden/clients/`, and the tests keep them equal to what llmctl writes.

Kimi rewrites `config.toml` itself, so llmctl finds its tables as TOML, never by text patterns (measured on the MSI Raider 18 Max HX, Kimi Code 2.5.3, no server):

- Kimi's first `default_model`, from `kimi web` or from a `/model` pick in the TUI, rewrote the whole file: every comment gone, `default_model` first, arrays written as `[ "tool_use" ]`, and a new inode (E4a, [file](../research/2026-11-17/llmctl-kimi-stub/exp/writer/a-web-set-default.toml)); a later default change edited only that line (E4a, [file](../research/2026-11-17/llmctl-kimi-stub/exp/writer/a3-web-switch-default.toml)).
- `kimi provider add` put the new provider and its `source` sub-table between `[providers.llmctl]` and llmctl's model table (E4b, [file](../research/2026-11-17/llmctl-kimi-stub/exp/writer/b-provider-add.toml)).
- `kimi provider remove` deleted only the other provider's lines and left runs of blank lines (E4c, [file](../research/2026-11-17/llmctl-kimi-stub/exp/writer/c-provider-remove.toml)).
- `/effort` in the TUI appended `[thinking]` directly after llmctl's last line, without a blank line (P3) (measured on the MSI Raider 18 Max HX, Kimi Code 2.5.3, recording stub).

llmctl parses the whole file with Python's `tomllib`, splits it into headers and statements, and recognises its tables by their decoded header paths, so a moved table, Kimi's spacing, a quoted or literal key and a multi-line array do not confuse it.
A table of llmctl's that already holds llmctl's values stays as it is, Kimi's spacing included; one that differs is replaced in place, and the tables below it and every other `llmctl/` alias are removed.
A missing provider table goes after the last `providers` table, and a missing model table after the last `models` table, else at the end of the file, which is where Kimi's own writer puts new ones (main.cjs:168489-168496; E4b, E4d).
Every line outside llmctl's tables and the `default_model` line keeps its bytes and its place, and new lines take the file's line ending, CRLF when the first line ends with one.
The design's prototype planned the eight E4 files, the two P3 files and a template file for four recipes with and without `--set-default`, and all 88 results parsed back with llmctl's tables and default as intended and everything else equal (measured on the MSI Raider 18 Max HX, Python 3.14 tomllib, no server; [prototype](../research/2026-11-17/llmctl-kimi-stub/spec/proto/kimi_proto.py)).

For example, a `config.toml` that holds Kimi's four template comment lines, a blank line and `telemetry = false` looks like this after `./llmctl up gemma-4-26b-a4b` without `--set-default`: the six lines that were there keep their bytes, and the tables follow after a blank line ([prototype output](../research/2026-11-17/llmctl-kimi-stub/spec/proto/out/owner--gemma--nodefault.toml)):

```toml
# ~/.kimi-code/config.toml
# Environment settings for Kimi Code.
# This file starts empty so built-in defaults can apply.
# Login will populate managed Kimi provider and model entries.

telemetry = false

[providers.llmctl]
type = "openai"
base_url = "http://127.0.0.1:8080/v1"
api_key = "no-key-required"
model_source = "static"

[models."llmctl/local"]
provider = "llmctl"
model = "local"
display_name = "gemma-4-26b-a4b on llama.cpp (llmctl)"
max_context_size = 131072
max_output_size = 16384
capabilities = ["tool_use"]
```

With `--set-default`, the same write adds one line, `default_model = "llmctl/local"`, right after `telemetry = false`, and nothing else changes ([prototype output](../research/2026-11-17/llmctl-kimi-stub/spec/proto/out/owner--gemma--default.toml)).
After a `kimi provider add`, llmctl's two tables change in place while the provider Kimi put between them, its `source` sub-table and Kimi's `[ "tool_use" ]` spacing stay ([prototype output](../research/2026-11-17/llmctl-kimi-stub/spec/proto/out/b-provider-add--gptoss-vllm--nodefault.toml)).

Before anything starts, llmctl refuses a `config.toml` it cannot change safely, and writes nothing:

- a file that is not UTF-8 text, or not valid TOML, a byte-order mark included, which TOML does not allow;
- `providers` or `models` that is not a table, such as `[[providers]]`;
- `providers` or `models` written at the top level as an inline table or with dotted keys, because a `[providers.llmctl]` header cannot be added after it;
- llmctl's provider, or an `llmctl/` alias, written with dotted keys or as an inline table, because llmctl replaces its tables line by line;
- a `default_model` that llmctl must change but cannot edit in place, such as a value over several lines or one that is not a string.

The messages and what to do are in [llmctl.md](llmctl.md#when-up-refuses).

Before its first change to a file that holds anything besides llmctl's tables and an llmctl `default_model` line, llmctl copies it once to `config.toml.llmctl-backup` beside it, with its mode; that copy also keeps the comments that Kimi's own first full rewrite drops.
A table under llmctl's names that llmctl could not have written, such as a `[providers.llmctl]` with another `type`, a remote `base_url` or a key of your own, counts as your content for that test, so its first replacement is backed up too.
A backup is written once, never overwritten and never restored by llmctl.
The write goes through `config.toml.llmctl-tmp` in the same folder and a rename, keeps the file's mode, and follows a symlink to its target; Kimi watches only `config.toml` (main.cjs:168816-168824, local-K46).
Kimi takes no lock on `config.toml` and reads it again before each of its own writes (main.cjs:169229-169277), so llmctl takes none either: it plans again right before the rename and reads the file back afterwards, and when the text changed meanwhile, probably through an open Kimi, `up` exits 6 with `<path>: the file changed while llmctl wrote it, probably through Kimi; check it, then run ./llmctl up again`, the other clients written.
A Kimi write that read the file before llmctl's rename and lands after the read-back can still undo llmctl's change (inferred from main.cjs:169229-169277); no open TUI wrote the file back in E8, and the next `./llmctl status` shows such a loss.

### Kimi thinking per type

| Thinking type | Recipes | Model keys | Main request (E2) | `thinking:` line of `up` |
|---|---|---|---|---|
| toggle, default off | Qwen3.6 (llama.cpp and SGLang), Gemma 4 26B-A4B, Gemma 4 12B and E4B (all three runtimes), Groot (llama.cpp and vLLM), Nemotron 3.5 Bolt and Lightning, Qwen3.5 (all three runtimes) | `capabilities = ["tool_use"]` | No thinking field, so the server's default applies | `off, as the server sets it` |
| toggle, default on | Mellum2.1 (llama.cpp and vLLM) | `capabilities = ["tool_use"]` | No thinking field | `on, as the server sets it` |
| effort with `none` | Qwen3.8-27B | `["tool_use", "thinking"]`, `support_efforts = ["low", "medium"]`, `default_effort = "medium"`, `off_effort = "none"` | `reasoning_effort: "medium"` | `medium effort; /effort in kimi for another (off, low, medium)` |
| effort without `none` | gpt-oss-20b (all three runtimes) | `["tool_use", "thinking"]`, `support_efforts = ["low", "medium", "high"]`, `default_effort = "medium"` | `reasoning_effort: "medium"` | `medium effort; /effort in kimi for another (low, medium, high)` |
| always | K2 Horizon 7B (all three runtimes), K2 Horizon Mova 36B, DeepSeek-R1 (llama.cpp and vLLM) | `["tool_use"]`, and `[]` for DeepSeek-R1 | No thinking field | `always on, at the effort the server sets` |
| none | Qwen3-Coder-Next | `["tool_use"]` | No thinking field | `none` |

That is 18 toggle, 4 effort, 6 always and 1 none recipe; the keys and requests come from E1 and E2 (measured on the MSI Raider 18 Max HX, Kimi Code 2.5.3, recording stub), which ran before SGLang was added, and the six SGLang recipes and the four llama.cpp recipes of Qwen3.5-9B, Gemma 4 12B, Gemma 4 E4B and Mellum2.1 get the keys of their thinking types (inferred from their golden files).
Kimi sends a thinking level on this wire only as `reasoning_effort` (local-K31), so a toggle recipe cannot be switched from Kimi, as in Hermes, and Nemotron 3.5 Bolt's `medium_kwargs` cannot be sent, because the wire carries no `chat_template_kwargs` (E2 never saw one).
Kimi replays reasoning verbatim under the field the server used, `reasoning_content` from llama.cpp and SGLang and `reasoning` from vLLM, and keeps `reasoning_effort` unchanged after a reply with reasoning (E2 `e2r-*` and `e2c-*` runs).
gpt-oss-20b cannot stop reasoning, so it has no `off_effort`, and Kimi refuses thinking off for it with exit 1 in 0.53 s, before any request: `Model '<name>' reasons by default but declares no off effort, so thinking cannot be turned off.` (E2 `e2t-*-gpt-oss-20b-off` runs); llmctl never sends another effort in its place.
`/effort` in the TUI offered `Off  Low  [ Medium ]` for Qwen3.8-27B (P3); for gpt-oss-20b its levels follow from `support_efforts` (inferred from main.cjs:481166-481180, not run).

Your global `[thinking]` table in `config.toml` reaches llmctl's entries too:

- `enabled = false`, or `effort = "off"`: Qwen3.8-27B sends `reasoning_effort: "none"`, gpt-oss-20b is refused before any request, and an entry without the `thinking` capability sends nothing (E2) (measured on the MSI Raider 18 Max HX, Kimi Code 2.5.3, recording stub).
- `effort = "<level>"` with `enabled` not false: print mode sends that level verbatim as `reasoning_effort` to every entry, also outside its `support_efforts` and to an entry without the `thinking` capability: `high`, `max` and `minimal` to Qwen3.8-27B, `max` to gpt-oss-20b and `high` to Qwen3.6-35B-A3B, each with exit 0 (P2, [summary](../research/2026-11-17/llmctl-kimi-stub/spec/runs/p2-summary.jsonl)) (measured on the MSI Raider 18 Max HX, Kimi Code 2.5.3, recording stub); Kimi validates no effort on the `openai` type (inferred from main.cjs:204832-204841 and 287940-287943).
- Kimi's TUI writes this table itself: a `/model` pick of an entry without `thinking` appended `enabled = false` (E4a), `/effort` Medium saved `enabled = true` and printed `Saved qwen3.8-27b on llama.cpp (llmctl) with thinking medium as default.`, and `/effort` Low saved `enabled = true` and `effort = "low"`, which the next request sent (P3, [Medium](../research/2026-11-17/llmctl-kimi-stub/spec/runs/p3-tui-effort-qwen38/), [Low](../research/2026-11-17/llmctl-kimi-stub/spec/runs/p3-tui-effort-low-qwen38/)) (measured on the MSI Raider 18 Max HX, Kimi Code 2.5.3, recording stub).
- `forced_effort` in `[thinking]` and the variable `KIMI_MODEL_THINKING_EFFORT` override the effort for every model (local-K17; main.cjs:287932-287939, not run); llmctl does not read them.

llmctl never writes `[thinking]`; instead `up`, after the Kimi lines, and `status`, while the running model's entry is the file's, print one line from this table, the first row that matches winning:

| `[thinking]` in the file | Running recipe | Line |
|---|---|---|
| `enabled = false`, or `effort = "off"` | effort, efforts without `none` | `warning: [thinking] enabled = false in <path> makes Kimi refuse every prompt to llmctl/<served> before sending it, ...` |
| `enabled = false`, or `effort = "off"` | effort, efforts with `none` | `warning: [thinking] enabled = false in <path> makes Kimi send reasoning_effort none to llmctl/<served>, so <id> runs without thinking; ...` |
| `effort` is a string other than `on` and `off`, and not one of the recipe's efforts | effort | `warning: [thinking] effort = "<x>" in <path> makes Kimi send reasoning_effort <x> to llmctl/<served>, which lists only <efforts>; ...` |
| `effort` is a string other than `on`, `off` and `none` | toggle, on SGLang | `warning: [thinking] effort = "<x>" in <path> makes Kimi send reasoning_effort <x> to llmctl/<served>, which turns thinking on for <id>, because SGLang reads every effort except none as thinking on; ...` |
| `effort` is a string other than `on` and `off` | toggle, always or none | `warning: [thinking] effort = "<x>" in <path> makes Kimi send reasoning_effort <x> to llmctl/<served>, which sets its thinking on the server (<type>); ...` |
| `effort` is one of the recipe's efforts and not `medium` | effort | `note: [thinking] effort = "<x>" in <path> makes Kimi use effort <x> for llmctl/<served> instead of medium` |
| Anything else, or no table | any | Nothing |

Each warning names its fix: `set enabled = true there`, or another effort with `/effort` in Kimi, or removing the effort line.
The SGLang row exists because SGLang v0.5.23 turns `enable_thinking` on for any `reasoning_effort` other than `none`, which on Qwen3.5-9B turned thinking on although the server default is off, while `none` kept it off (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17; [sglang/README.md](../sglang/README.md#tool-calling-and-reasoning)); other thinking types were not measured that way, so they keep the general row.
Kimi's effort selector saves the choice for every model; to choose an effort for llmctl's model in one session only, press Alt+S in the selector, which the TUI shows as `Alt+S session-only` (P3), and leave the saved table alone unless you want that effort for every model.
Probe G of the e2e sent the plain OK request with `reasoning_effort: "low"`, the field a global `[thinking] effort = "low"` makes Kimi send, to every recipe, a level that only Qwen3.8-27B and gpt-oss-20b list, and every server answered with status 200 and content ([table](#verified-on-this-laptop)) (measured on the MSI Raider 18 Max HX, llama.cpp b12407 and vLLM v0.33.0, 2026-11-17).
On llama.cpp no recipe whose server default is off returned reasoning, as without the field in probe D; on vLLM Qwen3.5-9B, Gemma 4 12B and Groot-19B-A3B returned reasoning, Gemma 4 E4B did not, and the run recorded no control without the field for them, so this does not show that the field caused it; K2 Horizon 7B on vLLM returned none.

### The Kimi default model

The rules of [llmctl.md](llmctl.md#defaults-and-the-follow-rule) apply to `default_model`:

1. With `--set-default` or `LLMCTL_SET_DEFAULT=1`, llmctl writes `default_model = "llmctl/<served name>"`.
2. Without it, a string `default_model` that starts with `llmctl/` follows the running alias, and any other value, or none, never changes.
3. Every recipe gets an entry, so Kimi can always point at the running model.
4. `down` changes nothing; it names Kimi when its default starts with `llmctl/` and prints how long Kimi retries a stopped server ([Kimi sessions, a stopped server and timeouts](#kimi-sessions-a-stopped-server-and-timeouts)).

Kimi's `/model` and `/effort` save `default_model` themselves (local-K16, P3), so a default you pick in Kimi for llmctl's model then follows every `up`.
llmctl changes a one-line string value in place and keeps the key text, the spacing around `=` and a comment after it, as Kimi's own edit of an existing default does (E4a); a missing key is inserted after the last top-level statement, or as the first line when there is none, because a line at the end of the file would belong to the last table: Kimi accepted such a line silently as `defaultModel` inside `[models."llmctl/local"]` (E6, [negative records](../research/2026-11-17/llmctl-kimi-stub/exp/neg/)) (measured on the MSI Raider 18 Max HX, Kimi Code 2.5.3, no server).

The default matters more for Kimi than for the other clients, because print mode resolves `default_model` even when `-m` names another alias (E5, [summary](../research/2026-11-17/llmctl-kimi-stub/exp/runs/e5e7-summary-e5.jsonl); main.cjs:499744-499746 and 261009-261018) (measured on the MSI Raider 18 Max HX, Kimi Code 2.5.3, recording stub):

- without a `default_model`, `kimi -p ... -m llmctl/local` fails with `no default model configured`;
- a `default_model` that names an alias the file lacks fails every `kimi -p`, with or without `-m`, with `Model "<x>" is not configured in config.toml.` or `model <x> does not resolve to a configured provider`, in about 0.5 s and before any request;
- the TUI with `-m` works without a default (E8 `e8-tui-m`), and shows `Model: not set, run /login or /provider` when there is none.

So `up` prints, for a model that is not Kimi's default, that `kimi -p` needs `--set-default` first, and `up` and `status` warn when `default_model` names an alias the file does not define: `warning: Kimi's default_model <x> names no model in <path>, so every kimi -p fails (...); pick a model with /model in kimi, or run ./llmctl up <model> --set-default`.
The follow rule keeps an llmctl default from naming a missing alias when the alias changes between `llmctl/local` on llama.cpp and `llmctl/<served name>` on vLLM or SGLang.

`KIMI_MODEL_NAME` in the shell outranks `default_model`, and only `-m` outranks it (docs-K031) (vendor-reported); it activates an alias of its own, `__kimi_env_model__`, on a provider of its own whose id is `__kimi_env__` and whose type is `KIMI_MODEL_PROVIDER_TYPE`, `kimi` by default, and neither is ever written to the file (local-K32).
`up` and `status` print `note: KIMI_MODEL_NAME is set in this shell, so kimi uses that model instead of Kimi's default_model; unset it, or pass -m llmctl/<served>` when llmctl's environment has it set.
Other `KIMI_MODEL_*` variables, such as `KIMI_MODEL_TEMPERATURE` and `KIMI_MODEL_MAX_TOKENS`, change Kimi's requests even without `KIMI_MODEL_NAME` (local-K32).

### Kimi checks before writing

Before anything starts, and again after the server is ready, llmctl checks its own result: the candidate parses with `tomllib`, llmctl's two tables and `default_model` parse to the intended values, no other `llmctl/` alias exists, everything else parses equal to the old file, and every line outside llmctl's tables and the `default_model` statement is the same, in order; otherwise `up` refuses with `<path>: llmctl's own check of the new config.toml failed; nothing was written`.

Then the installed Kimi reads the candidate, on every `up` that is not a dry run, also when llmctl's tables do not change, so a file from which Kimi no longer reads llmctl's entry after your edit, Kimi's own write or an update refuses:

1. llmctl makes a private temporary folder (`llmctl-kimi-*`, mode 0700) with `home/config.toml` (mode 0600, the whole candidate, your other tables included) and an empty `cache/`, and runs both commands with `KIMI_CODE_HOME` and `KIMI_CODE_CACHE_DIR` pointing there and `KIMI_DISABLE_TELEMETRY=1`, `KIMI_CODE_NO_AUTO_UPDATE=1` and `KIMI_CODE_REGION_MARKER=off`; the folder is removed afterwards.
2. `kimi --version`: a failure, or no output, refuses; a version other than 2.5.3 prints a warning after the clients line, and the check goes on.
3. `kimi provider list --json`: it passes only when it exits 0, its output is a JSON object, and the `llmctl` provider and the `llmctl/<served name>` model hold every value llmctl writes under Kimi's camelCase names (`baseUrl`, `apiKey`, `maxContextSize`, `supportEfforts` and the others, as `providerEntryFromToml` converts them, main.cjs:169306-169314); keys llmctl does not write are ignored.

The exit status alone proves nothing: on a file that is not valid TOML, `kimi provider list --json` exits 0 with `{"providers": {}, "models": {}}` and an empty stderr (E6, [record](../research/2026-11-17/llmctl-kimi-stub/exp/records/provider-list-invalid-toml.json)).
Its output holds every provider's `apiKey`, so llmctl never prints, logs or records it, and a refusal names only the llmctl key that is missing or differs.
`kimi --version` took 0.27 to 0.29 s and `kimi provider list --json` 0.31 to 0.32 s, with a fresh or a warm cache; a fresh cache folder got 9 files with 1,932,692 bytes and the home a 36-byte `device_id`, and neither command made a connect call other than four to the local `nscd` socket (E6, [summary](../research/2026-11-17/llmctl-kimi-stub/exp/runs/e6-summary.jsonl)) (measured on the MSI Raider 18 Max HX, Kimi Code 2.5.3, no server).
So one check costs about 0.6 s and writes about 1.9 MB into a folder that it removes.
`kimi doctor config` is not used, because it accepted a model without `max_context_size`, an unknown provider type and an unquoted dotted alias (local-K23; E6).

What the check proves: Kimi 2.5.3's own parser and schemas read llmctl's values from the whole candidate, and llmctl changed nothing else.
What it does not prove: that a request succeeds, which E2, the golden files and the e2e cover.
`up --dry-run` runs no `kimi` command and creates no folder; when Kimi is the only checking client it ends with `dry run: nothing was written, and Kimi has not checked llmctl's entry`, and with Pi or dsh also active it ends with one of the other endings in [llmctl.md](llmctl.md#clients), such as `dry run: nothing was written, and none of Pi, dsh and Kimi has checked llmctl's entries` on this laptop.

### Kimi chat-only recipes

Kimi has no per-model tool switch: `tool_use` gates only dynamic tool selection, and every request carries Kimi's 24 tools, the two DeepSeek-R1 entries with `capabilities = []` included (E2; local-K22).
On vLLM, whose DeepSeek-R1 recipe has no tool parser, a request with tools gets HTTP 400, and Kimi exited 1 after 0.61 s without retrying against a stub that answers as that server does (E3, [summary](../research/2026-11-17/llmctl-kimi-stub/exp/runs/e3-summary.jsonl)) (measured on the MSI Raider 18 Max HX, Kimi Code 2.5.3, recording stub).
SGLang ships no chat-only recipe; an SGLang recipe without a tool parser would answer a request with tools with HTTP 200 and any call as text instead (inferred from serving_chat.py, [sglang/README.md](../sglang/README.md#tool-calling-and-reasoning)).
As for dsh, llmctl writes both DeepSeek-R1 entries anyway, so Kimi's default stays on the running model, and `up` prints a chat-only line.

A prompt without tools works in print mode with an agent file whose front matter has `tools: []`: the request then has no `tools` key, and the agent's body is its system prompt (E3).
llmctl ships that agent file as `launcher/kimi-chat-only.md`, and `up` prints its absolute path:

```markdown
---
name: llmctl-chat-only
description: Chat-only agent for llmctl's chat-only recipes; it has no tools.
tools: []
---

You are a helpful assistant.
Answer the user directly in plain text.
You have no tools.
```

```bash
kimi -p "your prompt" --agent-file <path>
```

The name is kebab-case, as Kimi requires, and does not clash with an agent of yours called `chat-only` (main.cjs:69803-69835).
P1 ran the same text with the body's three sentences on one line, joined by single spaces: `kimi -p "Reply with exactly OK." --output-format stream-json --agent-file <file>` sent no `tools`, an 87-character system prompt, answered `PONG-ONE` and exited 0 in 0.49 s (P1, [summary](../research/2026-11-17/llmctl-kimi-stub/spec/runs/p1-summary.jsonl), [agent file](../research/2026-11-17/llmctl-kimi-stub/spec/agents/llmctl-chat-only.md)) (measured on the MSI Raider 18 Max HX, Kimi Code 2.5.3, recording stub).
The shipped file puts each sentence on its own line, which gives the model the same 87 characters with two line breaks in place of the two spaces (not measured).
The TUI ignores `--agent-file`: with it, with `-m` or with a default, the TUI sent all 24 tools and Kimi's default system prompt and showed `Error: [provider.api_error] "auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set`, as the control run without the file did (P1).
So `up` prints only the print-mode route, which needs Kimi's default ([The Kimi default model](#the-kimi-default-model)), and llmctl writes nothing into Kimi's `agents/` folder; on llama.cpp the TUI sends the tools too, and the model may write a tool call as text, as it did in OpenCode (inferred from the recipe's evidence).

### Kimi sessions, a stopped server and timeouts

Kimi watches `config.toml` and reloads it when it changes (local-K46); an open TUI picked up llmctl's rename within 5 s (E8) (measured on the MSI Raider 18 Max HX, Kimi Code 2.5.3, recording stub):

- When the alias changed, the next prompt failed with `Error: [config.invalid] Model "llmctl/local" is not configured in config.toml.`, sent no request, and the footer still named the old model (E8, [run](../research/2026-11-17/llmctl-kimi-stub/exp/runs/e8-tui/)).
- When the alias stayed, the next prompt went out with the new entry's values but the session's old thinking level, `reasoning_effort: "none"` after a toggle model's `off`, while the footer still showed the old model (E8, [run](../research/2026-11-17/llmctl-kimi-stub/exp/runs/e8-tui-same/)).
- In neither case did the TUI write the file back.

So after an `up` that changed llmctl's tables, `up` prints `note: an open Kimi session on llmctl/<old> fails its next turn until you pick llmctl/<served> with /model or start a new session`, or, when the alias stayed, `note: an open Kimi session on llmctl/<served> sends its next turn to <id> on <runtime> but keeps its old thinking level; start a new session`.

The lead's TUI checklist after the e2e drove Kimi's TUI through a pseudo-terminal against the real servers, in a private home whose `config.toml` was a copy of the real one with llmctl's default edit, through a recording proxy that kept every entry's `base_url` ([checklist](../research/2026-11-17/llmctl-kimi-e2e/lead-checks/tui/README.md)); all five items passed:

1. `/model` listed `gemma-4-26b-a4b on llama.cpp (llmctl)` with `Thinking  On (Unsupported)  [ Off ]`, and Enter picked it with `Saved gemma-4-26b-a4b on llama.cpp (llmctl) with thinking off as default.`, appending `[thinking]` with `enabled = false` to the private file, as E4a found (measured on the MSI Raider 18 Max HX, Kimi Code 2.5.3, llama.cpp b12407).
2. A prompt got the answer from the real server in about 4 s: one request with `model: "local"`, `max_tokens: 16384`, 24 tools and no `reasoning_effort`, and one task with a 12,109-token prompt in the llama.cpp log (measured on the MSI Raider 18 Max HX, Kimi Code 2.5.3, llama.cpp b12407).
3. After `./llmctl up qwen3.8-27b --replace`, `/effort` offered `Off  Low  [ Medium ]` with `Alt+S session-only`; Low with Alt+S printed `Thinking set to low for this session only.`, the request carried `reasoning_effort: "low"`, and the private `config.toml` stayed byte for byte as copied (measured on the MSI Raider 18 Max HX, Kimi Code 2.5.3, llama.cpp b12407).
4. Alias changed: after `./llmctl up gpt-oss-20b --runtime vllm --replace`, an open session on `llmctl/local` failed its next turn with `Error: [config.invalid] Model "llmctl/local" is not configured in config.toml.`, no request reached either server, the footer still named qwen3.8-27b, and the TUI did not write the file (measured on the MSI Raider 18 Max HX, Kimi Code 2.5.3, llama.cpp b12407 and vLLM v0.33.0).
5. Alias kept: after `./llmctl up gemma-4-26b-a4b --replace`, an open session on `llmctl/local` sent its next turn to gemma with the old `reasoning_effort: "medium"` and got the answer, and the TUI did not write the file; the footer still showed `qwen3.8-27b on llama.cpp (llmctl) thinking: medium`, but its context total changed from 64k to 128k, where E8 saw the footer keep the old values (measured on the MSI Raider 18 Max HX, Kimi Code 2.5.3, llama.cpp b12407).

The notes `up` printed before each switch matched what the sessions did, and at the end `./llmctl status` showed Kimi on `llmctl/local for gemma-4-26b-a4b on llama.cpp, the running model; default not set`.

A stopped server fails a `kimi -p` after 151.2 s of 10 connection attempts, with nine `turn.step.retrying` lines in stream-json and `error: failed to run prompt: provider.connection_error: Connection error.` (E7, [run](../research/2026-11-17/llmctl-kimi-stub/exp/runs/e7-refused-json/)) (measured on the MSI Raider 18 Max HX, Kimi Code 2.5.3, no server); `down` prints `Kimi retries a stopped server for about two and a half minutes before a prompt fails` when Kimi's default points at llmctl.
Kimi waits 300 s for the first response byte unless `KIMI_CODE_LLM_HEADERS_TIMEOUT_MS` is exported, and 10 minutes per attempt (docs-K055, docs-K056, local-K37) (vendor-reported); neither can be set in `config.toml`, and llmctl sets neither, so a long first prefill of an offloaded model may need the variable.

Kimi sends `max_tokens` as the smaller of `max_output_size` and `max_context_size` minus the tokens used, at most 131072 (local-K26; E2: 16384, and 32768 for llama.cpp K2 Horizon 7B).
Kimi's first request in the hands-on run `a-1` carried 12,202 prompt tokens (gemma-4-26b-a4b; [README](../research/2026-11-17/llmctl-kimi-stub/handson/README.md)) (measured on the MSI Raider 18 Max HX, Kimi Code 2.5.3, llama.cpp b12407), so a 65,536-token recipe keeps about 37,000 tokens for the conversation after the 16,384-token output reserve (estimate).
The docs tie Kimi's automatic compaction to `reserved_context_size` (docs-K081) (vendor-reported), which llmctl does not write; when it compacts with llmctl's entries was not measured.

### Kimi privacy and side effects

What Kimi Code 2.5.3 does beyond the model requests, from its docs and its bundle; llmctl writes none of the switches into Kimi's files:

| Item | What happens | Off switch | Evidence |
|---|---|---|---|
| Telemetry | On by default; events go by POST to the region's collector, and failed batches wait in `telemetry/` | `telemetry = false` in `config.toml`, or `KIMI_DISABLE_TELEMETRY=1`, which wins over `telemetry = true` and which print mode honours | docs-K142 to docs-K145 (vendor-reported); local-K40, local-K44 |
| Device id | `device_id` (0600) is created on the first run and sent as `X-Msh-Device-Id` only to `kimi`-type providers | Not needed for llmctl's provider; no hands-on request carried an `X-Msh-*` header | local-K29, local-K35, local-K42; hands-on [README](../research/2026-11-17/llmctl-kimi-stub/handson/README.md) |
| Region | Chosen locally from the login, a `region` marker in the Kimi home, the time zone and the locale, else `mainland-cn`, whose hosts are on kimi.com; `global` uses kimi.ai | `KIMI_CODE_REGION_MARKER=off` turns off the marker and the local inference, which falls back to `mainland-cn` | local-K44; docs-K168 (vendor-reported) |
| Update check | A request for `latest.json` from Kimi's CDN at every interactive and `-p` start, a background install, and a binary swap at the next start | `KIMI_CODE_NO_AUTO_UPDATE=1` stops all three; `[upgrade] auto_install = false` in `tui.toml` stops only the install and the swap | local-K41; docs-K161 to docs-K164 (vendor-reported) |
| Client configs | A POST to the managed API's `client_configs`, also without a login, cached for 24 h | None found; print mode did not call it in the hands-on runs, so it is likely a TUI path (inferred) | local-K43, local-K44 |
| ripgrep, fd | Downloaded from Kimi's CDN into the Kimi home when `rg`, or `fd` and `fdfind`, are not on `PATH` and Kimi's Grep or Glob runs | Install `ripgrep` and `fd-find` | local-K43 |
| FetchURL | Fetches directly from this machine without a login, and through Kimi's service with one | No login, or drop the tool in the global `[tools]` | local-K47; docs-K123, docs-K156 (vendor-reported) |
| Session titles | Generated only through the managed Kimi provider, so none without a login; with a login, the TUI's titles would send the session's prompts to Kimi even for a local model (inferred from local-K47) | `auto_session_title = false` | local-K47; docs-K154 on the kimi.com server API page, docs-K155 (vendor-reported) |
| `/feedback` | Can upload diagnostic logs and codebase context, which the command offers as optional | Do not use it, or leave them out | docs-K152 (vendor-reported) |
| Session files | Sessions keep `wire.jsonl` request traces with whole system prompts and tool schemas, and may hold prompts, command output and traces of credentials | None; they stay in the Kimi home | local-K42; docs-K131, docs-K138 (vendor-reported) |
| Native assets | Written under `~/.cache/kimi-code/native/<version>/`, or `KIMI_CODE_CACHE_DIR`, at every start, `--version` included | None | local-K11, local-K12 |

To keep Kimi's print mode (`kimi -p`) on this machine with llmctl's model: set `telemetry = false` in `config.toml`, export `KIMI_CODE_NO_AUTO_UPDATE=1` (and `KIMI_DISABLE_TELEMETRY=1` if you prefer the variable), do not log in, and install ripgrep and fd from apt with `sudo apt install ripgrep fd-find`, Ubuntu's `fd-find` providing `fdfind`.
This does not keep the TUI local: an idle TUI with `KIMI_DISABLE_TELEMETRY=1`, `KIMI_CODE_NO_AUTO_UPDATE=1` and `KIMI_CODE_REGION_MARKER=off` set still looked up `api.kimi.com` and `code.kimi.com` (E8, the idle-TUI sentence later in this section).
llmctl's own `kimi` commands set `KIMI_DISABLE_TELEMETRY=1`, `KIMI_CODE_NO_AUTO_UPDATE=1` and `KIMI_CODE_REGION_MARKER=off`, and llmctl never uses a region.
Which region Kimi picks on a given machine, and whether to change it, is left open here: llmctl reads no region, and print mode with the switches set contacted no kimi.com host.

With the switches set, print mode made no DNS lookup and no outside connection: the hands-on runs' only connect target was 127.0.0.1, besides failed lookups of the local `nscd` socket ([README](../research/2026-11-17/llmctl-kimi-stub/handson/README.md)) (measured on the MSI Raider 18 Max HX, Kimi Code 2.5.3, llama.cpp b12407).
The experiments agree: over all 67 straced runs, in an offline sandbox, no print-mode, `provider`, `doctor` or `kimi web` run made a DNS lookup, and the only run that did was the idle TUI below ([How every run worked](../research/2026-11-17/llmctl-kimi-stub/exp/findings.md#how-every-run-worked)) (measured on the MSI Raider 18 Max HX, Kimi Code 2.5.3, recording stub).
Without the switches, as the owner runs Kimi, the lead's L-c check after the e2e ran one print-mode session with only `KIMI_CODE_NO_AUTO_UPDATE=1` from the environment, `KIMI_DISABLE_TELEMETRY` and `KIMI_CODE_REGION_MARKER` unset, and `telemetry = false` from a byte copy of the owner's `config.toml`, against gemma-4-26b-a4b ([L-c](../research/2026-11-17/llmctl-kimi-e2e/lead-checks/README.md)).
It answered `OK` with exit 0 and made one connect, to 127.0.0.1:8080, and no other connect besides two failed ones to the missing `nscd` socket, no `sendto` call and no DNS lookup (measured on the MSI Raider 18 Max HX, Kimi Code 2.5.3, llama.cpp b12407).
Those runs used no Grep, Glob or TUI, so they do not cover the ripgrep and fd downloads or the client configs.
An idle TUI, started with all three switches, looked up `api.kimi.com` 12 times and `code.kimi.com` 4 times in 12 s (E8, [strace](../research/2026-11-17/llmctl-kimi-stub/exp/runs/e8-tui-idle-strace/strace.txt)) (measured on the MSI Raider 18 Max HX, Kimi Code 2.5.3, recording stub), so the switches do not keep the TUI off the network; the purposes, client configs and the CDN, are inferred from local-K43 and local-K44.

Print mode approves every tool call, sensitive files included, and leaves out the dangerous-command guard (docs-K100, docs-K114 (vendor-reported); local-K49), and 2.5.3 may not apply `config.toml` permission rules at session start in print mode, which 2.6.0 lists as a fix (docs-K111) (vendor-reported).
Kimi's Bash tool runs `/bin/bash -c` as a plain child process, with no sandbox and no systemd scope (local-K48), so unlike dsh's commands its commands stay under heavy's cap when `kimi` itself runs under `heavy`: a command run that way reported a scope inside `heavy.slice` ([run](../research/2026-11-17/llmctl-kimi-stub/handson/runs/c-1/)) (measured on the MSI Raider 18 Max HX, Kimi Code 2.5.3, llama.cpp b12407).

## Silent fallbacks and exit codes

These are the ways a client can answer from the wrong model without an error, and what guards each one:

- Pi with a `defaultModel` that `models.json` no longer has uses another provider's model with exit 0 (S3); llmctl moves a default that points at llmctl on every `up`, and never removes its entry on `down`.
- Pi with a `models.json` that fails its schema drops the whole file (`dist/core/model-config.js:144-150`, E7 `neg-format`); llmctl lets Pi check the candidate file first and refuses when the llmctl row is missing.
- Pi with a comment in `settings.json` ignores the whole file (S4); llmctl edits only plain JSON and refuses the rest.
- Hermes with a duplicate key in a profile serves it on default settings inside your gateway (E3); llmctl refuses an owned key outside its block, and checks its own result before writing.
- A Hermes or Pi session that was open before an `up` may keep the entry it started with ([Hermes](#sessions-and-interactive-paths), [Pi](#defaults-sessions-and-timeouts)); `up` prints a note to start a new session when an entry changed.
- dsh with a renamed row id would leave llmctl's row unapplied and answer from its own default provider; the dsh check before each write refuses when the installed dsh lacks one of the three ids or did not apply a row ([Checks before writing](#checks-before-writing)).
- dsh with an `llm-pi-ai` row in a profile patch hides that row behind llmctl's home row without a message (A2-home); llmctl refuses until the row is gone ([Moving a hand-made setup](#moving-a-hand-made-setup)).
- dsh's model picker cannot save another default while llmctl holds dsh's default, and says nothing; `up` prints a note when it takes the default ([The default model and the picker](#the-default-model-and-the-picker)).
- Kimi with `KIMI_MODEL_NAME` set in the shell uses that model instead of `default_model` (docs-K031, local-K32); `up` and `status` print a note when llmctl's environment has it set ([The Kimi default model](#the-kimi-default-model)).
- Kimi with a global `[thinking] effort` sends that level verbatim to every entry, also to one that does not list it or has no `thinking` capability (Kimi's P2); `up` and `status` warn ([Kimi thinking per type](#kimi-thinking-per-type)).
- `kimi provider list --json` exits 0 with empty maps on a `config.toml` it cannot read as TOML (Kimi's E6); llmctl's Kimi check therefore reads llmctl's values in its output, not the exit status ([Kimi checks before writing](#kimi-checks-before-writing)).
- Kimi's TUI ignores `--agent-file` and sends its tools for a chat-only recipe too (Kimi's P1); `up` prints only the print-mode route ([Kimi chat-only recipes](#kimi-chat-only-recipes)).
- A Kimi `default_model` that names an alias the file lacks fails every `kimi -p` loudly, with or without `-m` (Kimi's E5), rather than falling back; the follow rule and the missing-alias warning keep it from happening through llmctl ([The Kimi default model](#the-kimi-default-model)).

Exit codes: `pi --mode json` exits 0 when the provider fails, because only the text mode turns a failed reply into exit 1 (`dist/modes/print-mode.js:17` and `103-139`, `dist/main.js:868-879`), so look for an assistant `message_end` with `stopReason` `error`; Hermes exits 2 when it refuses a context below 64,000 tokens (R19).
`dsh --profile headless` exits 1 when a turn fails or is aborted and prints `dsh: <code>: <message>` on stderr only for a failed turn, and it exits 1 with an empty line when the run produced no assistant message (`runtime-13`; [headless/README.md:36](https://github.com/deepseek-ai/deepseek-harness/blob/38a6a5c965aa99ab525161efb7cc5cf1fe62fced/packages/bundle/headless/README.md?plain=1#L36), [136](https://github.com/deepseek-ai/deepseek-harness/blob/38a6a5c965aa99ab525161efb7cc5cf1fe62fced/packages/bundle/headless/README.md?plain=1#L136) and [190](https://github.com/deepseek-ai/deepseek-harness/blob/38a6a5c965aa99ab525161efb7cc5cf1fe62fced/packages/bundle/headless/README.md?plain=1#L190)); with `--json`, a failure outside a turn writes an `error` event and no `final` event (`runtime-14`, [104](https://github.com/deepseek-ai/deepseek-harness/blob/38a6a5c965aa99ab525161efb7cc5cf1fe62fced/packages/bundle/headless/README.md?plain=1#L104)).
`kimi -p` exits 0 on success, 1 for a startup or turn error (a missing model, a refused request, a stopped server after its retries), 4 when the session is occupied, and 129, 130 or 143 on signals; in `--output-format stream-json` a retry shows as a `turn.step.retrying` line, and thinking is never printed (local-K39, docs-K101).
llmctl exits 3 when a client file refuses the start, and 6 when the model is running but a client file or command failed ([llmctl.md](llmctl.md#when-up-refuses)).
llmctl waits up to the runtime's `start_timeout_s` for the server, then 2 s at most for Pi's settings lock.

## Verified on this laptop

On 2026-11-18, from 05:24:55 to 06:16:47 +03:00, the driver ran every shipped recipe over the three runtimes: `--matrix all --set-default-round-trip` over 29 recipes, 15 on llama.cpp, 8 on vLLM and 6 on SGLang, with OpenCode, Hermes, Pi, dsh and Kimi, 510 rows (measured on the MSI Raider 18 Max HX, llama.cpp b12407, vLLM v0.33.0 and SGLang v0.5.23, 2026-11-18); [README](../research/2026-11-18/llmctl-sglang-e2e/README.md), [summary](../research/2026-11-18/llmctl-sglang-e2e/summary.md), [records](../research/2026-11-18/llmctl-sglang-e2e/records/).
490 rows read Yes: every `up`, `status`, `long-grammar`, `config`, `hermes-config`, `dsh-config`, `dsh-web`, `kimi-config`, `gateway`, `probes`, `restore` and `down` row on all three runtimes, and the round trip's five sessions without a model flag.
The 20 No rows are client sessions, which the README classifies ([The No rows](../research/2026-11-18/llmctl-sglang-e2e/README.md#the-no-rows)): 8 failures the same model and client also had on 2026-11-17, 4 of the same models on their new SGLang recipes (K2 Horizon 7B in `hermes -z` and `hermes chat`, Gemma 4 E4B in dsh, Qwen3.6-35B-A3B in `hermes -z`), 5 model errors in sessions that passed on 2026-11-17, one Kimi retry after Mellum2.1's thinking-only reply on llama.cpp, the vLLM DeepSeek-R1 OpenCode session, whose entry was then removed ([opencode.md](opencode.md#the-validated-config)), and gpt-oss-20b's SGLang OpenCode session, which ended after a reply that SGLang's tool parser had emptied ([sglang/README.md](../sglang/README.md#known-issues)).

The long-grammar step passed on all six SGLang recipes: a prompt of 90% of the context, 58,009 to 115,772 tokens, then Hermes' title request with a strict `json_schema` and a forced tool call, all answered HTTP 200 with a title that parses, and `/ready` answered after them ([table](../research/2026-11-18/llmctl-sglang-e2e/README.md#long-grammar-step-on-sglang)) (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
The lowest free GPU memory during the step was 493 MiB on Qwen3.6-35B-A3B and 1,627 MiB on Gemma 4 E4B, most of each drop came with the first, short calibration request, and the title and the tool call moved no memory, so no server came near the grammar crash of [sglang/README.md](../sglang/README.md#grammar-requests-and-nccl).

Probe D, the plain OK request with `max_tokens` 256 and the recipe id as its model, went to llama.cpp and SGLang; probe G, the same request with the served name and `reasoning_effort: "low"`, went to every recipe; both are recorded without judging, and every one got HTTP 200 (measured on the MSI Raider 18 Max HX, llama.cpp b12407, vLLM v0.33.0 and SGLang v0.5.23, 2026-11-18); the thinking type is the recipe's in [Kimi thinking per type](#kimi-thinking-per-type):

| Recipe | Runtime | Thinking type | D content | D reasoning | G content | G reasoning | Records |
|---|---|---|---|---|---|---|---|
| qwen3.8-27b | llama.cpp | effort | Yes | Yes | Yes | Yes | [024](../research/2026-11-18/llmctl-sglang-e2e/records/024-01-qwen3.8-27b-llamacpp-probes-D), [025](../research/2026-11-18/llmctl-sglang-e2e/records/025-01-qwen3.8-27b-llamacpp-probes-G) |
| k2-horizon-7b | llama.cpp | always | Yes | Yes | Yes | No | [044](../research/2026-11-18/llmctl-sglang-e2e/records/044-02-k2-horizon-7b-llamacpp-probes-D), [046](../research/2026-11-18/llmctl-sglang-e2e/records/046-02-k2-horizon-7b-llamacpp-probes-G) |
| nemotron-3.5-bolt | llama.cpp | toggle, default off | Yes | No | Yes | No | [064](../research/2026-11-18/llmctl-sglang-e2e/records/064-03-nemotron-3.5-bolt-llamacpp-probes-D), [066](../research/2026-11-18/llmctl-sglang-e2e/records/066-03-nemotron-3.5-bolt-llamacpp-probes-G) |
| qwen3-coder-next | llama.cpp | none | Yes | No | Yes | No | [084](../research/2026-11-18/llmctl-sglang-e2e/records/084-04-qwen3-coder-next-llamacpp-probes-D), [085](../research/2026-11-18/llmctl-sglang-e2e/records/085-04-qwen3-coder-next-llamacpp-probes-G) |
| deepseek-r1-0528-qwen3-8b | llama.cpp | always | Yes | Yes | Yes | Yes | [103](../research/2026-11-18/llmctl-sglang-e2e/records/103-05-deepseek-r1-0528-qwen3-8b-llamacpp-probes-D), [104](../research/2026-11-18/llmctl-sglang-e2e/records/104-05-deepseek-r1-0528-qwen3-8b-llamacpp-probes-G) |
| gemma-4-26b-a4b | llama.cpp | toggle, default off | Yes | No | Yes | No | [122](../research/2026-11-18/llmctl-sglang-e2e/records/122-06-gemma-4-26b-a4b-llamacpp-probes-D), [123](../research/2026-11-18/llmctl-sglang-e2e/records/123-06-gemma-4-26b-a4b-llamacpp-probes-G) |
| nemotron-3.5-lightning | llama.cpp | toggle, default off | Yes | No | Yes | No | [141](../research/2026-11-18/llmctl-sglang-e2e/records/141-07-nemotron-3.5-lightning-llamacpp-probes-D), [142](../research/2026-11-18/llmctl-sglang-e2e/records/142-07-nemotron-3.5-lightning-llamacpp-probes-G) |
| groot-19b-a3b | llama.cpp | toggle, default off | Yes | No | Yes | No | [160](../research/2026-11-18/llmctl-sglang-e2e/records/160-08-groot-19b-a3b-llamacpp-probes-D), [161](../research/2026-11-18/llmctl-sglang-e2e/records/161-08-groot-19b-a3b-llamacpp-probes-G) |
| k2-horizon-mova-36b | llama.cpp | always | Yes | Yes | Yes | Yes | [180](../research/2026-11-18/llmctl-sglang-e2e/records/180-09-k2-horizon-mova-36b-llamacpp-probes-D), [181](../research/2026-11-18/llmctl-sglang-e2e/records/181-09-k2-horizon-mova-36b-llamacpp-probes-G) |
| gpt-oss-20b | llama.cpp | effort | Yes | Yes | Yes | Yes | [199](../research/2026-11-18/llmctl-sglang-e2e/records/199-10-gpt-oss-20b-llamacpp-probes-D), [200](../research/2026-11-18/llmctl-sglang-e2e/records/200-10-gpt-oss-20b-llamacpp-probes-G) |
| qwen3.5-9b | llama.cpp | toggle, default off | Yes | No | Yes | No | [218](../research/2026-11-18/llmctl-sglang-e2e/records/218-11-qwen3.5-9b-llamacpp-probes-D), [219](../research/2026-11-18/llmctl-sglang-e2e/records/219-11-qwen3.5-9b-llamacpp-probes-G) |
| gemma-4-12b | llama.cpp | toggle, default off | Yes | No | Yes | No | [237](../research/2026-11-18/llmctl-sglang-e2e/records/237-12-gemma-4-12b-llamacpp-probes-D), [238](../research/2026-11-18/llmctl-sglang-e2e/records/238-12-gemma-4-12b-llamacpp-probes-G) |
| gemma-4-e4b | llama.cpp | toggle, default off | Yes | No | Yes | No | [256](../research/2026-11-18/llmctl-sglang-e2e/records/256-13-gemma-4-e4b-llamacpp-probes-D), [257](../research/2026-11-18/llmctl-sglang-e2e/records/257-13-gemma-4-e4b-llamacpp-probes-G) |
| mellum2.1-12b | llama.cpp | toggle, default on | Yes | Yes | Yes | Yes | [275](../research/2026-11-18/llmctl-sglang-e2e/records/275-14-mellum2.1-12b-llamacpp-probes-D), [276](../research/2026-11-18/llmctl-sglang-e2e/records/276-14-mellum2.1-12b-llamacpp-probes-G) |
| qwen3.6-35b-a3b | llama.cpp | toggle, default off | Yes | No | Yes | No | [586](../research/2026-11-18/llmctl-sglang-e2e/records/586-29-qwen3.6-35b-a3b-llamacpp-probes-D), [587](../research/2026-11-18/llmctl-sglang-e2e/records/587-29-qwen3.6-35b-a3b-llamacpp-probes-G) |
| gpt-oss-20b | vLLM | effort | not sent | not sent | Yes | Yes | [294](../research/2026-11-18/llmctl-sglang-e2e/records/294-15-gpt-oss-20b-vllm-probes-G) |
| mellum2.1-12b | vLLM | toggle, default on | not sent | not sent | Yes | Yes | [312](../research/2026-11-18/llmctl-sglang-e2e/records/312-16-mellum2.1-12b-vllm-probes-G) |
| k2-horizon-7b | vLLM | always | not sent | not sent | Yes | No | [331](../research/2026-11-18/llmctl-sglang-e2e/records/331-17-k2-horizon-7b-vllm-probes-G) |
| qwen3.5-9b | vLLM | toggle, default off | not sent | not sent | Yes | Yes | [349](../research/2026-11-18/llmctl-sglang-e2e/records/349-18-qwen3.5-9b-vllm-probes-G) |
| gemma-4-12b | vLLM | toggle, default off | not sent | not sent | Yes | Yes | [367](../research/2026-11-18/llmctl-sglang-e2e/records/367-19-gemma-4-12b-vllm-probes-G) |
| gemma-4-e4b | vLLM | toggle, default off | not sent | not sent | Yes | Yes | [385](../research/2026-11-18/llmctl-sglang-e2e/records/385-20-gemma-4-e4b-vllm-probes-G) |
| groot-19b-a3b | vLLM | toggle, default off | not sent | not sent | Yes | Yes | [403](../research/2026-11-18/llmctl-sglang-e2e/records/403-21-groot-19b-a3b-vllm-probes-G) |
| deepseek-r1-0528-qwen3-8b | vLLM | always | not sent | not sent | No | Yes | [421](../research/2026-11-18/llmctl-sglang-e2e/records/421-22-deepseek-r1-0528-qwen3-8b-vllm-probes-G) |
| qwen3.5-9b | SGLang | toggle, default off | Yes | No | Yes | Yes | [444](../research/2026-11-18/llmctl-sglang-e2e/records/444-23-qwen3.5-9b-sglang-probes-D), [445](../research/2026-11-18/llmctl-sglang-e2e/records/445-23-qwen3.5-9b-sglang-probes-G) |
| k2-horizon-7b | SGLang | always | Yes | Yes | Yes | Yes | [469](../research/2026-11-18/llmctl-sglang-e2e/records/469-24-k2-horizon-7b-sglang-probes-D), [470](../research/2026-11-18/llmctl-sglang-e2e/records/470-24-k2-horizon-7b-sglang-probes-G) |
| gpt-oss-20b | SGLang | effort | Yes | Yes | Yes | Yes | [493](../research/2026-11-18/llmctl-sglang-e2e/records/493-25-gpt-oss-20b-sglang-probes-D), [494](../research/2026-11-18/llmctl-sglang-e2e/records/494-25-gpt-oss-20b-sglang-probes-G) |
| gemma-4-12b | SGLang | toggle, default off | Yes | No | Yes | Yes | [517](../research/2026-11-18/llmctl-sglang-e2e/records/517-26-gemma-4-12b-sglang-probes-D), [518](../research/2026-11-18/llmctl-sglang-e2e/records/518-26-gemma-4-12b-sglang-probes-G) |
| gemma-4-e4b | SGLang | toggle, default off | Yes | No | No | Yes | [541](../research/2026-11-18/llmctl-sglang-e2e/records/541-27-gemma-4-e4b-sglang-probes-D), [542](../research/2026-11-18/llmctl-sglang-e2e/records/542-27-gemma-4-e4b-sglang-probes-G) |
| qwen3.6-35b-a3b | SGLang | toggle, default off | Yes | No | Yes | Yes | [566](../research/2026-11-18/llmctl-sglang-e2e/records/566-28-qwen3.6-35b-a3b-sglang-probes-D), [567](../research/2026-11-18/llmctl-sglang-e2e/records/567-28-qwen3.6-35b-a3b-sglang-probes-G) |

On SGLang, probe D is the control that the vLLM rows lack: the four toggle recipes, whose server default is off, returned no reasoning without the field and reasoning with `reasoning_effort: "low"`, which matches SGLang turning `enable_thinking` on for any effort but `none` ([sglang/README.md](../sglang/README.md#tool-calling-and-reasoning)).
Two probe G replies had no content: vLLM DeepSeek-R1 spent all 256 tokens on reasoning (`finish_reason` `length`), and SGLang Gemma 4 E4B put its whole answer, `OK.`, into `reasoning_content`.
K2 Horizon 7B on llama.cpp returned no reasoning to probe G this time, against reasoning on 2026-11-17 and to probe D in the same run.

The Kimi run below, the day before, covered llama.cpp and vLLM before SGLang support existed.

On 2026-11-17, from 14:40 to 15:07, the driver ran every shipped recipe again with Kimi Code CLI as the fifth client: `--matrix all --set-default-round-trip` over all 19 recipes with OpenCode, Hermes, Pi, dsh and Kimi, 333 rows (measured on the MSI Raider 18 Max HX, Kimi Code 2.5.3, llama.cpp b12407 and vLLM v0.33.0, 2026-11-17); [README](../research/2026-11-17/llmctl-kimi-e2e/README.md), [summary](../research/2026-11-17/llmctl-kimi-e2e/summary.md), [records](../research/2026-11-17/llmctl-kimi-e2e/records/).
All 39 Kimi rows read Yes: 19 `kimi-config` rows, 19 `kimi -p` sessions, and the round trip's session without `-m`, which used the `default_model` that `./llmctl up qwen3.6-35b-a3b --set-default` wrote ([record 365](../research/2026-11-17/llmctl-kimi-e2e/records/365-round-trip-up), [record 372](../research/2026-11-17/llmctl-kimi-e2e/records/372-round-trip-kimi)).
The restore put Kimi's `default_model` back to the owner's earlier state, unset, with every other client's default, and read it back (the `restore` row).
Both DeepSeek-R1 recipes answered through the agent file `launcher/kimi-chat-only.md` ([record 100](../research/2026-11-17/llmctl-kimi-e2e/records/100-05-deepseek-r1-0528-qwen3-8b-llamacpp-kimi), [record 341](../research/2026-11-17/llmctl-kimi-e2e/records/341-18-deepseek-r1-0528-qwen3-8b-vllm-kimi)).
K2 Horizon 7B on vLLM passed in Kimi, which wrote and read `hello.py` ([record 250](../research/2026-11-17/llmctl-kimi-e2e/records/250-13-k2-horizon-7b-vllm-kimi)), while Hermes and Pi failed on its template with HTTP 400 `Assistant message is missing a thinking field` in `hermes -z`, `hermes chat` and Pi ([record 244](../research/2026-11-17/llmctl-kimi-e2e/records/244-13-k2-horizon-7b-vllm-hermes-z), [record 245](../research/2026-11-17/llmctl-kimi-e2e/records/245-13-k2-horizon-7b-vllm-hermes-chat), [record 246](../research/2026-11-17/llmctl-kimi-e2e/records/246-13-k2-horizon-7b-vllm-pi)).
The run's 11 No rows belong to the other clients and the probes, and the README classifies none as a defect in llmctl or the driver ([The No rows](../research/2026-11-17/llmctl-kimi-e2e/README.md#the-no-rows)): those three K2 Horizon 7B rows, vLLM DeepSeek-R1's probe A with its garbled title, and seven sessions that failed on model behaviour, Nemotron 3.5 Bolt's parallel write and read in Pi, the refused `justification` argument of gemma-4-26b-a4b and vLLM gemma-4-e4b in dsh, and four Hermes sessions whose `hello.py` landed outside the working folder or whose write Hermes refused.
The run used Claude Code's ripgrep 14.1.1 through a symlink named `rg` on `PATH`, because ripgrep from apt was installed after the run, so any Kimi Grep call in it ran that build.

Probe G of that run, the plain OK request with `max_tokens` 256 and `reasoning_effort: "low"`, recorded without judging, got these replies (measured on the MSI Raider 18 Max HX, llama.cpp b12407 and vLLM v0.33.0, 2026-11-17); the thinking type is the recipe's in [Kimi thinking per type](#kimi-thinking-per-type):

| Recipe | Runtime | Thinking type | Status | Content | Reasoning | Record |
|---|---|---|---|---|---|---|
| qwen3.8-27b | llama.cpp | effort | 200 | Yes | Yes | [025](../research/2026-11-17/llmctl-kimi-e2e/records/025-01-qwen3.8-27b-llamacpp-probes-G) |
| k2-horizon-7b | llama.cpp | always | 200 | Yes | Yes | [046](../research/2026-11-17/llmctl-kimi-e2e/records/046-02-k2-horizon-7b-llamacpp-probes-G) |
| nemotron-3.5-bolt | llama.cpp | toggle, default off | 200 | Yes | No | [066](../research/2026-11-17/llmctl-kimi-e2e/records/066-03-nemotron-3.5-bolt-llamacpp-probes-G) |
| qwen3-coder-next | llama.cpp | none | 200 | Yes | No | [085](../research/2026-11-17/llmctl-kimi-e2e/records/085-04-qwen3-coder-next-llamacpp-probes-G) |
| deepseek-r1-0528-qwen3-8b | llama.cpp | always | 200 | Yes | Yes | [104](../research/2026-11-17/llmctl-kimi-e2e/records/104-05-deepseek-r1-0528-qwen3-8b-llamacpp-probes-G) |
| gemma-4-26b-a4b | llama.cpp | toggle, default off | 200 | Yes | No | [123](../research/2026-11-17/llmctl-kimi-e2e/records/123-06-gemma-4-26b-a4b-llamacpp-probes-G) |
| nemotron-3.5-lightning | llama.cpp | toggle, default off | 200 | Yes | No | [142](../research/2026-11-17/llmctl-kimi-e2e/records/142-07-nemotron-3.5-lightning-llamacpp-probes-G) |
| groot-19b-a3b | llama.cpp | toggle, default off | 200 | Yes | No | [161](../research/2026-11-17/llmctl-kimi-e2e/records/161-08-groot-19b-a3b-llamacpp-probes-G) |
| k2-horizon-mova-36b | llama.cpp | always | 200 | Yes | Yes | [181](../research/2026-11-17/llmctl-kimi-e2e/records/181-09-k2-horizon-mova-36b-llamacpp-probes-G) |
| gpt-oss-20b | llama.cpp | effort | 200 | Yes | Yes | [200](../research/2026-11-17/llmctl-kimi-e2e/records/200-10-gpt-oss-20b-llamacpp-probes-G) |
| qwen3.6-35b-a3b | llama.cpp | toggle, default off | 200 | Yes | No | [364](../research/2026-11-17/llmctl-kimi-e2e/records/364-19-qwen3.6-35b-a3b-llamacpp-probes-G) |
| gpt-oss-20b | vLLM | effort | 200 | Yes | Yes | [218](../research/2026-11-17/llmctl-kimi-e2e/records/218-11-gpt-oss-20b-vllm-probes-G) |
| mellum2.1-12b | vLLM | toggle, default on | 200 | Yes | Yes | [236](../research/2026-11-17/llmctl-kimi-e2e/records/236-12-mellum2.1-12b-vllm-probes-G) |
| k2-horizon-7b | vLLM | always | 200 | Yes | No | [255](../research/2026-11-17/llmctl-kimi-e2e/records/255-13-k2-horizon-7b-vllm-probes-G) |
| qwen3.5-9b | vLLM | toggle, default off | 200 | Yes | Yes | [273](../research/2026-11-17/llmctl-kimi-e2e/records/273-14-qwen3.5-9b-vllm-probes-G) |
| gemma-4-12b | vLLM | toggle, default off | 200 | Yes | Yes | [291](../research/2026-11-17/llmctl-kimi-e2e/records/291-15-gemma-4-12b-vllm-probes-G) |
| gemma-4-e4b | vLLM | toggle, default off | 200 | Yes | No | [309](../research/2026-11-17/llmctl-kimi-e2e/records/309-16-gemma-4-e4b-vllm-probes-G) |
| groot-19b-a3b | vLLM | toggle, default off | 200 | Yes | Yes | [327](../research/2026-11-17/llmctl-kimi-e2e/records/327-17-groot-19b-a3b-vllm-probes-G) |
| deepseek-r1-0528-qwen3-8b | vLLM | always | 200 | Yes | Yes | [344](../research/2026-11-17/llmctl-kimi-e2e/records/344-18-deepseek-r1-0528-qwen3-8b-vllm-probes-G) |

On llama.cpp, probe D sent the same plain request without `reasoning_effort`, with the recipe id as its model, to every recipe, and the toggle recipes whose server default is off returned no reasoning there either.
On vLLM, probe D is not sent, so the run holds no control of the plain request without the field for Qwen3.5-9B, Gemma 4 12B and Groot-19B-A3B, whose server default is off but which returned reasoning to probe G; their only requests without the field were probe B's tool replays, a different request, which returned no reasoning.
K2 Horizon 7B on vLLM, which always thinks, returned no reasoning to probe G, as to probes A and C with `reasoning_effort: "none"`, while its probe B returned reasoning.

The dsh run below, earlier that day, covered four clients before Kimi support existed.

On 2026-11-17 the driver ran every shipped recipe, 11 on llama.cpp and 8 on vLLM, with `heavy python3 -m launcher.e2e.clients --out research/2026-11-17/llmctl-dsh-e2e --matrix all --set-default-round-trip`, after six of the seven vLLM estimates had been measured and the seventh, DeepSeek-R1-0528-Qwen3-8B, had failed, and after the owner's hand-made dsh setup moved to llmctl's block (measured on the MSI Raider 18 Max HX, llama.cpp b12407, vLLM v0.33.0, dsh 0.2.2-rc.3, 2026-11-17); [README](../research/2026-11-17/llmctl-dsh-e2e/README.md), [summary](../research/2026-11-17/llmctl-dsh-e2e/summary.md), [records](../research/2026-11-17/llmctl-dsh-e2e/records/).
Every recipe ran on Hermes, Pi and dsh, and every recipe except vLLM DeepSeek-R1-0528-Qwen3-8B, which had no OpenCode entry then, on OpenCode too.
That was the second full run of the day, and the one whose records are in the repository.
The first run exited 1 with 280 Yes and 14 No rows; twelve were client sessions, and two were probe rows holding three failed probes: K2 Horizon 7B's vLLM probe A got HTTP 400 `Unsupported reasoning_effort 'none'`, vLLM deepseek-r1-0528-qwen3-8b's probe A got a garbled title from its known tokenizer defect, and its probe B got HTTP 400 because the driver sent tools to a chat-only recipe ([First run](../research/2026-11-17/llmctl-dsh-e2e/README.md#first-run); that run's [summary](../research/2026-11-17/llmctl-dsh-e2e-run1/summary.md) and records [214](../research/2026-11-17/llmctl-dsh-e2e-run1/records/214-13-k2-horizon-7b-vllm-probes-A), [290](../research/2026-11-17/llmctl-dsh-e2e-run1/records/290-18-deepseek-r1-0528-qwen3-8b-vllm-probes-A) and [291](../research/2026-11-17/llmctl-dsh-e2e-run1/records/291-18-deepseek-r1-0528-qwen3-8b-vllm-probes-B)).
Two fixes followed: K2 Horizon 7B's vLLM recipe gained the `--reasoning-effort-map` of the gpt-oss-20b recipe ([Thinking per type](#thinking-per-type)), and the driver no longer sends probe B to a chat-only recipe.

The second run took about 22 minutes and exited 1 with 284 of 294 rows Yes.
Every `up`, `status`, config, `hermes-config`, `dsh-config`, `dsh-web`, `gateway` and `restore` row read Yes, as did every probes row except vLLM deepseek-r1-0528-qwen3-8b's, whose probe A again got a title that does not parse because the tokenizer garbles every reply ([record 289](../research/2026-11-17/llmctl-dsh-e2e/records/289-18-deepseek-r1-0528-qwen3-8b-vllm-probes-A)); K2 Horizon 7B's vLLM probe A now passed ([record 213](../research/2026-11-17/llmctl-dsh-e2e/records/213-13-k2-horizon-7b-vllm-probes-A)), and no record held a `TIMEOUT` or a queued step.
Seven No rows are model behaviour: Nemotron 3.5 Bolt's parallel write and read in Pi ([record 049](../research/2026-11-17/llmctl-dsh-e2e/records/049-03-nemotron-3.5-bolt-llamacpp-pi)); `SANDBOX_ESCALATION_INVALID_ARGUMENT` write retries of gemma-4-26b-a4b and, with an `FS_NOT_FOUND` read, of vLLM gemma-4-e4b in dsh ([record 099](../research/2026-11-17/llmctl-dsh-e2e/records/099-06-gemma-4-26b-a4b-llamacpp-dsh-headless), [record 256](../research/2026-11-17/llmctl-dsh-e2e/records/256-16-gemma-4-e4b-vllm-dsh-headless)); and Hermes sessions whose `hello.py` landed outside the working folder, Nemotron 3.5 Lightning's `hermes -z` ([record 112](../research/2026-11-17/llmctl-dsh-e2e/records/112-07-nemotron-3.5-lightning-llamacpp-hermes-z)), Qwen3.6-35B-A3B's `hermes -z` into the home folder and its `hermes chat`, whose write to that same `~/hello.py` Hermes refused because the session had not read it ([record 297](../research/2026-11-17/llmctl-dsh-e2e/records/297-19-qwen3.6-35b-a3b-llamacpp-hermes-z), [record 298](../research/2026-11-17/llmctl-dsh-e2e/records/298-19-qwen3.6-35b-a3b-llamacpp-hermes-chat)), and the round trip's `hermes -z` into the profile's `cache/scratch` folder ([record 310](../research/2026-11-17/llmctl-dsh-e2e/records/310-round-trip-hermes-z)).
The other two are K2 Horizon 7B on vLLM in `hermes -z` and Pi, which got HTTP 400 `Assistant message is missing a thinking field` ([record 207](../research/2026-11-17/llmctl-dsh-e2e/records/207-13-k2-horizon-7b-vllm-hermes-z), [record 209](../research/2026-11-17/llmctl-dsh-e2e/records/209-13-k2-horizon-7b-vllm-pi)), the template limitation of [Thinking per type](#thinking-per-type).
The round trip's `./llmctl up qwen3.6-35b-a3b --set-default` made the model the default of all four clients with nothing started ([record 307](../research/2026-11-17/llmctl-dsh-e2e/records/307-round-trip-up)); then `opencode run`, Pi and a dsh session in the real dsh home reached the server without `--model`, `-p` or `--patch` ([record 308](../research/2026-11-17/llmctl-dsh-e2e/records/308-round-trip-opencode), [record 311](../research/2026-11-17/llmctl-dsh-e2e/records/311-round-trip-pi), [record 312](../research/2026-11-17/llmctl-dsh-e2e/records/312-round-trip-dsh-headless)), and the restore put every default back and read it back (the `restore` row).
The stray `hello.py` files the sessions wrote were removed by hand afterwards.

The browser checklist of the dsh design ran the same day with `dsh web --port 3182 --no-open` in a private home that held a byte copy of the real home patch, against gemma-4-26b-a4b on llama.cpp and then gpt-oss-20b on vLLM (measured on the MSI Raider 18 Max HX, dsh 0.2.2-rc.3, 2026-11-17; [record](../research/2026-11-17/llmctl-dsh-web/README.md)).
All five items passed: the picker listed `gemma-4-26b-a4b on llama.cpp (llmctl)` with Effort Off (preselected) and Medium; a new session answered `pong`; the Models page refused a new OpenAI-compatible provider with `Configuration for "llm-pi-ai" is overridden by a home patch or command-line overlay`; an Effort change lived only in the open session; and after `./llmctl up gpt-oss-20b --runtime vllm --replace` and an atomic copy of the new home patch, the next new session's picker showed `gpt-oss-20b on vLLM (llmctl)` 2.4 s later.
Although `DSH_HOME`, `DSH_AGENTS_HOME` and `PWD` pointed into the private home, `dsh web` created an empty `~/Documents/deepseek-harness/default-workspace` when the browser first opened the page; neither folder existed before, and both were removed afterwards.

The 2026-11-16 run below covered OpenCode, Hermes and Pi before dsh support existed.

On 2026-11-16 the e2e driver ran the default matrix and the `--set-default` round trip with Hermes Agent v0.23.1, Pi 1.5.2 and OpenCode v2.3.0 against the images the recipes pin, llama.cpp `server-cuda-b12407` and vLLM `v0.33.0`, alone and one server at a time, in about 4 minutes (measured on the MSI Raider 18 Max HX, llama.cpp b12407 and vLLM v0.33.0, 2026-11-16; [summary](../research/2026-11-16/llmctl-clients-e2e/summary.md), [records](../research/2026-11-16/llmctl-clients-e2e/records/)).
This was the fourth run that day and the one whose records are in the repository; the runs before it are summarized at the end of this section.
The command was `heavy python3 -m launcher.e2e.clients --out research/2026-11-16/llmctl-clients-e2e --set-default-round-trip`; records 001 to 003 hold the version lines, and `nvidia-smi` reported the power limit as `[N/A]` (record 004).
A record leaves out the token events of `hermes chat` (`text`) and Pi (`message_update`) and counts them under `dropped_events`, because they split a path into pieces that redaction cannot see, and when the redacted output still names the home path, the user name, the host name or the work root, even split over several events, the driver withholds both streams and fails the step.
The driver gained both rules after run 3, whose records leaked the home path and the work root through those token events and were removed before any commit; run 4 was made with the fixed driver, and none of its records was withheld.
The matrix was Qwen3.8-27B (effort), K2 Horizon 7B (always) and Qwen3.6-35B-A3B (toggle, off) on llama.cpp, and gpt-oss-20b (effort, served as `gpt-oss-20b`) on vLLM.
For each recipe the driver ran `up --replace` and `status`, compared the Hermes block and Pi's entry and level with the generator, read `model`, `providers`, `model_overrides` and `gateway` back with `hermes -p llmctl config get`, and gave the `hello.py` write-and-read task to `opencode run`, `hermes -z`, `hermes chat -q --format stream-json` and `pi --mode json`, each in a fresh work folder.
Then it sent probes to the server: A, Hermes' title request with `reasoning_effort: "none"` and a strict `json_schema`; B, a replay whose assistant turn carries both `reasoning` and `reasoning_content` and a tool call; C, `reasoning_effort: "none"` to K2 Horizon 7B and Qwen3.6; D, the recipe id instead of `local` to llama.cpp; and E, a request with its own `chat_template_kwargs` to K2 Horizon 7B.

50 of the 55 steps passed, so the driver exited 1.
Every `up`, `status`, config check, `config get` and probe step passed for all four recipes, as did the `gateway` row after the first `up`, the round trip's `up`, OpenCode and Pi sessions and restore, and `down`.
All five failures are client sessions, and none of them failed to reach the server.
Probes A and B got HTTP 200 from every server, A with a title that parses; C, D and E, which the driver records without judging, got HTTP 200 too, K2 Horizon 7B reasoned despite `reasoning_effort: "none"`, and llama.cpp answered D from the loaded model.
Four failures are Hermes sessions that completed but left no `hello.py` in the working folder, as [Where files land](#where-files-land) describes:

- K2 Horizon 7B, `hermes -z` (record 025): the model wrote `hello.py` by absolute path into the profile folder `~/.hermes/profiles/llmctl/`.
- Qwen3.6-35B-A3B, `hermes -z` (record 052): the model wrote it into the home folder.
- Qwen3.6-35B-A3B, `hermes chat` (record 053): the model passed the profile folder path to `write_file`, and Hermes refused the write because a `hello.py` from K2 Horizon 7B's session (record 025) was already there and this session had not read it; the model read that file, answered from it without writing again, and Hermes appended a warning that a file edit had failed.
- The round trip's `hermes -z` (record 062), again Qwen3.6-35B-A3B: the file did not land in the working folder, and the record does not name the path the model used.

The fifth failure is Qwen3.8-27B in Pi (record 014): the model wrote `hello.py` into the working folder, then tried a second write to `/app/hello.py`, a path it made up, which failed with `EACCES`, then wrote and read `hello.py` again.
That session ended with a valid `hello.py` in the working folder, but the driver fails a session on any tool error.
The stray `hello.py` files in the home folder and the profile folder were removed by hand after the run.

The round trip's `./llmctl up qwen3.6-35b-a3b --set-default` set Pi's `defaultProvider` and `defaultModel`, ran `hermes profile use llmctl` (`Hermes default profile: default -> llmctl`), and left OpenCode's default, which already named the model (record 059).
Then `opencode run` without `--model`, `hermes -z` without `-p` and `pi` without `--model` all reached the running server: the llama.cpp log shows the OpenCode requests (record 061), Hermes' usage file named provider `custom` and model `local` (the driver's check, which passed although the session failed on `hello.py`), and Pi's events name `llmctl/local` (record 063).
The restore put Pi back to no default, Hermes back to the profile `default` and OpenCode on the model it had before the round trip, and read each one back (the `restore` row).
The live-gateway check passed: after the first `up`, Hermes' `gateway_state.json` showed no running gateway that serves `llmctl`, and no `status` printed the gateway warning; the profile existed from an earlier run, so the check ran without its 35 s wait.
Checked by hand the same day, the multiplexing gateway on this laptop kept serving only `default` across its 30 s rescans after the profile was first created (not recorded).

The notes below describe the three earlier runs the same day; they come from notes taken during the runs, and none of their records is in the repository.
Run 1 was stopped after its first recipe: the driver started `opencode run` in a work folder but left `PWD` naming the repository, OpenCode v2.3.0 took its working folder from `PWD`, and so it ignored the work folder's `opencode.json` with its permission denies, and the model wrote `hello.py` into the repository.
The driver now sets `PWD` for each step; a script that runs `opencode run` in another folder should do the same, or `cd` there in a shell ([opencode.md](opencode.md#config-shape-in-v230)).
Run 2 passed 54 of 57 steps: Qwen3.6's `hermes -z` wrote `hello.py` outside the working folder, gpt-oss-20b on vLLM once answered in OpenCode with the file content as text and no tool call (it used the tools there in runs 3 and 4, record 039 for run 4, and in Hermes and Pi in runs 2 to 4), and the round trip's OpenCode check failed because the driver read only the stdout of `docker logs` while llama.cpp logs on stderr, although the session had reached the server; the driver now reads both.
Run 3 passed 52 of 55 steps; its three failures were Qwen3.6-35B-A3B's three Hermes sessions, which wrote `hello.py` outside the working folder.
Across runs 2 to 4 every setup step passed, and the session failures changed from run to run but were always about where the model put the file or a tool error of the model: Qwen3.6 in Hermes wrote outside the working folder in 1 of 3 sessions in run 2 and in 3 of 3 in runs 3 and 4, K2 Horizon 7B once (run 4), and Qwen3.8-27B and gpt-oss-20b never in Hermes.

## Maintenance

- Re-check the pinned versions before relying on this document: Hermes Agent v0.23.1, Pi 1.5.2, DeepSeek Harness 0.2.2-rc.3 and Kimi Code CLI 2.5.3 (see [Versions](#versions)); on an upgrade, rerun the e2e below before trusting the entries.
- Watch Hermes' `gateway.standalone` shim in each Hermes release; if it goes, the profile llmctl may be served by the messaging gateway, which the gateway warning reports.
- Hermes' `_config_version` default (53 at v0.23.1) and its owned keys: a newer Hermes migrates the stamp itself, but a new key that must live in the block needs a change in `launcher/hermes_config.py`.
- Pi's `ModelDefinitionSchema` keys (`dist/core/model-config.js:32-49`): Pi 1.5.2 accepts keys its schemas do not list (`dist/core/model-config.js:32-89` sets no `additionalProperties: false`), so a key llmctl writes that a later Pi drops or renames may still pass the check during `up` while the setting stops working; compare the keys on every Pi upgrade and rerun the e2e below.
- DeepSeek Harness 0.2.2-rc.3 is `VERIFIED_VERSION` in `launcher/dsh_config.py`; on any other version `up` warns, and on an upgrade update the constant, regenerate the golden files with `python3 -m launcher.tests.regen_golden`, and rerun the dsh steps of the e2e before relying on the rows.
- The three row ids `llm-pi-ai`, `agent-default-model` and `acp`: the dsh check refuses when the installed dsh lacks one, which is the first sign of a rename.
- The dump format the dsh check reads: `# == ..., patched by ...` groups that list the patch files, and `- id: <id>` lines under them; if a later dsh prints another format, the check refuses with `dsh did not apply llmctl's ... row` and `launcher/dsh_config.py` needs a change.
- pi-ai's resolved version (`@earendil-works/pi-ai` 1.5.2 from `^1.3.0`, with no shrinkwrap): a reinstall of the same dsh can resolve a newer pi-ai, so check `npm ls -g @earendil-works/pi-ai` after each install.
- The Bolt budget rule: Nemotron 3.5 Bolt's medium level depends on pi-ai leaving out a budget of 0, so record B1-2 again on every pi-ai upgrade ([Rows per thinking type](#rows-per-thinking-type)).
- Kimi Code CLI 2.5.3 is `VERIFIED_VERSION` in `launcher/kimi_config.py`; on any other version `up` warns after the clients line and goes on, and on an upgrade update the constant, regenerate the golden files with `python3 -m launcher.tests.regen_golden`, and rerun the Kimi steps of the e2e before relying on the entry.
- The output the Kimi check reads: `kimi provider list --json` prints llmctl's keys in camelCase (`baseUrl`, `apiKey`, `modelSource`, `maxContextSize`, `maxOutputSize`, `displayName`, `supportEfforts`, `defaultEffort`, `offEffort`) ([record](../research/2026-11-17/llmctl-kimi-stub/exp/records/provider-list-llamacpp-qwen3.8-27b.json)); if a later Kimi prints other names, the check refuses with `Kimi ... does not read llmctl's entry` and `launcher/kimi_config.py` needs a change.
- Kimi's writer, which [The Kimi tables](#the-kimi-tables) relies on: on an upgrade, rerun E4's writer runs (`exp/bin/e4.sh`, `e4_web.sh` and `e4_registry.sh` in [the archive](../research/2026-11-17/llmctl-kimi-stub/README.md#rerun)) and check that llmctl still finds and keeps its tables in the files Kimi writes, because a writer that moves, splits or reformats tables in a new way is what the line-by-line edit must survive.
- P1 to P3 on an upgrade: whether the TUI still ignores `--agent-file` and print mode still drops `tools` with the shipped agent file (P1), whether print mode still sends a global `[thinking] effort` verbatim (P2), and what `/effort` saves (P3); the warnings and the chat-only line of `up` rest on them.
- The e2e run proves real sessions through OpenCode, Hermes, Pi, dsh and Kimi on all three runtimes; run it only alone, through heavy, after the unit suite passes: `PATH=$HOME/.kimi-code/bin:$PATH heavy python3 -m launcher.e2e.clients --out research/<date>/llmctl-clients-e2e [--matrix default|extended|all] [--keep-running] [--set-default-round-trip] [--skip-dsh] [--skip-kimi]`.
  `--matrix all` runs every shipped recipe.
  Every SGLang recipe gets a long-grammar step right after its `status` step: a calibration request of 20 copies of a fixed code paragraph, a request of as many copies as fill 90% of the context, Hermes' title request and, for a recipe with tools, a forced tool call, then a `GET /ready`; a step whose server stopped answering skips the recipe's remaining steps, because SGLang's [grammar path](../sglang/README.md#grammar-requests-and-nccl) is most at risk right after a long prompt.
  `--skip-dsh` leaves out the dsh steps and `--skip-kimi` the Kimi steps, and each sets `LLMCTL_CLIENTS` for every command to the clients that stay: `opencode,hermes,pi,kimi`, `opencode,hermes,pi,dsh`, or `opencode,hermes,pi` with both.
  Without `--skip-kimi` the driver puts `$HOME/.kimi-code/bin` first on `PATH` for itself and every command, so `./llmctl` and every step find the native `kimi` before any other.
  Before it starts anything, it refuses with exit 1 when `LLMCTL_CLIENTS` is set, when an llmctl model server is running (`error: an llmctl model server is running (<recipe> on <runtime>); stop it with ./llmctl down, then run the driver again`), or when the `--out` folder exists.
  Without `--skip-kimi` it also refuses when Kimi is not installed with that `PATH`, when any `KIMI_MODEL_` variable is set (they change which model Kimi uses and what it sends), when no `rg` executable is on `PATH`, when neither `fd` nor `fdfind` is (Kimi's Grep and Glob tools would download them from Kimi's CDN; a shell function or alias does not count), when `config.toml` cannot be read or is not valid TOML, or when its `default_model` is in a form that llmctl's default edit cannot change, such as an integer, a one-line multi-line string or a `[default_model]` table, because every Kimi session of the matrix gets that edit on its private copy; it reads only `config.toml` in the Kimi home and runs no Kimi command for these checks.
  Without `--skip-dsh` it also refuses when dsh is not installed, when a profile patch sets the `llm-pi-ai` row, when `DEEPSEEK_API_KEY` is set (dsh's `web_search` could send model-written queries to DeepSeek), when 127.0.0.1:3182 is in use, or when the home patch is unreadable or its markers are damaged; it never opens a `.env` or credential file.
  Per recipe, after the Pi session, it checks the home patch against what llmctl writes from the text it held at the start, runs one `dsh --profile headless` session and boots `dsh web --port 3182 --no-open`, each with telemetry off, `TMPDIR` under the work root and a private dsh home there that holds a byte copy of the real home patch; the web step records `dsh web: <address withheld>` instead of its launch address.
  The web step waits up to 30 s for the `dsh web:` line and, once it came, reads stderr 3 s more, then stops the process group with SIGTERM, and with SIGKILL when it is still there 15 s later (`WEB_READY_S`, `WEB_TAIL_S` and `WEB_KILL_S` in `launcher/e2e/clients.py`).
  Without the line after 30 s it sends the group SIGTERM at once, and SIGKILL 15 s later even when the process ignores SIGTERM, so the step always ends.
  `dsh --version` runs with telemetry off, `TMPDIR` under the work root and its working folder a fresh `<work root>/dsh-version`.
  Per recipe, after the dsh steps (after the Pi session with `--skip-dsh`), a `kimi-config` step checks in process that the real `config.toml` holds llmctl's two tables for the recipe by value, the `default_model` that `up` without `--set-default` gives from the one at the start, and the same lines outside llmctl's tables as at the start; on a mismatch its record holds only llmctl's tables and `default_model` as found and as wanted, with any other `api_key` written `<withheld>`.
  Then a `kimi` step runs `kimi -p <task> --output-format stream-json -m llmctl/<served name>` (a chat-only recipe gets the chat prompt and `--agent-file launcher/kimi-chat-only.md`) in its own work folder, with a private Kimi home `<work root>/<step>.kimi` (mode 0700) holding a copy of the real `config.toml` (mode 0600), with llmctl's own default edit when its `default_model` does not name the recipe, because print mode needs a default.
  The session also gets an empty `HOME` and a `TMPDIR` under the work root, one Kimi cache folder per run (`<work root>/kimi-cache`), `KIMI_DISABLE_TELEMETRY=1`, `KIMI_CODE_NO_AUTO_UPDATE=1` and `KIMI_CODE_REGION_MARKER=off`, so no Kimi session runs in your Kimi home and your `~/.agents` skills stay out.
  It passes on exit 0, JSON lines only, no retried request, with tools a `Write` and a `Read` call each answered by a tool line, no `Bash` call and a valid `hello.py`, without tools no tool call, a last assistant message with text, the private `config.toml` unchanged by the session, and a chat completion in the server log since the session started.
  `kimi --version` runs the same way, in a fresh `<work root>/kimi-version` with its own private Kimi home.
  The probes are those of [Verified on this laptop](#verified-on-this-laptop) plus probe G, the plain OK request with `reasoning_effort: "low"` for every recipe, which the run records without judging; a chat-only recipe gets no probe B, because none of the e2e's sessions sends it tools: OpenCode runs only for a recipe with an OpenCode entry, which a chat-only recipe has only on llama.cpp, with `"tools": false` ([opencode.md](opencode.md#the-validated-config)), Hermes' block has `platform_toolsets: {"cli": []}`, Pi runs with `--no-tools`, and the dsh session adds the no-tools `--patch` of [Chat-only recipes](#chat-only-recipes).
  A probe with tools would test a request no session sends, and vLLM refuses tools without `--enable-auto-tool-choice` (HTTP 400 for vLLM deepseek-r1-0528-qwen3-8b in the first run of 2026-11-17, that run's [record 291](../research/2026-11-17/llmctl-dsh-e2e-run1/records/291-18-deepseek-r1-0528-qwen3-8b-vllm-probes-B)).
  With `--set-default-round-trip` it also refuses when OpenCode's config has no top-level `"model"` to restore, when Kimi's `default_model` exists but is not a string, when `HERMES_HOME` names a profile in that shell, or when Pi's `enabledModels` would start a session without `--model` on another model, because the round trip runs `hermes` without `-p` and `pi` without `--model`.
  `--out` holds only the records: one file per command or probe under `records/`, and the `summary.md` table.
  The client sessions run in fresh work folders under one private folder per run, made with `mkdtemp` (prefix `llmctl-e2e-`, mode 0700) outside the repository and left in place; the first line names both, `records in <out>; work folders in <work root>, left in place after the run`, and every record shows that folder as `$WORK` and your home as `$HOME`.
  The live-gateway check runs 35 s after the first passing `up` that created the profile llmctl, or right after the first passing `up` when the profile existed before the run, and adds a `gateway` row; when no `up` created the profile, `summary.md` gets `- | gateway | No | no up created the profile llmctl (<path>), so the gateway check did not run`, and a `status` step that prints the gateway warning fails as well.
  The round trip runs `./llmctl up qwen3.6-35b-a3b --set-default`, then OpenCode, `hermes -z`, Pi, a dsh session in the real dsh home and a Kimi session in a private Kimi home whose `config.toml` stays a pure byte copy of the real one, without `--model`, `-p`, `--patch` or `-m`, and always restores the defaults it snapshotted, Kimi's `default_model` included, which it removes again when you had none; `summary.md` gets a `round-trip | check | No | <Type>: <error>` row when the check itself failed, and a `round-trip | restore` row with each value the restore could not put back.
  It writes your real client files and moves defaults that point at llmctl, ending on qwen3.6-35b-a3b; its run of 2026-11-16 is in [Verified on this laptop](#verified-on-this-laptop), and the request shapes above keep their stub labels, because the e2e records the client sessions and the probe replies, not the requests the clients send.
