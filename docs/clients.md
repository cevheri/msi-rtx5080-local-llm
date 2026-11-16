# Hermes Agent and Pi with the local servers

Last verified: 2026-11-16.

`./llmctl up <model>` writes one llmctl entry into Hermes Agent and one into Pi, so both reach the model llmctl runs with its context, output cap and default thinking, without a manual config edit.
This document shows those entries, how each client thinks with them, what llmctl checks, and what was measured; which files llmctl writes and when, `LLMCTL_CLIENTS`, `--set-default` and the refusals are in [llmctl.md](llmctl.md#clients), and OpenCode is in [opencode.md](opencode.md).

## Versions

- [Hermes Agent v0.23.1](https://github.com/NousResearch/hermes-agent/tree/v0.23.1): the tag `v0.23.1` is commit `3b421578588243f920886e6fe79a6faa87ea026e`, and every Hermes source file cited here is [at that commit](https://github.com/NousResearch/hermes-agent/tree/3b421578588243f920886e6fe79a6faa87ea026e).
- [Pi 1.5.2 on npm](https://www.npmjs.com/package/@earendil-works/pi-coding-agent/v/1.5.2) (`@earendil-works/pi-coding-agent`; that page answers scripts with HTTP 403, so a script checks the [registry record](https://registry.npmjs.org/@earendil-works/pi-coding-agent/1.5.2) instead) and [its source at v1.5.2](https://github.com/earendil-works/pi/tree/v1.5.2/packages/coding-agent), tag `v1.5.2` = commit `b4afdddf44e059c67ccf5097e5317d913521b470`; the `dist/` and `docs/` files cited here are in the 1.5.2 npm package as published, and the `packages/ai/src/` file is at that tag.
- OpenCode v2.3.0, as in [opencode.md](opencode.md).

These are the versions the entries were verified with on 2026-11-16; any other version is unverified.
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

Every llama.cpp recipe is served as `local`, so its block names the model `local`; a vLLM recipe uses its served name.
`context_length` equals the recipe's context, and `extra_body.max_tokens` its `[clients] output`, so every request carries the output cap.
Hermes refuses a context below 64,000 tokens (R19), so llmctl refuses such a recipe for Hermes; every shipped recipe has 65,536 or more.
The golden files for all 19 recipes are under `launcher/tests/golden/clients/`, and the tests keep them equal to what llmctl writes.

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
With Qwen3.6, and less often K2 Horizon 7B, in Hermes, check where files land, or name the target folder in the prompt (not tested).

## Pi

llmctl owns the provider `llmctl` in Pi's `models.json` and the `llmctl/` keys of `modelThinkingLevels` in `settings.json`, both in Pi's agent folder: `PI_CODING_AGENT_DIR` when set, else `~/.pi/agent` (`dist/config.js:510-516`).
Use it with `pi --model llmctl/<served name>`, which `up` prints (`llmctl/local` for every llama.cpp recipe), or make it Pi's default with `./llmctl up <model> --set-default`.

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

## Silent fallbacks and exit codes

These are the ways a client can answer from the wrong model without an error, and what guards each one:

- Pi with a `defaultModel` that `models.json` no longer has uses another provider's model with exit 0 (S3); llmctl moves a default that points at llmctl on every `up`, and never removes its entry on `down`.
- Pi with a `models.json` that fails its schema drops the whole file (`dist/core/model-config.js:144-150`, E7 `neg-format`); llmctl lets Pi check the candidate file first and refuses when the llmctl row is missing.
- Pi with a comment in `settings.json` ignores the whole file (S4); llmctl edits only plain JSON and refuses the rest.
- Hermes with a duplicate key in a profile serves it on default settings inside your gateway (E3); llmctl refuses an owned key outside its block, and checks its own result before writing.
- A Hermes or Pi session that was open before an `up` may keep the entry it started with ([Hermes](#sessions-and-interactive-paths), [Pi](#defaults-sessions-and-timeouts)); `up` prints a note to start a new session when an entry changed.

Exit codes: `pi --mode json` exits 0 when the provider fails, because only the text mode turns a failed reply into exit 1 (`dist/modes/print-mode.js:17` and `103-139`, `dist/main.js:868-879`), so look for an assistant `message_end` with `stopReason` `error`; Hermes exits 2 when it refuses a context below 64,000 tokens (R19).
llmctl exits 3 when a client file refuses the start, and 6 when the model is running but a client file or command failed ([llmctl.md](llmctl.md#when-up-refuses)).
llmctl waits up to the runtime's `start_timeout_s` for the server, then 2 s at most for Pi's settings lock.

## Verified on this laptop

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

- Re-check the pinned versions before relying on this document: Hermes Agent v0.23.1 and Pi 1.5.2 (see [Versions](#versions)); on an upgrade, rerun the e2e below before trusting the entries.
- Watch Hermes' `gateway.standalone` shim in each Hermes release; if it goes, the profile llmctl may be served by the messaging gateway, which the gateway warning reports.
- Hermes' `_config_version` default (53 at v0.23.1) and its owned keys: a newer Hermes migrates the stamp itself, but a new key that must live in the block needs a change in `launcher/hermes_config.py`.
- Pi's `ModelDefinitionSchema` keys (`dist/core/model-config.js:32-49`): Pi 1.5.2 accepts keys its schemas do not list (`dist/core/model-config.js:32-89` sets no `additionalProperties: false`), so a key llmctl writes that a later Pi drops or renames may still pass the check during `up` while the setting stops working; compare the keys on every Pi upgrade and rerun the e2e below.
- The e2e run proves real sessions through OpenCode, Hermes and Pi on both runtimes; run it only alone, through heavy, after the unit suite passes: `heavy python3 -m launcher.e2e.clients --out research/<date>/llmctl-clients-e2e [--matrix default|extended] [--keep-running] [--set-default-round-trip]`.
  Before it starts anything, it refuses with exit 1 when `LLMCTL_CLIENTS` is set, when an llmctl model server is running (`error: an llmctl model server is running (<recipe> on <runtime>); stop it with ./llmctl down, then run the driver again`), or when the `--out` folder exists.
  With `--set-default-round-trip` it also refuses when OpenCode's config has no top-level `"model"` to restore, when `HERMES_HOME` names a profile in that shell, or when Pi's `enabledModels` would start a session without `--model` on another model, because the round trip runs `hermes` without `-p` and `pi` without `--model`.
  `--out` holds only the records: one file per command or probe under `records/`, and the `summary.md` table.
  The client sessions run in fresh work folders under one private folder per run, made with `mkdtemp` (prefix `llmctl-e2e-`, mode 0700) outside the repository and left in place; the first line names both, `records in <out>; work folders in <work root>, left in place after the run`, and every record shows that folder as `$WORK` and your home as `$HOME`.
  The live-gateway check runs 35 s after the first passing `up` that created the profile llmctl, or right after the first passing `up` when the profile existed before the run, and adds a `gateway` row; when no `up` created the profile, `summary.md` gets `- | gateway | No | no up created the profile llmctl (<path>), so the gateway check did not run`, and a `status` step that prints the gateway warning fails as well.
  The round trip runs `./llmctl up qwen3.6-35b-a3b --set-default`, then OpenCode, `hermes -z` and Pi without `--model` or `-p`, and always restores the defaults it snapshotted; `summary.md` gets a `round-trip | check | No | <Type>: <error>` row when the check itself failed, and a `round-trip | restore` row with each value the restore could not put back.
  It writes your real client files and moves defaults that point at llmctl, ending on qwen3.6-35b-a3b; its run of 2026-11-16 is in [Verified on this laptop](#verified-on-this-laptop), and the request shapes above keep their stub labels, because the e2e records the client sessions and the probe replies, not the requests the clients send.
