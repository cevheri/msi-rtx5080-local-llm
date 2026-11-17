# Kimi Code 2.5.3 experiments before the llmctl Kimi design

Date: 2026-11-17.
Scope: the experiments E1 to E8 of the brief `$SCRATCH/prompts/kimi-experiments.md`, which settle the open risks of `docs/superpowers/research/2026-11-17-kimi-code/research-report.md` (its sections 8 and 10) for option A1.
Nothing in the repository, in the owner's `~/.kimi-code` or in any other client's files changed; there was no Kimi login, no request left the machine, and no model server was started, stopped or contacted (every check used the recording stub).

Paths below are relative to `$SCRATCH/kimi/exp/`.
Every kept record has the scratch root, the home folder, the user name and the host name replaced with `$SCRATCH`, `$HOME`, `<user>` and `<host>` (`bin/sanitize.py`), and the bearer tokens of the sandboxed `kimi web` runs replaced with `<redacted>`.
The private Kimi homes, work folders and socket folders were deleted after the runs, because Kimi's session files there hold whole system prompts; what remains is `runs/`, `writer/`, `entries/`, `records/`, `neg/`, `agents/`, `registry/` and `bin/`.

Labels: "(measured on the MSI Raider 18 Max HX, Kimi Code 2.5.3, recording stub)" for runs answered by `bin/stub.py`, "(measured on the MSI Raider 18 Max HX, Kimi Code 2.5.3, no server)" for runs that needed none, "(inferred from ...)" and "not run: ...".

## How every run worked

- `bin/case.sh` runs one case as one `heavy` invocation, one at a time: `bwrap --unshare-net --unshare-ipc --unshare-uts --new-session --die-with-parent`, a tmpfs over `/tmp` and over `$HOME` with only `$HOME/.kimi-code/bin/kimi` bound back read-only, `--clearenv`, and a private `HOME`, `KIMI_CODE_HOME` and `KIMI_CODE_CACHE_DIR` per case, with `KIMI_DISABLE_TELEMETRY=1`, `KIMI_CODE_NO_AUTO_UPDATE=1` and `KIMI_CODE_REGION_MARKER=off`.
- Bridge: inside the sandbox a socat listens on 127.0.0.1 at the entry's own port (8080 for llama.cpp, 8000 for vLLM) and forwards to a unix socket; outside, a socat forwards that socket to the recording stub `bin/stub.py` on host 127.0.0.1:18491, so every entry's `base_url` stayed exactly as generated.
- The stub answers streamed chat completions and records one reduced shape per request (`runs/<case>/records.jsonl`): body keys, `model`, `max_tokens`, `reasoning_effort`, `stream_options`, `prompt_cache_key` presence, tool count and names, message roles, and per message only key names, sizes and whether a replayed reasoning text equals the stub's; no message text is kept.
- `bin/inner.sh` times the command inside the sandbox (`runs/<case>/time`) and, unless a case says otherwise, records `connect`, `sendto` and `sendmsg` with strace (`runs/<case>/strace.txt`).
- Print mode checks `default_model` even when `-m` is given (E5), so the E2, E3 and E7 configs carry `default_model = "<alias>"` above llmctl's two tables, as `--set-default` would write it.
- Network, over all 67 straced runs: AF_INET connects only to 127.0.0.1 ports 8080 and 8000 (the bridge), 47400 (the in-sandbox registry) and 58627 (`kimi web`, by curl), plus DNS to 127.0.0.53 in the one straced idle TUI run (E8); no DNS in any print-mode, `provider`, `doctor` or `kimi web` run (measured on the MSI Raider 18 Max HX, Kimi Code 2.5.3, recording stub).
- Two harness faults happened on the way and their runs were discarded: an absolute unix socket path longer than the 108-byte limit (fixed with a relative path), and a stub left over from an interrupted run that answered a whole first E2 pass (fixed: `case.sh` now refuses a busy port and cleans up on exit, and E2 ran again from scratch).

## 1. Per experiment

### E1. Generated entries for all 19 recipes

What ran: `python3 -I -B bin/gen_kimi.py <repo> entries`, which imports `launcher.catalog.load_catalog` read-only and writes, per recipe, report 5.1's `[providers.llmctl]` (`type`, `base_url`, `api_key = "no-key-required"`, `model_source = "static"`) and `[models."llmctl/<served>"]` (`provider`, `model`, `display_name = "<id> on <runtime> (llmctl)"`, `max_context_size` = recipe context, `max_output_size` = `[clients] output`, `capabilities`, and for the effort recipes `support_efforts` without `none`, `default_effort = "medium"`, `off_effort = "none"` only for Qwen3.8-27B).
Result: 19 entries, 10 toggle, 3 effort, 5 always, 1 none; the two DeepSeek-R1 recipes get `capabilities = []`; `git status` stayed clean.
Evidence: `entries/<runtime>/<id>.toml`, `entries/index.json`.
Label: (measured on the MSI Raider 18 Max HX, Kimi Code 2.5.3, no server).

### E2. Recording-stub lane for all 19 entries

What ran, per entry (`bin/e2.py`): `kimi provider list --json` on the entry, then `kimi -p "Reply with exactly OK." --output-format stream-json -m llmctl/<served>`.
Result, plain lane: 19 of 19 accepted (the parsed provider and alias carry every intended value) and 19 of 19 exit 0 with the reply `OK` in 0.57 to 0.63 s, one request each, `config.toml` unchanged in every run (measured on the MSI Raider 18 Max HX, Kimi Code 2.5.3, recording stub).
All 72 requests the stub saw in this work (E2, E3, E8 and the extras): `POST /v1/chat/completions`, `stream: true`, `stream_options: {"include_usage": true}`, `prompt_cache_key` present, `User-Agent: kimi-code-cli/2.5.3`, `Authorization: Bearer no-key-required`, no `X-Msh-*` header, and never `temperature`, `top_p`, `tool_choice`, `parallel_tool_calls`, `chat_template_kwargs`, `max_completion_tokens`, `thinking` or `reasoning`; no `GET /models` (measured, same label).
First request of every entry: keys `max_tokens, messages, model, prompt_cache_key, stream, stream_options, tools`, plus `reasoning_effort: "medium"` for the three effort entries only; `model` is the served name; `max_tokens` is the recipe's output cap (32768 for llama.cpp K2 Horizon 7B, 16384 elsewhere); 24 tools, the chat-only DeepSeek-R1 entries included; roles `system, user, user, user` (table in section 2).
Reasoning lane: for one recipe of each kind the stub's first reply streamed reasoning deltas plus one `Write` call (`hello.txt` in the work folder), and its second reply was text; all 7 runs, and the 4 vLLM-field variants, exit 0 with `hello.txt` written and two requests each.
The second request replays the assistant turn as `content: null`, the `tool_calls` and the reasoning text verbatim under the same field the server used (`reasoning_content` from the llama.cpp-style reply, `reasoning` in the vLLM-style variants), and carries the same `reasoning_effort` as the first: absent for toggle off (Qwen3.6), toggle on (Mellum), always (vLLM K2 Horizon 7B), none (Qwen3-Coder-Next) and chat-only (vLLM DeepSeek-R1), `medium` for Qwen3.8-27B and gpt-oss-20b (measured, same label).
A reply with reasoning did not make Kimi add `reasoning_effort` for the entries without the `thinking` capability (measured, same label).
Thinking switches (print mode has no flag; the switch is the global `[thinking]` table): Qwen3.8-27B sends `none` with `enabled = false`, `low` with `effort = "low"` and `medium` with `enabled = true`, in both requests; gpt-oss-20b on llama.cpp and on vLLM sends `low` and `medium` the same way, and with `enabled = false` exits 1 in 0.53 s before any request with `error: failed to run prompt: provider.api_error: Model 'local' reasons by default but declares no off effort, so thinking cannot be turned off. Declare offEffort (for example 'none') for this model in the model catalog configuration.` (`'gpt-oss-20b'` on vLLM); Qwen3.6 sends nothing with `enabled` true or false (measured, same label).
Extra, continued sessions (`bin/e2c.sh`, vLLM K2 Horizon 7B, a second `kimi -p ... -c` in the same folder): the earlier text answer is replayed with its `reasoning` field when the server sent one, and with no thinking field at all when it did not (measured, same label).
Evidence: `runs/e2-summary-all.jsonl`, `runs/e2-<runtime>-<id>-list/stdout`, `runs/e2-<runtime>-<id>-p/`, `runs/e2r-*/`, `runs/e2t-*/`, `runs/e2c-*/`, `records/provider-list-*.json`.

### E3. Chat-only without tools

Agent file format (docs K109 and K123, the agents page, and `parseAgentFileText` at main.cjs:69803-69835): Markdown with YAML front matter; `description` and a non-empty body are required; `tools: []` (or `tools: ""`) is an empty allowlist; `--agent-file` loads one file for one launch.
The file used, `agents/chat-only.md`:

```markdown
---
name: chat-only
description: Chat-only agent for llmctl's chat-only recipes; it has no tools.
tools: []
---

You are a helpful assistant. Answer the user directly in plain text. You have no tools.
```

What ran (`bin/e3.py`): `kimi -p "Reply with exactly OK." --output-format stream-json -m llmctl/deepseek-r1-8b [--agent-file <file>]` against a stub that plays vLLM without a tool parser: HTTP 400 `"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set` when the body has `tools`, a plain reply otherwise.
Result: without an agent file the request carries 24 tools, Kimi does not retry the 400, and exits 1 in 0.61 s with `error: failed to run prompt: provider.api_error: "auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set`.
With `--agent-file agents/chat-only.md` the request has no `tools` key at all (keys `max_tokens, messages, model, prompt_cache_key, stream, stream_options`), the system message is the 87-character body, and the run exits 0 with `OK` in 0.57 s; `tools: ""` behaves the same; a body of `${base_prompt}` (`agents/chat-only-base.md`) keeps Kimi's default prompt (7,859 characters against 10,187 with tools) and also sends no `tools`; the llama.cpp DeepSeek-R1 entry behaves the same.
Evidence: `runs/e3-summary.jsonl`, `runs/e3-*/`, `agents/`.
Label: (measured on the MSI Raider 18 Max HX, Kimi Code 2.5.3, recording stub).

### E4. Kimi's config writer

Inputs: `writer/f0-template-plus-llmctl.toml` (Kimi's own 185-byte comment template, as Kimi wrote it into an empty private home, plus llmctl's gemma-4-26b-a4b tables) and `writer/f2-default-at-top-plus-other.toml` (template comments, `default_model` above the first table, llmctl's tables, and a hand-written second provider and model).
Every result is kept as a fixture with its note in `writer/index.json`; `writer/check.json` is the `tomllib` check of each (`bin/check_fixtures.py`): llmctl's provider and model tables parse equal to the generator's values in all eight files.

(a) First `default_model` written by Kimi into a file with tables and no `default_model`.
Non-interactive route found: `kimi web --no-open` in the sandbox and `POST /api/v1/models/llmctl%2Flocal:set_default` with the banner token (docs server-api page; `bin/e4_web.sh`).
Result: the whole file was rewritten: the four comment lines are gone, `default_model = "llmctl/local"` is the first line, arrays are re-serialized as `[ "tool_use" ]`, the file ends in a blank line, mode stays 0600, the inode changed (a new file renamed into place), and llmctl's values survived (`writer/a-web-set-default.toml`, `runs/e4a-web-set-default/`).
The TUI route was also run, through a pseudo-terminal (`bin/tui.py`): `/model`, Enter, Enter on `llmctl/local`, whose selector row shows thinking as `On (Unsupported) [ Off ]`.
It rewrote the file the same way and also appended `[thinking]` with `enabled = false` (`writer/a2-tui-model-pick.toml`, `runs/e8-tui/c1-after-pick.toml`; the mode of this one was not measured).
A `default_model` that already exists is edited in place: switching it to `other/m` changed one line and kept the comments and all tables (`writer/a3-web-switch-default.toml`).
Label: (measured on the MSI Raider 18 Max HX, Kimi Code 2.5.3, no server).

(b) `kimi provider add http://127.0.0.1:47400/api.json`, with `registry/v1/api.json` (format from `toProviderEntry` and `fetchCustomRegistry`, main.cjs:17147-17206) served by `python3 -m http.server` inside the sandbox.
Result: exit 0, `Imported 1 provider (1 model)`; the whole file was rewritten (comments gone), and the new tables landed in Kimi's order: `[providers.llmctl]`, `[providers.dummy]`, `[providers.dummy.source]`, `[models."llmctl/local"]`, `[models."dummy/dummy-model"]`, so the new provider sits between llmctl's two tables; the new provider has `api_key = ""` and a `source` sub-table with the camelCase key `apiKey`; llmctl's tables are equal in value (`writer/b-provider-add.toml`, `runs/e4b-provider-add/`); a second run from the same input gave a byte-identical file.
Label: (measured on the MSI Raider 18 Max HX, Kimi Code 2.5.3, no server).

(c) `kimi provider remove dummy` on the result of (d).
Result: exit 0, `Removed provider "dummy".`; the dummy provider, its `source` table and both dummy models went by line edits; llmctl's tables are byte-identical; three blank lines were left between llmctl's two tables and three at the end of the file (`writer/c-provider-remove.toml`, `runs/e4c-provider-remove/`).
Label: (measured on the MSI Raider 18 Max HX, Kimi Code 2.5.3, no server).

(d) Start-time catalog refresh.
The refresh with `refresh_on_start` (default true) runs in the `kimi web` server (`ModelCatalogRefreshScheduler`, main.cjs:393038-393110, created at main.cjs:403356); print mode showed no refresh.
With the registry changed to `registry/v2/api.json`, a `kimi web` start fetched `GET /api.json` once and rewrote the dummy tables (context 65536, a new `[models."dummy/dummy-model-2"]` after the last model table), while llmctl's `model_source = "static"` tables stayed byte-identical (`writer/d-web-start-refresh.toml`, `runs/e4d-web-start-refresh/`); a `kimi web` start on `f0`, whose only provider is static, left the file unchanged (`runs/e4a-web-set-default/config.after-start.toml`).
Label: (measured on the MSI Raider 18 Max HX, Kimi Code 2.5.3, no server).

### E5. A `default_model` that names a missing alias

What ran (`bin/e5e7.py e5`): `kimi -p "Reply with exactly OK."` on llmctl's gemma tables with the defaults below; no run sent a request and none changed `config.toml`.

| `default_model` | `-m` | Exit | Time | Message on stderr |
| --- | --- | --- | --- | --- |
| `llmctl/missing` | none | 1 | 0.51 s | `error: failed to run prompt: Model "llmctl/missing" is not configured in config.toml.` (the same with `--output-format text`, 0.50 s) |
| `llmctl/missing` | `llmctl/local` | 1 | 0.52 s | `error: failed to run prompt: model llmctl/missing does not resolve to a configured provider` |
| unset | `llmctl/local` | 1 | 0.53 s | `error: failed to run prompt: no default model configured` |
| unset | none | 1 | 0.42 s | ``error: failed to run prompt: No model configured. Run `kimi` and use /login to sign in, then retry; or set default_model in config.toml.`` |
| `llmctl/local` | `llmctl/missing` | 1 | 0.50 s | `error: failed to run prompt: Model "llmctl/missing" is not configured in config.toml.` |

Each stderr ends with `See log: <KIMI_CODE_HOME>/logs/kimi-code.log`; in stream-json the only stdout line is the `system.version` meta line.
Why `-m` does not help: print mode's `runNativeTurn` calls `ensureReady()` without the model override, so it resolves `default_model` (main.cjs:499744-499746, 261009-261018) (inferred from main.cjs, matches the runs).
The TUI differs: `kimi -m llmctl/local` with no `default_model` started, answered through the stub and left the file unchanged (`runs/e8-tui-m/`).
Evidence: `runs/e5e7-summary-e5.jsonl`, `runs/e5-*/`.
Label: (measured on the MSI Raider 18 Max HX, Kimi Code 2.5.3, recording stub; no request reached it).

### E6. The cost of llmctl's checks

What ran (`bin/e6.sh`): `kimi --version` and `kimi provider list --json` (on the gemma entry), each three times in a fresh private home and cache and three times again warm, without strace for the timing; one straced run of each; times are of the `kimi` process inside the sandbox.

| Command | Cache | Wall time (3 runs) | Written |
| --- | --- | --- | --- |
| `kimi --version` | fresh | 0.271, 0.276, 0.276 s | 9 files, 1,932,692 bytes (18 folders) under `KIMI_CODE_CACHE_DIR`; nothing in the home |
| `kimi --version` | warm | 0.277, 0.282, 0.285 s | nothing (sizes and modification times unchanged) |
| `kimi provider list --json` | fresh | 0.317, 0.308, 0.311 s | the same 1,932,692 cache bytes plus `device_id` (36 bytes) in the home |
| `kimi provider list --json` | warm | 0.314, 0.313, 0.315 s | nothing |

No network: `--version` made no connect call at all, and `provider list` only four AF_UNIX connects to `/var/run/nscd/socket` (ENOENT) (`runs/e6-strace-version/`, `runs/e6-strace-list/`).
A home without `config.toml` took 1.37 s, because Kimi first writes its 185-byte template (`runs/e6-empty-list/`).
The exact output for the gemma entry (`records/provider-list-llamacpp-gemma-4-26b-a4b.json`), camelCase and in Kimi's key order:

```json
{
  "providers": {
    "llmctl": {
      "modelSource": "static",
      "baseUrl": "http://127.0.0.1:8080/v1",
      "type": "openai",
      "apiKey": "no-key-required"
    }
  },
  "models": {
    "llmctl/local": {
      "provider": "llmctl",
      "model": "local",
      "maxContextSize": 131072,
      "maxOutputSize": 16384,
      "capabilities": [
        "tool_use"
      ],
      "displayName": "gemma-4-26b-a4b on llama.cpp (llmctl)"
    }
  }
}
```

The effort keys print as `supportEfforts`, `defaultEffort` and `offEffort` after `displayName` (`records/provider-list-llamacpp-qwen3.8-27b.json`), and `capabilities = []` prints as `[]` (`records/provider-list-vllm-deepseek-r1-0528-qwen3-8b.json`).
Negative candidates (`neg/`, `runs/neg-*/`, `runs/doctor-*/`): on a file that is not valid TOML, `provider list --json` exits 0 with `{"providers": {}, "models": {}}` and an empty stderr, and the parse error appears only in Kimi's log (`records/provider-list-invalid-toml.json`); `kimi doctor config` exits 1 in 0.28 s on the same file with `Invalid TOML ... (line 4, column 27)`, but exits 0 for a model without `max_context_size`, which `provider list` shows without `maxContextSize`; an unknown key in `[providers.llmctl]` is dropped from the output, an unknown key in the model table is kept; a `default_model` line appended after the last table is read as a key of that model table (`"defaultModel": "llmctl/local"` inside `models."llmctl/local"`) and accepted with exit 0.
Label: (measured on the MSI Raider 18 Max HX, Kimi Code 2.5.3, no server).

### E7. A stopped server

What ran (`bin/e5e7.py e7`): `kimi -p "Reply with exactly OK." --output-format stream-json` with `default_model` set and the entry's `base_url` (127.0.0.1:8080) on a port with nothing listening inside the sandbox.
Result: exit 1 after 151.2 s; 10 connect attempts to 127.0.0.1:8080; nine `turn.step.retrying` meta lines on stdout (`max_attempts: 10`, `error_name: "APIConnectionError"`, `error_message: "Connection error."`) with delays of 0.60, 1.03, 2.23, 4.58, 9.22, 16.28, 39.82, 37.03 and 39.76 s (150.5 s in total); stderr `error: failed to run prompt: provider.connection_error: Connection error.` and the log path.
A bridge that accepted and closed the connection (the harness fault above) gave the same ladder and message after 147.8 s (`runs/e7-reset-by-bridge/`, exit code not captured).
Evidence: `runs/e5e7-summary-e7.jsonl`, `runs/e7-refused-json/`.
Label: (measured on the MSI Raider 18 Max HX, Kimi Code 2.5.3, no server).

### E8. An open TUI when the file changes under it

What ran: the TUI through a pseudo-terminal inside the sandbox (`bin/tui.py`, `KIMI_CODE_TRUST_WORKSPACE=1`, `TERM=xterm-256color`), with the stub bridged on 8080; the file was replaced by a rename, as llmctl writes it.
Alias changed (`runs/e8-tui/`): after `/model` and one answered prompt on `llmctl/local`, the file was replaced with `default_model = "llmctl/qwen3.5-9b"` and the vLLM qwen3.5-9b tables (port kept on the bridge); five seconds later the next prompt failed in the TUI with `Error: [config.invalid] Model "llmctl/local" is not configured in config.toml.`, no request was sent, the footer still named gemma, and the TUI did not write the file again before or after exit.
Alias kept (`runs/e8-tui-same/`): with `llmctl/local` swapped from the gemma tables to the Qwen3.8-27B tables, the next prompt went out with the new entry's values and `reasoning_effort: "none"`, because the session kept the thinking level `off` it had on gemma, while the footer still showed gemma and 128k; the file was not written back.
No login, local provider only: the TUI starts, shows `Model: not set, run /login or /provider` when `default_model` is unset, and serves `-m llmctl/local` (`runs/e8-tui-m/`).
Side observation: an idle TUI started with the three off switches looked up `api.kimi.com` 12 times and `code.kimi.com` 4 times in 12 s (DNS to 127.0.0.53, which the sandbox cannot reach) (`runs/e8-tui-idle-strace/strace.txt`); print mode never did; the purposes, client configs and the CDN, are inferred from local K43 and K44.
Label: (measured on the MSI Raider 18 Max HX, Kimi Code 2.5.3, recording stub).

## 2. The 19 entries

Accepted means `kimi provider list --json` listed provider `llmctl` and the alias with every value the generator wrote.
The request columns are from the first request of the E2 plain run.

| Recipe | Runtime | Served name | Alias | Thinking type | `capabilities` | Effort keys | `max_context_size` | `max_output_size` | Accepted | First request's keys | `reasoning_effort` | Tools |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deepseek-r1-0528-qwen3-8b | llama.cpp | `local` | `llmctl/local` | always, chat-only | `[]` | none | 65536 | 16384 | yes | max_tokens, messages, model, prompt_cache_key, stream, stream_options, tools | absent | 24 |
| gemma-4-26b-a4b | llama.cpp | `local` | `llmctl/local` | toggle, off | `tool_use` | none | 131072 | 16384 | yes | max_tokens, messages, model, prompt_cache_key, stream, stream_options, tools | absent | 24 |
| gpt-oss-20b | llama.cpp | `local` | `llmctl/local` | effort | `tool_use`, `thinking` | `support_efforts = ["low", "medium", "high"]`, `default_effort = "medium"` | 131072 | 16384 | yes | max_tokens, messages, model, prompt_cache_key, reasoning_effort, stream, stream_options, tools | `medium` | 24 |
| groot-19b-a3b | llama.cpp | `local` | `llmctl/local` | toggle, off | `tool_use` | none | 131072 | 16384 | yes | max_tokens, messages, model, prompt_cache_key, stream, stream_options, tools | absent | 24 |
| k2-horizon-7b | llama.cpp | `local` | `llmctl/local` | always | `tool_use` | none | 81920 | 32768 | yes | max_tokens, messages, model, prompt_cache_key, stream, stream_options, tools | absent | 24 |
| k2-horizon-mova-36b | llama.cpp | `local` | `llmctl/local` | always | `tool_use` | none | 65536 | 16384 | yes | max_tokens, messages, model, prompt_cache_key, stream, stream_options, tools | absent | 24 |
| nemotron-3.5-bolt | llama.cpp | `local` | `llmctl/local` | toggle, off | `tool_use` | none | 131072 | 16384 | yes | max_tokens, messages, model, prompt_cache_key, stream, stream_options, tools | absent | 24 |
| nemotron-3.5-lightning | llama.cpp | `local` | `llmctl/local` | toggle, off | `tool_use` | none | 131072 | 16384 | yes | max_tokens, messages, model, prompt_cache_key, stream, stream_options, tools | absent | 24 |
| qwen3-coder-next | llama.cpp | `local` | `llmctl/local` | none | `tool_use` | none | 131072 | 16384 | yes | max_tokens, messages, model, prompt_cache_key, stream, stream_options, tools | absent | 24 |
| qwen3.6-35b-a3b | llama.cpp | `local` | `llmctl/local` | toggle, off | `tool_use` | none | 131072 | 16384 | yes | max_tokens, messages, model, prompt_cache_key, stream, stream_options, tools | absent | 24 |
| qwen3.8-27b | llama.cpp | `local` | `llmctl/local` | effort | `tool_use`, `thinking` | `support_efforts = ["low", "medium"]`, `default_effort = "medium"`, `off_effort = "none"` | 65536 | 16384 | yes | max_tokens, messages, model, prompt_cache_key, reasoning_effort, stream, stream_options, tools | `medium` | 24 |
| deepseek-r1-0528-qwen3-8b | vLLM | `deepseek-r1-8b` | `llmctl/deepseek-r1-8b` | always, chat-only | `[]` | none | 65536 | 16384 | yes | max_tokens, messages, model, prompt_cache_key, stream, stream_options, tools | absent | 24 |
| gemma-4-12b | vLLM | `gemma-4-12b` | `llmctl/gemma-4-12b` | toggle, off | `tool_use` | none | 131072 | 16384 | yes | max_tokens, messages, model, prompt_cache_key, stream, stream_options, tools | absent | 24 |
| gemma-4-e4b | vLLM | `gemma-4-e4b` | `llmctl/gemma-4-e4b` | toggle, off | `tool_use` | none | 65536 | 16384 | yes | max_tokens, messages, model, prompt_cache_key, stream, stream_options, tools | absent | 24 |
| gpt-oss-20b | vLLM | `gpt-oss-20b` | `llmctl/gpt-oss-20b` | effort | `tool_use`, `thinking` | `support_efforts = ["low", "medium", "high"]`, `default_effort = "medium"` | 65536 | 16384 | yes | max_tokens, messages, model, prompt_cache_key, reasoning_effort, stream, stream_options, tools | `medium` | 24 |
| groot-19b-a3b | vLLM | `groot-19b-a3b` | `llmctl/groot-19b-a3b` | toggle, off | `tool_use` | none | 131072 | 16384 | yes | max_tokens, messages, model, prompt_cache_key, stream, stream_options, tools | absent | 24 |
| k2-horizon-7b | vLLM | `k2-horizon-7b` | `llmctl/k2-horizon-7b` | always | `tool_use` | none | 65536 | 16384 | yes | max_tokens, messages, model, prompt_cache_key, stream, stream_options, tools | absent | 24 |
| mellum2.1-12b | vLLM | `mellum2.1` | `llmctl/mellum2.1` | toggle, on | `tool_use` | none | 131072 | 16384 | yes | max_tokens, messages, model, prompt_cache_key, stream, stream_options, tools | absent | 24 |
| qwen3.5-9b | vLLM | `qwen3.5-9b` | `llmctl/qwen3.5-9b` | toggle, off | `tool_use` | none | 131072 | 16384 | yes | max_tokens, messages, model, prompt_cache_key, stream, stream_options, tools | absent | 24 |

Source: `entries/index.json` and `runs/e2-summary-all.jsonl` (measured on the MSI Raider 18 Max HX, Kimi Code 2.5.3, recording stub).

## 3. Design consequences

### Where the design must differ from report section 5

1. `kimi -p -m llmctl/<served>` needs a resolvable `default_model` (E5).
   The owner's file has no `default_model` today, so without `--set-default` the print-mode route fails with `no default model configured`, and any `default_model` naming a missing alias breaks every `kimi -p`, with or without `-m`.
   The not-default lines of 5.7 can offer `kimi -m llmctl/<served>` and `/model` for the TUI (which works, E8), but not `kimi -p ... -m`, or they must say that print mode needs a default.
   The e2e of section 6 cannot use a byte copy of the real file: its private copy needs `default_model = "llmctl/<served>"` (above the first table) when the real one is unset or points elsewhere.
   `status` could warn when `default_model` names an alias that is not in the file, since that is a hard failure for print mode, not a fallback.
2. A TUI `/model` pick writes `[thinking]` (E4a, TUI route).
   Picking any of the 16 entries without the `thinking` capability writes `[thinking] enabled = false`, which then turns gpt-oss-20b (both runtimes) into an exit 1 before any request and makes Qwen3.8-27B send `none` (E2 thinking runs), until the owner picks an effort entry in `/model` again; picking an effort entry should write `enabled = true` with `effort = "medium"` (inferred from `thinkingEffortToConfig`, main.cjs:481166-481180, and the selector's `segmentsFor` and `commitEffort`, main.cjs:479437-479469; not run).
   Section 5.3's table holds, but the design must choose between documenting this, giving gpt-oss-20b an `off_effort` (which sends a level the owner did not choose), or touching `[thinking]`, which the report leaves to the owner.
3. The tables llmctl owns are not adjacent after Kimi writes (E4b).
   Kimi groups all `[providers.*]` tables before all `[models.*]` tables, so another provider lands between `[providers.llmctl]` and `[models."llmctl/<served>"]`; the finder must locate each table by its header, accept Kimi's `[ "tool_use" ]` array spacing and runs of blank lines (three after a `provider remove`), and skip other providers' sub-tables such as `[providers.dummy.source]`.
   Inserting a missing table after the last table of its kind, as 5.1 says, matches what Kimi does.
4. Kimi writes `default_model` as the first line after any full rewrite (E4a), which matches 5.2's insert-above-the-first-header rule; a line appended at the end is silently read as a key of the last table (E6 negative `n6`), so check 1 must compare the top-level key.
5. Check 3 of 5.6 cannot rely on the exit status: `kimi provider list --json` exits 0 with empty maps on a file it cannot parse (E6).
   It passes only when provider `llmctl` and the alias are present with the intended values; the output keys are camelCase in Kimi's order, so compare parsed JSON, mapping `baseUrl`, `apiKey`, `modelSource`, `maxContextSize`, `maxOutputSize`, `displayName`, `supportEfforts`, `defaultEffort` and `offEffort`.
   `kimi doctor config` adds a syntax verdict (exit 1 with line and column) if wanted, at 0.28 s more.
6. Cost of the checks: `kimi --version` 0.27 to 0.28 s; `provider list --json` 0.31 s and 1.93 MB written per fresh private cache folder, 0.31 s and nothing written with a warm one; no network either way (E6).
   A cache folder that llmctl keeps (for example under its own state folder) saves the 1.93 MB per `up`; a fresh temporary one per check is simpler; a lead decision.
7. Open sessions (5.7 note, E8): after an `up` that changes the alias, an open TUI session fails its next turn with `Model "<old alias>" is not configured in config.toml.` until the owner runs `/model`; after an `up` that keeps `llmctl/local`, the session silently uses the new entry with its old thinking level.
   The note should say "run /model or start a new session" rather than relying on Kimi's reload.
8. Chat-only (5.4, E3): with an agent file whose front matter has `tools: []`, Kimi sends no `tools`, so the e2e can drive both DeepSeek-R1 recipes; where the agent file lives (the e2e work root, or a file llmctl writes into the Kimi home's `agents/` for `--agent`) is a lead decision.
9. A stopped server costs 151 s of retries before exit 1 (E7); `down` stays silent as designed, and the e2e's 900 s step bound covers it, but the `down` line could say that Kimi will retry for about two and a half minutes.
10. Privacy (report section 7): the off switches do not keep an interactive TUI off the network; it resolves `api.kimi.com` and `code.kimi.com` (E8 side observation), so the privacy table's "print mode did not call it" now has a measured TUI counterpart.

### Report risks now settled

| Risk (report section 8) | Status | Evidence |
| --- | --- | --- |
| 1, Kimi's writer | Settled: a first `default_model` by the web API or the TUI rewrites the whole file without comments and keeps llmctl's values; an existing `default_model` is edited in place; `provider add` rewrote the whole file and put the new provider between llmctl's tables; `provider remove` removed only the dummy provider; no write-back of old values by an open TUI was seen | E4 (`writer/`), E8 |
| 2, vLLM | Half settled: Kimi sends the vLLM entries the same keys as llama.cpp's (served names as `model`); whether vLLM v0.33.0 accepts `prompt_cache_key`, `stream_options` and `reasoning_effort` is still not run (stub only) | E2 |
| 3, reasoning models | Settled for Kimi's side: reasoning is replayed verbatim under the field the server used, `reasoning_effort` is unchanged after a reply with reasoning, and a turn without reasoning is replayed without a thinking field, which K2 Horizon 7B's vLLM template rejects (inferred from docs/clients.md:107) | E2 reasoning lane, `runs/e2c-*` |
| 6, missing alias | Settled: exit 1 in about 0.5 s with `Model "<alias>" is not configured in config.toml.`, no fallback, and the same failure with a valid `-m` | E5 |
| 7, chat-only | Settled: `--agent-file` with `tools: []` sends no `tools` | E3 |
| 8, open sessions | Settled for the TUI: alias change fails the next turn, same alias switches silently, no write-back | E8 |
| 9, TUI without login | Settled: it starts, shows `Model: not set` without a default, and serves the local provider; it does DNS lookups of kimi.com hosts | E8 |
| 10, timeouts | Partly: a stopped server fails after 151 s of retries; the 300 s first-byte timeout was not run | E7 |
| 11, documentation | `max_output_size` is sent as `max_tokens` on the `openai` wire in 19 of 19 entries | E2 |

Risks 4 (the owner's own settings) and 5 (auto-update) are unchanged, except that risk 4 now includes Kimi's own `[thinking]` writes (consequence 2).
Not run: any request to the live llama.cpp server (the stub answered everything), a vLLM server, the `/effort` command, and the TUI pick of an effort entry.
