# llmctl: start a measured model by name

Last verified: 2026-11-18.

`./llmctl` starts a model from this repository's recipes with one command, applies the rules from [README.md](../README.md#rules-for-every-setup-here) for you, and points OpenCode, Hermes Agent, Pi, DeepSeek Harness (dsh) and Kimi Code CLI at it ([Clients](#clients)).
It runs llama.cpp, vLLM and SGLang in Docker, one model server at a time, on an NVIDIA GPU.
The recipes were measured on an MSI Raider 18 Max HX A2WI with an RTX 5080 Laptop GPU (16 GB) and 61 GiB of RAM; on another machine, read `./llmctl list` and write an override wherever yours differs.

## Quick start

```bash
./llmctl list                 # every recipe, its evidence, and what is downloaded
./llmctl pull qwen3.6         # asks, then downloads the model files, and the image if it is missing
./llmctl up qwen3.6           # checks the machine, starts the server, waits until it answers
./llmctl test                 # a reply of exactly OK, a tool call and a decode speed
./llmctl down                 # stops the server
```

`up` prints the OpenCode model to use, for example `llamacpp-docker/qwen3.6-35b-a3b`, which works with the provider config in [opencode.md](opencode.md#the-validated-config), and the Hermes, Pi, dsh and Kimi commands, `hermes -p llmctl`, `pi --model llmctl/local`, `dsh web` with the model `llmctl/local` and `kimi -m llmctl/local`, whose entries it writes itself ([clients.md](clients.md)).
`./llmctl up qwen3.6 --set-default` also makes the model the default of every installed client.
In the OpenCode screen, pick it with `/models` (or `ctrl+x m`); for a single prompt, pass it as `opencode run --model llamacpp-docker/qwen3.6-35b-a3b "..."`, because the interactive `opencode` command has no `-m` flag ([OpenCode models docs](https://opencode.ai/docs/models/)).
It needs Linux, Docker Engine, the NVIDIA Container Toolkit with a CDI spec ([host-setup.md](host-setup.md)), and Python 3.11 or newer; it uses only Python's standard library, so there is nothing to install.
Checked on the reference laptop on 2026-11-16: `up qwen3.6` was ready in 9 s and passed `test` and an OpenCode session, and `up gpt-oss --runtime vllm` was ready in 29 s with warm compile caches and passed `test` ([records](../research/2026-11-16/llmctl-e2e/)).
The clients e2e of 2026-11-18 ran all 29 recipes, 15 on llama.cpp, 8 on vLLM and 6 on SGLang, with OpenCode, Hermes, Pi, dsh and Kimi: 490 of 510 steps passed, every `up`, `status`, config, long-grammar, probe, restore and `down` step among them, and the 20 failed client sessions were model behaviour, known model limitations, gpt-oss-20b's dropped tool call on SGLang and the vLLM DeepSeek-R1 OpenCode entry, which was then removed (measured on the MSI Raider 18 Max HX, llama.cpp b12407, vLLM v0.33.0 and SGLang v0.5.23, 2026-11-18; [record](../research/2026-11-18/llmctl-sglang-e2e/README.md)).

## Commands

| Command | What it does |
|---|---|
| `./llmctl list [--runtime R]` | Every recipe: ID, runtime, status, aliases, evidence, measured decode speed, context, GPU need (for SGLang the measured peak), RAM need, download size, whether the files and the image are present, and whether an override replaces it; below the table, the running server's address and the client lines of `status` without paths |
| `./llmctl show <model> [--runtime R]` | The effective recipe, its file and SHA-256, and the exact `docker run` command `up` would run |
| `./llmctl show <model> [--runtime R] --as-override` | A complete copy of the shipped recipe, marked as an estimate, to save as an override and edit; redirecting into an override that already exists loses what it held ([Overrides for another machine](#overrides-for-another-machine)) |
| `./llmctl pull <model> [--runtime R] [--yes] [--hf-env-file F]` | Says how much it will download, asks, then pulls the image if it is missing and downloads the model files; without a terminal it needs `--yes` |
| `./llmctl up <model> [--runtime R] [--replace] [--dry-run] [--set-default]` | Runs the checks below, starts the container, waits until the server answers (for SGLang it writes the model overlay before the start and checks the KV pool and the free GPU memory after it, and for a vLLM recipe with `[model] tokenizer_class` it writes the tokenizer overlay before the start), writes the llmctl entries of the installed clients ([Clients](#clients)), and prints the endpoint, the served name, the context, the output cap, each client's model and thinking choices, and the evidence; with `--set-default` it also makes the model every installed client's default |
| `./llmctl status` | The container llmctl started: its recipe, source file and SHA-256, state, port, health, memory use and the GPU memory in use; then each client's llmctl entry and default, and whether they match the running model |
| `./llmctl logs [-f] [--tail N]` | The server's log, the last 100 lines by default |
| `./llmctl test` | Three requests against the running server: a reply that must be exactly OK (ignoring case and any whitespace, quotes, backticks, `*`, `_`, `.`, `!` or `,` around it); one tool call; and a decode of up to 512 tokens |
| `./llmctl down [--remove]` | Stops the server with a 60-second grace period; the stopped container stays for its logs unless `--remove`; then names the clients whose default still points at llmctl, and changes no client file |
| `./llmctl check` | Validates every shipped recipe and every override and lists each problem with its file |

A model is named by its recipe ID or an alias, such as `qwen3.6` for `qwen3.6-35b-a3b`.
When a name has recipes for more than one runtime, llama.cpp comes first, then vLLM, then SGLang, because llama.cpp is the main runtime here; `--runtime vllm` or `--runtime sglang` picks another, and every SGLang recipe needs `--runtime sglang`, because each shares its name with a llama.cpp or vLLM recipe.
Since the GGUF recipes of `gemma-4-12b`, `gemma-4-e4b`, `mellum2.1-12b` and `qwen3.5-9b`, those four names start on llama.cpp unless `--runtime vllm` is given, or `--runtime sglang` for the three that ship an SGLang recipe (`gemma-4-12b`, `gemma-4-e4b` and `qwen3.5-9b`; `mellum2.1-12b` ships none).
For `qwen3.5-9b` that trades speed for precision: its Q8_0 build decodes 66.0 tok/s on llama.cpp (measured on the MSI Raider 18 Max HX, llama.cpp b12407, 2026-11-17; [record](../research/2026-11-17/llamacpp-qwen3.5-9b/README.md)) against 90.8 for the NVFP4 build on vLLM (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-17; [record](../research/2026-11-17/vllm-qwen3.5-9b/README.md)) and 93.7 on SGLang (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17; [record](../research/2026-11-17/sglang-qwen3.5-9b/README.md)), so `--runtime vllm` or `--runtime sglang` is the faster choice for it.
Running `up` for the model that is already running from the same recipe file starts nothing, writes the client entries again where they differ and prints the summary again; `--replace` restarts it.
For an SGLang recipe such an `up` skips the memory check and says so: `note: <id> was already running from this recipe file, so llmctl did not check its memory again; ./llmctl up <id> --runtime sglang --replace restarts it and checks it`.
Run `./llmctl <command> --help` for every option.
In `list`, STATUS is `running` for the server that answers, `starting` while it loads, `running (older file)` when it was started from an earlier version of its recipe file, and `?` when Docker did not answer.

## Clients

`up` points five coding clients at the model it starts: OpenCode v2.3.0, Hermes Agent v0.23.1, Pi 1.5.2, DeepSeek Harness 0.2.2-rc.3 and Kimi Code CLI 2.5.3, the versions this was verified with.
It keeps one llmctl entry per client and rewrites it on every `up` to describe the running recipe, so you pick the model once per client, not once per recipe.
[clients.md](clients.md) shows the Hermes, Pi, dsh and Kimi entries, how each client thinks with them, and what was measured.

A client is installed when its binary is on `PATH` and its config folder exists: OpenCode `$XDG_CONFIG_HOME/opencode/` (or `~/.config/opencode/`), Hermes its root (`~/.hermes/` unless `HERMES_HOME` points outside it, as [clients.md](clients.md#the-profile) explains), Pi `PI_CODING_AGENT_DIR` (or `~/.pi/agent/`).
dsh counts as installed when `dsh` is on `PATH` and `$DSH_HOME/profiles/` exists, which dsh creates the first time any of its profiles runs; the home folder alone is not enough, because Ubuntu's unrelated `dsh` package (the distributed shell) also keeps its files in `~/.dsh`.
`DSH_HOME` follows dsh's own rule: unset, empty or blank means `~/.dsh`, `~` and `~/...` are expanded with your `HOME`, an absolute path is used as given, `$NAME` is never expanded, and any other relative value is a usage error, because dsh resolves it against each command's working folder.
Kimi counts as installed when `kimi` is on `PATH` and its home folder exists: `KIMI_CODE_HOME` when it is set, else `~/.kimi-code/`; the legacy Python kimi-cli keeps its data in `~/.kimi` instead, so the home folder tells the two apart.
`KIMI_CODE_HOME` must be an absolute path, which is used as given; an empty value is a usage error, because Kimi 2.5.3 then reads `config.toml` from the working folder in some commands and from `~/.kimi-code` in others, and any other value, `~` and `~/...` included, is a usage error too, because Kimi expands neither `~` nor `$NAME` and resolves a relative path against each command's working folder ([clients.md](clients.md#the-kimi-tables)).
Kimi's installer puts `kimi` in `~/.kimi-code/bin`, which is on `PATH` only in a shell whose profile adds it; llmctl started from another shell prints `(kimi: not installed, no kimi on PATH)`, so run it from that shell or add the folder to `PATH` first.
llmctl never creates a config folder; inside an existing Hermes root it creates only `profiles/` and `profiles/llmctl/`, mode 0700.

| Client | File | What changes | When |
|---|---|---|---|
| OpenCode | `opencode.json` (or `opencode.jsonc`) in the OpenCode folder | Only the top-level `"model"` line, then `opencode reload` so an open OpenCode sees it; the provider config stays the one from [opencode.md](opencode.md#the-validated-config) | When the config holds the recipe's entry, and the default already names a `llamacpp-docker`, `vllm-docker` or `sglang-docker` model or you pass `--set-default` |
| Hermes | `<root>/profiles/llmctl/config.yaml` | Only the llmctl block between its two marker lines (`model`, `providers`, `model_overrides`, `gateway`, and `platform_toolsets` for a chat-only recipe); created once with a header and `_config_version: 53` after the block | Every `up` that changes the block |
| Hermes | `<root>/active_profile` | Written by `hermes profile use llmctl`, which llmctl runs | Only with `--set-default`, when the sticky profile is not `llmctl` yet |
| Pi | `models.json` in Pi's folder | Only `providers.llmctl`, replaced in place or appended | Every `up` that changes the entry |
| Pi | `settings.json` in Pi's folder | `modelThinkingLevels["llmctl/<served name>"]` (other `llmctl/` keys are removed), `defaultModel` when the default already points at llmctl, `defaultProvider` and `defaultModel` with `--set-default` | Every `up` that changes them, while llmctl holds Pi's lock folder `settings.json.lock` |
| dsh | `$DSH_HOME/cordis.patch.yml` (`~/.dsh/cordis.patch.yml` by default) | Only the llmctl block between its two marker lines: the `llm-pi-ai` row, and the `agent-default-model` and `acp` rows while llmctl holds dsh's default; created with a header when missing, and appended after your rows when the file has no block yet | Every `up` that changes it |
| Kimi | `$KIMI_CODE_HOME/config.toml` (`~/.kimi-code/config.toml` by default) | `[providers.llmctl]`, one `[models."llmctl/<served name>"]` and `default_model` by the rules below; every other line keeps its bytes, and a missing file is created with mode 0600 | Every `up` that changes them |

Before anything starts, `up` reads every client file and plans each change, and a file it cannot change safely refuses the start with the reasons below; Pi checks the candidate `models.json` with `pi --offline --list-models llmctl` in a private temporary folder, dsh checks the candidate home patch with `dsh --version` and `dsh --profile acp --dump-config` in another one ([clients.md](clients.md#checks-before-writing)), and Kimi checks the candidate `config.toml` with `kimi --version` and `kimi provider list --json` in a third ([clients.md](clients.md#kimi-checks-before-writing)).
The writes happen after the server answers, inside llmctl's lock, and never when the start failed, timed out or was interrupted.
Each write goes through a temporary file and a rename, keeps the file's mode (0600 for a new file), follows a symlink to its target, and prints `wrote <path>: <what changed>`; a file that is already right is neither written nor printed.
Each client is written on its own: when one fails after the server is ready, the server keeps running, the others are still written, `up` prints `<Client>: not updated, see the error below` in place of that client's usage lines, and exits with code 6.
When only `hermes profile use llmctl` fails, the profile already describes the running model, so the Hermes lines stay and the first one reads `Hermes: profile llmctl, not the default (hermes profile use llmctl failed, see the error below)`; the failure is in the exit 6 message.
`up --dry-run` prints the `would write`, `would back up`, `would run hermes profile use llmctl` and `would set OpenCode's default model` lines instead, ends with `dry run: nothing was written` (`dry run: nothing was written, and Pi has not checked the entry` when Pi is in use, `dry run: nothing was written, and dsh has not checked llmctl's rows` when dsh is, `dry run: nothing was written, and Kimi has not checked llmctl's entry` when Kimi is, `dry run: nothing was written, and neither <A> nor <B> has checked llmctl's entries` when two of Pi, dsh and Kimi are, and `dry run: nothing was written, and none of Pi, dsh and Kimi has checked llmctl's entries` when all three are), and runs no client command.

`LLMCTL_CLIENTS` picks the clients: unset means every installed client, an empty value means none, and otherwise it is a comma-separated list of `opencode`, `hermes`, `pi`, `dsh` and `kimi`.
llmctl neither reads nor writes a client the variable leaves out, and `status` shows it as disabled; an unknown name is a usage error, and a listed client that is not installed refuses `up` before anything starts.

### Defaults and the follow rule

`./llmctl up <model> --set-default`, or `export LLMCTL_SET_DEFAULT=1` in your shell profile, makes the model every installed client's default once its server is ready: OpenCode's `"model"` line, Hermes' sticky profile through `hermes profile use llmctl`, Pi's `defaultProvider` and `defaultModel`, the `agent-default-model` and `acp` lines in dsh's block, which make it dsh's default in every dsh profile ([clients.md](clients.md#the-default-model-and-the-picker)), and Kimi's top-level `default_model` ([clients.md](clients.md#the-kimi-default-model)).
`LLMCTL_SET_DEFAULT` accepts only `1` and `0` (or empty).
After `hermes profile use llmctl`, plain `hermes` commands, `hermes gateway` and `hermes config` included, act on the profile llmctl; `up` says so and prints `Hermes default profile: <previous> -> llmctl`.

Without the flag, a default that already points at llmctl follows the running model, so no client keeps talking to a stopped recipe or falls back to another provider: an OpenCode `"model"` that names a `llamacpp-docker`, `vllm-docker` or `sglang-docker` model is moved to the running one, Pi's `defaultModel` is moved when `defaultProvider` is `llmctl`, dsh's default lines follow when the block already holds them, Kimi's `default_model` follows when it names an `llmctl/` alias, and Hermes' profile llmctl describes the running model anyway.
A default that points anywhere else never changes without the flag.
To opt out, point OpenCode's default at a model that is not llmctl's, or leave OpenCode out with `LLMCTL_CLIENTS=hermes,pi,dsh`; the same works for Hermes and Pi, for dsh you delete the `agent-default-model` and `acp` lines from its block, and for Kimi you point its default at another model with `/model`, or leave Kimi out with `LLMCTL_CLIENTS=opencode,hermes,pi,dsh`.
A default that dsh's model picker saved in a profile patch is not followed, because llmctl never edits the files the Web UI owns; `up` prints a note when it would go stale ([clients.md](clients.md#the-default-model-and-the-picker)).
A recipe without an OpenCode entry, an estimate recipe or the chat-only vLLM DeepSeek-R1 recipe, leaves OpenCode's default as it is and prints `OpenCode: <id> has no OpenCode entry in docs/opencode.md; its default stays <model>`; that is not an error.

`down` changes no client file: the entries and defaults stay and point at a stopped server, which fails loudly in each client, rather than vanish and let Pi or Hermes fall back to another model without a word.
It stops the server first and reads the client files afterwards, so a client problem never keeps the GPU busy; it then names each client whose default points at llmctl:

```text
stopped qwen3.6-35b-a3b in container 'llamacpp'
OpenCode, Hermes, Pi, dsh and Kimi still point at qwen3.6-35b-a3b and get no answer until the next ./llmctl up
  switch back: OpenCode /models; hermes profile use default; Pi /model; dsh: delete the agent-default-model and acp lines in /home/u/.dsh/cordis.patch.yml; Kimi /model
  Kimi retries a stopped server for about two and a half minutes before a prompt fails
```

Kimi counts when its `default_model` starts with `llmctl/`, and the last line appears whenever it does ([clients.md](clients.md#kimi-sessions-a-stopped-server-and-timeouts)).

A bad `LLMCTL_CLIENTS`, an unset `HOME`, a `~user` path, a relative `DSH_HOME` or an empty or relative `KIMI_CODE_HOME` is reported after the `stopped` line, and `down` then exits with code 1.

To switch back by hand: in OpenCode pick another model with `/models` (or change the `"model"` line), in Hermes run `hermes profile use default` (or the previous profile that `up --set-default` named), in Pi pick another model with `/model` (or change `defaultProvider` and `defaultModel` in `settings.json`), in dsh delete the `agent-default-model` and `acp` lines from llmctl's block in the home patch, and in Kimi pick another model with `/model` (or change `default_model` in `config.toml`).

### Backups and open sessions

Before its first change to a file that holds anything of yours, llmctl copies it to `<file>.llmctl-backup` beside it, with the same mode, and prints `backed up <path> to <path>.llmctl-backup before the first change`: OpenCode's config, Pi's `models.json` when it holds other providers or keys, Pi's `settings.json` when it holds other keys, dsh's home patch when it holds anything besides llmctl's header and block, and Kimi's `config.toml` when it holds anything besides llmctl's tables and an llmctl `default_model` line ([clients.md](clients.md#the-kimi-tables)).
A backup is written once, never overwritten and never restored by llmctl; the Hermes profile has none, because llmctl created it and refuses one it did not create.

Hermes and Pi sessions that were open before may keep the entry they started with ([clients.md](clients.md#sessions-and-interactive-paths)), so after an `up` that changed their entry, start a new session; `up` prints a note when that applies.
A running `dsh web` uses new rows within seconds, while dsh headless, acp and sdk processes that are already running keep the old rows until they restart ([clients.md](clients.md#profiles-running-processes-and-sessions)).
An OpenCode session that already picked a model with `/models` keeps it, and new sessions use the default ([OpenCode models docs](https://opencode.ai/docs/models/)).
An open Kimi session reloads `config.toml`: when the alias changed, its next turn fails until you pick the new one with `/model` or start a new session, and when the alias stayed, it uses the new entry with its old thinking level ([clients.md](clients.md#kimi-sessions-a-stopped-server-and-timeouts)); `up` prints the matching note.

### What up prints

For a user whose home is `/home/u`, with all five clients installed and the files below, `./llmctl up qwen3.6` prints the sample that follows, line for line:

- `opencode.json` names another llmctl model, `llamacpp-docker/gemma-4-26b-a4b`, as the default, and its `.llmctl-backup` exists from an earlier change, so no backup line appears for it; `opencode reload` succeeds.
- The Hermes profile llmctl holds the block llmctl wrote for gpt-oss-20b on vLLM, and Hermes' default profile is not `llmctl`.
- Pi's `models.json` holds only llmctl's entry for gpt-oss-20b on vLLM, so it needs no backup, and `settings.json` sets `defaultProvider` to `llmctl` and `defaultModel` to `gpt-oss-20b` beside Pi's own `lastChangelogVersion`, with no backup yet.
- dsh's home patch `/home/u/.dsh/cordis.patch.yml` holds your row that disables dsh's session telemetry, then llmctl's block for gpt-oss-20b on vLLM with the default lines, and its `.llmctl-backup` exists from the first change.
- Kimi's `/home/u/.kimi-code/config.toml` holds Kimi's template comments and `telemetry = false`, then `default_model = "llmctl/gpt-oss-20b"`, then llmctl's tables for gpt-oss-20b on vLLM, and its `.llmctl-backup` exists from the first change.

```text
qwen3.6-35b-a3b on llamacpp: shipped recipe, measured
clients: opencode, hermes, pi, dsh, kimi
waiting for qwen3.6-35b-a3b to load (up to 300 s)
ready: http://127.0.0.1:8080/v1, served as 'local', context 131072, output 16384
backed up /home/u/.pi/agent/settings.json to /home/u/.pi/agent/settings.json.llmctl-backup before the first change
wrote /home/u/.config/opencode/opencode.json: model llamacpp-docker/qwen3.6-35b-a3b (was llamacpp-docker/gemma-4-26b-a4b); OpenCode reloaded its configuration
wrote /home/u/.hermes/profiles/llmctl/config.yaml: llmctl block for qwen3.6-35b-a3b on llama.cpp (was gpt-oss-20b on vLLM)
wrote /home/u/.pi/agent/models.json: provider llmctl for qwen3.6-35b-a3b on llama.cpp, model local (was gpt-oss-20b on vLLM)
wrote /home/u/.pi/agent/settings.json: modelThinkingLevels llmctl/local off (was absent); defaultModel local (was gpt-oss-20b)
wrote /home/u/.dsh/cordis.patch.yml: llmctl rows for qwen3.6-35b-a3b on llama.cpp (was gpt-oss-20b on vLLM); dsh's default llmctl/local in every profile (was llmctl/gpt-oss-20b)
wrote /home/u/.kimi-code/config.toml: provider llmctl and model llmctl/local for qwen3.6-35b-a3b on llama.cpp (was gpt-oss-20b on vLLM); default_model llmctl/local (was llmctl/gpt-oss-20b)
OpenCode: llamacpp-docker/qwen3.6-35b-a3b, the default
  in the OpenCode screen: type /models (or press ctrl+x m) and pick it
  for a single prompt: opencode run --model llamacpp-docker/qwen3.6-35b-a3b "your prompt"
  thinking: off; add #thinking to the model to turn it on
Hermes: profile llmctl, not the default (./llmctl up qwen3.6-35b-a3b --set-default makes it one)
  hermes -p llmctl
  thinking: off, as the server sets it
Pi: llmctl/local, the default
  pi --model llmctl/local
  thinking: off; --thinking medium turns it on
dsh: llmctl/local, the default in every dsh profile
  dsh web, or for a single prompt: dsh --profile headless "your prompt"
  thinking: off; choose Medium under Effort in dsh's model picker to turn it on
Kimi: llmctl/local, the default
  kimi, or for a single prompt: kimi -p "your prompt"
  thinking: off, as the server sets it
note: Hermes and Pi sessions that were open before keep the model they started with; start a new session
note: dsh web uses the new rows within seconds, and its open sessions on llmctl/gpt-oss-20b fail until you pick llmctl/local; dsh headless, acp and sdk processes that are already running keep the old rows until they restart
note: an open Kimi session on llmctl/gpt-oss-20b fails its next turn until you pick llmctl/local with /model or start a new session
evidence: 122.5 tok/s decode with 128k context, 15,190 MiB GPU peak, OpenCode session passed (2026-11-16, research/2026-11-16/llamacpp-qwen3.6-35b-a3b/)
stop with ./llmctl down
```

A client that is not used gets a note on the `clients:` line instead, such as `(pi: not installed, no pi on PATH)`, `(dsh: not installed, no /home/u/.dsh/profiles)`, `(kimi: not installed, no /home/u/.kimi-code)` or `(hermes: left out by LLMCTL_CLIENTS)`.
When the Hermes profile is first created, its line reads `created the Hermes profile llmctl for <id> on <runtime> (standalone, not served by Hermes' gateway)`, and when the same recipe's block is rewritten, `(same recipe, block rewritten)`.
dsh's line reads `created it with the llmctl rows for <id> on <runtime>` for a new home patch and `added the llmctl rows for <id> on <runtime> after your rows` the first time it appends to yours, each followed by the default part (`dsh's default llmctl/<served name> in every profile (was not held by llmctl)` or `dsh's default stays with its model picker`); that first time, `up` also notes that dsh's Models page cannot add OpenAI-compatible providers while llmctl's row is there ([clients.md](clients.md#the-home-patch)).
`up` also warns when Hermes' gateway serves the profile llmctl, when the profile sets `agent.reasoning_effort` outside the block, when `HERMES_HOME` names a profile in this shell, when Pi's `enabledModels` would start a new session without `--model` on another model ([clients.md](clients.md#defaults-sessions-and-timeouts) has the matching rule), when Pi reports problems in `models.json` outside llmctl's entry, when your OpenCode config has no entry for the recipe, when your installed OpenCode entry differs from the recipe ([clients.md](clients.md#hermes-agent) explains the Hermes ones), and, right after the `clients:` line, when the installed dsh is not 0.2.2-rc.3 ([clients.md](clients.md#maintenance)).
For dsh it adds a line for a chat-only recipe, a compaction line for a recipe whose context leaves no room for dsh's compaction headroom ([clients.md](clients.md#context-compaction-and-timeouts)), a note when llmctl takes dsh's default, and a note when a profile patch sets a picker default that may go stale.

Kimi's `wrote` line reads `created it with provider llmctl and model llmctl/<served name> for <id> on <runtime>` for a new `config.toml`, followed by `; default_model llmctl/<served name>` when it sets one; on a later `up` the part in parentheses reads `(was <id> on <runtime>)`, `(was absent)` when the file had no llmctl model, `(was an unknown model)` when the old one had no readable `display_name`, `(same recipe, tables rewritten)` when a key was added or changed inside llmctl's tables, or `(same recipe)` when only the default changed, and `; removed the model <alias>` names each other `llmctl/` alias it removed.
When the model is not Kimi's default, Kimi's lines read `Kimi: llmctl/<served name>, not the default (./llmctl up <id> --set-default makes it one)`, then `kimi -m llmctl/<served name>, or pick <id> on <runtime> (llmctl) with /model in kimi`, then `kimi -p needs Kimi's default even with -m, so for a single prompt run ./llmctl up <id> --set-default first`, because print mode resolves the default even with `-m` ([clients.md](clients.md#the-kimi-default-model)).
For a chat-only recipe Kimi's section holds only the `Kimi:` line, the thinking line and a `chat only:` line with `kimi -p "your prompt" --agent-file <path>`, the agent file llmctl ships, and asks for `--set-default` first when the model is not Kimi's default; it leaves out the `kimi` and `kimi -m` lines, because Kimi's TUI always sends its tools ([clients.md](clients.md#kimi-chat-only-recipes)).
After Kimi's usage lines, `up` prints a warning or a note when your `[thinking]` table in `config.toml` changes what Kimi sends to the model ([clients.md](clients.md#kimi-thinking-per-type)), a warning when `default_model` names a model the file does not define, and a note when `KIMI_MODEL_NAME` is set in llmctl's environment, which outranks `default_model` ([clients.md](clients.md#the-kimi-default-model)); right after the `clients:` line it warns when the installed Kimi is not 2.5.3 ([clients.md](clients.md#maintenance)).

An SGLang recipe adds two lines to that sample: before the `waiting` line the model overlay that llmctl builds from the snapshot ([sglang/README.md](../sglang/README.md#sampling-and-the-model-overlay)), and before the `ready` line the post-ready check of the KV pool and the free GPU memory ([sglang/README.md](../sglang/README.md#readiness-and-the-memory-check)).
For `./llmctl up qwen3.5 --runtime sglang`, with the pool and the free memory that measurement D read for that recipe ([record](../research/2026-11-18/sglang-qwen3.5-9b/README.md)):

```text
qwen3.5-9b on sglang: shipped recipe, measured
clients: opencode, hermes, pi, dsh, kimi
wrote /home/u/.cache/llmctl/sglang/overlays/qwen3.5-9b: model overlay with 11 links into snapshot a4df85a20fb7 of RedHatAI/Qwen3.5-9B-NVFP4, and generation_config.json with temperature 0.6, top_p 0.95, top_k 20
waiting for qwen3.5-9b to load (up to 900 s)
memory: KV pool 140974 tokens for context 131072; 937 MiB of GPU memory free after the start, at least 768 MiB needed
ready: http://127.0.0.1:30000/v1, served as 'qwen3.5-9b', context 131072, output 16384
```

The client lines follow with `on SGLang` and port 30000.
A check that refuses prints the `memory:` line when it read one, then `error: <refusal>; llmctl stopped the container and kept it for its logs: ./llmctl logs, ./llmctl down --remove`, exits with code 4 and writes no client file; the refusals are in [When up refuses](#when-up-refuses).
`up --dry-run` prints `would write <folder>: model overlay with links into snapshot ...` instead of the `wrote` line and builds nothing.

A vLLM recipe with `[model] tokenizer_class` adds one line before the `waiting` line, `wrote <folder>: tokenizer overlay with <n> links into snapshot <revision> of <repo>, and tokenizer_config.json with tokenizer_class <class>`, such as `wrote /home/u/.cache/llmctl/vllm/tokenizers/deepseek-r1-0528-qwen3-8b: tokenizer overlay with 9 links into snapshot 6e8885a6ff5c of deepseek-ai/DeepSeek-R1-0528-Qwen3-8B, and tokenizer_config.json with tokenizer_class PreTrainedTokenizerFast` for `./llmctl up deepseek-r1 --runtime vllm`, whose snapshot gives the nine links of measurement C's folder ([record](../research/2026-11-18/vllm-deepseek-r1-0528-qwen3-8b-cold-start/README.md)).
`up --dry-run` prints `would write <folder>: tokenizer overlay with links into snapshot ...` instead and builds nothing; vLLM has no post-ready check, so no `memory:` line follows.

`status` and `list` end with the client lines (`list` without the paths), and warn when a default points at llmctl while no model runs, or when OpenCode's llmctl default is not the running model:

```text
clients:
  OpenCode: default llamacpp-docker/qwen3.6-35b-a3b, the running model (/home/u/.config/opencode/opencode.json)
  Hermes: profile llmctl for qwen3.6-35b-a3b on llama.cpp, the running model; default profile: llmctl (/home/u/.hermes/profiles/llmctl/config.yaml)
  Pi: llmctl/local for qwen3.6-35b-a3b on llama.cpp, the running model; default llmctl/local, thinking off (/home/u/.pi/agent)
  dsh: llmctl/local for qwen3.6-35b-a3b on llama.cpp, the running model; default llmctl/local, held by llmctl (/home/u/.dsh/cordis.patch.yml)
  Kimi: llmctl/local for qwen3.6-35b-a3b on llama.cpp, the running model; default llmctl/local (/home/u/.kimi-code/config.toml)
```

Other states read `profile llmctl not created yet (the next ./llmctl up creates it)`, `for gpt-oss-20b on vLLM, not the running model (run ./llmctl up again)`, `cannot read <path>: <reason>`, `disabled by LLMCTL_CLIENTS` and `not installed (<reason>)`; dsh's line also reads `no llmctl rows yet (the next ./llmctl up writes them)`, or `default left to dsh's model picker` in place of the whole `default <provider>/<model>, held by llmctl` part, and `status` reads no dsh command.
Kimi's line reads `no llmctl entry yet (the next ./llmctl up writes it)` when either table is missing, and `default <value>` or `default not set` for its default; after the client lines come the `[thinking]` line while the running model's entry is the file's, the missing-model warning, the `KIMI_MODEL_NAME` note, and `warning: Kimi's default llmctl/<x> points at llmctl, but no model is running` when no model runs; `status` runs no `kimi` command.

## What llmctl enforces

These rules live in the code, and no recipe, runtime file or override can change them:

- Ports are published on 127.0.0.1 only, because Docker-published ports bypass ufw ([host-setup.md](host-setup.md)).
- `--memory` and `--memory-swap` are equal, from the recipe's `memory_gib`, so a container cannot swap.
- No restart policy, `--rm`, privileged mode or host network; the GPU is requested only as `--device nvidia.com/gpu=all`.
- One model server runs at a time; `up` refuses while another runs, and `up --replace` stops it first.
- llmctl never stops or removes a container it did not start, and never downloads during `up`.
- Every llama.cpp container gets `GGML_CUDA_DISABLE_GRAPHS=1`, the workaround for the CUDA-graph hang on RTX 50 laptop GPUs ([llama.cpp #27330](https://github.com/ggml-org/llama.cpp/issues/27330)); a runtime file that tries to set it is refused.
- A Hugging Face token comes from `HF_TOKEN` or from `--hf-env-file` (a file with an `HF_TOKEN=` line), and never appears in a command line, an output line, a label or `--dry-run`.
  It is sent only to huggingface.co: when a download redirects to another host, such as the storage CDN, the token is dropped.
- A runtime file's `[env]` may not set a variable whose name ends in `_TOKEN`, `_KEY`, `_SECRET` or `_PASSWORD`, in any case, because `[env]` goes into the `docker run` command line; such a runtime file is refused with exit 2.
- Every SGLang container mounts the Hugging Face cache and the recipe's model overlay read-only, and an SGLang start that fails the post-ready check of its KV pool and free GPU memory is stopped and refused with exit 4 ([When up refuses](#when-up-refuses)).
- `pull` checks every llama.cpp file against the SHA-256 that Hugging Face publishes for it, and keeps a partial file when the network fails, so running `pull` again resumes it.

The containers carry labels that say what started them: `io.llmctl.managed`, `runtime`, `recipe`, `source`, `sha256` (of the recipe file as read), `evidence`, `runtime-source` and `runtime-sha256`.
`status` reads them back, so a running server always says which recipe file, and which version of it, started it.

## Recipes

Recipes live in [recipes/llamacpp/](../recipes/llamacpp/), [recipes/vllm/](../recipes/vllm/) and [recipes/sglang/](../recipes/sglang/), one TOML file per model, plus a `_runtime.toml` per runtime with what every model of that runtime shares.
The three runtimes never read each other's files, so a change to one runtime's file cannot change another runtime's command; the tests check this.

A llama.cpp recipe has these keys:

| Key | Meaning and bounds |
|---|---|
| `schema` | Always 1 |
| `id` | The file name without `.toml`, and the OpenCode model key |
| `name`, `aliases`, `doc` | A readable name; short names (lower-case letters, digits, `.`, `_`, `-`), unique within the runtime; the document section that shows the launch |
| `opencode`, `chat_only` | Whether [opencode.md](opencode.md) has an entry for it (false by default); whether the model has no tool calling (false by default) |
| `[model] repo`, `revision`, `files`, `size_bytes` | The Hugging Face repository, the 40-character commit it was measured with, the GGUF files (a split GGUF lists every part, and the first goes to `-m`), and their total size |
| `[container] memory_gib` | The container's memory cap, 4 to 56 GiB |
| `[server] context`, `args` | The context length (`-c`), 1024 to 1048576, and the other `llama-server` flags in order |
| `[clients] output` | The output cap every client gets (OpenCode `limit.output`, Hermes `extra_body.max_tokens`, Pi `maxTokens`), 1024 to half the context |
| `[clients] thinking` | How the model thinks and how a client chooses it: `toggle`, `effort`, `always` or `none` |
| `[clients] default` | The server's default for a toggle model, `off` or `on`; required for `toggle` and refused for the other types |
| `[clients] efforts` | The `reasoning_effort` levels a client may choose, a subset of `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max` in that order that contains `medium`, the server's default; required for `effort` and refused for the other types |
| `[clients] medium_kwargs` | For `toggle` only and optional: extra `chat_template_kwargs` for a middle level (OpenCode variant `medium`, Pi level `medium`), as an inline table of strings, booleans, integers or finite numbers without `enable_thinking`, such as Nemotron 3.5 Bolt's `{ medium_effort = true }` |
| `[needs] gpu_mib`, `ram_gib` | The GPU memory the server used at its measured peak, and the container's measured RAM peak (optional) |
| `[evidence] kind`, `summary` | `measured` or `estimate`, and one sentence on what was measured |
| `[evidence] date`, `hardware`, `record` | For a measured recipe: when, on what, and the record folder under `research/` |
| `[evidence] decode_tok_s` | For a measured recipe: the decode speed it measured, which `list` shows; the summary must state the same number, and an estimate may not state one |
| `[evidence] doc` | For an estimate: the document it comes from |

A vLLM recipe differs in three places: `[model]` has `include` and `exclude` patterns and a `served_name` instead of `files`; `args` are the flags after the model; and `[needs]` has only `ram_gib`, because the GPU need is the `--gpu-memory-utilization` share of the memory CUDA can use, which is vLLM's own start check.
A vLLM recipe without that flag uses `default_gpu_memory_utilization` from `recipes/vllm/_runtime.toml`, 0.92 for vLLM v0.33.0.
A vLLM recipe may also set `[model] tokenizer_class`, a Python class name such as `PreTrainedTokenizerFast`: `up` then builds the tokenizer overlay `<cache>/llmctl/vllm/tokenizers/<id>/`, a folder of links into the snapshot plus a real `tokenizer_config.json` that is the snapshot's own with `tokenizer_class` replaced, mounts it read-only at `/llmctl/tokenizer` and passes `--tokenizer /llmctl/tokenizer` right after `--max-model-len`.
With the key set, `args` may not hold `--tokenizer`, because vLLM would take the last of the two; without it the recipe gets no overlay and `--tokenizer` stays allowed.
The DeepSeek-R1 recipe sets it, because transformers 5.20.2 in the vLLM v0.33.0 image builds the model's `LlamaTokenizerFast` config as a Metaspace `LlamaTokenizer` and garbles every reply, while `PreTrainedTokenizerFast` reads the model's byte-level `tokenizer.json` as llama.cpp does ([vllm/models.md](../vllm/models.md#deepseek-r1-0528-qwen3-8b-chat-only)).
A snapshot whose `tokenizer_config.json` is missing or is not a JSON object is refused by `up` with exit 3 before anything stops, and `pull` fails on it with exit 5 ([When up refuses](#when-up-refuses)).

An SGLang recipe has the vLLM recipe's shape, with `args` the flags after llmctl's own, plus these keys:

| Key | Meaning and bounds |
|---|---|
| `[server] sampling` | Optional: an inline table of `temperature` (0 to 2), `top_p` (above 0, at most 1), `top_k` (-1 or 1 to 1,000,000), `min_p` (0 to 1) and `repetition_penalty` (above 0, at most 2), which llmctl writes into the model overlay's `generation_config.json`, because SGLang reads its sampling defaults from that file ([sglang/README.md](../sglang/README.md#sampling-and-the-model-overlay)) |
| `[needs] gpu_mib` | Required: the server's measured GPU peak minus the desktop's share at the time, which preflight compares with the free GPU memory, because SGLang sizes itself from whatever is free |
| `[needs] ram_gib` | Required: the container's measured peak of non-reclaimable memory (anon plus shmem), which for a Paged Experts recipe includes the pinned expert store; refused when larger than `memory_gib` |

Its `args` must hold `--mem-fraction-static` with a number above 0 and at most 0.90 and `--max-running-requests 1`, and `--disable-prefill-cuda-graph` unless `[evidence] kind` is `measured` ([sglang/README.md](../sglang/README.md#memory-budget)).
They may not hold `--disable-cuda-graph`, which SGLang v0.5.23 deprecates and which also turns the decode graph off, `--sampling-defaults` with any value but `model`, or `--preferred-sampling-params`, which does not reach chat requests in v0.5.23.
`--enable-paged-experts` needs `--paged-experts-num-resident` with a positive integer and `--moe-runner-backend marlin`, and `--paged-experts-num-resident` needs `--enable-paged-experts` ([sglang/README.md](../sglang/README.md#paged-experts)).
Every flag these rules read, and every owned or secret flag below, counts in its `--flag=value` form too, must be written in full and only once, because SGLang expands an abbreviated flag and keeps the last of two values; an abbreviation or a second copy is refused with exit 2.
Some rules depend on the model's architecture, which a recipe does not name, so llmctl cannot check them and the post-ready check of the KV pool catches what they miss: `--max-mamba-cache-size` for Gated DeltaNet models (8 for Qwen3.5-9B), `--swa-full-tokens-ratio` for sliding-window models (0.1 for gpt-oss-20b), FP8 KV written as `fp8_e4m3`, and BF16 KV where the pool holds the context ([sglang/README.md](../sglang/README.md#memory-budget)).

All three runtimes take the same `[clients]` table, and each checks it against its own server args, so the entries llmctl writes for the clients describe what the server does:

- llama.cpp: `default = "off"` exactly when the args hold `--reasoning off` or a `--chat-template-kwargs` that sets `enable_thinking` to false; `effort` needs `--reasoning-effort` absent or `medium`; `always` needs `--reasoning-effort high`, `reasoning_effort` `high` in `--chat-template-kwargs`, or `--reasoning-format deepseek`; `none` allows none of `--reasoning`, `--reasoning-effort`, `--reasoning-format` and `--chat-template-kwargs`.
- vLLM: `--reasoning-parser` is present exactly when thinking is not `none`; a toggle whose `--default-chat-template-kwargs` sets `enable_thinking` must match `default`; a recipe with tools needs `--enable-auto-tool-choice` and `--tool-call-parser`, and a chat-only recipe may set neither.
- SGLang: as vLLM, without `--enable-auto-tool-choice`, which SGLang lacks: `--reasoning-parser` is present exactly when thinking is not `none`, a toggle's `--default-chat-template-kwargs` must match `default`, a recipe with tools needs `--tool-call-parser`, a chat-only recipe may not set it, and both parsers need an explicit name, not `auto` ([sglang/README.md](../sglang/README.md#tool-calling-and-reasoning)).
- All: the served name (the llama.cpp runtime file's `served_name`, a vLLM or SGLang recipe's `served_name`) uses only letters, digits, `.`, `_` and `-`, and starts with a letter or a digit.

llmctl sets some flags itself, and a recipe that contains them is refused:

| Runtime | Flags llmctl owns | Flags refused because they would put a secret on the command line |
|---|---|---|
| llama.cpp | `-m`, `--model`, `-hf` and every `--hf-*` form, `-c`, `--ctx-size`, `-a`, `--alias`, `--host`, `--port` | `--api-key`, `--hf-token`, `-hft` |
| vLLM | the model argument, `--model`, `--revision`, `--served-model-name`, `--max-model-len`, `--host`, `--port`, and `--tokenizer` when `[model] tokenizer_class` is set | `--api-key`, `--hf-token` |
| SGLang | `--model-path`, `--model`, `--served-model-name`, `--context-length`, `--host`, `--port`, `--revision`, and any abbreviation of them | `--api-key`, `--admin-api-key`, `--config` (its YAML file could carry a key) |

A runtime file has `image`, `container_name`, `host_port`, `ready_path`, `start_timeout_s`, `opencode_provider`, and an `[env]` table; the llama.cpp file adds `served_name` (`local`), the vLLM file adds named `volumes` and `default_gpu_memory_utilization`, and the SGLang file adds named `volumes` and `min_free_at_ready_mib`.
`min_free_at_ready_mib`, 768 in `recipes/sglang/_runtime.toml`, is the free GPU memory, read with `nvidia-smi` after a `POST /flush_cache` of the idle server, that must remain once the server is ready, because long prompts and the first grammar request take more after ready; it comes from the margins measured with every shipped SGLang recipe ([sglang/benchmarking.md](../sglang/benchmarking.md#post-ready-margins)), and 0 switches the check off.
The SGLang file's `[env]` holds eight variables: `HF_HUB_OFFLINE=1` and `FLASHINFER_NO_DOWNLOAD=1`, so a start fetches nothing from the network; `HF_MODULES_CACHE=/tmp/hf_modules`, a writable folder for the code a `--trust-remote-code` model brings; `MAX_JOBS=2`; `SYNC_TOKEN_IDS_ACROSS_TP=1` and `NCCL_MAX_P2P_NCHANNELS=1`, which move the NCCL memory that SGLang v0.5.23 allocates on the first grammar-constrained request into the server's own warm-up and shrink it ([sglang/README.md](../sglang/README.md#grammar-requests-and-nccl)); `TIKTOKEN_ENCODINGS_BASE`, gpt-oss's Harmony vocabulary from the image; and `PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1`, which cut gpt-oss-20b's pinned expert store from 19.19 to 10.37 GiB (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17; [record](../research/2026-11-17/sglang-gpt-oss-20b/README.md)) by keeping PyTorch from rounding each pinned host allocation up to a power of two (inferred from the image's PyTorch source).
`min_free_at_ready_mib` holds only while `[env]` sets both NCCL variables: a runtime override without them gets no refusal or warning, so drop them only with an image that contains SGLang #48487, where the override may set `min_free_at_ready_mib = 0`, or measure your own margins ([sglang/README.md](../sglang/README.md#grammar-requests-and-nccl)).
The llama.cpp runtime file may not set `GGML_CUDA_DISABLE_GRAPHS`, and the vLLM file sets `HF_MODULES_CACHE` to a writable path inside the container, because the model cache is mounted read-only there and a `--trust-remote-code` model copies its code into that cache.
Unknown keys are errors with a "did you mean" hint, because a misspelled key that was ignored would run a configuration you did not write.

## Overrides for another machine

Put your own files in `~/.config/llmctl/recipes/llamacpp/`, `~/.config/llmctl/recipes/vllm/` and `~/.config/llmctl/recipes/sglang/` (`$XDG_CONFIG_HOME/llmctl/recipes` when that is set, or the directory in `LLMCTL_RECIPES_DIR`).

- A file with a shipped recipe's name replaces that recipe whole; nothing is merged across the two files, because half of one measurement beside half of another is a configuration nobody has run.
- A file with a new name adds a model.
- A `_runtime.toml` replaces that runtime's settings whole, for another image tag, port or cache volume; the rules above still hold, and when you change the port, change the OpenCode provider's `baseURL` to match.
- `mkdir -p ~/.config/llmctl/recipes/llamacpp && ./llmctl show qwen3.6 --as-override > ~/.config/llmctl/recipes/llamacpp/qwen3.6-35b-a3b.toml` writes a complete copy to start from, marked `kind = "estimate"` until you measure it; [recipes/examples/](../recipes/examples/llamacpp/qwen3.6-35b-a3b.toml) has one.
  For a shipped model it copies the shipped recipe and does not read the override folder for that model, so the redirect works although the shell creates the empty target file before llmctl runs.
  For a model that more than one runtime ships, `--runtime vllm`, `--runtime sglang` or `--runtime llamacpp` picks which recipe is copied; without it the copy follows the same runtime order as `show`, llama.cpp first, then vLLM, then SGLang, and the copy's first line names the path it belongs at.
  Redirecting into an override that already exists loses what that override held, because the shell empties the file before llmctl runs: for a shipped model the file then holds the shipped recipe, and for a model that exists only in the override folder it stays empty.
  To keep your own override, copy that file instead.
  A model that exists only in the override folder has no shipped recipe to copy, so `--as-override` refuses it and names its file.
- An override that does not validate is listed by `check` and `list`, and `up` refuses that model until it is fixed or removed, instead of quietly running the shipped recipe.

On a GPU with less memory, lower `--moe-cache-mib` first and then raise `--n-cpu-moe`, as [llamacpp/README.md](../llamacpp/README.md#moe-expert-offload) explains; for SGLang, lower `--mem-fraction-static` or the context, or the resident experts of a Paged Experts recipe ([sglang/README.md](../sglang/README.md#memory-budget)); on a machine with less RAM, lower `--cache-ram` or `memory_gib`.

## When up refuses

`up` checks, in order: the clients, Docker, the CDI spec, the image, the model files, other containers and the port, free GPU memory, and available RAM; for SGLang it then builds the model overlay, and after the server is ready it checks the KV pool and the free GPU memory; for a vLLM recipe with `[model] tokenizer_class` it builds the tokenizer overlay before the start.
Each refusal names what it found and exits with the code in its row: 1 for a usage error, 3 for a check that refused before anything started, 4 for an SGLang server that started but failed its post-ready check, which llmctl then stops and keeps for its logs, and 6 when the model is running but a client file or command failed.
A client problem that appears only after the server is ready, such as a file that changed while the model loaded, is reported inside the exit 6 message instead of refusing.
`status` and `list` stop with the same `LLMCTL_CLIENTS`, `HOME`, `~user`, relative `DSH_HOME` and empty or relative `KIMI_CODE_HOME` usage errors; `down` stops the server first and reports them afterwards, with exit code 1.

| Message starts with | Exit | What to do |
|---|---|---|
| `Docker is not answering` | 3 | Start Docker: `sudo systemctl start docker` |
| `Docker refused the connection (permission denied` | 3 | Add yourself to the `docker` group and log in again |
| `no CDI spec for nvidia.com/gpu` | 3 | Install the NVIDIA Container Toolkit and generate the spec ([host-setup.md](host-setup.md)) |
| `the image ... is not on this machine` | 3 | Run the `./llmctl pull` command the message names (with `--runtime` for a vLLM or SGLang recipe) |
| `the model files are not ready` | 3 | Run the `./llmctl pull` command the message names (with `--runtime` for a vLLM or SGLang recipe); for vLLM and SGLang this also finishes a download that stopped partway |
| `generation_config.json in snapshot ... is not a JSON object` (inside `the model files are not ready` for SGLang) | 3 | The snapshot's sampling file is damaged, so llmctl cannot add the recipe's sampling to the model overlay: delete that file and its blob from the Hugging Face cache, then pull again; `pull` checks the same file and fails with exit 5 ([sglang/README.md](../sglang/README.md#sampling-and-the-model-overlay)) |
| `llmctl's cache folder ... is not an absolute path` | 1 | Set `XDG_CACHE_HOME`, or `HOME` when it is unset, to an absolute path: llmctl builds the SGLang model overlay under `<cache>/llmctl/sglang/overlays/` and the vLLM tokenizer overlay under `<cache>/llmctl/vllm/tokenizers/`, never relative to the working folder |
| `cannot write the model overlay` | 3 | Check the free space and permissions of llmctl's cache folder, which the message names; when it names a missing snapshot instead, run `./llmctl pull <model> --runtime sglang`, and when it names the snapshot's `generation_config.json`, delete that file and its blob from the Hugging Face cache, then pull again ([sglang/README.md](../sglang/README.md#sampling-and-the-model-overlay)) |
| `tokenizer_config.json in snapshot ... is missing` (inside `the model files are not ready` for a vLLM recipe with `[model] tokenizer_class`) | 3 | The snapshot has no `tokenizer_config.json`, so the tokenizer overlay cannot set the class: remove `tokenizer_class` from the recipe, or make the download include that file, then pull again; `pull` checks the same file and fails with exit 5 |
| `tokenizer_config.json in snapshot ... is not a JSON object` (inside `the model files are not ready` for a vLLM recipe with `[model] tokenizer_class`) | 3 | The snapshot's tokenizer config is damaged: delete that file and its blob from the Hugging Face cache, then pull again; `pull` checks the same file and fails with exit 5 |
| `cannot write the tokenizer overlay` | 3 | Check the free space and permissions of llmctl's cache folder, which the message names; when it names a missing snapshot instead, run `./llmctl pull <model> --runtime vllm`, and when it names the snapshot's `tokenizer_config.json`, follow the two `tokenizer_config.json` rows above ([vllm/models.md](../vllm/models.md#deepseek-r1-0528-qwen3-8b-chat-only)) |
| `another model server is running` | 3 | `./llmctl down`, or `./llmctl up <model> --replace` |
| `a container named ... was not started by llmctl` | 3 | Check that you do not need it, then `docker rm -f <name>` |
| `port 127.0.0.1:... is already in use` | 3 | Stop the program that holds the port |
| `... needs N MiB of free GPU memory` | 3 | Close the programs it lists, or write the override the message suggests: a smaller `--moe-cache-mib` for an offloaded MoE model, a smaller context for other llama.cpp models, a smaller `--gpu-memory-utilization` or context for vLLM, and for SGLang a smaller `--mem-fraction-static` or context, or fewer resident experts (`--paged-experts-num-resident`) for a Paged Experts recipe, with the `gpu_mib` you measure with it |
| `... may use up to N GiB of RAM` | 3 | Close programs, or write an override with a smaller `memory_gib` or `--cache-ram` |
| `nvidia-smi was not found` or `nvidia-smi failed` | 3, or 4 in SGLang's check after the start | Install or load the NVIDIA driver; after an SGLang start llmctl stopped the container, so run `up` again once `nvidia-smi` answers |
| `the KV pool holds ... tokens, fewer than the context` | 4 | The SGLang server is stopped and kept for its logs; write an override with a smaller context, FP8 KV (`--kv-cache-dtype fp8_e4m3`) or fewer Mamba slots ([sglang/README.md](../sglang/README.md#memory-budget)), then run `up` again |
| `only ... MiB of GPU memory is free after the start` | 4 | Another program may have taken GPU memory while the model loaded: close other GPU programs and run the same `up` again, or write an override with a `--mem-fraction-static` 0.02 lower, which frees about 300 MiB ([sglang/README.md](../sglang/README.md#memory-budget)) |
| `could not read internal_states[0].memory_usage.token_capacity` | 4 | llmctl could not read the KV pool's size from the SGLang server, for the reason the message names; read `./llmctl logs`, then run `up` again, or `./llmctl down --remove` ([sglang/README.md](../sglang/README.md#readiness-and-the-memory-check)) |
| `POST http://127.0.0.1:30000/flush_cache answered` or `... failed` | 4 | llmctl could not empty SGLang's cache before it read the free memory; read `./llmctl logs`, then run `up` again ([sglang/README.md](../sglang/README.md#readiness-and-the-memory-check)) |
| `another llmctl command is running` | 3 | Wait for it to finish |
| `LLMCTL_CLIENTS: unknown client` | 1 | Use only `opencode`, `hermes`, `pi`, `dsh` and `kimi`, comma-separated, or an empty value for none |
| `LLMCTL_SET_DEFAULT must be 1 or 0` | 1 | Set it to `1` or `0`, or unset it |
| `HOME is not set` | 1 | Run llmctl with `HOME` set; it finds every client config from it |
| `...: llmctl expands only ~ and ~/, not ~user` | 1 | Write `HERMES_HOME`, `PI_CODING_AGENT_DIR` or `XDG_CONFIG_HOME` as a full path, `~` or `~/...` |
| `DSH_HOME is relative` | 1 | Set `DSH_HOME` to an absolute path, `~` or `~/...`, or unset it for `~/.dsh`; dsh itself would resolve a relative value against each command's working folder |
| `KIMI_CODE_HOME is set but empty` | 1 | Unset `KIMI_CODE_HOME` for `~/.kimi-code`, or set it to an absolute path; Kimi 2.5.3 reads an empty value two different ways ([clients.md](clients.md#the-kimi-tables)) |
| `KIMI_CODE_HOME is relative` | 1 | Set `KIMI_CODE_HOME` to an absolute path, or unset it for `~/.kimi-code`; Kimi expands neither `~` nor `$NAME` and would resolve the value against each command's working folder |
| `LLMCTL_CLIENTS names ..., but ... is not installed here` | 3 | Install the client, or drop it from `LLMCTL_CLIENTS`; installed means its binary is on `PATH` and its config folder exists, for dsh that `dsh` is on `PATH` and `$DSH_HOME/profiles` exists, which the first run of any dsh profile creates, and for Kimi that `kimi` is on `PATH` (`~/.kimi-code/bin` is on it only in a shell that adds it) and the Kimi home folder exists |
| `cannot read ...` | 3 | Fix the file's permissions or encoding (UTF-8), or leave that client out with `LLMCTL_CLIENTS` |
| `... is not plain JSON` | 3 | Remove the comments and trailing commas, or leave that client out with `LLMCTL_CLIENTS`; with `--set-default` this also applies to OpenCode's config |
| `... is not a JSON object` | 3 | The file must hold one JSON object; fix it, or leave that client out with `LLMCTL_CLIENTS` |
| `...: "providers" is not a JSON object` or `...: "modelThinkingLevels" is not a JSON object` | 3 | Fix that key in Pi's `models.json` or `settings.json`, or leave Pi out with `LLMCTL_CLIENTS` |
| `... holds an escaped lone surrogate` | 3 | Remove the `\ud800` to `\udfff` escape, which cannot be written back as UTF-8, or leave that client out |
| `could not change the top-level "model" in ... safely` | 3 | Set OpenCode's `"model"` to the value the message names yourself |
| `Pi rejects the llmctl entry for ...` | 3 | Read Pi's lines in the message: they name the problem in `models.json`, which may be outside llmctl's entry, because Pi rejects a file as a whole |
| `pi --list-models failed` | 3 | Check that `pi` runs in this shell, or leave Pi out with `LLMCTL_CLIENTS` |
| `cannot check ... with pi` | 3 | llmctl could not make its private temporary folder for Pi's check or run `pi` there; check `TMPDIR`, free space, and that `pi` runs in this shell |
| `... has no llmctl block, so the Hermes profile llmctl was not created by llmctl` | 3 | Rename your profile with `hermes profile rename llmctl <name>`, or leave Hermes out with `LLMCTL_CLIENTS` |
| `... holds a Hermes profile that llmctl did not create` | 3 | The same: rename that profile, or leave Hermes out |
| `...: the llmctl block markers are damaged` | 3 | Restore the two marker lines exactly, in the Hermes profile ([clients.md](clients.md#the-profile)) or in dsh's home patch ([clients.md](clients.md#the-home-patch)) |
| `... line N: '...' is set outside the llmctl block` | 3 | Delete that setting: llmctl owns it in this profile, and Hermes' loader rejects a duplicate key ([clients.md](clients.md#the-profile)) |
| `... line N: llmctl cannot check this top-level line` | 3 | In the Hermes profile, write top-level keys as plain `key:`; in dsh's home patch, write each row as a `- ` list item at column 0, and remove a lone carriage return |
| `... line N: the llmctl block holds a line llmctl did not write` | 3 | Delete that line from dsh's home patch, or move your row after the end marker; llmctl's block holds only its own rows ([clients.md](clients.md#the-home-patch)) |
| `... line N: dsh reads this file as one YAML list` | 3 | Delete the `[]` line from dsh's home patch; your rows stay valid as `- ` list items |
| `... line N: the llm-pi-ai row is set outside the llmctl block` | 3 | Delete that row from dsh's home patch: dsh applies only one of two rows with one id, and llmctl's row describes the running model ([clients.md](clients.md#the-home-patch)) |
| `... line N: the ... row is set outside the llmctl block while llmctl holds or sets dsh's default` | 3 | Delete your `agent-default-model` or `acp` row, or delete llmctl's `agent-default-model` and `acp` lines and run `up` without `--set-default` |
| `dsh's profile patches set the llm-pi-ai row` | 3 | Delete those rows from the profile patches the message names ([clients.md](clients.md#moving-a-hand-made-setup)), or leave dsh out with `LLMCTL_CLIENTS` |
| `dsh cannot send ... per level` | 3 | Leave dsh out with `LLMCTL_CLIENTS` for that recipe: its `medium_kwargs` value is not `true`, which dsh's levels cannot carry ([clients.md](clients.md#rows-per-thinking-type)) |
| `dsh --version failed` or `dsh --dump-config failed` | 3 | Check that `dsh` runs in this shell (Node 24.9 or newer), or leave dsh out with `LLMCTL_CLIENTS` |
| `dsh rejects the home patch llmctl would write for ...` | 3 | Read dsh's error line in the message: it names a line of your home patch, often one of your own rows; fix it, and nothing was written |
| `dsh ... has no ... row, so llmctl's row would not apply` | 3 | The installed dsh is not the version llmctl's rows were checked with; install `@deepseek-ai/dsh@0.2.2-rc.3`, or leave dsh out with `LLMCTL_CLIENTS` ([clients.md](clients.md#maintenance)) |
| `dsh did not apply llmctl's ... row` | 3 | dsh read the file but did not apply llmctl's row from the home patch; leave dsh out with `LLMCTL_CLIENTS` and report it with the dsh version |
| `cannot check ... with dsh` | 3 | llmctl could not make its private temporary folder for dsh's check or run `dsh` there; check `TMPDIR`, free space, and that `dsh` runs in this shell |
| `... is not valid TOML` | 3 | Fix Kimi's `config.toml` at the line and column the message names, or remove a byte-order mark at its start, or leave Kimi out with `LLMCTL_CLIENTS`; `kimi doctor config <path>` shows the same syntax problem, while whether Kimi accepts a byte-order mark was not measured |
| `...: "providers" is not a table` or `...: "models" is not a table` | 3 | Kimi needs `[providers.<name>]` and `[models."<alias>"]` tables; fix the key, such as a `[[providers]]` array, or leave Kimi out with `LLMCTL_CLIENTS` |
| `... line N: providers is written as an inline table or with dotted keys` (or `models`) | 3 | Write each provider as a `[providers.<name>]` table and each model as a `[models."<alias>"]` table, because llmctl cannot add its own table after an inline table or a dotted key |
| `...: llmctl's provider is written with dotted keys or as an inline table` (or `the model llmctl/... is written ...`) | 3 | Write it as a `[providers.llmctl]` or `[models."llmctl/<served name>"]` table, or delete it and let llmctl write it ([clients.md](clients.md#the-kimi-tables)) |
| `could not change the top-level default_model in ... safely` | 3 | Set `default_model` to the value the message names yourself, as one line with a quoted string |
| `kimi --version failed` or `kimi provider list --json failed` | 3 | Check that `kimi` runs in this shell, or leave Kimi out with `LLMCTL_CLIENTS` |
| `Kimi ... does not read llmctl's entry for ...` | 3 | On Kimi 2.5.3, run the `kimi doctor config <path>` the message names to see whether Kimi reads the rest of the file; on another version, leave Kimi out with `LLMCTL_CLIENTS` until llmctl supports it ([clients.md](clients.md#kimi-checks-before-writing)) |
| `cannot check ... with kimi` | 3 | llmctl could not make its private temporary folder for Kimi's check or run `kimi` there; check `TMPDIR`, free space, and that `kimi` runs in this shell |
| `Hermes marks the profile llmctl as deleted` | 3 | Delete the file `<root>/profiles/.deleted/llmctl` to let llmctl recreate the profile, or leave Hermes out |
| `Hermes refuses a context below 64,000 tokens` | 3 | Leave Hermes out with `LLMCTL_CLIENTS=opencode,pi` for that recipe |
| `... does not exist; llmctl creates a Hermes profile only inside an existing Hermes home` | 3 | Run Hermes once so it creates its home, or leave Hermes out |
| `...: llmctl's own check of the new profile failed`, `of the new models.json failed`, `of the new settings failed`, `of the new home patch failed` or `of the new config.toml failed` | 3 | Nothing was written; report it with the file, because llmctl refused a change that would have touched more than its own keys |
| `the model is running, but N client file(s) were not updated` | 6 | The server runs; fix each `<path>: <reason>` the message lists, then run `./llmctl up <model>` again |
| `...: Pi holds its settings lock ...` (inside the exit 6 message) | 6 | Close Pi; if no Pi is running, remove the lock folder the message names, then run `up` again; both Pi files stay as they were |
| `...: Pi took over its settings lock while llmctl held it` (inside the exit 6 message) | 6 | llmctl made its Pi changes, but another writer may have changed `settings.json` meanwhile; check it, then run `up` again |
| `putting ... back failed too` (after an interrupt; inside the exit 6 message when a write failed) | 130 or 6 | `models.json` and `settings.json` now name different llmctl models; run `./llmctl up <model>` again, or fix the two files by hand ([clients.md](clients.md#validation)) |
| `hermes profile use llmctl exited ...` (inside the exit 6 message) | 6 | Read Hermes' message, then run `hermes profile use llmctl` yourself |
| `...: the file changed while llmctl wrote it, probably through Kimi` (inside the exit 6 message) | 6 | An open Kimi saved `config.toml` while llmctl wrote it; check the file, then run `./llmctl up <model>` again ([clients.md](clients.md#the-kimi-tables)) |

Exit codes: 0 success; 1 a usage error, or a download you declined; 2 an invalid recipe or override; 3 a refused check; 4 a server that failed to start or become ready, an SGLang server that failed its post-ready check, or a failed `test`; 5 a failed download or checksum; 6 the model is running, but a client file or command failed; 130 interrupted.

## Adding or changing a recipe

1. Measure the launch with the method in [llamacpp/benchmarking.md](../llamacpp/benchmarking.md), [vllm/benchmarking.md](../vllm/benchmarking.md) or [sglang/benchmarking.md](../sglang/benchmarking.md), and keep the records under `research/`.
2. Write the launch into its section of [llamacpp/models.md](../llamacpp/models.md), [vllm/models.md](../vllm/models.md) or [sglang/models.md](../sglang/models.md), and the recipe file into `recipes/` with its `[clients]` table; set `opencode = true` only together with an entry in [opencode.md](opencode.md#the-validated-config), and otherwise put an "OpenCode entry" snippet in the model's section.
   A measured recipe always gets `opencode = true` and its entry, and a chat-only one keeps `"tools": false` in it; only an estimate goes without, and a chat-only recipe on vLLM, which refuses OpenCode's `tool_choice` `"auto"` without a tool parser, gets `opencode = false` and neither entry nor snippet ([opencode.md](opencode.md#the-validated-config)).
3. Run `python3 -m launcher.tests.regen_golden`, which also writes the Hermes and Pi entries under `launcher/tests/golden/clients/`, read the diff of `launcher/tests/golden/`, and run `python3 -m unittest discover -s launcher/tests -t .`.

The tests check that each recipe matches its document's launch, and that its `[clients]` table matches its OpenCode entry and the server args, so a recipe, its document and the OpenCode config change together or the tests fail.
They run without Docker, a GPU or the network.
