# llmctl: start a measured model by name

Last verified: 2026-11-16.

`./llmctl` starts a model from this repository's recipes with one command, applies the rules from [README.md](../README.md#rules-for-every-setup-here) for you, and points OpenCode, Hermes Agent and Pi at it ([Clients](#clients)).
It runs llama.cpp and vLLM in Docker, one model server at a time, on an NVIDIA GPU.
The recipes were measured on an MSI Raider 18 Max HX A2WI with an RTX 5080 Laptop GPU (16 GB) and 61 GiB of RAM; on another machine, read `./llmctl list` and write an override wherever yours differs.

## Quick start

```bash
./llmctl list                 # every recipe, its evidence, and what is downloaded
./llmctl pull qwen3.6         # asks, then downloads the model files, and the image if it is missing
./llmctl up qwen3.6           # checks the machine, starts the server, waits until it answers
./llmctl test                 # a reply, a tool call and a decode speed
./llmctl down                 # stops the server
```

`up` prints the OpenCode model to use, for example `llamacpp-docker/qwen3.6-35b-a3b`, which works with the provider config in [opencode.md](opencode.md#the-validated-config), and the Hermes and Pi commands, `hermes -p llmctl` and `pi --model llmctl/local`, whose entries it writes itself ([clients.md](clients.md)).
`./llmctl up qwen3.6 --set-default` also makes the model the default of every installed client.
In the OpenCode screen, pick it with `/models` (or `ctrl+x m`); for a single prompt, pass it as `opencode run --model llamacpp-docker/qwen3.6-35b-a3b "..."`, because the interactive `opencode` command has no `-m` flag ([OpenCode models docs](https://opencode.ai/docs/models/)).
It needs Linux, Docker Engine, the NVIDIA Container Toolkit with a CDI spec ([host-setup.md](host-setup.md)), and Python 3.11 or newer; it uses only Python's standard library, so there is nothing to install.
Checked on the reference laptop on 2026-11-16: `up qwen3.6` was ready in 9 s and passed `test` and an OpenCode session, and `up gpt-oss --runtime vllm` was ready in 29 s with warm compile caches and passed `test` ([records](../research/2026-11-16/llmctl-e2e/)).

## Commands

| Command | What it does |
|---|---|
| `./llmctl list [--runtime R]` | Every recipe: ID, runtime, status, aliases, evidence, measured decode speed, context, GPU need, RAM need, download size, whether the files and the image are present, and whether an override replaces it; below the table, the running server's address and the client lines of `status` without paths |
| `./llmctl show <model> [--runtime R]` | The effective recipe, its file and SHA-256, and the exact `docker run` command `up` would run |
| `./llmctl show <model> --as-override` | A complete copy of the recipe, marked as an estimate, to save in the override directory and edit |
| `./llmctl pull <model> [--runtime R] [--yes] [--hf-env-file F]` | Says how much it will download, asks, then pulls the image if it is missing and downloads the model files; without a terminal it needs `--yes` |
| `./llmctl up <model> [--runtime R] [--replace] [--dry-run] [--set-default]` | Runs the checks below, starts the container, waits until the server answers, writes the llmctl entries of the installed clients ([Clients](#clients)), and prints the endpoint, the served name, the context, the output cap, each client's model and thinking choices, and the evidence; with `--set-default` it also makes the model every installed client's default |
| `./llmctl status` | The container llmctl started: its recipe, source file and SHA-256, state, port, health, memory use and the GPU memory in use; then each client's llmctl entry and default, and whether they match the running model |
| `./llmctl logs [-f] [--tail N]` | The server's log, the last 100 lines by default |
| `./llmctl test` | Three requests against the running server: a reply, one tool call, and a decode of up to 512 tokens |
| `./llmctl down [--remove]` | Stops the server with a 60-second grace period; the stopped container stays for its logs unless `--remove`; then names the clients whose default still points at llmctl, and changes no client file |
| `./llmctl check` | Validates every shipped recipe and every override and lists each problem with its file |

A model is named by its recipe ID or an alias, such as `qwen3.6` for `qwen3.6-35b-a3b`.
When a name has a recipe for both runtimes, llama.cpp is used, because it is the main runtime here; `--runtime vllm` picks vLLM.
Running `up` for the model that is already running from the same recipe file starts nothing, writes the client entries again where they differ and prints the summary again; `--replace` restarts it.
Run `./llmctl <command> --help` for every option.
In `list`, STATUS is `running` for the server that answers, `starting` while it loads, `running (older file)` when it was started from an earlier version of its recipe file, and `?` when Docker did not answer.

## Clients

`up` points three coding clients at the model it starts: OpenCode v2.3.0, Hermes Agent v0.23.1 and Pi 1.5.2, the versions this was verified with.
It keeps one llmctl entry per client and rewrites it on every `up` to describe the running recipe, so you pick the model once per client, not once per recipe.
[clients.md](clients.md) shows the Hermes and Pi entries, how each client thinks with them, and what was measured.

A client is installed when its binary is on `PATH` and its config folder exists: OpenCode `$XDG_CONFIG_HOME/opencode/` (or `~/.config/opencode/`), Hermes its root (`~/.hermes/` unless `HERMES_HOME` points outside it, as [clients.md](clients.md#the-profile) explains), Pi `PI_CODING_AGENT_DIR` (or `~/.pi/agent/`).
llmctl never creates a config folder; inside an existing Hermes root it creates only `profiles/` and `profiles/llmctl/`, mode 0700.

| Client | File | What changes | When |
|---|---|---|---|
| OpenCode | `opencode.json` (or `opencode.jsonc`) in the OpenCode folder | Only the top-level `"model"` line, then `opencode reload` so an open OpenCode sees it; the provider config stays the one from [opencode.md](opencode.md#the-validated-config) | When the config holds the recipe's entry, and the default already names a `llamacpp-docker` or `vllm-docker` model or you pass `--set-default` |
| Hermes | `<root>/profiles/llmctl/config.yaml` | Only the llmctl block between its two marker lines (`model`, `providers`, `model_overrides`, `gateway`, and `platform_toolsets` for a chat-only recipe); created once with a header and `_config_version: 53` after the block | Every `up` that changes the block |
| Hermes | `<root>/active_profile` | Written by `hermes profile use llmctl`, which llmctl runs | Only with `--set-default`, when the sticky profile is not `llmctl` yet |
| Pi | `models.json` in Pi's folder | Only `providers.llmctl`, replaced in place or appended | Every `up` that changes the entry |
| Pi | `settings.json` in Pi's folder | `modelThinkingLevels["llmctl/<served name>"]` (other `llmctl/` keys are removed), `defaultModel` when the default already points at llmctl, `defaultProvider` and `defaultModel` with `--set-default` | Every `up` that changes them, while llmctl holds Pi's lock folder `settings.json.lock` |

Before anything starts, `up` reads every client file and plans each change, and a file it cannot change safely refuses the start with the reasons below; Pi checks the candidate `models.json` with `pi --offline --list-models llmctl` in a private temporary folder.
The writes happen after the server answers, inside llmctl's lock, and never when the start failed, timed out or was interrupted.
Each write goes through a temporary file and a rename, keeps the file's mode (0600 for a new file), follows a symlink to its target, and prints `wrote <path>: <what changed>`; a file that is already right is neither written nor printed.
Each client is written on its own: when one fails after the server is ready, the server keeps running, the others are still written, `up` prints `<Client>: not updated, see the error below` in place of that client's usage lines, and exits with code 6.
When only `hermes profile use llmctl` fails, the profile already describes the running model, so the Hermes lines stay and the first one reads `Hermes: profile llmctl, not the default (hermes profile use llmctl failed, see the error below)`; the failure is in the exit 6 message.
`up --dry-run` prints the `would write`, `would back up`, `would run hermes profile use llmctl` and `would set OpenCode's default model` lines instead, ends with `dry run: nothing was written` (`dry run: nothing was written, and Pi has not checked the entry` when Pi is in use), and runs no client command.

`LLMCTL_CLIENTS` picks the clients: unset means every installed client, an empty value means none, and otherwise it is a comma-separated list of `opencode`, `hermes` and `pi`.
llmctl neither reads nor writes a client the variable leaves out, and `status` shows it as disabled; an unknown name is a usage error, and a listed client that is not installed refuses `up` before anything starts.

### Defaults and the follow rule

`./llmctl up <model> --set-default`, or `export LLMCTL_SET_DEFAULT=1` in your shell profile, makes the model every installed client's default once its server is ready: OpenCode's `"model"` line, Hermes' sticky profile through `hermes profile use llmctl`, and Pi's `defaultProvider` and `defaultModel`.
`LLMCTL_SET_DEFAULT` accepts only `1` and `0` (or empty).
After `hermes profile use llmctl`, plain `hermes` commands, `hermes gateway` and `hermes config` included, act on the profile llmctl; `up` says so and prints `Hermes default profile: <previous> -> llmctl`.

Without the flag, a default that already points at llmctl follows the running model, so no client keeps talking to a stopped recipe or falls back to another provider: an OpenCode `"model"` that names a `llamacpp-docker` or `vllm-docker` model is moved to the running one, Pi's `defaultModel` is moved when `defaultProvider` is `llmctl`, and Hermes' profile llmctl describes the running model anyway.
A default that points anywhere else never changes without the flag.
To opt out, point OpenCode's default at a model that is not llmctl's, or leave OpenCode out with `LLMCTL_CLIENTS=hermes,pi`; the same works for Hermes and Pi.
A recipe without an OpenCode entry, such as the chat-only DeepSeek or a vLLM estimate, leaves OpenCode's default as it is and prints `OpenCode: <id> has no OpenCode entry in docs/opencode.md; its default stays <model>`; that is not an error.

`down` changes no client file: the entries and defaults stay and point at a stopped server, which fails loudly in each client, rather than vanish and let Pi or Hermes fall back to another model without a word.
It stops the server first and reads the client files afterwards, so a client problem never keeps the GPU busy; it then names each client whose default points at llmctl:

```text
stopped qwen3.6-35b-a3b in container 'llamacpp'
OpenCode, Hermes and Pi still point at qwen3.6-35b-a3b and get no answer until the next ./llmctl up
  switch back: OpenCode /models; hermes profile use default; Pi /model
```

A bad `LLMCTL_CLIENTS`, an unset `HOME` or a `~user` path is reported after the `stopped` line, and `down` then exits with code 1.

To switch back by hand: in OpenCode pick another model with `/models` (or change the `"model"` line), in Hermes run `hermes profile use default` (or the previous profile that `up --set-default` named), and in Pi pick another model with `/model` (or change `defaultProvider` and `defaultModel` in `settings.json`).

### Backups and open sessions

Before its first change to a file that holds anything of yours, llmctl copies it to `<file>.llmctl-backup` beside it, with the same mode, and prints `backed up <path> to <path>.llmctl-backup before the first change`: OpenCode's config, Pi's `models.json` when it holds other providers or keys, and Pi's `settings.json` when it holds other keys.
A backup is written once, never overwritten and never restored by llmctl; the Hermes profile has none, because llmctl created it and refuses one it did not create.

Hermes and Pi sessions that were open before may keep the entry they started with ([clients.md](clients.md#sessions-and-interactive-paths)), so after an `up` that changed their entry, start a new session; `up` prints a note when that applies.
An OpenCode session that already picked a model with `/models` keeps it, and new sessions use the default ([OpenCode models docs](https://opencode.ai/docs/models/)).

### What up prints

For a user whose home is `/home/u`, with all three clients installed and the files below, `./llmctl up qwen3.6` prints the sample that follows, line for line:

- `opencode.json` names another llmctl model, `llamacpp-docker/gemma-4-26b-a4b`, as the default, and its `.llmctl-backup` exists from an earlier change, so no backup line appears for it; `opencode reload` succeeds.
- The Hermes profile llmctl holds the block llmctl wrote for gpt-oss-20b on vLLM, and Hermes' default profile is not `llmctl`.
- Pi's `models.json` holds only llmctl's entry for gpt-oss-20b on vLLM, so it needs no backup, and `settings.json` sets `defaultProvider` to `llmctl` and `defaultModel` to `gpt-oss-20b` beside Pi's own `lastChangelogVersion`, with no backup yet.

```text
qwen3.6-35b-a3b on llamacpp: shipped recipe, measured
clients: opencode, hermes, pi
waiting for qwen3.6-35b-a3b to load (up to 300 s)
ready: http://127.0.0.1:8080/v1, served as 'local', context 131072, output 16384
backed up /home/u/.pi/agent/settings.json to /home/u/.pi/agent/settings.json.llmctl-backup before the first change
wrote /home/u/.config/opencode/opencode.json: model llamacpp-docker/qwen3.6-35b-a3b (was llamacpp-docker/gemma-4-26b-a4b); OpenCode reloaded its configuration
wrote /home/u/.hermes/profiles/llmctl/config.yaml: llmctl block for qwen3.6-35b-a3b on llama.cpp (was gpt-oss-20b on vLLM)
wrote /home/u/.pi/agent/models.json: provider llmctl for qwen3.6-35b-a3b on llama.cpp, model local (was gpt-oss-20b on vLLM)
wrote /home/u/.pi/agent/settings.json: modelThinkingLevels llmctl/local off (was absent); defaultModel local (was gpt-oss-20b)
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
note: Hermes and Pi sessions that were open before keep the model they started with; start a new session
evidence: 122.5 tok/s decode with 128k context, 15,190 MiB GPU peak, OpenCode session passed (2026-11-16, research/2026-11-16/llamacpp-qwen3.6-35b-a3b/)
stop with ./llmctl down
```

A client that is not used gets a note on the `clients:` line instead, such as `(pi: not installed, no pi on PATH)` or `(hermes: left out by LLMCTL_CLIENTS)`.
When the Hermes profile is first created, its line reads `created the Hermes profile llmctl for <id> on <runtime> (standalone, not served by Hermes' gateway)`, and when the same recipe's block is rewritten, `(same recipe, block rewritten)`.
`up` also warns when Hermes' gateway serves the profile llmctl, when the profile sets `agent.reasoning_effort` outside the block, when `HERMES_HOME` names a profile in this shell, when Pi's `enabledModels` would start a new session without `--model` on another model ([clients.md](clients.md#defaults-sessions-and-timeouts) has the matching rule), when Pi reports problems in `models.json` outside llmctl's entry, when your OpenCode config has no entry for the recipe, and when your installed OpenCode entry differs from the recipe ([clients.md](clients.md#hermes-agent) explains the Hermes ones).

`status` and `list` end with the client lines (`list` without the paths), and warn when a default points at llmctl while no model runs, or when OpenCode's llmctl default is not the running model:

```text
clients:
  OpenCode: default llamacpp-docker/qwen3.6-35b-a3b, the running model (/home/u/.config/opencode/opencode.json)
  Hermes: profile llmctl for qwen3.6-35b-a3b on llama.cpp, the running model; default profile: llmctl (/home/u/.hermes/profiles/llmctl/config.yaml)
  Pi: llmctl/local for qwen3.6-35b-a3b on llama.cpp, the running model; default llmctl/local, thinking off (/home/u/.pi/agent)
```

Other states read `profile llmctl not created yet (the next ./llmctl up creates it)`, `for gpt-oss-20b on vLLM, not the running model (run ./llmctl up again)`, `cannot read <path>: <reason>`, `disabled by LLMCTL_CLIENTS` and `not installed (<reason>)`.

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
- `pull` checks every llama.cpp file against the SHA-256 that Hugging Face publishes for it, and keeps a partial file when the network fails, so running `pull` again resumes it.

The containers carry labels that say what started them: `io.llmctl.managed`, `runtime`, `recipe`, `source`, `sha256` (of the recipe file as read), `evidence`, `runtime-source` and `runtime-sha256`.
`status` reads them back, so a running server always says which recipe file, and which version of it, started it.

## Recipes

Recipes live in [recipes/llamacpp/](../recipes/llamacpp/) and [recipes/vllm/](../recipes/vllm/), one TOML file per model, plus a `_runtime.toml` per runtime with what every model of that runtime shares.
The two runtimes never read each other's files, so a change to a llama.cpp file cannot change a vLLM command, and the reverse; the tests check this.

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

Both runtimes take the same `[clients]` table, and each checks it against its own server args, so the entries llmctl writes for the clients describe what the server does:

- llama.cpp: `default = "off"` exactly when the args hold `--reasoning off` or a `--chat-template-kwargs` that sets `enable_thinking` to false; `effort` needs `--reasoning-effort` absent or `medium`; `always` needs `--reasoning-effort high`, `reasoning_effort` `high` in `--chat-template-kwargs`, or `--reasoning-format deepseek`; `none` allows none of `--reasoning`, `--reasoning-effort`, `--reasoning-format` and `--chat-template-kwargs`.
- vLLM: `--reasoning-parser` is present exactly when thinking is not `none`; a toggle whose `--default-chat-template-kwargs` sets `enable_thinking` must match `default`; a recipe with tools needs `--enable-auto-tool-choice` and `--tool-call-parser`, and a chat-only recipe may set neither.
- Both: the served name (the llama.cpp runtime file's `served_name`, a vLLM recipe's `served_name`) uses only letters, digits, `.`, `_` and `-`, and starts with a letter or a digit.

llmctl sets some flags itself, and a recipe that contains them is refused:

| Runtime | Flags llmctl owns | Flags refused because they would put a secret on the command line |
|---|---|---|
| llama.cpp | `-m`, `--model`, `-hf` and every `--hf-*` form, `-c`, `--ctx-size`, `-a`, `--alias`, `--host`, `--port` | `--api-key`, `--hf-token`, `-hft` |
| vLLM | the model argument, `--model`, `--revision`, `--served-model-name`, `--max-model-len`, `--host`, `--port` | `--api-key`, `--hf-token` |

A runtime file has `image`, `container_name`, `host_port`, `ready_path`, `start_timeout_s`, `opencode_provider`, and an `[env]` table; the llama.cpp file adds `served_name` (`local`), and the vLLM file adds named `volumes` and `default_gpu_memory_utilization`.
The llama.cpp runtime file may not set `GGML_CUDA_DISABLE_GRAPHS`, and the vLLM file sets `HF_MODULES_CACHE` to a writable path inside the container, because the model cache is mounted read-only there and a `--trust-remote-code` model copies its code into that cache.
Unknown keys are errors with a "did you mean" hint, because a misspelled key that was ignored would run a configuration you did not write.

## Overrides for another machine

Put your own files in `~/.config/llmctl/recipes/llamacpp/` and `~/.config/llmctl/recipes/vllm/` (`$XDG_CONFIG_HOME/llmctl/recipes` when that is set, or the directory in `LLMCTL_RECIPES_DIR`).

- A file with a shipped recipe's name replaces that recipe whole; nothing is merged across the two files, because half of one measurement beside half of another is a configuration nobody has run.
- A file with a new name adds a model.
- A `_runtime.toml` replaces that runtime's settings whole, for another image tag, port or cache volume; the rules above still hold, and when you change the port, change the OpenCode provider's `baseURL` to match.
- `./llmctl show qwen3.6 --as-override > ~/.config/llmctl/recipes/llamacpp/qwen3.6-35b-a3b.toml` writes a complete copy to start from, marked `kind = "estimate"` until you measure it; [recipes/examples/](../recipes/examples/llamacpp/qwen3.6-35b-a3b.toml) has one.
- An override that does not validate is listed by `check` and `list`, and `up` refuses that model until it is fixed or removed, instead of quietly running the shipped recipe.

On a GPU with less memory, lower `--moe-cache-mib` first and then raise `--n-cpu-moe`, as [llamacpp/README.md](../llamacpp/README.md#moe-expert-offload) explains; on a machine with less RAM, lower `--cache-ram` or `memory_gib`.

## When up refuses

`up` checks, in order: the clients, Docker, the CDI spec, the image, the model files, other containers and the port, free GPU memory, and available RAM.
Each refusal names what it found and exits with the code in its row: 1 for a usage error, 3 for a check that refused before anything started, and 6 when the model is running but a client file or command failed.
A client problem that appears only after the server is ready, such as a file that changed while the model loaded, is reported inside the exit 6 message instead of refusing.
`status` and `list` stop with the same `LLMCTL_CLIENTS`, `HOME` and `~user` usage errors; `down` stops the server first and reports them afterwards, with exit code 1.

| Message starts with | Exit | What to do |
|---|---|---|
| `Docker is not answering` | 3 | Start Docker: `sudo systemctl start docker` |
| `Docker refused the connection (permission denied` | 3 | Add yourself to the `docker` group and log in again |
| `no CDI spec for nvidia.com/gpu` | 3 | Install the NVIDIA Container Toolkit and generate the spec ([host-setup.md](host-setup.md)) |
| `the image ... is not on this machine` | 3 | `./llmctl pull <model>` |
| `the model files are not ready` | 3 | `./llmctl pull <model>`; for vLLM this also finishes a download that stopped partway |
| `another model server is running` | 3 | `./llmctl down`, or `./llmctl up <model> --replace` |
| `a container named ... was not started by llmctl` | 3 | Check that you do not need it, then `docker rm -f <name>` |
| `port 127.0.0.1:... is already in use` | 3 | Stop the program that holds the port |
| `... needs N MiB of free GPU memory` | 3 | Close the programs it lists, or write the override the message suggests: a smaller `--moe-cache-mib` for an offloaded MoE model, a smaller context for other llama.cpp models, a smaller `--gpu-memory-utilization` or context for vLLM |
| `... may use up to N GiB of RAM` | 3 | Close programs, or write an override with a smaller `memory_gib` or `--cache-ram` |
| `nvidia-smi was not found` or `nvidia-smi failed` | 3 | Install or load the NVIDIA driver |
| `another llmctl command is running` | 3 | Wait for it to finish |
| `LLMCTL_CLIENTS: unknown client` | 1 | Use only `opencode`, `hermes` and `pi`, comma-separated, or an empty value for none |
| `LLMCTL_SET_DEFAULT must be 1 or 0` | 1 | Set it to `1` or `0`, or unset it |
| `HOME is not set` | 1 | Run llmctl with `HOME` set; it finds every client config from it |
| `...: llmctl expands only ~ and ~/, not ~user` | 1 | Write `HERMES_HOME`, `PI_CODING_AGENT_DIR` or `XDG_CONFIG_HOME` as a full path, `~` or `~/...` |
| `LLMCTL_CLIENTS names ..., but ... is not installed here` | 3 | Install the client, or drop it from `LLMCTL_CLIENTS`; installed means its binary is on `PATH` and its config folder exists |
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
| `...: the llmctl block markers are damaged` | 3 | Restore the two marker lines exactly ([clients.md](clients.md#the-profile)) |
| `... line N: '...' is set outside the llmctl block` | 3 | Delete that setting: llmctl owns it in this profile, and Hermes' loader rejects a duplicate key ([clients.md](clients.md#the-profile)) |
| `... line N: llmctl cannot check this top-level line` | 3 | Write top-level keys as plain `key:` |
| `Hermes marks the profile llmctl as deleted` | 3 | Delete the file `<root>/profiles/.deleted/llmctl` to let llmctl recreate the profile, or leave Hermes out |
| `Hermes refuses a context below 64,000 tokens` | 3 | Leave Hermes out with `LLMCTL_CLIENTS=opencode,pi` for that recipe |
| `... does not exist; llmctl creates a Hermes profile only inside an existing Hermes home` | 3 | Run Hermes once so it creates its home, or leave Hermes out |
| `...: llmctl's own check of the new profile failed`, `of the new models.json failed` or `of the new settings failed` | 3 | Nothing was written; report it with the file, because llmctl refused a change that would have touched more than its own keys |
| `the model is running, but N client file(s) were not updated` | 6 | The server runs; fix each `<path>: <reason>` the message lists, then run `./llmctl up <model>` again |
| `...: Pi holds its settings lock ...` (inside the exit 6 message) | 6 | Close Pi; if no Pi is running, remove the lock folder the message names, then run `up` again; both Pi files stay as they were |
| `...: Pi took over its settings lock while llmctl held it` (inside the exit 6 message) | 6 | llmctl made its Pi changes, but another writer may have changed `settings.json` meanwhile; check it, then run `up` again |
| `putting ... back failed too` (after an interrupt; inside the exit 6 message when a write failed) | 130 or 6 | `models.json` and `settings.json` now name different llmctl models; run `./llmctl up <model>` again, or fix the two files by hand ([clients.md](clients.md#validation)) |
| `hermes profile use llmctl exited ...` (inside the exit 6 message) | 6 | Read Hermes' message, then run `hermes profile use llmctl` yourself |

Exit codes: 0 success; 1 a usage error, or a download you declined; 2 an invalid recipe or override; 3 a refused check; 4 a server that failed to start or become ready, or a failed `test`; 5 a failed download or checksum; 6 the model is running, but a client file or command failed; 130 interrupted.

## Adding or changing a recipe

1. Measure the launch with the method in [llamacpp/benchmarking.md](../llamacpp/benchmarking.md) or [vllm/benchmarking.md](../vllm/benchmarking.md), and keep the records under `research/`.
2. Write the launch into its section of [llamacpp/models.md](../llamacpp/models.md) or [vllm/models.md](../vllm/models.md), and the recipe file into `recipes/` with its `[clients]` table; set `opencode = true` only together with an entry in [opencode.md](opencode.md#the-validated-config), and otherwise put an "OpenCode entry" snippet in the model's section.
3. Run `python3 -m launcher.tests.regen_golden`, which also writes the Hermes and Pi entries under `launcher/tests/golden/clients/`, read the diff of `launcher/tests/golden/`, and run `python3 -m unittest discover -s launcher/tests -t .`.

The tests check that each recipe matches its document's launch, and that its `[clients]` table matches its OpenCode entry and the server args, so a recipe, its document and the OpenCode config change together or the tests fail.
They run without Docker, a GPU or the network.
