# llmctl clients: stub evidence for Hermes Agent and Pi

Last verified: 2026-11-16.

This folder keeps the evidence behind docs/clients.md and behind the design work for it, whose documents are kept outside the published repository.
No model server, Docker container or real client config took part: Hermes Agent v0.23.1 and Pi 1.5.2 talked only to recording stub servers on 127.0.0.1.
The Phase 1 checks and the design experiments E1 to E9 were planned in those documents; the table below maps every claim id to the files here that back it.

## How the stub runs worked

Hermes ran only inside a bubblewrap sandbox (`bin/design/hxo`, `bin/research-hermes/hxo`).
`$HOME` was an empty tmpfs, and the real `~/.hermes` was mounted back as an overlay whose upper layer lived in the scratch folder, so every Hermes write landed in scratch.
`.env`, `auth.json` and `gateway.sock` were replaced by an empty read-only file, `/run/user/1000` was an empty tmpfs, and the sandbox had private network and PID namespaces and a cleared environment.
The Hermes stub (`bin/design/stub_server.py`, the same file as `bin/research-hermes/stub_server.py`) listened inside the sandbox on 127.0.0.1:8080 (llama.cpp flavor) or 127.0.0.1:8000 (vLLM flavor).

Pi ran through its bundle entry under `env -i` with a temporary `HOME`, `PI_CODING_AGENT_DIR`, `TMPDIR` and XDG folders, and `PI_MANAGED_INSTALL_ROOT` unset (`bin/design/pi/run_pi.sh`, `bin/research-pi/run_pi.sh`).
The Pi stub (`bin/design/pi/stub_server.py`, the same file as `bin/research-pi/stub_server.py`) listened on a free port on 127.0.0.1 and was steered per scenario through `POST /__control`.

Both stubs wrote one JSON file per request, named `NNN-<METHOD>-<path>.json`.
Those files are not kept here (see Privacy); `bin/extract_shapes.py` reduced them to the request shapes under `shapes/`.

## Layout

| Folder | Content |
|---|---|
| `shapes/` | Request shapes, one JSON line per request (fields below) |
| `hermes/design/`, `hermes/phase1/` | Hermes commands, exit status, reviewed output, profile snapshots, reduced events |
| `pi/design/`, `pi/phase1/` | Pi arguments, exit status, reviewed output, reduced events |
| `gen/` | The E9 comparison and rule check |
| `configs/` | Hermes profile configs made for the experiments |
| `bin/` | `extract_shapes.py` and the reviewed scripts, each under its source folder (`design/`, `design/pi/`, `research-hermes/`, `research-pi/`) |

A shape line holds `run`, `method`, `path`, `model`, `keys` (the sorted top-level body keys), `max_tokens`, `max_completion_tokens`, `reasoning_effort`, `chat_template_kwargs`, `response_format_type`, `stream`, `store`, `tools` (the function names) and `authorization` (the scheme word of the header), each left out when the request lacks it.
The `run` value names the source run, for example `design/hz/z-qwen36` or `phase1/req2/r03-mo-bare`.
An `events.jsonl` file is a client's event stream reduced to a few fields: for Pi `type`, `message.role`, `message.provider`, `message.model`, `message.stopReason`, `toolName` and `isError`; for Hermes stream-json `type`, `subtype`, `name`, `is_error` and `exit_code`.

## Claims and files

| Claim | What it shows | Files |
|---|---|---|
| E1 | `hermes -p llmctl config get` starts no daemon and makes no `connect()` (the filtered `strace.txt` is empty), takes about 1 s, but writes outside the profile: the bootstrap record `installs/<id>/bootstrap/llmctl.json`, the prune stamp `cache/scratch/.last_prune`, the `.install.lock` and `.prepare.lock` locks, two removed runtime leases and about 160 bytecode files; the `~/.local/bin` launcher-shim entries in `home-writes.txt` were a sandbox artifact of the tmpfs home, not a Hermes write (spec L3) | `hermes/design/hA1/`, `hermes/design/hA2/` |
| E2 | `hermes profile use llmctl` writes `active_profile`, touches the prune stamp and two locks, starts no daemon and makes no connect, in 0.66 s | `hermes/design/hB1/` |
| E3 | The sandboxed host gateway (the multiplexer) skips the profile as llmctl writes it, with `gateway.standalone`, and logs it as "standalone, not served by this gateway"; `state.txt` lists the served profiles `default`, `broken` and `plain`, so `plain` (no `standalone`) is served, and `broken` (a duplicate key) is served on default settings | `hermes/design/mux/state.txt`, `hermes/design/mux/plain.exit`, `shapes/design/mux.jsonl` |
| E4 | Hermes' own writes land inside the block when the block ends the file (`hw`), and after `_config_version` when that key ends it (`hw2`, where `3.yaml` also shows `model.default` changed in flow style inside the block) | `hermes/design/hw/`, `hermes/design/hw2/` |
| E5 | 19 of 19 generated configs load to the intended values in Hermes' loader, a duplicate top-level key is rejected, and `config get --json` returns the block values | `hermes/design/hy2/result.txt`, `hermes/design/hk/hk-*/keys/` |
| E6 | Seven recipes wrote and read `hello.py` through the final blocks with the request shapes of D1; title requests from `chat -q` send `reasoning_effort` `none` | `hermes/design/hz/`, `hermes/design/hz2/`, `hermes/design/hq/`, `shapes/design/hermes-e6.jsonl` |
| E7 | Pi accepts every generated entry in about 0.2 s; it exits 0 on every failure, a missing key prints nothing on stderr, and the filtered list does not run another provider's `!command` key (`cmdkey`) | `pi/design/pi-validate-all.txt`, `pi/design/val-*/`, `pi/design/neg-*/`, `pi/design/cmdkey/` |
| E8 | Pi's requests with the generated entries and levels reproduce the server defaults, and `--thinking medium` still turns thinking on; a stale `defaultModel` answers from another provider silently (`d6-stale`), and the follow write fixes it (`d6-follow`) | `pi/design/req-*/`, `pi/design/d6-stale/`, `pi/design/d6-follow/`, `shapes/design/pi-e8.jsonl` |
| E9 | The proposed metadata match docs/opencode.md and the vllm/models.md snippets except gemma-4-e4b (no variants), two llama.cpp recipes have no snippet, and all 19 recipes satisfy the server-args rules | `gen/metadata-vs-docs.txt`, `gen/server-args-rules.txt` |
| R02 | `model_overrides` keyed `custom:llmctl` has no effect: `reasoning_effort` `medium` is still sent | `hermes/phase1/req2/r02-mo-full/`, `shapes/phase1/hermes.jsonl` |
| R03 | `model_overrides` keyed `custom` with `supports_reasoning: false` removes `reasoning_effort` | `hermes/phase1/req2/r03-mo-bare/`, `shapes/phase1/hermes.jsonl` |
| R04 | An explicit `agent.reasoning_effort: medium` sends `medium` again despite the override; the `phase1/cfg/c2` shapes come from a profile made from the default config, which has that value | `hermes/phase1/req2/r04-mo-bare-effort-medium/`, `shapes/phase1/hermes.jsonl` |
| R12 | `hermes --reasoning high` sends `reasoning_effort` `high` for an effort model | `hermes/phase1/req2/r12-baseline-for-flag/`, `shapes/phase1/hermes.jsonl` |
| R13 | `--reasoning high` still sends `high` with the `supports_reasoning: false` override | `hermes/phase1/req2/r13-mo-bare-for-flag/`, `shapes/phase1/hermes.jsonl` |
| R19 | A `context_length` below 64,000 is refused with exit 2 before any chat request | `hermes/phase1/req2/r19-ctx-32k/`, `shapes/phase1/hermes.jsonl` |
| R22 | `platform_toolsets: {cli: []}` without `-t` sends no `tools` key | `hermes/phase1/req2/r22-platform-toolsets-empty/`, `shapes/phase1/hermes.jsonl` |
| `stick` | `hermes profile use` makes a profile sticky for every command without `-p`, a supervised child (`HERMES_SUPERVISED_CHILD=1`) ignores it, and `hermes profile use default` deletes `active_profile` | `hermes/phase1/stick/` |
| S3 | A `defaultModel` that is not in `models.json` makes Pi use the first available model, here another user provider, with exit 0 and an empty stderr | `pi/phase1/S3/`, `shapes/phase1/pi.jsonl` |
| S4 | A comment in `settings.json` makes Pi warn and ignore the whole file | `pi/phase1/S4/`, `shapes/phase1/pi.jsonl` |
| SESS2 | A session resumed with `--continue` after its model id disappeared falls back to the settings default silently (`SESS1` is the first run of that session) | `pi/phase1/SESS1/`, `pi/phase1/SESS2/`, `shapes/phase1/pi.jsonl` |
| X2 | `--model llmctl/<id>` with an id the entry lacks only warns on stderr and sends that id | `pi/phase1/X2-wrong-id/`, `shapes/phase1/pi.jsonl` |
| item 8 | Without `compat`, Pi sends `max_completion_tokens` and `store` to `http://127.0.0.1`, and the `developer` role (in the Phase 1 record, since shapes drop the messages) | `pi/phase1/M-baseline-nocompat/`, `shapes/phase1/pi.jsonl` |

## Rerun

The scripts expect the scratch layout they ran in: `$SCRATCH/clients/design` (with `bin/`, `pi/`, `configs/`), `$SCRATCH/clients/research-hermes` (with `bin/`) and `$SCRATCH/clients/research-pi`.
Set `SCRATCH` to an empty folder of your own, copy each `bin/<source>/` folder of this archive back to its place (`bin/design/pi/run_pi.sh` and `stub_server.py` go to `design/pi/` and `design/pi/stub/`, `bin/research-pi/stub_server.py` to `research-pi/stub/`), and copy `configs/` to `$SCRATCH/clients/design/configs/`.
Create the empty mask file the sandbox binds over the secrets: `mkdir -p $SCRATCH/clients/design/masks && : > $SCRATCH/clients/design/masks/empty`, and the same under `research-hermes/`.
Hermes runs need Hermes Agent v0.23.1 installed under `$HOME/.hermes` with its launcher in `$HOME/.local/bin/hermes`, and bubblewrap; Pi runs need Pi 1.5.2 under `$HOME/.pi/agent` and node v24.22.1 under nvm, as the scripts name them.

- Generated entries (E5, E7, E9): `python3 -I $SCRATCH/clients/design/bin/gen.py <repo> $SCRATCH/clients/design/gen` writes every entry and the comparison; `python3 -I $SCRATCH/clients/design/bin/check_rules.py <repo>` checks the server-args rules.
- E1 and E2: `$SCRATCH/clients/design/bin/probe.sh hA1 <profile config> hermes -p llmctl config get model --json` and `$SCRATCH/clients/design/bin/probe.sh hB1 - hermes profile use llmctl`.
- E3: `HXO_DROP_SOCK=1 $SCRATCH/clients/design/bin/hxo mux "" <workdir> $SCRATCH/clients/design/bin/mux_inside.sh $SCRATCH/clients/design/runs/mux`, with the profiles `plain` and `broken` seeded into the overlay from `configs/plain.yaml` and `configs/broken-dup.yaml` and the `llmctl` profile from a generated config.
- E6: `$SCRATCH/clients/design/bin/hcase.sh hz z-qwen36 <generated hermes-config.yaml> llamacpp local 131072 "" -p llmctl -z "Create hello.py that prints hello, then read it back and show its contents" -t file`, and the same per recipe.
- E7 and E8: start the Pi stub with `python3 -I $SCRATCH/clients/design/pi/stub/stub_server.py --log-root $SCRATCH/clients/design/pi/captures --port-file $SCRATCH/clients/design/pi/stub/port`, then `$SCRATCH/clients/design/pi/run_pi.sh <scenario> <folder with models.json and settings.json> '<stub json>' <pi args>`; the arguments of each run are in its `pi-args.txt`.
- Phase 1 Hermes cases: `python3 -I $SCRATCH/clients/research-hermes/bin/make_cases.py` writes the case configs, and `$SCRATCH/clients/research-hermes/bin/batch_req2.sh` runs R02 to R22; `stick` is `hermes/phase1/stick/steps.sh` run inside `hxo`.
- Phase 1 Pi scenarios: `python3 -I $SCRATCH/clients/research-pi/make_configs.py` and `make_edit_configs.py` write the configs, the stub runs as above with `research-pi` paths, and `$SCRATCH/clients/research-pi/matrix.sh` runs the thinking matrix.
- Shapes: `python3 -I bin/extract_shapes.py <captures folder> <out.jsonl> [--run NAME]`.

## Privacy

The sandboxed Hermes read the owner's real `~/.hermes` through the overlay, so a captured system prompt could hold the owner's persona, memories or skills.
Only request shapes were kept: message bodies, every other body field and every header except the `Authorization` scheme word were dropped, and the event streams were reduced to the fields above.
Left out entirely: the raw captures, stub and gateway logs, `gateway_state.json`, `log-grep.txt`, every overlay `upper` and `work` folder, `homes/`, `masks/`, `configs/real-copy.yaml` and the other copies of the owner's config, sessions, `state.db`, `.env` and auth files, the `research-pi/evidence/` hashes of the owner's Pi files, the `rollback/` folder, the `agent/`, `home/`, `tmp/`, `xdg/` and `work/` folders of each run, and every `__pycache__/` folder.
From `strace.txt` only `connect(`, `rename`, `unlink` and write-mode `openat(` lines were kept (the traces recorded only `connect` and `execve`, and no `connect` line occurred); the user name in `ps-after.txt` and in the sandbox wrappers became `$USER`.
The home folder became `$HOME` and the scratch folder `$SCRATCH`.
The only key-like strings left are the placeholders `no-key-required` and `other-placeholder` and the Pi stubs' header-masking code.
