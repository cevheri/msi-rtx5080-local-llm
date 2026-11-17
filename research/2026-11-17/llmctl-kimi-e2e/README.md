# llmctl clients e2e with Kimi Code, 2026-11-17

Last verified: 2026-11-17.

This folder holds the full run of the llmctl clients e2e with five clients on 2026-11-17, from 14:40 to 15:07 local time (1,635 s of steps): every shipped recipe, 11 on llama.cpp `server-cuda-b12407` and 8 on vLLM `v0.33.0`, with OpenCode v2.3.0, Hermes Agent v0.23.1, Pi 1.5.2, DeepSeek Harness (dsh) 0.2.2-rc.3 and Kimi Code CLI 2.5.3, then the `--set-default` round trip and the restore.
The results go into [docs/clients.md](../../../docs/clients.md#verified-on-this-laptop).
The driver wrote `summary.md` and `records/`; do not edit them by hand.

The command was:

```bash
cd $HOME/projects/llm && PATH=$HOME/.kimi-code/bin:$SCRATCH/kimi/rgbin:$PATH PYTHONDONTWRITEBYTECODE=1 heavy python3 -m launcher.e2e.clients --out research/2026-11-17/llmctl-kimi-e2e --matrix all --set-default-round-trip
```

`$SCRATCH` is the lead's session scratch folder.
ripgrep was not installed from apt at the time, so `$SCRATCH/kimi/rgbin` held only a symlink named `rg` to the Claude Code binary, which runs as ripgrep 14.1.1 under that name; it satisfied the driver's check that `rg` is on `PATH` (Kimi's Grep tool needs it), and `fd` came from Kimi's own `~/.kimi-code/bin`.
The driver put `$HOME/.kimi-code/bin` first on `PATH` itself as well.

## Matrix and pass rule

The matrix was all 19 recipes with five clients.
Per recipe the driver ran `up --replace` and `status`, compared each client's configuration with the generator (`config`, `hermes-config-*`, `dsh-config`, `kimi-config`), ran one write-and-read session through `opencode run`, `hermes -z`, `hermes chat`, `pi`, `dsh --profile headless` and `kimi -p --output-format stream-json`, started `dsh web` once, and sent the probes; a chat-only recipe got the chat prompt and no tools.
Then came the round trip with `./llmctl up qwen3.6-35b-a3b --set-default`, a session per client without a model flag, the restore of every client's default, and `./llmctl down`.

The pass rule of the run: exit 0, or every No row is a client session failing on model behaviour (a file written elsewhere, a model-made tool error, a `Bash` call in a Kimi session) or on an outcome the Kimi design names (K2 Horizon 7B on vLLM rejecting a replayed tool turn without a thinking field, in Hermes, Pi or Kimi), while every `up`, `status`, `config`, `hermes-config`, `dsh-config`, `dsh-web`, `kimi-config`, `gateway` and `restore` row and every `probes` row reads Yes, except vLLM deepseek-r1-0528-qwen3-8b's probes row, whose probe A gets a garbled title from the known tokenizer failure.
Probe G is recorded, not judged.

## Result

The driver exited 1 with 322 Yes and 11 No rows in 333.
All 39 Kimi rows read Yes: 19 `kimi-config` rows, 19 `kimi` sessions and the round trip's `kimi` session, K2 Horizon 7B on vLLM included.
Every `up`, `status`, `config`, `hermes-config`, `dsh-config`, `dsh-web`, `kimi-config`, `gateway` and `restore` row reads Yes, and every `probes` row but vLLM deepseek-r1-0528-qwen3-8b's.
The run passes the rule above: seven No rows are model behaviour, three are the named K2 Horizon 7B rejection, and one is the named DeepSeek-R1 probe A; no No row points at a defect in llmctl or the driver.
A search of the records for `TIMEOUT` and for `queued` found none.

## The No rows

Model behaviour:

- 03 Nemotron 3.5 Bolt, `pi` ([record 056](records/056-03-nemotron-3.5-bolt-llamacpp-pi)): the model sent `write` and `read` in one assistant message, the read ran before the write and failed with `ENOENT`, then the model read `hello.py` again with success; the work folder holds a valid `hello.py`, but the driver fails a session on any tool error.
- 06 gemma-4-26b-a4b, `dsh-headless` ([record 115](records/115-06-gemma-4-26b-a4b-llamacpp-dsh-headless)): the model added a `justification` argument to `write`, which dsh refused with `invalid escalation: justification is only valid together with sandbox_permissions` (`SANDBOX_ESCALATION_INVALID_ARGUMENT`), then wrote again without it.
- 16 gemma-4-e4b on vLLM, `dsh-headless` ([record 302](records/302-16-gemma-4-e4b-vllm-dsh-headless)): the same refused `justification` argument three times and four reads that failed with `FS_NOT_FOUND` before the file existed; the session ended with a valid `hello.py` and the answer "5 lines".
- 07 Nemotron 3.5 Lightning, `hermes-z` ([record 131](records/131-07-nemotron-3.5-lightning-llamacpp-hermes-z)): exit 0, three API calls, the answer "The file has 5 lines.", and no `hello.py` in the work folder; `hermes -z` prints no tool events, so the record does not show where the file went.
- 09 K2 Horizon Mova 36B, `hermes-chat` ([record 170](records/170-09-k2-horizon-mova-36b-llamacpp-hermes-chat)): the model passed the absolute path `$HOME/tmp/hello.py` to `write_file` and `read_file`, and Hermes wrote it there.
- 19 Qwen3.6-35B-A3B, `hermes-z` ([record 352](records/352-19-qwen3.6-35b-a3b-llamacpp-hermes-z)) and `hermes-chat` ([record 353](records/353-19-qwen3.6-35b-a3b-llamacpp-hermes-chat)): both sessions aimed `write_file` at `$HOME/hello.py`, which already existed; Hermes refused the write (`Refusing to overwrite ... this task has not seen its full current content`, `is_error: true`), the model read that file and answered from it, and Hermes appended its file-mutation warning.

The known K2 Horizon 7B rejection on vLLM (HTTP 400 "Assistant message is missing a thinking field"):

- 13 K2 Horizon 7B on vLLM, `hermes-z` ([record 244](records/244-13-k2-horizon-7b-vllm-hermes-z)): exit 2, `Provider said: HTTP 400: Assistant message is missing a thinking field. Provide one of: think, reasoning, reasoning_content, think_fast, think_faster.`
- 13 K2 Horizon 7B on vLLM, `hermes-chat` ([record 245](records/245-13-k2-horizon-7b-vllm-hermes-chat)): the write and the read of `hello.py` in the work folder succeeded, then the next request got the same HTTP 400 and the result has `exit_code: 1`.
- 13 K2 Horizon 7B on vLLM, `pi` ([record 246](records/246-13-k2-horizon-7b-vllm-pi)): after a successful write and read, the last assistant message has `stopReason: "error"` with the same 400 message.

The named DeepSeek-R1 tokenizer failure:

- 18 deepseek-r1-0528-qwen3-8b on vLLM, `probes` ([record 343](records/343-18-deepseek-r1-0528-qwen3-8b-vllm-probes-A)): probe A got status 200 with content and reasoning, but the title came back as `{"title":Ġ"AddingĠTwoĠNumbersĠinĠPython"}`, with the byte-level `Ġ` for every space, so `title_ok` is false.

## Rows that changed against the dsh run

The run of [../llmctl-dsh-e2e/](../llmctl-dsh-e2e/README.md) had 294 rows, 284 Yes and 10 No; this run has the same 294 rows plus 39 Kimi rows.
Three of the shared rows changed, all in Hermes sessions, and none through a change in llmctl:

- 09 K2 Horizon Mova 36B, `hermes-chat`: Yes before, when the model wrote the relative path `hello.py`; No now, with `$HOME/tmp/hello.py` (model behaviour).
- 13 K2 Horizon 7B on vLLM, `hermes-chat`: Yes before, when the session's follow-up request after the read was accepted; No now with the known HTTP 400, after the model wrote and read by absolute path in the work folder (the known rejection, which depends on whether a tool turn came back with reasoning (inferred)).
- round trip, `hermes-z`: No before, when the file landed in the profile's `cache/scratch` folder; Yes now, with `hello.py` in the work folder.

The other nine No rows of the dsh run are No again for the same reasons.

## Round trip and restore

`./llmctl up qwen3.6-35b-a3b --set-default` ([record 365](records/365-round-trip-up)) set every client's default, Kimi's included (`Kimi: llmctl/local, the default`).
OpenCode, `hermes -z` without `-p`, Pi, dsh and `kimi -p` without `-m` ([record 372](records/372-round-trip-kimi)) all passed; Kimi's session ran in a private home with a pure byte copy of the real `config.toml`, and the server log shows its request ([record 373](records/373-round-trip-kimi-docker-logs)).
The restore row reads Yes: the driver put every client's default back, Kimi's `default_model` to the owner's earlier state, and read it back.
`./llmctl down` stopped the server ([record 374](records/374-down)).

## Probe G

Probe G is the plain OK request with `max_tokens` 256 and `reasoning_effort: "low"`, sent to every recipe and recorded without judging.
Every recipe answered with status 200 and content; the table shows whether reasoning came back too.

| Recipe | Runtime | Status | Content | Reasoning | Record |
|---|---|---|---|---|---|
| qwen3.8-27b | llama.cpp | 200 | Yes | Yes | [025](records/025-01-qwen3.8-27b-llamacpp-probes-G) |
| k2-horizon-7b | llama.cpp | 200 | Yes | Yes | [046](records/046-02-k2-horizon-7b-llamacpp-probes-G) |
| nemotron-3.5-bolt | llama.cpp | 200 | Yes | No | [066](records/066-03-nemotron-3.5-bolt-llamacpp-probes-G) |
| qwen3-coder-next | llama.cpp | 200 | Yes | No | [085](records/085-04-qwen3-coder-next-llamacpp-probes-G) |
| deepseek-r1-0528-qwen3-8b | llama.cpp | 200 | Yes | Yes | [104](records/104-05-deepseek-r1-0528-qwen3-8b-llamacpp-probes-G) |
| gemma-4-26b-a4b | llama.cpp | 200 | Yes | No | [123](records/123-06-gemma-4-26b-a4b-llamacpp-probes-G) |
| nemotron-3.5-lightning | llama.cpp | 200 | Yes | No | [142](records/142-07-nemotron-3.5-lightning-llamacpp-probes-G) |
| groot-19b-a3b | llama.cpp | 200 | Yes | No | [161](records/161-08-groot-19b-a3b-llamacpp-probes-G) |
| k2-horizon-mova-36b | llama.cpp | 200 | Yes | Yes | [181](records/181-09-k2-horizon-mova-36b-llamacpp-probes-G) |
| gpt-oss-20b | llama.cpp | 200 | Yes | Yes | [200](records/200-10-gpt-oss-20b-llamacpp-probes-G) |
| gpt-oss-20b | vLLM | 200 | Yes | Yes | [218](records/218-11-gpt-oss-20b-vllm-probes-G) |
| mellum2.1-12b | vLLM | 200 | Yes | Yes | [236](records/236-12-mellum2.1-12b-vllm-probes-G) |
| k2-horizon-7b | vLLM | 200 | Yes | No | [255](records/255-13-k2-horizon-7b-vllm-probes-G) |
| qwen3.5-9b | vLLM | 200 | Yes | Yes | [273](records/273-14-qwen3.5-9b-vllm-probes-G) |
| gemma-4-12b | vLLM | 200 | Yes | Yes | [291](records/291-15-gemma-4-12b-vllm-probes-G) |
| gemma-4-e4b | vLLM | 200 | Yes | No | [309](records/309-16-gemma-4-e4b-vllm-probes-G) |
| groot-19b-a3b | vLLM | 200 | Yes | Yes | [327](records/327-17-groot-19b-a3b-vllm-probes-G) |
| deepseek-r1-0528-qwen3-8b | vLLM | 200 | Yes | Yes | [344](records/344-18-deepseek-r1-0528-qwen3-8b-vllm-probes-G) |
| qwen3.6-35b-a3b | llama.cpp | 200 | Yes | No | [364](records/364-19-qwen3.6-35b-a3b-llamacpp-probes-G) |

(measured on the MSI Raider 18 Max HX, llama.cpp b12407 and vLLM v0.33.0, 2026-11-17)

## Versions

Records 001 to 006 hold the version lines: Hermes Agent v0.23.1, Pi 1.5.2, OpenCode v2.3.0, dsh 0.2.2-rc.3, Kimi Code 2.5.3, and `nvidia-smi --query-gpu=power.limit`, which printed `[N/A]`.
The images were `ghcr.io/ggml-org/llama.cpp:server-cuda-b12407` and `vllm/vllm-openai:v0.33.0`, as the recipes pin them.

## Files

- [summary.md](summary.md): the driver's table, one row per step with its pass mark, reasons and seconds.
- [records/](records/): 374 files, one per command or probe, numbered in run order; each record shows the private work folder as `$WORK` and the home folder as `$HOME`.
  The Kimi steps add `<n>-<recipe>-kimi` and its `-kimi-docker-logs` record; `kimi-config` runs in process and writes a record only on a mismatch.
- [lead-checks/](lead-checks/README.md): the lead's L-c check of one Kimi session's connect calls, and in [lead-checks/tui/](lead-checks/tui/README.md) the TUI checklist.

The work folders and the private Kimi homes, which hold Kimi's session files with whole system prompts, stayed in the run's private temporary folder outside the repository and are not part of this folder.
