# llmctl clients e2e over the three runtimes, 2026-11-18

Last verified: 2026-11-18.

This folder holds the full run of the llmctl clients e2e over every shipped recipe on llama.cpp, vLLM and SGLang on 2026-11-18, from 05:24:55 to 06:16:47 +03:00 (3,058 s of steps), on branch `llmctl-sglang` at commit 5790b4c.
The matrix was 29 recipes, 15 on llama.cpp `server-cuda-b12407`, 8 on vLLM `v0.33.0` and 6 on SGLang `v0.5.23-runtime`, with OpenCode v2.3.0, Hermes Agent v0.23.1, Pi 1.5.2, DeepSeek Harness (dsh) 0.2.2-rc.3 and Kimi Code CLI 2.5.3, then the `--set-default` round trip and the restore.
The results go into [docs/clients.md](../../../docs/clients.md#verified-on-this-laptop), [docs/llmctl.md](../../../docs/llmctl.md), [sglang/README.md](../../../sglang/README.md) and [docs/opencode.md](../../../docs/opencode.md#context-budget).
The driver wrote `summary.md` and `records/`; do not edit them by hand.

The command was:

```bash
cd $HOME/projects/llm && PATH=$HOME/.kimi-code/bin:$PATH heavy python3 -m launcher.e2e.clients --out research/2026-11-18/llmctl-sglang-e2e --matrix all --set-default-round-trip
```

ripgrep 15.1.0 from apt (`/usr/bin/rg`) and `/usr/bin/fdfind` were installed, so no symlink stood in for `rg` this time.
No `--skip-dsh` or `--skip-kimi` was given, so all five clients ran.

## Matrix and what each step checks

The matrix was all 29 recipes in the order of `ALL_MATRIX`: 14 llama.cpp recipes (01 to 14), the 8 vLLM recipes (15 to 22), the 6 SGLang recipes (23 to 28) and Qwen3.6-35B-A3B on llama.cpp last (29), which leaves the default model running for the round trip.
Per recipe the driver ran `up --replace` and `status`, compared each client's configuration with the generator (`config`, `hermes-config-*`, `dsh-config`, `kimi-config`), ran one write-and-read session of `hello.py` through `opencode run`, `hermes -z`, `hermes chat`, `pi`, `dsh --profile headless` and `kimi -p --output-format stream-json`, started `dsh web` once, and sent the probes; a chat-only recipe got the chat prompt and no tools, and its Hermes block was also checked for `platform_toolsets` (2 rows).
Each SGLang recipe got a `long-grammar` step right after `status`: a calibration request of 20 copies of a fixed code paragraph, a request with as many copies as fill 90% of the context, Hermes' title request with a strict `json_schema`, a forced tool call, and a `GET /ready`, each with `nvidia-smi` free memory before and after.
Probes A, B (not for a chat-only recipe) and G went to every recipe, C to the ids in `PROBE_C`, D to llama.cpp and SGLang, E and F as before.
The live-gateway check ran once, after the first `up` (the `gateway` row).
Then came the round trip with `./llmctl up qwen3.6-35b-a3b --set-default`, one session per client without a model flag, the restore of every client's default, and `./llmctl down`.

The pass rule of Lead step 9 of the plan: exit 0, or every No row is a client session failing on model behaviour (a file written elsewhere, a model-made tool error) or on a limitation the documents name, while every `up`, `status`, `long-grammar`, `config`, `hermes-config`, `dsh-config`, `dsh-web`, `kimi-config`, `gateway`, `probes` and `restore` row reads Yes.
Probe G is recorded, not judged.

## Result

The driver exited 1 with 490 Yes and 20 No rows in 510.
Every `up`, `status`, `long-grammar`, `config`, `hermes-config`, `dsh-config`, `dsh-web`, `kimi-config`, `gateway`, `probes`, `restore` and `down` row reads Yes, so every setup step and every probe passed on all three runtimes (measured on the MSI Raider 18 Max HX, llama.cpp b12407, vLLM v0.33.0 and SGLang v0.5.23, 2026-11-18).
All 20 No rows are client sessions.
The lead classed them in the ledger: 8 are failures the same model and client also had on 2026-11-17, 4 are the same model's failures on its new SGLang recipe, 5 are model behaviour in sessions that passed on 2026-11-17, 1 is a new recipe's model behaviour, 1 is the vLLM DeepSeek-R1 OpenCode entry, which the lead removed after the run, and 1, gpt-oss-20b's SGLang OpenCode session, was reproduced in the [lead's checks](lead-checks/README.md#4-gpt-oss-20bs-sglang-opencode-session) and is now a known issue of SGLang.
Apart from that DeepSeek-R1 entry, which llmctl's documents had shipped, no No row points at a defect in llmctl or the driver.

## The No rows

Failures the same model and client also had on 2026-11-17 ([that run's No rows](../../2026-11-17/llmctl-kimi-e2e/README.md#the-no-rows)):

- 03 Nemotron 3.5 Bolt, `pi` ([record 056](records/056-03-nemotron-3.5-bolt-llamacpp-pi)): reads that ran before the write, or that named `$WORK/hello.py` one folder too high, failed with `ENOENT`; the model then wrote and read the file and answered, but the driver fails a session on any tool error.
- 06 gemma-4-26b-a4b, `dsh-headless` ([record 115](records/115-06-gemma-4-26b-a4b-llamacpp-dsh-headless)): the model added a `justification` argument to `write`, which dsh refused with `SANDBOX_ESCALATION_INVALID_ARGUMENT`, then wrote again without it and answered that the file has 5 lines.
- 17 K2 Horizon 7B on vLLM, `hermes-z` ([record 320](records/320-17-k2-horizon-7b-vllm-hermes-z)): exit 0, but the model aimed `write_file` at the profile's `cache/scratch/hello.py`, which already existed, and Hermes refused the overwrite and appended its file-mutation warning; on 2026-11-17 the same session failed with the HTTP 400 below.
- 17 K2 Horizon 7B on vLLM, `hermes-chat` ([record 321](records/321-17-k2-horizon-7b-vllm-hermes-chat)): the write and the read of `hello.py` in the work folder succeeded, then the next request got HTTP 400 `Assistant message is missing a thinking field` and the result has `exit_code: 1`.
- 17 K2 Horizon 7B on vLLM, `pi` ([record 322](records/322-17-k2-horizon-7b-vllm-pi)): after a write and a read, three assistant messages ended with `stopReason: "error"` and the same HTTP 400.
- 20 gemma-4-e4b on vLLM, `dsh-headless` ([record 378](records/378-20-gemma-4-e4b-vllm-dsh-headless)): two `write` calls refused for the `justification` argument and three reads that failed with `FS_NOT_FOUND` before the file existed.
- 29 Qwen3.6-35B-A3B, `hermes-z` ([record 575](records/575-29-qwen3.6-35b-a3b-llamacpp-hermes-z)): the model wrote `$HOME/hello.py` and answered from it, so no `hello.py` landed in the work folder.
- 29 Qwen3.6-35B-A3B, `hermes-chat` ([record 576](records/576-29-qwen3.6-35b-a3b-llamacpp-hermes-chat)): the model aimed `write_file` at `/tmp/hello.py`, which existed, Hermes refused the write (`is_error`), and the model read that file and answered from it.

The same model's failures on its new SGLang recipe:

- 24 K2 Horizon 7B on SGLang, `hermes-z` ([record 458](records/458-24-k2-horizon-7b-sglang-hermes-z)): as on vLLM, the model aimed `write_file` at the profile's `cache/scratch/hello.py`, and Hermes refused to overwrite the file there (most likely the one of record 320 (inferred)).
- 24 K2 Horizon 7B on SGLang, `hermes-chat` ([record 459](records/459-24-k2-horizon-7b-sglang-hermes-chat)): the write and the read in the work folder succeeded, then the next request got HTTP 400 `Assistant message is missing a thinking field`, the template refusal that [sglang/README.md](../../../sglang/README.md#tool-calling-and-reasoning) lists for a replayed turn without `reasoning_content`.
- 27 gemma-4-e4b on SGLang, `dsh-headless` ([record 534](records/534-27-gemma-4-e4b-sglang-dsh-headless)): three `write` calls refused for the `justification` argument, then a write and a read that succeeded and the answer "5 lines".
- 28 Qwen3.6-35B-A3B on SGLang, `hermes-z` ([record 555](records/555-28-qwen3.6-35b-a3b-sglang-hermes-z)): exit 0 and the answer that `/tmp/hello.py` has 6 lines, so the file landed outside the work folder.

Model behaviour in sessions that passed on 2026-11-17:

- 03 Nemotron 3.5 Bolt, `kimi` ([record 060](records/060-03-nemotron-3.5-bolt-llamacpp-kimi)): the model wrote `hello.py` and answered "5 lines" with no `Read` call.
- 04 Qwen3-Coder-Next, `hermes-z` ([record 074](records/074-04-qwen3-coder-next-llamacpp-hermes-z)): exit 0 and "The file has **6 lines**.", but no `hello.py` in the work folder; `hermes -z` prints no tool events, so the record does not show where the file went.
- 15 gpt-oss-20b on vLLM, `hermes-z` ([record 284](records/284-15-gpt-oss-20b-vllm-hermes-z)): the same, exit 0, "The file 5 lines long.", no `hello.py` in the work folder.
- 17 K2 Horizon 7B on vLLM, `dsh-headless` ([record 323](records/323-17-k2-horizon-7b-vllm-dsh-headless)): after a write and a read, the next request got the HTTP 400 `Assistant message is missing a thinking field` and dsh exited 1; on 2026-11-17 dsh passed with this recipe, so whether the template refuses depends on whether the model's tool turn came back with reasoning (inferred, as in [docs/clients.md](../../../docs/clients.md#thinking-per-type)).
- 19 gemma-4-12b on vLLM, `dsh-headless` ([record 360](records/360-19-gemma-4-12b-vllm-dsh-headless)): the model wrote and read the file but called `update_goal` twice without a goal (`GOAL_NOT_FOUND`), then `create_goal`, and the session ended with exit 1 and a final event without a reason.

A new recipe's model behaviour:

- 14 mellum2.1-12b on llama.cpp, `kimi` ([record 271](records/271-14-mellum2.1-12b-llamacpp-kimi)): Kimi retried one request after `APIEmptyResponseError`, a reply with thinking only, then wrote, read and answered "5 lines"; the driver fails a Kimi session on any retry.

The vLLM DeepSeek-R1 OpenCode entry, which the lead removed:

- 22 deepseek-r1-0528-qwen3-8b on vLLM, `opencode` ([record 411](records/411-22-deepseek-r1-0528-qwen3-8b-vllm-opencode)): exit 1 with HTTP 400 `"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set`, because OpenCode v2.3.0 sends `tool_choice` `auto` although the entry set `"tools": false`.
  The lead reversed plan ruling 39 after the run: a chat-only recipe on a runtime that refuses tools without a parser gets no OpenCode entry, the entry left docs/opencode.md, and the installed key left the owner's OpenCode config; OpenCode users take the llama.cpp DeepSeek-R1 recipe ([docs/opencode.md](../../../docs/opencode.md#the-validated-config)).

gpt-oss-20b's SGLang OpenCode session:

- 25 gpt-oss-20b on SGLang, `opencode` ([record 482](records/482-25-gpt-oss-20b-sglang-opencode)): a completed `write`, then a step with 6,006 input tokens, no text, no tool call and reason `stop`, so OpenCode ended the session without the `read`.
  It has no twin on 2026-11-17, so the lead had it reproduced: three rebuilt sessions passed, and 12 of 160 streamed replays of the turn after the write came back empty, because the model sometimes writes a malformed Harmony tool-call header, such as `to=functions.read json`, that SGLang's `gpt-oss` tool parser drops (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18, with the parser's part inferred for the streamed path; [lead checks](lead-checks/README.md#4-gpt-oss-20bs-sglang-opencode-session)).
  [sglang/README.md](../../../sglang/README.md#known-issues) lists it as a known issue.

A search of the summary for `TIMEOUT` and for `queued` finds none.

## Long-grammar step on SGLang

Every SGLang recipe passed: every request answered HTTP 200, the title parsed, and `/ready` answered 200 at the end, so no server died on a grammar request right after a prompt of 90% of its context (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
Free memory is `nvidia-smi`'s reading before the calibration request and the lowest reading after any request of the step.

| Recipe | Context | Long prompt (tokens) | Long prompt (s) | Title (s) | Forced tool | Free before (MiB) | Lowest free (MiB) | Records |
|---|---|---|---|---|---|---|---|---|
| qwen3.5-9b | 131,072 | 115,575 | 21.9 | 0.5 | 200 | 937 | 551 | [424](records/424-23-qwen3.5-9b-sglang-long-grammar-calibration) to [428](records/428-23-qwen3.5-9b-sglang-long-grammar-ready) |
| k2-horizon-7b | 65,536 | 58,082 | 30.0 | 10.3 | 200 | 1,152 | 754 | [448](records/448-24-k2-horizon-7b-sglang-long-grammar-calibration) to [452](records/452-24-k2-horizon-7b-sglang-long-grammar-ready) |
| gpt-oss-20b | 131,072 | 112,692 | 37.4 | 0.3 | 200 | 1,346 | 1,036 | [473](records/473-25-gpt-oss-20b-sglang-long-grammar-calibration) to [477](records/477-25-gpt-oss-20b-sglang-long-grammar-ready) |
| gemma-4-12b | 131,072 | 115,756 | 94.9 | 0.2 | 200 | 1,119 | 841 | [497](records/497-26-gemma-4-12b-sglang-long-grammar-calibration) to [501](records/501-26-gemma-4-12b-sglang-long-grammar-ready) |
| gemma-4-e4b | 65,536 | 58,009 | 20.5 | 0.1 | 200 | 1,923 | 1,627 | [521](records/521-27-gemma-4-e4b-sglang-long-grammar-calibration) to [525](records/525-27-gemma-4-e4b-sglang-long-grammar-ready) |
| qwen3.6-35b-a3b | 131,072 | 115,772 | 44.5 | 0.3 | 200 | 977 | 493 | [545](records/545-28-qwen3.6-35b-a3b-sglang-long-grammar-calibration) to [549](records/549-28-qwen3.6-35b-a3b-sglang-long-grammar-ready) |

(measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18)

Most of each drop came with the calibration request, 272 to 382 MiB; the long prompt took at most 108 MiB more (Qwen3.6-35B-A3B), and the title and the forced tool call moved nothing.
The SGLang `up` steps took 53, 133, 246, 46, 39 and 240 s in the order of the table.

## Probes D and G

Every probe D and G request got HTTP 200.
Probe D is the plain OK request with `max_tokens` 256 and the recipe id as its model, sent to llama.cpp and SGLang only; probe G is the same request with the served name and `reasoning_effort: "low"`, sent to every recipe and recorded without judging.
The thinking type is the recipe's in [Kimi thinking per type](../../../docs/clients.md#kimi-thinking-per-type).

| Recipe | Runtime | Thinking type | D content | D reasoning | G content | G reasoning | Records |
|---|---|---|---|---|---|---|---|
| qwen3.8-27b | llama.cpp | effort | Yes | Yes | Yes | Yes | [024](records/024-01-qwen3.8-27b-llamacpp-probes-D), [025](records/025-01-qwen3.8-27b-llamacpp-probes-G) |
| k2-horizon-7b | llama.cpp | always | Yes | Yes | Yes | No | [044](records/044-02-k2-horizon-7b-llamacpp-probes-D), [046](records/046-02-k2-horizon-7b-llamacpp-probes-G) |
| nemotron-3.5-bolt | llama.cpp | toggle, default off | Yes | No | Yes | No | [064](records/064-03-nemotron-3.5-bolt-llamacpp-probes-D), [066](records/066-03-nemotron-3.5-bolt-llamacpp-probes-G) |
| qwen3-coder-next | llama.cpp | none | Yes | No | Yes | No | [084](records/084-04-qwen3-coder-next-llamacpp-probes-D), [085](records/085-04-qwen3-coder-next-llamacpp-probes-G) |
| deepseek-r1-0528-qwen3-8b | llama.cpp | always | Yes | Yes | Yes | Yes | [103](records/103-05-deepseek-r1-0528-qwen3-8b-llamacpp-probes-D), [104](records/104-05-deepseek-r1-0528-qwen3-8b-llamacpp-probes-G) |
| gemma-4-26b-a4b | llama.cpp | toggle, default off | Yes | No | Yes | No | [122](records/122-06-gemma-4-26b-a4b-llamacpp-probes-D), [123](records/123-06-gemma-4-26b-a4b-llamacpp-probes-G) |
| nemotron-3.5-lightning | llama.cpp | toggle, default off | Yes | No | Yes | No | [141](records/141-07-nemotron-3.5-lightning-llamacpp-probes-D), [142](records/142-07-nemotron-3.5-lightning-llamacpp-probes-G) |
| groot-19b-a3b | llama.cpp | toggle, default off | Yes | No | Yes | No | [160](records/160-08-groot-19b-a3b-llamacpp-probes-D), [161](records/161-08-groot-19b-a3b-llamacpp-probes-G) |
| k2-horizon-mova-36b | llama.cpp | always | Yes | Yes | Yes | Yes | [180](records/180-09-k2-horizon-mova-36b-llamacpp-probes-D), [181](records/181-09-k2-horizon-mova-36b-llamacpp-probes-G) |
| gpt-oss-20b | llama.cpp | effort | Yes | Yes | Yes | Yes | [199](records/199-10-gpt-oss-20b-llamacpp-probes-D), [200](records/200-10-gpt-oss-20b-llamacpp-probes-G) |
| qwen3.5-9b | llama.cpp | toggle, default off | Yes | No | Yes | No | [218](records/218-11-qwen3.5-9b-llamacpp-probes-D), [219](records/219-11-qwen3.5-9b-llamacpp-probes-G) |
| gemma-4-12b | llama.cpp | toggle, default off | Yes | No | Yes | No | [237](records/237-12-gemma-4-12b-llamacpp-probes-D), [238](records/238-12-gemma-4-12b-llamacpp-probes-G) |
| gemma-4-e4b | llama.cpp | toggle, default off | Yes | No | Yes | No | [256](records/256-13-gemma-4-e4b-llamacpp-probes-D), [257](records/257-13-gemma-4-e4b-llamacpp-probes-G) |
| mellum2.1-12b | llama.cpp | toggle, default on | Yes | Yes | Yes | Yes | [275](records/275-14-mellum2.1-12b-llamacpp-probes-D), [276](records/276-14-mellum2.1-12b-llamacpp-probes-G) |
| qwen3.6-35b-a3b | llama.cpp | toggle, default off | Yes | No | Yes | No | [586](records/586-29-qwen3.6-35b-a3b-llamacpp-probes-D), [587](records/587-29-qwen3.6-35b-a3b-llamacpp-probes-G) |
| gpt-oss-20b | vLLM | effort | not sent | not sent | Yes | Yes | [294](records/294-15-gpt-oss-20b-vllm-probes-G) |
| mellum2.1-12b | vLLM | toggle, default on | not sent | not sent | Yes | Yes | [312](records/312-16-mellum2.1-12b-vllm-probes-G) |
| k2-horizon-7b | vLLM | always | not sent | not sent | Yes | No | [331](records/331-17-k2-horizon-7b-vllm-probes-G) |
| qwen3.5-9b | vLLM | toggle, default off | not sent | not sent | Yes | Yes | [349](records/349-18-qwen3.5-9b-vllm-probes-G) |
| gemma-4-12b | vLLM | toggle, default off | not sent | not sent | Yes | Yes | [367](records/367-19-gemma-4-12b-vllm-probes-G) |
| gemma-4-e4b | vLLM | toggle, default off | not sent | not sent | Yes | Yes | [385](records/385-20-gemma-4-e4b-vllm-probes-G) |
| groot-19b-a3b | vLLM | toggle, default off | not sent | not sent | Yes | Yes | [403](records/403-21-groot-19b-a3b-vllm-probes-G) |
| deepseek-r1-0528-qwen3-8b | vLLM | always | not sent | not sent | No | Yes | [421](records/421-22-deepseek-r1-0528-qwen3-8b-vllm-probes-G) |
| qwen3.5-9b | SGLang | toggle, default off | Yes | No | Yes | Yes | [444](records/444-23-qwen3.5-9b-sglang-probes-D), [445](records/445-23-qwen3.5-9b-sglang-probes-G) |
| k2-horizon-7b | SGLang | always | Yes | Yes | Yes | Yes | [469](records/469-24-k2-horizon-7b-sglang-probes-D), [470](records/470-24-k2-horizon-7b-sglang-probes-G) |
| gpt-oss-20b | SGLang | effort | Yes | Yes | Yes | Yes | [493](records/493-25-gpt-oss-20b-sglang-probes-D), [494](records/494-25-gpt-oss-20b-sglang-probes-G) |
| gemma-4-12b | SGLang | toggle, default off | Yes | No | Yes | Yes | [517](records/517-26-gemma-4-12b-sglang-probes-D), [518](records/518-26-gemma-4-12b-sglang-probes-G) |
| gemma-4-e4b | SGLang | toggle, default off | Yes | No | No | Yes | [541](records/541-27-gemma-4-e4b-sglang-probes-D), [542](records/542-27-gemma-4-e4b-sglang-probes-G) |
| qwen3.6-35b-a3b | SGLang | toggle, default off | Yes | No | Yes | Yes | [566](records/566-28-qwen3.6-35b-a3b-sglang-probes-D), [567](records/567-28-qwen3.6-35b-a3b-sglang-probes-G) |

(measured on the MSI Raider 18 Max HX, llama.cpp b12407, vLLM v0.33.0 and SGLang v0.5.23, 2026-11-18)

On SGLang, probe D is the control that the vLLM rows lack: all four toggle recipes, whose server default is off, returned no reasoning without the field and reasoning with `reasoning_effort: "low"`, as SGLang turns `enable_thinking` on for any effort other than `none`.
Two G replies had no content: vLLM DeepSeek-R1 spent all 256 tokens reasoning (`finish_reason` `length`), and SGLang gemma-4-e4b put its whole answer, `OK.`, into `reasoning_content` and left `content` empty.
K2 Horizon 7B on llama.cpp returned no reasoning to probe G, against reasoning on 2026-11-17 and to probe D in this run.

## Round trip and restore

`./llmctl up qwen3.6-35b-a3b --set-default` ([record 588](records/588-round-trip-up)) found the model running from the same recipe file, started nothing, and made it the default of all five clients.
OpenCode, `hermes -z` without `-p`, Pi, dsh in the real dsh home and `kimi -p` without `-m` all passed ([records 589 to 596](records/589-round-trip-opencode)); this time the round trip's `hermes -z` wrote into the work folder.
The restore row reads Yes: the driver put every client's default back and read it back.
`./llmctl down` stopped the server ([record 597](records/597-down)), so the run ended with no model running.

## End state

After the run no model server ran and every client's default was as before the run.
The lead's checks followed from 06:27 to 06:51 +03:00 and also ended with no model running ([lead-checks/](lead-checks/README.md#end-state)).

## Versions

Records 001 to 006 hold the version lines: Hermes Agent v0.23.1, Pi 1.5.2, OpenCode v2.3.0, dsh 0.2.2-rc.3, Kimi Code 2.5.3, and `nvidia-smi --query-gpu=power.limit`, which printed `[N/A]`.
The images were `ghcr.io/ggml-org/llama.cpp:server-cuda-b12407`, `vllm/vllm-openai:v0.33.0` and `lmsysorg/sglang:v0.5.23-runtime`, as the runtime files pin them.

## Files

- [summary.md](summary.md): the driver's table, one row per step with its pass mark, reasons and seconds.
- [records/](records/): 597 files, one per command, probe or long-grammar request, numbered in run order.
- [lead-checks/](lead-checks/README.md): the lead's checks of Lead step 10, a network snapshot of an SGLang start, OpenCode compaction on SGLang, the vLLM tokenizer overlay on the GPU, and the reproduction of gpt-oss-20b's SGLang OpenCode session, with the scripts that made every record.

Privacy: every record shows the private work folder as `$WORK` and the home folder as `$HOME`, and the lead-checks write the session's scratch folder as `$SCRATCH`.
In the 28 Pi records, the one installed agent skill that Pi's system prompt lists, which is unrelated to llmctl, was replaced with `<an installed agent skill unrelated to llmctl>` before the commit, its name, description and `SKILL.md` location included.
The Docker logs keep the request lines from Docker's bridge, 172.17.0.1 and the container's own 172.17.0.x, which identify no machine or person.
The work folders and the private Kimi and dsh homes stayed in the run's private temporary folder outside the repository and are not part of this folder.
