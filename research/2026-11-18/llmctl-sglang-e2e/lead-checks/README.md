# Lead checks of D15 after the clients e2e

Last verified: 2026-11-18.

These are the lead's checks of Lead step 10 of the SGLang plan, run on 2026-11-18 between 03:27 and 03:51 UTC on branch `llmctl-sglang` at commit 5790b4c, right after the [clients e2e](../summary.md).
Every figure comes from this laptop (MSI Raider 18 Max HX, RTX 5080 Laptop GPU 16 GB) with SGLang v0.5.23, vLLM v0.33.0 and OpenCode v2.3.0, one model server at a time.
Paths are written as `$HOME` and `$SCRATCH`, the session's scratch folder, and the scripts that made every record are in [scripts/](scripts/).
[image-sizes.txt](image-sizes.txt) holds the `docker image ls` sizes of the three runtime images that the disk budget of [docs/host-setup.md](../../../../docs/host-setup.md#disk-budget) gives, read once at 07:55 +03:00 on 2026-11-18 with Docker 29.9.3 (measured on the MSI Raider 18 Max HX).

| Check | Result | Evidence |
|---|---|---|
| 1. Network snapshot of an SGLang start (ruling 22) | Pass, as a snapshot | 63 samples, every remote address loopback, unspecified or 172.17.0.x |
| 2. OpenCode compaction on SGLang (D15, research risk 14) | Fail as written: compaction works, but 2 of the 5 compactions seen began with two HTTP 400s | The session ended with exit 0 and `DONE`; its largest served request held 77,439 tokens |
| 3. vLLM tokenizer overlay on the GPU (ruling 35) | Pass | The `wrote` line with 9 links, `prompt_tokens` 8, no `Ġ` or `Ċ` |
| 4. gpt-oss-20b's SGLang OpenCode session | 3 of 3 runs passed; the e2e failure reproduced at the turn level | 12 of 160 streamed replays of the turn after the write came back empty |

## 1. Network snapshot (ruling 22)

`LLMCTL_CLIENTS= ./llmctl up qwen3.5-9b --runtime sglang --replace` ran in the background while [netsample.py](scripts/netsample.py) ran `docker exec sglang cat /proc/net/tcp /proc/net/tcp6 /proc/net/udp /proc/net/udp6` every 0.5 s ([check1.sh](scripts/check1.sh)).
The sampler started at 03:27:03.885 UTC, before `up`; its first attempt found the old container stopped, and from 03:27:04.398 every attempt succeeded, so sampling began within 0.5 s of the new container's start ([sampler.log](network/sampler.log)).
After `ready:` the script sent three chat requests (`Say hello in one sentence.`, `max_tokens` 64), each answered HTTP 200 in 0.14 to 0.16 s ([requests.jsonl](network/requests.jsonl)), and stopped the sampler one second after the third reply, so the last sample is from 03:27:35.398 (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
`up` printed `memory: KV pool 140974 tokens for context 131072; 937 MiB of GPU memory free after the start, at least 768 MiB needed` and exited 0 ([up.txt](network/up.txt)); its line times are not exact, because llmctl's output reached the pipe in one block at its exit.

All 63 samples were decoded with the archived [proc_net.py](../../../2026-11-17/sglang-runtime/handson/scripts/proc_net.py) ([decoded.txt](network/decoded.txt)), and [netsummary.py](scripts/netsummary.py) lists each distinct remote address per table and state ([summary.txt](network/summary.txt), raw tables in [samples.txt](network/samples.txt)):

| Table | State | Remote address | Samples |
|---|---|---|---|
| tcp | LISTEN | 0.0.0.0 | 24 |
| tcp | ESTABLISHED, TIME_WAIT | 127.0.0.1 | 2, 5 |
| tcp | ESTABLISHED, TIME_WAIT | 172.17.0.1 (the bridge gateway, which carries the host's requests) | 1, 8 |
| tcp | ESTABLISHED, TIME_WAIT | 172.17.0.2 (the container itself) | 5, 5 |
| tcp6 | LISTEN | `::`, the IPv6 form of 0.0.0.0 | 25 |
| tcp6 | ESTABLISHED | `::ffff:127.0.0.1`, 127.0.0.1 on a dual-stack socket | 24 |

No sample held a UDP socket, and no remote address lies outside 127.0.0.1, ::1, 0.0.0.0 and 172.17.0.x (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
This is a snapshot, not proof: a connection that opens and closes between two samples, 0.5 s apart, leaves no trace here.

## 2. OpenCode compaction (D15, research risk 14)

The work folder `$SCRATCH/sglang/compaction/work` held the 18 Markdown files `README.md`, `docs/*.md`, `sglang/*.md`, `vllm/*.md` and `llamacpp/*.md` at commit 5790b4c, 1,132,607 bytes in all ([files.txt](compaction/files.txt)), and the e2e driver's [opencode.json](compaction/opencode.json).
On the server of check 1, [compaction.sh](scripts/compaction.sh) ran `heavy opencode run --standalone --archived --auto --format json -m sglang-docker/qwen3.5-9b "<prompt>"` there, with the prompt in [prompt.txt](compaction/prompt.txt).

The session ran from 03:28:45 to 03:30:03 UTC and exited 0 with an empty stderr; its last events are the text `DONE` and a `step_finish` with reason `stop` ([events.jsonl](compaction/run/events.jsonl), where each read output keeps only its length and hash).
It read 10 of the 18 files, six of them twice, never opened the eight files of docs/, and wrote one-line summaries for two files only (model behaviour).
[request_table.py](scripts/request_table.py) joins SGLang's prefill lines per request, and its table of the session's 16 chat requests matches the `input` token counts in OpenCode's events ([requests.txt](compaction/run/requests.txt), lines in [server-requests.log](compaction/run/server-requests.log)):

| Request | Time (UTC) | HTTP | Cached + new tokens | What it was |
|---|---|---|---|---|
| 9 | 03:29:09.459 to 03:29:16.098 | 200 | 39,168 + 38,271 = 77,439 | A step that read three files |
| 10, 11 | 03:29:20.526, 03:29:20.672 | 400 | not scheduled | Compaction attempts that SGLang refused |
| 12 | 03:29:20.838 to 03:29:21.751 | 200 | 0 + 14,432 | The compaction summary |
| 13 | 03:29:32.183 to 03:29:32.497 | 200 | 0 + 7,551 | The next step, on the summary |
| 14 | 03:29:37.107 to 03:29:43.954 | 200 | 7,936 + 53,348 = 61,284 | A step that read six files |
| 15 | 03:29:51.748 to 03:29:52.595 | 200 | 0 + 13,947 | The second compaction summary, with no 400 |
| 16 | 03:30:02.861 to 03:30:03.185 | 200 | 0 + 7,539 | The last step, `DONE` |

(measured on the MSI Raider 18 Max HX, SGLang v0.5.23 and OpenCode v2.3.0, 2026-11-18)

So OpenCode compacted twice and finished, but no served request exceeded 100,000 tokens, and SGLang answered two requests with HTTP 400, which fails both halves of the pass rule as written.
The three reads of request 9's step came to 50,309 tokens by SGLang's `/v1/tokenize`, so the conversation after it held about 77,439 + 295 + 50,309 = 128,043 tokens, more than the 114,688 that leave room for OpenCode's `max_completion_tokens` of 16,384 (inferred from those counts; requests 10 and 11 of this run were not captured).

Two more sessions with the same prompt and files ran under strace, tracing only the socket calls that carry OpenCode's HTTP traffic, so their requests could be rebuilt ([session_traced.sh](scripts/session_traced.sh), [strace_http.py](scripts/strace_http.py), [shapes.py](scripts/shapes.py)):

- [diag-1](compaction/diag-1/exchanges.txt): the model read ten files in one step after an 8,047-token request, and OpenCode compacted before sending them: a 14,679-token summary request, then 8,394 tokens and `DONE`, with no 400.
- [diag-2](compaction/diag-2/exchanges.txt): request 9 was served with 91,392 cached and 19,335 new tokens, 110,727 in all, and was followed by the compaction (12,263 tokens) and a 7,054-token step, the pattern the pass rule looks for ([requests.txt](compaction/diag-2/requests.txt)).
  Its compaction took three tries.
  Request 10 was the whole conversation with OpenCode's summary instruction appended, 8 tools and `max_completion_tokens` 6,344, and SGLang answered HTTP 400 `The input (131332 tokens) is longer than the model's context length (131072 tokens).`
  Request 11 put the conversation into one 344,159-character user message without tools, and SGLang answered HTTP 400 `Requested token count exceeds the model's maximum context length of 131072 tokens. You requested a total of 145252 tokens: 128868 tokens from the input messages and 16384 tokens for the completion. ...`
  Request 12 cut that message to 19,556 characters and got the summary, and the session went on and ended with exit 0 (measured on the MSI Raider 18 Max HX, SGLang v0.5.23 and OpenCode v2.3.0, 2026-11-18).

Two requests sent straight to the server show the same two refusals for a long user message ([overflow.txt](compaction/overflow.txt)).
The 400s come from OpenCode's own compaction attempts, which it absorbs: no session showed an error event.
OpenCode sent 131,332 tokens with `max_completion_tokens` 6,344, which fits only if it counted that input at no more than 124,728 tokens, so it counts this Markdown at least 5% low (inferred from the captured parameter and SGLang's count); the three reads of the first session ran at 2.6 characters per token.
llama.cpp and vLLM refuse a prompt that does not fit beside its output as well, so the same 400s are expected there (inferred, not measured here).
For the pass rule as written, the prompt needs nothing more, since compaction did happen: the rule's 100,000-token request is served only when OpenCode does not compact first, as in diag-2, and the 400s are part of OpenCode's compaction near the edge of a 131,072-token window.

## 3. vLLM tokenizer overlay (ruling 35)

[check3.sh](scripts/check3.sh) ran `./llmctl down`, then `LLMCTL_CLIENTS= ./llmctl up deepseek-r1 --runtime vllm --replace`, which printed ([up.txt](vllm-overlay/up.txt)):

```text
wrote $HOME/.cache/llmctl/vllm/tokenizers/deepseek-r1-0528-qwen3-8b: tokenizer overlay with 9 links into snapshot 6e8885a6ff5c of deepseek-ai/DeepSeek-R1-0528-Qwen3-8B, and tokenizer_config.json with tokenizer_class PreTrainedTokenizerFast
```

The server reached `ready:` 23 s after the command, and its engine line reads `tokenizer='/llmctl/tokenizer'` ([server-lines.txt](vllm-overlay/server-lines.txt)).
One `POST http://127.0.0.1:8000/v1/chat/completions` with the model `deepseek-r1-8b`, the user message `Reply with exactly: OK` and `max_tokens` 4096 answered HTTP 200 in 2.6 s with `usage.prompt_tokens` 8, 188 completion tokens of which 183 reasoning, the content `\nOK` and 880 characters of reasoning, and the whole reply holds no `Ġ` and no `Ċ` ([request.json](vllm-overlay/request.json)) (measured on the MSI Raider 18 Max HX, vLLM v0.33.0, 2026-11-18).
`./llmctl down` stopped it ([down-2.txt](vllm-overlay/down-2.txt)).
The recipe file in the checkout differed from commit 5790b4c only by `opencode = false`, the edit of the agent working beside this one, which writes no client file and changes nothing in the start.

## 4. gpt-oss-20b's SGLang OpenCode session

The e2e's record 482 holds a session that completed a `write` call and then got a step with 6,006 input tokens, 0 output and 11 reasoning tokens, reason `stop`, no text and no tool call.
`LLMCTL_CLIENTS= ./llmctl up gpt-oss-20b --runtime sglang --replace` started the server in 51 s ([up.txt](gpt-oss-opencode/up.txt)), and [check4.sh](scripts/check4.sh) rebuilt the driver's session three times from `launcher/e2e/clients.py`: the prompt `TASK` ([task.txt](gpt-oss-opencode/task.txt)), a fresh folder `$SCRATCH/sglang/gptoss-oc/run-<n>/25-gpt-oss-20b-sglang-opencode` holding only the driver's `opencode.json`, `opencode run --standalone --archived --auto --format json --model sglang-docker/gpt-oss-20b '<TASK>'`, and the driver's environment (`$HOME/.kimi-code/bin` first on `PATH`, `PWD` set to the folder, no `LLMCTL_SET_DEFAULT`), each through heavy and under the same socket-only strace.
[outcome.py](scripts/outcome.py) judged each run with the driver's own `opencode_outcome` and `hello_valid`:

| Run | Exit | Seconds | Steps (input tokens) | Driver's verdict |
|---|---|---|---|---|
| [1](gpt-oss-opencode/run-1/events.jsonl) | 0 | 12.0 | write (6,009), read (6,090), answer (6,193) | pass |
| [2](gpt-oss-opencode/run-2/events.jsonl) | 0 | 7.3 | write (6,009), read (6,090), answer (6,193) | pass |
| [3](gpt-oss-opencode/run-3/events.jsonl) | 0 | 6.4 | write (6,009), read (6,090), answer (6,180) | pass |

(measured on the MSI Raider 18 Max HX, SGLang v0.5.23 and OpenCode v2.3.0, 2026-11-18)

Each answered `The file **hello.py** contains **5 lines**.`, and each run's request lines and exchanges are beside its events.
OpenCode sends the earlier analysis back as `reasoning_content` (1,751 characters in run 1), yet the prompt grows by only 81 tokens from 6,009 to 6,090, so the template leaves it out, as open point 7 and #44931 say (inferred from the token counts).

No run failed, so to find the cause the turn after the write was replayed straight to `http://127.0.0.1:30000/v1/chat/completions`, sending run 1's captured request unchanged ([turn.txt](gpt-oss-opencode/replay/turn.txt), [replay.py](scripts/replay.py), [seedscan.py](scripts/seedscan.py)); [tally.txt](gpt-oss-opencode/replay/tally.txt) counts the replies:

- Streamed, as OpenCode streams: 148 of 160 replies held a `read` call and 12 were empty, with `content` "", no `tool_calls`, `finish_reason` `stop` and `matched_stop` 200012, which is `<|call|>` ([empty-reply.sse](gpt-oss-opencode/replay/empty-reply.sse)).
- Five of those twelve counted every completion token as reasoning and sent no reasoning text, 18 to 27 tokens, the shape of the e2e's step with 11.
- For one batch of 40, SGLang's request log was on (`POST /configure_logging` with `log_requests` true at level 2, then off again), so [logged_outputs.py](scripts/logged_outputs.py) paired each streamed reply with the raw text the model wrote ([logged-outputs.jsonl](gpt-oss-opencode/replay/logged-outputs.jsonl)).
  The 36 delivered calls all had the header `commentary to=functions.read<|constrain|>json` or `commentary to=functions.read <|constrain|>json`, two of them without an analysis message first.
  The four empty replies were calls with other headers: `to=functions.read json`, once after analysis and once as the whole reply, `to=functions.read<|constrain|>final code`, and `to=functions.glpjson`.
- Non-streamed, with `return_token_ids`, 106 of 120 replies held a call, 9 were empty and 5 put raw Harmony text in `content`; [rawparse.py](scripts/rawparse.py) shows each one's raw text ([plain.jsonl](gpt-oss-opencode/replay/plain.jsonl), [seeds-plain.jsonl](gpt-oss-opencode/replay/seeds-plain.jsonl)).
  Every empty one was a call written `to=functions.read json` or `to=functions.read<|channel|>commentary ...`.

(measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18)

SGLang's own parser endpoints, given fixed texts ([parser_probe.py](scripts/parser_probe.py), [parser-probe.jsonl](gpt-oss-opencode/replay/parser-probe.jsonl)), show where such a call goes: the `gpt-oss` reasoning parser passes it on after the analysis, and the `gpt-oss` tool parser returns `read` for the canonical headers but no call and no text for `to=functions.read json`, `to=functions.read<|channel|>commentary <|constrain|>json` and `to=functions.read<|channel|>commentary json` (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
The same tool parser returns as text a call written in the analysis channel and a call that comes without an analysis message first, even with a canonical header; the five non-streamed replies with raw Harmony `content` are two calls of the first kind and three of the second.
These endpoints run the non-streamed parsers; for the streamed path the logged replies show the loss of the malformed headers, while both call-first canonical calls there were delivered.

The cause, by part:

- The model: gpt-oss-20b at the recipe's temperature 1.0 sometimes writes the tool-call header in a non-canonical form; 12 of 160 streamed replays of this turn came back empty, and all four whose raw text was logged held such a header (measured on the replays above).
- SGLang's `gpt-oss` tool parser: it drops such a call entirely instead of returning it as a call or as text, so the streamed reply is empty with `finish_reason` `stop`, while the `gpt-oss` reasoning parser passes the call on (measured for the non-streamed parsers with the parser endpoints; for the streamed replies the loss is measured and its place in the tool parser inferred).
- OpenCode: it takes a reply with no text, no tool call and `finish_reason` `stop` as the end of the session and does not retry, as record 482 shows (inferred from the record).

Record 482's own raw output was not logged, so which malformed header it held is inferred from its shape, which matches the replays that wrote a malformed call as the whole reply.
That the e2e's sessions on llama.cpp and vLLM passed is one run each; whether their parsers accept these headers was not measured here.
`./llmctl down` stopped the server at 03:50:47 ([down.txt](gpt-oss-opencode/down.txt)).

## Deviations from the brief

- Check 2 added `--standalone --archived --auto --format json` and the e2e driver's `opencode.json` (bash, external directories, web, task and question denied) to the brief's command.
  Without `--standalone`, `opencode run` joins the owner's background service, which had run for two days and could hold a config older than the `sglang-docker` provider; `--archived` keeps the session out of the owner's session lists, the JSON events give each step's tokens, and the permission file keeps a non-interactive run from stopping on a question.
  The model was still chosen on the command line, and no client file changed.
- Check 2 also ran two traced diagnostic sessions and two direct oversized requests, to find out what the two HTTP 400s were.
- Check 4 ran each session under strace, tracing only `sendto`, `recvfrom`, `sendmsg` and `recvmsg`, so the requests could be replayed; the driver had run the whole e2e in one heavy scope, put `$HOME/.kimi-code/bin` on `PATH` twice and run the session after the long-grammar requests, while these runs each had their own heavy scope, the folder once on `PATH` and a fresh server.
- No check 4 run failed, so the brief's replay was not required; it was done anyway, 280 times, with SGLang's tokenize, detokenize and parser endpoints and, for 40 replays, its request log switched on and off through `/configure_logging`.
- `replay.py` first read the output token ids from the wrong field, so the first plain batch's raw texts were filled in afterwards by `rawparse.py`; the `seed` sent by `seedscan.py` did not make streamed and non-streamed replies repeat each other (3 of 80 pairs matched in length), so the two seed scans count as independent samples.
- Before the traced runs, a test of OpenCode with a recording proxy found that OpenCode v2.3.0 ignores a provider `baseURL` from `OPENCODE_CONFIG_CONTENT` or a project `opencode.json`, so the proxy was not used.
- A first strace test that also traced `read` caught OpenCode reading the owner's OpenCode config; 160 characters of that line (the schema line and the start of the MCP block) were printed once, the trace was deleted, and every later trace held socket calls only.

## End state

`./llmctl status` at 03:51 UTC lists no running model: `gpt-oss-20b on sglang`, `deepseek-r1-0528-qwen3-8b on vllm` and `qwen3.6-35b-a3b on llamacpp`, each exited.
The client lines are as before the checks, OpenCode's default `llamacpp-docker/qwen3.6-35b-a3b` and Hermes, Pi and dsh on `qwen3.6-35b-a3b` on llama.cpp, with the four warnings that no model is running.
`docker ps -a` shows `sglang`, `vllm` and `llamacpp` exited, the two `*-measured-2026-11-16` containers untouched, and the GPU holds 112 MiB.
