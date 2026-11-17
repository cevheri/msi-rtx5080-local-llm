# SGLang v0.5.23 measurement A: sampling, grammar ordering, readiness and cold start

Last verified: 2026-11-17.

This folder holds measurement A for the lead who adds SGLang as llmctl's third runtime: M8 (sampling, LD8), M6 (grammar warm-up, LD6), M7 (readiness, LD5) and M15 (cold start and network, LD15), as the brief `$SCRATCH/prompts/sglang-measure-a.md` asks.
Every start used `lmsysorg/sglang:v0.5.23-runtime` (image ID `3ce6da3e5ea0`), one container at a time named `sglang-measure`, `--device nvidia.com/gpu=all --ipc=host --memory 32g --memory-swap 32g`, `-p 127.0.0.1:30000:30000` (none for the network-none start), the Hugging Face cache mounted read-only, `HF_HUB_OFFLINE=1`, `HF_MODULES_CACHE=/tmp/hf_modules`, `MAX_JOBS=2` and no restart policy.
The settings are the hands-on lane's working ones: Qwen3.5-9B at a3, K2 Horizon 7B at a2 and gpt-oss-20b at a4 with Paged Experts (`$SCRATCH/sglang/handson/cmds/`); `sglang-research-cache` held `/root/.cache/sglang` in every start except the two M15 starts.
`nvidia-smi` read an enforced power limit of 145 W before the runs ([logs/llmctl-down.txt](logs/llmctl-down.txt)).
Labels: "(measured on the MSI Raider 18 Max HX, SGLang v0.5.23)" for runs here, "(inferred from <what>)" for readings of code and logs, "(estimate)" for arithmetic.
"Free" means `nvidia-smi` free memory of the whole GPU unless a line says the server's own kernel-load line ("free device mem") or `startup_available`.

## Answers for the design

Sampling: no v0.5.23 flag sets the defaults of OpenAI chat requests, but a model overlay does.
`--preferred-sampling-params` is merged under the request's sampling fields, and the chat layer always fills temperature, top_p, top_k, min_p and repetition_penalty itself, so a request without sampling fields ran at 1.0, 1.0 and -1 with `{"temperature":0.6,"top_p":0.95,"top_k":20}`, and five identical requests with `{"top_k":1}` gave five different replies (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).
A read-only overlay at `/llmctl/model` (links to the pinned snapshot plus a real `generation_config.json`) started Qwen3.5-9B with a3's weights, pool, free memory and parsers, logged `Using default chat sampling params from model generation config: {'temperature': 0.6, 'top_k': 20, 'top_p': 0.95}`, ran field-less requests at 0.6, 0.95 and 20, and with `top_k` 1 gave five identical replies; K2 Horizon 7B with `--trust-remote-code` started from an overlay and served a chat request and a tool call (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).

Grammar warm-up: at the working settings order 1 passed on every model, so the warm-up is insurance, not a fix, and the lead decides.
The first grammar-constrained request took 602 to 608 MiB on all three models and left 47 MiB free on Qwen3.5-9B, 34 MiB on K2 Horizon 7B and 468 MiB on gpt-oss-20b after a fresh 32k-token prompt; a 120k-token first prompt on Qwen also passed with 45 MiB left (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).
The tiny warm-up took 604 to 606 MiB and every later request passed, on Qwen also after a 120k-token prompt with 45 MiB free, so it holds, but it adds no headroom: both orders end with the same 34 to 49 MiB on Qwen and K2 (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).
Threshold: the first grammar request needs about 0.6 GiB free at that moment and a 32k to 120k prefill first grows PyTorch's cache by 304 to 388 MiB, so without a warm-up about 1.0 GiB must be free right after ready (estimate from the measured margins); `SYNC_TOKEN_IDS_ACROSS_TP=1` plus `NCCL_MAX_P2P_NCHANNELS=1` moved the allocation before ready and cut it to about 100 MiB, leaving 549 MiB after a 120k prompt and both grammar requests on Qwen (measured on the MSI Raider 18 Max HX, SGLang v0.5.23; Qwen only).

Readiness: `/ready` answering 200 is the signal, with the served name in `/v1/models`.
`/ready` and `/health` turned 200 at the same 0.5 s poll in all 15 polled starts, `/ready` answers in about 1 ms and `/health` in 1.0 s, and `/v1/models`, `/get_server_info` and `/server_info` answered 200 1.0 to 2.0 s before `/ready` in warm starts and 13 s before it in the cold start (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).
Both test `ServerStatus.Up`, which the server sets after its own warm-up request, and `/health` also sleeps a second and generates one token (inferred from `src/entrypoints/http_server.py:714-786`, `2490-2502`).
Read the memory from `/server_info`: v0.5.23 marks `/get_server_info` deprecated and logs a warning per call, and both carry `internal_states[0].memory_usage` (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).
On Qwen3.5-9B `startup_available` read 1.35 GiB but only 1,039 MiB were free after ready and a cache flush, 0.34 GiB less, while K2 Horizon 7B and gpt-oss-20b lost 40 and 56 MiB (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).
`startup_available` is logged before the server's own warm-up, which for Qwen3.5-9B is an image request because the vision tower loads, so free memory read after ready is the safer input for a threshold (inferred from `src/entrypoints/http_server.py:2402` and the figures above).

Exact values for the design (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, unless labelled):

- Overlay: mount the folder read-only at `/llmctl/model` and pass `--model-path /llmctl/model`; every snapshot file is a link to `/root/.cache/huggingface/hub/models--<org>--<name>/snapshots/<revision>/<file>` (subfolders recreated as folders of links), and `generation_config.json` is the snapshot's own file with the sampling keys added or replaced.
- SGLang reads only `repetition_penalty`, `temperature`, `top_k`, `top_p` and `min_p` from that file, and only while `--sampling-defaults` stays `model`, its default, so llmctl must refuse `--sampling-defaults openai`; a request's own fields still win (inferred from `src/configs/model_config.py:2210-2240`, `src/entrypoints/openai/protocol.py:1185-1222`).
- Files written: Qwen3.5-9B `{"_from_model_config": true, "eos_token_id": 248044, "transformers_version": "5.21.0", "use_cache": true, "temperature": 0.6, "top_p": 0.95, "top_k": 20}`; K2 Horizon 7B `{"bos_token_id": 0, "eos_token_id": [1, 250019], "transformers_version": "5.18.0.dev0", "temperature": 1.0, "top_p": 0.95}`.
- Warm-up body (order 2): `{"model": "<served>", "messages": [{"role": "user", "content": "Reply with a JSON object."}], "response_format": {"type": "json_schema", "json_schema": {"name": "w", "schema": {"type": "object"}}}, "max_tokens": 8}`; it returned 200 with `finish_reason` `length` on all three models.
- Server-side alternative: `SYNC_TOKEN_IDS_ACROSS_TP=1` (the NCCL communicator is created by the server's own warm-up before ready) and `NCCL_MAX_P2P_NCHANNELS=1` (the one-rank allocation falls from about 605 to about 100 MiB); decode 92.9 and 92.8 tok/s against 92.9 tok/s in the hands-on a3 run; measured on Qwen3.5-9B only, so K2 and gpt-oss need one check each.
- Readiness: `GET /ready` 200, then `GET /v1/models` listing the served name, then `GET /server_info` for `internal_states[0].memory_usage.token_capacity` (at least the context) and the free memory.
- `memory_usage` keys exactly as returned: `weight`, `kvcache`, `startup_available`, `token_capacity`, `token_capacity_swa`, `graph` (with `prefill`, `decode`, `target_verify`, `draft_prefill`, `draft_decode`, `draft_extend`); `max_total_num_tokens` is a top-level key only, equal to `token_capacity`; the top-level `status` read `ready`.

| Model and settings | `startup_available` (GiB) | `token_capacity` | Free after ready (MiB) | Free after `/flush_cache` (MiB) | First grammar request (MiB) | Left after the first grammar request (MiB) |
|---|---|---|---|---|---|---|
| Qwen3.5-9B a3 | 1.35 | 140,976 | 1,013 | 1,039 | 608 | 47 (32k prompt), 45 (120k prompt) |
| Qwen3.5-9B a3 + `SYNC_TOKEN_IDS_ACROSS_TP=1` | 1.35 | 140,976 | 407 | 433 | before ready | 45 (120k prompt) |
| Qwen3.5-9B a3 + both variables | 1.35 | 140,976 | 911 | 937 | before ready, about 100 | 549 (120k prompt) |
| K2 Horizon 7B a2 | 0.977 | 82,394 | 962 | 960 | 602 | 34 (32k prompt) |
| gpt-oss-20b a4 | 1.447 | 168,443 (`token_capacity_swa` 16,844) | 1,406 | 1,426 | 602 | 468 (32k prompt) |

All figures in this table are measured on the MSI Raider 18 Max HX, SGLang v0.5.23 ([logs/summary.txt](logs/summary.txt), `shapes/<model>/m6-*.jsonl`, `shapes/<model>/m6-o1-m7.json`).

## Starts

All rows are measured on the MSI Raider 18 Max HX, SGLang v0.5.23; "ready" is the "The server is fired up and ready to roll!" line in seconds after the `docker run` call (1 s resolution), from [logs/summary.txt](logs/summary.txt).

| Run | Model | Change against the working settings | Ready | Result |
|---|---|---|---|---|
| m8-s1-logreq | Qwen3.5-9B | `--log-requests --log-requests-level 2` | 28.3 s | Field-less chat requests at 1.0, 1.0, -1; five distinct replies |
| m8-s2-preferred | Qwen3.5-9B | as s1 plus `--preferred-sampling-params '{"temperature":0.6,"top_p":0.95,"top_k":20}'` | 27.6 s | Still 1.0, 1.0, -1; five distinct replies |
| m8-s3-preferred-topk1 | Qwen3.5-9B | as s1 plus `--preferred-sampling-params '{"top_k":1}'` | 27.4 s | Still -1; five distinct replies; `/model_info` lists `{"top_k": 1}` |
| m8-s4-overlay | Qwen3.5-9B | as s1 from the overlay (0.6, 0.95, 20) | 33.3 s | Overlay values logged and applied |
| m8-s5-overlay-topk1 | Qwen3.5-9B | as s4 with `top_k` 1 | 26.8 s | Five identical replies |
| m8-s6-overlay | K2 Horizon 7B | log requests, overlay (1.0, 0.95) | 35.0 s | Chat and tool call served, overlay values applied |
| m6-o1 | Qwen3.5-9B | none (M7 poll) | 26.9 s | Order 1 passed |
| m6-o2 | Qwen3.5-9B | none | 26.8 s | Order 2 passed, also the 120k phase |
| m6-o1 | K2 Horizon 7B | none (M7 poll) | 33.9 s | Order 1 passed |
| m6-o2 | K2 Horizon 7B | none | 35.8 s | Order 2 passed |
| m6-o1 | gpt-oss-20b | none (M7 poll) | 106.2 s | Order 1 passed |
| m6-o2 | gpt-oss-20b | none | 115.8 s | Order 2 passed |
| m6-e1-sync | Qwen3.5-9B | `-e SYNC_TOKEN_IDS_ACROSS_TP=1`, 120k first prompt | 28.7 s | Passed; NCCL allocated before ready |
| m6-e2-sync-p2p1 | Qwen3.5-9B | as e1 plus `-e NCCL_MAX_P2P_NCHANNELS=1` | 28.1 s | Passed with about 100 MiB for NCCL |
| m15-cold | Qwen3.5-9B | new empty `sglang-measure-cold`, `FLASHINFER_NO_DOWNLOAD=1`; then order 1 with a 120k first prompt | 46.2 s | Ready, no download line; order 1 passed |
| m15-netnone | Qwen3.5-9B | as m15-cold on a recreated empty volume, `--network none`, no port | 46.6 s | Ready, chat answered through `docker exec` |

Besides the server starts, four kinds of throwaway containers ran: the NCCL probe (GPU, `--network none`), the harmony check (no GPU, `--network none`), offline source reads and volume listings (`--rm --network none --memory 2g --memory-swap 2g`).

## M8: sampling

Request logging: `--log-requests-level` 1 is the lowest level that logs sampling parameters and 2 also logs the raw OpenAI request (`src/arg_groups/fields/observability.py:40-46`, `src/entrypoints/openai/serving_base.py:96`), so level 2 was used (inferred from the sources).
No level logs the final merged `SamplingParams`: the `Receive:` line shows `GenerateReqInput.sampling_params` before `_create_tokenized_object` merges the preferred values under it with `{**self.preferred_sampling_params, **obj.sampling_params}` (`src/managers/tokenizer_manager.py:1549-1550`) (inferred from the sources).
The chat layer fills five fields on every request through `get_param`, "Priority: user value > model generation_config > OpenAI defaults", with the defaults temperature 1.0, top_p 1.0, top_k -1, min_p 0.0 and repetition_penalty 1.0 (`src/entrypoints/openai/protocol.py:1062-1068`, `1185-1222`), so a preferred value for those fields can never win in a chat request (inferred from the sources).

Step 1, a3 plus request logging ([logs/qwen3.5-9b/m8-s1-logreq.reqlog.txt](../../sglang-qwen3.5-9b/measure-a/logs/m8-s1-logreq.reqlog.txt)): the raw request carried no sampling field (the `Receive OpenAI` line lists `messages`, `model`, `max_tokens` and the non-null defaults that the request model adds: `frequency_penalty` 0.0, `presence_penalty` 0.0, `stream` and `tool_choice`), and the server built `'temperature': 1.0, ... 'top_p': 1.0, 'top_k': -1, 'min_p': 0.0, ... 'repetition_penalty': 1.0` for it (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).
Five identical requests without sampling fields gave five different replies ([shapes/qwen3.5-9b/m8-s1-logreq.jsonl](../../sglang-qwen3.5-9b/measure-a/shapes/m8-s1-logreq.jsonl)) (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).

Step 2, `--preferred-sampling-params '{"temperature":0.6,"top_p":0.95,"top_k":20}'`: the log line of every field-less request still showed temperature 1.0, top_p 1.0 and top_k -1, and five identical requests gave five different replies ([logs/qwen3.5-9b/m8-s2-preferred.reqlog.txt](../../sglang-qwen3.5-9b/measure-a/logs/m8-s2-preferred.reqlog.txt), [shapes/qwen3.5-9b/m8-s2-preferred.jsonl](../../sglang-qwen3.5-9b/measure-a/shapes/m8-s2-preferred.jsonl)) (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).
With `'{"top_k":1}'`, `/model_info` returned `"preferred_sampling_params":{"top_k":1}`, the log lines showed top_k -1, and five identical requests at the default temperature gave five different replies, so the preferred value did not apply ([shapes/qwen3.5-9b/m8-s3-model-info.json](../../sglang-qwen3.5-9b/measure-a/shapes/m8-s3-model-info.json), [shapes/qwen3.5-9b/m8-s3-preferred-topk1.jsonl](../../sglang-qwen3.5-9b/measure-a/shapes/m8-s3-preferred-topk1.jsonl)) (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).

Step 3, the overlay (`overlay/qwen3.5-9b/`, made by [scripts/make_overlay.py](scripts/make_overlay.py), 11 links and one real file):
- Same as a3: `Load weight end ... mem usage=8.39 GB`, `#tokens: 140976`, `available_gpu_mem=1.35 GB`, `Auto-detected template features: ... reasoning_parser=qwen3, tool_call_parser=qwen3_coder`, content format `openai` ([logs/qwen3.5-9b/m8-s4-overlay.log](../../sglang-qwen3.5-9b/measure-a/logs/m8-s4-overlay.log)) (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).
- The start logged `Using default chat sampling params from model generation config: {'temperature': 0.6, 'top_k': 20, 'top_p': 0.95}`, and every field-less request was built with `'temperature': 0.6`, `'top_p': 0.95`, `'top_k': 20` ([logs/qwen3.5-9b/m8-s4-overlay.reqlog.txt](../../sglang-qwen3.5-9b/measure-a/logs/m8-s4-overlay.reqlog.txt)) (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).
- With `top_k` 1 in the overlay (`overlay/qwen3.5-9b-topk1/`) the log showed `{'temperature': 0.6, 'top_k': 1, 'top_p': 0.95}` and five identical requests gave one reply, sha256 prefix `649f17c5e7e8783d` five times ([shapes/qwen3.5-9b/m8-s5-overlay-topk1.jsonl](../../sglang-qwen3.5-9b/measure-a/shapes/m8-s5-overlay-topk1.jsonl)) (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).
- `/model_info` then reports `model_path` and `tokenizer_path` as `/llmctl/model` ([shapes/qwen3.5-9b/m8-s4-model-info.json](../../sglang-qwen3.5-9b/measure-a/shapes/m8-s4-model-info.json)); `/v1/models` still lists the served name (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).
- The server's own start-up warm-up, which for this vision model is a chat request with an image at temperature 0 (`src/entrypoints/http_server.py:2402`), also took the overlay's top_p and top_k (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).
- No transformers warning about `do_sample` or the newer `transformers_version` (5.21.0 in the file, 5.19.0 in the image) appeared in the cut log, whose patterns include `do_sample`, `GenerationConfig` and `UserWarning` (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).

K2 Horizon 7B from an overlay (`overlay/k2-horizon-7b/`, 16 links including the remote-code files `configuration_k2_horizon.py` and `modeling_k2_horizon.py` and the `assets` folder) at a2 with `--trust-remote-code`:
- Same as a2: `type=K2HorizonForCausalLM`, `mem usage=7.89 GB`, FP8 pool `#tokens: 82394`, `available_gpu_mem=0.98 GB`, `k2_horizon` parsers, content format `string` ([logs/k2-horizon-7b/m8-s6-overlay.log](../../sglang-k2-horizon-7b/measure-a/logs/m8-s6-overlay.log)) (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).
- It logged `{'temperature': 1.0, 'top_p': 0.95}`, a chat request returned 200 with `391` and 1,395 characters of reasoning, and a tool call returned `get_weather` with `{"city": "Paris"}` and `finish_reason` `tool_calls` ([shapes/k2-horizon-7b/m8-s6-overlay.jsonl](../../sglang-k2-horizon-7b/measure-a/shapes/m8-s6-overlay.jsonl), [logs/k2-horizon-7b/m8-s6-overlay.reqlog.txt](../../sglang-k2-horizon-7b/measure-a/logs/m8-s6-overlay.reqlog.txt)) (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).

Step 4, which mechanism works and where the overlay breaks:
- Only the overlay pins the sampling of chat requests; `--preferred-sampling-params` fills only fields that reach the tokenizer manager unset, as in native `/generate` requests, while chat and completion requests arrive with temperature set (inferred from `src/managers/tokenizer_manager.py:1549-1550`, `src/entrypoints/openai/protocol.py:386-397` and the step 2 runs).
- llmctl must write, per start: the folder of links from the pinned snapshot's file list, the edited `generation_config.json` (atomically; [scripts/make_overlay.py](scripts/make_overlay.py) writes a temporary file and renames it), the read-only mount and `--model-path /llmctl/model` (inferred from the runs).
- A file added to the snapshot later (a wider `pull` into the same revision) is missing from an overlay written before it, so llmctl should rebuild the folder on every `up` from the current listing rather than keep it (inferred).
- A snapshot without `generation_config.json` is fine for SGLang, whose loader treats a missing file as no defaults (`src/utils/hf_transformers_utils.py` re-exports `get_generation_config`, which catches `OSError` in the image's `hf_transformers/common.py:717-723`), so llmctl can write a file holding only the sampling keys; not run (inferred from the source).
- A remote-code model works from the overlay (K2 Horizon 7B above), because the code files are links that resolve inside the container; the module name then comes from the folder name `model` (measured for K2; module naming inferred).
- The links resolve only inside a container that mounts the cache at `/root/.cache/huggingface`; on the host they dangle, so host-side checks must look at the snapshot, not the overlay (inferred).
- A new revision changes the snapshot path, so the overlay must be rebuilt when a recipe's revision changes (inferred).
- Sampling fields that SGLang does not read from the file (for example `presence_penalty`) cannot be pinned this way (inferred from `src/configs/model_config.py:2228-2234`).

## M6: grammar ordering

Protocol: right after the M7 poll found all endpoints at 200 (one `/health` generation and a few reads), `POST /flush_cache`, a fresh prompt of about 32k tokens (a random session line, then 314 to 335 copies of a code paragraph, `max_tokens` 32), then a Hermes-shaped title request (system and user message, `reasoning_effort: "none"`, `response_format` `json_schema` `session_title` with `strict: true`), then `tool_choice: "required"` with one tool; order 2 sent the warm-up first ([scripts/m6.py](scripts/m6.py)).
`nvidia-smi` was read before and after each request, and a 1 s trace of GPU and cgroup memory ran for every start (`logs/<model>/<run>-trace.csv`).
All figures below are measured on the MSI Raider 18 Max HX, SGLang v0.5.23 ([logs/summary.txt](logs/summary.txt), `shapes/<model>/m6-*.jsonl`, the `device-loaded` lines in `logs/<model>/m6-*.full.log`).

| Model | Order | Free at start | Warm-up | Long prompt (tokens) | Title (first grammar in order 1) | Forced `tool_choice` | Kernel line at the first grammar request |
|---|---|---|---|---|---|---|---|
| Qwen3.5-9B a3 | 1 | 1,013 | none | 32,087: 1,039 to 655 | 655 to 47, title parses | 200, `get_weather` | 0.64 GiB |
| Qwen3.5-9B a3 | 2 | 1,013 | 1,013 to 407 | 32,087: 433 to 49 | 49 to 47, title parses | 200, `get_weather` | 0.99 GiB (warm-up) |
| Qwen3.5-9B a3, second phase of order 2 | 2 | 47 | done | 119,913: 433 to 45 | 45 to 45, title parses | 200, `get_weather` | already loaded |
| Qwen3.5-9B a3, cold volume | 1 | 709 | none | 119,911: 1,039 to 653 | 653 to 45, title parses | 200, `get_weather` | 0.64 GiB |
| K2 Horizon 7B a2 | 1 | 962 | none | 32,016: 960 to 636 | 636 to 34, title parses | 200, `get_weather` | 0.62 GiB |
| K2 Horizon 7B a2 | 2 | 962 | 962 to 358 | 32,021: 360 to 34 | 34 to 34, title parses | 200, `get_weather` | 0.94 GiB (warm-up) |
| gpt-oss-20b a4 | 1 | 1,406 | none | 31,929: 1,426 to 1,070 | 1,070 to 468, title parses | 200, `finish_reason` `stop`, no `tool_calls` | not logged |
| gpt-oss-20b a4 | 2 | 1,406 | 1,406 to 800 | 31,933: 822 to 518 | 518 to 518, title parses | 200, `finish_reason` `stop`, no `tool_calls` | not logged |

Memory columns are `nvidia-smi` free MiB before and after each request; every request returned HTTP 200 and every server stayed up.
gpt-oss-20b's forced tool choice came back as raw Harmony text in `content`, the known #41747 behaviour, and the server logged `Required tool call missing from gpt-oss output (98 chars)` (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).

What the numbers say:
- The first grammar request, or the warm-up, took 602 to 608 MiB on every model (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).
- A chunked prefill of 32k to 120k tokens grew PyTorch's cache by 304 to 388 MiB from the flushed level, about the same for 32k and 120k on Qwen (384 and 386 MiB) (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).
- `/flush_cache` returns PyTorch's cached memory to the device: free rose from 47 to 433 MiB before the 120k phase of Qwen's order 2 and from 709 to 1,039 MiB right after the cold start, so the order-1 margins above were measured with an emptied cache, which real sessions do not do (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).
- Order 1 passed on every model at these settings; the warm-up may still be cheap insurance, and the lead decides (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).
- The warm-up protects later grammar requests but does not add headroom, because the same NCCL buffer and the same prefill cache are allocated in either order; Qwen ended at 45 to 49 MiB and K2 at 34 MiB both ways (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).
- If the total does not fit, order 1 fails in NCCL's allocation at the first grammar request (the hands-on crashes) and order 2 would fail in PyTorch during a later prefill instead, which was not provoked here (inferred).

Threshold (step 5):
- At the moment of the first grammar request, crashes happened at 0.29 to 0.50 GiB free by the kernel line in the hands-on runs, and passes here at 0.62 and 0.64 GiB, with 602 to 608 MiB taken (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, hands-on figures from research-report section 3.4).
- At ready, without a warm-up, the free memory after a cache flush must cover about 600 MiB plus the prefill growth: about 996 MiB for Qwen3.5-9B (608 + 388), 928 MiB for K2 Horizon 7B (602 + 326) and 958 MiB for gpt-oss-20b (602 + 356), so about 1.0 GiB (estimate from the measured figures).
- In `startup_available` terms that is about 1.31 GiB for Qwen3.5-9B, of which 343 MiB is no longer free after ready even after a cache flush (1.35 GiB logged, 1,039 MiB free; that the image warm-up holds it is inferred), and about 0.95 GiB for K2 and 0.99 GiB for gpt-oss (estimate).
- The warm-up held with the memory left after a 120k-token prompt on Qwen3.5-9B: after it, the 119,913-token prompt left 45 MiB and the title and forced tool requests passed (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).

Step 4, variables and flags (sources in `src/`, read offline; probe in [logs/nccl-probe/](logs/nccl-probe/)):
- The sampler runs `torch.distributed.all_reduce(..., op=MIN, group=self.tp_sync_group)` when `SYNC_TOKEN_IDS_ACROSS_TP or sampling_info.grammars` (`src/layers/sampler.py:81`, `715-730`), so no variable or flag of v0.5.23 skips it while a grammar is active; the variable can only force it on for every batch (inferred from the source).
- For this single-node start the engine sets `NCCL_CUMEM_ENABLE=0` and `NCCL_NVLS_ENABLE=0` (`src/entrypoints/engine.py:1898-1915`), and no SGLang flag sizes NCCL's buffers (inferred from the sources).
- A one-rank probe of the sampler's call in the image ([scripts/nccl_probe.py](scripts/nccl_probe.py), [cmds/nccl-probe.sh](cmds/nccl-probe.sh)) took 672 MiB on the first `all_reduce`, of which `transport/net.cc:675 (sharedNetBuffersInit) Cuda Calloc Size 536870912` with `64 p2p channels` and `P2P Chunksize set to 524288`, which fits 64 x 8 steps x 512 KiB x 2 (measured on the MSI Raider 18 Max HX, NCCL 2.30.7 in the image; the decomposition is inferred).
- `NCCL_MAX_P2P_NCHANNELS=1` or `NCCL_MAX_NCHANNELS=1` cut that buffer to 8 MiB and the first `all_reduce` to 168 MiB, while `NCCL_NET_SHARED_BUFFERS=0` and `NCCL_P2P_NET_CHUNKSIZE=65536` changed nothing (measured on the MSI Raider 18 Max HX, NCCL 2.30.7).
- Extra start 1, `SYNC_TOKEN_IDS_ACROSS_TP=1` on Qwen3.5-9B a3: 407 MiB free after ready instead of 1,013 (the kernel lines after ready read 0.42 GiB instead of 0.99), so the server's own warm-up allocated the buffer before ready; a 119,911-token first prompt left 47 MiB, the title and forced tool requests passed, and decode ran at 92.9 tok/s on a 512-token answer, the hands-on a3 figure (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).
- Extra start 2, both variables: 911 MiB free after ready, so the allocation before ready was about 100 MiB; a 119,912-token first prompt left 551 MiB, both grammar requests passed with 549 MiB left, decode 92.8 tok/s (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).
- With both variables a start that cannot afford the allocation should fail before ready instead of in a session, because the server's warm-up request fails and the server stops (inferred from `src/entrypoints/http_server.py:2490-2502`, `2531-2535`; not provoked).

## M7: readiness

One poller thread per endpoint, every 0.5 s from the `docker run` call until all four required endpoints had answered 200 ([scripts/poll_ready.py](scripts/poll_ready.py), `shapes/<model>/<run>-m7.json`); `/server_info` was polled as well because v0.5.23 names it as the replacement.
The figures below are measured on the MSI Raider 18 Max HX, SGLang v0.5.23, on the M6 order-1 start of each model and on the cold start; all 15 polled starts are in [logs/summary.txt](logs/summary.txt).

| Model | Ready line | `/v1/models`, `/get_server_info`, `/server_info` first 200 | `/health` and `/ready` first 200 | `/ready` 200 after `/v1/models` 200 |
|---|---|---|---|---|
| Qwen3.5-9B a3 | 26.9 s | 26.05 s | 28.05 s | 2.0 s |
| K2 Horizon 7B a2 | 33.9 s | 33.55 s | 34.55 s | 1.0 s |
| gpt-oss-20b a4 | 106.2 s | 106.06 s | 107.06 s | 1.0 s |
| Qwen3.5-9B a3, cold volume | 46.2 s | 34.55 s | 47.55 s | 13.0 s |

- Before 200 every endpoint answered `ConnectionRefusedError` for about 0.5 s (nothing listening), then `ConnectionResetError` or `RemoteDisconnected` (Docker's port proxy accepted, nothing behind it) until the HTTP server listened; from then `/health` and `/ready` answered 503 until ready, while `/v1/models`, `/get_server_info` and `/server_info` answered 200 at once (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).
- `/health` and `/ready` turned 200 at the same poll in all 15 polled starts, and the other three endpoints 1.0 to 2.0 s earlier in warm starts (measured on the MSI Raider 18 Max HX, SGLang v0.5.23); both test `ServerStatus.Up`, which the server sets after its own warm-up request (inferred from `src/entrypoints/http_server.py:714-735`, `2490-2502`).
- One call once up: `/health` 1.001 to 1.002 s (it sleeps one second and generates one token, `src/entrypoints/http_server.py:751-786`), `/ready` 0.000 to 0.001 s, `/v1/models` up to 0.001 s, `/server_info` 0.005 to 0.006 s, `/get_server_info` 0.004 to 0.006 s (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).
- Every `/get_server_info` call logs `Endpoint '/get_server_info' is deprecated and will be removed in a future version. Please use '/server_info' instead.` (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).
- At ready `internal_states[0].memory_usage` held, for Qwen3.5-9B, `{'weight': 8.391, 'kvcache': 4.302, 'startup_available': 1.35, 'token_capacity': 140976, 'token_capacity_swa': None, 'graph': {'prefill': 0.0, 'decode': 0.096, ...}}`, for K2 `weight` 7.887, `kvcache` 5.658, `startup_available` 0.977, `token_capacity` 82394, and for gpt-oss `weight` 11.455, `kvcache` 2.12, `startup_available` 1.447, `token_capacity` 168443, `token_capacity_swa` 16844 (measured on the MSI Raider 18 Max HX, SGLang v0.5.23; [shapes/qwen3.5-9b/m6-o1-m7-server-info.json](../../sglang-qwen3.5-9b/measure-a/shapes/m6-o1-m7-server-info.json) is a full body).
- `max_total_num_tokens` is only a top-level key (140976, 82394, 168443), `internal_states[0]` repeats the server arguments plus `memory_usage`, `last_gen_throughput`, `world_size`, `effective_max_running_requests_per_dp`, `decode_cuda_graph_bs` and `decode_cuda_graph_memory_gb`, and the top-level `status` read `ready` (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).
- Scheduler death: not observed, because no grammar request crashed in this lane.
- The hands-on K2 a1 crash shows what happens: GPU memory fell to 114 MiB within a second, the request got 500 after 5 s, and the container's cgroup was gone 6 s after the crash, after `SIGQUIT received` and `kill_process_tree called: parent_pid=1, include_parent=True`, so the container exits instead of staying up with a dead scheduler; what the endpoints answered in those seconds was not recorded (measured by the hands-on lane, re-read from `$SCRATCH/sglang/handson/logs/k2-horizon-7b/trace.csv` and `attempt1.log`).

## M15: cold start and network

| Start | Cache volume | Network | Ready line | `/health` 200 | Lines about downloading or fetching |
|---|---|---|---|---|---|
| m15-cold | new, empty `sglang-measure-cold`, `FLASHINFER_NO_DOWNLOAD=1` | bridge, 127.0.0.1:30000 | 46.2 s | 47.55 s | none |
| m15-netnone | recreated empty, `FLASHINFER_NO_DOWNLOAD=1` | `--network none`, only `lo` | 46.6 s | 47.26 s (`docker exec` poll every 1 s) | none |

Both rows are measured on the MSI Raider 18 Max HX, SGLang v0.5.23 ([logs/qwen3.5-9b/m15-cold.full.log](../../sglang-qwen3.5-9b/measure-a/logs/m15-cold.full.log), [logs/qwen3.5-9b/m15-netnone.full.log](../../sglang-qwen3.5-9b/measure-a/logs/m15-netnone.full.log), [shapes/qwen3.5-9b/m15-netnone-exec-poll.txt](../../sglang-qwen3.5-9b/measure-a/shapes/m15-netnone-exec-poll.txt)).
- Warm Qwen3.5-9B starts at a3 reached the ready line in 26.8 to 28.7 s (33.3 s for the first overlay start); the cold starts spent 10.34 and 10.05 s capturing the decode graph instead of 2.27 to 2.28 s and 13 s between the pool line and ready instead of 2 s (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).
- A case-insensitive search of both full logs for `download`, `fetch`, `urm.nvidia`, `huggingface.co` and `http` found only the image's NGC licence banner URL, the `'download_dir': None` field of `server_args`, the `Uvicorn running on http://0.0.0.0:30000` line, a FastAPI deprecation warning naming `http_server.py`, and the access-log lines (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).
- `FLASHINFER_NO_DOWNLOAD=1` raises instead of downloading a missing artifact (`get_artifact` in the image's `flashinfer/jit/cubin_loader.py:221-226`), and neither start raised, so the image's cubins and JIT cache covered Qwen3.5-9B (measured; mechanism inferred from the source).
- The network-none start served `Reply with exactly: OK` with `OK` through `docker exec`, and its only network-related lines were the warnings `The hostname of the client socket cannot be retrieved. err=-3` (c10d) and `Unable to resolve hostname to a (local) address. Using the loopback address as fallback.` (Gloo) (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).
- The cache volume held 19 MB after a cold start (421 files: `triton` 16 MB, `jit` 1.9 MB, `inductor` 684 KB, `.cache/flashinfer`, `nv`) and 397 files after the network-none start ([logs/qwen3.5-9b/m15-cold.volume-after.txt](../../sglang-qwen3.5-9b/measure-a/logs/m15-cold.volume-after.txt)); `sglang-measure-cold` was removed afterwards (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).
- After the cold start's ready, order 1 with a 119,911-token first prompt passed (table in M6) (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).
- This covers Qwen3.5-9B only: gpt-oss-20b downloads a vocabulary at every start unless `TIKTOKEN_ENCODINGS_BASE` is set (next section).

## Other findings

- gpt-oss-20b downloads its Harmony vocabulary at every start: for `model_type` `gpt_oss`, `OpenAIServingResponses.__init__` calls `get_stop_tokens_for_assistant_actions()` (`src/entrypoints/openai/serving_responses.py:213-224`), openai-harmony 0.0.4 then fetches `o200k_base.tiktoken` from `https://openaipublic.blob.core.windows.net/encodings/` into a temporary-folder cache unless `TIKTOKEN_ENCODINGS_BASE` is set (strings of its library), and the HTTP server builds that object in its start-up (`src/entrypoints/http_server.py:395-416`) (inferred from the sources).
- That start-up took 52 and 63 s in the two gpt-oss starts here ("Waiting for application startup" to "Application startup complete", with no log line in between; 52 s from `Engine startup timings` to `Uvicorn running` in m6-o1) against 6 s from `Engine startup timings` to `Uvicorn running` in the three hands-on gpt-oss starts, which put the ready line at 106.2 and 115.8 s instead of 61 s (measured on the MSI Raider 18 Max HX, SGLang v0.5.23; that the gap is the download is inferred).
- Offline, `load_harmony_encoding(HARMONY_GPT_OSS)` failed at once with `HarmonyError: error downloading or loading vocab file`, and with `TIKTOKEN_ENCODINGS_BASE=/opt/sglang/lib/python3.12/site-packages/tml_renderers/data` (a copy that ships in the image) it loaded in 0.08 s ([logs/harmony-offline/harmony-check.txt](logs/harmony-offline/harmony-check.txt), [scripts/harmony_check.py](scripts/harmony_check.py)) (measured on the MSI Raider 18 Max HX, SGLang v0.5.23 image, no GPU).
- Without the network the server would log `OpenAI Responses API (/v1/responses) disabled: ...` and keep serving chat completions, because that failure is caught (`src/entrypoints/http_server.py:406-416`); not run (inferred from the source).
- gpt-oss-20b with Paged Experts held the container's cgroup at 31.88 of 32 GiB from about 44 s after `docker run` to the end of both starts, with 4.6 GiB `anon` and 27.1 GiB `file` (page cache and shared memory, which the trace does not split), against a 28.24 GiB peak in the hands-on run; no OOM kill ([logs/gpt-oss-20b/m6-o1-trace.csv](../../sglang-gpt-oss-20b/measure-a/logs/m6-o1-trace.csv)) (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).
- GPU peaks in this lane were 15,796 to 15,808 MiB on Qwen3.5-9B and K2 Horizon 7B and 15,374 MiB on gpt-oss-20b, with power up to 148.5 W in 1 s samples (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, the trace files).

## What contradicts the research report

- Sections 1, 3.4 and 8 say the first grammar request makes NCCL allocate 512 MiB: the 512 MiB is NCCL's shared network buffer of a one-rank communicator, the whole first grammar request took 602 to 608 MiB in the server (672 MiB in a fresh process), and `NCCL_MAX_P2P_NCHANNELS=1` shrinks it to about 100 MiB (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).
- Section 3.4 expects a first grammar request after a long prompt to be the dangerous order: at the working settings order 1 passed on all three models, also after 120k tokens on Qwen, though with 34 to 49 MiB to spare on Qwen and K2 (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).
- Section 2.7 says free memory fell to 0.04 GiB during a 124,667-token prefill: in that hands-on run the grammar requests had already taken the NCCL buffer; a 120k prefill alone grew PyTorch's cache by 386 MiB (1,039 to 653 MiB free) (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).
- Section 5.4 and decision D6 take `/health` as the readiness path and `/get_server_info` for the memory check: `/ready` turns 200 at the same moment in 1 ms instead of 1.0 s, and v0.5.23 deprecates `/get_server_info` in favour of `/server_info` (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).
- Sections 2.4 and 2.7 leave `/ready` during start-up open: it answers 503 from the moment the HTTP server listens until ready, like `/health` (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).
- Sections 1, 2.5 and 8 say v0.5.23 has no server-side sampling override: the overlay is one, tested on Qwen3.5-9B and K2 Horizon 7B (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).
- Section 2.4 gives warm starts of 28 to 61 s: gpt-oss-20b a4 took 106 and 116 s here, 52 and 63 s of it in the start-up step that downloads the Harmony vocabulary, a network use that section 7 does not list (measured on the MSI Raider 18 Max HX, SGLang v0.5.23); the download as the cause is inferred.
- Section 5.4 and decision D6 put `min_free_at_ready_mib` (900) on `startup_available`: on Qwen3.5-9B 0.34 GiB of `startup_available` was no longer free after ready, against 40 and 56 MiB on K2 Horizon 7B and gpt-oss-20b, so the same threshold means different headroom per model (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).

## Start and end state

- Before: `gemma-4-26b-a4b` on llama.cpp running and ready, 15,008 MiB of GPU memory, every client on it ([logs/llmctl-status-before.txt](logs/llmctl-status-before.txt)) (measured on the MSI Raider 18 Max HX, llmctl).
- `./llmctl down` at 20:15:58 (+03:00) left 112 MiB in use ([logs/llmctl-down.txt](logs/llmctl-down.txt)) (measured on the MSI Raider 18 Max HX, llmctl and nvidia-smi).
- `PATH=$HOME/.kimi-code/bin:$PATH ./llmctl up gemma-4-26b-a4b` at 20:48:06 printed `ready` with no `wrote` line ([logs/llmctl-up.txt](logs/llmctl-up.txt)), and `./llmctl status` read running, health ready, 15,008 MiB of 16,303 MiB, and OpenCode, Hermes, Pi, dsh and Kimi on the running model ([logs/llmctl-status-after.txt](logs/llmctl-status-after.txt)) (measured on the MSI Raider 18 Max HX, llmctl).
- Left behind: no `sglang-measure` container, `sglang-measure-cold` removed, `sglang-research-cache` kept (the warm starts added their caches to it); the repository was not edited.

## Files

- `cmds/`: one `docker run` file per start, the NCCL probe, and the M15 pair; they read `$HOME` and `$SCRATCH` from the environment.
- `logs/<model>/`: per run the cut log (`<run>.log`), the full log for the M6 and M15 runs (`<run>.full.log`), the request-log extract of the M8 runs (`<run>.reqlog.txt`), the 1 s GPU and cgroup trace (`<run>-trace.csv`), the `docker run` time (`<run>.t0`), the poller and probe output, and the container state.
- [logs/nccl-probe/](logs/nccl-probe/), [logs/harmony-offline/](logs/harmony-offline/), [logs/summary.txt](logs/summary.txt) and the `llmctl` outputs.
- `shapes/<model>/`: M6 and M8 records (`*.jsonl`, replies as a sha256 prefix and at most 80 characters), M7 records (`<run>-m7.json`), one full `/get_server_info` body per model, and `/model_info` of three M8 starts.
- `overlay/`: the three overlay folders and `*.make.json`, the generation config before and after.
- [scripts/](scripts/): `run.sh` and `run_netnone.sh` drive a start; `poll_ready.py`, `m6.py`, `m8.py`, `make_overlay.py`, `nccl_probe.py`, `harmony_check.py`, `reqlog.py`, `summarize.py` and a modified `cutlog.py` (the hands-on `trace.sh` was used unchanged); run them with `python3 -I`.
- `src/`: the v0.5.23 source files read offline from the image, cited by path and line; vendor code, read as data.
