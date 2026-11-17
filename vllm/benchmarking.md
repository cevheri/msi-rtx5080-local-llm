# Benchmarking vLLM on this laptop

Last verified: 2026-11-17.

Benchmark a running vLLM server with `vllm bench serve`, run from the same `vllm/vllm-openai:v0.33.0` image with `--entrypoint vllm`, against `http://127.0.0.1:8000/v1/chat/completions`.
The numbers that matter for OpenCode on this machine are decode speed at concurrency 1 (TPOT), TTFT at 8k to 64k prompt tokens, the prefix-cache effect on a repeated system prompt, the tool-call success rate, and thinking on versus off.
The vLLM measurements on this laptop (gpt-oss-20b on 2026-11-16, seven more recipes on 2026-11-17) are in the results table at the end; every other speed figure linked from here is a third-party measurement or an estimate.

## What to measure and why

OpenCode is one user driving one agent loop, sometimes with a subagent beside it, so single-stream latency matters far more than aggregate throughput.

| Metric | Why it matters here | How to get it |
|---|---|---|
| Decode speed (TPOT, tok/s per stream) | Sets how long every tool call and answer takes to stream | Scenario S1, concurrency 1 |
| TTFT at 8k, 32k, 64k (and 120k where the context allows) | OpenCode resends its full context every turn; the fixed prompt alone is about 5.5k-7k tokens (estimate, see [OpenCode notes](../docs/opencode.md)) | Scenario S2 |
| Prefix-cache effect | A warm cache should make follow-up turns pay only for the new suffix | Scenario S3, server with and without prefix caching |
| Concurrency 2 | A subagent or a title/compaction request runs beside the main turn | Scenario S4 |
| Thinking on versus off | Thinking can eat the whole token budget and return no answer, as on the earlier DGX Spark runbook, but it can also raise agent quality | Scenario S5 |
| Tool-call success rate | A wrong parser does not error: calls arrive as plain text | Scenario S6, plus the optional SQL benchmark |
| VRAM headroom and KV capacity | 16303 MiB in total; vLLM reserves `--gpu-memory-utilization` x total up front (default 0.92, about 14.65 GiB, [cache.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/config/cache.py)) | Server startup log, `/metrics`, `nvidia-smi` |

A rough ceiling for decode speed is memory bandwidth divided by the bytes read per token: 896 GB/s ([NVIDIA spec](https://www.nvidia.com/en-us/geforce/laptops/50-series/)) divided by the weights a token touches (dense: roughly the non-embedding weights; MoE: the active experts plus shared weights).
This is an upper bound (estimate), not a prediction; real runs land below it.
Per-model weight sizes are in [models.md](models.md).

## The tool: `vllm bench serve`

`vllm bench serve` is an HTTP client that load-tests an OpenAI-compatible endpoint and reports request throughput, output token throughput, TTFT, TPOT, ITL and optionally E2EL ([vLLM benchmark CLI docs at v0.33.0](https://github.com/vllm-project/vllm/blob/v0.33.0/docs/benchmarking/cli.md)).
It ships in the server image, so no extra install is needed for the `random` and `prefix_repetition` datasets.
The image entrypoint is `vllm serve` ([Dockerfile at v0.33.0](https://github.com/vllm-project/vllm/blob/v0.33.0/docker/Dockerfile)), so a client run must override it with `--entrypoint vllm`.
This client recipe is derived from the Dockerfile and the docs and has not been run on this machine yet.

The vLLM docs say `vllm bench` is mainly for feature and regression testing and recommend [GuideLLM](https://github.com/vllm-project/guidellm) for benchmarking production servers ([cli.md](https://github.com/vllm-project/vllm/blob/v0.33.0/docs/benchmarking/cli.md)).
For a single-user laptop, `vllm bench serve` is enough and needs nothing beyond the image.
GuideLLM has not been reviewed for this knowledge base; its version and container image are unknown.

### Defaults that silently shape results

These are the v0.33.0 defaults ([serve.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/benchmarks/serve.py), [datasets.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/benchmarks/datasets/datasets.py)).
Always set prompts, concurrency, lengths and seed explicitly.

| Option | Default | What to do |
|---|---|---|
| `--backend` / `--endpoint` | `openai`, `/v1/completions` | Use `openai-chat` and `/v1/chat/completions`, so the chat template and the tool and reasoning parsers are on the measured path, as in OpenCode traffic |
| `--num-prompts` | 1000 | Set 4-16 per scenario |
| `--request-rate` | `inf` (all requests at t=0) | Leave it and cap with `--max-concurrency` |
| `--max-concurrency` | unlimited | Set 1 or 2 |
| `--random-input-len` / `--random-output-len` | 1024 / 128 | Set per scenario |
| `--random-range-ratio` | 0.0 (fixed lengths) | Leave at 0.0 |
| `--seed` | 0 | Change it on every run (see the prefix-cache warning below) |
| `--num-warmups` | 0 | Keep 0; warm up with a separate run instead (see below) |
| `--percentile-metrics` | `ttft,tpot,itl` | Add `e2el` |
| `--metric-percentiles` | `99` | Use `50,90,99` |
| `--model` | first model from `/v1/models` | Pass the server's `--served-model-name` and a matching `--tokenizer` |

The client loads a tokenizer to build random prompts and count tokens, from `--tokenizer` if given, otherwise from `--model` ([serve.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/benchmarks/serve.py)).
Mount the Hugging Face cache read-only and set `HF_HUB_OFFLINE=1`, so the client uses the tokenizer the server already downloaded and fails loudly if it is missing.

The warmup requests reuse the first prompt of the measured set ([serve.py L876-946](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/benchmarks/serve.py)).
With prefix caching on, `--num-warmups 1` therefore puts the first measured prompt in the cache and lowers its TTFT.
Warm up with a short separate run on a different seed instead.

### Prefix-cache inflation

Prefix caching is on by default in v0.33.0 ([cache.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/config/cache.py)).
The docs warn that repeating `vllm bench serve` against the same server can reuse prompts left in the prefix cache and inflate throughput, and that the random dataset is reproducible for a fixed `--seed`, which defaults to 0 ([cli.md](https://github.com/vllm-project/vllm/blob/v0.33.0/docs/benchmarking/cli.md)).
Their remedies: vary `--seed`, reset or restart the server, or use `vllm bench sweep serve`, which resets caches between runs.

On this machine, use a fresh seed per run (`--seed $RANDOM`) and record it.
To reset the cache without a restart, start the server with `-e VLLM_SERVER_DEV_MODE=1`, which exposes `POST /reset_prefix_cache` ([envs.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/envs.py), [dev cache router](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/entrypoints/serve/dev/cache/api_router.py)).
vLLM logs "SECURITY WARNING: Development endpoints are enabled!" in that mode ([serve/__init__.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/entrypoints/serve/__init__.py)); that is acceptable only because the port is published on 127.0.0.1.

```bash
curl -s -X POST http://127.0.0.1:8000/reset_prefix_cache
```

### `bench latency`, `bench throughput` and `bench sweep`

`vllm bench latency` and `vllm bench throughput` are offline benchmarks: they load the model inside the benchmark process ([latency.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/benchmarks/latency.py), [cli.md](https://github.com/vllm-project/vllm/blob/v0.33.0/docs/benchmarking/cli.md)).
On a 16 GB GPU they cannot run beside a running vLLM or llama.cpp server, because a vLLM server has already reserved about 14.65 GiB at the default 0.92, or more where a recipe raises it (the gpt-oss-20b recipe in [models.md](models.md) uses 0.965 with `--enforce-eager`, and K2 Horizon 7B uses 0.94).
Stop the server first if you want them, and give the benchmark container the same GPU and memory flags as a server (see [README.md](README.md)).
They measure the engine without the HTTP frontend, the chat template or the parsers, so they say less about OpenCode than `bench serve` does.
`bench latency` defaults to `--input-len 32 --output-len 128 --batch-size 8`, which is nothing like an agent turn; set them explicitly ([latency.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/benchmarks/latency.py)).

`vllm bench sweep serve` starts its own `vllm serve` for each parameter combination and runs 3 repetitions by default ([sweeps.md](https://github.com/vllm-project/vllm/blob/v0.33.0/docs/benchmarking/sweeps.md)).
Its cache resets need `VLLM_SERVER_DEV_MODE=1`, which it sets itself in the environment of each server it starts ([sweep/server.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/benchmarks/sweep/server.py)).
Here it would have to run inside one GPU container started with a shell entrypoint, with no other server running.
Treat it as optional; the manual scenarios below are enough to start.

## Before a run

1. Run one GPU server at a time: stop any llama.cpp container first.
2. Start the vLLM server with the run template in [README.md](README.md) and the per-model flags in [models.md](models.md), published on 127.0.0.1:8000 only, with a container memory cap and no restart policy (see [host setup](../docs/host-setup.md)).
3. Run the smoke tests in [README.md](README.md) first, including the tool-call test; a benchmark of a broken parser setup is worthless.
4. Copy two lines from the server startup log into the results: the KV cache size in tokens and `Maximum concurrency for N tokens per request` ([kv_cache_utils.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/core/kv_cache_utils.py)).
   The docs give `max_concurrency = kv_cache_size / max_model_len` ([cli.md](https://github.com/vllm-project/vllm/blob/v0.33.0/docs/benchmarking/cli.md)).
5. Plug in AC power, keep the same power profile for every run, and note it.
   The GPU power limit observed on 2026-11-15 was 140 W; a lower limit on battery or a quieter profile would change every number.
6. Do not run other heavy work at the same time.
   The user's `heavy` wrapper does not cap Docker containers ([host setup](../docs/host-setup.md)); the client container's `--memory` flag does.

Sample the GPU during each run, in a second terminal:

```bash
mkdir -p vllm/results
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,power.draw,clocks.sm,temperature.gpu \
  --format=csv -l 1 > vllm/results/gpu-$(date +%Y%m%d-%H%M%S).csv
# stop with Ctrl-C when the run ends
```

`memory.used` under vLLM shows the up-front reservation, not live KV use.
For live KV use and cache hits, read the server's Prometheus endpoint ([metric names in loggers.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/metrics/loggers.py)):

```bash
curl -s http://127.0.0.1:8000/metrics \
  | grep -E '^vllm:(kv_cache_usage_perc|prefix_cache_hits|prefix_cache_queries|num_preemptions|num_requests_running)'
```

Preemptions above zero mean the KV pool was too small for the scenario.

## Client setup

Run from the project folder.
Set the model variables to match the running server; the served names below are the ones used in [models.md](models.md) and the [OpenCode config](../docs/opencode.md).

| Model | `MODEL` (served name) | `TOKENIZER` (HF repo the server loaded) |
|---|---|---|
| gpt-oss-20b (rank 9) | `gpt-oss-20b` | `openai/gpt-oss-20b` |
| Groot-19B-A3B-261023 NVFP4 (rank 8) | `groot-19b-a3b` | `GrootTeam/Groot-19B-A3B-261023-NVFP4` |
| K2 Horizon 7B AWQ-INT4 (rank 4) | `k2-horizon-7b` | `cyankiwi/K2-Horizon-7B-AWQ-INT4` |
| Gemma 4 12B QAT W4A16 (honorable mention) | `gemma-4-12b` | `google/gemma-4-12B-it-qat-w4a16-ct` |

```bash
export MODEL=groot-19b-a3b
export TOKENIZER=GrootTeam/Groot-19B-A3B-261023-NVFP4
export META="gpu=rtx5080-laptop power_limit_w=140 driver=595.99.02 image=vllm-openai:v0.33.0"
mkdir -p vllm/results vllm/bench-prompts

vbench() {
  docker run --rm --network host \
    --memory 8g --memory-swap 8g \
    -e HF_HUB_OFFLINE=1 \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface:ro" \
    -v "$PWD/vllm/results:/results" \
    -v "$PWD/vllm/bench-prompts:/prompts:ro" \
    --entrypoint vllm vllm/vllm-openai:v0.33.0 \
    bench serve \
      --backend openai-chat --endpoint /v1/chat/completions \
      --base-url http://127.0.0.1:8000 \
      --model "$MODEL" --tokenizer "$TOKENIZER" \
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
      --save-result --save-detailed --result-dir /results \
      "$@"
}
```

The client needs no GPU and publishes no port; `--network host` lets it reach the server's 127.0.0.1:8000.
The 8 GiB cap is a choice for a tokenizer-only client, not a measured need.
Result files are written as root; `sudo chown -R "$USER" vllm/results` afterwards if needed.
`--save-result` names files `{label}-{request_rate}qps-concurrency{N}-{model}-{datetime}.json` unless `--result-filename` is set ([serve.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/benchmarks/serve.py)).

Warm-up (not recorded), run once after each server start:

```bash
vbench --label warmup --dataset-name random --random-input-len 1024 --random-output-len 64 \
  --ignore-eos --num-prompts 2 --max-concurrency 1 --seed 1
```

## Scenarios

Each command is one run; repeat each scenario three times with fresh seeds and report the median run.
Keep input plus output below the server's `--max-model-len`.
`--ignore-eos` fixes the output length, which the docs recommend for random runs ([cli.md](https://github.com/vllm-project/vllm/blob/v0.33.0/docs/benchmarking/cli.md)).

### S1: decode speed, single stream

```bash
vbench --label s1-decode --dataset-name random \
  --random-input-len 1024 --random-output-len 512 --ignore-eos \
  --num-prompts 8 --max-concurrency 1 --seed $RANDOM \
  --metadata $META scenario=s1 thinking=server-default
```

Read the median TPOT; per-stream decode speed is 1000 / TPOT(ms) tok/s.

### S2: TTFT ladder at OpenCode-like prompt sizes

```bash
for LEN in 8192 32768 61440; do
  vbench --label s2-ttft-$LEN --dataset-name random \
    --random-input-len $LEN --random-output-len 128 --ignore-eos \
    --num-prompts 4 --max-concurrency 1 --seed $RANDOM \
    --metadata $META scenario=s2 input_len=$LEN
done
```

Add a 120000 rung only for servers started with `--max-model-len 131072` (Groot, Gemma 4 12B, Qwen3.5-9B and Mellum2.1 in [models.md](models.md)).
gpt-oss-20b on vLLM runs at 65536 at most on this card, so 61440 is its top rung.
Random prompts do not share content, so this measures cold prefill: prefill speed is about input length / TTFT (estimate).

### S3: prefix-cache effect

This mimics a long fixed system prompt and tool schema followed by a new user turn.
`prefix_repetition` builds prompts from shared prefixes plus unique suffixes ([datasets.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/benchmarks/datasets/datasets.py), [cli.md](https://github.com/vllm-project/vllm/blob/v0.33.0/docs/benchmarking/cli.md)).

```bash
vbench --label s3-prefix --dataset-name prefix_repetition \
  --prefix-repetition-prefix-len 7000 --prefix-repetition-suffix-len 1000 \
  --prefix-repetition-num-prefixes 1 --prefix-repetition-output-len 256 \
  --ignore-eos --num-prompts 16 --max-concurrency 1 --seed $RANDOM \
  --metadata $META scenario=s3 prefix_caching=on
```

Then restart the same server with `--no-enable-prefix-caching` (the flag pair comes from a boolean option in [arg_utils.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/engine/arg_utils.py)) and repeat with `prefix_caching=off`.
The TTFT difference is the cache benefit; check `vllm:prefix_cache_hits` against `vllm:prefix_cache_queries` on `/metrics` to confirm hits happened.
Hybrid Mamba or GDN models (Groot and Qwen3.5-9B are Gated DeltaNet hybrids) use a different prefix-cache mode, `mamba_cache_mode "align"`, which caches the recurrent state only at block-aligned positions and step ends ([cache.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/config/cache.py)), so measure their benefit rather than assume it.

### S4: concurrency 2

```bash
vbench --label s4-c2-decode --dataset-name random \
  --random-input-len 1024 --random-output-len 512 --ignore-eos \
  --num-prompts 8 --max-concurrency 2 --seed $RANDOM \
  --metadata $META scenario=s4 concurrency=2

vbench --label s4-c2-32k --dataset-name random \
  --random-input-len 32768 --random-output-len 256 --ignore-eos \
  --num-prompts 4 --max-concurrency 2 --seed $RANDOM \
  --metadata $META scenario=s4 concurrency=2 input_len=32768
```

Compare per-stream TPOT with S1, and TTFT with the 32k rung of S2.
The server's `--max-num-seqs` must be at least 2, and every ranked recipe in [models.md](models.md) uses 1, so restart the server with `--max-num-seqs 2` for this scenario, only where the KV pool leaves room.

### S5: thinking on versus off

Random tokens make thinking meaningless, so use a few real coding prompts in a custom JSONL dataset ([datasets.py CustomDataset](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/benchmarks/datasets/datasets.py)).
Create `vllm/bench-prompts/coding.jsonl`, one JSON object per line:

```json
{"prompt": "Write a Python function that parses an ISO 8601 duration such as PT1H30M into seconds, with tests.", "output_tokens": 4096}
{"prompt": "Explain why this SQL is slow and rewrite it: SELECT * FROM orders WHERE YEAR(created_at) = 2026;", "output_tokens": 4096}
{"prompt": "Refactor a 200-line Bash deploy script into functions. List the functions you would create and why.", "output_tokens": 4096}
{"prompt": "A Go HTTP handler leaks goroutines under load. List the five most likely causes and how to confirm each.", "output_tokens": 4096}
```

Do not pass `--ignore-eos` here: the point is when the model stops on its own.

```bash
for THINK in false true; do
  vbench --label s5-think-$THINK --dataset-name custom --dataset-path /prompts/coding.jsonl \
    --custom-output-len -1 --num-prompts 4 --max-concurrency 1 --seed $RANDOM \
    --extra-body "{\"chat_template_kwargs\":{\"enable_thinking\":$THINK}}" \
    --metadata $META scenario=s5 thinking=$THINK
done
```

For gpt-oss-20b, thinking cannot be turned off; compare `--extra-body '{"reasoning_effort":"low"}'` with `"medium"` instead.
The `custom` loader reads JSONL with pandas, and whether the image ships pandas is unverified; if it fails with an import error, note it in the results and skip S5 until a derived image or GuideLLM is set up.

How to read S5:

- Reasoning deltas count toward TTFT and ITL, and only answer text is collected as generated text ([cli.md](https://github.com/vllm-project/vllm/blob/v0.33.0/docs/benchmarking/cli.md), [endpoint_request_func.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/benchmarks/lib/endpoint_request_func.py)).
  TTFT therefore stays low with thinking on; compare E2EL and output tokens instead.
- An empty entry in `generated_texts` in the saved JSON means the whole budget went to reasoning, the failure the DGX Spark runbook recorded with thinking on.
- Never pass `--extra-body '{"include_reasoning": false}'`: the docs warn, in their Responses API section, that the server then emits no events for reasoning tokens, so the whole reasoning phase is absorbed into TTFT ([cli.md](https://github.com/vllm-project/vllm/blob/v0.33.0/docs/benchmarking/cli.md)); the same effect on the chat backend is an inference, not a documented statement.

### S6: tool-call success rate

`vllm bench serve` can replay BFCL tool-calling traffic (`--dataset-name hf --dataset-path gorilla-llm/Berkeley-Function-Calling-Leaderboard --bfcl-categories simple,live_simple,multiple`), but it measures latency, not whether the calls were right, and multi-turn categories are not supported ([cli.md](https://github.com/vllm-project/vllm/blob/v0.33.0/docs/benchmarking/cli.md)).
It also needs the Hugging Face `datasets` package and a network download, neither verified in the image.

A direct check of correctness is a short loop against the chat endpoint.
It sends no `tool_choice`; v0.33.0 then defaults to `"auto"` when tools are present ([protocol.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/entrypoints/openai/chat_completion/protocol.py)), the same as OpenCode.
`THINK` unset leaves the server default; `EFFORT` is for gpt-oss.

```bash
MODEL=groot-19b-a3b N=20 THINK=0 python3 -I - <<'EOF'
import json, os, urllib.request

url = "http://127.0.0.1:8000/v1/chat/completions"
model = os.environ["MODEL"]
n = int(os.environ.get("N", "20"))
tools = [{"type": "function", "function": {
    "name": "read_file",
    "description": "Read a text file from the repository.",
    "parameters": {"type": "object",
                   "properties": {"path": {"type": "string", "description": "Repository-relative path"}},
                   "required": ["path"]}}}]
body = {"model": model, "max_tokens": 2048, "tools": tools,
        "messages": [{"role": "user", "content": "Open src/app/main.py and tell me what it does."}]}
if "THINK" in os.environ:
    body["chat_template_kwargs"] = {"enable_thinking": os.environ["THINK"] == "1"}
if "EFFORT" in os.environ:
    body["reasoning_effort"] = os.environ["EFFORT"]

counts = {"ok": 0, "wrong_args": 0, "malformed_args": 0, "leaked_as_text": 0, "no_call": 0}
for _ in range(n):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        msg = json.load(resp)["choices"][0]["message"]
    calls = msg.get("tool_calls") or []
    if calls:
        try:
            args = json.loads(calls[0]["function"]["arguments"])
        except ValueError:
            counts["malformed_args"] += 1
            continue
        good = calls[0]["function"]["name"] == "read_file" and args.get("path") == "src/app/main.py"
        counts["ok" if good else "wrong_args"] += 1
    elif "read_file" in (msg.get("content") or ""):
        counts["leaked_as_text"] += 1
    else:
        counts["no_call"] += 1
print(json.dumps({"model": model, "n": n, "think": os.environ.get("THINK"),
                  "effort": os.environ.get("EFFORT"), **counts}))
EOF
```

`leaked_as_text` above zero almost always means the wrong `--tool-call-parser` (see [README.md](README.md)).
This tests the first call only; multi-turn reliability after compaction needs a real OpenCode session or the SQL benchmark below.

### OpenCode's real prompt size

The 5.5k to 7k token fixed prompt in [OpenCode notes](../docs/opencode.md) is an estimate from source character counts.
To replace it with a measured figure, read the per-request prompt-token histogram before and after one short OpenCode turn against a server that nothing else is using.
vLLM records `vllm:request_prompt_tokens` once per finished request with that request's full prompt length ([loggers.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/metrics/loggers.py)).

```bash
curl -s http://127.0.0.1:8000/metrics | grep -E '^vllm:request_prompt_tokens_(sum|count)'
```

The increase in `_sum` divided by the increase in `_count` is the mean prompt size of the requests that turn sent.
If `_count` rose by more than one, OpenCode sent more than one request for that turn, so the mean understates the main turn; record both deltas.

### Optional: agentic text-to-SQL benchmark

The [LLM SQL Benchmark](https://github.com/nlothian/llm-sql-benchmark) by the author of a [r/LocalLLaMA post](https://www.reddit.com/r/LocalLLaMA/comments/1s7r9wu/) is an author-built, 25-question tool loop (`run_sql_query` and `results_ok` over DuckDB, at most 20 tool calls per question) that can point at any OpenAI-compatible endpoint and stores per-question traces.
Its CLI help shows `npx tsx apps/cli/src/benchmark.ts --endpoint <url> --model <name> [--timeout <seconds>]`, where the endpoint is the full chat-completions URL, here `http://127.0.0.1:8000/v1/chat/completions`.
It needs Node.js and the repository's dependencies, which this knowledge base has not installed or tested; follow the repository's own setup instructions.

Caveats that limit what its scores mean ([community benchmarks](../docs/community-benchmarks.md)):

- Temperature is hard-coded to 0.1 for every model, overriding vendor sampling ([apps/cli/src/openai-tool-call.ts](https://github.com/nlothian/llm-sql-benchmark/blob/main/apps/cli/src/openai-tool-call.ts); [issue #4](https://github.com/nlothian/llm-sql-benchmark/issues/4) about it was open when reviewed on 2026-11-15).
- The default per-question timeout is 120 s ([apps/cli/src/benchmark.ts](https://github.com/nlothian/llm-sql-benchmark/blob/main/apps/cli/src/benchmark.ts)); the author's published local small-model runs used 300 s ([result files](https://github.com/nlothian/llm-sql-benchmark/tree/main/data/benchmarks)).
  Record the value you use.
- Scoring checks row and column counts, column names and the first-row values only, from a single run per model ([write-up](https://sql-benchmark.nicklothian.com/)); a desktop RTX 5080 user who ran it saw 1 to 3 tasks change between identical runs ([r/LocalLLaMA, 2026-04-01](https://www.reddit.com/r/LocalLLaMA/comments/1s9mkm1/benchmarked_18_models_that_i_can_run_on_my_rtx/)), so run it three times.
- Treat it as a smoke test for tool-loop reliability on this server, not as a model ranking.

## How to read the metrics

All latencies are measured at the client ([cli.md](https://github.com/vllm-project/vllm/blob/v0.33.0/docs/benchmarking/cli.md)).

| Metric | Definition | What it tells you here |
|---|---|---|
| TTFT | Time from sending the request to the first streamed output | Prefill cost of the prompt; for OpenCode, the pause before anything appears |
| TPOT | (E2EL - TTFT) / (output tokens - 1), per request, then aggregated | Decode speed per stream; 1000 / TPOT(ms) = tok/s |
| ITL | Gap between consecutive streamed outputs, over all requests | Smoothness; with speculative decoding it undercounts, so compare TPOT instead |
| E2EL | Request send to last token | The whole turn; the metric to compare thinking on versus off |
| Output token throughput | Generated tokens / benchmark duration, all streams | Only meaningful at concurrency above 1 |

Report p50 as the headline and p90 and p99 as the spread.
With 4-16 prompts per run, p99 is effectively the maximum.
An OpenCode turn costs roughly TTFT for the uncached part of the context plus output tokens x TPOT (estimate).

The example outputs in the vLLM docs (for example 382.89 output tok/s for Hermes-3-Llama-3.1-8B) come from unspecified hardware and versions; they show the output format only and are not performance data for any GPU.
Third-party numbers that do exist for RTX 5080 cards come mostly from the desktop card (360 W total graphics power per [NVIDIA](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5080/), 960 GB/s as modelled by community calculators), so they are not laptop numbers; they are collected and labeled in [community benchmarks](../docs/community-benchmarks.md).

## Results

Keep the raw JSON from `--save-result --save-detailed` in `vllm/results/` and add one summary row per scenario here.
Use the median of three runs, and link the JSON file in the Notes column.
Compare runtimes only through the same scenarios; the llama.cpp equivalents are in [llama.cpp benchmarking](../llamacpp/benchmarking.md).

| Date | Image | Model and checkpoint | Server flags (non-default) | max-model-len / KV tokens (log) | Scenario | Thinking | Conc. | In / out tokens | TTFT p50 / p90 (ms) | TPOT p50 (ms) | ITL p99 (ms) | E2EL p50 (ms) | Output tok/s | Peak `memory.used` (MiB) | Power and profile | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-16 | vllm/vllm-openai:v0.33.0 | gpt-oss-20b, `openai/gpt-oss-20b` (MXFP4, Marlin) | util 0.965, `--enforce-eager`, `--max-num-batched-tokens 256`, `--max-num-seqs 1`, FP8 KV, TRITON_ATTN | 65536 / 81,504 | short prompt, 1,500 tokens out (3 runs) | always, medium | 1 | 93 / 1500 | 65 to 90 (client) | about 12.0 | | | 83.5 | 15700 | about 144 W, 68 C | Probe script, not `vllm bench serve`; [record](../research/2026-11-16/vllm-gpt-oss-20b/probe-results.json) |
| 2026-11-16 | vllm/vllm-openai:v0.33.0 | gpt-oss-20b | same | 65536 / 81,504 | 8k fresh prompt, then the same prompt again | always, medium | 1 | 8058 / 64 | 1809, repeat 73 (client) | | | | | 15700 | | Prefix cache hit on the repeat |
| 2026-11-16 | vllm/vllm-openai:v0.33.0 | gpt-oss-20b | same | 65536 / 81,504 | 30k prompt with the 8k prefix cached, then again | always, medium | 1 | 30003 / 64 | 8867, repeat 87 (client) | | | | | 15700 | | llama.cpp: 4048 and 95 ms for the same test |
| 2026-11-16 | vllm/vllm-openai:v0.33.0 | gpt-oss-20b | same | 65536 / 81,504 | OpenCode v2.3.0 session, write then read | always, medium | 1 | 3652 first request | | | | | | 15700 | | Passed in 8 s; [events](../research/2026-11-16/vllm-gpt-oss-20b/opencode-e2e-events.jsonl) |
| 2026-11-17 | vllm/vllm-openai:v0.33.0 | Mellum2.1 12B-A2.5B Thinking, `cyankiwi/Mellum2.1-12B-A2.5B-Thinking-AWQ-INT4` (AWQ-INT4) | util 0.92 (default), `--max-num-seqs 1`, BF16 KV | 131072 / 426,946 | short prompt, up to 1,500 tokens out (3 runs) | toggle, on (recipe default) | 1 | 39 / 1,337 (runs of 1,171 to 1,500) | 22 to 40 (client) | about 3.9 | | | 257.2 | 14816 | about 143 W, 56 to 60 C, 145 W limit | Probe script, not `vllm bench serve`; two of three runs stopped by themselves; [record](../research/2026-11-17/vllm-mellum2.1-12b/probe-results.json) |
| 2026-11-17 | vllm/vllm-openai:v0.33.0 | Mellum2.1 12B-A2.5B Thinking | same | 131072 / 426,946 | 8k fresh prompt, then the same prompt again | toggle, on (recipe default) | 1 | 9,932 / 64 | 895, repeat 32 (client) | | | | | 14816 | | Prefix cache hit on the repeat |
| 2026-11-17 | vllm/vllm-openai:v0.33.0 | Mellum2.1 12B-A2.5B Thinking | same | 131072 / 426,946 | 30k fresh prompt, then the same prompt again | toggle, on (recipe default) | 1 | 37,190 / 64 | 4311, repeat 73 (client) | | | | | 14816 | | |
| 2026-11-17 | vllm/vllm-openai:v0.33.0 | K2 Horizon 7B, `cyankiwi/K2-Horizon-7B-AWQ-INT4` (AWQ-INT4) | util 0.94, `--max-num-seqs 1`, FP8 KV, TRITON_ATTN, `--trust-remote-code` | 65536 / 95,168 | short prompt, up to 1,500 tokens out (3 runs) | always, high | 1 | 36 / 1500 | 23 to 41 (client) | about 10.5 | | | 94.8 | 15384 | 136 to 142 W, 54 to 63 C, 145 W limit | Probe script, not `vllm bench serve`; the shipped recipe at util 0.92 did not start ([attempts](../research/2026-11-17/vllm-k2-horizon-7b/attempts.md)); [record](../research/2026-11-17/vllm-k2-horizon-7b/probe-results.json) |
| 2026-11-17 | vllm/vllm-openai:v0.33.0 | K2 Horizon 7B | same | 65536 / 95,168 | 8k fresh prompt, then the same prompt again | always, high | 1 | 8,085 / 64 | 2227, repeat 37 (client) | | | | | 15384 | | Prefix cache hit on the repeat |
| 2026-11-17 | vllm/vllm-openai:v0.33.0 | K2 Horizon 7B | same | 65536 / 95,168 | 30k fresh prompt, then the same prompt again | always, high | 1 | 30,261 / 64 | 10869, repeat 89 (client) | | | | | 15384 | | |
| 2026-11-17 | vllm/vllm-openai:v0.33.0 | Qwen3.5-9B, `RedHatAI/Qwen3.5-9B-NVFP4` (NVFP4, FlashInfer CUTLASS) | util 0.92 (default), `--language-model-only`, `--max-num-seqs 1`, BF16 KV | 131072 / 147,663 | short prompt, up to 1,500 tokens out (3 runs) | toggle, off (recipe default) | 1 | 40 / 676 (runs of 676 to 791) | 37 to 55 (client) | about 11.0 | | | 90.8 | 13726 | 128.6 to 131.7 W, 57 to 61 C, 145 W limit | Probe script, not `vllm bench serve`; every run stopped by itself before 1,500 tokens; [record](../research/2026-11-17/vllm-qwen3.5-9b/probe-results.json) |
| 2026-11-17 | vllm/vllm-openai:v0.33.0 | Qwen3.5-9B | same | 131072 / 147,663 | 8k fresh prompt, then the same prompt again | toggle, off (recipe default) | 1 | 8,591 / 46 | 592, repeat 65 (client) | | | | | 13726 | | Prefix cache hit on the repeat |
| 2026-11-17 | vllm/vllm-openai:v0.33.0 | Qwen3.5-9B | same | 131072 / 147,663 | 30k fresh prompt, then the same prompt again | toggle, off (recipe default) | 1 | 32,153 / 46 | 2424, repeat 119 (client) | | | | | 13726 | | |
| 2026-11-17 | vllm/vllm-openai:v0.33.0 | Gemma 4 12B IT, `google/gemma-4-12B-it-qat-w4a16-ct` (QAT W4A16) | util 0.92 (default), `--max-num-seqs 1`, FP8 KV, TRITON_ATTN, no image or audio | 131072 / 366,455 | short prompt, up to 1,500 tokens out (3 runs) | toggle, off (recipe default) | 1 | 42 / 933 (runs of 885 to 933) | 38 to 47 (client) | about 14.4 | | | 69.5 | 15822 | 134 to 136 W, 53 to 61 C, 145 W limit | Probe script, not `vllm bench serve`; every run stopped by itself before 1,500 tokens; the peak is from startup profiling; [record](../research/2026-11-17/vllm-gemma-4-12b/probe-results.json) |
| 2026-11-17 | vllm/vllm-openai:v0.33.0 | Gemma 4 12B IT | same | 131072 / 366,455 | 8k fresh prompt, then the same prompt again | toggle, off (recipe default) | 1 | 9,936 / 29 | 3878, repeat 43 (client) | | | | | 15822 | | Prefix cache hit on the repeat |
| 2026-11-17 | vllm/vllm-openai:v0.33.0 | Gemma 4 12B IT | same | 131072 / 366,455 | 30k fresh prompt, then the same prompt again | toggle, off (recipe default) | 1 | 37,194 / 40 | 17180, repeat 91 (client) | | | | | 15822 | | |
| 2026-11-17 | vllm/vllm-openai:v0.33.0 | Gemma 4 E4B IT, `google/gemma-4-E4B-it-qat-w4a16-ct` (QAT W4A16) | util 0.92 (default), `--enforce-eager`, `--language-model-only`, `--max-num-seqs 2`, BF16 KV | 65536 / 204,533 | short prompt, up to 1,500 tokens out (3 runs) | toggle, off (recipe default) | 1 | 38 / 1,130 (runs of 969 to 1,190) | 49 to 66 (client) | about 22.1 | | | 45.2 | 15044 | 87 to 91 W, 52 to 60 C, 145 W limit | Probe script, not `vllm bench serve`; every run stopped by itself before 1,500 tokens; without `--enforce-eager` the start ran out of GPU memory ([attempts](../research/2026-11-17/vllm-gemma-4-e4b/attempts.md)); [record](../research/2026-11-17/vllm-gemma-4-e4b/probe-results.json) |
| 2026-11-17 | vllm/vllm-openai:v0.33.0 | Gemma 4 E4B IT | same | 65536 / 204,533 | 8k fresh prompt, then the same prompt again | toggle, off (recipe default) | 1 | 9,932 / 36 | 1522, repeat 63 (client) | | | | | 15044 | | Prefix cache hit on the repeat |
| 2026-11-17 | vllm/vllm-openai:v0.33.0 | Gemma 4 E4B IT | same | 65536 / 204,533 | 30k fresh prompt, then the same prompt again | toggle, off (recipe default) | 1 | 37,190 / 41 | 7899, repeat 109 (client) | | | | | 15044 | | |
| 2026-11-17 | vllm/vllm-openai:v0.33.0 | Groot-19B-A3B-261023, `GrootTeam/Groot-19B-A3B-261023-NVFP4` (NVFP4, Marlin weight-only) | util 0.92 (default), `--language-model-only`, `--max-num-seqs 1`, FP8 KV, TRITON_ATTN | 131072 / 167,480 | short prompt, up to 1,500 tokens out (3 runs) | toggle, off (recipe default) | 1 | 40 / 829 (runs of 780 to 829) | 56 to 76 (client) | about 5.2 | | | 193.4 | 14812 | 138 to 140 W, 54 to 57 C, 145 W limit | Probe script, not `vllm bench serve`; every run stopped by itself before 1,500 tokens; [record](../research/2026-11-17/vllm-groot-19b-a3b/probe-results.json) |
| 2026-11-17 | vllm/vllm-openai:v0.33.0 | Groot-19B-A3B-261023 | same | 131072 / 167,480 | 8k fresh prompt, then the same prompt again | toggle, off (recipe default) | 1 | 8,591 / 58 | 861, repeat 69 (client) | | | | | 14812 | | Prefix cache hit on the repeat |
| 2026-11-17 | vllm/vllm-openai:v0.33.0 | Groot-19B-A3B-261023 | same | 131072 / 167,480 | 30k fresh prompt, then the same prompt again | toggle, off (recipe default) | 1 | 32,153 / 46 | 3675, repeat 148 (client) | | | | | 14812 | | |
| 2026-11-17 | vllm/vllm-openai:v0.33.0 | DeepSeek-R1-0528-Qwen3-8B, `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` (online `fp8_per_tensor`) | util 0.965, `--max-num-seqs 1`, FP8 KV, TRITON_ATTN | 65536 / 80,064 | short prompt, 1,500 tokens out (3 runs) | always | 1 | 31 / 1500 | 32 to 64 (client) | about 13.9 | | | 72.1 | 15750 | 138 to 141 W, 59 to 64 C, 140 W limit | Failed: replies are garbled by the tokenizer class, see below; [record](../research/2026-11-17/vllm-deepseek-r1-0528-qwen3-8b/) |

The default flags from [models.md](models.md#gpt-oss-20b) (util 0.95, CUDA graphs on) did not start; see the [attempt log](../research/2026-11-16/vllm-gpt-oss-20b/attempts.md).
The 2026-11-17 rows come from `./llmctl up` of each shipped recipe with the 2026-11-16 probe script, one run per recipe ([records](../research/2026-11-17/README.md)); none of them ran an OpenCode session.
K2 Horizon 7B needed `--gpu-memory-utilization 0.94` and Gemma 4 E4B needed `--enforce-eager`, and both recipes now carry the flag ([models.md](models.md#k2-horizon-7b-awq-int4)).
DeepSeek-R1-0528-Qwen3-8B failed: the shipped recipe's first start at util 0.92, with an empty torch.compile cache, lacked 0.67 GiB of KV cache, the measured start above used util 0.965, and both full e2e runs of 2026-11-17 later started the shipped recipe at 0.92 with the cache warm ([record 276](../research/2026-11-17/llmctl-dsh-e2e/records/276-18-deepseek-r1-0528-qwen3-8b-vllm-up), [Known issues](README.md#known-issues)); either way the transformers 5.20.2 in the image loads its tokenizer as a Metaspace `LlamaTokenizer` over a byte-level BPE `tokenizer.json`, so prompts lose their spaces and every reply carries raw byte-level markers ([tokenizer check](../research/2026-11-17/vllm-deepseek-r1-0528-qwen3-8b/tokenizer-check.json)).
Its token counts are therefore not comparable, its recipe stays an estimate, and the [llama.cpp recipe](../llamacpp/models.md#10-deepseek-r1-0528-qwen3-8b-chat-and-reasoning-only) is the one to use.

For S3 add the prefix-cache hit ratio in Notes; for S6 put the `ok / n` count and the other categories in Notes.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Client cannot find the tokenizer | `HF_HUB_OFFLINE=1` and the repo is not in the cache, or `TOKENIZER` differs from what the server loaded | Use the exact repo id from the server command |
| HTTP 404 model not found | `MODEL` is not the server's `--served-model-name` | `curl -s http://127.0.0.1:8000/v1/models` |
| Requests fail at long inputs | Input plus output exceeds `--max-model-len` | Lower the rung; check the KV line in the startup log |
| TTFT much lower on the second run | Prefix cache hit from a repeated seed | Fresh `--seed`, or `POST /reset_prefix_cache` with dev mode |
| Preemptions in `/metrics` | KV pool too small for the concurrency | Lower concurrency or context; see the memory budget in [README.md](README.md) |
| `vllm bench serve` will not start without a GPU | Not observed; untested assumption that the client needs none | Report it; adding the GPU device to the client is the workaround to try |
| Tool calls arrive as text | Wrong or missing `--tool-call-parser` | See parsers in [README.md](README.md) |

## Sources

- [vLLM benchmark suites index](https://docs.vllm.ai/en/latest/benchmarking/)
- [vLLM benchmark CLI docs, v0.33.0](https://github.com/vllm-project/vllm/blob/v0.33.0/docs/benchmarking/cli.md)
- [vLLM parameter sweeps docs, v0.33.0](https://github.com/vllm-project/vllm/blob/v0.33.0/docs/benchmarking/sweeps.md)
- [vllm/benchmarks/serve.py, v0.33.0](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/benchmarks/serve.py)
- [vllm/benchmarks/datasets/datasets.py, v0.33.0](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/benchmarks/datasets/datasets.py)
- [vllm/benchmarks/lib/endpoint_request_func.py, v0.33.0](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/benchmarks/lib/endpoint_request_func.py)
- [vllm/benchmarks/latency.py, v0.33.0](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/benchmarks/latency.py)
- [vllm/benchmarks/sweep/server.py, v0.33.0](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/benchmarks/sweep/server.py)
- [docker/Dockerfile, v0.33.0](https://github.com/vllm-project/vllm/blob/v0.33.0/docker/Dockerfile)
- [vllm/envs.py, v0.33.0](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/envs.py) and [dev cache router](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/entrypoints/serve/dev/cache/api_router.py)
- [vllm/v1/metrics/loggers.py, v0.33.0](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/metrics/loggers.py)
- [vllm/config/cache.py, v0.33.0](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/config/cache.py)
- [chat_completion/protocol.py, v0.33.0](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/entrypoints/openai/chat_completion/protocol.py)
- [GuideLLM](https://github.com/vllm-project/guidellm)
- [LLM SQL Benchmark repository](https://github.com/nlothian/llm-sql-benchmark) and [r/LocalLLaMA post, 2026-03-30](https://www.reddit.com/r/LocalLLaMA/comments/1s7r9wu/)
- [NVIDIA GeForce RTX 50 Series laptops](https://www.nvidia.com/en-us/geforce/laptops/50-series/)
- Full catalog with evidence classes: [sources](../docs/sources.md)

## Maintenance

Before relying on this document, re-verify:

- The image tag: if `vllm/vllm-openai` moves past v0.33.0, re-check the `bench serve` defaults, the dataset names and the entrypoint in the new tag's `serve.py`, `datasets.py` and Dockerfile.
- That the image actually contains a working `vllm bench serve` and, for S5 and BFCL, pandas and `datasets` (`docker run --rm --entrypoint python3 vllm/vllm-openai:v0.33.0 -c 'import pandas, datasets'`).
- The served names and tokenizer repos against [models.md](models.md).
- The power limit and profile with `nvidia-smi -q -d POWER` before each benchmark session.
- After the first real runs, replace "nothing has been run" in the summary and fill the results table.
