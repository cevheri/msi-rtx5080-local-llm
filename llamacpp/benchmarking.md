# Benchmarking llama.cpp on this laptop

Last verified: 2026-11-16.

Benchmark with two tools: `llama-bench` from the `full-cuda-b12407` image for raw prompt and generation speed at several context depths, and `llama-server` with `--metrics` plus the per-response `timings` object for OpenCode-like turns, time to first token and prompt-cache reuse.
Measurements from this laptop since 2026-11-16 are in the results tables at the end; every other number is a third-party measurement on other hardware or an estimate, and each one is labeled.
Desktop RTX 5080 numbers are upper bounds here: the laptop GPU has less bandwidth and a much lower power limit, and it runs with CUDA graphs disabled.
Run the procedures below, fill in the results template at the end, and only then rank models on speed.

How to start the server, which flags to use and why are in [README.md](README.md).
Per-model files and offload values are in [models.md](models.md).
Container GPU access, memory caps and localhost-only ports are in [../docs/host-setup.md](../docs/host-setup.md).

## What to measure

| Metric | Why it matters for OpenCode | Tool |
|---|---|---|
| Prompt processing (pp) tok/s at depth 0, 8k, 32k, 64k | Every turn re-sends the conversation; prefill speed sets how long a large turn takes before the first token | `llama-bench -p ... -d ...` |
| Generation (tg) tok/s at the same depths | Decode speed of tool calls and answers; it falls as the context fills | `llama-bench -n 128 -d ...` |
| Time to first token for an OpenCode-size prompt | OpenCode's fixed prompt overhead alone is about 5.5k-7k tokens (estimate, see [../docs/opencode.md](../docs/opencode.md)) | `llama-server` `timings.prompt_ms` |
| Prompt-cache reuse across turns | A good turn re-processes only the new suffix; a cache miss re-processes the whole history | `llama-server` `timings.cache_n` versus `timings.prompt_n` |
| Offload trade-off | For MoE models with experts in RAM, each extra layer on the CPU costs decode and prefill speed but frees VRAM for context | `--n-cpu-moe` sweep |
| Peak VRAM and host RAM | The GPU has 16303 MiB and about 40 GiB of host RAM is free beside the desktop | `nvidia-smi`, `free`, cgroup `memory.peak` |
| GPU power, clocks, temperature | A 140 W laptop GPU can throttle in long runs; a slow run with throttle flags set is not a model result | `nvidia-smi --query-gpu` |

Use the runtime's own counters, not wall-clock arithmetic, as the [TechFuelHQ RTX 5080 dataset](https://huggingface.co/datasets/iBlessi/rtx-5080-llm-throughput) does.
Make every repetition a fresh prefill, for example with a unique nonce at the start of the prompt: TechFuelHQ found that warm repeats hit the prompt cache and reported about 60,000 tok/s "prefill" that measured nothing ([dataset README](https://huggingface.co/datasets/iBlessi/rtx-5080-llm-throughput)).
`--cache-prompt` is on by default in llama-server ([server README at b12418](https://github.com/ggml-org/llama.cpp/blob/b12418/tools/server/README.md)), so the same trap applies here.

## Ground rules for every run

- Run one model container at a time, and never while a vLLM container holds the GPU.
- Pass `-e GGML_CUDA_DISABLE_GRAPHS=1` as a docker flag before the image name, for llama-bench as well as llama-server: CUDA graphs hang the GPU on RTX 50 Laptop parts ([#27330](https://github.com/ggml-org/llama.cpp/issues/27330), open), and the variable is read by the ggml CUDA backend, not by a specific binary.
  Details in [README.md](README.md).
- Give every container `--memory` with an equal `--memory-swap`.
  The `heavy` wrapper caps only the docker CLI process, not the container, which dockerd starts under `system.slice` (see [../docs/host-setup.md](../docs/host-setup.md)).
  Exit code 137 means the cap was hit: record it as a result and lower the RAM load, do not raise the cap blindly.
- Publish server ports on `127.0.0.1` only, use no restart policy, and stop the container when done so the dGPU can power down.
- Close heavy desktop applications and keep the laptop on AC power with the same power profile for every run; write the profile into the results.
- Record the GGUF file name and size, the image tag and the build number in every row; numbers from different builds are not comparable.

## Which image has llama-bench

The `server-cuda` image copies only the `llama` and `llama-server` binaries (entrypoint `/app/llama-server`), and the `light-cuda` image only `llama`, `llama-cli` and `llama-completion`, so neither has `llama-bench` ([cuda.Dockerfile at b12418](https://github.com/ggml-org/llama.cpp/blob/b12418/.devops/cuda.Dockerfile)).
The `full-cuda` image copies every built binary into `/app` and uses [`/app/tools.sh`](https://github.com/ggml-org/llama.cpp/blob/b12418/.devops/tools.sh) as entrypoint; `--bench` (or `-b`) runs `llama-bench` with the remaining arguments.
The newest pinned tag is `ghcr.io/ggml-org/llama.cpp:full-cuda-b12407`: built 2026-11-15 from build b12407, CUDA 12.8.1, `NVIDIA_REQUIRE_CUDA=cuda>=12.8`, about 3.62 GB compressed for amd64 (GHCR manifest and image config read on 2026-11-15; [package page](https://github.com/ggml-org/llama.cpp/pkgs/container/llama.cpp)).
`full-cuda-b12418` does not exist yet (HTTP 404), and `full-cuda13-b12407` has the same CUDA 13.5 start-check problem as the server image, so use `full-cuda` for the reasons given in [README.md](README.md).
The binary list comes from the Dockerfile; the image itself has not been pulled on this machine.

## llama-bench

### Options that matter here

All of these exist at b12418 ([llama-bench README](https://github.com/ggml-org/llama.cpp/blob/b12418/tools/llama-bench/README.md)).
Every test parameter accepts a comma list or a range such as `14-26+2`, and llama-bench runs every combination.

| Option | Meaning | Default |
|---|---|---|
| `-p N` | prompt-processing test with N tokens (`pp N`) | 512 |
| `-n N` | text-generation test with N tokens (`tg N`) | 128 |
| `-pg P,N` | prompt of P tokens followed by N generated tokens | none |
| `-d N` | prefill the KV cache with N tokens first, then run the test at that depth | 0 |
| `-b N` / `-ub N` | logical and physical batch size | 2048 / 512 |
| `-fa on\|off\|auto` | flash attention | auto |
| `-ctk` / `-ctv` | KV cache type, for example `q8_0` | f16 |
| `-ngl N` | layers on the GPU | -1 |
| `-ncmoe N` | experts of the first N layers stay in host RAM | 0 |
| `-lm MODE` | load mode (`none` replaces the removed `--no-mmap`) | auto |
| `-t N` | CPU threads | system dependent |
| `-r N` | repetitions per test, reported as mean ± standard deviation | 5 |
| `-o md\|csv\|json\|jsonl\|sql` | output format on stdout | md |
| `-hf repo:quant` | download from Hugging Face into the shared cache, like llama-server | unused |
| `-fitt` / `-fitc` | automatic fit to device memory, with a margin in MiB / a minimum context | off / 0 |

Three differences from llama-server matter for interpreting results:

- `--fit` is off in llama-bench (`-fitt` defaults to off), while llama-server fits by default.
  llama-bench does not move experts to the CPU on its own: set `-ncmoe` explicitly.
- llama-bench allocates a context of only `p + n + d` tokens ([llama-bench.cpp at b12418](https://github.com/ggml-org/llama.cpp/blob/b12418/tools/llama-bench/llama-bench.cpp)), not the server's `-c`.
  An `-ncmoe` value that fits in llama-bench at depth 0 can run out of VRAM in a llama-server with `-c 131072`.
  Run the sweep at the depth you plan to serve, then confirm the chosen value on llama-server with its real `-c`.
- llama-bench times exclude tokenization and sampling (README note), so server numbers will be a little lower.

### Example: gpt-oss-20b fully on the GPU

This reproduces the shape of the only official RTX 5080 run in the [ggml-org gpt-oss guide](https://github.com/ggml-org/llama.cpp/discussions/15396) (`-fa 1 -b 4096 -ub 2048`) and adds depths that match OpenCode use.

```bash
mkdir -p ~/llm-bench
docker run --rm --name llama-bench \
  --device nvidia.com/gpu=all \
  --memory 16g --memory-swap 16g \
  -e GGML_CUDA_DISABLE_GRAPHS=1 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  ghcr.io/ggml-org/llama.cpp:full-cuda-b12407 \
  --bench -hf ggml-org/gpt-oss-20b-GGUF:MXFP4 \
  -ngl 99 -fa on -b 4096 -ub 2048 \
  -p 512,2048,8192 -n 128 -d 0,8192,32768,65536 \
  -r 3 -o jsonl | tee ~/llm-bench/$(date +%F)-gptoss20b-depth.jsonl
```

Each JSONL row carries `build_commit`, `build_number`, `gpu_info`, `model_size`, every parameter, `avg_ts`, `stddev_ts` and the per-repetition `samples_ts` ([output formats](https://github.com/ggml-org/llama.cpp/blob/b12418/tools/llama-bench/README.md#jsonl)), so the file is the raw record and the results table is a summary of it.
`--device nvidia.com/gpu=all` needs the CDI spec from the NVIDIA Container Toolkit, which is installed and verified since 2026-11-15 ([../docs/host-setup.md](../docs/host-setup.md)).

### Example: a cross-machine anchor

Run the model from the [llama.cpp CUDA scoreboard](https://github.com/ggml-org/llama.cpp/discussions/15013) once, so this laptop can be placed against other GPUs on identical work.
The scoreboard uses `llama-2-7b.Q4_0.gguf` from `TheBloke/Llama-2-7B-GGUF` with `llama-bench -m llama-2-7b.Q4_0.gguf -ngl 99 -fa 0,1` (default `pp512` and `tg128`).

```bash
docker run --rm --name llama-bench-anchor \
  --device nvidia.com/gpu=all \
  --memory 8g --memory-swap 8g \
  -e GGML_CUDA_DISABLE_GRAPHS=1 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  ghcr.io/ggml-org/llama.cpp:full-cuda-b12407 \
  --bench -hf TheBloke/Llama-2-7B-GGUF -hff llama-2-7b.Q4_0.gguf \
  -ngl 99 -fa 0,1 -o jsonl | tee ~/llm-bench/$(date +%F)-anchor-llama2-7b-q4_0.jsonl
```

Note in the results that this run has CUDA graphs disabled, while scoreboard entries normally do not.

## llama-server timings and metrics

### The `timings` object

Every `/v1/chat/completions` response from llama-server carries a `timings` object with `cache_n` (prompt tokens reused from the cache), `prompt_n` (prompt tokens processed now), `prompt_ms`, `prompt_per_second`, `predicted_n`, `predicted_ms` and `predicted_per_second` ([server README at b12418, "Timings and context usage"](https://github.com/ggml-org/llama.cpp/blob/b12418/tools/server/README.md)).
The context in use is `prompt_n + cache_n + predicted_n`.
`prompt_ms` is the server-side prefill time, which is the time to first token minus HTTP and tokenization overhead.
The request field `timings_per_token` adds speed data to each streamed chunk, and `return_progress` adds a `prompt_progress` object (`total`, `cache`, `processed`, `time_ms`) in stream mode while a long prompt is processed (same README).

### The `/metrics` endpoint

`/metrics` exists only when the server starts with `--metrics` (default disabled, env `LLAMA_ARG_ENDPOINT_METRICS`) and exports Prometheus counters ([server README, GET /metrics](https://github.com/ggml-org/llama.cpp/blob/b12418/tools/server/README.md)).
Useful ones here: `llamacpp:prompt_tokens_total`, `llamacpp:prompt_seconds_total`, `llamacpp:tokens_predicted_total`, `llamacpp:tokens_predicted_seconds_total`, `llamacpp:prompt_tokens_seconds`, `llamacpp:predicted_tokens_seconds`, `llamacpp:n_tokens_max` (the highest context seen) and `llamacpp:requests_deferred`.
Read it before and after an OpenCode session and take the difference: that gives real-session prefill and decode rates without changing OpenCode.
`/slots` is on by default (`--no-slots` turns it off) and shows per-slot state.

### Start a server for measurement

```bash
docker run --rm -d --name llama-bench-srv \
  --device nvidia.com/gpu=all \
  --memory 16g --memory-swap 16g \
  -e GGML_CUDA_DISABLE_GRAPHS=1 \
  -p 127.0.0.1:8080:8080 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  ghcr.io/ggml-org/llama.cpp:server-cuda-b12407 \
  -hf ggml-org/gpt-oss-20b-GGUF:MXFP4 --no-mmproj \
  -c 65536 -np 1 -fa on --jinja --metrics --alias gpt-oss-20b
```

For another model, take its line from [models.md](models.md) and add `--metrics`.
llama-server keeps a host-RAM prompt cache of up to 8192 MiB by default (`--cache-ram`, [server README](https://github.com/ggml-org/llama.cpp/blob/b12418/tools/server/README.md)); add that to the expert RAM when you size `--memory` for an offloaded model.
If the server uses `--api-key`, add `-H "Authorization: Bearer ..."` to the requests below.

### Time to first token and cache reuse across two turns

The probe below sends a fresh, nonce-prefixed prompt of about 8k tokens (estimate; the server reports the real count) and then a second turn that extends the same conversation, the way OpenCode does.
It is a small host-side probe and needs only Python's standard library.

```bash
python3 -I - <<'EOF'
import json, time, uuid, urllib.request

URL = "http://127.0.0.1:8080/v1/chat/completions"
filler = open("/usr/share/common-licenses/GPL-3").read()  # about 35 KB of text

def ask(messages, max_tokens=256):
    body = json.dumps({"model": "bench", "messages": messages,
                       "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.load(r)
    wall = time.monotonic() - t0
    t = out["timings"]
    print(f"cache_n={t['cache_n']} prompt_n={t['prompt_n']} prompt_ms={t['prompt_ms']:.0f} "
          f"pp={t['prompt_per_second']:.1f} tok/s predicted_n={t['predicted_n']} "
          f"tg={t['predicted_per_second']:.1f} tok/s wall={wall:.1f}s")
    return out["choices"][0]["message"]

nonce = uuid.uuid4().hex
msgs = [{"role": "system", "content": f"run {nonce}"},
        {"role": "user", "content": filler + "\n\nSummarize section 2 in three sentences."}]
reply = ask(msgs)                                   # turn 1: cold prefill, expect cache_n near 0
msgs += [{"role": "assistant", "content": reply.get("content") or ""},
         {"role": "user", "content": "Now list the defined terms in section 0."}]
ask(msgs)                                           # turn 2: expect cache_n close to turn 1's context
EOF
```

How to read it:

- Turn 1 `prompt_ms` is the cold time to first token for an 8k prompt; `prompt_per_second` should be close to `llama-bench` `pp8192` at depth 0.
- In turn 2, `cache_n` should cover almost all of turn 1 and `prompt_n` only the new tail.
  A `prompt_n` close to the whole history means the cache was invalidated and the turn paid a full re-prefill.
- For Qwen3.5 and newer hybrid models that full re-prefill is a known open bug when the model emits non-canonical whitespace ([#30357](https://github.com/ggml-org/llama.cpp/issues/30357), open).
  Run the probe several times on those models and record how often `cache_n` collapses.
- The probe sends back only the visible `content`, as a client would.
  For thinking models, also record whether OpenCode sends the reasoning back, because a different history text breaks the prefix match.
- Thinking tokens are generated tokens: they cost decode time and count against `max_tokens`.
  Measure thinking on and off for models that can toggle it, because the user's DGX Spark runbook measured a thinking-on reply that spent the whole 1024-token budget with no answer while thinking-off answered in 1.5 s (see [../docs/opencode.md](../docs/opencode.md) for the variants).

## MoE offload sweep (`--n-cpu-moe`)

The goal is the smallest `-ncmoe` (the most experts on the GPU) that still leaves room for the target context, and the speed cost of each step above it.
`-ncmoe N` keeps the experts of the first N layers in host RAM, counting block indices from 0 ([llama-bench.cpp](https://github.com/ggml-org/llama.cpp/blob/b12418/tools/llama-bench/llama-bench.cpp), [server README](https://github.com/ggml-org/llama.cpp/blob/b12418/tools/server/README.md)).
Starting values per model are in [models.md](models.md); for Qwen3.6-35B-A3B Q4_K_M the research estimate is about 0.42 GiB of experts per layer (16.875 GiB over 40 layers), with about 21 to 22 layers on the CPU at 128k context and fewer at 64k with q8_0 KV (estimate, from the [model config](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/raw/main/config.json)).
Measured on 2026-11-16: 21 left about 2.1 GiB free at 128k with f16 KV, and spending that VRAM on the expert cache (`--moe-cache-mib`) beat moving blocks back to the GPU; the variants are under [results](#qwen36-35b-a3b-offload-variants).

### Monitoring, in a second terminal

```bash
nvidia-smi --query-gpu=timestamp,pstate,memory.used,utilization.gpu,power.draw,clocks.sm,clocks.mem,temperature.gpu,clocks_event_reasons.active \
  --format=csv -l 1 | tee ~/llm-bench/$(date +%F-%H%M)-gpu.csv
```

```bash
free -m -s 2 | tee ~/llm-bench/$(date +%F-%H%M)-ram.log
```

While a container runs, its cgroup peak memory is in `/sys/fs/cgroup/system.slice/docker-<full id>.scope/memory.peak` (path from the research on this host's systemd cgroup driver; see [../docs/host-setup.md](../docs/host-setup.md)), and `docker stats --no-stream` gives a snapshot.
All of these query fields were checked against `nvidia-smi --help-query-gpu` on this machine with driver 595.99.02.
`clocks_event_reasons.active` is a bitmask in which 0x1 only means the GPU is idle; any other bit set during a run means the clocks were held back (power cap, thermal or hardware slowdown), so flag that row.

### Sweep procedure

1. Pick the depth you will serve at.
   llama-bench sizes the KV cache to `p + n + d`, so run the sweep at `-d 32768` or `-d 65536` with the KV type you will use, not only at depth 0.
2. Run one container per `-ncmoe` value, from high (safe) to low, so one out-of-memory load does not end the whole sweep.
3. Use `-lm none` so the CPU-side experts sit in anonymous RAM rather than an mmap of the file.
   This is the current spelling of the old `--no-mmap`, which one desktop RTX 5080 user found essential for prefill speed with offloaded experts (community advice, [r/LocalLLaMA thread](https://www.reddit.com/r/LocalLLaMA/comments/1s9mkm1/benchmarked_18_models_that_i_can_run_on_my_rtx/)).
   It also makes the expert RAM count against the container cap, which is what you want to see.
   Measured here on 2026-11-16, it doubled prefill on Qwen3.6-35B-A3B, because the pinned experts upload at full link speed (see [results](#qwen36-35b-a3b-offload-variants)).
4. Watch the monitoring terminals.
   Stop when `memory.used` passes about 15.5 GiB or the cgroup peak approaches the cap.

```bash
mkdir -p ~/llm-bench
for n in 26 24 22 20 18 16 14; do
  docker run --rm --name llama-sweep \
    --device nvidia.com/gpu=all \
    --memory 24g --memory-swap 24g \
    -e GGML_CUDA_DISABLE_GRAPHS=1 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    ghcr.io/ggml-org/llama.cpp:full-cuda-b12407 \
    --bench -hf ggml-org/Qwen3.6-35B-A3B-GGUF:Q4_K_M \
    -ngl 99 -ncmoe "$n" -lm none -fa on -ctk q8_0 -ctv q8_0 \
    -b 4096 -ub 2048 -p 4096 -n 128 -d 0,32768 -r 3 -o jsonl \
    >> ~/llm-bench/$(date +%F)-qwen36-ncmoe.jsonl \
    || echo "ncmoe=$n exit=$?" >> ~/llm-bench/$(date +%F)-qwen36-ncmoe.log
done
```

The 24g cap covers about 11 GiB of experts at `-ncmoe 26` plus runtime overhead (estimate from 0.42 GiB per layer).
For Qwen3-Coder-Next, 26.4 GiB of experts stay in pinned RAM and the container peaked at 31.9 GiB under a 35g cap (measured on 2026-11-16, [models.md](models.md#6-qwen3-coder-next)), so close large applications before a sweep.

5. Plot `avg_ts` for `pp4096` and `tg128` against `-ncmoe` at each depth.
   Expect both to fall as N rises, decode because expert reads move to system RAM (DDR5-6400 dual channel per [Intel's spec](https://www.intel.com/content/www/us/en/products/sku/245960/intel-core-ultra-9-processor-290hx-plus-36m-cache-up-to-5-50-ghz/specifications.html), so about 102.4 GB/s peak (estimate: 6400 MT/s x 8 bytes x 2 channels; Intel lists no bandwidth figure); the configured DIMM speed is unverified, so check `sudo dmidecode -t memory` once), prefill because experts are copied to the GPU per layer.
6. Take the lowest N that loaded, add one or two layers of margin, and confirm it on llama-server with the real `-c`, `-np 1` and the KV type.
   Compare with what `--fit` picks on its own when `-ncmoe` is left out (see [README.md](README.md)).
7. Optionally sweep threads at the chosen N with `-t 8,12,16,24 -p 0 -n 128`.
   One desktop RTX 5080 user saw speed drop by more than 20% when he set 16 threads on an 8-core, 16-thread Ryzen 7 9800X3D, where the default of 8 was best (community advice, [u/grumd comment, 2026-03-30](https://www.reddit.com/r/LocalLLaMA/comments/1s7lo0a/comment/odbkdoe/)); this CPU has 24 cores and 24 threads, so measure rather than guess.

On the prefill side, offloaded-MoE prompt processing can leave the GPU idle while expert weights are copied serially ([#25859](https://github.com/ggml-org/llama.cpp/issues/25859), open, measured on an RTX 3060 with PCIe 4.0).
This laptop's GPU link is PCIe 5.0 x16 (`nvidia-smi -q`).
Measured on 2026-11-16 with Qwen3.6-35B-A3B and 21 blocks in RAM: 934 tok/s prefill at 8.6k with the default mmap load, and 2,112 tok/s with `-lm none`, which pins the host experts so they upload at full link speed.

## Micro-batch (`-ub`) tuning

Community advice, not measured here: on a desktop RTX 5080 with Qwen3.5-122B-A10B experts in RAM, prefill was under 300 tok/s at the default `-ub 512` and rose to about 1500 tok/s at `-ub 2048` to `4096` ([u/grumd, r/LocalLLaMA, 2026-04](https://www.reddit.com/r/LocalLLaMA/comments/1s9mkm1/benchmarked_18_models_that_i_can_run_on_my_rtx/)).
That fits the serial-copy explanation above: a bigger micro-batch amortizes each expert upload over more tokens.
The defaults are `-b 2048 -ub 512` in both llama-bench and llama-server ([server README](https://github.com/ggml-org/llama.cpp/blob/b12418/tools/server/README.md)).

```bash
# inside the same docker run template as above, at the chosen -ncmoe:
--bench -hf ggml-org/Qwen3.6-35B-A3B-GGUF:Q4_K_M -ngl 99 -ncmoe 22 -lm none -fa on \
  -b 4096 -ub 512,1024,2048,4096 -p 4096 -n 0 -r 3 -o jsonl
```

A larger `-ub` also grows the compute buffer in VRAM, so a value that helps prefill can push the model out of its `-ncmoe` fit; re-check VRAM after changing it.
For models fully on the GPU the gain is usually smaller: the gpt-oss guide's RTX 5080 run measured `-ub 2048` and `-ub 4096`, and `-ub 2048` was as fast or faster on every prefill size ([#15396](https://github.com/ggml-org/llama.cpp/discussions/15396)).

## KV cache type at depth

Measure `-ctk/-ctv f16` against `q8_0` at `-d 0,32768,65536` before relying on q8_0 for long contexts.
Measured on 2026-11-16 on K2 Horizon 7B (head size 128, dense): after a 30k prompt f16 KV decoded at 41.4 tok/s against 40.0 for q8_0, with prefill about 3% faster, so the q8_0 penalty was small on this build; the drop from 67 tok/s at a short context came from the attention reads themselves ([record](../research/2026-11-16/llamacpp-k2-horizon-7b/variants/)).
Models with 256- or 512-wide heads may behave differently: Gemma 4's 512-wide global heads have no vector flash-attention kernel, so its recipe keeps f16 ([models.md](models.md#5-gemma-4-26b-a4b-it)).
Measured on 2026-11-16 on Qwen3-Coder-Next (head size 256, 16 query and 2 KV heads, 12 attention layers of 48): from a short context to a 101k prompt, decode fell 40% with q8_0 KV (80.9 to 48.7 tok/s) and 18% with f16 KV (71.5 to 58.6 tok/s), while prefill was the same ([record](../research/2026-11-16/llamacpp-qwen3-coder-next/variants/)).
[PR #31802](https://github.com/ggml-org/llama.cpp/pull/31802) (open on 2026-11-16) describes the likely cause: for a quantized KV cache, the MMA flash-attention kernels first convert the whole K/V to f16 in a separate pass, on models with large heads and a GQA ratio above 4; its author measured +63% decode at 131k with the conversion fused (RTX 2000 Ada, not this laptop).
Qwen3.8-27B (256-wide heads, q8_0 KV in its recipe) fell from 36.4 to 29.6 tok/s after 32k, so its open f16 check (item 3 in [model-selection.md](../docs/model-selection.md#what-to-measure-first)) matters more now.
[#27623](https://github.com/ggml-org/llama.cpp/issues/27623) (open) reports Qwen3.8-27B decode collapsing past about 80k tokens on an RTX 4080 SUPER; our research found the cliff in its title retracted, and what remains in dispute is a q8_0 KV decode penalty, with a fix in PR [#31802](https://github.com/ggml-org/llama.cpp/pull/31802) (open on 2026-11-15).
Do not use q4_0 KV for coding agents: community testing reports very poor answers, and Qwen3.5-class models are sensitive to KV quantization (community advice, [same r/LocalLLaMA thread](https://www.reddit.com/r/LocalLLaMA/comments/1s9mkm1/benchmarked_18_models_that_i_can_run_on_my_rtx/)).

## Quality smoke tests

Speed alone does not rank models.
The [LLM SQL Benchmark CLI](https://github.com/nlothian/llm-sql-benchmark) can be pointed at llama-server's `/v1/chat/completions` as a short tool-calling check with per-question traces; it hardcodes temperature 0.1 and defaults to a 120 s per-question timeout (`--timeout`), so record both (per the source review in [../docs/sources.md](../docs/sources.md); not yet run here).
Run it at least three times per model: the author of the 18-model RTX 5080 comparison saw 1 to 3 tasks change between identical runs, and slow offloaded models need a longer timeout or they score as failures ([r/LocalLLaMA thread](https://www.reddit.com/r/LocalLLaMA/comments/1s9mkm1/benchmarked_18_models_that_i_can_run_on_my_rtx/)).
To drive llama-server with the same load generator used for vLLM, `vllm bench serve --backend openai-chat --endpoint /v1/chat/completions --base-url http://127.0.0.1:8080` is a candidate that has not been tested against llama.cpp; check that its token counts match the server's `usage` block before comparing (see [../vllm/benchmarking.md](../vllm/benchmarking.md)).

## Community reference numbers

None of these were measured on this laptop.
Every row is a third-party measurement on desktop hardware, and the Qwen3.6 row is a figure recalled in a comment rather than a benchmark table.
They are listed to show what the same software does on related hardware, and to detect a broken setup (a result far below them).
The normalized, cross-runtime collection lives in [../docs/community-benchmarks.md](../docs/community-benchmarks.md).

| Source | Hardware | Runtime | Model and settings | Prefill tok/s | Decode tok/s |
|---|---|---|---|---|---|
| [ggml-org gpt-oss guide](https://github.com/ggml-org/llama.cpp/discussions/15396) | Desktop RTX 5080, CUDA 12.8 | llama.cpp build 6210 (Aug 2025), llama-bench | gpt-oss-20b MXFP4 all on GPU, `-fa 1 -b 4096 -ub 2048` | pp2048 7476.55, pp8192 7047.73, pp32768 5531.03 | tg128 204.85 |
| [CUDA scoreboard #15013](https://github.com/ggml-org/llama.cpp/discussions/15013), with FA | Desktop RTX 5080 | llama.cpp commit 8a4280c, llama-bench | Llama 2 7B Q4_0, pp512 / tg128 | 9487.70 ± 21.89 | 184.68 ± 0.05 |
| [CUDA scoreboard #15013](https://github.com/ggml-org/llama.cpp/discussions/15013), with FA | Desktop RTX 5070 Ti (896 GB/s, the laptop 5080's spec bandwidth) | llama.cpp commit 933414c, llama-bench | Llama 2 7B Q4_0, pp512 / tg128 | 8419.56 ± 35.50 | 182.43 ± 0.09 |
| [TechFuelHQ Ollama vs llama.cpp](https://techfuelhq.com/self-hosted/ollama-vs-llamacpp-rtx-5080-2026/) | Desktop RTX 5080 | llama-server b10507 (upstream GGUF) vs Ollama 0.32.1, 4k context, 256-token generations | gpt-oss-20b | not used here | 222-238 (llama.cpp), 195.8 (Ollama) |
| [u/grumd llama-benchy table](https://www.reddit.com/r/LocalLLaMA/comments/1sdu8oz/bartowski_vs_unsloth_for_gemma_4/oem5j5k/) | Desktop RTX 5080 (overclocked), Ryzen 7 9800X3D, DDR5-6000 | self-built llama.cpp (comment dated 2026-04-06), llama-benchy | Gemma 4 26B-A4B UD-Q6_K_XL, `--n-cpu-moe 23`, q8_0 KV, `-b 4096 -ub 1024`, depth 0 | pp4096 2590.02 | tg128 40.92 |
| same | same | same | same, depth 100000 | pp4096 2104.35 (TTFT 49.6 s) | tg128 30.89 |
| [u/grumd comment](https://www.reddit.com/r/LocalLLaMA/comments/1so3rsx/comment/ogrgfvj/) | Desktop RTX 5080, DDR5-6000 | self-built llama.cpp (comment dated 2026-04-17) | Qwen3.6-35B-A3B UD-Q6_K_XL, 262k f16 KV, `--fit`, 15 GB VRAM + 29 GB RAM | not given | 35 (single recalled figure) |
| [TechFuelHQ dataset (iBlessi)](https://huggingface.co/datasets/iBlessi/rtx-5080-llm-throughput) | Desktop RTX 5080, Windows 11, driver 610.88, Ryzen 7 7800X3D | Ollama 0.32.1, median of 3 | gpt-oss:20b MXFP4, 4k and 16k context, 503-1658 token prompts | 9078-11001 | 178.34-190.14 |
| same | same | same | qwen3-coder:30b Q4_K_M, partial offload (74% on GPU, measured at 8k) | 1525-2369 | 51.62-61.89 |
| same | same | same | qwen2.5:14b Q4_K_M, all on GPU | 4865-5202 | 94.58-96.65 |

Notes on these rows:

- The gpt-oss gap between Ollama and llama.cpp on the same desktop card comes mostly from different model conversions (Ollama's own blob versus the upstream GGUF), not wrapper overhead, per the [TechFuelHQ companion page](https://techfuelhq.com/self-hosted/ollama-vs-llamacpp-rtx-5080-2026/).
- TechFuelHQ attributes its offloaded qwen3-coder spread (0.7, 62 and 83 tok/s under three host states) to the load-time memory state and the resulting GPU/CPU split, not only free RAM ([canonical dataset page](https://techfuelhq.com/data/rtx-5080-llm-throughput/)).
- The Qwen3.6 figure is a recalled number from a comment, not a benchmark table.
- The ubatch advice in the previous section is from the same desktop user.

### Why these are upper bounds here

- Bandwidth: the laptop GPU has 896 GB/s ([NVIDIA RTX 50 laptop specs](https://www.nvidia.com/en-us/geforce/laptops/50-series/)) against 960 GB/s for the desktop RTX 5080, and decode on a fully resident model is bandwidth bound.
- Power: the laptop GPU's limit is 140 W on this machine (observed), against 360 W for the desktop card, so long prefill runs can hit power or thermal limits that the desktop never sees.
- CUDA graphs are disabled here because of [#27330](https://github.com/ggml-org/llama.cpp/issues/27330).
  The reference runs presumably had them on, and graphs exist to cut per-token launch overhead, so expect some decode loss (expected, not measured).
- Offload rows depend on the host: the desktop users had DDR5-6000 with 8-core X3D CPUs.
  This laptop's RAM is DDR5-6400 dual channel by spec, so the RAM side is not necessarily slower, but its configured speed is unverified.
- Builds differ: the gpt-oss guide row is from build 6210 (August 2025) and the scoreboard rows from older commits; b12407 may be faster or slower.

### Estimates to check against

- gpt-oss-20b tg128 in llama.cpp: 204.85 × 896 / 960 ≈ 191 tok/s by bandwidth alone (estimate; ignores the power limit and the disabled CUDA graphs).
  The research range is about 180 to 195 tok/s (estimate).
- Dense decode ceiling, following the bandwidth method from the user's DGX Spark runbook: 896 GB/s divided by the weight size in GB.
  For Qwen3.8-27B UD-Q3_K_XL (12.24 GiB, 13.14 GB) that is about 68 tok/s, a ceiling that real decode does not reach (estimate).

A laptop result well below these, with no throttle flags in the nvidia-smi log, points to a setup problem: wrong image, experts mmapped from disk, a missing `-ngl`, or another process on the GPU.

## Results template

Copy a row per test into this table; keep the raw JSONL and the nvidia-smi CSV next to it under `~/llm-bench/`.

### llama-bench results

| Date | Image tag / build | Model repo:file (GiB) | -ngl / -ncmoe | KV type | -b / -ub | -fa | -t | Graphs off | Test | t/s (mean ± sd, reps) | Peak VRAM MiB | Peak RAM GiB (cgroup) | Power W / temp C / throttle | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-mm-dd | full-cuda-b12407 / b12407 | ggml-org/gpt-oss-20b-GGUF:MXFP4 (11.28) | 99 / 0 | f16 | 4096 / 2048 | on | default | Yes | pp2048 @ d0 | not measured yet | | | | |

### llama-server scenario results

| Date | Image tag / build | Model and launch line link | -c / -np | Thinking | Scenario | prompt_n / cache_n | prompt_ms (TTFT proxy) | pp tok/s | predicted_n | tg tok/s | Peak VRAM MiB | Peak RAM GiB | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-16 | server-cuda-b12407 / b12407 | gpt-oss-20b MXFP4, [models.md](models.md#9-gpt-oss-20b) | 131072 / 1 | always, medium | short prompt, 1,500 tokens out (3 runs) | 33 / 60 | under 0.1 s | 936 to 963 | 1500 | 165.7 to 166.2 | 13408 | 2.5 | q8_0 KV, graphs off, about 143 W |
| 2026-11-16 | server-cuda-b12407 / b12407 | gpt-oss-20b MXFP4 | 131072 / 1 | always, medium | turn 1, 8k fresh prompt | 7998 / 60 | about 1,076 ms (client TTFT 1.12 s) | 7436 | 64 | 145.1 | 13408 | 2.5 | |
| 2026-11-16 | server-cuda-b12407 / b12407 | gpt-oss-20b MXFP4 | 131072 / 1 | always, medium | turn 2, same 8k prompt | 1 / 8057 | 8 ms (client TTFT 0.047 s) | | 64 | 145.1 | 13408 | 2.5 | prompt cache hit |
| 2026-11-16 | server-cuda-b12407 / b12407 | gpt-oss-20b MXFP4 | 131072 / 1 | always, medium | 30k prompt, 8k prefix cached | 21960 / 8043 | 3,959 ms (client TTFT 4.05 s) | 5546 | 64 | 105.5 | 13408 | 2.5 | |
| 2026-11-16 | server-cuda-b12407 / b12407 | gpt-oss-20b MXFP4 | 131072 / 1 | always, medium | turn 2, same 30k prompt | 1 / 30002 | 11 ms (client TTFT 0.095 s) | | 64 | 108.4 | 13408 | 2.5 | prompt cache hit |
| 2026-11-16 | server-cuda-b12407 / b12407 | gpt-oss-20b MXFP4 | 131072 / 1 | always, medium | OpenCode v2.3.0 session, write then read | 3648, then 78 and 86 new | 481 ms for the first request | 7590 | 444, 26, 60 | 154 to 164 | 13408 | 2.5 | cache read 4036 and 4123 tokens on later steps |
| 2026-11-16 | server-cuda-b12407 / b12407 | Qwen3.8-27B UD-Q3_K_XL, [models.md](models.md#1-qwen38-27b) | 65536 / 1 | toggle, effort medium | short prompt, 1,500 tokens out (3 runs) | short | under 0.2 s | | 1500 | 36.2 to 36.5 | 14790 | 3.3 | q8_0 KV, graphs off, about 146 W, 72 C |
| 2026-11-16 | server-cuda-b12407 / b12407 | Qwen3.8-27B UD-Q3_K_XL | 65536 / 1 | toggle, effort medium | turn 1, 8.6k fresh prompt | 8589 / 0 | about 6,480 ms (client TTFT 6.53 s) | 1326 | 64 | 34.2 | 14790 | 3.3 | |
| 2026-11-16 | server-cuda-b12407 / b12407 | Qwen3.8-27B UD-Q3_K_XL | 65536 / 1 | toggle, effort medium | 32k prompt, 8k prefix cached | 24078 / 8073 | about 21,680 ms (client TTFT 21.7 s) | 1111 | 64 | 29.6 | 14790 | 3.3 | repeat of the same prompt: 0.12 s, cache_n 32147 |
| 2026-11-16 | server-cuda-b12407 / b12407 | Qwen3.8-27B UD-Q3_K_XL | 65536 / 1 | toggle, effort medium | OpenCode v2.3.0 session, write then read | 4180, then 22 and 70 new | 3,174 ms for the first request | 1317 | 98, 36, 115 | 35.5 to 35.9 | 14790 | 3.3 | passed in 14 s |
| 2026-11-16 | server-cuda-b12407 / b12407 | Qwen3.6-35B-A3B Q4_K_M, baseline: `-ncmoe 21`, mmap, auto threads ([launch](../research/2026-11-16/llamacpp-qwen3.6-35b-a3b/baseline/run-command.sh)) | 131072 / 1 | toggle, off | short prompt, 600-word answer (3 runs) | 40 / 0 | about 0.3 s | | 745 to 865 | 66.0 to 67.5 | 13696 | 1.9, plus mmapped experts | f16 KV, graphs off, up to 101 W, 64 C |
| 2026-11-16 | server-cuda-b12407 / b12407 | Qwen3.6-35B-A3B Q4_K_M, baseline | 131072 / 1 | toggle, off | turn 1, 8.6k fresh prompt | 8591 / 0 | 9,196 ms (client TTFT 9.24 s) | 934 | 40 | 64.3 | 13696 | 1.9 | |
| 2026-11-16 | server-cuda-b12407 / b12407 | Qwen3.6-35B-A3B Q4_K_M, baseline | 131072 / 1 | toggle, off | 32k prompt, 8k prefix cached | 24078 / 8075 | 26,473 ms (client TTFT 26.5 s) | 910 | 36 | 61.2 | 13696 | 1.9 | |
| 2026-11-16 | server-cuda-b12407 / b12407 | Qwen3.6-35B-A3B Q4_K_M, baseline | 131072 / 1 | toggle, off | 107k prompt, 32k prefix cached | 75690 / 31637 | 93,243 ms (client TTFT 93.3 s) | 812 | 48 | 56.4 | 13696 | 1.9 | repeat of the same prompt: 0.16 s, cache_n 107323 |
| 2026-11-16 | server-cuda-b12407 / b12407 | Qwen3.6-35B-A3B Q4_K_M, tuned: `-ncmoe 25 -lm none -t 8 --moe-cache-mib 3200`, [models.md](models.md#2-qwen36-35b-a3b) | 131072 / 1 | toggle, off | short prompt, 600-word answer (3 runs) | 40 / 0 | about 0.15 s | | 727 to 838 | 119.7 to 122.7 | 15190 | 14.9 | f16 KV, graphs off, up to 144 W, 71 C; `-lv 4` on |
| 2026-11-16 | server-cuda-b12407 / b12407 | Qwen3.6-35B-A3B Q4_K_M, tuned | 131072 / 1 | toggle, off | code generation, LRU cache module with tests | 40 / 0 | 119 ms | | 1193 | 114.8 | 15190 | 14.9 | |
| 2026-11-16 | server-cuda-b12407 / b12407 | Qwen3.6-35B-A3B Q4_K_M, tuned | 131072 / 1 | toggle, off | turn 1, 8.6k fresh prompt | 8591 / 0 | 4,317 ms (client TTFT 4.34 s) | 1990 | 45 | 117.7 | 15190 | 14.9 | |
| 2026-11-16 | server-cuda-b12407 / b12407 | Qwen3.6-35B-A3B Q4_K_M, tuned | 131072 / 1 | toggle, off | 32k prompt, 8k prefix cached | 24078 / 8075 | 12,983 ms (client TTFT 13.0 s) | 1855 | 36 | 109.9 | 15190 | 14.9 | |
| 2026-11-16 | server-cuda-b12407 / b12407 | Qwen3.6-35B-A3B Q4_K_M, tuned | 131072 / 1 | toggle, off | 107k prompt, 32k prefix cached | 75690 / 31637 | 50,225 ms (client TTFT 50.3 s) | 1507 | 42 | 92.6 | 15190 | 14.9 | repeat of the same prompt: 0.13 s, cache_n 107323 |
| 2026-11-16 | server-cuda-b12407 / b12407 | Qwen3.6-35B-A3B Q4_K_M, tuned | 131072 / 1 | toggle, off | OpenCode v2.3.0 session, write then read | 4186, then 24 and 136 new | 2,584 ms for the first request | 1620 | 145, 85, 120 | 108.5 to 114.1 | 15190 | 14.9 | passed in 9 s |
| 2026-11-16 | server-cuda-b12407 / b12407 | Gemma 4 26B-A4B QAT Q4_0: `-ncmoe 12 -lm none -t 8 --moe-cache-mib 2368 -ctxcp 16`, [models.md](models.md#5-gemma-4-26b-a4b-it) | 131072 / 1 | toggle, off | short prompt, 600-word answer (3 runs) | 41 / 0 | about 0.1 s | | 923 to 1075 | 147.7 to 150.0 | 15064 | 10.6 | f16 KV, graphs off, up to 145 W, 71 C; `-lv 4` on; a model download ran at the same time |
| 2026-11-16 | server-cuda-b12407 / b12407 | Gemma 4 26B-A4B QAT Q4_0 | 131072 / 1 | toggle, off | code generation, LRU cache module with tests | 40 / 0 | 60 ms | | 1273 | 148.1 | 15064 | 10.6 | |
| 2026-11-16 | server-cuda-b12407 / b12407 | Gemma 4 26B-A4B QAT Q4_0 | 131072 / 1 | toggle, off | turn 1, 9.9k fresh prompt (the 8k target in Gemma tokens) | 9935 / 1 | 2,674 ms (client TTFT 2.73 s) | 3716 | 37 | 132.4 | 15064 | 10.6 | |
| 2026-11-16 | server-cuda-b12407 / b12407 | Gemma 4 26B-A4B QAT Q4_0 | 131072 / 1 | toggle, off | 37k prompt, 10k prefix cached | 27279 / 9915 | 9,293 ms (client TTFT 9.33 s) | 2935 | 34 | 124.9 | 15064 | 10.6 | |
| 2026-11-16 | server-cuda-b12407 / b12407 | Gemma 4 26B-A4B QAT Q4_0 | 131072 / 1 | toggle, off | 124k prompt, 37k prefix cached | 86987 / 37173 | 48,667 ms (client TTFT 48.8 s) | 1787 | 30 | 94.9 | 15064 | 10.6 | repeat of the same prompt: 0.17 s, cache_n 124159 |
| 2026-11-16 | server-cuda-b12407 / b12407 | Gemma 4 26B-A4B QAT Q4_0 | 131072 / 1 | toggle, off | OpenCode v2.3.0 session, write then read | 3824, then 71 and 82 new | 1,273 ms for the first request | 3005 | 54, 19, 29 | 138.3 to 150.4 | 15064 | 10.6 | passed in 6 s |
| 2026-11-16 | server-cuda-b12407 / b12407 | K2 Horizon 7B Q8_0, all on the GPU, [models.md](models.md#4-k2-horizon-7b) | 81920 / 1 | always, high | short prompt, 1,500 tokens out (3 runs) | 30 to 34 / 0 | under 0.1 s | | 1500 | 66.8 to 67.0 | 15008 | 5.5 | q8_0 KV, graphs off, up to 145 W, 70 C; a model download ran at the same time |
| 2026-11-16 | server-cuda-b12407 / b12407 | K2 Horizon 7B Q8_0 | 81920 / 1 | always, high | code generation, LRU cache module with tests | 40 / 0 | | | 3000 (limit) | 65.7 | 15008 | 5.5 | all reasoning, no answer within 3,000 tokens |
| 2026-11-16 | server-cuda-b12407 / b12407 | K2 Horizon 7B Q8_0 | 81920 / 1 | always, high | turn 1, 8k fresh prompt | 8081 / 4 | 1,865 ms (client TTFT 1.91 s) | 4333 | 64 | 58.0 | 15008 | 5.5 | |
| 2026-11-16 | server-cuda-b12407 / b12407 | K2 Horizon 7B Q8_0 | 81920 / 1 | always, high | 30k prompt, 8k prefix cached | 22194 / 8067 | 10,165 ms (client TTFT 10.2 s) | 2183 | 64 | 40.0 | 15008 | 5.5 | f16 KV at `-c 40960`: 2,245 tok/s and 41.4 tok/s |
| 2026-11-16 | server-cuda-b12407 / b12407 | K2 Horizon 7B Q8_0 | 81920 / 1 | always, high | 61k prompt, 30k prefix cached | 30354 / 30243 | 27,809 ms (client TTFT 27.9 s) | 1092 | 64 | 27.7 | 15008 | 5.5 | repeat of the same prompt: 0.11 s, cache_n 60596 |
| 2026-11-16 | server-cuda-b12407 / b12407 | K2 Horizon 7B Q8_0 | 81920 / 1 | always, high | OpenCode v2.3.0 session, write then read | 3482, then 16 and 107 new | 673 ms for the first request | 5177 | 145, 64, 127 | 61.5 to 63.0 | 15008 | 5.5 | passed in 10 s |
| 2026-11-16 | server-cuda-b12407 / b12407 | Nemotron 3.5 Lightning 30B-A3B Q4_0: `-ncmoe 25 -lm none -t 8 --moe-cache-mib 3200`, [models.md](models.md#nemotron-35-lightning-30b-a3b-fallback-if-bolt-does-not-fit) | 131072 / 1 | toggle, off | short prompt, 600-word answer (3 runs) | 40 / 0 | about 0.1 s | | 1054 to 1459 | 143.4 to 145.0 | 15106 | 10.3 | f16 KV, graphs off, up to 146 W, 72 C; `-lv 4` on; a model download ran at the same time |
| 2026-11-16 | server-cuda-b12407 / b12407 | Nemotron 3.5 Lightning 30B-A3B Q4_0 | 131072 / 1 | toggle, off | code generation, LRU cache module with tests | 40 / 0 | | | 1254 | 141.0 | 15106 | 10.3 | |
| 2026-11-16 | server-cuda-b12407 / b12407 | Nemotron 3.5 Lightning 30B-A3B Q4_0 | 131072 / 1 | toggle, off | turn 1, 8.3k fresh prompt | 8338 / 5 | 2,863 ms (client TTFT 2.89 s) | 2912 | 24 | 131.0 | 15106 | 10.3 | |
| 2026-11-16 | server-cuda-b12407 / b12407 | Nemotron 3.5 Lightning 30B-A3B Q4_0 | 131072 / 1 | toggle, off | 31k prompt, 8k prefix cached | 23385 / 7827 | 7,731 ms (client TTFT 7.77 s) | 3025 | 25 | 137.6 | 15106 | 10.3 | |
| 2026-11-16 | server-cuda-b12407 / b12407 | Nemotron 3.5 Lightning 30B-A3B Q4_0 | 131072 / 1 | toggle, off | 104k prompt, 31k prefix cached | 73479 / 30696 | 29,833 ms (client TTFT 29.9 s) | 2463 | 21 | 128.2 | 15106 | 10.3 | repeat of the same prompt: 0.14 s, cache_n 104171 |
| 2026-11-16 | server-cuda-b12407 / b12407 | Nemotron 3.5 Lightning 30B-A3B Q4_0 | 131072 / 1 | toggle, off | OpenCode v2.3.0 session, write then read | 4370, then 24 and 69 new | 1,814 ms for the first request | 2409 | 62, 25, 8 | 120.7 to 136.4 | 15106 | 10.3 | passed in 6 s; the first launch (`-ncmoe 21`, 1,792 MiB cache, 64% hit rate) decoded 136.5 tok/s |
| 2026-11-16 | server-cuda-b12407 / b12407 | K2 Horizon MoVA 36B A4B Q4_K_M: `-ncmoe 48 -lm none -t 8 --moe-cache-mib 1536`, [models.md](models.md#3-k2-horizon-mova-36b-a4b) | 65536 / 1 | always, high | short prompt, up to 1,500 tokens out (3 runs) | 30 to 34 / 0 | under 0.1 s | | 891 to 1500 | 36.9 to 37.5 | 14848 | 19.7 | q8_0 KV, graphs off, up to 144 W, 70 C; `-lv 4` on; a model download ran at the same time |
| 2026-11-16 | server-cuda-b12407 / b12407 | K2 Horizon MoVA 36B A4B Q4_K_M | 65536 / 1 | always, high | code generation, LRU cache module with tests | 40 / 0 | | | 3000 (limit) | 38.2 | 14848 | 19.7 | all reasoning, no answer within 3,000 tokens |
| 2026-11-16 | server-cuda-b12407 / b12407 | K2 Horizon MoVA 36B A4B Q4_K_M | 65536 / 1 | always, high | turn 1, 8k fresh prompt | 8081 / 4 | 6,590 ms (client TTFT 6.64 s) | 1226 | 64 | 31.8 | 14848 | 19.7 | |
| 2026-11-16 | server-cuda-b12407 / b12407 | K2 Horizon MoVA 36B A4B Q4_K_M | 65536 / 1 | always, high | 30k prompt, 8k prefix cached | 22194 / 8067 | 22,753 ms (client TTFT 22.8 s) | 975 | 64 | 26.2 | 14848 | 19.7 | repeat of the same prompt: 0.09 s, cache_n 30260 |
| 2026-11-16 | server-cuda-b12407 / b12407 | K2 Horizon MoVA 36B A4B Q4_K_M | 65536 / 1 | always, high | OpenCode v2.3.0 session, write then read | 3489, then 140 and 172 new | 2,916 ms for the first request | 1197 | 220, 77, 152 | 33.4 to 34.4 | 14848 | 19.7 | passed in 20 s; at equal VRAM, 2 whole blocks plus an 806 MiB cache decoded 32.7 tok/s |
| 2026-11-16 | server-cuda-b12407 / b12407 | DeepSeek-R1-0528-Qwen3-8B Q8_0, all on the GPU, [models.md](models.md#10-deepseek-r1-0528-qwen3-8b-chat-and-reasoning-only) | 65536 / 1 | always | short prompt, up to 1,500 tokens out (3 runs) | 30 to 34 / 0 | under 0.1 s | | 1089 to 1500 | 70.3 to 70.7 | 13286 | 3.0 | q8_0 KV, graphs off, up to 145 W, 70 C; a model download ran at the same time |
| 2026-11-16 | server-cuda-b12407 / b12407 | DeepSeek-R1-0528-Qwen3-8B Q8_0 | 65536 / 1 | always | code generation, LRU cache module with tests | 40 / 0 | | | 2045 | 69.3 | 13286 | 3.0 | complete answer after 2,109 characters of reasoning |
| 2026-11-16 | server-cuda-b12407 / b12407 | DeepSeek-R1-0528-Qwen3-8B Q8_0 | 65536 / 1 | always | turn 1, 8k fresh prompt | 8076 / 2 | 1,882 ms (client TTFT 1.96 s) | 4291 | 64 | 60.5 | 13286 | 3.0 | |
| 2026-11-16 | server-cuda-b12407 / b12407 | DeepSeek-R1-0528-Qwen3-8B Q8_0 | 65536 / 1 | always | 30k prompt, 8k prefix cached | 22189 / 8065 | 10,245 ms (client TTFT 10.3 s) | 2166 | 64 | 41.1 | 13286 | 3.0 | repeat of the same prompt: 0.09 s, cache_n 30253 |
| 2026-11-16 | server-cuda-b12407 / b12407 | DeepSeek-R1-0528-Qwen3-8B Q8_0 | 65536 / 1 | always | OpenCode v2.3.0 session with tools enabled for a test | 4000 / 0 | 795 ms | 5030 | 3673 | 63.0 | 13286 | 3.0 | failed: the tool call was written as text, no file created |
| 2026-11-16 | server-cuda-b12407 / b12407 | Groot-19B-A3B-261023 Q4_K_M, all on the GPU, [models.md](models.md#8-groot-19b-a3b-261023) | 131072 / 1 | toggle, off | short prompt, 600-word answer (3 runs) | 40 / 0 | under 0.1 s | | 861 to 1030 | 169.5 to 170.0 | 13822 | 4.4 | f16 KV, graphs off, up to 146 W, 67 C; a model download ran at the same time |
| 2026-11-16 | server-cuda-b12407 / b12407 | Groot-19B-A3B-261023 Q4_K_M | 131072 / 1 | toggle, off | code generation, LRU cache module with tests | 40 / 0 | | | 1147 | 166.5 | 13822 | 4.4 | |
| 2026-11-16 | server-cuda-b12407 / b12407 | Groot-19B-A3B-261023 Q4_K_M | 131072 / 1 | toggle, off | turn 1, 8.6k fresh prompt | 8591 / 0 | 1,498 ms (client TTFT 1.53 s) | 5737 | 42 | 164.5 | 13822 | 4.4 | |
| 2026-11-16 | server-cuda-b12407 / b12407 | Groot-19B-A3B-261023 Q4_K_M | 131072 / 1 | toggle, off | 32k prompt, 8k prefix cached | 24078 / 8075 | 5,229 ms (client TTFT 5.26 s) | 4605 | 52 | 150.1 | 13822 | 4.4 | |
| 2026-11-16 | server-cuda-b12407 / b12407 | Groot-19B-A3B-261023 Q4_K_M | 131072 / 1 | toggle, off | 107k prompt, 32k prefix cached | 75690 / 31637 | 27,592 ms (client TTFT 27.7 s) | 2743 | 42 | 112.6 | 13822 | 4.4 | repeat of the same prompt: 0.12 s, cache_n 107323 |
| 2026-11-16 | server-cuda-b12407 / b12407 | Groot-19B-A3B-261023 Q4_K_M | 131072 / 1 | toggle, off | OpenCode v2.3.0 session, write then read | 4186, then 24 and 136 new | 749 ms for the first request | 5586 | 149, 85, 144 | 162.3 to 167.0 | 13822 | 4.4 | passed in 6 s |
| 2026-11-16 | server-cuda-b12407 / b12407 | Nemotron 3.5 Bolt 74B-A9B 24GB GGUF: `-ncmoe 60 -lm none -t 8 --moe-cache-mib 2816 -ctxcp 16`, [models.md](models.md#7-nvidia-nemotron-35-bolt-74b-a9b) | 131072 / 1 | toggle, off | short prompt, 600-word answer (3 runs) | 40 / 0 | about 0.1 s | | 657 to 714 | 73.1 to 73.7 | 15064 | 17.8 | f16 KV, graphs off, up to 145 W, 71 C; `-lv 4` on; a model download ran at the same time |
| 2026-11-16 | server-cuda-b12407 / b12407 | Nemotron 3.5 Bolt 74B-A9B 24GB GGUF | 131072 / 1 | toggle, off | code generation, LRU cache module with tests | 40 / 0 | | | 1327 | 72.2 | 15064 | 17.8 | |
| 2026-11-16 | server-cuda-b12407 / b12407 | Nemotron 3.5 Bolt 74B-A9B 24GB GGUF | 131072 / 1 | toggle, off | turn 1, 8.3k fresh prompt | 8338 / 5 | 5,488 ms (client TTFT 5.54 s) | 1519 | 44 | 69.1 | 15064 | 17.8 | |
| 2026-11-16 | server-cuda-b12407 / b12407 | Nemotron 3.5 Bolt 74B-A9B 24GB GGUF | 131072 / 1 | toggle, off | 31k prompt, 8k prefix cached | 23385 / 7827 | 14,475 ms (client TTFT 14.5 s) | 1616 | 36 | 71.1 | 15064 | 17.8 | |
| 2026-11-16 | server-cuda-b12407 / b12407 | Nemotron 3.5 Bolt 74B-A9B 24GB GGUF | 131072 / 1 | toggle, off | 104k prompt, 31k prefix cached | 73479 / 30696 | 52,061 ms (client TTFT 52.2 s) | 1411 | 52 | 65.8 | 15064 | 17.8 | repeat of the same prompt: 0.19 s, cache_n 104171 |
| 2026-11-16 | server-cuda-b12407 / b12407 | Nemotron 3.5 Bolt 74B-A9B 24GB GGUF | 131072 / 1 | toggle, off | OpenCode v2.3.0 session, write then read | 4372, then 113, 82 and 131 new | 3,540 ms for the first request | 1235 | 209, 27, 87, 63 | 70.2 to 74.1 | 15064 | 17.8 | passed in 13 s after a parallel write and read raced and the model recovered with glob |
| 2026-11-16 | server-cuda-b12407 / b12407 | Qwen3-Coder-Next UD-IQ4_XS: `-ncmoe 38 -lm none -t 8 --moe-cache-mib 1504 --cache-ram 4096 -ctxcp 8`, [models.md](models.md#6-qwen3-coder-next) | 131072 / 1 | none | short prompt, 600-word answer (3 runs) | 36 / 0 | about 0.2 s | | 913 to 1212 | 71.1 to 72.0 | 15076 | 31.9 | f16 KV, graphs off, up to 141 W, 71 C; `-lv 4` on; Ollama downloads ran on the E-cores at the same time |
| 2026-11-16 | server-cuda-b12407 / b12407 | Qwen3-Coder-Next UD-IQ4_XS | 131072 / 1 | none | code generation, LRU cache module with tests | 36 / 0 | 244 ms | | 1445 | 68.8 | 15076 | 31.9 | |
| 2026-11-16 | server-cuda-b12407 / b12407 | Qwen3-Coder-Next UD-IQ4_XS | 131072 / 1 | none | turn 1, 8k fresh prompt | 8083 / 0 | 7,257 ms (client TTFT 7.29 s) | 1114 | 46 | 64.7 | 15076 | 31.9 | q8_0 KV with a 2,944 MiB cache: 1,116 tok/s and 72.7 tok/s |
| 2026-11-16 | server-cuda-b12407 / b12407 | Qwen3-Coder-Next UD-IQ4_XS | 131072 / 1 | none | 30k prompt, 8k prefix cached | 22692 / 7567 | 19,725 ms (client TTFT 19.75 s) | 1150 | 47 | 66.0 | 15076 | 31.9 | q8_0 KV: 1,153 tok/s and 68.1 tok/s |
| 2026-11-16 | server-cuda-b12407 / b12407 | Qwen3-Coder-Next UD-IQ4_XS | 131072 / 1 | none | 101k prompt, 30k prefix cached | 71268 / 29743 | 69,437 ms (client TTFT 69.5 s) | 1026 | 56 | 58.6 | 15076 | 31.9 | repeat of the same prompt: 0.16 s, cache_n 101007; q8_0 KV: 1,029 tok/s and 48.7 tok/s |
| 2026-11-16 | server-cuda-b12407 / b12407 | Qwen3-Coder-Next UD-IQ4_XS | 131072 / 1 | none | OpenCode v2.3.0 session, write then read | 4249, then 20 and 128 new | 5,396 ms for the first request | 787 | 119, 81, 57 | 63.9 to 65.5 | 15076 | 31.9 | passed in 13 s; q8_0 KV passed in 11 s |

Raw records for these rows: [research/2026-11-16/llamacpp-gpt-oss-20b/](../research/2026-11-16/llamacpp-gpt-oss-20b/), [research/2026-11-16/llamacpp-qwen3.8-27b/](../research/2026-11-16/llamacpp-qwen3.8-27b/), [research/2026-11-16/llamacpp-qwen3.6-35b-a3b/](../research/2026-11-16/llamacpp-qwen3.6-35b-a3b/README.md), [research/2026-11-16/llamacpp-gemma-4-26b-a4b/](../research/2026-11-16/llamacpp-gemma-4-26b-a4b/), [research/2026-11-16/llamacpp-k2-horizon-7b/](../research/2026-11-16/llamacpp-k2-horizon-7b/), [research/2026-11-16/llamacpp-nemotron-3.5-lightning/](../research/2026-11-16/llamacpp-nemotron-3.5-lightning/), [research/2026-11-16/llamacpp-k2-horizon-mova/](../research/2026-11-16/llamacpp-k2-horizon-mova/), [research/2026-11-16/llamacpp-deepseek-r1-0528-qwen3-8b/](../research/2026-11-16/llamacpp-deepseek-r1-0528-qwen3-8b/), [research/2026-11-16/llamacpp-groot-19b-a3b/](../research/2026-11-16/llamacpp-groot-19b-a3b/), [research/2026-11-16/llamacpp-nemotron-3.5-bolt/](../research/2026-11-16/llamacpp-nemotron-3.5-bolt/) and [research/2026-11-16/llamacpp-qwen3-coder-next/](../research/2026-11-16/llamacpp-qwen3-coder-next/) (probe JSON, server timing log, GPU trace, OpenCode events, launch arguments).

### Qwen3.6-35B-A3B offload variants

Measured on 2026-11-16 with `server-cuda-b12407`, 131072 context, f16 KV, `-np 1` and thinking off.
Each variant ran 3 decodes of a 600-word answer and one 8.6k prompt ([script](../research/2026-11-16/llamacpp-qwen3.6-35b-a3b/variants/variant.sh), [one-line results](../research/2026-11-16/llamacpp-qwen3.6-35b-a3b/variants/variants.jsonl)).
Decode is the server's `predicted_per_second`; the baseline and tuned rows come from the full probe above.

| Variant | Flags on top of `-ngl all -fa on -c 131072 -np 1` | Decode tok/s (3 runs) | Prefill tok/s at 8.6k | Peak VRAM MiB | Container RAM GiB |
|---|---|---|---|---|---|
| baseline | `--n-cpu-moe 21`, mmap, auto threads (4) | 66.0 to 67.5 | 934 | 13696 | 1.9, plus mmapped experts |
| v2 | baseline plus `-lm none` | 66.6 to 67.4 | 2112 | 13714 | 9.6 to 10.7 |
| v3 | v2 plus `-t 8` | 68.0 to 68.9 | 2098 | 13714 | 9.6 to 10.7 |
| v4 | v3 plus `--moe-cache-mib 1536` (CPU share 0) | 111.8 to 112.8 | 2162 | 15250 | 9.6 to 10.7 |
| v5 | v4 with `--moe-cache-cpu-share 0.95` | 83.7 to 86.1 | 2158 | 15250 | 9.6 to 10.7 |
| v6 | v3 with `--n-cpu-moe 25 --moe-cache-mib 3200` | 120.2 to 120.9 | 1989 | 15190 | 11.3 to 12.4 |
| v7 | v3 with `--n-cpu-moe 40 --moe-cache-mib 9600`, 28g cap | 125.3 to 126.6 | 1679 | 15122 | 17.7 to 18.7 |
| tuned | v6 plus `-lv 4`, full probe | 119.7 to 122.7 | 1990 | 15190 | 11.3 to 14.9 |

v6 was adopted: v7 decodes 5% faster but costs 16% of prefill and 6 GiB more pinned RAM, and v4 keeps prefill 8% higher but decodes 7% slower.
Over the tuned run the cache logged an 83.3% hit rate for `ubatch <= 8` (decode) and 39.3% for `ubatch > 8` ([server log](../research/2026-11-16/llamacpp-qwen3.6-35b-a3b/server-timings.log)).

### Gemma 4 26B-A4B offload split

Measured on 2026-11-16 with the same image, 131072 context, f16 KV, `-lm none -t 8 -ctxcp 16` and thinking off; the full rows come from the full probe and the variant from the short one ([records](../research/2026-11-16/llamacpp-gemma-4-26b-a4b/)).

| Launch | Decode tok/s (3 runs) | Prefill tok/s at 9.9k | Peak VRAM MiB | Container RAM GiB | Decode cache hit rate |
|---|---|---|---|---|---|
| `--n-cpu-moe 9 --moe-cache-mib 1152`, full probe (first launch) | 140.0 to 141.9 | 3758 | 15072 | 4.9 to 9.2 | 78.3% |
| `--n-cpu-moe 12 --moe-cache-mib 2368`, short probe | 148.0 to 149.7 | 3550 | 15064 | 5.9 to 7.4 | 91.8% |
| `--n-cpu-moe 12 --moe-cache-mib 2368`, full probe (adopted) | 147.7 to 150.0 | 3716 | 15064 | 5.9 to 10.6 | 92.1% |

As on Qwen3.6, giving the VRAM of whole blocks to the cache raised decode (5.4%) for a small prefill cost.

### Qwen3-Coder-Next offload variants

Measured on 2026-11-16 with the same image, 131072 context, `-lm none -t 8 --cache-ram 4096 -ctxcp 8` and no thinking, while Ollama downloads ran on the E-cores.
The full rows come from the full probe; the short probe ran 3 decodes and one 8k prompt, and the f16 short run added the 30k and 101k prompts ([one-line results](../research/2026-11-16/llamacpp-qwen3-coder-next/variants/variants.jsonl)).

| Launch | KV | Decode tok/s (3 runs) | Decode after 8k / 30k / 101k | Prefill tok/s at 8k | Peak VRAM MiB | Decode cache hit rate |
|---|---|---|---|---|---|---|
| `--n-cpu-moe 38 --moe-cache-mib 2944`, full probe (first launch) | q8_0 | 80.0 to 80.9 | 72.7 / 68.1 / 48.7 | 1116 | 15082 | 68.5% |
| `--n-cpu-moe 38 --moe-cache-mib 2944`, short probe | q8_0 | 80.8 to 81.4 | 74.9 | 1112 | 15080 | 70.6% |
| `--n-cpu-moe 38 --moe-cache-mib 3200`, short probe | q8_0 | 81.2 to 82.9 | 72.9 | 1113 | 15336 | 72.1% |
| `--n-cpu-moe 36 --moe-cache-mib 1664`, 33g cap, short probe | q8_0 | 74.0 to 74.7 | 68.1 | 1155 | 15220 | 59.5% |
| `--n-cpu-moe 38 --moe-cache-mib 1504`, short probe | f16 | 71.1 to 71.7 | 66.6 / 67.8 / 59.8 | 1115 | 15076 | 56.7% |
| `--n-cpu-moe 38 --moe-cache-mib 1504`, full probe (adopted) | f16 | 71.1 to 72.0 | 64.7 / 66.0 / 58.6 | 1114 | 15076 | 55.5% |

The 3,200 MiB cache gained 1% and left 505 MiB free, at the floor, so the 2,944 MiB launch kept more margin.
Two more whole blocks on the GPU (`--n-cpu-moe 36`) decoded 8% slower than the larger cache, the fifth model on which the cache beats whole blocks.
At equal VRAM, f16 KV with the smaller cache was 12% slower at a short context and 3% slower after 30k, but 20% faster after 101k, with the same prefill; it was adopted because it holds decode near 60 tok/s across the 128k window and stores the KV losslessly (see [KV cache type at depth](#kv-cache-type-at-depth)).

Scenario names to reuse: "turn 1, 8k fresh prompt", "turn 2, same conversation", "OpenCode session delta from /metrics", "ncmoe sweep confirm at -c N".

## Sources

- llama.cpp b12418 sources: [llama-bench README](https://github.com/ggml-org/llama.cpp/blob/b12418/tools/llama-bench/README.md), [llama-bench.cpp](https://github.com/ggml-org/llama.cpp/blob/b12418/tools/llama-bench/llama-bench.cpp), [server README](https://github.com/ggml-org/llama.cpp/blob/b12418/tools/server/README.md), [cuda.Dockerfile](https://github.com/ggml-org/llama.cpp/blob/b12418/.devops/cuda.Dockerfile), [tools.sh](https://github.com/ggml-org/llama.cpp/blob/b12418/.devops/tools.sh), [docker.yml](https://github.com/ggml-org/llama.cpp/blob/b12418/.github/workflows/docker.yml)
- [llama.cpp container package on GHCR](https://github.com/ggml-org/llama.cpp/pkgs/container/llama.cpp) (tags `full-cuda-b12407`, `server-cuda-b12407`; registry API read 2026-11-15)
- llama.cpp issues and PRs: [#27330](https://github.com/ggml-org/llama.cpp/issues/27330), [#30357](https://github.com/ggml-org/llama.cpp/issues/30357), [#25859](https://github.com/ggml-org/llama.cpp/issues/25859), [#27623](https://github.com/ggml-org/llama.cpp/issues/27623), [PR #31802](https://github.com/ggml-org/llama.cpp/pull/31802)
- [ggml-org gpt-oss guide, discussion #15396](https://github.com/ggml-org/llama.cpp/discussions/15396)
- [llama.cpp CUDA scoreboard, discussion #15013](https://github.com/ggml-org/llama.cpp/discussions/15013)
- [TechFuelHQ RTX 5080 throughput dataset (iBlessi on Hugging Face)](https://huggingface.co/datasets/iBlessi/rtx-5080-llm-throughput), its [canonical TechFuelHQ page](https://techfuelhq.com/data/rtx-5080-llm-throughput/) and [TechFuelHQ Ollama vs llama.cpp on RTX 5080](https://techfuelhq.com/self-hosted/ollama-vs-llamacpp-rtx-5080-2026/)
- [u/grumd, "Benchmarked 18 models ... RTX 5080 16GB"](https://www.reddit.com/r/LocalLLaMA/comments/1s9mkm1/benchmarked_18_models_that_i_can_run_on_my_rtx/), [the Gemma 4 llama-benchy comment](https://www.reddit.com/r/LocalLLaMA/comments/1sdu8oz/bartowski_vs_unsloth_for_gemma_4/oem5j5k/), [the Qwen3.6 262k comment](https://www.reddit.com/r/LocalLLaMA/comments/1so3rsx/comment/ogrgfvj/) and [the thread-count comment](https://www.reddit.com/r/LocalLLaMA/comments/1s7lo0a/comment/odbkdoe/)
- [LLM SQL Benchmark](https://github.com/nlothian/llm-sql-benchmark)
- [NVIDIA GeForce RTX 50 Series laptops](https://www.nvidia.com/en-us/geforce/laptops/50-series/) and [Intel Core Ultra 9 290HX Plus specifications](https://www.intel.com/content/www/us/en/products/sku/245960/intel-core-ultra-9-processor-290hx-plus-36m-cache-up-to-5-50-ghz/specifications.html)
- Evidence classes and credibility for each source: [../docs/sources.md](../docs/sources.md)

## Maintenance

Before relying on this document, re-check:

- whether a newer `full-cuda-bNNNNN` and `server-cuda-bNNNNN` tag exists, and keep both images on the same build;
- the llama-bench and llama-server option tables against the README of the build you use (flags were renamed before, for example `--no-mmap`);
- the state of [#27330](https://github.com/ggml-org/llama.cpp/issues/27330); if it is fixed, measure once with CUDA graphs on and note the difference;
- [#30357](https://github.com/ggml-org/llama.cpp/issues/30357), [#27623](https://github.com/ggml-org/llama.cpp/issues/27623) with [PR #31802](https://github.com/ggml-org/llama.cpp/pull/31802), and [#25859](https://github.com/ggml-org/llama.cpp/issues/25859), since each changes what the depth and offload results mean;
- the community rows, whose measurements date from 2025-08 to 2026-08; replace them with this laptop's own numbers as soon as the results tables have entries.
