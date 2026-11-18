# SGLang v0.5.23 on the RTX 5080 Laptop: first hands-on run, 2026-11-17

Result: SGLang ran on this laptop for the first time.
Three of four models served requests: Qwen3.5-9B NVFP4 at 128k, gpt-oss-20b at 128k only with Paged Experts, and K2 Horizon 7B at 64k.
Gemma 4 E4B did not load.
Every "measured" figure below is (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, `lmsysorg/sglang:v0.5.23-runtime`, 2026-11-17) unless it names another runtime.
The GPU power limit read 140 W after the runs; draws up to 150.4 W were sampled during them ([logs/power-limit.txt](logs/power-limit.txt)).

## What ran

1. Disk check: 351.1 GB free on `/`, which holds `/var/lib/docker` (containerd snapshotter); 301.4 GB were free after the pull (measured).
2. `docker pull lmsysorg/sglang:v0.5.23-runtime` took 22 min 45 s.
   Index digest `sha256:3ce6da3e5ea0fee2dac6a23b528cb294fedd31ca74a55a4511f709c5e01e8652`, the digest the docs lane recommended; amd64 manifest `354700f4404a`.
   `docker image ls --tree` shows 49.7 GB disk usage and 14.5 GB content size (measured, [logs/image-inspect.txt](logs/image-inspect.txt)).
   Inside the image: sglang 0.5.23, torch 2.14.1+cu130, flashinfer 0.7.2, sglang-kernel 0.4.10, transformers 5.19.0, triton 3.8.0 and xgrammar 0.2.7; CDI injects `nvidia-smi`, which lists the RTX 5080 Laptop GPU with 16303 MiB and driver 595.99.02, so `NVIDIA_REQUIRE_CUDA=cuda>=13.0` passes (measured).
3. `./llmctl down` stopped `gemma-4-26b-a4b` on llama.cpp at 18:31; `nvidia-smi` then showed 112 MiB in use and no compute process (measured).
4. Twelve SGLang starts by hand as `sglang-research` (`cmds/`), one at a time, each removed after its model.
5. `PATH=$HOME/.kimi-code/bin:$PATH ./llmctl up gemma-4-26b-a4b` brought the owner's model back at 19:08 in 8 s; `./llmctl status` reads running, health ready, 15,008 MiB of GPU memory in use, and every client entry pointing at the running model; no client file was written ([logs/llmctl-up.txt](logs/llmctl-up.txt), [logs/llmctl-status-after.txt](logs/llmctl-status-after.txt)).

Every start used `--device nvidia.com/gpu=all`, `--ipc=host` (the upstream quick start runs "docker run --rm --gpus all --ipc=host", [quickstart.mdx](https://github.com/sgl-project/sglang/blob/v0.5.23/docs/docs/get-started/quickstart.mdx)), `--memory 32g --memory-swap 32g`, `-p 127.0.0.1:30000:30000`, no restart policy, the Hugging Face cache mounted read-only at `/root/.cache/huggingface`, `HF_HUB_OFFLINE=1`, `HF_MODULES_CACHE=/tmp/hf_modules`, `MAX_JOBS=2`, and one named volume, `sglang-research-cache`, on `/root/.cache/sglang`.
The image has no server entrypoint, so each command runs `python3 -m sglang.launch_server ... --host 0.0.0.0 --port 30000`; it works and warns that `sglang serve` is the recommended entrypoint (measured).

| Start | Change against the one before | Result | Ready line after `docker run` |
|---|---|---|---|
| qwen3.5-9b a1 | vLLM recipe flags translated, `--mem-fraction-static 0.85`, `--language-model-only` | Refused at argument resolution | none (13 s) |
| qwen3.5-9b a2 | without `--language-model-only` | Ran; KV pool 74,233 tokens, short of 131072 | 54 s, first start ever |
| qwen3.5-9b a3 | `--mem-fraction-static 0.88 --max-mamba-cache-size 8 --disable-prefill-cuda-graph` | Ran; KV pool 140,976 tokens | 28 s |
| qwen3.5-9b run 4 | the a2 configuration again, 300 s sustained load | Ran, no error | 35 s |
| gpt-oss-20b a1 | 64k, FP8 KV, `--swa-full-tokens-ratio 0.1`, 1,024-token chunks, no prefill graph, fraction 0.95 | Refused: weights leave no KV memory | none |
| gpt-oss-20b a2 | fraction 0.97, as the a1 error asked | Ran with an 11,896-token pool, then crashed on the first grammar request | 95 s |
| gpt-oss-20b a3 | `--moe-runner-backend marlin`, fraction 0.95 | Ran with a 34,726-token pool, then crashed on the first grammar request | 217 s, cold Marlin compile |
| gpt-oss-20b a4 | 128k, Paged Experts with 24 of 32 experts resident, fraction 0.90 | Ran, all probes passed except a forced tool choice | 61 s |
| k2-horizon-7b a1 | vLLM recipe flags translated, fraction 0.90, FP8 KV | Ran; crashed on the first grammar request | 131 s, cold prefill-graph capture |
| k2-horizon-7b a2 | `--disable-prefill-cuda-graph` | Ran, grammar requests passed | 34 s |
| gemma-4-e4b a1 | vLLM recipe flags translated, fraction 0.85, no prefill graph | Model construction failed | none |
| gemma-4-e4b a2 | `--json-model-override-args '{"architectures":["Gemma4ForCausalLM"]}'` | Refused: no processor | none |

Times are from the `docker run` call to the log line "The server is fired up and ready to roll!" ([logs/ready-times.txt](logs/ready-times.txt)) (measured).

## Qwen3.5-9B NVFP4 (`RedHatAI/Qwen3.5-9B-NVFP4` at `a4df85a20fb7`)

- a1 failed: `ValueError: --language-model-only does not support ['Qwen3_5ForConditionalGeneration']. Supported: ['DeepseekV4ForCausalLM', 'MuseGlimmerForConditionalGeneration', 'Cosmos3ForConditionalGeneration', 'Cosmos3EdgeForConditionalGeneration'].` (measured).
  So the vision tower always loads: weights take 8.39 GB in SGLang against 7.48 GiB language-only on vLLM (measured).
  SGLang's "GB" in these log lines are GiB (inferred from its 15.06 GB free before loading against the 16,611,278,848-byte device total that the same build logs).
- a2 memory (measured): `Mamba Cache is allocated. max_mamba_cache_size: 41, conv_state size: 0.05GB, ssm_state size: 1.97GB`; `KV Cache is allocated. dtype: torch.bfloat16, #tokens: 74233, K size: 1.13 GB, V size: 1.13 GB`; prefill CUDA graph (42 shapes, breakable) 0.71 GB in 20.95 s; decode graph `bs=[1]` 0.05 GB; 1.15 GB free after start, 0.76 GiB after the lazily loaded Triton kernels.
  SGLang warned: "KV pool holds 74233 tokens, fewer than --context-length 131072.
  One full-length request cannot fit, so long requests will be refused or truncated at serving time."
  The mamba pool is auto-sized from `mamba_full_memory_ratio` 0.9, and with one running request it spent 2 GB on 41 state slots (inferred from the log and [kv_cache_configurator.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/mem_cache/kv_cache_configurator.py): "Use ratio-based calculation to auto-fit available memory").
- a3 memory (measured): 8 mamba slots took 0.43 GB, the KV pool held 140,976 BF16 tokens (2.15 + 2.15 GB), and 1.35 GB stayed free after start.
- GPU peak by `nvidia-smi`: 15,806 MiB (a2) and 15,810 MiB (a3), against 13,726 MiB for vLLM (measured; vLLM figure from the repository record `research/2026-11-17/vllm-qwen3.5-9b/`).
  Host RAM: anonymous memory peaked at 5.61 GiB (measured, cgroup `memory.stat`).
- Attention: "Attention backend not specified.
  Use flashinfer backend by default."; GDN layers on Triton kernels; the FP4 GEMM runner stayed `auto` and the log does not name the kernel it picked (measured).
- Decode: 92.7 tok/s on three 512-token answers (a2), 92.9 tok/s (a3); the repository probe gave a median of 92.8 tok/s over answers of 643 to 813 tokens, against 90.8 tok/s on vLLM v0.33.0 (measured).
- Prefill after `/flush_cache`: 0.580 s to the first token for 8,591 tokens and 2.426 s for 32,153 tokens with the prefill graph (a2), 0.566 s and 2.394 s without it (a3), against 0.592 s and 2.424 s on vLLM (measured).
- Long prompts: 67,955 tokens took 8.27 s to the first token (8,217 tok/s) on a2, and 124,667 tokens took 23.13 s (5,390 tok/s) on a3; the same prompts again took 0.042 s and 0.057 s, so the radix cache works for this hybrid model; decode at 124.7k tokens of depth ran at 56.8 tok/s by the server log (measured).
- Sustained load (run 4, both CUDA graph types on): 66 back-to-back requests in 300 s with fresh prompts of up to 13,448 tokens, no error, decode 86.7 to 93.5 tok/s (median 91.3, first half 91.7, second half 90.6), at most 70 C and 144.5 W (measured, [shapes/qwen3.5-9b/sustained-a4.json](../../sglang-qwen3.5-9b/handson/shapes/sustained-a4.json)).

## gpt-oss-20b (`openai/gpt-oss-20b` at `6cee5e81ee83`)

- a1 failed (measured): "Detected SM120 and MXFP4 quantization format for GPT-OSS model, enabling FlashInfer CUTLASS MXFP4 MOE kernel.", then `Load weight end. ... avail mem=0.60 GB, mem usage=14.45 GB.` and `ValueError: Loaded weights leave no GPU memory for the KV cache under --mem-fraction-static=0.95. Raise --mem-fraction-static above 0.961 (minimum viable = 1 - available/pre = 0.9601).`
  Weight loading also logged 15 `CUDACachingAllocator` OOM warnings for 289,406,976-byte allocations, which it survived.
- a2 (fraction 0.97) started with `Full KV Cache ... #tokens: 11896` and 1,189 SWA tokens, 0.46 GB free; FlashInfer autotuned the MoE GEMMs for 55 s on the first start, cached under `/root/.cache/sglang/flashinfer/autotune/0.7.2/sm120/` (measured).
  Short replies decoded at 124 to 136 tok/s by the server log (measured).
- a3 (Marlin MoE runner, fraction 0.95): weights 13.86 GB, the same as vLLM's Marlin path; FP8 KV pool 34,726 tokens plus 3,472 SWA tokens; the first decode-graph capture took 173.49 s, which the log does not explain; 0.67 GB free (measured).
  Decode 146.9 tok/s over three 512-token runs (145.9 to 146.9); a 32,473-token prompt took 4.578 s to the first token (7,093 tok/s) (measured).
- a2 and a3 crashed on the first grammar-constrained request (`tool_choice: "required"`), with the same error (measured):

  ```text
  Triton kernel 'apply_token_bitmask_inplace_kernel' device-loaded after serving started (free device mem: 0.44 GiB).
  Scheduler hit an exception ... layers/sampler.py:726 in _sync_token_ids_across_tp
  torch.distributed.DistBackendError: NCCL error ... unhandled cuda error ... NCCL version 2.30.7
  Failed to CUDA calloc 536870912 bytes
  ```

  The server then exited; the client got HTTP 500 and every later request a connection reset.
  Upstream knows the cause: the open PR [#43302](https://github.com/sgl-project/sglang/pull/43302) says "the final-token-id `all_reduce` is gated only on `SYNC_TOKEN_IDS_ACROSS_TP` (env, default off) or non-empty `sampling_info.grammars`.
  It does not check the size of the sync group.", and [#48487](https://github.com/sgl-project/sglang/pull/48487), "[Fix] sampler: skip the token-id all-reduce when the TP sync group has one rank", was merged on 2026-11-06, after the v0.5.23 tag of 2026-10-29, so v0.5.23 does not have it (inferred from the GitHub search API, read 2026-11-17, `data-gh/`).
- a4, Paged Experts (new in v0.5.23): `--moe-runner-backend marlin --enable-paged-experts --paged-experts-num-resident 24` kept 24 of 32 experts per layer on the GPU (measured):
  - weights 11.46 GB on the GPU, FP8 KV pool 168,443 tokens plus 16,844 SWA tokens (2.12 GB), 1.45 GB free;
  - decode 133.8 tok/s over 512-token runs (133.0 to 134.1), and a median of 125.5 tok/s in the repository probe over answers of 1,464 to 1,500 tokens;
  - 1.452 s to the first token for 8,058 tokens and 4.893 s for 30,003 tokens after `/flush_cache`;
  - 123,958 tokens took 41.69 s to the first token (2,973 tok/s), 0.209 s again, and decode at that depth ran at 56.7 to 72.5 tok/s by the server log;
  - host anonymous memory peaked at 13.94 GiB against 5.47 GiB without paging, because the pinned host store holds all 32 experts of every layer, and the cgroup peak with page cache was 28.24 GiB of the 32 GiB cap; Docker's default memlock limit did not stop the pinning;
  - the grammar requests did not crash this server, which had about 1.4 GiB free.
  The docs say the store "is page-locked, so the host must be able to pin that much memory" ([paged_experts.mdx](https://github.com/sgl-project/sglang/blob/v0.5.23/docs/docs/advanced_features/paged_experts.mdx)) (vendor-reported).
- For comparison on the same probe script: llama.cpp b12407 decodes 166.2 tok/s with 1.124 s and 4.048 s to the first token at 128k, and vLLM v0.33.0 in eager mode 83.5 tok/s with 1.809 s and 8.867 s at 64k (measured, repository records `research/2026-11-16/llamacpp-gpt-oss-20b/` and `research/2026-11-16/vllm-gpt-oss-20b/`).

## K2 Horizon 7B AWQ-INT4 (`cyankiwi/K2-Horizon-7B-AWQ-INT4` at `530d5ceccd8e`)

- The native `K2HorizonForCausalLM` class loaded with `--trust-remote-code`; weights 7.89 GB, FP8 KV pool 82,394 tokens (2.83 + 2.83 GB), FlashInfer attention, auto-detected `k2_horizon` parsers and content format `string` (measured).
- a1: the first prefill-graph capture took 96.23 s and 0.58 GB, which left 0.40 GB free; the first grammar request then crashed the server with `Failed to CUDA calloc 536870912 bytes` at 0.29 GiB free (measured).
- a2 without the prefill graph: 0.98 GB free; `tool_choice: "required"` returned a parsed `get_weather` call with `finish_reason` `tool_calls`, `json_schema` returned `{"capital":"Paris"}`, and the server stayed up (measured).
- Decode 90.1 tok/s over 512-token runs and a median of 89.9 tok/s over 1,500-token runs, against 94.8 tok/s on vLLM v0.33.0 (measured; vLLM from `research/2026-11-17/vllm-k2-horizon-7b/`).
- 2.426 s to the first token for 8,085 tokens and 9.142 s for 30,261 tokens, against 2.227 s and 10.869 s on vLLM; 59,253 tokens took 27.73 s (2,137 tok/s), and 0.077 s again (measured).
- GPU peak 15,814 MiB, host anonymous memory 4.89 GiB at most (measured).

## Gemma 4 E4B QAT W4A16 (`google/gemma-4-E4B-it-qat-w4a16-ct` at `6cd26aaa2357`)

- a1 failed while building the audio tower (measured): `models/gemma4_audio.py:785` to `layers/clippable_linear.py:210`, `orig_loader = self.linear.weight.weight_loader`, `AttributeError: 'MergedColumnParallelLinear' object has no attribute 'weight'`.
  The encoders are built with the checkpoint's quantization config ([gemma4_mm.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/models/gemma4_mm.py): `self.audio_tower = Gemma4AudioEncoder(config=config.audio_config, quant_config=quant_config, ...)`), so the W4A16 linear has packed weights where the clippable layer expects `weight` (inferred from the traceback and that source).
- a2 failed: `ValueError: No processor registered for architecture: ['Gemma4ForCausalLM'].`, because the config still counts as multimodal (measured).
- Stopped after two attempts: the listed fixes (attention backend, CUDA graphs, memory fraction, context) act after model construction, `--language-model-only` accepts only four other architectures (a1 of Qwen), and `enable_multimodal` is switched off by default only for "Gemma3ForConditionalGeneration", "Llama4ForConditionalGeneration", "Step3VLForConditionalGeneration" and "InklingForConditionalGeneration" ([model_config.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/configs/model_config.py)) (inferred from source).
  A search for `ClippableGLUParallelLinear` in the SGLang issues returned nothing on 2026-11-17 (`data-gh/`).

## Working docker run commands

Qwen3.5-9B at 128k ([cmds/qwen3.5-9b-a3.sh](../../sglang-qwen3.5-9b/handson/cmds/qwen3.5-9b-a3.sh)):

```bash
docker run -d --name sglang-research \
  --device nvidia.com/gpu=all --ipc=host \
  --memory 32g --memory-swap 32g \
  -p 127.0.0.1:30000:30000 \
  -e HF_HUB_OFFLINE=1 -e HF_MODULES_CACHE=/tmp/hf_modules -e MAX_JOBS=2 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface:ro" \
  -v sglang-research-cache:/root/.cache/sglang \
  lmsysorg/sglang:v0.5.23-runtime \
  python3 -m sglang.launch_server \
    --model-path /root/.cache/huggingface/hub/models--RedHatAI--Qwen3.5-9B-NVFP4/snapshots/a4df85a20fb7e54740f564f15af0c2f5d9c6616c \
    --served-model-name qwen3.5-9b --context-length 131072 \
    --host 0.0.0.0 --port 30000 \
    --mem-fraction-static 0.88 --max-running-requests 1 \
    --max-mamba-cache-size 8 --disable-prefill-cuda-graph \
    --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking":false}'
```

gpt-oss-20b at 128k ([cmds/gpt-oss-20b-a4.sh](../../sglang-gpt-oss-20b/handson/cmds/gpt-oss-20b-a4.sh)), same `docker run` options:

```bash
  python3 -m sglang.launch_server \
    --model-path /root/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee \
    --served-model-name gpt-oss-20b --context-length 131072 \
    --host 0.0.0.0 --port 30000 \
    --mem-fraction-static 0.90 --max-running-requests 1 \
    --kv-cache-dtype fp8_e4m3 --swa-full-tokens-ratio 0.1 \
    --chunked-prefill-size 1024 --disable-prefill-cuda-graph \
    --moe-runner-backend marlin --enable-paged-experts --paged-experts-num-resident 24 \
    --tool-call-parser gpt-oss --reasoning-parser gpt-oss
```

K2 Horizon 7B at 64k ([cmds/k2-horizon-7b-a2.sh](../../sglang-k2-horizon-7b/handson/cmds/k2-horizon-7b-a2.sh)), same `docker run` options:

```bash
  python3 -m sglang.launch_server \
    --model-path /root/.cache/huggingface/hub/models--cyankiwi--K2-Horizon-7B-AWQ-INT4/snapshots/530d5ceccd8ea660ceaa2a9f3a4bcbb68ba44291 \
    --served-model-name k2-horizon-7b --context-length 65536 \
    --host 0.0.0.0 --port 30000 \
    --mem-fraction-static 0.90 --max-running-requests 1 --disable-prefill-cuda-graph \
    --trust-remote-code --kv-cache-dtype fp8_e4m3 \
    --tool-call-parser k2_horizon --reasoning-parser k2_horizon \
    --default-chat-template-kwargs '{"reasoning_effort":"high","tool_call_format":"xml"}'
```

## API behaviour the clients will meet

Measured with [scripts/api_probe.py](scripts/api_probe.py) on Qwen3.5-9B, gpt-oss-20b and K2 Horizon 7B unless a line names one model; the shapes are in `shapes/`.

- `/v1/models` lists the served name with `max_model_len` equal to `--context-length`, also when the KV pool is smaller (131072 listed over a 74,233-token pool); keys `created`, `id`, `max_model_len`, `object`, `owned_by` (`sglang`), `parent`, `root`.
- `/health` answered 503 during warmup and 200 afterwards, and each call takes 1.0 s because it generates a token; `/ready` answered 200 in 1 ms once up; its behaviour during startup was not probed.
- The request `model` is not checked: `not-the-served-name` got HTTP 200.
- Replies carry `reasoning_content`, never `reasoning`, and the key is present even with thinking off; responses add a top-level `metadata` key and a per-choice `matched_stop`.
- `usage` adds a top-level `reasoning_tokens`; `completion_tokens_details` is null when nothing was reasoned and `{"reasoning_tokens": N}` otherwise; `prompt_tokens_details` is null.
- Streaming: delta keys `role`, `content`, `reasoning_content` and `tool_calls`; with `stream_options.include_usage` a final chunk has `choices: []` and the usage, without it no `usage` key appears; the stream ends with `data: [DONE]`.
- Tool calls: `id` like `call_36c73926dfc244e099ab9541`, `type` `function`, `index`, `arguments` as a JSON string, `finish_reason` `tool_calls`, and `content` an empty string rather than null; streamed, Qwen sent a call in 4 fragments and gpt-oss and K2 in 1.
  Replaying the assistant call and a `role: tool` result produced a correct final answer on all three models.
- `"tools": []`, `tool_choice: "auto"` without tools, `prompt_cache_key` and `max_completion_tokens` all returned 200; vLLM returns 400 for the first two (repository vllm/README.md).
- `max_tokens` above the context returns HTTP 400 with `{"object":"error","message":"max_completion_tokens is too large: 148466.This model supports at most 131072 completion tokens.","type":"BadRequestError","param":null,"code":"400"}`.
- Grammar-constrained requests (`tool_choice: "required"`, `response_format` `json_schema`) work with xgrammar, but on v0.5.23 the first one allocates 512 MiB through NCCL and kills the whole server when less than that is free (three crashes above); keep about 0.6 GiB free after start until a release carries #48487.
- Forced tool choice on gpt-oss returned HTTP 200, `finish_reason` `stop`, no `tool_calls`, and raw Harmony text (`<|channel|>commentary to=functions.get_weather<|constrain|>json<|message|>{"city...`) in `content`, as issue [#41747](https://github.com/sgl-project/sglang/issues/41747) reports (third-party report as quoted by the docs lane: "comes back as HTTP 200 with `finish_reason: "stop"`, no `tool_calls`").
- Qwen thinking: `chat_template_kwargs.enable_thinking` true and false work, the server default off holds without the field, and `reasoning_effort` `none` keeps it off, but `low` and `high` turn it on despite the server default (858 and 771 characters of reasoning).
  With thinking on, `content` starts with two newlines (`"\n\n391"`).
- gpt-oss: every `reasoning_effort` (`none`, `minimal`, `low`, `medium`, `high`, `xhigh`) returned 200 with no effort map; `none` still reasoned (43 characters), and `enable_thinking: false` changed nothing.
- K2 Horizon always reasons; every effort returned 200, `none` included (1,174 characters).
  A replayed assistant turn with only a `reasoning` field returned HTTP 400 "Assistant message is missing a thinking field.
  Provide one of: think, reasoning, reasoning_content, think_fast, think_faster.", and the same turn with `reasoning_content` returned 200; SGLang drops `reasoning` before the template sees it, as [#44931](https://github.com/sgl-project/sglang/issues/44931) describes (docs lane quote: "`reasoning` and a top-level `thinking` key are dropped at request validation.").
- Sampling: Qwen3.5-9B's `generation_config.json` holds no sampling values, so with the default `--sampling-defaults model` SGLang falls back to "SGLang/OpenAI defaults (temperature=1.0, top_p=1.0, etc.)" ([serving.py](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/arg_groups/fields/serving.py)); `--preferred-sampling-params` is documented only as "json-formatted sampling settings that will be returned in /get_model_info" (vendor-reported).

## Startup, caches and network

- First starts compiled and tuned: Qwen's prefill-graph capture took 20.95 s cold and 6.98 s warm, gpt-oss FlashInfer autotune 55 s, the first Marlin decode-graph capture for gpt-oss 173.49 s, and K2's first prefill-graph capture 96.23 s; later starts reached the ready line in 28 to 61 s (measured).
- Host anonymous memory stayed at or below 5.61 GiB in every start without Paged Experts, and the 300 s watchdog never fired (measured); whether `MAX_JOBS=2` was honoured was not checked.
- `sglang-research-cache` held 60 MB after all runs (FlashInfer autotune, Inductor, JIT 26 MB, Triton 25 MB); it is still on the machine (measured).
- With the cache mounted read-only and `HF_HUB_OFFLINE=1`, no log shows a write error; socket tables read inside the container after the probes list only its own address 172.17.0.2, the bridge gateway 172.17.0.1 and loopback (measured, a snapshot rather than proof of no traffic; [scripts/proc_net.py](scripts/proc_net.py)).
- CUDA graphs: decode graphs ran in every start and both graph types survived 300 s of sustained load on Qwen; no hang like llama.cpp's #27330 (measured).

## Unknowns

- not run: Gemma 4 12B, Groot-19B-A3B, Mellum2.1, DeepSeek-R1-0528-Qwen3-8B and the larger MoE models, for time.
- not run: the five clients end to end; OpenCode's provider needs `reasoningField` `reasoning_content` (inferred from the reply shape).
- not run: FP8 KV accuracy (the gpt-oss log warns "Using FP8 KV cache but no scaling factors provided. Defaulting to scaling factors of 1.0.").
- not run: Paged Experts at other K values, its page-in statistics, and the MoE models it targets (Qwen3.6-35B-A3B, Nemotron 3.5 Lightning, Gemma 4 26B-A4B).
- not run: MTP speculative decoding, although the Qwen3.5-9B checkpoint ships `model_mtp.safetensors`.
- Unknown: which FP4 GEMM kernel Qwen's `auto` runner chose, what the 173 s Marlin capture spent its time on, and whether a later release fixes the Gemma 4 audio-tower failure.
- Unknown: whether `/ready` returns 503 during startup on this build, and whether `--preferred-sampling-params` changes request defaults.

## Files

- `cmds/`: every `docker run`, one file per start.
- `logs/`: cut container logs per start (`attemptN.log`, `run4-sustained.log`), `nvidia-smi` and cgroup traces (`trace*.csv`, one row per second), start times, container arguments (`*.cmd.json`), socket tables, the pull log, image facts, power limit, and the `llmctl` outputs with `$HOME` for the home path.
- `shapes/`: request and reply shapes from `api_probe.py`, the repository probe `research/2026-11-16/llm_probe.py` and `sustained.py`.
- [scripts/](scripts/): the helpers, run with `python3 -I` or bash.
- `data-gh/`: GitHub search API results, read as data only.
