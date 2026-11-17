# K2 Horizon 7B AWQ-INT4 on SGLang, 2026-11-18

Last verified: 2026-11-18.

Result: measured in measurement D: at `--mem-fraction-static 0.88` with the eight variables of the runtime file, K2 Horizon 7B holds a 77,988-token FP8 pool for its 65,536 context, left 1,152 MiB free right after ready and after D8's `POST /flush_cache`, held a fresh 62,254-token prompt, the first grammar request and 300 s of load with at least 822 MiB free, decoded 89.6 tok/s and passed the smoke test and tool call (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
Its 330 MiB drop and 1,152 MiB after ready pass the gate of D8's rule over measurements B, C and D (`MIN_FREE` 768 MiB, ship gate 896 MiB), which B's 865 MiB at fraction 0.90 did not (estimate for the rule; the figures are measured).
The start ran by hand with `docker run` before llmctl had an SGLang recipe, at 22:24 UTC on 2026-11-17, which is 01:24 on 2026-11-18 local time (+03:00).
The earlier records of this model, from the hands-on run and measurements A and B at fraction 0.90, are in [research/2026-11-17/sglang-k2-horizon-7b/](../../2026-11-17/sglang-k2-horizon-7b/README.md).

Image: `lmsysorg/sglang:v0.5.23-runtime`.
Model: `cyankiwi/K2-Horizon-7B-AWQ-INT4` at revision `530d5ceccd8ea660ceaa2a9f3a4bcbb68ba44291`, served as `k2-horizon-7b`.
Shipped settings (d1-f088, [measure-d/cmds/d1-f088.sh](measure-d/cmds/d1-f088.sh)): context 65536, `--mem-fraction-static 0.88`, `--max-running-requests 1`, `--disable-prefill-cuda-graph`, `--trust-remote-code`, FP8 KV, parsers `k2_horizon` and `--default-chat-template-kwargs '{"reasoning_effort":"high","tool_call_format":"xml"}'`, with `--device nvidia.com/gpu=all --ipc=host --memory 32g --memory-swap 32g`, the Hugging Face cache mounted read-only, a model overlay whose `generation_config.json` carries temperature 1.0 and top_p 0.95 (B's overlay, rebuilt), and the eight variables of the runtime file: `HF_HUB_OFFLINE=1`, `HF_MODULES_CACHE=/tmp/hf_modules`, `MAX_JOBS=2`, `FLASHINFER_NO_DOWNLOAD=1`, `SYNC_TOKEN_IDS_ACROSS_TP=1`, `NCCL_MAX_P2P_NCHANNELS=1`, `TIKTOKEN_ENCODINGS_BASE=/opt/sglang/lib/python3.12/site-packages/tml_renderers/data` and `PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1`.
The reports are [measurement D's README for this model](measure-d/README.md) and [measurement D's lane README](../../2026-11-17/sglang-runtime/measure-d/README.md), where every figure below comes from.

## Measurement D

- Readiness: the ready line came 36.0 s after `docker run` and `/ready` answered 200 at 36.55 s; the sampling line read `{'temperature': 1.0, 'top_p': 0.95}` (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
- Pool: 77,988 FP8 tokens, 4,409 fewer than B's 82,397 at fraction 0.90 and 12,452 above the context; `startup_available` read 1.258 GiB against B's 0.977 (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
- Margins: 1,152 MiB free right after ready and after D8's flush, 828 MiB after the fresh 62,254-token prompt, the first grammar request and the forced tool request, and 822 MiB at the lowest after ready over the whole protocol, a 330 MiB drop against B's 333 MiB (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
- Decode 89.6 tok/s median in the repository probe against B's 89.5, and 2.196 s and 9.233 s to the first token for 8,085 and 30,261 tokens (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
- M12: 51 requests in 300 s with no error, decode 78.3 to 89.7 tok/s (median 86.5), and a second 62,257-token prompt left 822 MiB (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
- Peaks: 15,020 MiB of GPU memory, 147.0 W in 1 s samples, 74 C, and 4.92 GiB of anon plus shmem, so the draft's `gpu_mib` 14908 and `ram_gib` 5 by D5's rule, with `memory_gib` 32 unchanged (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
- Smoke requests: plain and streamed `OK`, a `get_weather {"city": "Paris"}` call and its second turn, `json_schema` and Hermes' title request passed; the model reasoned for every effort and switch, as in B (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
- `PYTORCH_ALLOC_CONF` changed nothing measurable on this model, which has no pinned expert store: shmem stayed at 0.09 GiB as in B (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-18).
- M13 and M15 were not run again at 0.88; B measured both at 0.90 (measurement D's README for this model).
- The power limit read 140 W before the start, against 145 W in B (measured, [measure-d/logs/d1-f088.meta](measure-d/logs/d1-f088.meta)).

## Files

| Path | Holds |
|---|---|
| [measure-d/README.md](measure-d/README.md) | Measurement D's report on this model, as the lane wrote it; no link needed a rewrite |
| [measure-d/cmds/](measure-d/cmds/) | The `docker run` of the start d1-f088; it reads `$HOME` and `$SCRATCH` |
| [measure-d/logs/](measure-d/logs/) | The cut and full logs, the `docker run` output, the 1 s GPU and cgroup trace, the readiness poll, the process memory, the mappings inside the container at ready, the request log, the sequence output, the state, the metadata and the `/dev/shm` readings |
| [measure-d/shapes/](measure-d/shapes/) | Readiness and `/server_info`, the sequence records, the repository probe, the api probe, the sustained load and the summary |
| [measure-d/overlay/](measure-d/overlay/) | The overlay's generation config before and after, and its link count |
