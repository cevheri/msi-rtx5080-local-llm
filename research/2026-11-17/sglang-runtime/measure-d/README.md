# SGLang v0.5.23 measurement D: K2 Horizon 7B at fraction 0.88 and the wave 1 and wave 2 margins under the new runtime variable

Last verified: 2026-11-18.

This folder holds measurement D for the lead who adds SGLang as llmctl's third runtime, as the brief `$SCRATCH/prompts/sglang-measure-d.md` asks, with measurements B and C as the template for method, layout and wording.
Every start used `lmsysorg/sglang:v0.5.23-runtime` (image ID `3ce6da3e5ea0`), one container at a time named `sglang-measure`, `--device nvidia.com/gpu=all --ipc=host`, `--memory 32g --memory-swap 32g`, `-p 127.0.0.1:30000:30000`, the Hugging Face cache read-only, no restart policy, the warm volume `sglang-research-cache`, the model overlay rebuilt under `overlay/` with B's `make_overlay.py` (each equal to B's, link for link and byte for byte in `generation_config.json`), B's draft settings for the recipe, and all eight variables of the runtime file: `HF_HUB_OFFLINE=1`, `HF_MODULES_CACHE=/tmp/hf_modules`, `MAX_JOBS=2`, `FLASHINFER_NO_DOWNLOAD=1`, `SYNC_TOKEN_IDS_ACROSS_TP=1`, `NCCL_MAX_P2P_NCHANNELS=1`, `TIKTOKEN_ENCODINGS_BASE=/opt/sglang/lib/python3.12/site-packages/tml_renderers/data` and `PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1`.
Every llmctl command ran as the copy of main at 5ad8883 (`$SCRATCH/llmctl-main/llmctl`), and the repository was only read.
The enforced GPU power limit read 140 W before the K2 Horizon and Qwen3.5-9B starts and 145 W before the two Gemma starts, against 145 W throughout B (measured, `<record>/logs/<run>.meta`).
Labels: "(measured on the MSI Raider 18 Max HX, SGLang v0.5.23)" for runs here, "(inferred from <what>)" for readings of logs and records, "(estimate)" for arithmetic and for the D8 rule.
"Free" means `nvidia-smi` free memory of the whole GPU; "flushed" means read right after D8's `POST /flush_cache`; times are UTC and dates local (+03:00).

## Answers for the lead

### K2 Horizon 7B at fraction 0.88: passes, draft updated

| Figure | D d1-f088, fraction 0.88 | B b0-a2, fraction 0.90 |
|---|---|---|
| `token_capacity` (FP8) | 77,988, 12,452 above the 65,536 context | 82,397 |
| Free right after ready, and after D8's flush | 1,152 and 1,152 MiB | 865 MiB, flush not read |
| After the fresh 95% prompt (62,254 tokens), the first grammar request and the forced tool request | 828, 828, 828 MiB | 543, 543, 543 MiB |
| Lowest after ready over the whole protocol with M12, and the drop | 822 MiB, 330 MiB | 532 MiB, 333 MiB |
| Decode, probe median | 89.6 tok/s | 89.5 tok/s |
| First token, 8,085 and 30,261 tokens | 2.196 and 9.233 s | 2.198 and 9.249 s |
| M12: requests in 300 s, errors, decode median, second 95% prompt | 51, 0, 86.5 tok/s, 62,257 tokens with 822 MiB left | 52, 0, 87.55 tok/s, 533 MiB left |
| GPU peak; anon plus shmem peak | 15,020 MiB; 4.92 GiB | 15,311 MiB; 4.86 GiB |
| Smoke requests and thinking (always) | all passed: plain, streamed, tool call and its second turn, `json_schema`, Hermes' title, thinking on for every effort and switch, every answer 391 | the same |

All figures are measured on the MSI Raider 18 Max HX, SGLang v0.5.23 ([sglang-k2-horizon-7b/](../../../2026-11-18/sglang-k2-horizon-7b/measure-d/README.md)).
- Pass: pool 77,988 against 65,536, 1,152 MiB against the 896 MiB gate, and a 330 MiB drop against 486 MiB (measured; the gate is the D8 rule's estimate).
- `$SCRATCH/sglang/overrides/sglang/k2-horizon-7b.toml` is updated in place: fraction 0.88, `gpu_mib = 14908` (15,020 minus the desktop's 112 MiB), `ram_gib = 5` (4.92 GiB rounded up), `decode_tok_s = 89.6`, `date = "2026-11-18"`, `record = "$SCRATCH/sglang/measure-d/sglang-k2-horizon-7b/"`, and the summary "89.6 tok/s decode over runs of 1,286 to 1,500 tokens with 64k context and FP8 KV, 77,988-token pool at fraction 0.88, 1,152 MiB free after ready and 828 MiB after a 62,254-token prompt and the first grammar request, 300 s of sustained load without error, 15,020 MiB GPU peak, smoke test and tool call passed"; `memory_gib = 32` is unchanged, and the version before is in [drafts-before/](drafts-before/).
- M13 and M15 were not re-run at 0.88 (not in the brief); B measured both at 0.90: 20 of 20 with FP8 and BF16 KV, and a cold start without a network ready in 137.9 s (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, in B's record).

### Qwen3.5-9B, Gemma 4 12B and Gemma 4 E4B with the new variable: all pass

| Recipe, start | Free right after ready (B) | Flushed | After the 95% prompt (B) | After the first grammar request (B) | After the forced tool request (B) | Lowest to the end of the probe (B) | Drop from ready (B) | Drop from flushed | Decode (B) | Gate |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-9b d2-env | 911 (911) | 937 | 551 (545) | 549 (545) | 549 (545) | 541 (537) | 370 (374) | 396 | 92.8 (93.7) tok/s | passes: 937 and 911 against 896, 396 against 486 |
| gemma-4-12b d2-env | 1,099 (1,099) | 1,119 | 841 (839) | 841 (839) | 841 (839) | 841 (839) | 258 (260) | 278 | 70.1 (70.4) tok/s | passes: 1,119 and 1,099, 278 |
| gemma-4-e4b d2-env | 1,407 (1,407) | 1,923 | 1,627 (1,403) | 1,627 (1,403) | 1,627 (1,403) | 1,627 (1,403) | none, free rose by 220 (4) | 296 | 140.0 (141.3) tok/s | passes: 1,923 and 1,407, 296 |

All values are MiB unless named, measured on the MSI Raider 18 Max HX, SGLang v0.5.23 ([lane-logs/margins-to-probe.txt](lane-logs/margins-to-probe.txt), by [scripts/margins_d.py](scripts/margins_d.py) over each record and over B's); B's values are B's margin-table row, and B's lowest is taken over the same requests, which for all three equals B's lowest over its whole protocol with M12.
- Qwen3.5-9B and Gemma 4 12B: no figure moved by more than 64 MiB; the readings right after ready are equal to B's, the later ones within 6 MiB, and the flush released 26 and 20 MiB, which is the whole difference between the two drops (measured).
- Gemma 4 E4B: every reading after the flush moved by more than 64 MiB, plainly: 1,627 against B's 1,403 MiB after the prompt and both grammar requests (224 MiB more), and a drop of 296 MiB from the flushed reading against B's 4 MiB (measured).
  The cause is D8's flush, not the variables: the server held 516 MiB of PyTorch's cache after its own warm-up, which the flush released (1,407 to 1,923 MiB) and B's unflushed prompt reused; B's own flush before its probe read 1,919 MiB (measured; the mechanism is inferred from the readings, [sglang-gemma-4-e4b/](../../../2026-11-18/sglang-gemma-4-e4b/measure-d/README.md)).
- The variables changed nothing measurable on the three models at B's settings, none of which has a pinned expert store: pools within 9 tokens of B's, equal readings right after ready, flushed readings within 4 MiB of B's flush before its probe, shmem 0.07 to 0.10 GiB as in B, and decode within 1.0% of B's; K2 Horizon's shmem also stayed at B's 0.09 GiB (measured).
- Qwen3.5-9B stays the closest to the gate: 15 MiB above it unflushed and 41 MiB flushed (arithmetic on the measured figures).
- The drafts `qwen3.5-9b.toml`, `gemma-4-12b.toml` and `gemma-4-e4b.toml` are not edited.

### The largest drop over B, C and D, and the threshold

| Lane | Largest drop | Start |
|---|---|---|
| B | 374 MiB | qwen3.5-9b m12-a3 |
| C | 486 MiB | qwen3.6-35b-a3b w3-k88-f088 |
| D | 396 MiB from the flushed reading (370 from the unflushed one) | qwen3.5-9b d2-env |

Measured on the MSI Raider 18 Max HX, SGLang v0.5.23 ([lane-logs/margins-bcd.txt](lane-logs/margins-bcd.txt), every start of B, C and D by `margins_d.py`, which reproduces B's and C's tables).
- The largest drop stays C's 486 MiB, so D8's rule still gives `MIN_FREE` = 768 MiB (486 + 256 = 742, rounded up to a multiple of 64) and the ship gate `F_ready >= 896` MiB (estimate from the measured margins).
- Every recipe measured here passes it: K2 Horizon 7B 1,152, Qwen3.5-9B 937, Gemma 4 12B 1,119 and Gemma 4 E4B 1,923 MiB flushed (measured).
- B's and C's drops are taken from the unflushed reading, D's from both; the flushed reading is the one D8's check makes and gives drops 0 to 26 MiB larger on the three models without a large warm-up cache (measured).
- Open risk (inferred): Qwen3.6's warm flushed reading was never taken; if it equals its cold one after the flush (969 MiB, C's M15), its drop from the flushed reading is 504 MiB and the rule still gives 768 MiB, but from 512 MiB on the rule gives 832 MiB and a 960 MiB gate, which Qwen3.5-9B's 937 MiB would fail.

### End state

- Start: `$SCRATCH/llmctl-main/llmctl status` at 01:23:37 (+03:00) read gemma-4-26b-a4b on llama.cpp running and ready at 15,008 MiB of 16,303 MiB, with OpenCode, Hermes, Pi and dsh on it and Kimi "not installed (no kimi on PATH)"; with the `.kimi-code/bin` prefix Kimi was on it too; `down` at 01:23:44 left 112 MiB in use ([lane-logs/llmctl-status-before.txt](lane-logs/llmctl-status-before.txt), [lane-logs/llmctl-status-before-kimi-path.txt](lane-logs/llmctl-status-before-kimi-path.txt), [lane-logs/llmctl-down.txt](lane-logs/llmctl-down.txt)) (measured).
- End: `PATH=$HOME/.kimi-code/bin:$PATH $SCRATCH/llmctl-main/llmctl up gemma-4-26b-a4b` at 01:43:59 removed llmctl's stopped `llamacpp` container, as `up` does, printed `ready` and no `wrote` line and exited 0; `status` with the same prefix at 01:44:12 read running, health ready, 15,008 MiB of 16,303 MiB, and OpenCode, Hermes, Pi, dsh and Kimi on the running model ([lane-logs/llmctl-up.txt](lane-logs/llmctl-up.txt), [lane-logs/llmctl-status-after.txt](lane-logs/llmctl-status-after.txt)) (measured).
- Containers: llmctl's `llamacpp` running; no `sglang-measure` container; the `-measured-2026-11-16` containers untouched (measured).
- Volumes: none created; `sglang-research-cache`, `vllm-cache`, `flashinfer-cache` and `ollama-models` kept (measured).
- `/dev/shm`: the same 93 files and 16,519,168 bytes used as at the lane start; each start's own root-owned files (four `sglang_loads_*.shm` and one `sem.mp-*`) were removed after its container was gone and are named in its record (measured, [lane-logs/devshm-lane-start.txt](lane-logs/devshm-lane-start.txt)).
- Disk: `/home` 196,301,697,024 bytes free and `/` 301,188,571,136 bytes ([lane-logs/end-disk-volumes.txt](lane-logs/end-disk-volumes.txt)) (measured).
- The owner's model was down from 01:23:44 to 01:43:59 (+03:00), by the lane logs' timestamps; the repository was not edited.

## Starts

| Time (UTC) | Record | Start | Ready line | Poll | Sequence |
|---|---|---|---|---|---|
| 22:24:03 | [sglang-k2-horizon-7b](../../../2026-11-18/sglang-k2-horizon-7b/measure-d/README.md) | d1-f088 | 36.0 s | 36.81 s | flushed reading, 95% prompt, grammar requests, probe, api probe, 300 s load, second 95% prompt |
| 22:36:31 | [sglang-qwen3.5-9b](../../../2026-11-18/sglang-qwen3.5-9b/measure-d/README.md) | d2-env | 28.3 s | 29.27 s | flushed reading, 95% prompt, grammar requests, probe |
| 22:38:06 | [sglang-gemma-4-12b](../../../2026-11-18/sglang-gemma-4-12b/measure-d/README.md) | d2-env | 28.5 s | 29.72 s | the same |
| 22:41:31 | [sglang-gemma-4-e4b](../../../2026-11-18/sglang-gemma-4-e4b/measure-d/README.md) | d2-env | 28.5 s | 29.79 s | the same |

All rows are measured on the MSI Raider 18 Max HX, SGLang v0.5.23 ([lane-logs/starts.txt](lane-logs/starts.txt), by `scripts/starts_table_c.py`); "ready line" is SGLang's "The server is fired up and ready to roll!" after the `docker run` call and "poll" the readiness poll (`/ready` 200 and the served name in `/v1/models`); every container ended `running exit=0 oom=false` before its removal.
Besides these, four throwaway `--rm --ipc=host --network none --memory 256m` containers of the same image removed the `/dev/shm` files each start left.

## Deviations from the brief

1. D8's flush comes before the 95% prompt: each start reads free memory right after ready (the poller, B's reading), then sends `POST /flush_cache` and reads it again (D8's reading), and only then the 95% prompt, so the prompt follows a flush, where B sent it with no flush before it; this is the order every llmctl start has, and both readings and both drops are reported.
2. For the three margin-only starts the lowest free memory runs to the end of the probe, and B's lowest is recomputed over the same requests for the comparison (`margins_d.py --until probe`); B's whole-protocol lowest is the same number for all three.
3. `$SCRATCH/llmctl-main/llmctl status` also ran once at the start with the `.kimi-code/bin` prefix, read-only, so that the start and end states list Kimi alike.
4. The K2 draft's comments on the pool, the cgroup and the draft's origin were updated with the fraction, so that they state this start's figures; nothing else in the draft changed beyond what the brief names.
5. The `/dev/shm` cleanup script compares listings in byte order (`LC_ALL=C`) from the second start on, because `comm` warned about the sort order under `en_US.UTF-8` in the first start; that start's cleanup found and removed its one file correctly.

## Files

- One folder per model, `sglang-<id>/`, with `cmds/` (the `docker run` file, reading `$HOME` and `$SCRATCH`), `logs/` (cut and full logs, the 1 s GPU and cgroup trace, process memory and mappings at ready, `/dev/shm` readings, state, metadata), `shapes/` (readiness and `/server_info`, the sequence record, probe, api probe and sustained load for K2 Horizon, the summary) and a `README.md`.
- [scripts/](scripts/): `run_d.sh` drives a start (C's `run_c.sh` with D's folder and `seq_d.py`), `seq_d.py` is C's `seq_c.py` with the `flushready` phase, `mkcmd_d.py` writes the `docker run` files with the runtime file's eight variables, `margins_d.py` is C's `margins_c.py` with the flushed reading and `--until`, `chain_d1.sh` and `chain_d2.sh` queued the starts, `shm_cleanup.sh` is C's with byte-order comparison; copied unchanged from C: `poll_c.py`, `trace_c.sh`, `procmem.sh`, `maps_inside.py`, `mcommon.py`, `cutlog.py`, `reqlog.py`, `sustained.py`, `api_probe.py`, `anonshmem.py`, `nonreclaim.py`, `make_overlay.py`, `summarize_c.py`, `starts_table_c.py`, `scrub.py`, `margins_c.py`, and `llm_probe.py` (sha256 `ee3fdc0a...`, as in the repository); run the Python scripts with `python3 -I`.
- `overlay/`: the four model overlays and their `*.make.json`; [drafts-before/](drafts-before/): the K2 draft before this lane; [lane-logs/](lane-logs/): llmctl outputs, chain logs, the margin tables, the starts table, the `/dev/shm` reading at the lane start and the end disk and volume reading.
- Local paths in the records were replaced with `$HOME` and `$SCRATCH`, and the user and host names with `<user>` and `<host>`, by `scripts/scrub.py`.
