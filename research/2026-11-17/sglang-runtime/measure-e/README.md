# SGLang v0.5.23 measurement E: Qwen3.6-35B-A3B's post-ready margin from D8's flushed reading

Last verified: 2026-11-18.

This folder holds measurement E for the lead who adds SGLang as llmctl's third runtime, as the brief `$SCRATCH/prompts/sglang-measure-e.md` asks, with measurements C and D as the template for method, layout and wording.
It is one warm start, e1-k88-f088-mem27, of measurement C's shipped Qwen3.6-35B-A3B settings as the draft `$SCRATCH/sglang/overrides/sglang/qwen3.6-35b-a3b.toml` and C's command file `w3-k88-f088-mem27.sh` give them: K 88 of 256 experts resident, Marlin, fraction 0.88, BF16 KV, 8 Mamba slots, 128k, the runtime file's eight variables including `PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1`, `--memory 27g --memory-swap 27g` and the warm volume `sglang-research-cache` ([sglang-qwen3.6-35b-a3b/](../../../2026-11-18/sglang-qwen3.6-35b-a3b/measure-e/README.md)).
The container was `lmsysorg/sglang:v0.5.23-runtime` (image ID `3ce6da3e5ea0`) named `sglang-measure`, with `--device nvidia.com/gpu=all --ipc=host`, `-p 127.0.0.1:30000:30000`, the Hugging Face cache read-only, no restart policy, and C's model overlay `$SCRATCH/sglang/measure-c/overlay/qwen3.6-35b-a3b/` mounted read-only at `/llmctl/model`, as C's command file names it.
Every llmctl command ran as the copy of main at 5ad8883 (`$SCRATCH/llmctl-main/llmctl`), and the repository was only read.
The enforced GPU power limit read 145 W before the start, and the desktop held 112 MiB ([sglang-qwen3.6-35b-a3b/logs/e1-k88-f088-mem27.meta](../../../2026-11-18/sglang-qwen3.6-35b-a3b/measure-e/logs/e1-k88-f088-mem27.meta)) (measured).
Labels: "(measured on the MSI Raider 18 Max HX, SGLang v0.5.23)" for this run and for figures read from the records of measurements B, C and D, "(inferred from <what>)" for conclusions drawn from logs and records, "(estimate)" for arithmetic and for the D8 rule.
"Free" means `nvidia-smi` free memory of the whole GPU; "flushed" means read right after D8's `POST /flush_cache`; times are UTC and dates local (+03:00).

## Answers for the lead

### The figures

| Figure | e1-k88-f088-mem27 |
|---|---|
| Free right after ready, without a flush | 951 MiB, as in both of C's starts of these settings |
| Free after D8's `POST /flush_cache`, the reading llmctl's check takes | 977 MiB, at once and 2 s later |
| After the fresh 124,502-token prompt (95.0% of 131,072), after the first grammar request (Hermes' title `json_schema`) and after a forced tool request | 505, 505 and 505 MiB |
| Lowest over everything after the flushed reading, in the 1 s trace and every reading, through 300 s of sustained load and a second 124,503-token prompt | 505 MiB |
| Drop from the flushed reading | 472 MiB (446 MiB from the unflushed reading) |
| M12 | 91 requests in 300 s, no error, decode median 155.25 tok/s |

All figures are measured on the MSI Raider 18 Max HX, SGLang v0.5.23 ([sglang-qwen3.6-35b-a3b/README.md](../../../2026-11-18/sglang-qwen3.6-35b-a3b/measure-e/README.md), [lane-logs/margins-qwen3.6.txt](lane-logs/margins-qwen3.6.txt)).
- The flush released 26 MiB (951 to 977 MiB); the first 95% prompt then took 472 MiB, and nothing after it took more: every reading and every 1 s sample from 22:58:58 to the end read 505 MiB (measured).
- The draft's `ram_gib = 23` and `memory_gib = 27` still hold: anon plus shmem peaked at 22.76 GiB and the non-reclaimable total at 22.98 GiB under the 27 GiB cap, without OOM (measured).

### The threshold and the gate over B, C, D and E

| Lane | Largest drop | Start | Drop taken from |
|---|---|---|---|
| B | 374 MiB | qwen3.5-9b m12-a3 | the unflushed reading |
| C | 486 MiB | qwen3.6-35b-a3b w3-k88-f088 | the unflushed reading |
| D | 396 MiB | qwen3.5-9b d2-env | the flushed reading |
| E | 472 MiB | qwen3.6-35b-a3b e1-k88-f088-mem27 | the flushed reading |

Measured on the MSI Raider 18 Max HX, SGLang v0.5.23 ([lane-logs/margins-bcde.txt](lane-logs/margins-bcde.txt): all 15 starts of B, C, D and E by [scripts/margins_e.py](scripts/margins_e.py), which reproduces D's table for B, C and D row for row in [lane-logs/margins-bcd-check.txt](lane-logs/margins-bcd-check.txt)).
The lowest of each start runs to the end of its sequence, from the flushed reading where there is one (D and E) and from the ready reading otherwise (B and C), as B, C and D took it.
- Over B's, C's, D's and this run's drops the largest stays C's 486 MiB, so D8's rule gives `MIN_FREE` = 768 MiB (486 + 256 = 742, rounded up to a multiple of 64) and the ship gate `F_ready >= 896` MiB, both unchanged (estimate from the measured margins).
- Without C's two Qwen3.6 rows, whose unflushed reading this run replaces, the largest drop is this run's 472 MiB and the rule gives the same 768 and 896 MiB (472 + 256 = 728) (estimate).
- No shipped recipe misses 896 MiB: Qwen3.5-9B 937 MiB flushed (41 above), Qwen3.6 977 (81 above), Gemma 4 12B 1,119, K2 Horizon 7B 1,152, Gemma 4 E4B 1,923, and gpt-oss-20b 1,346 to 1,347 in C, read without a flush, which a flush can only raise (readings measured; the gate is the rule's estimate; that a flush cannot lower free memory is inferred from what it releases).

### The case the brief names: a drop of exactly 512 MiB

- The brief's arithmetic uses C's lowest of 465 MiB, and against this run's flushed 977 MiB, the reading above "about 976" that the brief names, it gives a drop of exactly 512 MiB (arithmetic on measured figures).
- That 465 MiB came after requests this run did not send, C's repository probe and api probe between the forced tool request and M12, which left that start at 469 and 467 MiB; the same probe left C's mem27 start at 503 MiB, and in the brief's order the lowest here was 505 MiB (measured).
- Counted that way, 512 + 256 = 768 is already a multiple of 64, so the brief's rule keeps `MIN_FREE` = 768 and the gate 896 MiB (estimate).
- The design's D8 text says "the next multiple of 64 above", D's `margins_d.py` implements that wording, and the brief's "Why" paragraph reads the same way ("larger than 511 MiB" raises the threshold to 832); for a sum of exactly 768 that wording gives 832 MiB and a 960 MiB gate, which Qwen3.5-9B (937 MiB flushed) would miss and Qwen3.6 (977) would pass by 17 MiB (estimate).
- For every measured drop the two wordings agree (742 and 728 are not multiples of 64); they part only at this combined 512 MiB, so the spec's D8 sentence and the brief's rule should say the same thing before `min_free_at_ready_mib` goes into the runtime file (inferred; the choice is the lead's).

### End state

- Start: `$SCRATCH/llmctl-main/llmctl status` at 01:57:06 (+03:00) read gemma-4-26b-a4b on llama.cpp running and ready at 15,008 MiB of 16,303 MiB, with OpenCode, Hermes, Pi and dsh on it and Kimi "not installed (no kimi on PATH)"; with the `.kimi-code/bin` prefix at 01:57:07 Kimi was on it too; `down` at 01:57:14 left 112 MiB in use ([lane-logs/llmctl-status-before.txt](lane-logs/llmctl-status-before.txt), [lane-logs/llmctl-status-before-kimi-path.txt](lane-logs/llmctl-status-before-kimi-path.txt), [lane-logs/llmctl-down.txt](lane-logs/llmctl-down.txt)) (measured).
- End: `PATH=$HOME/.kimi-code/bin:$PATH $SCRATCH/llmctl-main/llmctl up gemma-4-26b-a4b` at 02:07:19 removed llmctl's stopped `llamacpp` container, as `up` does, printed `ready` and no `wrote` line and exited 0; `status` with the same prefix at 02:07:32 read running, health ready, 15,008 MiB of 16,303 MiB, and OpenCode, Hermes, Pi, dsh and Kimi on the running model ([lane-logs/llmctl-up.txt](lane-logs/llmctl-up.txt), [lane-logs/llmctl-status-after.txt](lane-logs/llmctl-status-after.txt)) (measured).
- A plain `status` without the prefix at 02:12:15 read the same except Kimi "not installed (no kimi on PATH)", because `kimi` is not on this shell's PATH, while with the prefix the same Kimi file reads the running model ([lane-logs/llmctl-status-after-plain-path.txt](lane-logs/llmctl-status-after-plain-path.txt)) (measured).
- Containers: llmctl's `llamacpp` running; no `sglang-measure` container; the `-measured-2026-11-16` containers untouched ([lane-logs/end-disk-volumes.txt](lane-logs/end-disk-volumes.txt)) (measured).
- Volumes: none created; `sglang-research-cache`, `vllm-cache`, `flashinfer-cache` and `ollama-models` kept (measured).
- `/dev/shm`: the start's two root-owned files, `sem.mp-ek533pzo` and `sglang_loads_f28b231c59c2_c4057475.shm`, were removed after its container was gone and are named in its record; at the end `/dev/shm` held the same 93 files as at the lane start and 18,636,800 bytes used against 18,681,856 then, both with the owner's model running ([lane-logs/devshm-lane-start.txt](lane-logs/devshm-lane-start.txt)) (measured).
- Disk: `/home` 196,314,148,864 bytes free and `/` 301,188,530,176 bytes (measured).
- The owner's model was down from 01:57:14 to 02:07:19 (+03:00), by the lane logs' timestamps (measured); the repository was not edited, and no git command that changes anything was run in it.

## Start

| Time (UTC) | Record | Start | Ready line | Poll | Sequence |
|---|---|---|---|---|---|
| 22:57:22 | [sglang-qwen3.6-35b-a3b](../../../2026-11-18/sglang-qwen3.6-35b-a3b/measure-e/README.md) | e1-k88-f088-mem27 | 63.8 s | 64.08 s | ready reading, D8's flush and flushed reading, 95% prompt, title and forced tool requests, 300 s load, second 95% prompt |

Measured on the MSI Raider 18 Max HX, SGLang v0.5.23 ([lane-logs/starts.txt](lane-logs/starts.txt), by `scripts/starts_table_c.py`); "ready line" is SGLang's "The server is fired up and ready to roll!" after the `docker run` call and "poll" the readiness poll (`/ready` 200 and the served name in `/v1/models`); the container ended `running exit=0 oom=false` before its removal.
Besides it, one throwaway `--rm --ipc=host --network none --memory 256m` container of the same image removed the `/dev/shm` files the start left.
The brief's order has no repository probe and no api probe, so this run has no probe decode or first-token figure; M12's median is its decode figure.
No host-side command was heavy: the server ran in its capped container, and the host ran only the HTTP clients, the 1 s trace and readers of `/proc` and the cgroup, which C's rules exempt from `heavy`.

## Deviations from the brief

1. The docker run file is C's `w3-k88-f088-mem27.sh` with one line added, `-e TIKTOKEN_ENCODINGS_BASE=/opt/sglang/lib/python3.12/site-packages/tml_renderers/data`, because the brief asks for the eight runtime variables and C's file sets seven; nothing else differs ([sglang-qwen3.6-35b-a3b/logs/e1-k88-f088-mem27.cmd-diff-vs-c.txt](../../../2026-11-18/sglang-qwen3.6-35b-a3b/measure-e/logs/e1-k88-f088-mem27.cmd-diff-vs-c.txt)).
2. `$SCRATCH/llmctl-main/llmctl status` ran both plain, as the brief names it, and with the `.kimi-code/bin` prefix, at the start and at the end, all read-only, because the plain runs list Kimi as "not installed (no kimi on PATH)" and only the prefixed runs show every client, as D did.

Not deviations, but differences from D's method that the brief sets: `margins_e.py` rounds as the brief says (a sum that is already a multiple of 64 stays), where D's `margins_d.py` and the design's D8 text use "the next multiple of 64 above", and it prints both when they differ; for every measured drop they agree.

## Files

- [sglang-qwen3.6-35b-a3b/](../../../2026-11-18/sglang-qwen3.6-35b-a3b/measure-e/README.md): the record of the start, with `cmds/`, `logs/`, `shapes/` and its README.
- [scripts/](scripts/): `run_e.sh` drives the start (D's `run_d.sh` with E's folder), `chain_e1.sh` queued it, `margins_e.py` is D's `margins_d.py` with the brief's window and rounding and an `--exclude` view, and `check_inputs_e.py` checks the docker run file against the draft and the overlay against C's make record and the snapshot; copied unchanged from D: `seq_d.py`, `poll_c.py`, `trace_c.sh`, `procmem.sh`, `maps_inside.py`, `mcommon.py`, `cutlog.py`, `reqlog.py`, `sustained.py` (the hands-on script, sha256 `47830acf...`), `anonshmem.py`, `nonreclaim.py`, `summarize_c.py`, `starts_table_c.py`, `scrub.py`, `shm_cleanup.sh` and `margins_d.py`; run the Python scripts with `python3 -I`.
- [lane-logs/](lane-logs/): the llmctl outputs, the chain log, the margin tables (`margins-bcde.txt` over all 15 starts, `margins-qwen3.6.txt` over the three Qwen3.6 starts, `margins-bcd-check.txt` with the list of B's, C's and D's starts in `margins-runs-bcd.list`), the starts table, the `/dev/shm` reading at the lane start, and the end reading of disk, volumes, containers and `/dev/shm`.
- Local paths in the records were replaced with `$HOME` and `$SCRATCH`, and the user and host names with `<user>` and `<host>`, by `scripts/scrub.py`.
