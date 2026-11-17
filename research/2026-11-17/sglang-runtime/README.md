# SGLang v0.5.23 on this laptop: the cross-model records, 2026-11-17

Last verified: 2026-11-17.

This folder holds the records behind the SGLang runtime of llmctl that apply to more than one model: the hands-on lane's first run of SGLang on this laptop, measurement A (M6, M7, M8 and M15), and the design's offline probes P1 and P2.
The per-model records of the same runs are in [sglang-qwen3.5-9b/](../sglang-qwen3.5-9b/README.md), [sglang-k2-horizon-7b/](../sglang-k2-horizon-7b/README.md), [sglang-gpt-oss-20b/](../sglang-gpt-oss-20b/README.md) and [sglang-gemma-4-e4b/](../sglang-gemma-4-e4b/README.md).
Every model server and P1 ran in `lmsysorg/sglang:v0.5.23-runtime` (index digest `sha256:3ce6da3e5ea0fee2dac6a23b528cb294fedd31ca74a55a4511f709c5e01e8652`), started by hand with `docker run`, one container at a time, before llmctl had an SGLang adapter.
P2 ran on the host with Python and `jsonschema`, without the image.
Do not edit these records by hand.

## What each lane measured

| Lane | Question | Result | Label and record |
|---|---|---|---|
| Hands-on | Does SGLang v0.5.23 run the cached checkpoints on this GPU? | 12 starts: Qwen3.5-9B NVFP4 at 128k (140,976-token pool, 92.8 tok/s), K2 Horizon 7B AWQ-INT4 at 64k (82,394 FP8 tokens, 89.9 tok/s), gpt-oss-20b at 128k only with Paged Experts K 24 (168,443 FP8 tokens, 125.5 tok/s, 13.94 GiB anonymous host memory); Gemma 4 E4B failed in its audio tower; three starts crashed on the first grammar request | (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17); [handson/README.md](handson/README.md) |
| M8 (measurement A) | How can a recipe pin sampling? | No flag sets the defaults of chat requests; `--preferred-sampling-params` never reached a chat request; a read-only overlay at `/llmctl/model` (links into the snapshot plus a real `generation_config.json`) was logged as `Using default chat sampling params from model generation config: {'temperature': 0.6, 'top_k': 20, 'top_p': 0.95}` and applied, also for the remote-code K2 Horizon 7B | (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17); [measure-a/README.md](measure-a/README.md), section M8 |
| M6 (measurement A) | Does a warm-up protect later grammar requests, and what must be free? | The first grammar request took 602 to 608 MiB; a warm-up protects but adds no headroom; `SYNC_TOKEN_IDS_ACROSS_TP=1` moves the NCCL allocation before ready and `NCCL_MAX_P2P_NCHANNELS=1` cuts it to about 100 MiB, leaving 549 MiB after a 120k-token prompt and both grammar requests on Qwen3.5-9B, at unchanged decode (92.8 against 92.9 tok/s) | (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17); [measure-a/README.md](measure-a/README.md), section M6 |
| M7 (measurement A) | Which endpoint says ready? | `/ready` and `/health` turned 200 at the same 0.5 s poll in 15 starts, `/ready` in about 1 ms and `/health` in 1.0 s; `/v1/models` and `/server_info` answered 1 to 13 s before ready; `startup_available` overstated Qwen3.5-9B's headroom by 0.34 GiB | (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17); [measure-a/README.md](measure-a/README.md), section M7 |
| M15 (measurement A) | Does a start need the network? | Qwen3.5-9B reached ready from an empty cache volume with `FLASHINFER_NO_DOWNLOAD=1` and with `--network none`, without a download line; gpt-oss-20b spends 52 to 63 s of each start fetching its Harmony vocabulary unless `TIKTOKEN_ENCODINGS_BASE` points at the image's copy, which loaded offline in 0.08 s | (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17); [measure-a/README.md](measure-a/README.md), sections M15 and "Other findings" |
| Cold and warm free memory (measurement A) | Is the free memory right after ready repeatable? | Qwen3.5-9B a3: 1,013 MiB free after a warm start and 1,039 after `POST /flush_cache`; 709 MiB after a cold start and 1,039 after the flush | (measured on the MSI Raider 18 Max HX, SGLang v0.5.23, 2026-11-17); [measure-a/README.md](measure-a/README.md), the M6 tables |
| P1 (design probe) | Does the image's Hugging Face CLI run as the host user for `pull`? | `docker run --rm --network none --memory 2g --memory-swap 2g --user <uid>:<gid> -e HOME=/tmp -e HF_HOME=/hf --entrypoint python3 lmsysorg/sglang:v0.5.23-runtime -m huggingface_hub.cli.hf download --help` exited 0 and lists `--revision`, `--include` and `--exclude`; `hf version` reports 1.33.0; `python3` is `/opt/sglang/bin/python3`; no `hf` is on `PATH`; `/opt/sglang/lib/python3.12/site-packages/tml_renderers/data` holds `o200k_base.tiktoken` | (measured on the MSI Raider 18 Max HX, offline probe of the image, 2026-11-17); [spec-probes/offline-image-probe.txt](spec-probes/offline-image-probe.txt) |
| P2 (design probe) | Is the `sglang-docker` OpenCode provider of the design valid config? | docs/opencode.md's validated config with that provider added: 0 errors for 21 models against https://opencode.ai/config.json fetched 2026-11-17 with Python `jsonschema` 4.19.2 (Draft 2020-12); a V1 `npm` key and a string `tools` value in the new provider are rejected | (measured on the MSI Raider 18 Max HX, Python 3 jsonschema, 2026-11-17); [spec-probes/validate-result.json](spec-probes/validate-result.json), [spec-probes/validate-result-provider.json](spec-probes/validate-result-provider.json), [spec-probes/validate_opencode.py](spec-probes/validate_opencode.py) |

The hands-on runs took place between 18:31 and 19:08 (+03:00) and measurement A between 20:15 and 20:48 (+03:00), both on 2026-11-17, with the owner's `gemma-4-26b-a4b` on llama.cpp stopped before and started again after; the `llmctl` outputs of both lanes record that.
The GPU power limit read 140 W after the hands-on runs and 145 W before measurement A.

## Layout

| Path | Holds | Source |
|---|---|---|
| [handson/README.md](handson/README.md) | The hands-on lane's report: twelve starts, the API behaviour the clients meet, caches and network | `$SCRATCH/sglang/handson/README.md`, links rewritten |
| `handson/logs/` | [image-inspect.txt](handson/logs/image-inspect.txt), [ready-times.txt](handson/logs/ready-times.txt), [power-limit.txt](handson/logs/power-limit.txt), the `llmctl` outputs and the image pull's [log](handson/logs/pull.log), start and end times | `$SCRATCH/sglang/handson/logs/` |
| [handson/scripts/](handson/scripts/) | The hands-on helpers (`api_probe.py`, `cutlog.py`, `proc_net.py`, `sustained.py`, `trace.sh` and others), cited by the README | `$SCRATCH/sglang/handson/scripts/` |
| [handson/links.diff](handson/links.diff) | The link rewrite of the README against the scratch copy | `diff -u` |
| [measure-a/README.md](measure-a/README.md) | Measurement A's report: M8, M6, M7, M15 and other findings | `$SCRATCH/sglang/measure-a/README.md`, links rewritten |
| `measure-a/logs/` | [summary.txt](measure-a/logs/summary.txt), the `llmctl` outputs, [nccl-probe/](measure-a/logs/nccl-probe/) and [harmony-offline/](measure-a/logs/harmony-offline/) | `$SCRATCH/sglang/measure-a/logs/` |
| [measure-a/scripts/](measure-a/scripts/) | Measurement A's helpers (`run.sh`, `poll_ready.py`, `m6.py`, `m8.py`, `make_overlay.py`, `nccl_probe.py`, `harmony_check.py` and others) | `$SCRATCH/sglang/measure-a/scripts/` |
| [measure-a/cmds/nccl-probe.sh](measure-a/cmds/nccl-probe.sh) | The one-rank NCCL probe's `docker run` | `$SCRATCH/sglang/measure-a/cmds/` |
| [measure-a/links.diff](measure-a/links.diff) | The link rewrite of the README against the scratch copy | `diff -u` |
| `spec-probes/` | P1's [offline-image-probe.txt](spec-probes/offline-image-probe.txt), in which the archive removed the `hf` CLI's four ANSI colour escapes and wrote its check-mark character as the word OK, under the repository's no-emoji rule, P2's two results and its validator | `$SCRATCH/sglang-spec/` |

The per-model files of both lanes (`cmds/<id>-*.sh`, `logs/<id>/`, `shapes/<id>/` and A's `overlay/<id>*.make.json`) are in the `handson/` and `measure-a/` folders of each `sglang-<id>/` folder, under `cmds/`, `logs/`, `shapes/` and `overlay/` without the model folder level.

## Links and the rewrite

The two lane READMEs keep their text as the lanes wrote it, with their relative links rewritten to this layout by a script outside the repository, so that every link still resolves.
A link to `cmds/<id>-...`, `logs/<id>/...` or `shapes/<id>/...` of the four models now points to `../../sglang-<id>/<lane>/cmds/...`, `.../logs/...` or `.../shapes/...`.
Links within the kept files stay as they were.
A link to a folder that is not archived here as a whole (`cmds/`, `shapes/`, `logs/`, `overlay/`, `src/`, `data-gh/` and the overlay folders) became its link text in backticks.
Each lane's `links.diff` is `diff -u` of the scratch copy against the archived copy at that point and shows only link changes.
After that, a script put the sentences of the hands-on README on lines of their own, which its diff does not show; measurement A's README already had one sentence per line and was not changed.

## Left out

- `$SCRATCH/sglang/measure-a/src/`: the v0.5.23 source files read offline from the image, which the README cites by path and line; they are vendor code, and the same files can be read at tag v0.5.23 on GitHub.
- Every overlay folder of measurement A: each holds links into `/root/.cache/huggingface` that resolve only inside a container, so they dangle here; the `*.make.json` files, which record each overlay's link count and its generation config before and after, are kept.
- `$SCRATCH/sglang/handson/data-gh/`: GitHub search API listings, read as data only.
- `$SCRATCH/sglang/docs/` and `$SCRATCH/sglang/data/`: the docs lane's raw copies of SGLang's documentation and source.
- `$SCRATCH/sglang-spec/schema/`: the saved copy of https://opencode.ai/config.json that P2 validated against, which is OpenCode's file; it was fetched at 2026-11-17T18:26:39Z and its sha256 is `828719e5c07c6806c002e764a98a4ee9d3904b0d2b084a8d984b714b25d44c62`.
- `$SCRATCH/sglang-spec/bin/gen_sglang.py`: probe P3's generator, which imports the repository's launcher at 5ad8883 to render the files of `gen/` below from recipe values written into the script; like its output, it was rendered rather than measured on the GPU.
- `$SCRATCH/sglang-spec/gen/`: probe P3's docker arguments and client entries for three recipes, rendered from the repository's code at 5ad8883 rather than measured on the GPU.
- Every `__pycache__/` folder.

## Rerunning

The scripts and command files read `$HOME` and `$SCRATCH` from the environment, where `$SCRATCH` was the session's scratch folder and `$HOME` the owner's home folder.
Measurement A's `run.sh` and `run_netnone.sh` also need `M`, the lane folder they write `logs/<model>/` and `shapes/<model>/` into, and call the hands-on `trace.sh` from `$SCRATCH/sglang/handson/scripts/`, so a rerun rebuilds the lanes' scratch layout first, with the per-model command files copied back from the `sglang-<id>/` folders.
Every start needs the Hugging Face cache with the pinned snapshots, the image above and a free GPU; stop any other model server first, and run the Python helpers with `python3 -I`.
P2's validator runs as `python3 -I spec-probes/validate_opencode.py <saved config.json> docs/opencode.md <result.json>` with Python `jsonschema` installed, against a fresh copy of the schema whose sha256 may differ from the one above.

## Privacy

A script replaced local paths in these records with `$HOME` and `$SCRATCH` before they were copied, and a search for the user name, the host name and the session's temporary path finds nothing.
The access logs and socket tables name only 172.17.0.1 and 172.17.0.2, Docker's default bridge gateway and the container's own address, which identify no machine or person.
The P1 probe writes the host user's IDs as `<uid>:<gid>`.
