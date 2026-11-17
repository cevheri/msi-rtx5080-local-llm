# gpt-oss-20b on SGLang v0.5.23: the pinned expert store (measurement C)

Last verified: 2026-11-17.

Result: B's 1.84 factor is PyTorch's pinned host allocator rounding every expert tensor of the store up to a power of two, and the documented setting `PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1` removes it: the store fell from B's 19.19 GiB to 10.37 GiB of shared memory, the non-reclaimable peak from 24.35 to 15.45 GiB, and the server ran the margin protocol and 300 s of load under a 20 GiB cap at B's speed (measured on the MSI Raider 18 Max HX, SGLang v0.5.23).
Decode was 123.3 and 124.6 tok/s by the repository probe against B's 125.5, and 126.4 and 126.2 tok/s median over M12 against 127.5, within the 4.3 tok/s that B's three runs of one K spread (measured), so wave 3 used the setting.
Two starts were used, the most the brief allows.

## Why the store was 1.84 times its data

- `HostExpertStore` allocates each paged tensor of a layer with one `torch.empty(shape, dtype, device="cpu", pin_memory=True)` (`offline/sglang-src/store.py`, lines 19 to 30); with Marlin the post-load repack builds a new pinned store of the repacked tensors, K experts at a time, and the staged checkpoint layout before it is pageable (`offline/sglang-src/formats.py`, lines 64 to 145; `offline/sglang-src/method.py`, lines 147 to 165) (inferred from the source).
- PyTorch 2.14.1 in the image serves `pin_memory=True` from its `CachingHostAllocator`, which rounds a request up to the next power of two unless the request is above `pinned_max_round_threshold` or `pinned_max_cached_size`, both unlimited by default (`offline/torch-src/ATen_core_CachingHostAllocator.h`, lines 340 to 349; `offline/torch-src/c10_core_AllocatorConfig.h`, lines 358 to 363) (inferred from the source).
- The Marlin MXFP4 path pads gpt-oss's intermediate size from 2,880 to 2,944 and its hidden size to 3,072 (`offline/sglang-src/mxfp4.py`, lines 463 to 468), so one layer's `w13_weight` is 32 x 5,888 x 1,536 bytes = 276 MiB, rounded to 512 MiB, and `w2_weight` 138 MiB, rounded to 256 MiB; with the scales and biases rounded likewise that is about 817 MiB per layer for about 440 MiB of data, which matches B's measured 821 MiB (estimate from the source and the shapes).
- The setting is documented: "pinned_max_round_threshold_mb option sets the maximum allocation size (in MB) that will be rounded up to the nearest power-of-2" under `PYTORCH_ALLOC_CONF` ([PyTorch CUDA semantics notes](https://docs.pytorch.org/docs/main/notes/cuda.html), read on 2026-11-17) (vendor-reported), and the image's `libc10.so` holds the key `pinned_max_round_threshold_mb` ([offline/torch-libc10-pinned-keys.txt](../../sglang-runtime/measure-c/offline/torch-libc10-pinned-keys.txt)) (measured offline).
- SGLang's server code sets neither `PYTORCH_ALLOC_CONF` nor `PYTORCH_CUDA_ALLOC_CONF`, and the image's environment sets neither, so the variable changes only the pinned host allocator; in `sglang/srt` only the weight-cache daemon and the Mooncake disaggregation helper read the variables, both to refuse `expandable_segments`, and only the multimodal generator sets one ([offline/sglang-alloc-conf-search.txt](../../sglang-runtime/measure-c/offline/sglang-alloc-conf-search.txt)) (inferred from the source).
- With the threshold at 1 MiB every store tensor keeps its size, while pinned requests of at most 1 MiB (SGLang's sampling and grammar buffers) are still rounded and cached as before (inferred from the source).

## The two starts

Both repeat B's b0-a4 ([../../measure-b/sglang-gpt-oss-20b/cmds/b0-a4.sh](../measure-b/cmds/b0-a4.sh)): `lmsysorg/sglang:v0.5.23-runtime`, C's overlay of the pinned snapshot (temperature 1.0 and top_p 1.0, equal to B's), the lane environment with `TIKTOKEN_ENCODINGS_BASE`, Paged Experts with 24 of 32 experts resident, Marlin, fraction 0.90, FP8 KV, SWA ratio 0.1, 1,024-token chunks, no prefill graph, parsers `gpt-oss`, plus `PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1` ([cmds/pe-round1.sh](cmds/pe-round1.sh), [cmds/pe-round2-mem20.sh](cmds/pe-round2-mem20.sh)).
The enforced GPU power limit read 140 W before each (B's read 145 W).

| Item | B b0-a4 (32 GiB cap) | C pe-round1 (32 GiB cap) | C pe-round2-mem20 (20 GiB cap) |
|---|---|---|---|
| Pinned store (cgroup `shmem`; scheduler `RssShmem`) | 19.19 GiB; 19.25 GiB | 10.37 GiB; 10.43 GiB | 10.37 GiB; 10.43 GiB |
| Non-reclaimable peak (anon + shmem + kernel) | 24.35 GiB | 15.44 GiB | 15.45 GiB |
| Peak of anon + shmem (the D5 `ram_gib` rule) | 24.16 GiB | 15.30 GiB | 15.29 GiB |
| Anonymous peak while loading (the staged checkpoint layout) | 14.11 GiB | 14.02 GiB | 13.87 GiB |
| Cgroup `memory.peak` | 32.00 GiB (page cache filled the cap) | 29.12 GiB | 17.04 GiB |
| Ready line after `docker run` | 54.9 s | 54.6 s | 50.7 s |
| `token_capacity` (SWA); `startup_available` | 168,456 (16,845); 1.447 GiB | 169,835 (16,983); 1.465 GiB | 169,853 (16,985); 1.465 GiB |
| Free right after ready, after a 95% prompt, lowest after ready, drop | 1,329, 989, 989, 340 MiB | 1,346, 1,036, 1,022, 324 MiB | 1,347, 1,006, 1,006, 341 MiB |
| Decode, repository probe (three 1,500-token runs) | 125.5 tok/s (126.0, 123.8, 125.5) | 123.3 tok/s (123.6, 123.3, 121.1) | 124.6 tok/s (126.0, 123.5, 124.6) |
| First token, 8,058 and 30,003 tokens | 1.486, 4.903 s | 1.511, 5.042 s | 1.472, 4.935 s |
| M12: requests in 300 s, errors, decode min / median / max | 74, 0, 118.8 / 127.5 / 134.9 | 73, 0, 115.4 / 126.4 / 131.4 | 73, 0, 118.0 / 126.2 / 132.9 |
| Second 95% prompt after the load | 124,462 tokens, 44.70 s | 124,460 tokens, 44.78 s | 124,460 tokens, 44.87 s |
| GPU peak; power; temperature | 14,854 MiB; 145.3 W; 73 C | 14,820 MiB; 150.1 W; 72 C | 14,836 MiB; 146.1 W; 73 C |
| OOM | none | none | none (`oom=false`) |

C's columns are measured on the MSI Raider 18 Max HX, SGLang v0.5.23 ([shapes/pe-round1-summary.json](shapes/pe-round1-summary.json), [shapes/pe-round2-mem20-summary.json](shapes/pe-round2-mem20-summary.json), `logs/pe-round*-trace.csv`, `logs/pe-round*.procmem.txt`, `shapes/pe-round*-sustained.json`, `$SCRATCH/sglang/measure-c/scripts/anonshmem.py`); B's column is from [its record](../measure-b/README.md).
- The sampling line read `Using default chat sampling params from model generation config: {'temperature': 1.0, 'top_p': 1.0}` in both starts, and the smoke requests of pe-round1 gave B's shapes: `OK`, a `get_weather {"city": "Paris"}` call and a second turn naming 18 C, every effort answered 391, the forced `tool_choice` came back as Harmony text (#41747), and `json_schema` gave `{"capital":"Paris"}` (measured; [shapes/pe-round1-api.json](shapes/pe-round1-api.json)).
- The store measured 442 MiB per layer (10.37 GiB over 24 layers), the 440 MiB that the padded Marlin shapes give, so against the checkpoint index's 9.467 GiB of expert tensors (403.95 MiB per layer, `$SCRATCH/sglang/measure-c/estimates/gpt-oss-20b.expert-bytes.json`) the factor is now 1.096 instead of 2.03, and C estimated the wave 3 stores with 1.096 (measured; the factor is arithmetic).

## The store is shared memory, not a /dev/shm file

- Inside the container the scheduler held the store in 112 shared mappings of `/dev/zero (deleted)` (permissions `rw-s`, flag `sh`), 10,589 MiB resident, which is how the CUDA driver's pinned allocations appear: anonymous shared memory, counted as cgroup `shmem` and as `RssShmem` ([logs/pe-round1.maps.txt](logs/pe-round1.maps.txt), written by `$SCRATCH/sglang/measure-c/scripts/maps_inside.py`) (measured; that `cudaHostAlloc` makes these mappings is inferred).
- It does not sit in `/dev/shm`: with `--ipc=host` the host's `/dev/shm` grew only from 1,519,616 to 35,758,080 bytes at ready and was back at 1,540,096 bytes after the container was gone ([logs/pe-round1.devshm.txt](logs/pe-round1.devshm.txt)) (measured).
- Each start left one root-owned file there, `sglang_loads_<hex>_<hex>.shm` of 16,396 bytes: `sglang_loads_98d6767489dc_9d639f58.shm` after pe-round1 and `sglang_loads_14ff0f3b2eec_617b9980.shm` after pe-round2-mem20, which C removed once each container was gone, with a throwaway `--ipc=host` container because the files are root's; `/dev/shm` then read 1,519,616 bytes used, as before (measured; the removal lines are at the end of each `logs/<run>.devshm.txt`).
- 49 such `sglang_loads_*.shm` files and 42 `sem.mp-*` files from earlier lanes' starts (dated 2026-11-16 20:54 to 2026-11-17 23:03 local time, all root's, 16,396 and 32 bytes) were already in `/dev/shm`, and the same 93 files (with two of the desktop's `aomshm.*`) were there at the end of the lane: C left them as they are (measured; listing in the same file).

## What the setting would change in B's gpt-oss draft (not edited)

- `recipes/sglang/_runtime.toml` `[env]` gains `PYTORCH_ALLOC_CONF = "pinned_max_round_threshold_mb:1"`, which every SGLang recipe then gets; in the non-paged recipes it touches only pinned requests above 1 MiB, of which none were seen (inferred).
- `[needs] ram_gib` 25 becomes 16 (the 15.30 GiB peak of anon plus shmem, rounded up) and `[container] memory_gib` 28 becomes 20 (pe-round2-mem20 held the protocol and M12 under it at a 17.04 GiB cgroup peak) (measured).
- `[needs] gpu_mib` 14,742 becomes 14,724 at most (the 14,836 MiB peak minus 112), the pool 168,456 becomes about 169,850 tokens, and the summary's figures change with them; decode and margins stay within noise (measured).
- The header comment's "19.19 GiB of shared memory" becomes 10.37 GiB.

## Files

- [cmds/](cmds/): `pe-round1.sh` and `pe-round2-mem20.sh`, reading `$HOME` and `$SCRATCH`.
- `logs/<run>.*` and `shapes/<run>-*`: as in B's records, plus `.maps.txt` (the mappings inside the container at ready), `.devshm.txt` (`/dev/shm` before, at ready, after, and the removal) and `.devshm-before.list`.
