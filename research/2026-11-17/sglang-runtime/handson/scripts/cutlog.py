"""Cut an SGLang container log to what matters: startup, backend and memory lines, warnings, errors and a few batch lines.
Run with python3 -I: cutlog.py <raw.log> <cut.log>"""
import re
import sys

KEY = re.compile(r"server_args=|Attention backend|attention backend|Use triton|Detected SM120|Hybrid SWA|Auto-detected|content format|"
                 r"Init parallel|Memory baseline|Load weight|Mamba Cache|KV Cache is allocated|SWAKVPool|Reserving|Memory pool end|"
                 r"Linear attention|hybrid linear|GDN kernel|autotune|Disable prefill|Capture target|KV pool holds|max_total_num_tokens|"
                 r"Tree cache|Radix Cache|startup timings|Uvicorn running|fired up|device-loaded after serving|not fully optimized|"
                 r"FP8 KV cache|paged|Paged|Multimodal|UserWarning|Scheduler hit an exception|SIGQUIT|kill_process_tree|"
                 r"Failed to CUDA calloc|ncclUnhandledCudaError|DistBackendError|Error:|ValueError|AttributeError|500 Internal|"
                 r"missing a thinking field|is too large")
BATCH = re.compile(r"(Prefill|Decode) batch")
FRAME = re.compile(r'^\s+File "(/sgl-workspace/sglang/python/sglang/srt/[^"]+)", line (\d+), in (\S+)')
ALLOC = re.compile(r"CUDACachingAllocator.*memory allocation failed with OOM")
PROGRESS = re.compile(r"\d+%\|")

raw = open(sys.argv[1], errors="replace").read().splitlines()
out, batches, allocs, in_tb = [], [], 0, False
for line in raw:
    if ALLOC.search(line):
        allocs += 1
        if allocs == 1:
            out.append(line[:300])
        continue
    if PROGRESS.search(line):
        continue
    if BATCH.search(line):
        batches.append(line[:330])
        continue
    if line.startswith("Traceback") or "Traceback (most recent call last)" in line:
        in_tb = True
        out.append(line[:300])
        continue
    if in_tb:
        m = FRAME.match(line)
        if m:
            out.append(f"  File {m.group(1).split('/srt/')[-1]}:{m.group(2)} in {m.group(3)}")
            continue
        if line.startswith("  ") or line.strip().startswith("^") or not line.strip():
            continue
        in_tb = False
    if KEY.search(line):
        out.append(line[:2000] if "server_args=" in line else line[:400])
if allocs > 1:
    out.append(f"[cut] {allocs} CUDACachingAllocator OOM warnings in total during weight loading")
if batches:
    out.append(f"[cut] {len(batches)} batch lines; first 3 and last 6:")
    out.extend(batches[:3] + (["..."] if len(batches) > 9 else []) + batches[-6:])
open(sys.argv[2], "w").write("\n".join(out) + "\n")
print(sys.argv[2].rsplit("/", 2)[-2] + "/" + sys.argv[2].rsplit("/", 1)[-1], len(raw), "->", len(out), "lines")
