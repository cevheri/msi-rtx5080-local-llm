"""Cut an SGLang container log to what matters: startup, backend and memory lines, warnings, errors and a few batch lines.
measure-a changes: sampling-default, download and fetch, NCCL, health-check and SIGQUIT lines are kept; HTTP access
lines are counted per method, path and status (500s kept); request-log lines (Receive, Finish) are counted, not kept,
because they carry prompt text (reqlog.py extracts their sampling fields); deprecation warnings are counted.
Run with python3 -I: cutlog.py <raw.log> <cut.log>"""
import re
import sys

KEY = re.compile(r"server_args=|Attention backend|attention backend|Use triton|Detected SM120|Hybrid SWA|Auto-detected|content format|"
                 r"Init parallel|Memory baseline|Load weight|Mamba Cache|KV Cache is allocated|SWAKVPool|Reserving|Memory pool end|"
                 r"Linear attention|hybrid linear|GDN kernel|autotune|Disable prefill|Capture target|KV pool holds|max_total_num_tokens|"
                 r"Tree cache|Radix Cache|startup timings|Uvicorn running|fired up|device-loaded after serving|not fully optimized|"
                 r"FP8 KV cache|paged|Paged|Multimodal|UserWarning|Scheduler hit an exception|SIGQUIT|kill_process_tree|"
                 r"Failed to CUDA calloc|ncclUnhandledCudaError|DistBackendError|Error:|ValueError|AttributeError|500 Internal|"
                 r"missing a thinking field|is too large|"
                 r"Using default chat sampling|generation config|GenerationConfig|do_sample|"
                 r"[Ff]etching|[Dd]ownload|urm\.nvidia|huggingface\.co|NO_DOWNLOAD|"
                 r"NCCL|nccl|Health check|SIGQUIT|exit code|Killed|RuntimeError|OSError|Traceback")
ACCESS = re.compile(r'"(GET|POST) (\S+) HTTP/1\.1" (\d{3})')
REQLOG = re.compile(r"Receive OpenAI: obj=|Receive: obj=|Finish: obj=")
DEPRECATED = re.compile(r"is deprecated and will be removed")
BATCH = re.compile(r"(Prefill|Decode) batch")
FRAME = re.compile(r'^\s+File "(/sgl-workspace/sglang/python/sglang/srt/[^"]+)", line (\d+), in (\S+)')
ALLOC = re.compile(r"CUDACachingAllocator.*memory allocation failed with OOM")
PROGRESS = re.compile(r"\d+%\|")

raw = open(sys.argv[1], errors="replace").read().splitlines()
out, batches, allocs, in_tb = [], [], 0, False
access, reqlog, deprecated = {}, 0, {}
for line in raw:
    m_access = ACCESS.search(line)
    if m_access:
        key = " ".join(m_access.groups())
        access[key] = access.get(key, 0) + 1
        if m_access.group(3).startswith("5"):
            out.append(line[:300])
        continue
    if REQLOG.search(line):
        reqlog += 1
        continue
    if DEPRECATED.search(line):
        key = line.split("] ", 1)[-1][:120]
        deprecated[key] = deprecated.get(key, 0) + 1
        continue
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
if access:
    out.append("[cut] HTTP access lines by method, path and status: " + ", ".join(f"{k} x{v}" for k, v in sorted(access.items())))
if reqlog:
    out.append(f"[cut] {reqlog} request-log lines (Receive OpenAI, Receive, Finish) left out; see the reqlog extract")
for key, count in deprecated.items():
    out.append(f"[cut] {count} x {key}")
if allocs > 1:
    out.append(f"[cut] {allocs} CUDACachingAllocator OOM warnings in total during weight loading")
if batches:
    out.append(f"[cut] {len(batches)} batch lines; first 3 and last 6:")
    out.extend(batches[:3] + (["..."] if len(batches) > 9 else []) + batches[-6:])
open(sys.argv[2], "w").write("\n".join(out) + "\n")
print(sys.argv[2].rsplit("/", 2)[-2] + "/" + sys.argv[2].rsplit("/", 1)[-1], len(raw), "->", len(out), "lines")
