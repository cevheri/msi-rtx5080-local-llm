"""Write one SGLang docker run file for measurement D (measurement C's mkcmd_c.py, SGLang part only, with D's overlay
folder and all eight environment variables of the shipped runtime file in its [env] order). The file reads HOME and
SCRATCH from the environment.
The model overlay of measurement A's mechanism (LD8 revised) is mounted read-only at /llmctl/model; the shared container
rules of measurements B and C; python3 -m sglang.launch_server.
Run with python3 -I: mkcmd_d.py --out FILE --comment TEXT --overlay ID --served NAME --context N [--memory 32g] -- ARGS..."""
import argparse
import os
import shlex

RUNTIME_ENV = ["HF_HUB_OFFLINE=1", "HF_MODULES_CACHE=/tmp/hf_modules", "MAX_JOBS=2", "FLASHINFER_NO_DOWNLOAD=1",
               "SYNC_TOKEN_IDS_ACROSS_TP=1", "NCCL_MAX_P2P_NCHANNELS=1",
               "TIKTOKEN_ENCODINGS_BASE=/opt/sglang/lib/python3.12/site-packages/tml_renderers/data",
               "PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1"]


def q(arg):
    return shlex.quote(arg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--comment", required=True)
    ap.add_argument("--overlay", required=True, help="overlay folder name under $SCRATCH/sglang/measure-d/overlay/")
    ap.add_argument("--served", required=True)
    ap.add_argument("--context", type=int, required=True)
    ap.add_argument("--memory", default="32g")
    ap.add_argument("args", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    args = a.args[1:] if a.args and a.args[0] == "--" else a.args
    lines = ["#!/usr/bin/env bash", f"# {a.comment}", "set -euo pipefail", "docker run -d --name sglang-measure \\",
             "  --device nvidia.com/gpu=all \\", "  --ipc=host \\", f"  --memory {a.memory} --memory-swap {a.memory} \\",
             "  -p 127.0.0.1:30000:30000 \\", "  -v sglang-research-cache:/root/.cache/sglang \\",
             f'  -v "$SCRATCH/sglang/measure-d/overlay/{a.overlay}:/llmctl/model:ro" \\']
    for e in RUNTIME_ENV:
        lines.append(f"  -e {q(e)} \\")
    lines += ['  -v "$HOME/.cache/huggingface:/root/.cache/huggingface:ro" \\', "  lmsysorg/sglang:v0.5.23-runtime \\",
              "  python3 -m sglang.launch_server \\", "    --model-path /llmctl/model \\",
              f"    --served-model-name {q(a.served)} \\", f"    --context-length {a.context} \\",
              "    --host 0.0.0.0 --port 30000 \\"]
    # group flag and value on one line where the next item is not a flag
    out, i = [], 0
    while i < len(args):
        if i + 1 < len(args) and not args[i + 1].startswith("--"):
            out.append(f"    {args[i]} {q(args[i + 1])}")
            i += 2
        else:
            out.append(f"    {args[i]}")
            i += 1
    lines += [x + " \\" for x in out[:-1]] + out[-1:]
    if not out:
        lines[-1] = lines[-1].rstrip(" \\")
    with open(a.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(a.out, 0o755)


if __name__ == "__main__":
    main()
