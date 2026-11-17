"""Write one docker run file for measurement C (copied from measurement B's mkcmd.py; C's overlay folder and an
optional read-only bind mount added). The file reads HOME and SCRATCH from the environment.
SGLang: the model overlay of measurement A's mechanism (LD8 revised) mounted read-only at /llmctl/model, the lane's
environment (LD6 and LD15 revised), the shared container rules, and python3 -m sglang.launch_server.
vLLM: mirrors recipes/vllm/_runtime.toml and launcher/runtimes/vllm.py container_spec (no llmctl labels).
Run with python3 -I: mkcmd.py --kind sglang --out FILE --comment TEXT --overlay ID --served NAME --context N
                              [--env K=V ...] [--volume NAME] [--network-none] [--memory 32g] -- ARGS..."""
import argparse
import os
import shlex

COMMON_SGLANG_ENV = ["HF_HUB_OFFLINE=1", "HF_MODULES_CACHE=/tmp/hf_modules", "MAX_JOBS=2",
                     "SYNC_TOKEN_IDS_ACROSS_TP=1", "NCCL_MAX_P2P_NCHANNELS=1", "FLASHINFER_NO_DOWNLOAD=1"]
VLLM_ENV = ["HF_HUB_OFFLINE=1", "VLLM_NO_USAGE_STATS=1", "MAX_JOBS=2", "HF_MODULES_CACHE=/tmp/hf_modules"]


def q(arg):
    return shlex.quote(arg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["sglang", "vllm"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--comment", required=True)
    ap.add_argument("--overlay", help="SGLang: overlay folder name under $SCRATCH/sglang/measure-c/overlay/")
    ap.add_argument("--snapshot", help="vLLM: container path of the snapshot")
    ap.add_argument("--served", required=True)
    ap.add_argument("--context", type=int, required=True)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--volume", action="append", default=[], help="NAME:/path; default the research cache volume(s)")
    ap.add_argument("--network-none", action="store_true")
    ap.add_argument("--memory", default="32g")
    ap.add_argument("--bind-ro", action="append", default=[], help="HOSTPATH:CTRPATH with $SCRATCH or $HOME left for the shell")
    ap.add_argument("args", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    args = a.args[1:] if a.args and a.args[0] == "--" else a.args
    name = "sglang-measure" if a.kind == "sglang" else "vllm-measure"
    port = 30000 if a.kind == "sglang" else 8000
    lines = ["#!/usr/bin/env bash", f"# {a.comment}", "set -euo pipefail", f"docker run -d --name {name} \\",
             "  --device nvidia.com/gpu=all \\", "  --ipc=host \\", f"  --memory {a.memory} --memory-swap {a.memory} \\"]
    if a.network_none:
        lines.append("  --network none \\")
    else:
        lines.append(f"  -p 127.0.0.1:{port}:{port} \\")
    if a.kind == "sglang":
        vols = a.volume or ["sglang-research-cache:/root/.cache/sglang"]
        for v in vols:
            lines.append(f"  -v {v} \\")
        lines.append(f'  -v "$SCRATCH/sglang/measure-c/overlay/{a.overlay}:/llmctl/model:ro" \\')
        for e in COMMON_SGLANG_ENV + a.env:
            lines.append(f"  -e {q(e)} \\")
        lines.append('  -v "$HOME/.cache/huggingface:/root/.cache/huggingface:ro" \\')
        lines.append("  lmsysorg/sglang:v0.5.23-runtime \\")
        lines.append("  python3 -m sglang.launch_server \\")
        lines.append("    --model-path /llmctl/model \\")
        lines.append(f"    --served-model-name {q(a.served)} \\")
        lines.append(f"    --context-length {a.context} \\")
        lines.append("    --host 0.0.0.0 --port 30000 \\")
    else:
        for e in VLLM_ENV + a.env:
            lines.append(f"  -e {q(e)} \\")
        lines.append('  -v "$HOME/.cache/huggingface:/root/.cache/huggingface:ro" \\')
        vols = a.volume or ["vllm-cache:/root/.cache/vllm", "flashinfer-cache:/root/.cache/flashinfer"]
        for v in vols:
            lines.append(f"  -v {v} \\")
        for b in a.bind_ro:
            lines.append(f'  -v "{b}:ro" \\')
        lines.append("  vllm/vllm-openai:v0.33.0 \\")
        lines.append(f"    {a.snapshot} \\")
        lines.append(f"    --served-model-name {q(a.served)} \\")
        lines.append(f"    --max-model-len {a.context} \\")
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
