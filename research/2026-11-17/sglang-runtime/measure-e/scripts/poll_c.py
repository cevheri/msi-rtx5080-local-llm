"""Readiness poller for measurement C (measurement B's poll_b.py unchanged but for this line), after measurement A's poll_ready.py.
SGLang: polls /ready, /v1/models and /server_info every 0.5 s from the docker run call; ready means /ready answered 200
and /v1/models lists the served name (LD5 revised). vLLM: polls /health and /v1/models; ready means /v1/models lists
the served name (llmctl's vLLM readiness). Right after ready it reads nvidia-smi and, for SGLang, /server_info
(internal_states[0].memory_usage), and the container's cgroup memory, and saves the full /server_info body.
Run with python3 -I: poll_b.py --t0 EPOCH --container NAME --served NAME --kind sglang|vllm --out FILE.json
Exit codes: 0 ready, 2 the container stopped first, 3 timeout."""
import argparse
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mcommon import container_state, gpu, http  # noqa: E402


def cgroup_memory(name):
    import subprocess
    out = subprocess.run(["docker", "inspect", "--format", "{{.Id}}", name], capture_output=True, text=True, timeout=20)
    cid = out.stdout.strip()
    base = f"/sys/fs/cgroup/system.slice/docker-{cid}.scope"
    res = {}
    try:
        with open(f"{base}/memory.stat") as f:
            for line in f:
                key, value = line.split()
                if key in ("anon", "file", "shmem", "file_mapped", "unevictable", "active_file", "inactive_file", "kernel"):
                    res[key + "_gib"] = round(int(value) / 2**30, 3)
        for key in ("memory.current", "memory.peak"):
            with open(f"{base}/{key}") as f:
                res[key.replace("memory.", "") + "_gib"] = round(int(f.read().strip()) / 2**30, 3)
    except OSError as err:
        res["error"] = str(err)[:200]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t0", type=float, required=True)
    ap.add_argument("--container", required=True)
    ap.add_argument("--served", required=True)
    ap.add_argument("--kind", choices=["sglang", "vllm"], default="sglang")
    ap.add_argument("--base", default="http://127.0.0.1:30000")
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=float, default=900)
    a = ap.parse_args()
    t0 = a.t0
    endpoints = ["/ready", "/v1/models", "/server_info"] if a.kind == "sglang" else ["/health", "/v1/models"]
    state = {p: {"first_200_s": None, "changes": [], "polls": 0, "last": None} for p in endpoints}
    listed = {"first_listed_s": None}
    lock = threading.Lock()
    stop = threading.Event()

    def label(head):
        return head["status"] if head["status"] is not None else head.get("error", "error").split(":")[0]

    def worker(path):
        while not stop.is_set():
            start = time.time()
            head, body = http(a.base, path, timeout=10)
            lab = label(head)
            with lock:
                s = state[path]
                s["polls"] += 1
                if lab != s["last"]:
                    s["changes"].append({"at_s": round(start - t0, 2), "status": lab, "call_s": head["seconds"]})
                    s["last"] = lab
                if lab == 200 and s["first_200_s"] is None:
                    s["first_200_s"] = round(start - t0, 2)
                if path == "/v1/models" and lab == 200 and listed["first_listed_s"] is None:
                    ids = [m.get("id") for m in (body or {}).get("data", [])] if isinstance(body, dict) else []
                    if a.served in ids:
                        listed["first_listed_s"] = round(start - t0, 2)
            stop.wait(max(0.0, 0.5 - (time.time() - start)))

    threads = [threading.Thread(target=worker, args=(p,), daemon=True) for p in endpoints]
    for t in threads:
        t.start()
    code = 3
    gate = "/ready" if a.kind == "sglang" else "/v1/models"
    while time.time() - t0 < a.timeout:
        time.sleep(0.25)
        with lock:
            if state[gate]["first_200_s"] is not None and listed["first_listed_s"] is not None:
                code = 0
                break
        st = container_state(a.container)
        if (not st.startswith("running") and st != "missing") or (st == "missing" and time.time() - t0 > 30):
            code = 2
            state["container_at_stop"] = st
            break
    ready_at = round(time.time() - t0, 2)
    stop.set()
    result = {"t0": t0, "kind": a.kind, "exit": {0: "ready", 2: "container stopped", 3: "timeout"}[code],
              "ready_seen_s": ready_at if code == 0 else None, "served_listed_s": listed["first_listed_s"]}
    if code == 0:
        result["gpu_right_after_ready"] = gpu()
        if a.kind == "sglang":
            head, info = http(a.base, "/server_info", timeout=30)
            result["server_info_call"] = head
            if isinstance(info, dict):
                states = info.get("internal_states") or [{}]
                result["memory_usage"] = states[0].get("memory_usage")
                result["max_total_num_tokens"] = info.get("max_total_num_tokens")
                result["status"] = info.get("status")
                result["selected"] = {k: states[0].get(k) for k in (
                    "decode_cuda_graph_bs", "decode_cuda_graph_memory_gb", "effective_max_running_requests_per_dp",
                    "kv_cache_dtype", "mem_fraction_static", "context_length", "attention_backend",
                    "swa_full_tokens_ratio", "max_mamba_cache_size", "enable_paged_experts", "paged_experts_num_resident",
                    "moe_runner_backend", "fp4_gemm_runner_backend", "quantization", "chunked_prefill_size",
                    "disable_prefill_cuda_graph", "tool_call_parser", "reasoning_parser") if k in states[0]}
                with open(a.out.replace(".json", "-server-info.json"), "w") as f:
                    json.dump(info, f, indent=1, sort_keys=True, default=str)
        result["cgroup_right_after_ready"] = cgroup_memory(a.container)
        result["gpu_after_reads"] = gpu()
    for t in threads:
        t.join(timeout=15)
    result["endpoints"] = {p: state[p] for p in endpoints}
    with open(a.out, "w") as f:
        json.dump(result, f, indent=1, default=str)
    print(json.dumps({k: result.get(k) for k in ("exit", "ready_seen_s", "served_listed_s", "memory_usage",
                                                  "gpu_right_after_ready")}, default=str))
    sys.exit(code)


if __name__ == "__main__":
    main()
