"""M7 readiness poller: polls /health, /ready, /v1/models, /get_server_info (and /server_info, which v0.5.23 names as the
replacement of the deprecated /get_server_info) every 0.5 s from the docker run call until all of them have answered 200.
Records each status change with its time after t0, then reads /get_server_info once and times single calls.
Run with python3 -I: poll_ready.py --t0 <epoch> --container NAME --out FILE.json [--base URL] [--timeout S]
Exit codes: 0 all answered 200, 2 the container stopped first, 3 timeout."""
import argparse
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mcommon import container_state, http  # noqa: E402

ENDPOINTS = ["/health", "/ready", "/v1/models", "/get_server_info", "/server_info"]
REQUIRED = ["/health", "/ready", "/v1/models", "/get_server_info"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t0", type=float, required=True)
    ap.add_argument("--container", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="http://127.0.0.1:30000")
    ap.add_argument("--timeout", type=float, default=900)
    a = ap.parse_args()
    t0 = a.t0
    state = {p: {"first_200_s": None, "changes": [], "polls": 0, "last": None} for p in ENDPOINTS}
    lock = threading.Lock()
    stop = threading.Event()

    def label(head):
        return head["status"] if head["status"] is not None else head.get("error", "error").split(":")[0]

    def worker(path):
        while not stop.is_set():
            start = time.time()
            head, _ = http(a.base, path, timeout=10)
            lab = label(head)
            with lock:
                s = state[path]
                s["polls"] += 1
                if lab != s["last"]:
                    s["changes"].append({"at_s": round(start - t0, 2), "status": lab, "call_s": head["seconds"],
                                         "error": head.get("error")})
                    s["last"] = lab
                if lab == 200 and s["first_200_s"] is None:
                    s["first_200_s"] = round(start - t0, 2)
            stop.wait(max(0.0, 0.5 - (time.time() - start)))

    threads = [threading.Thread(target=worker, args=(p,), daemon=True) for p in ENDPOINTS]
    for t in threads:
        t.start()
    code = 3
    while time.time() - t0 < a.timeout:
        time.sleep(0.5)
        with lock:
            if all(state[p]["first_200_s"] is not None for p in REQUIRED):
                code = 0
                break
        st = container_state(a.container)
        if not st.startswith("running") and st != "missing" or (st == "missing" and time.time() - t0 > 30):
            code = 2
            state["container_at_stop"] = st
            break
    stop.set()
    for t in threads:
        t.join(timeout=15)
    result = {"t0": t0, "exit": {0: "all four answered 200", 2: "container stopped", 3: "timeout"}[code],
              "elapsed_s": round(time.time() - t0, 2), "endpoints": {p: state[p] for p in ENDPOINTS}}
    if code == 0:
        head, info = http(a.base, "/get_server_info", timeout=30)
        result["get_server_info_call"] = head
        if isinstance(info, dict):
            result["get_server_info_top_keys"] = sorted(info.keys())
            states = info.get("internal_states") or []
            result["internal_states_len"] = len(states)
            if states:
                st0 = states[0]
                result["internal_states0_keys"] = sorted(st0.keys())
                result["internal_states0_memory_usage"] = st0.get("memory_usage")
                result["internal_states0_selected"] = {k: st0.get(k) for k in (
                    "max_total_num_tokens", "max_running_requests", "context_len", "available_gpu_mem",
                    "last_gen_throughput", "load") if k in st0}
            result["top_level_selected"] = {k: info.get(k) for k in (
                "max_total_num_tokens", "max_running_requests", "context_length", "mem_fraction_static", "version",
                "startup_time", "served_model_name", "tp_size") if k in info}
            with open(a.out.replace(".json", "-server-info.json"), "w") as f:
                json.dump(info, f, indent=1, sort_keys=True, default=str)
        timings = {}
        for path, n in (("/health", 3), ("/ready", 3), ("/v1/models", 3), ("/server_info", 1), ("/get_server_info", 1)):
            timings[path] = [http(a.base, path, timeout=30)[0] for _ in range(n)]
        result["single_call_timings"] = {p: [{"status": h["status"], "seconds": h["seconds"]} for h in hs]
                                         for p, hs in timings.items()}
    with open(a.out, "w") as f:
        json.dump(result, f, indent=1, default=str)
    print(json.dumps({"exit": result["exit"], "elapsed_s": result["elapsed_s"],
                      "first_200_s": {p: state[p]["first_200_s"] for p in ENDPOINTS}}))
    sys.exit(code)


if __name__ == "__main__":
    main()
