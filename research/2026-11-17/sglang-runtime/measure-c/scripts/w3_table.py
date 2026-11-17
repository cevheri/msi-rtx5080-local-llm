"""Tabulate the starts of one wave 3 record: per start the ready line and poll, the pool, weights, startup_available,
free GPU memory right after ready, decode and first tokens by the repository probe, the pinned store (scheduler RssShmem
and cgroup shmem at ready), the anon+shmem peak, the GPU peak, and the M12 figures when present.
Run with python3 -I: w3_table.py <record-dir> [run ...]"""
import csv
import datetime
import json
import os
import re
import sys

d = sys.argv[1]
runs = sys.argv[2:] or sorted(f[:-5] for f in os.listdir(os.path.join(d, "logs")) if f.endswith(".meta"))
G = 2**30
for run in runs:
    L, S = os.path.join(d, "logs"), os.path.join(d, "shapes")
    out = {"run": run}
    try:
        t0 = float(open(os.path.join(L, f"{run}.t0")).read())
    except OSError:
        t0 = None
    for name in (f"{run}.log", f"{run}.full.log"):
        p = os.path.join(L, name)
        if os.path.exists(p) and t0:
            for line in open(p, errors="replace"):
                m = re.match(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] The server is fired up", line)
                if m:
                    ts = datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=datetime.timezone.utc).timestamp()
                    out["ready_line_s"] = round(ts - t0, 1)
                    break
            if "ready_line_s" in out:
                break
    rj = os.path.join(S, f"{run}-ready.json")
    if os.path.exists(rj):
        r = json.load(open(rj))
        mu = r.get("memory_usage") or {}
        out.update(exit=r.get("exit"), poll_ready_s=r.get("ready_seen_s"), weight=mu.get("weight"), kv=mu.get("kvcache"),
                   startup_available=mu.get("startup_available"), pool=mu.get("token_capacity"),
                   pool_swa=mu.get("token_capacity_swa"), free_after_ready=(r.get("gpu_right_after_ready") or {}).get("free_mib"))
    ep = os.path.join(S, f"{run}-exec-poll.txt")
    if os.path.exists(ep):
        for line in open(ep):
            m = re.match(r"at ([\d.]+) s served name listed", line)
            if m:
                out["exec_poll_ready_s"] = float(m.group(1))
            m = re.match(r"free_mib_after_ready=(\d+)", line)
            if m:
                out["free_after_ready"] = int(m.group(1))
            if line.startswith("chat "):
                out["chat"] = line.strip()[:160]
    pj = os.path.join(S, f"{run}-probe.json")
    if os.path.exists(pj):
        p = json.load(open(pj))
        out["decode"] = p.get("decode_tok_s_median")
        out["decode_runs"] = [x.get("decode_tok_s") for x in p.get("decode_runs", [])]
        out["ttft"] = [((l["first"].get("usage") or {}).get("prompt_tokens"), l["first"].get("ttft_s")) for l in p.get("long_prompt", [])]
    pm = os.path.join(L, f"{run}.procmem.txt")
    if os.path.exists(pm):
        txt = open(pm).read().split("== at the end")[0]
        sh = [int(x) for x in re.findall(r"RssShmem_kB=(\d+)", txt)]
        out["rss_shmem_gib"] = round(max(sh) / 2**20, 2) if sh else None
        m = re.search(r"cgroup shmem=([\d.]+) GiB", txt)
        out["cg_shmem_at_ready_gib"] = float(m.group(1)) if m else None
    tr = os.path.join(L, f"{run}-trace.csv")
    if os.path.exists(tr):
        rows = [x for x in csv.DictReader(open(tr))]
        gpu = [float(x["gpu_mem_used_mib"]) for x in rows if x.get("gpu_mem_used_mib")]
        out["gpu_peak_mib"] = int(max(gpu)) if gpu else None
        cg = [x for x in rows if x.get("cg_anon_b")]
        if cg:
            out["anon_shmem_peak_gib"] = round(max(float(x["cg_anon_b"]) + float(x["cg_shmem_b"]) for x in cg) / G, 2)
            out["nonreclaim_peak_gib"] = round(max(float(x["cg_anon_b"]) + float(x["cg_shmem_b"]) + float(x["cg_kernel_b"]) for x in cg) / G, 2)
            out["anon_peak_gib"] = round(max(float(x["cg_anon_b"]) for x in cg) / G, 2)
            out["memory_peak_gib"] = round(max(float(x["cg_peak_b"]) for x in cg if x.get("cg_peak_b")) / G, 2)
        pw = [float(x["gpu_power_w"]) for x in rows if x.get("gpu_power_w")]
        out["power_max_w"] = max(pw) if pw else None
    sj = os.path.join(S, f"{run}-sustained.json")
    if os.path.exists(sj):
        s = json.load(open(sj))
        out["m12"] = {k: s.get(k) for k in ("requests", "decode_tok_s", "first_half_median", "second_half_median", "prompt_tokens_max")}
        out["m12"]["errors"] = len(s.get("errors") or [])
    st = os.path.join(L, f"{run}.state.txt")
    if os.path.exists(st):
        out["state"] = open(st).read().split(" started")[0].strip()
    print(json.dumps(out))
