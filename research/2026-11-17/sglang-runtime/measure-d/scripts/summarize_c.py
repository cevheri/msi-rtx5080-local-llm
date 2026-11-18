"""Summarize one measurement-B start from its record folder: readiness, memory at ready, the margin table, probe
decode and first-token figures, smoke and thinking results, sustained load, M13, trace peaks and the key log lines.
Run with python3 -I: summarize_b.py <record-dir> <run>"""
import csv
import datetime
import json
import os
import re
import sys

d, run = sys.argv[1], sys.argv[2]
logs, shapes = os.path.join(d, "logs"), os.path.join(d, "shapes")
out = {"run": run}


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


t0 = None
try:
    t0 = float(open(os.path.join(logs, f"{run}.t0")).read())
except OSError:
    pass
ready = load(os.path.join(shapes, f"{run}-ready.json"))
if ready:
    out["ready"] = {k: ready.get(k) for k in ("exit", "ready_seen_s", "served_listed_s", "memory_usage",
                                               "max_total_num_tokens")}
    out["ready"]["free_mib_right_after_ready"] = (ready.get("gpu_right_after_ready") or {}).get("free_mib")
    out["ready"]["cgroup_at_ready"] = ready.get("cgroup_right_after_ready")
    out["ready"]["first_200_s"] = {p: s.get("first_200_s") for p, s in (ready.get("endpoints") or {}).items()}
log = os.path.join(logs, f"{run}.log")
lines = []
if os.path.exists(log):
    for line in open(log, errors="replace"):
        if re.search(r"Load weight end|KV Cache is allocated|SWAKVPool|Mamba Cache|Memory pool end|available_gpu_mem|"
                     r"Using default chat sampling|Auto-detected template|KV pool holds|fired up|Detected SM120|"
                     r"Attention backend|fp4|FP4|Marlin|marlin|Capture|content format|Error|error", line):
            lines.append(line.rstrip()[:260])
        m = re.match(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] The server is fired up", line)
        if m and t0:
            ts = datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=datetime.timezone.utc).timestamp()
            out["ready_line_s"] = round(ts - t0, 1)
out["log_lines"] = lines[:40]
seq = os.path.join(shapes, f"{run}-seq.jsonl")
if os.path.exists(seq):
    rows = [json.loads(x) for x in open(seq)]
    margins, before = [], None
    for r in rows:
        if r["event"] == "gpu_before":
            before = r["free_mib"]
        elif r["event"] == "request":
            margins.append({"step": r["step"], "status": r["status"], "seconds": r.get("seconds"),
                            "prompt_tokens": (r.get("usage") or {}).get("prompt_tokens"),
                            "finish": r.get("finish_reason"), "content": r.get("content"),
                            "tool_calls": r.get("tool_calls"), "title_parses": r.get("title_parses"),
                            "free_before_mib": before})
        elif r["event"] == "gpu_after" and margins:
            margins[-1]["free_after_mib"] = r["free_mib"]
            margins[-1]["cgroup_after"] = r.get("cgroup")
        elif r["event"] in ("nearfull_plan", "tokcheck_counts", "tokcheck_reply", "m13_summary", "end", "death_watch_start",
                            "flush_cache", "skip"):
            out.setdefault("events", []).append({k: v for k, v in r.items() if k not in ("t", "utc")})
        elif r["event"] == "phase_end":
            out.setdefault("phases", []).append({"step": r["step"], "rc": r["rc"], "seconds": r["seconds"],
                                                 "free_mib_after": r.get("free_mib"), "cgroup": r.get("cgroup")})
    out["requests"] = margins
probe = load(os.path.join(shapes, f"{run}-probe.json"))
if probe:
    out["probe"] = {"decode_tok_s_median": probe.get("decode_tok_s_median"),
                    "decode_runs": [{"tok_s": x.get("decode_tok_s"), "completion_tokens": (x.get("usage") or {}).get("completion_tokens")}
                                    for x in probe.get("decode_runs", [])],
                    "long": [{"prompt_tokens": (x["first"].get("usage") or {}).get("prompt_tokens"),
                              "ttft_first_s": x["first"].get("ttft_s"), "ttft_repeat_s": x["repeat_same_prompt"].get("ttft_s")}
                             for x in probe.get("long_prompt", [])],
                    "ok_test": probe.get("ok_test"), "tool_test": probe.get("tool_test")}
api = load(os.path.join(shapes, f"{run}-api.json"))
if api:
    def short(x):
        if not isinstance(x, dict):
            return x
        return {k: x.get(k) for k in ("status", "content", "finish_reason", "reasoning_content_chars", "tool_calls",
                                      "decode_tok_s", "error_body", "mentions_18", "delta_keys", "done_marker") if k in x}
    out["smoke"] = {k: short(api.get(k)) for k in ("plain", "plain_stream_usage", "tool_auto", "tool_auto_stream",
                                                   "tool_second_turn", "tool_second_turn_stream")}
    out["smoke"]["thinking"] = {k: {"status": v.get("status"), "content": v.get("content"),
                                    "reasoning_content_chars": v.get("reasoning_content_chars")}
                                for k, v in (api.get("thinking") or {}).items()}
    out["smoke"]["grammar"] = api.get("grammar")
    out["smoke"]["decode_512_median_tok_s"] = api.get("decode_512_median_tok_s")
sus = load(os.path.join(shapes, f"{run}-sustained.json"))
if sus:
    out["sustained"] = {k: sus.get(k) for k in ("requests", "errors", "decode_tok_s", "prompt_tokens_max",
                                                "first_half_median", "second_half_median")}
m13 = load(os.path.join(shapes, f"{run}-m13.json"))
if m13:
    out["m13"] = {"correct": sum(1 for r in m13 if r.get("correct")), "total": len(m13),
                  "wrong": [{"i": r["i"], "answer": r.get("answer"), "content": r.get("content")} for r in m13 if not r.get("correct")]}
trace = os.path.join(logs, f"{run}-trace.csv")
if os.path.exists(trace):
    rows = list(csv.DictReader(open(trace)))
    def mx(key, scale=1.0):
        vals = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
        return round(max(vals) / scale, 3) if vals else None
    out["trace"] = {"gpu_used_peak_mib": mx("gpu_mem_used_mib"), "gpu_free_min_mib": None,
                    "power_max_w": mx("gpu_power_w"), "temp_max_c": mx("gpu_temp_c"),
                    "cg_anon_peak_gib": mx("cg_anon_b", 2**30), "cg_shmem_peak_gib": mx("cg_shmem_b", 2**30),
                    "cg_file_peak_gib": mx("cg_file_b", 2**30), "cg_peak_gib": mx("cg_peak_b", 2**30),
                    "cg_unevictable_peak_gib": mx("cg_unevictable_b", 2**30), "samples": len(rows)}
    frees = [float(r["gpu_mem_free_mib"]) for r in rows if r.get("gpu_mem_free_mib")]
    out["trace"]["gpu_free_min_mib"] = min(frees) if frees else None
for name in ("state.txt", "meta", "procmem.txt"):
    p = os.path.join(logs, f"{run}.{name}")
    if os.path.exists(p):
        out[name] = open(p).read().strip().splitlines()[:30]
print(json.dumps(out, indent=1, default=str))
