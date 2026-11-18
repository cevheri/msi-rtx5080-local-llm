"""Summarize one M16 llama.cpp start: probe (decode, first token and server timings), llmctl test, the grammar and smoke
requests, thinking per type, the code decode, GPU and power peaks from the 1 s trace, the docker stats peak and the
cgroup split. Run with python3 -I: m16_summary.py <record-dir> <run>"""
import csv
import json
import os
import sys

d, run = sys.argv[1], sys.argv[2]
logs, shapes = os.path.join(d, "logs"), os.path.join(d, "shapes")
out = {"run": run}
p = json.load(open(os.path.join(shapes, f"{run}-probe-summary.json")))
out["ok"] = {k: p["ok"].get(k) for k in ("content", "reasoning_chars", "finish")}
out["tool"] = {k: p["tool"].get(k) for k in ("finish", "calls")}
out["decode_median"] = p["decode_tok_s_median"]
out["decode_runs"] = [(x["decode_tok_s"], (x.get("usage") or {}).get("completion_tokens"),
                       round((x.get("server_timings") or {}).get("predicted_per_second") or 0, 1)) for x in p["decode_runs"]]
out["long"] = []
for l in p["long"]:
    f = l.get("server_first") or {}
    out["long"].append({"target": l["target"], "prompt_tokens": l["prompt_tokens"], "new": f.get("prompt_n"),
                        "cached": f.get("cache_n"), "pp_tok_s": round(f.get("prompt_per_second") or 0),
                        "ttft_s": l["ttft_first_s"], "tg_tok_s": round(f.get("predicted_per_second") or 0, 1),
                        "repeat_ttft_s": l["ttft_repeat_s"]})
out["test"] = open(os.path.join(logs, f"{run}.test.txt")).read().strip().splitlines()
reqs = []
for line in open(os.path.join(shapes, f"{run}-seq.jsonl")):
    r = json.loads(line)
    if r["event"] == "request":
        reqs.append({k: r.get(k) for k in ("step", "status", "finish_reason", "content", "tool_calls", "title_parses",
                                            "reasoning_content_chars")})
out["grammar_requests"] = reqs
a = json.load(open(os.path.join(shapes, f"{run}-api.json")))
out["tool_second_turn"] = {k: a["tool_second_turn"].get(k) for k in ("status", "content", "mentions_18")} \
    if isinstance(a.get("tool_second_turn"), dict) else a.get("tool_second_turn")
out["tool_auto_stream"] = {k: (a.get("tool_auto_stream") or {}).get(k) for k in ("status", "tool_calls", "finish_reasons")}
out["thinking"] = {k: (v.get("status"), v.get("content"), v.get("reasoning_content_chars"))
                   for k, v in (a.get("thinking") or {}).items()}
out["json_schema"] = {k: (v.get("status"), v.get("finish_reason"), v.get("content")) for k, v in (a.get("grammar") or {}).items()}
try:
    b = json.load(open(os.path.join(shapes, f"{run}-code-decode.json")))
    t = b.get("timings") or {}
    m = b["choices"][0]["message"]
    out["code_decode"] = {"tokens": t.get("predicted_n"), "tok_s": round(t.get("predicted_per_second") or 0, 1),
                          "finish": b["choices"][0]["finish_reason"], "reasoning_chars": len(m.get("reasoning_content") or ""),
                          "content_chars": len(m.get("content") or "")}
except (OSError, ValueError, KeyError) as err:
    out["code_decode"] = f"unreadable: {err}"
rows = list(csv.DictReader(open(os.path.join(logs, f"{run}-trace.csv"))))
num = lambda k: [float(r[k]) for r in rows if r.get(k) not in (None, "")]
G = 2**30
out["gpu_peak_mib"] = int(max(num("gpu_mem_used_mib")))
out["power_max_w"] = max(num("gpu_power_w"))
out["temp_max_c"] = max(num("gpu_temp_c"))
out["cgroup_peak_gib"] = round(max(num("cg_peak_b")) / G, 2)
out["anon_peak_gib"] = round(max(num("cg_anon_b")) / G, 2)
out["shmem_peak_gib"] = round(max(num("cg_shmem_b")) / G, 2)
out["file_peak_gib"] = round(max(num("cg_file_b")) / G, 2)
cur_minus_inactive = [(float(r["cg_current_b"]) - float(r["cg_inactive_file_b"])) / G for r in rows if r.get("cg_current_b")]
out["current_minus_inactive_file_peak_gib"] = round(max(cur_minus_inactive), 2)
stats = []
for line in open(os.path.join(logs, f"{run}.docker-stats.log")):
    parts = line.split()
    if len(parts) >= 2 and parts[1].endswith("iB"):
        v = parts[1]
        unit = {"KiB": 2**10, "MiB": 2**20, "GiB": 2**30}[v[-3:]]
        stats.append(float(v[:-3]) * unit / G)
out["docker_stats_peak_gib"] = round(max(stats), 3) if stats else None
out["meta"] = open(os.path.join(logs, f"{run}.meta")).read().split()
print(json.dumps(out, indent=1, ensure_ascii=False))
