#!/usr/bin/env python3
"""Summarize proxy records of real llama.cpp runs: real_summary.py <case> [...]."""
import json
import os
import sys

LB = "$SCRATCH/laneB.GwYhns"

for case in sys.argv[1:]:
    meta = open(os.path.join(LB, "runs", case, "meta.txt")).read().strip()
    print(f"## {case}: {meta}")
    path = os.path.join(LB, "rec", case + ".jsonl")
    recs = [json.loads(line) for line in open(path) if line.strip()] if os.path.exists(path) else []
    recs.sort(key=lambda r: r["t_arrive"])
    t0 = recs[0]["t_arrive"] if recs else 0
    for r in recs:
        b = r.get("body") or {}
        system = (b.get("messages") or [{}])[0].get("content", "")
        kind = "title" if isinstance(system, str) and "Create a concise title" in system else "main"
        last = (b.get("messages") or [{}])[-1]
        if isinstance(last.get("content"), str) and "compaction engine" in last["content"]:
            kind = "compaction"
        s = r.get("summary") or {}
        timings = s.get("timings") or {}
        print("  ", kind, "status", r.get("status"),
              "arrive", round(r["t_arrive"] - t0, 2),
              "first_tok", round(r["t_first_token"] - t0, 2) if r.get("t_first_token") else None,
              "end", round(r["t_end"] - t0, 2),
              "sent", json.dumps({k: b[k] for k in ("reasoning_effort", "chat_template_kwargs", "max_tokens")
                                  if k in b}),
              "tools", len(b.get("tools") or []),
              "got", json.dumps(s.get("fields")), "finish", s.get("finish"),
              "usage", json.dumps(s.get("usage")),
              "prompt_n", timings.get("prompt_n"), "cache_n", timings.get("cache_n"),
              "prompt_ms", round(timings.get("prompt_ms", 0)), "pred_n", timings.get("predicted_n"))
        if r.get("error_body"):
            print("     error_body", r["error_body"][:500])
    for line in open(os.path.join(LB, "runs", case, "stdout.jsonl")):
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get("type") in ("thinking", "text", "final", "error"):
            text = json.dumps(ev, ensure_ascii=False)
            print("   EV", text[:260])
