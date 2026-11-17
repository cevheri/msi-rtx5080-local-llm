#!/usr/bin/env python3
"""One line per case: exit, the thinking fields of each recorded request, and any error event.

summary.py <case> [<case> ...]
"""
import json
import os
import sys

LB = "$SCRATCH/laneB.GwYhns"


def fields(body):
    out = {}
    for key in ("reasoning_effort", "chat_template_kwargs", "max_tokens", "max_completion_tokens", "store",
                "temperature", "top_p", "response_format", "tool_choice"):
        if key in body:
            out[key] = body[key]
    msgs = body.get("messages") or []
    out["sys_role"] = msgs[0].get("role") if msgs else None
    out["tools"] = len(body.get("tools") or []) if "tools" in body else 0
    return out


for case in sys.argv[1:]:
    meta = open(os.path.join(LB, "runs", case, "meta.txt")).read().strip()
    rec_path = os.path.join(LB, "rec", case + ".jsonl")
    recs = []
    if os.path.exists(rec_path):
        recs = [json.loads(line) for line in open(rec_path) if line.strip()]
        recs = [r for r in recs if r.get("method") == "POST"]
        recs.sort(key=lambda r: r["t_arrive"])
    print(f"## {case}: {meta}")
    for r in recs:
        print(f"   {r['kind']}#{r['kind_n']}: {json.dumps(fields(r['body']))}")
    events = []
    for line in open(os.path.join(LB, "runs", case, "stdout.jsonl")):
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get("type") in ("error",) or (ev.get("type") == "final" and ev.get("reason", {}).get("kind") != "completed"):
            events.append(ev)
        if ev.get("type") == "status" and ev.get("phase") == "turn_end" and ev.get("reason", {}).get("kind") != "completed":
            events.append(ev)
    for ev in events:
        print("   EVENT", json.dumps(ev)[:600])
    err = open(os.path.join(LB, "runs", case, "stderr.txt")).read().strip()
    if err:
        print("   STDERR", err[:800].replace("\n", " | "))
