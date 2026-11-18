"""Summarize the measure-a runs: M7 first-200 times per endpoint and the ready log line after the docker run call,
and the M6 nvidia-smi free memory before and after each request with the outcome. Run with python3 -I: summarize.py <measure-a dir>"""
import datetime
import glob
import json
import os
import re
import sys

M = sys.argv[1]
print("M7: seconds after the docker run call (poll every 0.5 s); ready = the 'fired up' log line (1 s resolution)")
for t0_file in sorted(glob.glob(f"{M}/logs/*/*.t0")):
    model, run = t0_file.split("/")[-2], os.path.basename(t0_file)[:-3]
    t0 = float(open(t0_file).read())
    m7 = f"{M}/shapes/{model}/{run}-m7.json"
    log = f"{M}/logs/{model}/{run}.log"
    ready = None
    if os.path.exists(log):
        for line in open(log, errors="replace"):
            m = re.match(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] The server is fired up", line)
            if m:
                ts = datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=datetime.timezone.utc).timestamp()
                ready = round(ts - t0, 1)
                break
    first = {}
    before = {}
    if os.path.exists(m7):
        d = json.load(open(m7))
        for p, s in d["endpoints"].items():
            first[p] = s["first_200_s"]
            before[p] = sorted({str(c["status"]) for c in s["changes"] if c["status"] != 200})
    print(f"  {model:14} {run:22} ready_line={ready} first_200={first}")
    if run.endswith("m6-o1"):
        print(f"  {'':14} {'':22} statuses before 200: {before}")
print()
print("M6: nvidia-smi free MiB before -> after each request, and the outcome")
for f in sorted(glob.glob(f"{M}/shapes/*/m6-*.jsonl") + glob.glob(f"{M}/shapes/*/m15-*.jsonl")):
    rows = [json.loads(x) for x in open(f)]
    model, run = f.split("/")[-2], os.path.basename(f)[:-6]
    out, before = [], None
    for r in rows:
        if r["event"] == "start":
            out.append(f"start {r['free_mib']}")
        elif r["event"] == "gpu_before":
            before = r["free_mib"]
        elif r["event"] == "request":
            tag = r["step"]
            res = r["status"]
            if r.get("title_parses") is not None:
                res = f"{res} title_parses={r['title_parses']}"
            if r.get("tool_calls"):
                res = f"{res} tool={r['tool_calls'][0]['name']}"
            elif r["step"].startswith("forced"):
                res = f"{res} finish={r.get('finish_reason')} no tool_calls"
            pt = (r.get("usage") or {}).get("prompt_tokens")
            out.append(f"{tag}({pt} tok) {before}->{{after}} {res}")
        elif r["event"] == "gpu_after":
            out[-1] = out[-1].replace("{after}", str(r["free_mib"]))
        elif r["event"] == "decode_check":
            out.append(f"decode {r['decode_tok_s']} tok/s")
        elif r["event"] == "end":
            out.append(f"end: {r['result']}")
    print(f"  {model} {run}:")
    for o in out:
        print(f"      {o}")
