"""List every start of measurement B: record folder, run, docker run time (UTC), seconds to the ready line or to the
readiness poll, the outcome and the container state. Run with python3 -I: starts_table.py <measure-b dir>"""
import datetime
import glob
import json
import os
import re
import sys

M = sys.argv[1]
rows = []
for t0_file in glob.glob(f"{M}/*/logs/*.t0"):
    rec = t0_file.split("/")[-3]
    run = os.path.basename(t0_file)[:-3]
    t0 = float(open(t0_file).read())
    logs = os.path.dirname(t0_file)
    shapes = os.path.join(os.path.dirname(logs), "shapes")
    meta = open(f"{logs}/{run}.meta").read() if os.path.exists(f"{logs}/{run}.meta") else ""
    state = open(f"{logs}/{run}.state.txt").read().strip() if os.path.exists(f"{logs}/{run}.state.txt") else ""
    ready_line = None
    for name in (f"{logs}/{run}.log", f"{logs}/{run}.full.log"):
        if os.path.exists(name):
            for line in open(name, errors="replace"):
                m = re.match(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] The server is fired up", line)
                if m:
                    ts = datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=datetime.timezone.utc).timestamp()
                    ready_line = round(ts - t0, 1)
                    break
            if ready_line is not None:
                break
    poll = None
    rj = f"{shapes}/{run}-ready.json"
    if os.path.exists(rj):
        d = json.load(open(rj))
        poll = d.get("ready_seen_s") if d.get("exit") == "ready" else d.get("exit")
    exec_poll = f"{shapes}/{run}-exec-poll.txt"
    if os.path.exists(exec_poll):
        for line in open(exec_poll):
            m = re.match(r"at ([\d.]+) s served name listed", line)
            if m:
                poll = float(m.group(1))
    rows.append((t0, rec, run, ready_line, poll, "ready=yes" in meta or (isinstance(poll, float)), state.split(" started")[0]))
for t0, rec, run, ready_line, poll, ok, state in sorted(rows):
    print(f"{datetime.datetime.fromtimestamp(t0, datetime.timezone.utc):%H:%M:%S} {rec:34} {run:18} ready_line={ready_line} poll={poll} {state}")
