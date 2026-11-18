"""The margin table of measurements B and C, per start, with measurement D's flushed reading added: free GPU memory right
after ready (the poller's reading, without a flush, as B and C read it), after D8's POST /flush_cache (seq_d.py's
flushready phase; none for B's and C's starts), after the near-full prompt, after the first grammar request, after the
forced tool request, the lowest free memory in the 1 s trace after ready, and the drop from each ready reading.
With --until STEP the lowest is taken only up to the end of that phase of the start (for example probe), so that a
margin-only start can be compared with a full-protocol start over the same requests.
Run with python3 -I: margins_d.py [--until STEP] <base dir>/<record>/<run> [...]"""
import argparse
import csv
import json
import math
import os

ap = argparse.ArgumentParser()
ap.add_argument("--until", default=None)
ap.add_argument("runs", nargs="+")
a = ap.parse_args()
drops = {}


def fmt(v, missing=""):
    return f"{v:,}" if isinstance(v, int) else missing


print("| Start | Free right after ready | After D8's flush | Near-full prompt (tokens, share) | After it | "
      "After the first grammar request | After the forced tool request | Lowest after ready"
      + (f" (to the end of {a.until})" if a.until else "") + " | Drop from ready | Drop from the flushed reading |")
print("|---|---|---|---|---|---|---|---|---|---|")
for item in a.runs:
    rec_dir, run = os.path.split(item)
    base, rec = os.path.split(rec_dir)
    lane = os.path.basename(base).replace("measure-", "").upper()
    shapes, logs = f"{rec_dir}/shapes", f"{rec_dir}/logs"
    ready = json.load(open(f"{shapes}/{run}-ready.json"))
    f_ready = ready["gpu_right_after_ready"]["free_mib"]
    t_ready = ready["gpu_right_after_ready"]["t"]
    rows = [json.loads(x) for x in open(f"{shapes}/{run}-seq.jsonl")]
    after, plan, flushed, t_end = {}, None, None, None
    for r in rows:
        if r["event"] == "nearfull_plan" and plan is None:
            plan = r
        if r["event"] == "flush_ready" and flushed is None:
            flushed = r["free_mib_after"]
        if r["event"] == "gpu_after" and r["step"] not in after:
            after[r["step"]] = r["free_mib"]
        if a.until and r["event"] == "phase_end" and r["step"] == a.until and t_end is None:
            t_end = r["t"]
    trace = list(csv.DictReader(open(f"{logs}/{run}-trace.csv")))
    window = [t for t in trace if float(t["epoch"]) >= t_ready and t["gpu_mem_free_mib"]
              and (t_end is None or float(t["epoch"]) <= t_end)]
    low = min(float(t["gpu_mem_free_mib"]) for t in window)
    low = int(min(low, *(v for v in after.values())))
    drop = f_ready - low
    drop_f = flushed - low if isinstance(flushed, int) else None
    drops[item] = drop_f if drop_f is not None else drop
    share = f"{plan['prompt_tokens']:,}, {plan['share_of_context'] * 100:.1f}%" if plan else "none"
    print(f"| {lane} {rec.replace('sglang-', '')} {run} | {f_ready:,} | {fmt(flushed, 'not read')} | {share} | "
          f"{fmt(after.get('nearfull_p1'))} | {fmt(after.get('title_first_grammar'))} | "
          f"{fmt(after.get('forced_tool'), 'not sent')} | {low:,} | {drop} | {fmt(drop_f, 'none')} |")
worst = max(drops.values())
min_free = math.ceil((worst + 256) / 64) * 64
if (worst + 256) % 64 == 0:
    min_free += 64
print(f"\nlargest drop over these starts {worst} MiB (the flushed reading where there is one); D8 rule: MIN_FREE = next "
      f"multiple of 64 above {worst} + 256 = {min_free} MiB; ship gate F_ready >= {min_free + 128} MiB")
