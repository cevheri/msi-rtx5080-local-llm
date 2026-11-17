"""The margin table of measurement B, per start: free GPU memory right after ready, after the near-full prompt, after the
first grammar request, after the forced tool request, the lowest free memory in the 1 s trace after ready (the whole
protocol, the 300 s load included), the drop, and the design's D8 threshold rule over the given runs.
Run with python3 -I: margins.py <measure-b dir> <record>/<run> [...]"""
import csv
import json
import math
import os
import sys

M = sys.argv[1]
drops = {}


def fmt(v, missing=''):
    return f'{v:,}' if isinstance(v, int) else missing


print("| Start | Free right after ready | Near-full prompt (tokens, share) | After it | After the first grammar request | After the forced tool request | Lowest in the trace after ready | Drop |")
print("|---|---|---|---|---|---|---|---|")
for item in sys.argv[2:]:
    rec, run = item.split("/")
    shapes, logs = f"{M}/{rec}/shapes", f"{M}/{rec}/logs"
    ready = json.load(open(f"{shapes}/{run}-ready.json"))
    f_ready = ready["gpu_right_after_ready"]["free_mib"]
    t_ready = ready["gpu_right_after_ready"]["t"]
    rows = [json.loads(x) for x in open(f"{shapes}/{run}-seq.jsonl")]
    after, plan, last_step = {}, None, None
    for r in rows:
        if r["event"] == "nearfull_plan" and plan is None:
            plan = r
        if r["event"] == "request":
            last_step = r["step"]
        if r["event"] == "gpu_after" and r["step"] not in after:
            after[r["step"]] = r["free_mib"]
    trace = list(csv.DictReader(open(f"{logs}/{run}-trace.csv")))
    low = min(float(t["gpu_mem_free_mib"]) for t in trace if float(t["epoch"]) >= t_ready and t["gpu_mem_free_mib"])
    low = int(min(low, *(v for v in after.values())))
    drop = f_ready - low
    drops[item] = drop
    share = f"{plan['prompt_tokens']:,}, {plan['share_of_context'] * 100:.1f}%" if plan else "none"
    print(f"| {rec.replace('sglang-', '')} {run} | {f_ready:,} | {share} | {fmt(after.get('nearfull_p1'))} | "
          f"{fmt(after.get('title_first_grammar'))} | {fmt(after.get('forced_tool'), 'not sent')} | {low:,} | {drop} |")
worst = max(drops.values())
min_free = math.ceil((worst + 256) / 64) * 64
if (worst + 256) % 64 == 0:
    min_free += 64
print(f"\nlargest drop {worst} MiB; D8 rule: MIN_FREE = next multiple of 64 above {worst} + 256 = {min_free} MiB; "
      f"ship gate F_ready >= {min_free + 128} MiB")
