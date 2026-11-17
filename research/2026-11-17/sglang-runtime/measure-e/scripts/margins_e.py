"""The margin table of measurements B, C and D (measurement D's margins_d.py) with measurement E's rules:
- free GPU memory right after ready (the poller's reading, without a flush), after D8's POST /flush_cache (seq_d.py's
  flushready phase; "not read" for B's and C's starts), after the near-full prompt, after the first grammar request and
  after the forced tool request;
- the lowest free memory over the protocol: for a start with a flushed reading, over everything after that reading (the
  1 s trace and every seq_d.py reading from the flush to the seq's end event); for a start without one, over everything
  after the ready reading, as B, C and D took it; the window ends at the seq's end event (or at the end of --until STEP);
- the drop from each reading, and D8's threshold over the starts given: the largest drop (from the flushed reading where
  there is one) plus 256 MiB, rounded up to a multiple of 64 where a sum that is already a multiple stays (the rule as
  measurement E's brief states it), and the ship gate 128 MiB above that; the line also prints what the design's wording
  "the next multiple of 64 above" (margins_d.py) would give when the two differ.
With --exclude RECORD/RUN (repeatable) the rule is also printed without those starts.
Run with python3 -I: margins_e.py [--until STEP] [--exclude REC/RUN ...] <base dir>/<record>/<run> [...]"""
import argparse
import csv
import json
import math
import os

ap = argparse.ArgumentParser()
ap.add_argument("--until", default=None)
ap.add_argument("--exclude", action="append", default=[])
ap.add_argument("runs", nargs="+")
a = ap.parse_args()
drops = {}


def fmt(v, missing=""):
    return f"{v:,}" if isinstance(v, int) else missing


def rule(worst):
    s = worst + 256
    brief = math.ceil(s / 64) * 64
    above = brief + 64 if s % 64 == 0 else brief
    return s, brief, above


print("| Start | Free right after ready | After D8's flush | Near-full prompt (tokens, share) | After it | "
      "After the first grammar request | After the forced tool request | Lowest over the protocol"
      + (f" (to the end of {a.until})" if a.until else "") + " | Drop from ready | Drop from the flushed reading |")
print("|---|---|---|---|---|---|---|---|---|---|")
for item in a.runs:
    rec_dir, run = os.path.split(item.rstrip("/"))
    base, rec = os.path.split(rec_dir)
    lane = os.path.basename(base).replace("measure-", "").upper()
    shapes, logs = f"{rec_dir}/shapes", f"{rec_dir}/logs"
    ready = json.load(open(f"{shapes}/{run}-ready.json"))
    f_ready = ready["gpu_right_after_ready"]["free_mib"]
    t_ready = ready["gpu_right_after_ready"]["t"]
    rows = [json.loads(x) for x in open(f"{shapes}/{run}-seq.jsonl")]
    after, plan, flushed, t_flush, t_end, t_seq_end = {}, None, None, None, None, None
    for r in rows:
        if r["event"] == "nearfull_plan" and plan is None:
            plan = r
        if r["event"] == "flush_ready" and flushed is None:
            flushed, t_flush = r["free_mib_after"], r["t_after"]
        if r["event"] == "gpu_after" and r["step"] not in after:
            after[r["step"]] = r["free_mib"]
        if a.until and r["event"] == "phase_end" and r["step"] == a.until and t_end is None:
            t_end = r["t"]
        if r["event"] == "end":
            t_seq_end = r["t"]
    t_start = t_flush if t_flush is not None else t_ready
    t_stop = t_end if t_end is not None else t_seq_end
    readings = [r["free_mib"] for r in rows if isinstance(r.get("free_mib"), int)
                and r["t"] >= t_start and (t_stop is None or r["t"] <= t_stop)]
    if t_flush is not None:
        readings.append(flushed)
    trace = list(csv.DictReader(open(f"{logs}/{run}-trace.csv")))
    window = [float(t["gpu_mem_free_mib"]) for t in trace if t["gpu_mem_free_mib"] and float(t["epoch"]) >= t_start
              and (t_stop is None or float(t["epoch"]) <= t_stop)]
    low = int(min(window + readings))
    drop = f_ready - low
    drop_f = flushed - low if isinstance(flushed, int) else None
    drops[f"{rec}/{run}"] = drop_f if drop_f is not None else drop
    share = f"{plan['prompt_tokens']:,}, {plan['share_of_context'] * 100:.1f}%" if plan else "none"
    print(f"| {lane} {rec.replace('sglang-', '')} {run} | {f_ready:,} | {fmt(flushed, 'not read')} | {share} | "
          f"{fmt(after.get('nearfull_p1'))} | {fmt(after.get('title_first_grammar'))} | "
          f"{fmt(after.get('forced_tool'), 'not sent')} | {low:,} | {drop} | {fmt(drop_f, 'none')} |")


def report(label, d):
    worst_key = max(d, key=d.get)
    worst = d[worst_key]
    s, brief, above = rule(worst)
    line = (f"{label}: largest drop {worst} MiB ({worst_key}); {worst} + 256 = {s}; MIN_FREE = {brief} MiB (rounded up to a "
            f"multiple of 64, a multiple stays); ship gate F_ready >= {brief + 128} MiB")
    if above != brief:
        line += f"; the design's 'next multiple of 64 above' would give {above} and {above + 128} MiB"
    print(line)


print()
report(f"over these {len(drops)} starts (the flushed reading where there is one)", drops)
if a.exclude:
    kept = {k: v for k, v in drops.items() if k not in a.exclude}
    report(f"without {', '.join(a.exclude)}", kept)
