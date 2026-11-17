"""Compare two M13 runs (FP8 KV against BF16 KV) prompt by prompt: correct answers per run and matching answers.
An arithmetic or code answer matches when the extracted answers are equal; a tool answer when the same function is
called with the same arguments. Run with python3 -I: m13_compare.py FP8.json BF16.json"""
import json
import sys

a, b = (json.load(open(p)) for p in sys.argv[1:3])
rows, match = [], 0
for x, y in zip(a, b):
    same = x.get("answer") == y.get("answer")
    match += same
    rows.append({"i": x["i"], "kind": x["kind"], "fp8": x.get("answer"), "bf16": y.get("answer"),
                 "fp8_correct": x.get("correct"), "bf16_correct": y.get("correct"), "match": same,
                 "fp8_reasoning_chars": x.get("reasoning_chars"), "bf16_reasoning_chars": y.get("reasoning_chars")})
res = {"prompts": len(rows), "fp8_correct": sum(1 for r in rows if r["fp8_correct"]),
       "bf16_correct": sum(1 for r in rows if r["bf16_correct"]), "matching_answers": match,
       "by_kind": {k: {"n": sum(1 for r in rows if r["kind"] == k),
                       "fp8_correct": sum(1 for r in rows if r["kind"] == k and r["fp8_correct"]),
                       "bf16_correct": sum(1 for r in rows if r["kind"] == k and r["bf16_correct"]),
                       "match": sum(1 for r in rows if r["kind"] == k and r["match"])} for k in ("arith", "code", "tool")},
       "differences": [r for r in rows if not r["match"] or not r["fp8_correct"] or not r["bf16_correct"]]}
print(json.dumps(res, indent=1, default=str))
