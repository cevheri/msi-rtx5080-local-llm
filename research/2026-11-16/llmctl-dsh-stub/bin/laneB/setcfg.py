#!/usr/bin/env python3
"""Point the stub at rec/<case>.jsonl and merge JSON overrides into its config.

setcfg.py <case> ['{"single_slot": true, "rules": [...]}']
"""
import json
import os
import sys

LB = "$SCRATCH/laneB.GwYhns"
case = sys.argv[1]
overrides = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
record = os.path.join(LB, "rec", case + ".jsonl")
if os.path.exists(record):
    os.remove(record)
cfg = {
    "record": record,
    "single_slot": False,
    "reasoning_field": "reasoning_content",
    "models": ["local"],
    "default": {"type": "text", "text": "Hello from stub.", "reasoning": "Thinking briefly.",
                "usage": {"prompt_tokens": 1200, "completion_tokens": 12}},
    "rules": [],
}
cfg.update(overrides)
tmp = os.path.join(LB, "cfg", "stub.json.tmp")
with open(tmp, "w", encoding="utf-8") as handle:
    json.dump(cfg, handle, indent=1)
os.replace(tmp, os.path.join(LB, "cfg", "stub.json"))
print("stub ->", record)
