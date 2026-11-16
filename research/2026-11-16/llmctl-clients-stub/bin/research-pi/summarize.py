#!/usr/bin/env python3
"""Summarize captured chat-completion requests: python3 -I summarize.py <capture-dir> [--full N]."""
import json
import os
import sys

SKIP = {"messages", "tools"}


def summarize(path, full=False):
    entry = json.load(open(path))
    body = entry.get("body") or {}
    out = {"file": os.path.basename(path), "path": entry["path"]}
    if not isinstance(body, dict):
        out["body"] = body
        return out
    out["top_level_keys"] = sorted(body.keys())
    out["fields"] = {k: v for k, v in body.items() if k not in SKIP}
    tools = body.get("tools")
    out["tools"] = None if tools is None else [t.get("function", {}).get("name") for t in tools]
    msgs = body.get("messages") or []
    out["roles"] = [m.get("role") for m in msgs]
    replay = []
    for m in msgs:
        if m.get("role") == "assistant":
            replay.append({k: (v if k != "tool_calls" else [tc["function"]["name"] for tc in v]) for k, v in m.items()})
    out["assistant_messages"] = replay
    sys_msgs = [m for m in msgs if m.get("role") in ("system", "developer")]
    out["instruction_role"] = [m.get("role") for m in sys_msgs]
    out["system_prompt_chars"] = sum(len(m.get("content") or "") for m in sys_msgs if isinstance(m.get("content"), str))
    hdr = entry.get("headers", {})
    out["headers"] = {k: v for k, v in hdr.items() if k.lower() not in ("host", "content-length")}
    if full:
        out["body_full"] = body
    return out


if __name__ == "__main__":
    directory = sys.argv[1]
    full_n = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[2] == "--full" else None
    for i, name in enumerate(sorted(os.listdir(directory)), 1):
        print(json.dumps(summarize(os.path.join(directory, name), full=(full_n == i)), indent=1, ensure_ascii=False))
