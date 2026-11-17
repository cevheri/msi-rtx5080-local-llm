#!/usr/bin/env python3
"""Print request shapes from stub records: shapes.py <record.jsonl> [--tools] [--times]."""
import json
import sys

SAMPLING = ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "frequency_penalty",
            "repetition_penalty", "seed", "stop", "n", "logprobs")


def shape(rec, show_tools=False):
    body = rec.get("body") or {}
    msgs = body.get("messages") or []
    roles = [m.get("role") for m in msgs]
    first = roles[0] if roles else None
    out = {
        "n": rec.get("n"), "kind": rec.get("kind"),
        "model": body.get("model"),
        "keys": sorted(body.keys()),
        "max_tokens": body.get("max_tokens"),
        "max_completion_tokens": body.get("max_completion_tokens"),
        "stream": body.get("stream"),
        "stream_options": body.get("stream_options"),
        "store": body.get("store", "<absent>"),
        "first_role": first,
        "roles": roles,
        "tools": len(body.get("tools") or []) if "tools" in body else "<absent>",
        "tool_choice": body.get("tool_choice", "<absent>"),
        "reasoning_effort": body.get("reasoning_effort", "<absent>"),
        "chat_template_kwargs": body.get("chat_template_kwargs", "<absent>"),
        "response_format": body.get("response_format", "<absent>"),
        "sampling": {k: body[k] for k in SAMPLING if k in body},
        "other": {k: body[k] for k in body if k not in (
            "model", "messages", "stream", "stream_options", "max_tokens", "max_completion_tokens", "store",
            "tools", "tool_choice", "reasoning_effort", "chat_template_kwargs", "response_format", *SAMPLING)},
        "status": rec.get("action", {}).get("type"),
        "ua": (rec.get("headers") or {}).get("user-agent"),
        "auth": (rec.get("headers") or {}).get("authorization_scheme"),
    }
    if show_tools:
        out["tool_names"] = [t.get("function", {}).get("name") for t in body.get("tools") or []]
    return out


def main():
    path = sys.argv[1]
    show_tools = "--tools" in sys.argv
    times = "--times" in sys.argv
    recs = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    recs = [r for r in recs if r.get("method") == "POST"]
    recs.sort(key=lambda r: r["t_arrive"])
    t0 = recs[0]["t_arrive"] if recs else 0
    for rec in recs:
        s = shape(rec, show_tools)
        if times:
            s["t"] = {"arrive": round(rec["t_arrive"] - t0, 2), "start": round(rec["t_start"] - t0, 2),
                      "end": round(rec["t_end"] - t0, 2)}
        print(json.dumps(s, ensure_ascii=False))


if __name__ == "__main__":
    main()
