#!/usr/bin/env python3
"""Reduce stub records to request shapes and dsh JSON streams to events.

Usage:
  python3 -I dsh_shapes.py shapes <record.jsonl> <out.jsonl>
  python3 -I dsh_shapes.py events <stdout.jsonl> <out.jsonl>

A record line is one request as the recording stub (or lane B's proxy) wrote it:
n, kind, kind_n, t_arrive, method, path, headers, body, action, t_start, t_end.
A shape line keeps the request's form and drops its content: no messages, no other
body field, no user agent and no reply text.
An event line keeps the structure of a `dsh --json` stream and drops its text,
tool input and output, session id, working folder and usage.
Standard library only.
"""
import json
import sys
from pathlib import Path

BODY_FIELDS = ("max_tokens", "max_completion_tokens", "reasoning_effort", "reasoning",
               "chat_template_kwargs", "stream", "stream_options", "store")
EVENT_FIELDS = ("type", "phase", "turn", "step", "callId", "tool", "status")


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def shape(record, run, t0):
    out = {"run": run}
    for key in ("n", "kind", "kind_n", "method", "path"):
        if key in record:
            out[key] = record[key]
    body = record.get("body")
    if isinstance(body, dict):
        if "model" in body:
            out["model"] = body["model"]
        out["keys"] = sorted(body)
        for key in BODY_FIELDS:
            if key in body:
                out[key] = body[key]
        messages = body.get("messages")
        if isinstance(messages, list) and messages and isinstance(messages[0], dict):
            out["first_role"] = messages[0].get("role")
        if "tools" in body:
            out["tools"] = len(body["tools"] or [])
    headers = record.get("headers")
    if isinstance(headers, dict) and "authorization_scheme" in headers:
        out["authorization"] = headers["authorization_scheme"]
    action = record.get("action")
    if isinstance(action, dict) and "type" in action:
        out["reply"] = action["type"]
    times = {}
    for name in ("arrive", "start", "end"):
        if "t_" + name in record:
            times[name] = round(record["t_" + name] - t0, 2)
    if times:
        out["t"] = times
    return out


def shapes(src, dst):
    records = read_jsonl(src)
    t0 = min((r["t_arrive"] for r in records if "t_arrive" in r), default=0)
    write_jsonl(dst, [shape(r, Path(src).stem, t0) for r in records])


def event(obj):
    out = {key: obj[key] for key in EVENT_FIELDS if key in obj}
    if isinstance(obj.get("input"), dict):
        out["input_keys"] = sorted(obj["input"])
    error = obj.get("error")
    if isinstance(error, dict):
        out["error"] = {key: error[key] for key in ("name", "code") if key in error}
    reason = obj.get("reason")
    if isinstance(reason, dict):
        kept = {key: reason[key] for key in ("kind",) if key in reason}
        if isinstance(reason.get("error"), dict) and "code" in reason["error"]:
            kept["error"] = {"code": reason["error"]["code"]}
        out["reason"] = kept
    return out


def events(src, dst):
    write_jsonl(dst, [event(obj) for obj in read_jsonl(src)])


def main(argv):
    if len(argv) != 4 or argv[1] not in ("shapes", "events"):
        raise SystemExit(__doc__)
    {"shapes": shapes, "events": events}[argv[1]](argv[2], argv[3])


if __name__ == "__main__":
    main(sys.argv)
