#!/usr/bin/env python3
"""Reduce recording-stub captures to request shapes.

Usage: python3 -I extract_shapes.py <capture file or folder> <out.jsonl> [--run NAME]

The stub servers (bin/design/stub_server.py, bin/design/pi/stub_server.py and
their Phase 1 copies) write one JSON file per request, named
NNN-<METHOD>-<path>.json, into a captures/ folder.
Each record holds the full request body and headers, so it may hold a system
prompt read from the owner's home.
This script keeps only the request shape, one JSON line per request:

  run, method, path, model, keys (sorted top-level body keys), max_tokens,
  max_completion_tokens, reasoning_effort, chat_template_kwargs,
  response_format_type, stream, store, tools (function names),
  authorization (the scheme word of the Authorization header, or null when
  the header has no scheme word)

A key is left out when the request lacks it.
The run is the capture folder relative to the given folder (without a
trailing captures/ part), or the given name with --run.
"""
import json
import pathlib
import re
import sys
from urllib.parse import urlsplit

CAPTURE_NAME = re.compile(r"^\d{3}-[A-Z]+-.*\.json$")
VALUE_KEYS = ("max_tokens", "max_completion_tokens", "reasoning_effort", "chat_template_kwargs", "stream", "store")


def run_name(capture, root):
    folder = capture.parent
    if folder.name == "captures":
        folder = folder.parent
    rel = folder.relative_to(root) if folder != root and root in folder.parents else pathlib.Path(".")
    return rel.as_posix() if rel.as_posix() != "." else folder.name


def tool_names(tools):
    names = []
    for tool in tools:
        if not isinstance(tool, dict):
            raise ValueError(f"unexpected tool entry: {type(tool).__name__}")
        function = tool.get("function")
        name = function.get("name") if isinstance(function, dict) else tool.get("name")
        if not isinstance(name, str):
            raise ValueError("tool entry without a name")
        names.append(name)
    return names


def shape(record, run):
    out = {"run": run, "method": record["method"], "path": urlsplit(record["path"]).path}
    body = record.get("body")
    if isinstance(body, dict):
        if "model" in body:
            out["model"] = body["model"]
        out["keys"] = sorted(body)
        for key in VALUE_KEYS[:4]:
            if key in body:
                out[key] = body[key]
        response_format = body.get("response_format")
        if isinstance(response_format, dict) and "type" in response_format:
            out["response_format_type"] = response_format["type"]
        for key in VALUE_KEYS[4:]:
            if key in body:
                out[key] = body[key]
        if "tools" in body:
            out["tools"] = tool_names(body["tools"])
    for name, value in (record.get("headers") or {}).items():
        if name.lower() == "authorization":
            scheme, sep, _ = value.partition(" ")
            out["authorization"] = scheme if sep and scheme else None
    return out


def captures(source):
    if source.is_file():
        return source.parent, [source]
    files = sorted(p for p in source.rglob("*.json") if CAPTURE_NAME.match(p.name))
    return source, files


def main(argv):
    args = list(argv)
    run_override = None
    if "--run" in args:
        i = args.index("--run")
        run_override = args[i + 1]
        del args[i:i + 2]
    if len(args) != 2:
        sys.exit(__doc__.split("\n\n")[1])
    source, out_path = pathlib.Path(args[0]), pathlib.Path(args[1])
    root, files = captures(source)
    if not files:
        sys.exit(f"no capture files under {source}")
    lines = []
    for capture in files:
        record = json.loads(capture.read_text(encoding="utf-8"))
        run = run_override or run_name(capture, root)
        lines.append(json.dumps(shape(record, run), ensure_ascii=False, sort_keys=False))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(lines)} requests -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
