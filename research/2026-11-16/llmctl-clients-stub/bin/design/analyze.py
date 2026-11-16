#!/usr/bin/env python3
"""Summarize the requests captured by stub_server.py for one or more run folders.
usage: analyze.py <run-dir> [<run-dir> ...] [--full]"""
import json
import sys
from pathlib import Path

MAXTOK = ("max_tokens", "max_completion_tokens", "max_output_tokens", "n_predict")
SHOWN = {"model", "messages", "tools", "tool_choice", "stream", "stream_options", "reasoning_effort",
         "chat_template_kwargs", *MAXTOK}


def summarize(path: Path, full: bool):
    rec = json.loads(path.read_text())
    line = f"{path.name}: {rec['method']} {rec['path']} -> {rec.get('reply_status')}"
    if rec["method"] != "POST" or not isinstance(rec.get("body"), dict):
        return line
    b = rec["body"]
    tools = b.get("tools") or []
    names = [t.get("function", {}).get("name") for t in tools]
    roles = [m.get("role") for m in b.get("messages", [])]
    out = [line,
           f"    model={b.get('model')!r} stream={b.get('stream')!r} stream_options={b.get('stream_options')!r}",
           f"    max-token fields={ {k: b[k] for k in MAXTOK if k in b} }",
           f"    reasoning_effort={b.get('reasoning_effort', '<absent>')!r} chat_template_kwargs={b.get('chat_template_kwargs', '<absent>')!r}",
           f"    tools={len(tools)} {names} tool_choice={b.get('tool_choice', '<absent>')!r}",
           f"    roles={roles}",
           f"    other top-level keys={ {k: b[k] for k in b if k not in SHOWN} }"]
    h = rec.get("headers", {})
    out.append(f"    Authorization={h.get('Authorization', '<absent>')!r} User-Agent={h.get('User-Agent')!r}")
    extra = {k: v for k, v in h.items() if k.lower().startswith(("x-", "openai", "hermes")) and not k.lower().startswith("x-stainless")}
    stainless = sorted(k for k in h if k.lower().startswith("x-stainless"))
    out.append(f"    other headers={extra} x-stainless headers={stainless}")
    reply = rec.get("reply") or {}
    if isinstance(reply, dict):
        tc = [c["function"]["name"] for c in (reply.get("tool_calls") or [])]
        out.append(f"    stub reply: tool_calls={tc} text={(reply.get('text') or '')[:60]!r}")
    if full:
        out.append(json.dumps(b, indent=1)[:6000])
    return "\n".join(out)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    full = "--full" in sys.argv
    for run in args:
        run = Path(run)
        print(f"===== {run}")
        for f in ("command.txt", "exit.txt"):
            if (run / f).exists():
                print(f"{f}: {(run / f).read_text().strip()}")
        for p in sorted((run / "captures").glob("*.json")):
            print(summarize(p, full))


if __name__ == "__main__":
    main()
