"""Shared helpers for the measure-a probes: HTTP without proxies, nvidia-smi and docker state, JSONL records.
Standard library only; the probe scripts import it after putting their own folder on sys.path."""
import json
import subprocess
import time
import urllib.error
import urllib.request

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

TOOL = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}


def http(base, path, payload=None, timeout=1800, method=None):
    """One request; returns (head, parsed_json_or_None). head has status (None on a socket error), seconds, error."""
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"},
                                 method=method or ("POST" if data is not None else "GET"))
    t0 = time.time()
    try:
        with OPENER.open(req, timeout=timeout) as resp:
            body, status = resp.read(), resp.status
    except urllib.error.HTTPError as err:
        body, status = err.read(), err.code
    except (urllib.error.URLError, OSError) as err:
        reason = getattr(err, "reason", err)
        return {"status": None, "error": f"{type(reason).__name__}: {reason}"[:200], "t": round(t0, 3),
                "seconds": round(time.time() - t0, 3)}, None
    head = {"status": status, "t": round(t0, 3), "seconds": round(time.time() - t0, 3)}
    try:
        return head, json.loads(body) if body else None
    except ValueError:
        head["body_text"] = body.decode(errors="replace")[:300]
        return head, None


def gpu():
    """nvidia-smi view of the whole GPU, in MiB."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
    used, free, total = (int(x) for x in out.strip().split(","))
    return {"t": round(time.time(), 3), "used_mib": used, "free_mib": free, "total_mib": total}


def container_state(name):
    out = subprocess.run(["docker", "inspect", "--format",
                          "{{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}}", name],
                         capture_output=True, text=True, timeout=20)
    return out.stdout.strip() if out.returncode == 0 else "missing"


def preview(text, n=80):
    if text is None:
        return None
    text = str(text)
    return text if len(text) <= n else text[:n] + f"... ({len(text)} chars)"


def chat_summary(head, body):
    """Status, finish reason, short content, tool call names and usage of a chat completion."""
    out = dict(head)
    if not isinstance(body, dict):
        return out
    if "choices" not in body:
        out["error_body"] = {k: preview(v, 300) for k, v in body.items()}
        return out
    choice = body["choices"][0]
    msg = choice.get("message") or {}
    out["finish_reason"] = choice.get("finish_reason")
    out["content"] = preview(msg.get("content"))
    out["reasoning_content_chars"] = len(msg.get("reasoning_content") or "")
    out["tool_calls"] = [{"name": (c.get("function") or {}).get("name"),
                          "arguments": preview((c.get("function") or {}).get("arguments"), 80)}
                         for c in (msg.get("tool_calls") or [])]
    out["usage"] = {k: v for k, v in (body.get("usage") or {}).items() if k in ("prompt_tokens", "completion_tokens")}
    return out


class Records:
    """Append-only JSONL file, one event per line."""

    def __init__(self, path):
        self.path = path
        open(path, "w").close()

    def add(self, **event):
        event.setdefault("t", round(time.time(), 3))
        event["utc"] = time.strftime("%H:%M:%S", time.gmtime(event["t"]))
        with open(self.path, "a") as f:
            f.write(json.dumps(event) + "\n")
        print(json.dumps(event)[:600], flush=True)
        return event
