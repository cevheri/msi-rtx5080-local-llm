#!/usr/bin/env python3
"""Drive `pi --mode rpc` to see whether a running Pi picks up a rewritten models.json and whether RPC model
or thinking changes write settings.json. Usage: python3 -I rpc_live.py <scenario> <models-before> <models-after>"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request

R = "$SCRATCH/clients/research-pi"
scenario, models_before, models_after = sys.argv[1], sys.argv[2], sys.argv[3]
port = open(f"{R}/stub/port").read().strip()
run = f"{R}/runs/{scenario}"
shutil.rmtree(run, ignore_errors=True)
shutil.rmtree(f"{R}/captures/{scenario}", ignore_errors=True)
for sub in ("agent", "home", "work", "tmp", "xdg/config", "xdg/data", "xdg/cache", "xdg/state"):
    os.makedirs(f"{run}/{sub}", exist_ok=True)


def install(src):
    text = open(src).read().replace("__PORT__", port)
    tmp = f"{run}/agent/models.json.tmp"
    open(tmp, "w").write(text)
    os.replace(tmp, f"{run}/agent/models.json")


install(models_before)
settings = {"defaultThinkingLevel": "off"}
open(f"{run}/agent/settings.json", "w").write(json.dumps(settings, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16] if os.path.exists(path) else None


def control(**kw):
    body = {"scenario": scenario, "mode": "text", "model_id": "local", "reasoning_field": "reasoning_content"}
    body.update(kw)
    req = urllib.request.Request(f"http://127.0.0.1:{port}/__control", data=json.dumps(body).encode(), method="POST")
    urllib.request.urlopen(req, timeout=5).read()


control(reset_seq=True)
env = {
    "HOME": f"{run}/home", "TMPDIR": f"{run}/tmp", "PI_CODING_AGENT_DIR": f"{run}/agent",
    "XDG_CONFIG_HOME": f"{run}/xdg/config", "XDG_DATA_HOME": f"{run}/xdg/data",
    "XDG_CACHE_HOME": f"{run}/xdg/cache", "XDG_STATE_HOME": f"{run}/xdg/state",
    "PATH": "$HOME/.nvm/versions/node/v24.22.1/bin:/usr/local/bin:/usr/bin:/bin",
    "TERM": "dumb", "LANG": "C.UTF-8", "PI_OFFLINE": "1", "PI_SKIP_VERSION_CHECK": "1", "PI_TELEMETRY": "0",
}
cmd = ["$HOME/.pi/agent/bin/pi", "--mode", "rpc", "--no-session", "--no-approve", "--no-context-files",
       "--offline", "--tools", "read,write", "--model", "llmctl/local"]
proc = subprocess.Popen(cmd, cwd=f"{run}/work", env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=open(f"{run}/stderr.txt", "w"), text=True, bufsize=1)
events = []
lock = threading.Condition()


def reader():
    with open(f"{run}/stdout.jsonl", "w") as log:
        for line in proc.stdout:
            log.write(line)
            log.flush()
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            with lock:
                events.append(ev)
                lock.notify_all()


threading.Thread(target=reader, daemon=True).start()


def send(obj, wait_for):
    """Send a command and wait until an event matching wait_for appears after it."""
    with lock:
        start = len(events)
    proc.stdin.write(json.dumps(obj) + "\n")
    proc.stdin.flush()
    deadline = time.time() + 30
    with lock:
        while time.time() < deadline:
            for ev in events[start:]:
                if wait_for(ev):
                    return ev
            lock.wait(0.5)
    raise SystemExit(f"timeout waiting after {obj}")


def response(cmd_id):
    return lambda ev: ev.get("type") == "response" and ev.get("id") == cmd_id


def settled(ev):
    return ev.get("type") == "agent_settled"


def model_view(m):
    return {k: m.get(k) for k in ("provider", "id", "name", "contextWindow", "maxTokens", "reasoning", "compat")} if m else None


report = {"settings_sha_start": sha(f"{run}/agent/settings.json"), "models_sha_start": sha(f"{run}/agent/models.json")}
st = send({"id": "s1", "type": "get_state"}, response("s1"))
report["state_before"] = {"model": model_view(st["data"].get("model")), "thinkingLevel": st["data"].get("thinkingLevel")}
send({"id": "p1", "type": "prompt", "message": "First prompt."}, settled)
install(models_after)
report["models_sha_after_rewrite"] = sha(f"{run}/agent/models.json")
av = send({"id": "a1", "type": "get_available_models"}, response("a1"))
report["available_after_rewrite"] = [model_view(m) for m in av["data"]["models"]]
send({"id": "p2", "type": "prompt", "message": "Second prompt, after the rewrite."}, settled)
send({"id": "n1", "type": "new_session"}, response("n1"))
send({"id": "p3", "type": "prompt", "message": "Third prompt, new session."}, settled)
sm = send({"id": "m1", "type": "set_model", "provider": "llmctl", "modelId": "local"}, response("m1"))
report["set_model_response"] = {"success": sm.get("success"), "model": model_view(sm.get("data")), "error": sm.get("error")}
send({"id": "t1", "type": "set_thinking_level", "level": "medium"}, response("t1"))
send({"id": "p4", "type": "prompt", "message": "Fourth prompt, after set_model and set_thinking_level medium."}, settled)
st = send({"id": "s2", "type": "get_state"}, response("s2"))
report["state_end"] = {"model": model_view(st["data"].get("model")), "thinkingLevel": st["data"].get("thinkingLevel")}
proc.stdin.close()
try:
    report["exit_code"] = proc.wait(timeout=20)
except subprocess.TimeoutExpired:
    proc.kill()
    report["exit_code"] = "killed after stdin close"
report["settings_sha_end"] = sha(f"{run}/agent/settings.json")
report["settings_end"] = open(f"{run}/agent/settings.json").read()
caps = sorted(os.listdir(f"{R}/captures/{scenario}"))
report["requests"] = []
for name in caps:
    e = json.load(open(f"{R}/captures/{scenario}/{name}"))
    b = e["body"]
    if isinstance(b, dict) and "messages" in b:
        report["requests"].append({"file": name, "model": b.get("model"), "max_tokens": b.get("max_tokens"),
                                   "max_completion_tokens": b.get("max_completion_tokens"),
                                   "chat_template_kwargs": b.get("chat_template_kwargs"),
                                   "reasoning_effort": b.get("reasoning_effort"),
                                   "user_messages": [m["content"][0]["text"] if isinstance(m["content"], list) else m["content"]
                                                     for m in b["messages"] if m["role"] == "user"]})
json.dump(report, open(f"{run}/report.json", "w"), indent=1)
print(json.dumps(report, indent=1))
