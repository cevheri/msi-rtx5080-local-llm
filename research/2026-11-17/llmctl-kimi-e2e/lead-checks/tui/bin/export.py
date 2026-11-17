"""Copy one run into the records: steps, transcript, config copies, server log lines, and the proxy's request
shapes reduced to keys and the values Kimi sets (no message text, no system prompt, no session ids).
Every text gets $HOME, $SCRATCH, <user> and <host> in place of the real values.
usage: python3 -I export.py <run dir> <out dir>"""
import getpass, json, os, re, socket, sys
from pathlib import Path
run, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
SCRATCH = "$SCRATCH"
SUBS = [(SCRATCH, "$SCRATCH"), (os.path.expanduser("~"), "$HOME"), (getpass.getuser(), "<user>"),
        (socket.gethostname(), "<host>")]
SESSION = re.compile(r"session_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
def clean(text):
    for old, new in SUBS:
        text = text.replace(old, new)
    # the TUI shortens the work folder to its last parts
    text = re.sub(r"…/scratchpad/kimi-tui-[A-Za-z0-9]+/work", "…/$W/work", text)
    text = re.sub(r"kimi-tui-[A-Za-z0-9]{6}", "kimi-tui-XXXXXX", text)
    return SESSION.sub("session_<id>", text)
for name in sorted(os.listdir(run)):
    p = run / name
    if name.endswith(".toml") or name in ("tui-transcript.txt", "tui-steps.jsonl", "exit", "start-utc", "end-utc",
                                          "up-start-utc", "up-rc", "prep.txt") or name.startswith("server-log-"):
        (out / name).write_text(clean(p.read_text(encoding="utf-8", errors="replace")), encoding="utf-8")
shapes = []
for p in sorted((run / "shapes").glob("*.json")):
    d = json.loads(p.read_text())
    r = d.get("request", {})
    shape = {"file": p.name, "time": d.get("time"), "path": d.get("path"), "status": d.get("status"),
             "user_agent": d.get("headers", {}).get("User-Agent"),
             "keys": r.get("keys")}
    for k, v in r.items():
        if k in ("keys", "messages", "tool_names"):
            continue
        shape[k] = "<session id>" if k == "prompt_cache_key" else v
    shape["roles"] = [m.get("role") for m in r.get("messages", [])]
    shape["response"] = d.get("response")
    if "error" in d:
        shape["error"] = d["error"]
    shapes.append(shape)
(out / "request-shapes.json").write_text(clean(json.dumps(shapes, indent=1)) + "\n", encoding="utf-8")
print(out, len(shapes), "requests")
