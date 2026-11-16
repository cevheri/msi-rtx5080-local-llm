"""Check the proposed server-args consistency rules against every shipped recipe and the proposed metadata."""
import json, sys, tomllib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from gen import META
repo = Path(sys.argv[1])

def value(args, flag):
    for i, a in enumerate(args):
        if a == flag and i + 1 < len(args):
            return args[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None

problems = []
for (runtime, rid), (output, thinking, default, efforts, medium) in META.items():
    r = tomllib.loads((repo / "recipes" / runtime / f"{rid}.toml").read_text())
    args = r["server"]["args"]; chat_only = r.get("chat_only", False)
    def bad(msg): problems.append(f"{runtime}/{rid}: {msg}")
    if output > r["server"]["context"] // 2: bad("output above context/2")
    if runtime == "llamacpp":
        reasoning = value(args, "--reasoning"); effort = value(args, "--reasoning-effort")
        kwargs = json.loads(value(args, "--chat-template-kwargs") or "{}")
        if thinking == "toggle" and (reasoning == "off") != (default == "off"): bad(f"toggle default {default} but --reasoning {reasoning}")
        if thinking == "effort" and effort not in (None, "medium"): bad(f"effort default must be medium, args say {effort}")
        if thinking == "none" and (reasoning or effort or value(args, "--reasoning-format") or kwargs): bad("thinking none but reasoning flags present")
        if thinking == "always" and not (effort == "high" or kwargs.get("reasoning_effort") == "high" or value(args, "--reasoning-format") == "deepseek"): bad("always: no server-side high effort or deepseek format")
    else:
        parser = value(args, "--reasoning-parser"); kwargs = json.loads(value(args, "--default-chat-template-kwargs") or "{}")
        if (thinking == "none") != (parser is None): bad(f"thinking {thinking} vs reasoning parser {parser}")
        if thinking == "toggle" and "enable_thinking" in kwargs and (kwargs["enable_thinking"] is False) != (default == "off"): bad("toggle default vs kwargs")
        if not chat_only and not ("--enable-auto-tool-choice" in args and value(args, "--tool-call-parser")): bad("tools without a tool parser")
        if chat_only and "--enable-auto-tool-choice" in args: bad("chat only but tool choice enabled")
print("\n".join(problems) or f"all {len(META)} recipes satisfy the proposed rules")
