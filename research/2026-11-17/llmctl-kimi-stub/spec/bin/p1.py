"""P1 for the llmctl Kimi design (lead decision L-k): does --agent-file with tools: [] also keep the TUI from sending
tools, and does the shipped agent file behave as E3's in print mode?

usage: python3 -I p1.py   (every kimi command runs through `heavy bash case.sh`, offline bwrap sandbox, private home)
The stub plays vLLM without a tool parser: a body with `tools` gets HTTP 400 with vLLM's message, a body without gets
PONG-ONE. Writes runs/<case>/ and runs/p1-summary.jsonl.
"""

import json
import os
import subprocess
from pathlib import Path

SPEC = Path(__file__).resolve().parents[1]
EXP = SPEC.parent / "exp"
ENTRY = (EXP / "entries" / "vllm" / "deepseek-r1-0528-qwen3-8b.toml").read_text(encoding="utf-8")
ALIAS = "llmctl/deepseek-r1-8b"
DEFAULT = f'default_model = "{ALIAS}"\n\n'
AGENT = SPEC / "agents" / "llmctl-chat-only.md"
VLLM_400 = {"error": {"message": "\"auto\" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set",
                      "type": "BadRequestError", "param": None, "code": 400}}
TUI_ENV = "KIMI_CODE_TRUST_WORKSPACE=1 TERM=xterm-256color"


def home(case, text):
    folder = SPEC / "homes" / case
    subprocess.run(["rm", "-rf", str(folder)], check=True)
    (folder / ".kimi-code").mkdir(parents=True)
    (folder / ".cache").mkdir()
    cfg = folder / ".kimi-code" / "config.toml"
    cfg.write_text(text, encoding="utf-8")
    cfg.chmod(0o600)
    return folder, cfg


def stub(case):
    path = SPEC / "runs" / f"{case}.stub.json"
    path.write_text(json.dumps({"record": str(SPEC / "runs" / case / "records.jsonl"), "canned_reasoning": "",
                                "rules": [], "default": {"type": "error_if_tools", "status": 400, "body": VLLM_400,
                                                         "otherwise": {"type": "text", "text": "PONG-ONE"}}}),
                    encoding="utf-8")
    return path


def run(case, text, args, *, tui):
    folder, cfg = home(case, text)
    env = dict(os.environ, HOMEDIR=str(folder), STRACE="0", TO="120", BPORT="8000", STUBCFG=str(stub(case)),
               EXTRA_RO=str(AGENT.parent))
    if tui:
        env["XENV"] = TUI_ENV
        args = ["python3", "-I", str(SPEC / "bin" / "tui.py"), str(SPEC / "runs" / case),
                str(SPEC / "bin" / "tui-p1.json"), "--", *args]
    out = subprocess.run(["heavy", "bash", str(SPEC / "bin" / "case.sh"), case, *args], env=env,
                         capture_output=True, text=True)
    runs = SPEC / "runs" / case
    recs = [json.loads(line) for line in (runs / "records.jsonl").read_text().splitlines()
            if line.strip()] if (runs / "records.jsonl").exists() else []
    posts = [r for r in recs if r.get("method") == "POST"]
    transcript = (runs / "tui-transcript.txt").read_text(errors="replace") if tui and (runs / "tui-transcript.txt").exists() else ""
    row = {"case": case, "tui": tui, "argv": args[args.index("--") + 1:] if tui else args,
           "exit": (runs / "exit").read_text().strip() if (runs / "exit").exists() else None,
           "heavy": out.stdout.strip()[-200:],
           "requests": [{"keys": r["request"].get("keys"), "tools_count": r["request"].get("tools_count"),
                         "system_chars": next((m.get("content_chars") for m in r["request"].get("messages", [])
                                               if m.get("role") == "system"), None),
                         "status": r.get("status")} for r in posts],
           "answered_pong": "PONG-ONE" in transcript if tui else "PONG-ONE" in (runs / "stdout").read_text(),
           "error_in_tui": ("rror" in transcript.split("<<<MARK start>>>")[-1]) if tui else None,
           "config_unchanged": cfg.read_text(encoding="utf-8") == text}
    print(json.dumps(row), flush=True)
    return row


def main():
    rows = [
        run("p1-tui-agent-m", ENTRY, ["kimi", "-m", ALIAS, "--agent-file", str(AGENT)], tui=True),
        run("p1-tui-agent-default", DEFAULT + ENTRY, ["kimi", "--agent-file", str(AGENT)], tui=True),
        run("p1-tui-noagent-m", ENTRY, ["kimi", "-m", ALIAS], tui=True),
        run("p1-print-agent", DEFAULT + ENTRY, ["kimi", "-p", "Reply with exactly OK.", "--output-format", "stream-json",
                                                "--agent-file", str(AGENT)], tui=False),
    ]
    (SPEC / "runs" / "p1-summary.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


if __name__ == "__main__":
    main()
