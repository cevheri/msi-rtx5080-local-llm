"""P2 for the llmctl Kimi design: what Kimi 2.5.3 does in print mode when the owner's global [thinking] effort is
outside an effort entry's support_efforts (docs K074 and K089 disagree), on llmctl's generated entries.

usage: python3 -I p2.py   (every kimi command runs through `heavy bash case.sh`, offline bwrap sandbox, private home)
Writes runs/<case>/ and runs/p2-summary.jsonl.
"""

import json
import os
import subprocess
from pathlib import Path

SPEC = Path(__file__).resolve().parents[1]
EXP = SPEC.parent / "exp"
CASES = [("p2-qwen38-high", "qwen3.8-27b", "high"), ("p2-qwen38-max", "qwen3.8-27b", "max"),
         ("p2-qwen38-minimal", "qwen3.8-27b", "minimal"), ("p2-gptoss-max", "gpt-oss-20b", "max")]


def main():
    rows = []
    for case, rid, effort in CASES:
        entry = (EXP / "entries" / "llamacpp" / f"{rid}.toml").read_text(encoding="utf-8")
        text = 'default_model = "llmctl/local"\n\n' + entry + f'\n[thinking]\neffort = "{effort}"\n'
        folder = SPEC / "homes" / case
        subprocess.run(["rm", "-rf", str(folder)], check=True)
        (folder / ".kimi-code").mkdir(parents=True)
        (folder / ".cache").mkdir()
        cfg = folder / ".kimi-code" / "config.toml"
        cfg.write_text(text, encoding="utf-8")
        cfg.chmod(0o600)
        stub = SPEC / "runs" / f"{case}.stub.json"
        stub.write_text(json.dumps({"record": str(SPEC / "runs" / case / "records.jsonl"), "canned_reasoning": "",
                                    "rules": [], "default": {"type": "text", "text": "OK"}}), encoding="utf-8")
        env = dict(os.environ, HOMEDIR=str(folder), STRACE="0", TO="120", BPORT="8080", STUBCFG=str(stub))
        args = ["kimi", "-p", "Reply with exactly OK.", "--output-format", "stream-json"]
        subprocess.run(["heavy", "bash", str(SPEC / "bin" / "case.sh"), case, *args], env=env, capture_output=True,
                       text=True)
        runs = SPEC / "runs" / case
        recs = [json.loads(line) for line in (runs / "records.jsonl").read_text().splitlines()
                if line.strip()] if (runs / "records.jsonl").exists() else []
        row = {"case": case, "recipe": rid, "thinking_effort": effort,
               "exit": (runs / "exit").read_text().strip(),
               "reasoning_effort": [r["request"].get("reasoning_effort", "<absent>") for r in recs
                                    if r.get("method") == "POST"],
               "stderr": (runs / "stderr").read_text()[-400:], "config_unchanged": cfg.read_text() == text}
        print(json.dumps(row), flush=True)
        rows.append(row)
    (SPEC / "runs" / "p2-summary.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


if __name__ == "__main__":
    main()
