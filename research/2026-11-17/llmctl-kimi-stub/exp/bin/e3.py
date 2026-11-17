"""E3 driver: a chat-only recipe through Kimi with and without an agent file that allows no tools.

The stub plays vLLM without a tool parser: a body with `tools` gets HTTP 400 with vLLM's message, a body without
gets a plain reply. usage: python3 -I e3.py
"""

import json
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "bin"))
import e2  # noqa: E402

VLLM_400 = {"error": {"message": "\"auto\" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set",
                      "type": "BadRequestError", "param": None, "code": 400}}
TASK = "Reply with exactly OK."


def case(name, row, agent, summary):
    text = e2.default_line(row) + (EXP / "entries" / row["file"]).read_text()
    home, cfg = e2.fresh_home(name, text)
    stub = e2.stub_config(name, [], {"type": "error_if_tools", "status": 400, "body": VLLM_400,
                                      "otherwise": {"type": "text", "text": "OK"}})
    args = ["kimi", "-p", TASK, "--output-format", "stream-json", "-m", row["alias"]]
    if agent:
        args += ["--agent-file", str(EXP / "agents" / agent)]
    import os
    os.environ["EXTRA_RO"] = str(EXP / "agents")
    res = e2.run_case(name, home, args, stubcfg=stub, bport=row["port"], to=300)
    recs = e2.records(name)
    summary.append({"case": name, "agent": agent, "exit": res["exit"], "seconds": res["seconds"],
                    "stream": e2.stream_check(name),
                    "requests": [{**e2.shape(r), "system_chars": next((m.get("content_chars") for m in r["request"]["messages"]
                                                                       if m.get("role") == "system"), None),
                                  "tools_present": r["request"].get("tools_present")} for r in recs if r.get("method") == "POST"],
                    "stderr_tail": (EXP / "runs" / name / "stderr").read_text()[-700:],
                    "config_unchanged": cfg.read_text() == text})
    print(name, res["exit"], res["seconds"], flush=True)


def main():
    summary = []
    vllm = e2.entry("vllm", "deepseek-r1-0528-qwen3-8b")
    llama = e2.entry("llamacpp", "deepseek-r1-0528-qwen3-8b")
    case("e3-vllm-deepseek-noagent", vllm, None, summary)
    case("e3-vllm-deepseek-agent", vllm, "chat-only.md", summary)
    case("e3-vllm-deepseek-agent-base", vllm, "chat-only-base.md", summary)
    case("e3-vllm-deepseek-agent-string", vllm, "chat-only-string.md", summary)
    case("e3-llamacpp-deepseek-agent", llama, "chat-only.md", summary)
    (EXP / "runs" / "e3-summary.jsonl").write_text("".join(json.dumps(s) + "\n" for s in summary))


if __name__ == "__main__":
    main()
