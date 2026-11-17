"""E5 (default_model naming a missing alias, and the -m cases next to it) and E7 (a stopped server).

usage: python3 -I e5e7.py [e5|e7|all]
"""

import json
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "bin"))
import e2  # noqa: E402

TASK = "Reply with exactly OK."


def run(name, config_text, args, summary, stub=True, bport=None, to=180):
    home, cfg = e2.fresh_home(name, config_text)
    row = e2.entry("llamacpp", "gemma-4-26b-a4b")
    stubcfg = e2.stub_config(name, [], {"type": "text", "text": "OK"}) if stub else None
    res = e2.run_case(name, home, ["kimi", "-p", TASK, *args], stubcfg=stubcfg,
                      bport=row["port"] if bport is None else bport, to=to)
    recs = e2.records(name)
    run_dir = EXP / "runs" / name
    summary.append({"case": name, "args": args, "exit": res["exit"], "seconds": res["seconds"],
                    "requests": len([r for r in recs if r.get("method") == "POST"]),
                    "models_sent": [r["request"].get("model") for r in recs if r.get("method") == "POST"],
                    "stdout": (run_dir / "stdout").read_text()[-1500:], "stderr": (run_dir / "stderr").read_text()[-1500:],
                    "config_unchanged": cfg.read_text() == config_text})
    print(name, res["exit"], res["seconds"], flush=True)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    tables = (EXP / "entries" / "llamacpp" / "gemma-4-26b-a4b.toml").read_text()
    summary = []
    if which in ("e5", "all"):
        dangling = 'default_model = "llmctl/missing"\n\n' + tables
        good = 'default_model = "llmctl/local"\n\n' + tables
        sj = ["--output-format", "stream-json"]
        run("e5-dangling-nom-json", dangling, sj, summary)
        run("e5-dangling-nom-text", dangling, ["--output-format", "text"], summary)
        run("e5-dangling-with-m", dangling, [*sj, "-m", "llmctl/local"], summary)
        run("e5-nodefault-with-m", tables, [*sj, "-m", "llmctl/local"], summary)
        run("e5-nodefault-nom", tables, sj, summary)
        run("e5-good-with-missing-m", good, [*sj, "-m", "llmctl/missing"], summary)
    if which in ("e7", "all"):
        good = 'default_model = "llmctl/local"\n\n' + tables
        run("e7-refused-json", good, ["--output-format", "stream-json"], summary, stub=False, bport=0, to=400)
    out = EXP / "runs" / f"e5e7-summary-{which}.jsonl"
    out.write_text("".join(json.dumps(s) + "\n" for s in summary))


if __name__ == "__main__":
    main()
