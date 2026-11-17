"""E2 driver: for each generated entry, kimi provider list --json, then kimi -p against the recording stub.

usage: python3 -I e2.py [plain|reasoning|thinking|all] [case filter substring]
Every kimi command runs through `heavy bash case.sh ...` (offline bwrap sandbox, private home and cache).
Writes runs/<case>/ per case and runs/e2-summary.jsonl.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
INDEX = json.loads((EXP / "entries" / "index.json").read_text())
CANNED = "The user wants hello.txt with hello in it, so I will call Write once and then answer."
PLAIN_TASK = "Reply with exactly OK."
TOOL_TASK = "Create hello.txt in the current folder containing the word hello, then reply with Done."
SNAKE = {"modelSource": "model_source", "baseUrl": "base_url", "apiKey": "api_key", "maxContextSize": "max_context_size",
         "maxOutputSize": "max_output_size", "displayName": "display_name", "supportEfforts": "support_efforts",
         "defaultEffort": "default_effort", "offEffort": "off_effort"}

# one recipe of each kind for the reasoning-plus-tool replies (brief E2)
REASONING = [("llamacpp", "qwen3.6-35b-a3b", "toggle, off by default"),
             ("vllm", "mellum2.1-12b", "toggle, on by default"),
             ("llamacpp", "qwen3.8-27b", "effort with off_effort"),
             ("llamacpp", "gpt-oss-20b", "effort without off_effort"),
             ("vllm", "k2-horizon-7b", "always"),
             ("llamacpp", "qwen3-coder-next", "none"),
             ("vllm", "deepseek-r1-0528-qwen3-8b", "always, chat-only")]
# vLLM streams its reasoning as `reasoning`: the vLLM rows again with that field
REASONING_VLLM_FIELD = [("vllm", "mellum2.1-12b"), ("vllm", "k2-horizon-7b"), ("vllm", "deepseek-r1-0528-qwen3-8b"),
                        ("vllm", "gpt-oss-20b")]
# thinking switches for the effort entries ([thinking] is the only switch print mode reads)
THINKING = {"default": "", "off": "[thinking]\nenabled = false\n\n", "low": "[thinking]\neffort = \"low\"\n\n",
            "on": "[thinking]\nenabled = true\n\n"}


def default_line(row):
    """print mode checks default_model even with -m (main.cjs runNativeTurn -> ensureReady()), so set it as --set-default would."""
    return f'default_model = "{row['alias']}"\n\n'


def entry(runtime, rid):
    for row in INDEX:
        if row["runtime"] == runtime and row["id"] == rid:
            return row
    raise KeyError((runtime, rid))


def run_case(case, home, args, stubcfg=None, bport=0, strace=1, to=180):
    env = dict(os.environ, HOMEDIR=str(home), STRACE=str(strace), TO=str(to), BPORT=str(bport))
    env.pop("STUBCFG", None)
    if stubcfg is not None:
        env["STUBCFG"] = str(stubcfg)
    out = subprocess.run(["heavy", "bash", str(EXP / "bin" / "case.sh"), case, *args], env=env,
                         capture_output=True, text=True)
    run = EXP / "runs" / case
    t0, t1 = map(float, (run / "time").read_text().split()) if (run / "time").exists() else (0.0, 0.0)
    exit_file = run / "exit"
    return {"case": case, "exit": int(exit_file.read_text()) if exit_file.exists() else None, "seconds": round(t1 - t0, 2),
            "heavy": out.stdout.strip()[-200:]}


def fresh_home(case, config_text):
    home = EXP / "homes" / case
    subprocess.run(["rm", "-rf", str(home)], check=True)
    (home / ".kimi-code").mkdir(parents=True)
    (home / ".cache").mkdir()
    cfg = home / ".kimi-code" / "config.toml"
    cfg.write_text(config_text, encoding="utf-8")
    cfg.chmod(0o600)
    return home, cfg


def check_list(row, stdout):
    """Whether provider list --json shows the provider and alias with the intended values (camelCase -> snake)."""
    try:
        doc = json.loads(stdout)
    except ValueError as exc:
        return False, f"not JSON: {exc}"
    provider = doc.get("providers", {}).get("llmctl")
    model = doc.get("models", {}).get(row["alias"])
    if provider is None or model is None:
        return False, "provider or alias missing"
    got_p = {SNAKE.get(k, k): v for k, v in provider.items()}
    got_m = {SNAKE.get(k, k): v for k, v in model.items()}
    problems = [f"provider {k}: {got_p.get(k)!r} != {v!r}" for k, v in row["provider"].items() if got_p.get(k) != v]
    problems += [f"model {k}: {got_m.get(k)!r} != {v!r}" for k, v in row["model"].items() if got_m.get(k) != v]
    extra = sorted(set(got_m) - set(row["model"])) + sorted(set(got_p) - set(row["provider"]))
    return not problems, "; ".join(problems) or ("extra keys " + ",".join(extra) if extra else "ok")


def records(case):
    path = EXP / "runs" / case / "records.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stream_check(case):
    lines = (EXP / "runs" / case / "stdout").read_text().splitlines()
    objs, bad = [], 0
    for line in lines:
        try:
            objs.append(json.loads(line))
        except ValueError:
            bad += 1
    finals = [o for o in objs if o.get("role") == "assistant" and o.get("content")]
    calls = [c["function"]["name"] for o in objs if o.get("role") == "assistant" for c in o.get("tool_calls") or []]
    metas = [o.get("type") for o in objs if o.get("role") == "meta"]
    return {"lines": len(lines), "non_json": bad, "final": finals[-1]["content"] if finals else None,
            "tool_calls": calls, "tool_results": sum(1 for o in objs if o.get("role") == "tool"), "meta": metas}


def shape(rec):
    req = rec.get("request", {})
    assistant = [m for m in req.get("messages", []) if m.get("role") == "assistant"]
    return {"n": rec.get("n"), "keys": req.get("keys"), "model": req.get("model"), "max_tokens": req.get("max_tokens"),
            "reasoning_effort": req.get("reasoning_effort", "<absent>"), "tools_count": req.get("tools_count"),
            "stream_options": req.get("stream_options"), "prompt_cache_key": "prompt_cache_key" in (req.get("keys") or []),
            "roles": req.get("roles"),
            "assistant_messages": [{k: v for k, v in m.items() if k != "role"} for m in assistant],
            "x_msh_headers": rec.get("x_msh_headers"), "authorization": rec.get("authorization"),
            "user_agent": rec.get("user_agent")}


def stub_config(case, rules, default):
    cfg = EXP / "runs" / f"{case}.stub.json"
    (EXP / "runs").mkdir(exist_ok=True)
    cfg.write_text(json.dumps({"record": str(EXP / "runs" / case / "records.jsonl"), "canned_reasoning": CANNED,
                               "rules": rules, "default": default}), encoding="utf-8")
    return cfg


def plain(row, summary):
    case = f"e2-{row['runtime']}-{row['id']}"
    text = default_line(row) + (EXP / "entries" / row["file"]).read_text()
    home, cfg = fresh_home(case, text)
    lst = run_case(case + "-list", home, ["kimi", "provider", "list", "--json"], strace=0)
    ok, detail = check_list(row, (EXP / "runs" / (case + "-list") / "stdout").read_text())
    stub = stub_config(case + "-p", [], {"type": "text", "text": "OK"})
    res = run_case(case + "-p", home, ["kimi", "-p", PLAIN_TASK, "--output-format", "stream-json", "-m", row["alias"]],
                   stubcfg=stub, bport=row["port"])
    recs = records(case + "-p")
    summary.append({"lane": "plain", "case": case, "runtime": row["runtime"], "id": row["id"], "alias": row["alias"],
                    "list_exit": lst["exit"], "accepted": ok and lst["exit"] == 0, "list_detail": detail,
                    "exit": res["exit"], "seconds": res["seconds"], "stream": stream_check(case + "-p"),
                    "requests": [shape(r) for r in recs if r.get("method") == "POST"],
                    "other_requests": [r for r in recs if r.get("method") != "POST"],
                    "config_unchanged": cfg.read_text() == text})


def reasoning(row, label, field, summary, thinking_key=None):
    suffix = "" if field == "reasoning_content" else "-" + field
    case = f"e2r-{row['runtime']}-{row['id']}{suffix}"
    prefix = ""
    if thinking_key is not None:
        case = f"e2t-{row['runtime']}-{row['id']}-{thinking_key}"
        prefix = THINKING[thinking_key]
    text = default_line(row) + prefix + (EXP / "entries" / row["file"]).read_text()
    home, cfg = fresh_home(case, text)
    stub = stub_config(case, [{"nth": 1, "action": {"type": "tool", "name": "Write",
                                                     "arguments": {"path": "hello.txt", "content": "hello\n"},
                                                     "reasoning": CANNED, "field": field}}],
                       {"type": "text", "text": "Done.", "reasoning": "Written.", "field": field})
    res = run_case(case, home, ["kimi", "-p", TOOL_TASK, "--output-format", "stream-json", "-m", row["alias"]],
                   stubcfg=stub, bport=row["port"])
    recs = records(case)
    work = EXP / "work" / case / "hello.txt"
    summary.append({"lane": "thinking" if thinking_key else "reasoning", "case": case, "kind": label,
                    "runtime": row["runtime"], "id": row["id"], "alias": row["alias"], "field": field,
                    "thinking_table": prefix.strip() or None, "exit": res["exit"], "seconds": res["seconds"],
                    "stream": stream_check(case), "requests": [shape(r) for r in recs if r.get("method") == "POST"],
                    "other_requests": [r for r in recs if r.get("method") != "POST"],
                    "hello_txt": work.read_text() if work.exists() else None,
                    "stderr_tail": (EXP / "runs" / case / "stderr").read_text()[-600:],
                    "config_unchanged": cfg.read_text() == text})


def main():
    lane = sys.argv[1] if len(sys.argv) > 1 else "all"
    filt = sys.argv[2] if len(sys.argv) > 2 else ""
    summary = []
    if lane in ("plain", "all"):
        for row in INDEX:
            if filt in f"{row['runtime']}-{row['id']}":
                plain(row, summary)
                print(summary[-1]["case"], summary[-1]["accepted"], summary[-1]["exit"], flush=True)
    if lane in ("reasoning", "all"):
        for runtime, rid, label in REASONING:
            if filt in f"{runtime}-{rid}":
                reasoning(entry(runtime, rid), label, "reasoning_content", summary)
                print(summary[-1]["case"], summary[-1]["exit"], flush=True)
        for runtime, rid in REASONING_VLLM_FIELD:
            if filt in f"{runtime}-{rid}":
                reasoning(entry(runtime, rid), "vLLM field", "reasoning", summary)
                print(summary[-1]["case"], summary[-1]["exit"], flush=True)
    if lane in ("thinking", "all"):
        for runtime, rid in (("llamacpp", "qwen3.8-27b"), ("llamacpp", "gpt-oss-20b"), ("vllm", "gpt-oss-20b")):
            for key in ("off", "low", "on"):
                if filt in f"{runtime}-{rid}":
                    reasoning(entry(runtime, rid), "effort", "reasoning_content", summary, thinking_key=key)
                    print(summary[-1]["case"], summary[-1]["exit"], flush=True)
        for runtime, rid in (("llamacpp", "qwen3.6-35b-a3b"),):
            for key in ("on", "off"):
                if filt in f"{runtime}-{rid}":
                    reasoning(entry(runtime, rid), "toggle", "reasoning_content", summary, thinking_key=key)
                    print(summary[-1]["case"], summary[-1]["exit"], flush=True)
    out = EXP / "runs" / f"e2-summary-{lane}{('-' + filt) if filt else ''}.jsonl"
    out.write_text("".join(json.dumps(s) + "\n" for s in summary))
    print("wrote", out)


if __name__ == "__main__":
    main()
