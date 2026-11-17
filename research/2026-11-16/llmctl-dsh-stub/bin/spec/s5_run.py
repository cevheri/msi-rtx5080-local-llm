#!/usr/bin/env python3
"""S5: run every generated home patch (port 18090) through `dsh --profile headless --json` against the stub.

Each case gets a fresh DSH_HOME, DSH_AGENTS_HOME and work folder (cwd = PWD), DSH_TELEMETRY_DISABLED=1, and runs
through heavy. Everything it writes goes under $SCRATCH/llmctl-dsh-stub-rerun/ (it refuses when SCRATCH is unset).
The stub (bin/spec/stub.py 18090 $SCRATCH/llmctl-dsh-stub-rerun/out/stub.json) must be running; this script rewrites
its config per case.
Usage: s5_run.py [case ...]   (default: every recipe plus migration-profile-rows; with case names, add the argument
migration next to llamacpp-gemma-4-26b-a4b to record migration-profile-rows)
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

if not os.environ.get("SCRATCH"):
    raise SystemExit("s5_run.py: set SCRATCH to a work folder; records and runs go under $SCRATCH/llmctl-dsh-stub-rerun/")
S = Path(os.environ["SCRATCH"]) / "llmctl-dsh-stub-rerun"
REC = S / "out" / "rec"
RUNS = S / "out" / "runs"
CFG = S / "out" / "stub.json"
PORT = 18090
TASK = "Reply with exactly the word pong."
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_dsh  # noqa: E402


def set_stub(case, served, field):
    record = REC / f"{case}.jsonl"
    if record.exists():
        record.unlink()
    cfg = {"record": str(record), "single_slot": False, "reasoning_field": field, "models": [served],
           "default": {"type": "text", "text": "pong", "reasoning": "Thinking briefly.",
                       "usage": {"prompt_tokens": 1200, "completion_tokens": 2}}, "rules": []}
    tmp = CFG.with_suffix(".tmp")
    tmp.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    os.replace(tmp, CFG)
    return record


def run_case(case, home_text, served, field, profile_patches=None):
    record = set_stub(case, served, field)
    home = Path(tempfile.mkdtemp(prefix=f"h-{case}.", dir=S / "homes"))
    agents = Path(tempfile.mkdtemp(prefix=f"a-{case}.", dir=S / "homes"))
    work = Path(tempfile.mkdtemp(prefix=f"w-{case}.", dir=S / "work"))
    (home / "cordis.patch.yml").write_text(home_text, encoding="utf-8")
    for profile, text in (profile_patches or {}).items():
        # dump-config initializes the profile, then the case's own profile patch replaces the template
        subprocess.run(["heavy", "dsh", "--profile", profile, "--dump-config"], cwd=work, capture_output=True,
                       env=dict(os.environ, DSH_HOME=str(home), DSH_TELEMETRY_DISABLED="1"))
        (home / "profiles" / profile / "cordis.patch.yml").write_text(text, encoding="utf-8")
    env = dict(os.environ, DSH_HOME=str(home), DSH_AGENTS_HOME=str(agents), DSH_TELEMETRY_DISABLED="1",
               PWD=str(work))
    out = RUNS / case
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    done = subprocess.run(["heavy", "dsh", "--profile", "headless", "--json", TASK], cwd=work, env=env,
                          capture_output=True, text=True, timeout=180)
    seconds = time.monotonic() - start
    (out / "stdout.jsonl").write_text(done.stdout, encoding="utf-8")
    (out / "stderr.txt").write_text(done.stderr, encoding="utf-8")
    (out / "home.yml").write_text(home_text, encoding="utf-8")
    time.sleep(0.3)
    requests = []
    if record.exists():
        for raw in record.read_text(encoding="utf-8").splitlines():
            item = json.loads(raw)
            if item.get("method") != "POST":
                continue
            body = item["body"]
            requests.append({
                "kind": item["kind"], "model": body.get("model"), "max_tokens": body.get("max_tokens"),
                "max_completion_tokens": body.get("max_completion_tokens"),
                "chat_template_kwargs": body.get("chat_template_kwargs"),
                "reasoning_effort": body.get("reasoning_effort", "<absent>"), "store": body.get("store", "<absent>"),
                "first_role": (body.get("messages") or [{}])[0].get("role"),
                "tools": len(body.get("tools") or []), "stream": body.get("stream"),
                "stream_options": body.get("stream_options"),
                "authorization_scheme": item["headers"].get("authorization_scheme"),
            })
    final = None
    for raw in done.stdout.splitlines():
        try:
            event = json.loads(raw)
        except ValueError:
            continue
        if event.get("type") == "final":
            final = event.get("text")
    result = {"case": case, "exit": done.returncode, "seconds": round(seconds, 1), "final": final,
              "stderr": done.stderr.strip()[:300], "requests": requests}
    (out / "result.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    return result


def main(argv):
    (S / "homes").mkdir(parents=True, exist_ok=True)
    (S / "work").mkdir(parents=True, exist_ok=True)
    REC.mkdir(parents=True, exist_ok=True)
    wanted = set(argv)
    results = []
    for runtime, recipe_id, target in gen_dsh.targets():
        case = f"{runtime}-{recipe_id}"
        if wanted and case not in wanted:
            continue
        field = "reasoning_content" if runtime == "llamacpp" else "reasoning"
        text = gen_dsh.home_text(target, default=True, port=PORT)
        results.append(run_case(case, text, target.served, field))
        if runtime == "llamacpp" and recipe_id == "gemma-4-26b-a4b" and (not wanted or "migration" in wanted):
            # the owner's hand-made profile rows under llmctl's home block that holds the default
            profile = ("- id: llm-pi-ai\n  config:\n    providers:\n      llmctl:\n        api: openai-completions\n"
                       "        baseURL: http://127.0.0.1:9/v1\n        headers:\n"
                       "          Authorization: Bearer no-key-required\n        models:\n          - id: hidden\n"
                       "- id: agent-default-model\n  config:\n    provider: llmctl\n    model: local\n"
                       "    reasoningEffort: medium\n")
            results.append(run_case("migration-profile-rows", text, target.served, field, {"headless": profile}))
    for result in results:
        main_requests = [r for r in result["requests"] if r["kind"] == "main"]
        title = [r for r in result["requests"] if r["kind"] == "title"]
        first = main_requests[0] if main_requests else {}
        print(json.dumps({"case": result["case"], "exit": result["exit"], "s": result["seconds"],
                          "final": result["final"], "main": {k: first.get(k) for k in (
                              "model", "max_tokens", "chat_template_kwargs", "reasoning_effort", "store", "first_role",
                              "tools", "authorization_scheme")},
                          "title": {k: (title[0].get(k) if title else None) for k in (
                              "max_tokens", "chat_template_kwargs", "reasoning_effort")},
                          "stderr": result["stderr"][:160]}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
