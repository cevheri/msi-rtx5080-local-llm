"""Measurement D sequence (measurement C's seq_c.py with a flushready phase added) for one running server:
  flushready right after ready, the reading D8's post-ready check makes: nvidia-smi, POST /flush_cache, nvidia-smi at
            once and again 2 s later (added in measurement D; the near-full prompt that follows then comes after this
            flush, as the first request after an llmctl start does)
  nearfull  a fresh prompt at about 95% of the context (at least 90%, no /flush_cache before it), sized with /v1/tokenize
  grammar   Hermes' title request (json_schema, strict, reasoning_effort none), then a forced tool_choice request
  tokcheck  token counts of three fixed messages against the llama.cpp record, and byte-level markers in replies
  probe     /flush_cache, then the repository probe research/2026-11-16/llm_probe.py (decode, 8k and 30k first token)
  smoke     the hands-on api_probe.py with --skip-long (plain, streamed, tool call and second turn, thinking, json_schema)
  sustained the hands-on sustained.py for --sustained-s seconds
  nearfull2 a second fresh near-full prompt after the sustained load (M12)
  m13       twenty short arithmetic, code and tool prompts at temperature 0 (FP8 against BF16 KV comparison)
Before and after every request it records nvidia-smi and the container's cgroup memory; when a request fails or the
server stops answering it watches the container and stops. Standard library only; run with python3 -I:
  seq_d.py --model NAME --context N --thinking toggle|effort|always|none --phases flushready,nearfull,grammar,... --out-dir DIR --run TAG
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mcommon import TOOL, Records, chat_summary, container_state, gpu, http, preview  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PARAGRAPH = (
    "def process_batch(items, config):\n"
    "    # Validate the configuration, normalise every record and collect per-key statistics.\n"
    "    results = {}\n"
    "    for index, item in enumerate(items):\n"
    "        key = f'{item.category}-{index % 17}'\n"
    "        value = item.weight * config.scale + config.offset\n"
    "        results.setdefault(key, []).append(round(value, 3))\n"
    "    return {k: sum(v) / len(v) for k, v in results.items()}\n\n"
)
TAIL = "\nIn one sentence, what does the function above compute?"
TITLE_SCHEMA = {"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"],
                "additionalProperties": False}
TOOLS_M13 = [
    TOOL,
    {"type": "function", "function": {
        "name": "convert_currency", "description": "Convert an amount of money from one currency to another",
        "parameters": {"type": "object", "properties": {
            "amount": {"type": "number"}, "from_currency": {"type": "string", "description": "ISO 4217 code"},
            "to_currency": {"type": "string", "description": "ISO 4217 code"}},
            "required": ["amount", "from_currency", "to_currency"]}}},
    {"type": "function", "function": {
        "name": "get_time", "description": "Get the current local time in a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
]
M13 = [
    ("arith", "What is 17 * 23? Answer with the number only.", "391"),
    ("arith", "What is 1234 + 5678? Answer with the number only.", "6912"),
    ("arith", "What is 144 / 12? Answer with the number only.", "12"),
    ("arith", "What is 2 to the power of 10? Answer with the number only.", "1024"),
    ("arith", "What is 999 - 457? Answer with the number only.", "542"),
    ("arith", "What is 15% of 240? Answer with the number only.", "36"),
    ("arith", "What is the sum of the integers from 1 to 20? Answer with the number only.", "210"),
    ("arith", "What is 37 * 41? Answer with the number only.", "1517"),
    ("code", "What does this Python code print? print(sorted([5, 2, 9, 1])[::-1]) Answer with the output only.", "[9, 5, 2, 1]"),
    ("code", "What does this Python code print? print(len('hello world'.split())) Answer with the output only.", "2"),
    ("code", "What does this Python code print? print({'a': 1, 'b': 2}.get('c', 3)) Answer with the output only.", "3"),
    ("code", "What does this Python code print? print(sum(x * x for x in range(4))) Answer with the output only.", "14"),
    ("code", "What does this Python code print? print('abc'.upper() + str(2 + 3)) Answer with the output only.", "ABC5"),
    ("code", "What does this Python code print? print([i for i in range(10) if i % 3 == 0]) Answer with the output only.", "[0, 3, 6, 9]"),
    ("tool", "What is the weather in Paris? Use the tool.", ("get_weather", {"city": "paris"})),
    ("tool", "Convert 100 US dollars to euros. Use the tool.", ("convert_currency", {"amount": 100, "from_currency": "usd", "to_currency": "eur"})),
    ("tool", "What time is it in Tokyo? Use the tool.", ("get_time", {"city": "tokyo"})),
    ("tool", "What is the weather in Berlin? Use the tool.", ("get_weather", {"city": "berlin"})),
    ("tool", "Convert 250 euros to Japanese yen. Use the tool.", ("convert_currency", {"amount": 250, "from_currency": "eur", "to_currency": "jpy"})),
    ("tool", "What time is it in London? Use the tool.", ("get_time", {"city": "london"})),
]
LLAMACPP_COUNTS = {"ok": 8, "probe_8k": 8078, "probe_30k": 30254}  # research/2026-11-16/llamacpp-deepseek-r1-0528-qwen3-8b/


def cgroup(container):
    out = subprocess.run(["docker", "inspect", "--format", "{{.Id}}", container], capture_output=True, text=True, timeout=20)
    base = f"/sys/fs/cgroup/system.slice/docker-{out.stdout.strip()}.scope"
    res = {}
    try:
        with open(f"{base}/memory.stat") as f:
            for line in f:
                key, value = line.split()
                if key in ("anon", "file", "shmem", "unevictable"):
                    res[key + "_gib"] = round(int(value) / 2**30, 3)
        with open(f"{base}/memory.peak") as f:
            res["peak_gib"] = round(int(f.read()) / 2**30, 3)
    except OSError as err:
        res["error"] = str(err)[:120]
    return res


def normalise(text):
    text = (text or "").strip().strip("`").strip()
    text = re.sub(r"^(python|text|output)\s*\n", "", text).strip()
    return text


def last_number(text):
    nums = re.findall(r"-?\d+(?:\.\d+)?", (text or "").replace(",", ""))
    if not nums:
        return None
    value = float(nums[-1])
    return str(int(value)) if value == int(value) else str(value)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:30000")
    ap.add_argument("--kind", choices=["sglang", "vllm", "llamacpp"], default="sglang")
    ap.add_argument("--container", default="sglang-measure")
    ap.add_argument("--model", required=True)
    ap.add_argument("--context", type=int, required=True)
    ap.add_argument("--thinking", choices=["toggle", "effort", "always", "none"], required=True)
    ap.add_argument("--chat-only", action="store_true")
    ap.add_argument("--phases", required=True)
    ap.add_argument("--fill", type=float, default=0.95, help="share of the context for the near-full prompt")
    ap.add_argument("--grammar-max-tokens", type=int, default=2048)
    ap.add_argument("--extra", default="{}", help="JSON merged into the probe, api_probe and sustained requests")
    ap.add_argument("--sustained-s", type=int, default=300)
    ap.add_argument("--m13-max-tokens", type=int, default=4096)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    rec = Records(os.path.join(a.out_dir, f"{a.run}-seq.jsonl"))
    base, model = a.base, a.model
    tok_path = "/v1/tokenize" if a.kind == "sglang" else "/tokenize"
    extra = json.loads(a.extra)
    info = {}
    if a.kind == "sglang":
        head, body = http(base, "/server_info", timeout=30)
        if isinstance(body, dict):
            info = (body.get("internal_states") or [{}])[0].get("memory_usage") or {}
    capacity = info.get("token_capacity")

    def alive():
        head, _ = http(base, "/v1/models", timeout=10)
        return head["status"] == 200

    def watch_death(reason):
        rec.add(event="death_watch_start", reason=reason, container=container_state(a.container))
        t_start = time.time()
        while time.time() - t_start < 60:
            st = container_state(a.container)
            rec.add(event="death_watch", after_s=round(time.time() - t_start, 1), container=st)
            if not st.startswith("running"):
                break
            time.sleep(3)

    def step(name, payload, timeout=3600):
        rec.add(event="gpu_before", step=name, **gpu())
        head, body = http(base, "/v1/chat/completions", payload, timeout=timeout)
        summary = chat_summary(head, body)
        if name.startswith("title") and isinstance(body, dict) and body.get("choices"):
            content = body["choices"][0]["message"].get("content") or ""
            try:
                summary["title_parses"] = isinstance(json.loads(content).get("title"), str)
            except (ValueError, AttributeError):
                summary["title_parses"] = False
        rec.add(event="request", step=name, **summary)
        rec.add(event="gpu_after", step=name, cgroup=cgroup(a.container), **gpu())
        if head["status"] != 200 or not alive():
            watch_death(f"{name}: status {head['status']} {head.get('error') or ''}")
            return False, body
        return True, body

    def count(messages):
        head, body = http(base, tok_path, {"model": model, "messages": messages}, timeout=120)
        if isinstance(body, dict) and "count" in body:
            return body["count"]
        raise RuntimeError(f"tokenize failed: {head} {preview(json.dumps(body), 300)}")

    def nearfull(tag):
        session = f"Session {uuid.uuid4().hex}. Read the code below.\n\n"
        msg = lambda n: [{"role": "user", "content": session + PARAGRAPH * n + TAIL}]  # noqa: E731
        c100, c200 = count(msg(100)), count(msg(200))
        per = (c200 - c100) / 100.0
        overhead = c100 - 100 * per
        target = int(a.fill * a.context)
        cap = a.context - 96
        if capacity:
            cap = min(cap, int(capacity) - 96)
        target = min(target, cap)
        n = int((target - overhead) / per)
        total = count(msg(n))
        while total > cap:
            n -= max(1, int((total - cap) / per) + 1)
            total = count(msg(n))
        rec.add(event="nearfull_plan", step=tag, per_paragraph=round(per, 2), overhead=round(overhead, 1), paragraphs=n,
                prompt_tokens=total, context=a.context, token_capacity=capacity, share_of_context=round(total / a.context, 4))
        payload = {"model": model, "messages": msg(n), "max_tokens": 32}
        ok, _ = step(f"nearfull_{tag}", payload)
        return ok

    def grammar():
        title = {"model": model, "messages": [
            {"role": "system", "content": "You write short titles for chat sessions. Reply with JSON only."},
            {"role": "user", "content": "Write a short title for a chat about adding two numbers in Python."}],
            "reasoning_effort": "none",
            "response_format": {"type": "json_schema", "json_schema": {"name": "session_title", "strict": True,
                                                                       "schema": TITLE_SCHEMA}},
            "max_tokens": a.grammar_max_tokens, "stream": False}
        ok, _ = step("title_first_grammar", title, timeout=900)
        if not ok:
            return False
        if a.chat_only:
            rec.add(event="skip", step="forced_tool", reason="chat-only recipe")
            return True
        forced = {"model": model, "messages": [{"role": "user", "content": "What is the weather in Paris? Use the tool."}],
                  "tools": [TOOL], "tool_choice": "required", "max_tokens": a.grammar_max_tokens}
        ok, _ = step("forced_tool", forced, timeout=900)
        return ok

    def tokcheck():
        msgs = {"ok": "Reply with exactly: OK", "probe_8k": PARAGRAPH * (8000 // 95) + TAIL,
                "probe_30k": PARAGRAPH * (30000 // 95) + TAIL}
        counts = {k: count([{"role": "user", "content": v}]) for k, v in msgs.items()}
        raw = {}
        for k, v in msgs.items():
            head, body = http(base, tok_path, {"model": model, "prompt": v, "add_special_tokens": False}, timeout=120)
            raw[k] = body.get("count") if isinstance(body, dict) else f"error {head['status']}"
        rec.add(event="tokcheck_counts", chat_counts=counts, llamacpp_chat_counts=LLAMACPP_COUNTS,
                chat_diff={k: counts[k] - LLAMACPP_COUNTS[k] for k in counts},
                raw_text_counts=raw, llamacpp_text_counts_implied={k: v - 3 for k, v in LLAMACPP_COUNTS.items()},
                note="llama.cpp's chat count is BOS + <|User|> + text + <|Assistant|>, so its text count is the chat count minus 3")
        for label, text in (("say_hello", "Say hello world."), ("sum", "What is 17 + 25? Answer briefly.")):
            ok, body = step(f"tokcheck_{label}", {"model": model, "messages": [{"role": "user", "content": text}],
                                                  "max_tokens": 2048})
            if isinstance(body, dict) and body.get("choices"):
                msg = body["choices"][0]["message"]
                content = msg.get("content") or ""
                reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
                rec.add(event="tokcheck_reply", step=label, content_head=content[:200], reasoning_head=reasoning[:160],
                        content_has_byte_markers=("Ġ" in content or "Ċ" in content),
                        reasoning_has_byte_markers=("Ġ" in reasoning or "Ċ" in reasoning),
                        reasoning_field=[k for k in ("reasoning_content", "reasoning") if msg.get(k)])
            if not ok:
                return False
        tools_req = {"model": model, "messages": [{"role": "user", "content": "What is the weather in Paris? Use the tool."}],
                     "tools": [TOOL], "max_tokens": 2048}
        rec.add(event="gpu_before", step="m9_tools", **gpu())
        head, body = http(base, "/v1/chat/completions", tools_req, timeout=900)
        summary = chat_summary(head, body)
        if isinstance(body, dict) and body.get("choices"):
            summary["content_full_head"] = (body["choices"][0]["message"].get("content") or "")[:400]
        rec.add(event="request", step="m9_tools", **summary)
        return alive()

    def subprocess_step(name, argv, timeout):
        rec.add(event="phase_start", step=name, argv=[x if len(x) < 200 else x[:200] for x in argv], **gpu())
        t = time.time()
        try:
            out = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
            rc, tail = out.returncode, (out.stdout[-3000:] + "\n" + out.stderr[-1500:]).strip()
        except subprocess.TimeoutExpired:
            rc, tail = "timeout", ""
        rec.add(event="phase_end", step=name, rc=rc, seconds=round(time.time() - t, 1), output_tail=tail[-2500:],
                cgroup=cgroup(a.container), **gpu())
        return alive()

    def m13():
        rows = []
        for i, (kind, prompt, expect) in enumerate(M13):
            payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0,
                       "max_tokens": a.m13_max_tokens}
            if kind == "tool":
                payload["tools"] = TOOLS_M13
            payload.update({k: v for k, v in extra.items() if k not in payload})
            head, body = http(base, "/v1/chat/completions", payload, timeout=900)
            row = {"i": i, "kind": kind, "status": head["status"]}
            if isinstance(body, dict) and body.get("choices"):
                choice = body["choices"][0]
                msg = choice.get("message") or {}
                content = msg.get("content") or ""
                row.update(finish=choice.get("finish_reason"), content=content[:200],
                           reasoning_chars=len(msg.get("reasoning_content") or ""),
                           completion_tokens=(body.get("usage") or {}).get("completion_tokens"))
                if kind == "arith":
                    row["answer"] = last_number(content)
                    row["correct"] = row["answer"] == expect
                elif kind == "code":
                    row["answer"] = normalise(content).splitlines()[-1].strip() if normalise(content) else ""
                    row["correct"] = row["answer"].replace("'", "").replace('"', "") == expect.replace("'", "")
                else:
                    calls = msg.get("tool_calls") or []
                    parsed = []
                    for c in calls:
                        fn = c.get("function") or {}
                        try:
                            args = json.loads(fn.get("arguments") or "{}")
                        except ValueError:
                            args = {"unparsable": fn.get("arguments")}
                        parsed.append({"name": fn.get("name"), "arguments": args})
                    row["answer"] = parsed
                    name, want = expect
                    good = False
                    for c in parsed:
                        if c["name"] != name:
                            continue
                        got = {k: (str(v).lower() if not isinstance(v, (int, float)) else v) for k, v in c["arguments"].items()}
                        good = all((str(got.get(k, "")).lower().find(str(v)) >= 0) if isinstance(v, str)
                                   else (float(got.get(k, -1)) == float(v)) for k, v in want.items())
                        if good:
                            break
                    row["correct"] = good
            rows.append(row)
            rec.add(event="m13", **row)
            if head["status"] != 200 and not alive():
                watch_death(f"m13 {i}: status {head['status']}")
                return False
        n_ok = sum(1 for r in rows if r.get("correct"))
        rec.add(event="m13_summary", correct=n_ok, total=len(rows))
        with open(os.path.join(a.out_dir, f"{a.run}-m13.json"), "w") as f:
            json.dump(rows, f, indent=1)
        return True

    def diag():
        """Tool and plain requests with the thinking switch unset, on and off, temperature 0, each recorded with the
        head of its content and reasoning, its tool calls and finish reason (added in measurement C)."""
        weather = [{"role": "user", "content": "What is the weather in Paris? Use the tool."}]
        capital = [{"role": "user", "content": "What is the capital of France? Answer in one word."}]
        variants = (("default", {}), ("thinking_on", {"chat_template_kwargs": {"enable_thinking": True}}),
                    ("thinking_off", {"chat_template_kwargs": {"enable_thinking": False}}))
        for label, kw in variants:
            for name, msgs, tools in (("tool", weather, [TOOL]), ("capital", capital, None)):
                payload = dict({"model": model, "messages": msgs, "max_tokens": 2048, "temperature": 0}, **kw)
                if tools:
                    payload["tools"] = tools
                head, body = http(base, "/v1/chat/completions", payload, timeout=900)
                row = {"variant": label, "request": name, "status": head["status"], "seconds": head.get("seconds")}
                if isinstance(body, dict) and body.get("choices"):
                    msg = body["choices"][0].get("message") or {}
                    row.update(finish=body["choices"][0].get("finish_reason"), content_head=(msg.get("content") or "")[:400],
                               reasoning_head=(msg.get("reasoning_content") or "")[:200],
                               reasoning_chars=len(msg.get("reasoning_content") or ""),
                               tool_calls=[((c.get("function") or {}).get("name"), (c.get("function") or {}).get("arguments"))
                                           for c in msg.get("tool_calls") or []],
                               usage=body.get("usage"))
                else:
                    row["body"] = preview(json.dumps(body), 300)
                rec.add(event="diag", **row)
        return alive()

    rec.add(event="start", model=model, phases=a.phases, token_capacity=capacity, memory_usage=info,
            container=container_state(a.container), cgroup=cgroup(a.container), **gpu())
    py = [sys.executable, "-I"]
    ok = True
    for phase in a.phases.split(","):
        if not ok:
            break
        if phase == "flushready":
            before = gpu()
            head, _ = http(base, "/flush_cache", {}, timeout=30)
            after = gpu()
            time.sleep(2)
            later = gpu()
            rec.add(event="flush_ready", status=head["status"], seconds=head.get("seconds"),
                    free_mib_before=before["free_mib"], free_mib_after=after["free_mib"],
                    free_mib_after_2s=later["free_mib"], used_mib_after=after["used_mib"], t_after=after["t"],
                    cgroup=cgroup(a.container))
            ok = head["status"] == 200 and alive()
        elif phase == "nearfull":
            ok = nearfull("p1")
        elif phase == "nearfull2":
            ok = nearfull("p2")
        elif phase == "grammar":
            ok = grammar()
        elif phase == "tokcheck":
            ok = tokcheck()
        elif phase == "probe":
            if a.kind == "sglang":
                head, _ = http(base, "/flush_cache", {}, timeout=60)
                rec.add(event="flush_cache", status=head["status"], **gpu())
            script = "llm_probe_chat_only.py" if a.chat_only else "llm_probe.py"
            ok = subprocess_step("probe", py + [os.path.join(HERE, script), "--base", base + "/v1", "--model", model,
                                                "--label", a.run, "--out", os.path.join(a.out_dir, f"{a.run}-probe.json"),
                                                "--extra", json.dumps(extra)], 3600)
        elif phase == "smoke":
            ok = subprocess_step("smoke", py + [os.path.join(HERE, "api_probe.py"), "--base", base, "--model", model,
                                                "--thinking", a.thinking, "--context", str(a.context), "--skip-long",
                                                "--out", os.path.join(a.out_dir, f"{a.run}-api.json")], 3600)
        elif phase == "sustained":
            ok = subprocess_step("sustained", py + [os.path.join(HERE, "sustained.py"), "--base", base, "--model", model,
                                                    "--seconds", str(a.sustained_s), "--extra", json.dumps(extra),
                                                    "--out", os.path.join(a.out_dir, f"{a.run}-sustained.json")],
                                 a.sustained_s + 900)
        elif phase == "m13":
            ok = m13()
        elif phase == "diag":
            ok = diag()
        else:
            raise SystemExit(f"unknown phase {phase}")
    final = {}
    if a.kind == "sglang":
        head, body = http(base, "/server_info", timeout=30)
        if isinstance(body, dict):
            final = (body.get("internal_states") or [{}])[0].get("memory_usage") or {}
    rec.add(event="end", result="completed" if ok else "stopped early", container=container_state(a.container),
            cgroup=cgroup(a.container), memory_usage=final, **gpu())


if __name__ == "__main__":
    main()
