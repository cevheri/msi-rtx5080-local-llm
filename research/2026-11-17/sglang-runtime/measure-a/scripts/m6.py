"""M6 grammar-ordering probe for SGLang v0.5.23: a long fresh prompt, then the first grammar-constrained requests.
Order 1: flush, long prompt, Hermes-shaped title request (json_schema, strict), forced tool_choice request.
Order 2: the same after a tiny json_schema warm-up sent first.
--then-long N repeats flush, a fresh prompt of N paragraphs, the title and the forced tool request (the 120k check).
Before and after each request it records nvidia-smi; when a request fails or the server stops answering it watches the
endpoints and the container (M7, scheduler death) and stops.
Run with python3 -I: m6.py --model NAME --order 1|2 --paragraphs N --max-tokens N --out FILE.jsonl [--then-long N]"""
import argparse
import json
import os
import sys
import time
import urllib.request
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mcommon import OPENER, TOOL, Records, chat_summary, container_state, gpu, http  # noqa: E402

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
TITLE_SCHEMA = {"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"],
                "additionalProperties": False}


def decode_check(base, model):
    """One streamed answer of up to 512 tokens; decode tok/s from the first to the last content chunk."""
    payload = {"model": model, "max_tokens": 512, "stream": True, "stream_options": {"include_usage": True},
               "messages": [{"role": "user", "content": "Write a long essay on the history of computing."}]}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    first = last = None
    usage = None
    t0 = time.time()
    with OPENER.open(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            chunk = json.loads(line[5:])
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("content") or delta.get("reasoning_content"):
                    first = first or time.time()
                    last = time.time()
    tokens = (usage or {}).get("completion_tokens") or 0
    rate = round((tokens - 1) / (last - first), 1) if first and last and last > first and tokens > 1 else None
    return {"completion_tokens": tokens, "ttft_s": round((first or t0) - t0, 3), "decode_tok_s": rate}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:30000")
    ap.add_argument("--container", default="sglang-measure")
    ap.add_argument("--model", required=True)
    ap.add_argument("--order", type=int, choices=[1, 2], required=True)
    ap.add_argument("--paragraphs", type=int, required=True)
    ap.add_argument("--then-long", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, required=True, help="for the title and the forced tool request")
    ap.add_argument("--decode-check", action="store_true", help="at the end, one streamed 512-token answer for decode tok/s")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rec = Records(a.out)
    base, model = a.base, a.model

    def alive():
        head, _ = http(base, "/v1/models", timeout=10)
        return head["status"] == 200

    def watch_death(reason):
        """Scheduler death: what each endpoint answers while the container is up, and whether it exits."""
        rec.add(event="death_watch_start", reason=reason, container=container_state(a.container))
        t_start = time.time()
        while time.time() - t_start < 120:
            st = container_state(a.container)
            row = {"container": st}
            if st.startswith("running"):
                for path in ("/health", "/ready", "/v1/models", "/get_server_info", "/server_info"):
                    head, _ = http(base, path, timeout=8)
                    row[path] = head["status"] if head["status"] is not None else head.get("error")
            rec.add(event="death_watch", after_s=round(time.time() - t_start, 1), **row)
            if not st.startswith("running"):
                break
            time.sleep(2)
        rec.add(event="death_watch_end", container=container_state(a.container))

    def step(name, payload, timeout=1800):
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
        rec.add(event="gpu_after", step=name, **gpu())
        ok = head["status"] == 200
        if not ok or not alive():
            watch_death(f"{name}: status {head['status']} {head.get('error') or ''}")
            return False
        return True

    def long_phase(tag, paragraphs):
        head, _ = http(base, "/flush_cache", {}, timeout=60)
        rec.add(event="flush_cache", step=tag, status=head["status"])
        text = (f"Session {uuid.uuid4().hex}. Read the code below.\n\n" + PARAGRAPH * paragraphs
                + "\nIn one sentence, what does the function above compute?")
        if not step(f"long_prompt_{tag}", {"model": model, "messages": [{"role": "user", "content": text}],
                                          "max_tokens": 32}, timeout=3600):
            return False
        title = {"model": model, "messages": [
            {"role": "system", "content": "You write short titles for chat sessions. Reply with JSON only."},
            {"role": "user", "content": "Write a short title for a chat about adding two numbers in Python."}],
            "reasoning_effort": "none",
            "response_format": {"type": "json_schema", "json_schema": {"name": "session_title", "strict": True,
                                                                       "schema": TITLE_SCHEMA}},
            "max_tokens": a.max_tokens, "stream": False}
        if not step(f"title_{tag}", title):
            return False
        forced = {"model": model, "messages": [{"role": "user", "content": "What is the weather in Paris? Use the tool."}],
                  "tools": [TOOL], "tool_choice": "required", "max_tokens": a.max_tokens}
        return step(f"forced_tool_{tag}", forced)

    rec.add(event="start", model=model, order=a.order, paragraphs=a.paragraphs, then_long=a.then_long,
            container=container_state(a.container), **gpu())
    if a.order == 2:
        warm = {"model": model, "messages": [{"role": "user", "content": "Reply with a JSON object."}],
                "response_format": {"type": "json_schema", "json_schema": {"name": "w", "schema": {"type": "object"}}},
                "max_tokens": 8}
        if not step("warmup", warm, timeout=300):
            rec.add(event="end", result="failed at warmup")
            return
    if not long_phase("p1", a.paragraphs):
        rec.add(event="end", result="failed")
        return
    if a.then_long and not long_phase("p2", a.then_long):
        rec.add(event="end", result="failed in the second long phase")
        return
    if a.decode_check:
        rec.add(event="decode_check", **decode_check(base, model))
    rec.add(event="end", result="all requests answered 200 and the server stayed up", **gpu())


if __name__ == "__main__":
    main()
