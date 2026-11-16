"""Smoke tests and simple speed measurements for an OpenAI-compatible LLM server.

Standard library only. Measures, per server:
  - /v1/models
  - a short "Reply with exactly: OK" request
  - a tool-call request (expects finish_reason "tool_calls")
  - decode speed from a streamed ~600-word answer (repeated runs)
  - time to first token for long prompts, and the same long prompt again (prefix or prompt cache)
"""
import argparse
import json
import statistics
import time
import urllib.request

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


def post(base, payload, stream=False, timeout=900):
    req = urllib.request.Request(
        base + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=timeout)


def chat(base, payload):
    t0 = time.time()
    with post(base, payload) as r:
        body = json.load(r)
    return body, time.time() - t0


def stream_chat(base, payload):
    payload = dict(payload, stream=True, stream_options={"include_usage": True})
    t0 = time.time()
    t_first = None
    t_last = None
    usage = None
    timings = None
    pieces = 0
    with post(base, payload, stream=True) as r:
        for raw in r:
            line = raw.decode(errors="ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            if chunk.get("timings"):
                timings = chunk["timings"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta") or {}
                text = (delta.get("content") or "") + (delta.get("reasoning_content") or "") + (delta.get("reasoning") or "")
                if text or delta.get("tool_calls"):
                    now = time.time()
                    if t_first is None:
                        t_first = now
                    t_last = now
                    pieces += 1
    t_end = time.time()
    out = {
        "ttft_s": round((t_first or t_end) - t0, 3),
        "total_s": round(t_end - t0, 3),
        "usage": usage,
        "stream_pieces": pieces,
    }
    if usage and t_first and t_last and t_last > t_first:
        ct = usage.get("completion_tokens") or 0
        out["decode_tok_s"] = round((ct - 1) / (t_last - t_first), 1) if ct > 1 else None
    if timings:
        out["server_timings"] = {k: timings.get(k) for k in ("prompt_n", "prompt_per_second", "predicted_n", "predicted_per_second", "cache_n")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--long-tokens", default="8000,30000")
    ap.add_argument("--extra", default="{}", help="JSON merged into every request body")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    base, extra = a.base.rstrip("/"), json.loads(a.extra)
    res = {"label": a.label, "base": base, "model": a.model, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}

    with urllib.request.urlopen(base + "/models", timeout=30) as r:
        res["models"] = [m["id"] for m in json.load(r)["data"]]

    body, dt = chat(base, dict({"model": a.model, "messages": [{"role": "user", "content": "Reply with exactly: OK"}], "max_tokens": 1024}, **extra))
    msg = body["choices"][0]["message"]
    res["ok_test"] = {
        "content": msg.get("content"),
        "reasoning_chars": len(msg.get("reasoning_content") or msg.get("reasoning") or ""),
        "finish": body["choices"][0].get("finish_reason"),
        "usage": body.get("usage"),
        "seconds": round(dt, 2),
    }

    tool = {"type": "function", "function": {"name": "get_weather", "description": "Get the weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
    body, dt = chat(base, dict({"model": a.model, "messages": [{"role": "user", "content": "What is the weather in Paris? Use the tool."}],
                                "tools": [tool], "max_tokens": 1024}, **extra))
    ch = body["choices"][0]
    calls = ch["message"].get("tool_calls") or []
    res["tool_test"] = {
        "finish": ch.get("finish_reason"),
        "calls": [{"name": c["function"]["name"], "arguments": c["function"]["arguments"]} for c in calls],
        "content": ch["message"].get("content"),
        "seconds": round(dt, 2),
    }

    decode = []
    for i in range(a.runs):
        decode.append(stream_chat(base, dict({"model": a.model, "max_tokens": 1500, "messages": [
            {"role": "user", "content": f"Run {i}: write a detailed, roughly 600-word explanation of how a hash map works, with one short Python example."}]}, **extra)))
    res["decode_runs"] = decode
    tps = [d["decode_tok_s"] for d in decode if d.get("decode_tok_s")]
    res["decode_tok_s_median"] = statistics.median(tps) if tps else None

    longs = []
    for n in [int(x) for x in a.long_tokens.split(",") if x]:
        text = PARAGRAPH * max(1, n // 95)
        msgs = [{"role": "user", "content": text + "\nIn one sentence, what does the function above compute?"}]
        first = stream_chat(base, dict({"model": a.model, "max_tokens": 64, "messages": msgs}, **extra))
        again = stream_chat(base, dict({"model": a.model, "max_tokens": 64, "messages": msgs}, **extra))
        longs.append({"target_tokens": n, "first": first, "repeat_same_prompt": again})
    res["long_prompt"] = longs
    res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({
        "label": a.label,
        "models": res["models"],
        "ok": res["ok_test"],
        "tool": res["tool_test"],
        "decode_tok_s_median": res["decode_tok_s_median"],
        "decode_runs": [{k: d.get(k) for k in ("ttft_s", "decode_tok_s", "usage", "server_timings")} for d in decode],
        "long": [{"target": l["target_tokens"],
                  "prompt_tokens": (l["first"].get("usage") or {}).get("prompt_tokens"),
                  "ttft_first_s": l["first"]["ttft_s"], "ttft_repeat_s": l["repeat_same_prompt"]["ttft_s"],
                  "server_first": l["first"].get("server_timings"), "server_repeat": l["repeat_same_prompt"].get("server_timings")}
                 for l in longs],
    }, indent=1))


if __name__ == "__main__":
    main()
