"""API-shape probe for an OpenAI-compatible server, written for the SGLang hands-on lane.

Standard library only; run with python3 -I. It keeps shapes (keys and the values that matter), not whole texts.
Usage:
  python3 -I api_probe.py --base http://127.0.0.1:30000 --model qwen3.5-9b --thinking toggle --out shapes.json
Thinking types: toggle (enable_thinking switch), effort (gpt-oss style reasoning_effort), always, none.
"""
import argparse
import json
import statistics
import time
import urllib.error
import urllib.request

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

TOOL = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}

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


def preview(text, n=80):
    if text is None:
        return None
    text = str(text)
    return text if len(text) <= n else text[:n] + f"... ({len(text)} chars)"


def http(base, path, payload=None, timeout=1800):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"},
                                 method="POST" if data is not None else "GET")
    t0 = time.time()
    try:
        with OPENER.open(req, timeout=timeout) as resp:
            body = resp.read()
            status = resp.status
    except urllib.error.HTTPError as err:
        body = err.read()
        status = err.code
    except (urllib.error.URLError, OSError) as err:
        return {"status": None, "error": repr(err)[:300], "seconds": round(time.time() - t0, 3)}, None
    seconds = round(time.time() - t0, 3)
    try:
        parsed = json.loads(body)
    except ValueError:
        parsed = None
    head = {"status": status, "seconds": seconds}
    if parsed is None:
        head["body_text"] = preview(body.decode(errors="replace"), 300)
    return head, parsed


def call_shape(call):
    fn = call.get("function") or {}
    args = fn.get("arguments")
    try:
        parsed = json.loads(args) if isinstance(args, str) else None
    except ValueError:
        parsed = "unparsable"
    return {"keys": sorted(call.keys()), "id": preview(call.get("id"), 40), "type": call.get("type"),
            "index": call.get("index"), "name": fn.get("name"), "arguments_type": type(args).__name__,
            "arguments": preview(args, 120), "arguments_json": parsed}


def chat_shape(head, body):
    out = dict(head)
    if not isinstance(body, dict):
        return out
    if "choices" not in body:
        out["error_body"] = {k: preview(v, 300) for k, v in body.items()}
        return out
    out["top_keys"] = sorted(body.keys())
    choice = body["choices"][0]
    msg = choice.get("message") or {}
    out["choice_keys"] = sorted(choice.keys())
    out["message_keys"] = sorted(msg.keys())
    out["content"] = preview(msg.get("content"))
    out["content_chars"] = len(msg.get("content") or "")
    out["reasoning_content_chars"] = len(msg.get("reasoning_content") or "") if "reasoning_content" in msg else "absent"
    out["reasoning_chars"] = len(msg.get("reasoning") or "") if "reasoning" in msg else "absent"
    out["reasoning_preview"] = preview(msg.get("reasoning_content") or msg.get("reasoning"), 60)
    out["tool_calls"] = [call_shape(c) for c in (msg.get("tool_calls") or [])]
    out["finish_reason"] = choice.get("finish_reason")
    out["matched_stop"] = choice.get("matched_stop")
    out["usage"] = body.get("usage")
    out["model"] = body.get("model")
    return out


def stream(base, payload, timeout=1800):
    payload = dict(payload, stream=True)
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    out = {"chunks": 0, "delta_keys": set(), "chunk_keys": set(), "finish_reasons": [], "usage_chunks": [],
           "content_chars": 0, "reasoning_content_chars": 0, "reasoning_chars": 0, "done_marker": False}
    calls = {}
    first = last = None
    content = []
    try:
        resp = OPENER.open(req, timeout=timeout)
    except urllib.error.HTTPError as err:
        return {"status": err.code, "error_body": preview(err.read().decode(errors="replace"), 400)}
    with resp:
        out["status"] = resp.status
        out["content_type"] = resp.headers.get("Content-Type")
        for raw in resp:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                out["done_marker"] = True
                break
            chunk = json.loads(data)
            out["chunks"] += 1
            out["chunk_keys"].update(chunk.keys())
            if chunk.get("usage"):
                out["usage_chunks"].append({"choices_len": len(chunk.get("choices") or []), "usage": chunk["usage"]})
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                out["delta_keys"].update(k for k, v in delta.items() if v is not None)
                if choice.get("finish_reason"):
                    out["finish_reasons"].append(choice["finish_reason"])
                text = delta.get("content") or ""
                rc = delta.get("reasoning_content") or ""
                rr = delta.get("reasoning") or ""
                out["content_chars"] += len(text)
                out["reasoning_content_chars"] += len(rc)
                out["reasoning_chars"] += len(rr)
                content.append(text)
                for tc in delta.get("tool_calls") or []:
                    slot = calls.setdefault(tc.get("index", 0), {"id": None, "type": None, "name": None, "arguments": "",
                                                                  "fragments": 0})
                    slot["fragments"] += 1
                    slot["id"] = slot["id"] or tc.get("id")
                    slot["type"] = slot["type"] or tc.get("type")
                    fn = tc.get("function") or {}
                    slot["name"] = slot["name"] or fn.get("name")
                    slot["arguments"] += fn.get("arguments") or ""
                if text or rc or rr or delta.get("tool_calls"):
                    now = time.time()
                    first = first or now
                    last = now
    end = time.time()
    out["delta_keys"] = sorted(out["delta_keys"])
    out["chunk_keys"] = sorted(out["chunk_keys"])
    out["content"] = preview("".join(content))
    out["tool_calls"] = [dict(v, index=k, arguments=preview(v["arguments"], 120)) for k, v in sorted(calls.items())]
    out["ttft_s"] = round((first or end) - t0, 3)
    out["total_s"] = round(end - t0, 3)
    usage = out["usage_chunks"][-1]["usage"] if out["usage_chunks"] else None
    if usage and first and last and last > first and (usage.get("completion_tokens") or 0) > 1:
        out["decode_tok_s"] = round((usage["completion_tokens"] - 1) / (last - first), 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--thinking", choices=["toggle", "effort", "always", "none"], required=True)
    ap.add_argument("--context", type=int, required=True)
    ap.add_argument("--long-margin", type=int, default=6144, help="tokens left free below the context")
    ap.add_argument("--skip-long", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    base, model = a.base.rstrip("/"), a.model
    res = {"model": model, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    off = {"chat_template_kwargs": {"enable_thinking": False}} if a.thinking == "toggle" else {}
    big = 4096 if a.thinking in ("effort", "always") else 256

    def save():
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1, default=sorted)

    head, body = http(base, "/v1/models")
    res["models"] = dict(head, data=[{"keys": sorted(m.keys()), "id": m.get("id"), "max_model_len": m.get("max_model_len"),
                                      "owned_by": m.get("owned_by"), "root": preview(m.get("root"), 60)}
                                     for m in (body or {}).get("data", [])])
    for path in ("/health", "/ready", "/health_generate"):
        head, _ = http(base, path)
        res[path] = head
    head, info = http(base, "/server_info")
    if isinstance(info, dict):
        wanted = ("mem_fraction_static", "context_length", "max_total_tokens", "attention_backend", "prefill_attention_backend",
                  "decode_attention_backend", "sampling_backend", "chunked_prefill_size", "max_prefill_tokens",
                  "max_running_requests", "kv_cache_dtype", "page_size", "tool_call_parser", "reasoning_parser",
                  "cuda_graph_max_bs", "cuda_graph_max_bs_decode", "cuda_graph_max_bs_prefill", "disable_cuda_graph",
                  "disable_prefill_cuda_graph", "disable_decode_cuda_graph", "cuda_graph_backend_decode",
                  "cuda_graph_backend_prefill", "grammar_backend", "quantization", "fp4_gemm_backend", "moe_runner_backend",
                  "language_model_only", "sampling_defaults", "enable_metrics", "watchdog_timeout", "served_model_name",
                  "mamba_full_memory_ratio", "max_mamba_cache_size", "schedule_policy", "disable_radix_cache", "version")
        res["server_info"] = {k: info.get(k) for k in wanted if k in info}
        res["server_info"]["internal_states"] = info.get("internal_states")
        res["server_info"]["key_count"] = len(info)
    else:
        res["server_info"] = head
    save()

    ok = [{"role": "user", "content": "Reply with exactly: OK"}]
    res["plain"] = chat_shape(*http(base, "/v1/chat/completions", dict({"model": model, "messages": ok, "max_tokens": big}, **off)))
    res["plain_stream_usage"] = stream(base, dict({"model": model, "messages": ok, "max_tokens": big,
                                                   "stream_options": {"include_usage": True}}, **off))
    res["plain_stream_no_usage"] = stream(base, dict({"model": model, "messages": ok, "max_tokens": big}, **off))
    save()

    ask = [{"role": "user", "content": "What is the weather in Paris? Use the tool."}]
    tool_req = dict({"model": model, "messages": ask, "tools": [TOOL], "tool_choice": "auto", "max_tokens": big}, **off)
    head, body = http(base, "/v1/chat/completions", tool_req)
    res["tool_auto"] = chat_shape(head, body)
    res["tool_auto_stream"] = stream(base, tool_req)
    res["tool_no_tool_choice"] = chat_shape(*http(base, "/v1/chat/completions", {k: v for k, v in tool_req.items() if k != "tool_choice"}))
    save()

    if isinstance(body, dict) and body.get("choices") and body["choices"][0]["message"].get("tool_calls"):
        msg = body["choices"][0]["message"]
        call = msg["tool_calls"][0]
        assistant = {"role": "assistant", "content": msg.get("content") or "", "tool_calls": [
            {"id": call.get("id"), "type": "function", "function": {"name": call["function"]["name"],
                                                                     "arguments": call["function"]["arguments"]}}]}
        if msg.get("reasoning_content"):
            assistant["reasoning_content"] = msg["reasoning_content"]
        result = {"role": "tool", "tool_call_id": call.get("id"),
                  "content": json.dumps({"city": "Paris", "temperature_c": 18, "condition": "sunny"})}
        turn2 = dict({"model": model, "messages": ask + [assistant, result], "tools": [TOOL], "max_tokens": big}, **off)
        res["tool_second_turn"] = chat_shape(*http(base, "/v1/chat/completions", turn2))
        res["tool_second_turn"]["mentions_18"] = "18" in (res["tool_second_turn"].get("content") or "")
        res["tool_second_turn_stream"] = stream(base, turn2)
        res["tool_second_turn_sent_reasoning_content"] = "reasoning_content" in assistant
    else:
        res["tool_second_turn"] = "skipped: no parsed tool call in tool_auto"
    save()

    q = [{"role": "user", "content": "What is 17 * 23? Answer with the number only."}]
    thinking = {}
    if a.thinking == "toggle":
        for label, extra in (("enable_thinking_true", {"chat_template_kwargs": {"enable_thinking": True}}),
                             ("enable_thinking_false", {"chat_template_kwargs": {"enable_thinking": False}}),
                             ("no_field_server_default", {}),
                             ("reasoning_effort_none", {"reasoning_effort": "none"}),
                             ("reasoning_effort_low", {"reasoning_effort": "low"}),
                             ("reasoning_effort_high", {"reasoning_effort": "high"})):
            thinking[label] = chat_shape(*http(base, "/v1/chat/completions", dict({"model": model, "messages": q, "max_tokens": 4096}, **extra)))
    else:
        for label, extra in (("no_field", {}), ("reasoning_effort_low", {"reasoning_effort": "low"}),
                             ("reasoning_effort_medium", {"reasoning_effort": "medium"}),
                             ("reasoning_effort_high", {"reasoning_effort": "high"}),
                             ("reasoning_effort_none", {"reasoning_effort": "none"}),
                             ("reasoning_effort_minimal", {"reasoning_effort": "minimal"}),
                             ("reasoning_effort_xhigh", {"reasoning_effort": "xhigh"}),
                             ("enable_thinking_false", {"chat_template_kwargs": {"enable_thinking": False}})):
            thinking[label] = chat_shape(*http(base, "/v1/chat/completions", dict({"model": model, "messages": q, "max_tokens": 4096}, **extra)))
    res["thinking"] = thinking
    res["thinking_stream_on"] = stream(base, dict({"model": model, "messages": q, "max_tokens": 4096,
                                                   "stream_options": {"include_usage": True}},
                                                  **({"chat_template_kwargs": {"enable_thinking": True}} if a.thinking == "toggle" else {})))
    save()

    compat = {}
    base_req = dict({"model": model, "messages": ok, "max_tokens": big}, **off)
    for label, extra in (("tools_empty_list", {"tools": []}),
                         ("prompt_cache_key", {"prompt_cache_key": "llmctl-probe"}),
                         ("max_completion_tokens", {"max_completion_tokens": big}),
                         ("tool_choice_auto_without_tools", {"tool_choice": "auto"}),
                         ("tool_choice_none_with_tools", {"tools": [TOOL], "tool_choice": "none"}),
                         ("wrong_model_name", {"model": "not-the-served-name"}),
                         ("max_tokens_above_context", {"max_tokens": a.context * 2}),
                         ("history_reasoning_field", {"messages": [{"role": "user", "content": "Hi"},
                                                                   {"role": "assistant", "content": "Hello.", "reasoning": "The user greets."},
                                                                   {"role": "user", "content": "Reply with exactly: OK"}]}),
                         ("history_reasoning_content_field", {"messages": [{"role": "user", "content": "Hi"},
                                                                           {"role": "assistant", "content": "Hello.", "reasoning_content": "The user greets."},
                                                                           {"role": "user", "content": "Reply with exactly: OK"}]})):
        req = dict(base_req)
        req.update(extra)
        shape = chat_shape(*http(base, "/v1/chat/completions", req))
        compat[label] = {k: shape.get(k) for k in ("status", "finish_reason", "content", "tool_calls", "error_body", "usage",
                                                   "reasoning_content_chars") if k in shape}
    res["compat"] = compat
    save()

    essay = [{"role": "user", "content": "Write a long, detailed essay of at least 1500 words on the history of computing, "
                                         "from mechanical calculators to modern GPUs."}]
    runs = []
    for i in range(3):
        runs.append(stream(base, dict({"model": model, "messages": [{"role": "user", "content": f"Run {i}. " + essay[0]["content"]}],
                                       "max_tokens": 512, "stream_options": {"include_usage": True}}, **off)))
    res["decode_512"] = [{k: r.get(k) for k in ("status", "ttft_s", "total_s", "decode_tok_s", "finish_reasons", "usage_chunks",
                                                "reasoning_content_chars", "content_chars")} for r in runs]
    speeds = [r["decode_tok_s"] for r in runs if r.get("decode_tok_s")]
    res["decode_512_median_tok_s"] = statistics.median(speeds) if speeds else None
    save()

    if not a.skip_long:
        calib = stream(base, dict({"model": model, "max_tokens": 8, "stream_options": {"include_usage": True},
                                   "messages": [{"role": "user", "content": PARAGRAPH * 100 + "\nIn one sentence, what does the function above compute?"}]}, **off))
        per_100 = ((calib.get("usage_chunks") or [{}])[-1].get("usage") or {}).get("prompt_tokens")
        res["long_calibration"] = {"paragraphs": 100, "prompt_tokens": per_100}
        if per_100:
            per = per_100 / 100.0
            n = int((a.context - a.long_margin) / per)
            msgs = [{"role": "user", "content": PARAGRAPH * n + "\nIn one sentence, what does the function above compute?"}]
            first = stream(base, dict({"model": model, "max_tokens": 64, "messages": msgs, "stream_options": {"include_usage": True}}, **off), timeout=3600)
            again = stream(base, dict({"model": model, "max_tokens": 64, "messages": msgs, "stream_options": {"include_usage": True}}, **off), timeout=3600)
            keep = ("status", "error_body", "ttft_s", "total_s", "decode_tok_s", "finish_reasons", "usage_chunks", "content")
            res["long_prompt"] = {"paragraphs": n, "first": {k: first.get(k) for k in keep},
                                  "repeat_same_prompt": {k: again.get(k) for k in keep}}
            pt = ((first.get("usage_chunks") or [{}])[-1].get("usage") or {}).get("prompt_tokens")
            if pt and first.get("ttft_s"):
                res["long_prompt"]["prefill_tok_s_first"] = round(pt / first["ttft_s"], 1)
    # Grammar-constrained requests last: on gpt-oss attempt 2 the first one crashed the server.
    grammar_res = {}
    base_req = dict({"model": model, "messages": ok, "max_tokens": big}, **off)
    for label, extra in (("tool_choice_required", {"tools": [TOOL], "tool_choice": "required"}),
                         ("response_format_json_schema", {"messages": [{"role": "user", "content": "Give the capital of France as JSON."}],
                                                          "response_format": {"type": "json_schema", "json_schema": {"name": "capital", "schema": {
                                                              "type": "object", "properties": {"capital": {"type": "string"}},
                                                              "required": ["capital"]}}}})):
        req = dict(base_req)
        req.update(extra)
        shape = chat_shape(*http(base, "/v1/chat/completions", req))
        grammar_res[label] = {k: shape.get(k) for k in ("status", "finish_reason", "content", "tool_calls", "error_body", "usage",
                                                        "reasoning_content_chars", "error") if k in shape}
        res["grammar"] = grammar_res
        save()
    res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    save()
    print(json.dumps({"plain": {k: res["plain"].get(k) for k in ("status", "content", "finish_reason", "reasoning_content_chars")},
                      "tool_auto": {k: res["tool_auto"].get(k) for k in ("status", "finish_reason", "tool_calls")},
                      "decode_512_median_tok_s": res["decode_512_median_tok_s"],
                      "long": {k: (res.get("long_prompt") or {}).get(k) for k in ("paragraphs", "prefill_tok_s_first")}},
                     indent=1, default=sorted))


if __name__ == "__main__":
    main()
