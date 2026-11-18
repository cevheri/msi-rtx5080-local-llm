"""Replay one captured chat request straight to SGLang and record the raw replies. Run with python3 -I.

Usage: replay.py <n>-request.json <out folder> <streamed count> <plain count> [<seed>]
Streamed replays send the captured body unchanged (stream true, as OpenCode sent it) and keep the event stream with
the joined content, reasoning_content, tool_calls, finish_reason and usage. Plain replays send the same body with
stream false and return_token_ids true, then turn the generated token ids back into text with special tokens
(/v1/detokenize, skip_special_tokens false) and run that text through SGLang's own gpt-oss parsers
(/separate_reasoning and /parse_function_call), so the raw Harmony output can be told apart from what the parsers made
of it. With a seed, every plain replay sends it, so a reply can be reproduced.
Writes stream-<i>.sse and stream-<i>.json, plain-<i>.json, and summary.jsonl with one line per replay.
"""
import copy
import json
import pathlib
import sys
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:30000"


def post(path: str, body: dict, timeout: int = 600) -> tuple[int, bytes]:
    request = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def join_stream(raw: bytes) -> dict:
    content, reasoning, calls, finish, usage, matched = "", "", {}, None, None, None
    for line in raw.decode(errors="replace").splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        chunk = json.loads(line[6:])
        if chunk.get("usage"):
            usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            content += delta.get("content") or ""
            reasoning += delta.get("reasoning_content") or ""
            for call in delta.get("tool_calls") or []:
                slot = calls.setdefault(call.get("index", 0), {"name": "", "arguments": "", "id": None})
                function = call.get("function") or {}
                slot["name"] += function.get("name") or ""
                slot["arguments"] += function.get("arguments") or ""
                slot["id"] = call.get("id") or slot["id"]
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
                matched = choice.get("matched_stop")
    return {"content": content, "reasoning_content": reasoning, "tool_calls": list(calls.values()),
            "finish_reason": finish, "matched_stop": matched, "usage": usage}


def main() -> None:
    record = json.loads(pathlib.Path(sys.argv[1]).read_text())
    out = pathlib.Path(sys.argv[2])
    streamed, plain = int(sys.argv[3]), int(sys.argv[4])
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else None
    out.mkdir(parents=True, exist_ok=True)
    body = record["body"]
    tools = body.get("tools") or []
    with open(out / "summary.jsonl", "a", encoding="utf-8") as summary:
        for index in range(1, streamed + 1):
            status, raw = post("/v1/chat/completions", body)
            (out / f"stream-{index}.sse").write_bytes(raw)
            joined = join_stream(raw) if status == 200 else {"error": raw.decode(errors="replace")}
            (out / f"stream-{index}.json").write_text(json.dumps(joined, indent=1, ensure_ascii=False))
            summary.write(json.dumps({"replay": f"stream-{index}", "status": status, **joined},
                                     ensure_ascii=False) + "\n")
            summary.flush()
        for index in range(1, plain + 1):
            plain_body = copy.deepcopy(body)
            plain_body["stream"] = False
            plain_body.pop("stream_options", None)
            plain_body["return_token_ids"] = True
            if seed is not None:
                plain_body["seed"] = seed
            status, raw = post("/v1/chat/completions", plain_body)
            reply = json.loads(raw)
            result = {"status": status, "reply": reply}
            if status == 200:
                choice = reply["choices"][0]
                output_ids = choice.get("response_token_ids") or []
                _, text = post("/v1/detokenize", {"model": body["model"], "tokens": output_ids,
                                                  "skip_special_tokens": False})
                raw_text = json.loads(text)["text"]
                _, separated = post("/separate_reasoning", {"text": raw_text, "reasoning_parser": "gpt-oss"})
                _, parsed = post("/parse_function_call", {"text": raw_text, "tool_call_parser": "gpt-oss",
                                                          "tools": tools})
                result.update(raw_text=raw_text, separate_reasoning=json.loads(separated),
                              parse_function_call=json.loads(parsed))
                message = choice.get("message") or {}
                line = {"replay": f"plain-{index}", "status": status, "content": message.get("content"),
                        "reasoning_content": message.get("reasoning_content"),
                        "tool_calls": message.get("tool_calls"), "finish_reason": choice.get("finish_reason"),
                        "matched_stop": choice.get("matched_stop"), "usage": reply.get("usage"),
                        "raw_text": raw_text}
            else:
                line = {"replay": f"plain-{index}", "status": status, "error": reply}
            (out / f"plain-{index}.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
            summary.write(json.dumps(line, ensure_ascii=False) + "\n")
            summary.flush()


if __name__ == "__main__":
    main()
