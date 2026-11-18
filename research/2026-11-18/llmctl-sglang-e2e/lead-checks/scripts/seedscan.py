"""Replay one captured chat request with a range of sampling seeds and record each raw reply. Run with python3 -I.

Usage: seedscan.py <n>-request.json <out.jsonl> <first seed> <last seed> [stream]
Without "stream": the body with stream false, return_token_ids true and the seed; each line holds the seed, the
finish_reason, matched_stop, usage, content, reasoning_content, the tool call names and the raw text with special
tokens (/v1/detokenize of response_token_ids). With "stream": the body as captured plus the seed, streamed as OpenCode
streams, and each line holds what the event stream delivered.
"""
import copy
import json
import pathlib
import sys
import urllib.request

BASE = "http://127.0.0.1:30000"


def post(path: str, body: dict) -> bytes:
    request = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        return response.read()


def main() -> None:
    body = json.loads(pathlib.Path(sys.argv[1]).read_text())["body"]
    first, last = int(sys.argv[3]), int(sys.argv[4])
    stream = len(sys.argv) > 5 and sys.argv[5] == "stream"
    with open(sys.argv[2], "a", encoding="utf-8") as out:
        for seed in range(first, last + 1):
            sent = copy.deepcopy(body)
            sent["seed"] = seed
            if stream:
                raw = post("/v1/chat/completions", sent).decode()
                content, reasoning, calls, finish, matched, usage = "", "", [], None, None, None
                for line in raw.splitlines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    chunk = json.loads(line[6:])
                    usage = chunk.get("usage") or usage
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        content += delta.get("content") or ""
                        reasoning += delta.get("reasoning_content") or ""
                        calls += [(call.get("function") or {}).get("name") for call in delta.get("tool_calls") or []
                                  if (call.get("function") or {}).get("name")]
                        finish = choice.get("finish_reason") or finish
                        matched = choice.get("matched_stop") if choice.get("finish_reason") else matched
                line = {"seed": seed, "mode": "stream", "finish_reason": finish, "matched_stop": matched,
                        "usage": usage, "content": content, "reasoning_content": reasoning, "tool_calls": calls}
            else:
                sent["stream"] = False
                sent.pop("stream_options", None)
                sent["return_token_ids"] = True
                reply = json.loads(post("/v1/chat/completions", sent))
                choice = reply["choices"][0]
                message = choice.get("message") or {}
                raw_text = json.loads(post("/v1/detokenize", {"model": body["model"],
                                                              "tokens": choice["response_token_ids"],
                                                              "skip_special_tokens": False}))["text"]
                line = {"seed": seed, "mode": "plain", "finish_reason": choice.get("finish_reason"),
                        "matched_stop": choice.get("matched_stop"), "usage": reply.get("usage"),
                        "content": message.get("content"), "reasoning_content": message.get("reasoning_content"),
                        "tool_calls": [call["function"]["name"] for call in message.get("tool_calls") or []],
                        "raw_text": raw_text}
            out.write(json.dumps(line, ensure_ascii=False) + "\n")
            out.flush()


if __name__ == "__main__":
    main()
