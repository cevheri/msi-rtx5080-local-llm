"""Add the raw Harmony text and SGLang's own parse of it to saved plain replays. Run with python3 -I.

Usage: rawparse.py <replay folder> <n>-request.json
For each plain-<i>.json written by replay.py, turns the reply's response_token_ids back into text with special tokens
(/v1/detokenize, skip_special_tokens false), runs that text through /separate_reasoning (gpt-oss) and
/parse_function_call (gpt-oss, with the request's tools), stores the three in the file, and writes raw.jsonl with one
line per plain replay: finish_reason, matched_stop, content, reasoning_content, tool call names, the raw text and the
two parser results.
"""
import json
import pathlib
import sys
import urllib.request

BASE = "http://127.0.0.1:30000"


def post(path: str, body: dict) -> dict:
    request = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())


def main() -> None:
    folder = pathlib.Path(sys.argv[1])
    body = json.loads(pathlib.Path(sys.argv[2]).read_text())["body"]
    tools, model = body.get("tools") or [], body["model"]
    names = sorted(folder.glob("plain-*.json"), key=lambda path: int(path.stem.split("-")[1]))
    with open(folder / "raw.jsonl", "w", encoding="utf-8") as out:
        for path in names:
            result = json.loads(path.read_text())
            if result["status"] != 200:
                continue
            choice = result["reply"]["choices"][0]
            raw_text = post("/v1/detokenize", {"model": model, "tokens": choice["response_token_ids"],
                                               "skip_special_tokens": False})["text"]
            separated = post("/separate_reasoning", {"text": raw_text, "reasoning_parser": "gpt-oss"})
            parsed = post("/parse_function_call", {"text": raw_text, "tool_call_parser": "gpt-oss", "tools": tools})
            result.update(raw_text=raw_text, separate_reasoning=separated, parse_function_call=parsed)
            path.write_text(json.dumps(result, indent=1, ensure_ascii=False))
            message = choice.get("message") or {}
            calls = [call["function"]["name"] for call in message.get("tool_calls") or []]
            out.write(json.dumps({"replay": path.stem, "finish_reason": choice.get("finish_reason"),
                                  "matched_stop": choice.get("matched_stop"), "content": message.get("content"),
                                  "reasoning_content": message.get("reasoning_content"), "tool_calls": calls,
                                  "completion_tokens": result["reply"]["usage"]["completion_tokens"],
                                  "raw_text": raw_text, "separate_reasoning": separated,
                                  "parse_function_call": parsed}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
