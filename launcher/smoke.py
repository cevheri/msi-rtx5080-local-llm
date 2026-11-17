"""`llmctl test`: three requests against the running server, as in llamacpp/README.md#smoke-tests."""

import json
import re

from launcher.errors import StartError
from launcher.runtimes import ADAPTERS

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}


def _chat(host, url: str, served: str, payload: dict) -> tuple[dict, float]:
    body = json.dumps({"model": served, **payload}).encode()
    started = host.monotonic()
    try:
        response = host.http("POST", url, headers={"Content-Type": "application/json"}, body=body, timeout=600)
    except OSError as exc:
        raise StartError(f"the server at {url} did not answer: {exc}; check ./llmctl status") from None
    elapsed = host.monotonic() - started
    if response.status != 200:
        raise StartError(f"the server answered HTTP {response.status}: {response.body[:300].decode(errors='replace')}")
    return json.loads(response.body), elapsed


def smoke(host, recipe, runtime) -> int:
    served = ADAPTERS[recipe.runtime].served_name(recipe, runtime)
    url = f"http://127.0.0.1:{runtime.host_port}/v1/chat/completions"
    out = host.out
    failures = []

    data, _ = _chat(host, url, served, {"messages": [{"role": "user", "content": "Reply with exactly the word OK."}],
                                        "max_tokens": 1024})
    message = data["choices"][0]["message"]
    content = (message.get("content") or "").strip()
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    if re.fullmatch(r"[\s*_`\"'.!,]*ok[\s*_`\"'.!,]*", content.lower()):
        out.write(f"reply: ok ({content[:40]!r})\n")
    elif not content and reasoning:
        out.write("reply: ok (reasoning only; the model spent its budget thinking)\n")
    else:
        out.write(f"reply: FAILED, expected OK, got {content[:80]!r}\n")
        failures.append("reply")

    if recipe.chat_only:
        out.write("tool call: skipped, this recipe is chat only\n")
    else:
        data, _ = _chat(host, url, served, {"messages": [{"role": "user", "content": "What is the weather in Paris? Use the tool."}],
                                            "tools": [WEATHER_TOOL], "max_tokens": 2048})
        choice = data["choices"][0]
        calls = choice["message"].get("tool_calls") or []
        names = [call.get("function", {}).get("name") for call in calls]
        if choice.get("finish_reason") == "tool_calls" and "get_weather" in names:
            arguments = next(call["function"].get("arguments", "") for call in calls if call["function"]["name"] == "get_weather")
            out.write(f"tool call: ok (get_weather {arguments})\n")
        else:
            text = (choice["message"].get("content") or "")[:80]
            out.write(f"tool call: FAILED, the tool call did not arrive (finish {choice.get('finish_reason')!r}, text {text!r})\n")
            failures.append("tool call")

    data, elapsed = _chat(host, url, served, {"messages": [{"role": "user", "content":
                                                            "Write a short story of about 300 words about a lighthouse keeper."}],
                                              "max_tokens": 512})
    tokens = (data.get("usage") or {}).get("completion_tokens", 0)
    speed = (data.get("timings") or {}).get("predicted_per_second") or (tokens / elapsed if elapsed > 0 else 0.0)
    out.write(f"decode: {speed:.1f} tok/s ({tokens} tokens)\n")
    out.write(f"recipe evidence: {recipe.evidence.summary}\n")
    if failures:
        out.write(f"smoke test failed: {', '.join(failures)}\n")
        return 4
    return 0
