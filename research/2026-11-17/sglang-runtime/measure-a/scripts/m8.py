"""M8 sampling probe: chat requests that send no sampling field, so the server's defaults decide.
plain: one short request (its log line shows the sampling parameters the server filled in).
five: five identical requests; identical replies mean top_k 1 (greedy) applied.
k2: one chat request and one tool call (the overlay check on a remote-code model).
Replies are kept as a sha256 and a short excerpt only.
Run with python3 -I: m8.py --model NAME --mode plain five | k2 --out FILE.jsonl"""
import argparse
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mcommon import TOOL, Records, chat_summary, http  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:30000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", nargs="+", choices=["plain", "five", "k2"], required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rec = Records(a.out)
    if "plain" in a.mode:
        payload = {"model": a.model, "messages": [{"role": "user", "content": "Reply with exactly: OK"}], "max_tokens": 16}
        head, body = http(a.base, "/v1/chat/completions", payload)
        rec.add(event="plain", request_keys=sorted(payload), **chat_summary(head, body))
    if "five" in a.mode:
        payload = {"model": a.model, "messages": [{"role": "user", "content": "Write one sentence about a cat who finds a key."}],
                   "max_tokens": 40}
        digests = []
        for i in range(5):
            head, body = http(a.base, "/v1/chat/completions", payload)
            content = ((body or {}).get("choices") or [{}])[0].get("message", {}).get("content") or ""
            digest = hashlib.sha256(content.encode()).hexdigest()[:16]
            digests.append(digest)
            rec.add(event="five", i=i, request_keys=sorted(payload), status=head["status"], sha256_16=digest,
                    excerpt=content[:60], chars=len(content))
        rec.add(event="five_summary", distinct_replies=len(set(digests)), identical=len(set(digests)) == 1)
    if "k2" in a.mode:
        payload = {"model": a.model, "messages": [{"role": "user", "content": "What is 17 * 23? Answer with the number only."}],
                   "max_tokens": 4096}
        rec.add(event="k2_chat", **chat_summary(*http(a.base, "/v1/chat/completions", payload)))
        payload = {"model": a.model, "messages": [{"role": "user", "content": "What is the weather in Paris? Use the tool."}],
                   "tools": [TOOL], "tool_choice": "auto", "max_tokens": 4096}
        rec.add(event="k2_tool_call", **chat_summary(*http(a.base, "/v1/chat/completions", payload)))


if __name__ == "__main__":
    main()
