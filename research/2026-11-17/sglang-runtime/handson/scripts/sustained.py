"""Sustained-load check: back-to-back streamed requests with fresh prompts of varying length for a fixed time.

Standard library only; run with python3 -I. Records per request: prompt tokens, TTFT, decode tok/s, finish reason, errors.
Usage: python3 -I sustained.py --base http://127.0.0.1:30000 --model qwen3.5-9b --seconds 300 --out sustained.json
"""
import argparse
import json
import random
import statistics
import time
import urllib.error
import urllib.request

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
WORDS = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon "
         "phi chi psi omega river stone cloud forest engine signal vector matrix kernel buffer packet thread").split()


def one(base, model, extra, n_words, max_tokens, rng):
    text = " ".join(rng.choice(WORDS) for _ in range(n_words))
    payload = dict({"model": model, "max_tokens": max_tokens, "stream": True, "stream_options": {"include_usage": True},
                    "messages": [{"role": "user", "content": f"Nonce {rng.random()}. Summarise this text, then continue it at length: {text}"}]},
                   **extra)
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    first = last = None
    usage = None
    finish = None
    try:
        with OPENER.open(req, timeout=600) as resp:
            for raw in resp:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for ch in chunk.get("choices") or []:
                    d = ch.get("delta") or {}
                    if d.get("content") or d.get("reasoning_content"):
                        now = time.time()
                        first = first or now
                        last = now
                    finish = ch.get("finish_reason") or finish
    except (urllib.error.URLError, OSError, ValueError) as err:
        return {"error": repr(err)[:200], "seconds": round(time.time() - t0, 2)}
    out = {"prompt_tokens": (usage or {}).get("prompt_tokens"), "completion_tokens": (usage or {}).get("completion_tokens"),
           "ttft_s": round((first or time.time()) - t0, 3), "finish": finish}
    ct = out["completion_tokens"] or 0
    if first and last and last > first and ct > 1:
        out["decode_tok_s"] = round((ct - 1) / (last - first), 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seconds", type=int, default=300)
    ap.add_argument("--extra", default="{}")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = random.Random(20261117)
    extra = json.loads(a.extra)
    runs = []
    end = time.time() + a.seconds
    while time.time() < end:
        n_words = rng.choice([200, 1000, 3000, 6000, 12000])
        r = one(a.base.rstrip("/"), a.model, extra, n_words, rng.choice([256, 512]), rng)
        r["t"] = round(time.time(), 1)
        runs.append(r)
        if "error" in r:
            break
    speeds = [r["decode_tok_s"] for r in runs if r.get("decode_tok_s")]
    res = {"requests": len(runs), "errors": [r for r in runs if "error" in r],
           "decode_tok_s": {"min": min(speeds), "median": statistics.median(speeds), "max": max(speeds)} if speeds else None,
           "prompt_tokens_max": max((r.get("prompt_tokens") or 0) for r in runs) if runs else None,
           "first_half_median": statistics.median(speeds[: len(speeds) // 2]) if len(speeds) > 3 else None,
           "second_half_median": statistics.median(speeds[len(speeds) // 2:]) if len(speeds) > 3 else None,
           "runs": runs}
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "runs"}))


if __name__ == "__main__":
    main()
