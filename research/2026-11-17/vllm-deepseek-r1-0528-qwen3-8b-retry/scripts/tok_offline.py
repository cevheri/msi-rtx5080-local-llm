"""Offline tokenizer check for DeepSeek-R1-0528-Qwen3-8B inside vllm/vllm-openai:v0.33.0 (CPU only, no network).

For each candidate tokenizer option it loads the tokenizer the way vLLM v0.33.0 does (vllm.tokenizers.get_tokenizer),
round-trips a sentence, and counts the chat prompt of three fixed messages, which the llama.cpp record counted as
8 ("Reply with exactly: OK"), 8,078 (the probe's 8k prompt) and 30,254 (the probe's 30k prompt) tokens.
Run inside the image: python3 tok_offline.py OUT.json [LABEL ...]
"""
import json
import sys
import traceback

from vllm.tokenizers import get_tokenizer

SNAP = "/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-0528-Qwen3-8B/snapshots/6e8885a6ff5c1dc5201574c8fd700323f23c25fa"
GGUF = ("/root/.cache/huggingface/hub/models--unsloth--DeepSeek-R1-0528-Qwen3-8B-GGUF/snapshots/"
        "eb48357c179d34dbf515983f798dfb8752a0f261/DeepSeek-R1-0528-Qwen3-8B-Q8_0.gguf")
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
MESSAGES = {
    "ok": "Reply with exactly: OK",
    "probe_8k": PARAGRAPH * (8000 // 95) + TAIL,
    "probe_30k": PARAGRAPH * (30000 // 95) + TAIL,
}
LLAMACPP = {"ok": 8, "probe_8k": 8078, "probe_30k": 30254}

CANDIDATES = [
    ("auto", {"tokenizer_name": SNAP, "tokenizer_mode": "auto"}),
    ("deepseek_v32", {"tokenizer_name": SNAP, "tokenizer_mode": "deepseek_v32"}),
    ("deepseek_v4", {"tokenizer_name": SNAP, "tokenizer_mode": "deepseek_v4"}),
    ("slow", {"tokenizer_name": SNAP, "tokenizer_mode": "slow"}),
    ("gguf_tokenizer", {"tokenizer_name": GGUF, "tokenizer_mode": "auto"}),
]


def run(label, kw):
    out = {"label": label, "options": kw}
    try:
        tok = get_tokenizer(kw["tokenizer_name"], tokenizer_mode=kw["tokenizer_mode"], trust_remote_code=False)
    except Exception as err:  # record the exact error and go on
        out["load_error"] = f"{type(err).__name__}: {err}"[:600]
        out["traceback_tail"] = traceback.format_exc().splitlines()[-4:]
        return out
    out["class"] = type(tok).__name__
    backend = getattr(tok, "_tokenizer", None) or getattr(tok, "backend_tokenizer", None)
    try:
        out["pre_tokenizer"] = type(backend.pre_tokenizer).__name__ if backend is not None else None
        out["decoder"] = type(backend.decoder).__name__ if backend is not None else None
    except Exception as err:
        out["pre_tokenizer"] = f"unreadable: {err}"
    ids = tok.encode("Say hello world.", add_special_tokens=False)
    out["encode_say_hello"] = ids
    out["decode_say_hello"] = tok.decode(ids)
    counts = {}
    for name, text in MESSAGES.items():
        try:
            rendered = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                               tokenize=False)
            enc = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                          tokenize=True)
            if isinstance(enc, dict) or hasattr(enc, "keys"):
                enc = enc["input_ids"]
            counts[name] = {"tokens": len(enc), "llamacpp": LLAMACPP[name], "diff": len(enc) - LLAMACPP[name],
                            "rendered_head": rendered[:60], "rendered_tail": rendered[-40:]}
        except Exception as err:
            counts[name] = {"error": f"{type(err).__name__}: {err}"[:400]}
    out["chat_counts"] = counts
    return out


def main():
    results = []
    only = sys.argv[2:]
    for label, kw in CANDIDATES:
        if only and label not in only:
            continue
        res = run(label, kw)
        results.append(res)
        print(json.dumps({k: res.get(k) for k in ("label", "class", "pre_tokenizer", "decode_say_hello", "load_error")}),
              flush=True)
        for name, c in (res.get("chat_counts") or {}).items():
            print("   ", name, json.dumps(c, ensure_ascii=True)[:300], flush=True)
    with open(sys.argv[1], "w") as f:
        json.dump(results, f, indent=1, ensure_ascii=True)


if __name__ == "__main__":
    main()
