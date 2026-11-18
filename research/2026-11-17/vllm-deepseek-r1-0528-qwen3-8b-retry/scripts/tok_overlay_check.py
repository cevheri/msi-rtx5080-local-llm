"""Load the DeepSeek-R1-0528-Qwen3-8B tokenizer from a folder whose tokenizer_config.json names PreTrainedTokenizerFast,
the way vLLM v0.33.0 loads a --tokenizer path, and count the three fixed chat prompts against the llama.cpp record.
Run inside vllm/vllm-openai:v0.33.0, CPU only: python3 tok_overlay_check.py FOLDER OUT.json"""
import json
import sys

from vllm.tokenizers import get_tokenizer

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
MSGS = {"ok": "Reply with exactly: OK", "probe_8k": PARAGRAPH * 84 + TAIL, "probe_30k": PARAGRAPH * 315 + TAIL}
LLAMACPP = {"ok": 8, "probe_8k": 8078, "probe_30k": 30254}
tok = get_tokenizer(sys.argv[1], tokenizer_mode="auto")
backend = getattr(tok, "backend_tokenizer", None) or getattr(tok, "_tokenizer", None)
res = {"class": type(tok).__name__, "pre_tokenizer": type(backend.pre_tokenizer).__name__ if backend else None}
ids = tok.encode("Say hello world.", add_special_tokens=False)
res["say_hello"] = {"ids": ids, "decode": tok.decode(ids)}
res["chat_counts"] = {}
for k, v in MSGS.items():
    enc = tok.apply_chat_template([{"role": "user", "content": v}], add_generation_prompt=True, tokenize=True)
    enc = enc["input_ids"] if hasattr(enc, "keys") else enc
    res["chat_counts"][k] = {"tokens": len(enc), "llamacpp": LLAMACPP[k], "diff": len(enc) - LLAMACPP[k]}
res["rendered_ok"] = tok.apply_chat_template([{"role": "user", "content": MSGS["ok"]}], add_generation_prompt=True, tokenize=False)
print(json.dumps(res, ensure_ascii=True))
json.dump(res, open(sys.argv[2], "w"), indent=1, ensure_ascii=True)
