"""Compare the prompt that vLLM v0.33.0's deepseek_v32 tokenizer mode renders with the model's own chat template
(rendered with transformers' TokenizersBackend, which keeps the byte-level tokenizer) for a few conversations.
Run inside vllm/vllm-openai:v0.33.0, CPU only: python3 template_compare.py OUT.json"""
import json
import sys

from transformers import TokenizersBackend
from vllm.tokenizers import get_tokenizer

SNAP = "/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-0528-Qwen3-8B/snapshots/6e8885a6ff5c1dc5201574c8fd700323f23c25fa"
CONVS = {
    "single_user": [{"role": "user", "content": "Reply with exactly: OK"}],
    "system_user": [{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is 17 + 25?"}],
    "multi_turn": [{"role": "system", "content": "Be brief."},
                   {"role": "user", "content": "Hi"},
                   {"role": "assistant", "content": "Hello! How can I help?"},
                   {"role": "user", "content": "Say hello world."}],
    "multi_turn_with_reasoning": [{"role": "user", "content": "Hi"},
                                  {"role": "assistant", "content": "Hello!", "reasoning_content": "The user greets."},
                                  {"role": "user", "content": "Say hello world."}],
}
own = TokenizersBackend.from_pretrained(SNAP)
v32 = get_tokenizer(SNAP, tokenizer_mode="deepseek_v32")
out = {}
for name, conv in CONVS.items():
    a = own.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    b = v32.apply_chat_template(conv, add_generation_prompt=True, tokenize=False, thinking=True)
    ids_a = own.apply_chat_template(conv, add_generation_prompt=True, tokenize=True)
    ids_b = v32.apply_chat_template(conv, add_generation_prompt=True, tokenize=True, thinking=True)
    ids_a = ids_a["input_ids"] if hasattr(ids_a, "keys") else ids_a
    ids_b = ids_b["input_ids"] if hasattr(ids_b, "keys") else ids_b
    out[name] = {"own_template": a, "deepseek_v32_thinking": b, "identical": a == b,
                 "v32_equals_own_plus_think": b == a + "<think>", "own_tokens": len(ids_a), "v32_tokens": len(ids_b)}
    print(name, json.dumps({k: v for k, v in out[name].items() if k not in ("own_template", "deepseek_v32_thinking")}))
    if not out[name]["identical"]:
        print("   own:", repr(a)[:300])
        print("   v32:", repr(b)[:300])
json.dump(out, open(sys.argv[1], "w"), indent=1, ensure_ascii=True)
