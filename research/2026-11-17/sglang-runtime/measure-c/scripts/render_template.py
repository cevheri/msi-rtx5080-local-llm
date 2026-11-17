"""Render a snapshot's chat template offline with transformers (CPU only), for one user message with and without the
get_weather tool and with enable_thinking false, and print the prompt text and its token count.
Run inside the SGLang image: python3 -I render_template.py <snapshot-dir>"""
import json
import sys

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(sys.argv[1])
tool = {"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
msgs = [{"role": "user", "content": "What is the weather in Berlin? Use the tool."}]
for label, kw in (("with tools", {"tools": [tool]}), ("without tools", {})):
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False, **kw)
    ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False, **kw)
    if isinstance(ids, dict):
        ids = ids["input_ids"]
    print(f"===== {label}: {len(ids)} tokens")
    print(text)
