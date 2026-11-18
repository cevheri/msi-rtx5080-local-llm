"""Offline check: can openai_harmony load the gpt-oss encoding (o200k_base vocab) without network? Prints the time and result.
Run inside the image with --network none: python3 -I /probe/harmony_check.py"""
import os
import time

from openai_harmony import HarmonyEncodingName, load_harmony_encoding

t0 = time.time()
try:
    enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    result = f"loaded, {len(enc.stop_tokens_for_assistant_actions())} stop tokens"
except Exception as err:  # the probe records the failure
    result = f"{type(err).__name__}: {str(err)[:300]}"
print(f"TIKTOKEN_ENCODINGS_BASE={os.environ.get('TIKTOKEN_ENCODINGS_BASE')!r} seconds={time.time() - t0:.2f} result={result}")
