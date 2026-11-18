"""Rebuild measurement B's DeepSeek-R1 tokenizer folder (retry 2) for measurement C: links to the pinned snapshot's
config.json and tokenizer.json at their container paths, and a real tokenizer_config.json that is the snapshot's file
with its one "tokenizer_class": "LlamaTokenizerFast" line changed to "PreTrainedTokenizerFast", every other byte kept.
Run with python3 -I: make_tokenizer_overlay.py --snapshot HOST_SNAPSHOT_DIR --container-snapshot CONTAINER_DIR --out DIR"""
import argparse
import json
import os
import shutil

ap = argparse.ArgumentParser()
ap.add_argument("--snapshot", required=True)
ap.add_argument("--container-snapshot", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
if os.path.lexists(a.out):
    shutil.rmtree(a.out)
os.makedirs(a.out)
for name in ("config.json", "tokenizer.json"):
    os.symlink(os.path.join(a.container_snapshot, name), os.path.join(a.out, name))
text = open(os.path.join(a.snapshot, "tokenizer_config.json"), encoding="utf-8").read()
old, new = '"tokenizer_class": "LlamaTokenizerFast"', '"tokenizer_class": "PreTrainedTokenizerFast"'
if text.count(old) != 1:
    raise SystemExit(f"expected one {old!r} in the snapshot's tokenizer_config.json, found {text.count(old)}")
out = text.replace(old, new)
json.loads(out)
with open(os.path.join(a.out, "tokenizer_config.json"), "w", encoding="utf-8") as f:
    f.write(out)
os.chmod(os.path.join(a.out, "tokenizer_config.json"), 0o644)
print(json.dumps({"links": ["config.json", "tokenizer.json"], "changed": [old, new],
                  "tokenizer_class": json.loads(out)["tokenizer_class"]}))
