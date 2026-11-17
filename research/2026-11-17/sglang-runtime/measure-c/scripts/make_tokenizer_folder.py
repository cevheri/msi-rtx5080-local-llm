"""Build the DeepSeek-R1 tokenizer folder as llmctl will ship it (the lead's message of 2026-11-18, plan ruling 35):
every file of the pinned snapshot as a symbolic link to its container path
/root/.cache/huggingface/hub/<repo dir>/snapshots/<revision>/<relative path>, subfolders recreated as real folders
(os.walk in sorted order), and one real file, tokenizer_config.json: the snapshot's JSON object with "tokenizer_class"
set to "PreTrainedTokenizerFast" (replaced where it is, appended where it is not), written as
json.dumps(config, indent=2, ensure_ascii=False) + "\n", mode 0644. Prints the file list (links and the real file).
Run with python3 -I: make_tokenizer_folder.py --snapshot HOST_SNAPSHOT_DIR --container-snapshot CONTAINER_DIR --out DIR"""
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
listing = []
for root, dirs, files in os.walk(a.snapshot):
    dirs.sort()
    rel_root = os.path.relpath(root, a.snapshot)
    out_root = a.out if rel_root == "." else os.path.join(a.out, rel_root)
    for d in dirs:
        os.makedirs(os.path.join(out_root, d), exist_ok=True)
    for name in sorted(files):
        rel = name if rel_root == "." else os.path.join(rel_root, name)
        if rel == "tokenizer_config.json":
            continue
        target = os.path.join(a.container_snapshot, rel)
        os.symlink(target, os.path.join(out_root, name))
        listing.append({"path": rel, "link": target})
with open(os.path.join(a.snapshot, "tokenizer_config.json"), encoding="utf-8") as f:
    config = json.load(f)
if not isinstance(config, dict):
    raise SystemExit("the snapshot's tokenizer_config.json is not a JSON object")
before = config.get("tokenizer_class")
config["tokenizer_class"] = "PreTrainedTokenizerFast"
path = os.path.join(a.out, "tokenizer_config.json")
with open(path, "w", encoding="utf-8") as f:
    f.write(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    f.flush()
    os.fsync(f.fileno())
os.chmod(path, 0o644)
listing.append({"path": "tokenizer_config.json", "real_file": True, "tokenizer_class_before": before,
                "tokenizer_class_after": "PreTrainedTokenizerFast", "bytes": os.path.getsize(path)})
print(json.dumps({"links": sum(1 for x in listing if "link" in x), "files": listing}, indent=1))
