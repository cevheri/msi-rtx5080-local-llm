"""Build a model overlay folder the way llmctl would: for every file of a pinned Hugging Face snapshot, a symbolic link
to that file's path inside the container, except generation_config.json, which is a real file: the snapshot's own
generation_config.json with the given keys added or replaced, written atomically.
Subfolders are recreated as real folders holding links, so every file is a link.
Run with python3 -I: make_overlay.py --snapshot HOST_SNAPSHOT_DIR --container-snapshot CONTAINER_DIR --out DIR
                                     --set temperature=0.6 --set top_p=0.95 --set top_k=20"""
import argparse
import json
import os
import shutil
import tempfile


def parse_value(text):
    try:
        return json.loads(text)
    except ValueError:
        return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--container-snapshot", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    a = ap.parse_args()
    if os.path.lexists(a.out):
        shutil.rmtree(a.out)
    os.makedirs(a.out)
    links = 0
    for root, dirs, files in os.walk(a.snapshot):
        rel_root = os.path.relpath(root, a.snapshot)
        out_root = a.out if rel_root == "." else os.path.join(a.out, rel_root)
        for d in sorted(dirs):
            os.makedirs(os.path.join(out_root, d), exist_ok=True)
        for name in sorted(files):
            rel = name if rel_root == "." else os.path.join(rel_root, name)
            if rel == "generation_config.json":
                continue
            os.symlink(os.path.join(a.container_snapshot, rel), os.path.join(out_root, name))
            links += 1
    source = os.path.join(a.snapshot, "generation_config.json")
    config = json.load(open(source)) if os.path.exists(source) else {}
    before = dict(config)
    for item in a.set:
        key, _, value = item.partition("=")
        config[key] = parse_value(value)
    fd, tmp = tempfile.mkstemp(dir=a.out, prefix=".generation_config.", suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")
    os.chmod(tmp, 0o644)
    os.replace(tmp, os.path.join(a.out, "generation_config.json"))
    print(json.dumps({"links": links, "generation_config_source_exists": os.path.exists(source),
                      "before": before, "after": config}, indent=1))


if __name__ == "__main__":
    main()
