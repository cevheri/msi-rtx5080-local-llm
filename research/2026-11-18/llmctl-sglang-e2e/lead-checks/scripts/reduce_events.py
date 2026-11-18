"""Shorten an `opencode run --format json` event stream for the record. Run with python3 -I.

Usage: reduce_events.py <events.jsonl> <work folder> <shown work folder> [<max chars>]
Every string longer than <max chars> (default 400) becomes "[<n> chars, sha256 <first 12 hex>]", so a read tool's
output, which repeats a repository file, keeps only its size and hash; the work folder's path becomes the shown name.
"""
import hashlib
import json
import sys


def shorten(value, limit: int):
    if isinstance(value, str):
        if len(value) > limit:
            digest = hashlib.sha256(value.encode()).hexdigest()[:12]
            return f"[{len(value)} chars, sha256 {digest}]"
        return value
    if isinstance(value, list):
        return [shorten(item, limit) for item in value]
    if isinstance(value, dict):
        return {key: shorten(item, limit) for key, item in value.items()}
    return value


def main() -> None:
    path, work, shown = sys.argv[1], sys.argv[2], sys.argv[3]
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else 400
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        event = json.loads(line.replace(work, shown))
        print(json.dumps(shorten(event, limit), ensure_ascii=False))


if __name__ == "__main__":
    main()
