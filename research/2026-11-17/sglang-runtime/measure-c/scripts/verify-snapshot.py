"""Compare a downloaded snapshot folder with the Hugging Face API's tree at the same revision.

Usage: python3 -I verify-snapshot.py <tree.json> <snapshot-dir> [--hash]
Checks every file the tree lists (name and size, following the cache's links) and lists extra files.
With --hash, LFS files are checked against their SHA-256 (lfs.oid) and the rest against their git blob id (oid).
Prints the total and exits 1 on any difference.
"""
import hashlib
import json
import os
import sys
import time

tree = [t for t in json.load(open(sys.argv[1])) if t.get("type") == "file"]
snap = sys.argv[2]
do_hash = "--hash" in sys.argv
problems = []
total = 0
t0 = time.time()
for t in tree:
    path = os.path.join(snap, t["path"])
    if not os.path.exists(path):
        problems.append(f"missing {t['path']}")
        continue
    size = os.stat(path).st_size
    total += size
    if size != t["size"]:
        problems.append(f"{t['path']}: size {size}, API {t['size']}")
        continue
    if do_hash:
        lfs = t.get("lfs")
        h = hashlib.sha256() if lfs else hashlib.sha1(b"blob %d\0" % size)
        with open(path, "rb") as f:
            while chunk := f.read(1 << 24):
                h.update(chunk)
        want = lfs["oid"] if lfs else t["oid"]
        if h.hexdigest() != want:
            problems.append(f"{t['path']}: {'sha256' if lfs else 'git blob id'} {h.hexdigest()}, API {want}")
listed = {t["path"] for t in tree}
for root, dirs, files in os.walk(snap):
    for name in files:
        rel = os.path.relpath(os.path.join(root, name), snap)
        if rel not in listed:
            problems.append(f"extra file {rel}")
lfs_n = sum(1 for t in tree if t.get("lfs"))
print(f"{len(tree)} files in the API tree ({lfs_n} LFS), {total} bytes = {total / 2**30:.2f} GiB in the snapshot; "
      f"{'hashed' if do_hash else 'names and sizes only'} in {time.time() - t0:.0f} s; "
      f"{'no difference' if not problems else str(len(problems)) + ' problem(s)'}")
for p in problems:
    print("  " + p)
sys.exit(1 if problems else 0)
