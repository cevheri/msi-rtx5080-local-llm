"""usage: python3 -I tree_bytes.py <dir>...  -> JSON {dir: {"files": n, "bytes": n, "dirs": n}} (symlinks not followed)."""
import json, os, sys
out = {}
for root in sys.argv[1:]:
    files = size = dirs = 0
    for base, dnames, fnames in os.walk(root):
        dirs += len(dnames)
        for name in fnames:
            p = os.path.join(base, name)
            if os.path.islink(p):
                continue
            files += 1
            size += os.path.getsize(p)
    out[os.path.basename(root.rstrip("/")) or root] = {"files": files, "bytes": size, "dirs": dirs}
print(json.dumps(out))
