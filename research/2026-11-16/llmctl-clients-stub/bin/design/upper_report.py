#!/usr/bin/env python3
"""List what a sandboxed run wrote into the ~/.hermes overlay: each path in the upper layer is
NEW (absent from the real ~/.hermes), MODIFIED (exists in the real tree: overlayfs copied it up,
so the run would have changed the real file), or DELETED (a whiteout). Directories that only
exist because something below them was written are skipped.
usage: upper_report.py <upper-dir> [--max N]"""
import os
import stat
import sys
from pathlib import Path

REAL = Path("$HOME/.hermes")


def main():
    upper = Path(sys.argv[1])
    limit = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[2] == "--max" else 400
    rows = []
    for dirpath, dirnames, filenames in os.walk(upper):
        for name in sorted(filenames) + sorted(d for d in dirnames if os.path.islink(os.path.join(dirpath, d))):
            p = Path(dirpath) / name
            rel = p.relative_to(upper)
            st = p.lstat()
            real = REAL / rel
            if stat.S_ISCHR(st.st_mode) and st.st_rdev == 0:
                kind = "DELETED"
            elif os.path.lexists(real):
                kind = "MODIFIED"
            else:
                kind = "NEW"
            rows.append((kind, str(rel), st.st_size))
    for kind in ("MODIFIED", "DELETED", "NEW"):
        sel = [r for r in rows if r[0] == kind]
        print(f"{kind}: {len(sel)}")
        for _, rel, size in sel[:limit]:
            print(f"  {rel} ({size} B)")


if __name__ == "__main__":
    main()
