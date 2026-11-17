"""Replace local paths and names in the lane's text files: the scratchpad path with $SCRATCH, the home path with $HOME,
and any remaining user or host name with <user> and <host>. Symbolic links are left alone (they point at container
paths). Run with python3 -I: scrub.py SCRATCH HOME USER HOST DIR [DIR ...]"""
import os
import sys

scratch, home, user, host = sys.argv[1:5]
changed = 0
for top in sys.argv[5:]:
    for root, dirs, files in os.walk(top):
        for name in files:
            path = os.path.join(root, name)
            if os.path.islink(path):
                continue
            try:
                with open(path, "rb") as f:
                    data = f.read()
            except OSError:
                continue
            if b"\0" in data:
                continue
            text = data.decode("utf-8", errors="surrogateescape")
            new = text.replace(scratch, "$SCRATCH").replace(home, "$HOME")
            if user:
                new = new.replace(user, "<user>")
            if host:
                new = new.replace(host, "<host>")
            if new != text:
                with open(path, "wb") as f:
                    f.write(new.encode("utf-8", errors="surrogateescape"))
                changed += 1
print(f"{changed} files changed")
