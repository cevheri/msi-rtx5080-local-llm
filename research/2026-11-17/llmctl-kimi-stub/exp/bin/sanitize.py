"""Replace machine details in kept records: scratch root, home folder, host name, user name, bearer tokens.

usage: python3 -I sanitize.py <scratch root> <home> <host> <user> <dir>...   (rewrites text files in place)
"""

import re
import sys
from pathlib import Path

SCRATCH, HOME, HOST, USER = sys.argv[1:5]
TOKENS = [
    (re.compile(r"(Authorization: Bearer )[A-Za-z0-9._~+/=-]{16,}"), r"\1<redacted>"),
    (re.compile(r"(#token=)[A-Za-z0-9._~+/=-]+"), r"\1<redacted>"),
    (re.compile(r"(Token:\s+)[A-Za-z0-9._~+/=-]{16,}"), r"\1<redacted>"),
]
changed = 0
for root in sys.argv[5:]:
    for path in sorted(Path(root).rglob("*")):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        new = text.replace(SCRATCH, "$SCRATCH").replace(HOME, "$HOME").replace(HOST, "<host>").replace(USER, "<user>")
        for pattern, repl in TOKENS:
            new = pattern.sub(repl, new)
        if new != text:
            path.write_text(new, encoding="utf-8")
            changed += 1
print(f"rewrote {changed} files")
