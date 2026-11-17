#!/usr/bin/env python3
"""Replace the lines between llmctl's two marker lines in a dsh patch file, keeping every byte outside them.
usage: rewrite_block.py <file> <line>...   (each <line> is one full block line)"""
import os, sys
BEGIN = '# >>> llmctl managed block: rewritten by llmctl on every start; edits here are lost >>>'
END = '# <<< llmctl managed block <<<'
path, lines = sys.argv[1], sys.argv[2:]
text = open(path, encoding='utf-8').read()
rows = text.split('\n')
b = [i for i, r in enumerate(rows) if r == BEGIN]
e = [i for i, r in enumerate(rows) if r == END]
if len(b) != 1 or len(e) != 1 or b[0] > e[0]:
    raise SystemExit(f'{path}: llmctl markers damaged (begin {b}, end {e})')
new = '\n'.join(rows[:b[0] + 1] + lines + rows[e[0]:])
assert new.startswith('\n'.join(rows[:b[0] + 1])) and new.endswith('\n'.join(rows[e[0]:]))
mode = os.stat(path).st_mode & 0o777
tmp = path + '.llmctl-tmp'
with open(tmp, 'w', encoding='utf-8') as f:
    f.write(new); f.flush(); os.fsync(f.fileno())
os.chmod(tmp, mode)
os.replace(tmp, path)
print('wrote', path)
