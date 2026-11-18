"""Turn an SGLang server log into one row per chat request: its prefill lines joined. Run with python3 -I.

Usage: request_table.py <docker logs --timestamps file>
SGLang v0.5.23 logs a long prompt as several "Prefill batch" lines of at most --chunked-prefill-size new tokens; the
first carries the radix-cache hit as #cached-token, the later ones 0, and the access line of a streamed request comes
after its prefill, when the first token is out. So the prompt of a request is the first line's #cached-token plus the
#new-token of every prefill line since the previous access line; a request that SGLang refuses before scheduling has
an access line and no prefill line.
"""
import re
import sys

PREFILL = re.compile(r"^(\S+) .*Prefill batch, #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+)")
ACCESS = re.compile(r'^(\S+) .*INFO: +\S+ - "(POST|GET) (\S+) HTTP/1\.1" (\d+)')


def main() -> None:
    chunks: list[tuple[str, int, int]] = []
    number = 0
    print("| # | First prefill line (UTC) | Access line (UTC) | Status | Prefill lines | Cached | New | Prompt |")
    print("|---|---|---|---|---|---|---|---|")
    for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
        match = PREFILL.match(line)
        if match:
            chunks.append((match.group(1)[11:23], int(match.group(3)), int(match.group(4))))
            continue
        match = ACCESS.match(line)
        if not match or match.group(3) != "/v1/chat/completions":
            continue
        number += 1
        cached = chunks[0][2] if chunks else 0
        new = sum(chunk[1] for chunk in chunks)
        first = chunks[0][0] if chunks else "-"
        print(f"| {number} | {first} | {match.group(1)[11:23]} | {match.group(4)} | {len(chunks)} | {cached} | {new} | "
              f"{cached + new if chunks else '-'} |")
        chunks = []


if __name__ == "__main__":
    main()
