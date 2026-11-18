"""Rebuild the HTTP exchanges with one local port from strace logs. Run with python3 -I.

Usage: strace_http.py <port> <out folder> <strace log>...
The logs come from `strace -f -ff -tt -yy -s 100000000 -e trace=sendto,sendmsg,write,writev,recvfrom,recvmsg,read,readv
-o <prefix> <command>`, one file per thread; -yy names each socket with its two ends, so the bytes written to and read
from 127.0.0.1:<port> can be joined per connection, and -tt orders the calls of all threads.
Writes <out folder>/<n>-request.json (time, request line, headers and JSON body) and <n>-response.txt (the status line,
headers and body as received, chunked framing removed), numbered in the order the requests were sent.
"""
import codecs
import json
import pathlib
import re
import sys

CALL = re.compile(r'^(\d+:\d+:\d+\.\d+) (sendto|sendmsg|write|writev|recvfrom|recvmsg|read|readv)'
                  r'\((\d+)<TCP(?:v6)?:\[([^\]]+)\]>, (.*)\) += (-?\d+)')
STRING = re.compile(r'"((?:[^"\\]|\\.)*)"')


def unescape(text: str) -> bytes:
    return codecs.escape_decode(text.encode("latin-1"))[0]


def payload(call: str, rest: str, result: int) -> bytes:
    """The bytes one call moved: its quoted strings (an iov array holds several), cut to the return value."""
    if result <= 0:
        return b""
    data = b"".join(unescape(s) for s in STRING.findall(rest))
    return data[:result]


def dechunk(body: bytes) -> tuple[bytes, int]:
    """The body of a chunked message and the bytes it took, or (partial body, -1) when it is incomplete."""
    out, at = b"", 0
    while True:
        end = body.find(b"\r\n", at)
        if end < 0:
            return out, -1
        size = int(body[at:end].split(b";")[0] or b"0", 16)
        at = end + 2
        if size == 0:
            trailer = body.find(b"\r\n", at)
            return out, (trailer + 2 if trailer >= 0 else len(body))
        out += body[at:at + size]
        at += size + 2


def messages(stream: bytes) -> list[tuple[int, bytes, bytes]]:
    """Split a byte stream of HTTP/1.1 messages into (offset, head, body)."""
    found, at = [], 0
    while at < len(stream):
        sep = stream.find(b"\r\n\r\n", at)
        if sep < 0:
            break
        head = stream[at:sep]
        lines = head.split(b"\r\n")
        fields = {k.strip().lower(): v.strip() for k, _, v in (l.partition(b":") for l in lines[1:])}
        start, rest = at, sep + 4
        if b"content-length" in fields:
            size = int(fields[b"content-length"])
            body, at = stream[rest:rest + size], rest + size
        elif fields.get(b"transfer-encoding", b"").lower() == b"chunked":
            body, used = dechunk(stream[rest:])
            at = len(stream) if used < 0 else rest + used
        else:
            body, at = stream[rest:], len(stream)
        found.append((start, head, body))
    return found


def main() -> None:
    port, out, logs = sys.argv[1], pathlib.Path(sys.argv[2]), sys.argv[3:]
    out.mkdir(parents=True, exist_ok=True)
    calls = []
    for log in logs:
        for line in open(log, encoding="latin-1"):
            match = CALL.match(line)
            if match:
                calls.append(match.groups())
    calls.sort(key=lambda call: call[0])
    sent: dict[str, bytearray] = {}
    got: dict[str, bytearray] = {}
    stamps: dict[str, list[tuple[int, str]]] = {}
    for stamp, call, _, ends, rest, result in calls:
        local, _, remote = ends.partition("->")
        if remote.rsplit(":", 1)[-1] != port:
            continue
        data = payload(call, rest, int(result))
        if not data:
            continue
        if call.startswith(("send", "write")):
            stamps.setdefault(ends, []).append((len(sent.get(ends, b"")), stamp))
            sent.setdefault(ends, bytearray()).extend(data)
        else:
            got.setdefault(ends, bytearray()).extend(data)
    exchanges = []
    for ends, stream in sent.items():
        replies = messages(bytes(got.get(ends, b"")))
        for index, (offset, head, body) in enumerate(messages(bytes(stream))):
            stamp = [s for o, s in stamps[ends] if o <= offset][-1]
            exchanges.append((stamp, ends, head, body, replies[index] if index < len(replies) else None))
    exchanges.sort(key=lambda item: item[0])
    for number, (stamp, ends, head, body, reply) in enumerate(exchanges, 1):
        try:
            shown = json.loads(body) if body else None
        except ValueError:
            shown = body.decode(errors="replace")
        record = {"n": number, "time": stamp, "connection": ends,
                  "head": head.decode(errors="replace").split("\r\n"), "body": shown}
        (out / f"{number:03d}-request.json").write_text(json.dumps(record, indent=1, ensure_ascii=False))
        text = b"(no reply)" if reply is None else reply[1] + b"\r\n\r\n" + reply[2]
        (out / f"{number:03d}-response.txt").write_bytes(text)
    print(f"{len(exchanges)} exchanges with port {port}")


if __name__ == "__main__":
    main()
