"""Decode /proc/net/tcp and tcp6 dumps: print each distinct (state, remote address) pair with a count. Run with python3 -I."""
import collections
import ipaddress
import sys

STATES = {"01": "ESTABLISHED", "02": "SYN_SENT", "06": "TIME_WAIT", "0A": "LISTEN", "08": "CLOSE_WAIT", "07": "CLOSE"}


def addr(hexaddr):
    host, port = hexaddr.split(":")
    raw = bytes.fromhex(host)
    if len(raw) == 4:
        ip = ipaddress.IPv4Address(raw[::-1])
    else:
        ip = ipaddress.IPv6Address(b"".join(raw[i:i + 4][::-1] for i in range(0, 16, 4)))
    return f"{ip}:{int(port, 16)}"


for path in sys.argv[1:]:
    seen = collections.Counter()
    for line in open(path):
        parts = line.split()
        if len(parts) < 4 or parts[0] == "sl":
            continue
        remote = addr(parts[2]).rsplit(":", 1)[0]
        seen[(STATES.get(parts[3], parts[3]), remote)] += 1
    print(path.rsplit("/", 2)[-2] + "/" + path.rsplit("/", 1)[-1], dict(seen))
