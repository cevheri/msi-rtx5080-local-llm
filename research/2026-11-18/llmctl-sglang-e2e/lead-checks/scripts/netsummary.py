"""Summarise socket-table samples: every distinct remote address per table and state, and whether each lies in the
allowed set. Run with python3 -I.

Usage: netsummary.py <sample file>...
Each sample is the output of `cat /proc/net/tcp /proc/net/tcp6 /proc/net/udp /proc/net/udp6`, named <UTC time>.txt;
the tables are told apart by their header line (UDP's ends in "drops") and by the address width, and the addresses
are decoded as in research/2026-11-17/sglang-runtime/handson/scripts/proc_net.py.
"""
import collections
import ipaddress
import pathlib
import sys

STATES = {"01": "ESTABLISHED", "02": "SYN_SENT", "06": "TIME_WAIT", "0A": "LISTEN", "08": "CLOSE_WAIT", "07": "CLOSE"}
ALLOWED = [ipaddress.ip_network(net) for net in ("127.0.0.1/32", "::1/128", "0.0.0.0/32", "172.17.0.0/24")]


def addr(hexaddr: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    host = hexaddr.split(":")[0]
    raw = bytes.fromhex(host)
    if len(raw) == 4:
        return ipaddress.IPv4Address(raw[::-1])
    return ipaddress.IPv6Address(b"".join(raw[i:i + 4][::-1] for i in range(0, 16, 4)))


def allowed(ip) -> str:
    """The allowed address the remote address is, or "" when it is none of them."""
    plain = ip
    if ip.version == 6 and ip.ipv4_mapped is not None:
        plain = ip.ipv4_mapped  # ::ffff:a.b.c.d is the IPv4 address a.b.c.d on a dual-stack socket
    if ip.version == 6 and ip == ipaddress.IPv6Address("::"):
        return "0.0.0.0 (the IPv6 unspecified address ::)"
    for net in ALLOWED:
        if plain.version == net.version and plain in net:
            return str(plain) if plain == ip else f"{plain} (written {ip})"
    return ""


def main() -> int:
    seen: dict[tuple[str, str, str], list[str]] = collections.defaultdict(list)
    files = sorted(sys.argv[1:])
    for path in files:
        stamp = pathlib.Path(path).stem
        proto = None
        for line in open(path, encoding="utf-8"):
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "sl":
                proto = "udp" if parts[-1] == "drops" else "tcp"
                continue
            ip = addr(parts[2])
            table = proto + ("6" if ip.version == 6 else "")
            seen[(table, STATES.get(parts[3], parts[3]), str(ip))].append(stamp)
    print(f"samples: {len(files)}, first {pathlib.Path(files[0]).stem}, last {pathlib.Path(files[-1]).stem}")
    bad = 0
    for (table, state, remote), stamps in sorted(seen.items()):
        verdict = allowed(ipaddress.ip_address(remote))
        bad += not verdict
        print(f"{table:5} {state:12} {remote:20} in {len(set(stamps)):3} samples, {stamps[0]} to {stamps[-1]}: "
              f"{verdict or 'NOT ALLOWED'}")
    print(f"remote addresses outside 127.0.0.1, ::1, 0.0.0.0 and 172.17.0.x: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
