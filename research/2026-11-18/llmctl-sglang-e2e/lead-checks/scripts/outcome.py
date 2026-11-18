"""Judge one OpenCode session the way the e2e driver does (launcher/e2e/clients.py). Run with python3 -I -B.

Usage: outcome.py <repository> <events.jsonl> <work folder> <exit status>
Prints the reasons the driver would give (opencode_outcome with tools, hello_valid, the exit status), or "pass".
"""
import pathlib
import sys

sys.path.insert(0, sys.argv[1])
from launcher.e2e import clients  # noqa: E402


def main() -> None:
    lines = pathlib.Path(sys.argv[2]).read_text(encoding="utf-8").splitlines()
    folder, code = pathlib.Path(sys.argv[3]), int(sys.argv[4])
    reasons = [] if code == 0 else [f"exit status {code}"]
    reasons += list(clients.opencode_outcome(lines, tools=True).reasons)
    if not clients.hello_valid(folder / "hello.py"):
        reasons.append("hello.py is missing or lacks def add and add(2, 3)")
    print("; ".join(reasons) if reasons else "pass")


if __name__ == "__main__":
    main()
