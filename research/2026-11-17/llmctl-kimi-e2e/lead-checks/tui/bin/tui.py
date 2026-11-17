"""Drive Kimi's TUI through a pseudo-terminal, inside the sandbox.

usage: python3 -I tui.py <run dir> <steps.json> -- <argv...>
Steps (a JSON list), each one of:
  ["wait", regex, seconds]      wait until the ANSI-stripped screen text matches (records a miss, goes on)
  ["send", text]                write keys ("\\r" Enter, "\\u001b" Escape, "\\u0003" Ctrl+C)
  ["type", text]                write text one character at a time
  ["sleep", seconds]
  ["copy", src, name]           copy a file into the run dir under name
  ["replace", src, dst]         os.replace(src, dst) (an atomic rename, as llmctl writes)
  ["mark", label]               put a marker into the transcript
  ["touch", path]               create a file (a signal to the outside driver)
  ["waitfile", path, seconds]   wait until a file exists (a signal from the outside driver)
Writes <run dir>/tui-transcript.txt (ANSI stripped) and <run dir>/tui-steps.jsonl.
"""

import fcntl
import json
import os
import pty
import re
import select
import shutil
import signal
import struct
import sys
import termios
import time

ANSI = re.compile(r"\x1b\[[0-9;?<>=]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[()][0-9A-Za-z]|\x1b[=>78DMEc]")
RUN = sys.argv[1]
STEPS = json.load(open(sys.argv[2]))
ARGV = sys.argv[sys.argv.index("--") + 1:]

pid, fd = pty.fork()
if pid == 0:
    os.execvp(ARGV[0], ARGV)
fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 140, 0, 0))
raw = []
log = open(os.path.join(RUN, "tui-steps.jsonl"), "w")


def pump(seconds):
    end = time.time() + seconds
    while time.time() < end:
        ready, _, _ = select.select([fd], [], [], 0.05)
        if fd in ready:
            try:
                data = os.read(fd, 65536)
            except OSError:
                return False
            if not data:
                return False
            text = data.decode("utf-8", "replace")
            raw.append(text)
            # answer a cursor position query, which some TUIs wait for
            if "\x1b[6n" in text:
                os.write(fd, b"\x1b[1;1R")
    return True


def screen():
    return ANSI.sub("", "".join(raw))


t0 = time.time()
for step in STEPS:
    kind = step[0]
    rec = {"t": round(time.time() - t0, 2), "step": step}
    if kind == "wait":
        end = time.time() + step[2]
        hit = False
        while time.time() < end:
            pump(0.2)
            if re.search(step[1], screen()):
                hit = True
                break
        rec["hit"] = hit
    elif kind == "send":
        os.write(fd, step[1].encode())
        pump(0.3)
    elif kind == "type":
        for ch in step[1]:
            os.write(fd, ch.encode())
            pump(0.05)
    elif kind == "sleep":
        pump(step[1])
    elif kind == "copy":
        try:
            shutil.copyfile(os.path.expandvars(step[1]), os.path.join(RUN, step[2]))
            rec["ok"] = True
        except OSError as exc:
            rec["error"] = repr(exc)
    elif kind == "replace":
        os.replace(os.path.expandvars(step[1]), os.path.expandvars(step[2]))
    elif kind == "touch":
        open(os.path.expandvars(step[1]), "w").close()
    elif kind == "waitfile":
        end = time.time() + step[2]
        hit = False
        while time.time() < end:
            pump(0.5)
            if os.path.exists(os.path.expandvars(step[1])):
                hit = True
                break
        rec["hit"] = hit
    elif kind == "mark":
        raw.append(f"\n<<<MARK {step[1]}>>>\n")
    log.write(json.dumps(rec) + "\n")
    log.flush()

pump(0.5)
try:
    os.kill(pid, signal.SIGTERM)
except ProcessLookupError:
    pass
for _ in range(50):
    done, status = os.waitpid(pid, os.WNOHANG)
    if done:
        break
    pump(0.1)
else:
    os.kill(pid, signal.SIGKILL)
    os.waitpid(pid, 0)
open(os.path.join(RUN, "tui-transcript.txt"), "w").write(screen())
print("tui done", round(time.time() - t0, 1), "s")
