#!/usr/bin/env python3
"""S8: boot `dsh web --port 3182 --no-open` in a fresh DSH_HOME with a given home patch, wait for the `dsh web:` line,
read stderr for activation warnings, then stop it with SIGTERM. Never prints the launch token.

s8_web_boot.py <case> <home patch file>
"""
import os
import re
import select
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

if not os.environ.get("SCRATCH"):
    raise SystemExit("s8_web_boot.py: set SCRATCH to a work folder; homes go under $SCRATCH/llmctl-dsh-stub-rerun/")
S = Path(os.environ["SCRATCH"]) / "llmctl-dsh-stub-rerun"
(S / "work").mkdir(parents=True, exist_ok=True)
case, patch = sys.argv[1], Path(sys.argv[2])
home = Path(tempfile.mkdtemp(prefix=f"home-{case}.", dir=S))
agents = Path(tempfile.mkdtemp(prefix=f"agents-{case}.", dir=S))
work = Path(tempfile.mkdtemp(prefix=f"w-{case}.", dir=S / "work"))
(home / "cordis.patch.yml").write_text(patch.read_text(encoding="utf-8"), encoding="utf-8")
before = set(os.listdir("/tmp"))
env = dict(os.environ, DSH_HOME=str(home), DSH_AGENTS_HOME=str(agents), DSH_TELEMETRY_DISABLED="1", PWD=str(work))
start = time.monotonic()
proc = subprocess.Popen(["heavy", "dsh", "web", "--port", "3182", "--no-open"], cwd=work, env=env,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
ready, out_lines, err_lines = None, [], []
deadline = start + 40
while time.monotonic() < deadline:
    readable, _, _ = select.select([proc.stdout, proc.stderr], [], [], 0.5)
    for stream in readable:
        line = stream.readline()
        if not line:
            continue
        (out_lines if stream is proc.stdout else err_lines).append(line)
        if stream is proc.stdout and line.startswith("dsh web:") and ready is None:
            ready = time.monotonic() - start
    if ready is not None and time.monotonic() - start > ready + 3:
        break
    if proc.poll() is not None:
        break
os.killpg(proc.pid, signal.SIGTERM)
try:
    proc.wait(timeout=15)
except subprocess.TimeoutExpired:
    os.killpg(proc.pid, signal.SIGKILL)
    proc.wait()
rest_out, rest_err = proc.communicate()
err_lines += rest_err.splitlines(keepends=True)
after = set(os.listdir("/tmp"))
redact = lambda text: re.sub(r"token=[^\s&]+", "token=<redacted>", text).replace(str(home), "$H")
print(f"{case}: ready after {ready if ready is None else round(ready, 1)} s, exit {proc.returncode}")
print("stdout:", redact("".join(out_lines + [rest_out]).strip())[:200])
print("stderr:", redact("".join(err_lines).strip())[:1500] or "(empty)")
print("new /tmp entries:", sorted(name for name in after - before if name.startswith("dsh")))
