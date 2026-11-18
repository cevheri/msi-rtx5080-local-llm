"""Sample the socket tables of the container `sglang` every 0.5 s until a stop file appears. Run with python3 -I.

Usage: netsample.py <out folder> <stop file>
Each successful `docker exec sglang cat /proc/net/tcp /proc/net/tcp6 /proc/net/udp /proc/net/udp6` is written to
<out folder>/raw/<UTC time>.txt; every attempt, failed ones included, goes to <out folder>/sampler.log.
"""
import datetime
import pathlib
import subprocess
import sys
import time

PERIOD_S = 0.5
CMD = ["docker", "exec", "sglang", "cat", "/proc/net/tcp", "/proc/net/tcp6", "/proc/net/udp", "/proc/net/udp6"]


def main() -> None:
    out, stop = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    with open(out / "sampler.log", "a", buffering=1, encoding="utf-8") as log:
        log.write("# UTC time, result, docker exec seconds, detail\n")
        tick = time.monotonic()
        while not stop.exists():
            stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%f")[:-3] + "Z"
            start = time.monotonic()
            try:
                done = subprocess.run(CMD, capture_output=True, text=True, timeout=10)
                code, stdout, stderr = done.returncode, done.stdout, done.stderr
            except subprocess.TimeoutExpired:
                code, stdout, stderr = None, "", "timeout"
            seconds = time.monotonic() - start
            if code == 0:
                (raw / f"{stamp}.txt").write_text(stdout, encoding="utf-8")
                log.write(f"{stamp} ok {seconds:.3f} {len(stdout.splitlines())} lines\n")
            else:
                log.write(f"{stamp} exit {code} {seconds:.3f} {stderr.strip()[:160]}\n")
            tick += PERIOD_S
            delay = tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                tick = time.monotonic()


if __name__ == "__main__":
    main()
