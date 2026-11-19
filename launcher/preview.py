"""python3 -m launcher.preview: build the model guide from this checkout, serve it on 127.0.0.1 and rebuild it when
a source changes, so a change shows in the browser within seconds (docs/site.md). Run it through heavy.

It lives outside launcher/site on purpose: the site generator reads no environment and starts no process, while the
preview runs git, a fresh Python for every build and a local web server."""

import argparse
import http.server
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from launcher.site import cli, data

REPO = cli.REPO
SOURCES = ("README.md", "docs", "recipes", "research", "launcher")
SKIP = {"__pycache__", "superpowers"}
CHILD = ("import sys\nfrom pathlib import Path\nfrom launcher.site import cli\n"
         "sys.exit(cli.main(['build', '--out', sys.argv[1], '--commit', sys.argv[2], '--commit-time', sys.argv[3], "
         "'--repository', sys.argv[4]], root=Path(sys.argv[5])))")


def watched(root: Path) -> tuple[tuple[str, int], ...]:
    """The modification time of every file the site may be built from, sorted by path; bytecode and the git-ignored
    docs/superpowers folder are left out, so a build's own .pyc files never start another build."""
    seen = []
    for name in SOURCES:
        path = root / name
        if path.is_file():
            seen.append((name, path.stat().st_mtime_ns))
            continue
        for folder, subfolders, names in os.walk(path):
            subfolders[:] = [sub for sub in subfolders if sub not in SKIP]
            for name_ in names:
                file = Path(folder) / name_
                try:
                    seen.append((file.relative_to(root).as_posix(), file.stat().st_mtime_ns))
                except FileNotFoundError:  # removed while the walk ran; the next poll sees that change
                    continue
    return tuple(sorted(seen))


def head(root: Path, repository: str) -> data.Build:
    """HEAD's commit and commit time, which the preview's footer and source links use."""
    try:
        commit = _git(root, "rev-parse", "HEAD")
        when = _git(root, "log", "-1", "--format=%ct")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise data.SiteError(f"preview: git could not name HEAD in {root}: {exc}") from None
    return data.Build(repository, commit, int(when))


def _stop(signum: int, frame: object) -> None:
    raise KeyboardInterrupt


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True).stdout.strip()


def build_into(root: Path, facts: data.Build) -> Path:
    """Build the site into a new temporary folder with a fresh Python, so an edit to the generator's own code shows
    too. The folder; a failed build raises SiteError with the build's message and leaves no folder behind."""
    out = Path(tempfile.mkdtemp(prefix="model-guide-"))
    done = subprocess.run([sys.executable, "-B", "-c", CHILD, str(out), facts.commit, str(facts.commit_time),
                           facts.repository, str(root)], cwd=REPO, capture_output=True, text=True)
    if done.returncode != 0:
        shutil.rmtree(out)
        raise data.SiteError(done.stderr.strip().removeprefix("site: ") or f"the build exited with {done.returncode}")
    return out


def serve(root: Path, port: int, repository: str) -> None:
    """Serve the latest build that worked until Ctrl+C or SIGTERM, rebuilding a second after any source changes."""
    signal.signal(signal.SIGINT, signal.default_int_handler)  # a shell that starts it in the background ignores SIGINT
    signal.signal(signal.SIGTERM, _stop)
    facts = head(root, repository)
    current = [build_into(root, facts)]

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(current[0]), **kwargs)

        def log_message(self, format, *args):  # a line per request would bury the rebuild messages
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"preview: http://127.0.0.1:{port}/ (it rebuilds when a source changes; Ctrl+C stops it)", flush=True)
    seen = watched(root)
    try:
        while True:
            time.sleep(1)
            now = watched(root)
            if now == seen:
                continue
            seen = now
            try:
                fresh = build_into(root, facts)
            except data.SiteError as exc:
                print(f"site: {exc}\npreview: still showing the last build that worked", flush=True)
                continue
            old, current[0] = current[0], fresh
            shutil.rmtree(old)
            print(f"preview: rebuilt at {time.strftime('%H:%M:%S')}", flush=True)
    except KeyboardInterrupt:
        print(flush=True)
    finally:
        server.shutdown()
        server.server_close()
        shutil.rmtree(current[0])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m launcher.preview",
                                     description="Serve the model guide from this checkout and rebuild it on change.")
    parser.add_argument("--port", type=int, default=8700, help="the port on 127.0.0.1 (default 8700)")
    parser.add_argument("--repository", default=cli.REPOSITORY, help=f"the GitHub repository (default {cli.REPOSITORY})")
    args = parser.parse_args(argv)
    try:
        serve(REPO, args.port, args.repository)
    except data.SiteError as exc:
        print(f"site: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:  # the port is taken, or the temporary folder cannot be written
        print(f"preview: {exc.strerror or exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
