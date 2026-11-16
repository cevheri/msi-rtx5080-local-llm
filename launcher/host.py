"""Everything llmctl does outside its own process goes through a Host.

RealHost talks to the machine; the tests use a fake with the same methods.
"""

import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TextIO

from launcher.errors import DownloadError

CHUNK = 1 << 20


@dataclass(frozen=True)
class Result:
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class HttpResponse:
    status: int
    headers: dict[str, str]  # lower-case names
    body: bytes


class Host(Protocol):
    env: Mapping[str, str]
    out: TextIO
    err: TextIO

    def run(self, argv: list[str], *, input: str | None = None,
            env_extra: Mapping[str, str] | None = None) -> Result: ...

    def run_attached(self, argv: list[str]) -> int: ...

    def http(self, method: str, url: str, *, headers: Mapping[str, str] | None = None,
             body: bytes | None = None, timeout: float = 10.0,
             follow_redirects: bool = True) -> HttpResponse: ...

    def download(self, url: str, headers: Mapping[str, str], dest: Path, offset: int,
                 on_chunk: Callable[[bytes], None]) -> None: ...

    def port_open(self, port: int) -> bool: ...

    def uid_gid(self) -> tuple[int, int]: ...

    def isatty(self) -> bool: ...

    def ask(self, prompt: str) -> str: ...

    def monotonic(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...

    def meminfo(self) -> str: ...

    def cdi_specs(self) -> list[str]: ...

    def which(self, name: str) -> str | None: ...


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class _SameHostAuth(urllib.request.HTTPRedirectHandler):
    """Follow redirects, but drop the Authorization header when one leads to another host (for example a CDN)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlsplit(newurl).netloc != urllib.parse.urlsplit(req.full_url).netloc:
            new.remove_header("Authorization")
        return new


def _lower(headers) -> dict[str, str]:
    return {name.lower(): value for name, value in headers.items()}


class RealHost:
    def __init__(self, cdi_dirs=(Path("/etc/cdi"), Path("/var/run/cdi")), env=None,
                 out: TextIO = sys.stdout, err: TextIO = sys.stderr):
        self.cdi_dirs = tuple(cdi_dirs)
        self.env = dict(os.environ if env is None else env)
        self.out = out
        self.err = err

    def run(self, argv, *, input=None, env_extra=None):
        env = None
        if env_extra:
            env = dict(os.environ)
            env.update(env_extra)
        try:
            done = subprocess.run(argv, input=input, capture_output=True, text=True, env=env)
        except FileNotFoundError:
            return Result(127, "", f"{argv[0]}: command not found")
        except PermissionError as exc:
            return Result(126, "", f"{argv[0]}: {exc}")
        return Result(done.returncode, done.stdout, done.stderr)

    def run_attached(self, argv):
        """Run with the terminal attached, for output that streams, such as docker logs -f."""
        try:
            return subprocess.run(argv).returncode
        except FileNotFoundError:
            self.err.write(f"{argv[0]}: command not found\n")
            return 127

    def http(self, method, url, *, headers=None, body=None, timeout=10.0, follow_redirects=True):
        request = urllib.request.Request(url, data=body, method=method, headers=dict(headers or {}))
        opener = urllib.request.build_opener(_SameHostAuth if follow_redirects else _NoRedirect)
        try:
            with opener.open(request, timeout=timeout) as response:
                return HttpResponse(response.status, _lower(response.headers), response.read())
        except urllib.error.HTTPError as exc:
            with exc:
                return HttpResponse(exc.code, _lower(exc.headers), exc.read())

    def download(self, url, headers, dest, offset, on_chunk):
        request = urllib.request.Request(url, headers=dict(headers))
        with urllib.request.build_opener(_SameHostAuth).open(request, timeout=60) as response:
            if offset and response.status != 206:
                raise DownloadError(
                    f"the server ignored the resume request for {url}; delete {dest} and pull again"
                )
            with open(dest, "ab" if offset else "wb") as handle:
                while chunk := response.read(CHUNK):
                    handle.write(chunk)
                    on_chunk(chunk)

    def port_open(self, port):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            return False

    def uid_gid(self):
        return os.getuid(), os.getgid()

    def isatty(self):
        return sys.stdin.isatty()

    def ask(self, prompt):
        return input(prompt)

    def monotonic(self):
        return time.monotonic()

    def sleep(self, seconds):
        time.sleep(seconds)

    def meminfo(self):
        return Path("/proc/meminfo").read_text()

    def cdi_specs(self):
        specs = []
        for directory in self.cdi_dirs:
            if not directory.is_dir():
                continue
            for path in sorted(directory.iterdir()):
                if path.is_file() and path.suffix in (".yaml", ".yml", ".json"):
                    specs.append(path.read_text())
        return specs

    def which(self, name):
        """The path of an executable on the PATH of this host's environment, or None."""
        return shutil.which(name, path=self.env.get("PATH", ""))
