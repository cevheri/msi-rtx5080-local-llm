"""A Host for tests: commands, HTTP, time and the terminal are scripted, and every call is recorded."""

import io
import shlex
from pathlib import Path

from launcher.host import HttpResponse, Result


class FakeHost:
    def __init__(self, env=None, uid=1000, gid=1000, tty=False):
        self.env = dict({"HOME": "/home/user"} if env is None else env)
        self.uid = uid
        self.gid = gid
        self.tty = tty
        self.out = io.StringIO()
        self.err = io.StringIO()
        self.calls: list[list[str]] = []
        self.env_extras: list[dict[str, str]] = []
        self.http_calls: list[tuple[str, str, dict[str, str]]] = []
        self.answers: list[str] = []
        self.prompts: list[str] = []
        self.open_ports: set[int] = set()
        self.clock = 0.0
        self.sleeps: list[float] = []
        self.on_sleep = None
        self.meminfo_text = "MemTotal:       64000000 kB\nMemAvailable:   50000000 kB\n"
        self.cdi = ["kind: nvidia.com/gpu\n"]
        self.downloads: dict[str, bytes] = {}
        self.download_calls: list[tuple[str, dict[str, str], int]] = []
        self.attached: list[list[str]] = []
        self.installed: dict[str, str] = {}
        self._routes: list[tuple[tuple[str, ...], object]] = []
        self._http: dict[tuple[str, str], object] = {}

    def on(self, argv_prefix, result):
        """Answer commands that start with argv_prefix; the longest prefix, then the latest, wins."""
        self._routes.append((tuple(argv_prefix), result))

    def run(self, argv, *, input=None, env_extra=None):
        argv = list(argv)
        self.calls.append(argv)
        self.env_extras.append(dict(env_extra or {}))
        best = None
        for prefix, result in self._routes:
            if tuple(argv[: len(prefix)]) == prefix and (best is None or len(prefix) >= len(best[0])):
                best = (prefix, result)
        if best is None:
            raise AssertionError(f"unexpected command: {shlex.join(argv)}")
        result = best[1]
        return result(argv) if callable(result) else result

    def run_attached(self, argv):
        self.attached.append(list(argv))
        return 0

    def on_http(self, method, url, response):
        """response: an HttpResponse, an exception to raise, or a callable returning either."""
        self._http[(method, url)] = response

    def http(self, method, url, *, headers=None, body=None, timeout=10.0, follow_redirects=True):
        self.http_calls.append((method, url, dict(headers or {})))
        if (method, url) not in self._http:
            raise ConnectionRefusedError(f"no fake route for {method} {url}")
        response = self._http[(method, url)]
        if callable(response) and not isinstance(response, HttpResponse):
            response = response(method, url, headers, body)
        if isinstance(response, BaseException):
            raise response
        return response

    def download(self, url, headers, dest, offset, on_chunk):
        self.download_calls.append((url, dict(headers), offset))
        data = self.downloads[url]
        if isinstance(data, BaseException):
            raise data
        chunk = data[offset:] if offset else data
        with open(dest, "ab" if offset else "wb") as handle:
            handle.write(chunk)
        on_chunk(chunk)

    def port_open(self, port):
        return port in self.open_ports

    def uid_gid(self):
        return self.uid, self.gid

    def isatty(self):
        return self.tty

    def ask(self, prompt):
        self.prompts.append(prompt)
        return self.answers.pop(0)

    def monotonic(self):
        return self.clock

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.clock += seconds
        if self.on_sleep is not None:
            self.on_sleep(self)

    def meminfo(self):
        return self.meminfo_text

    def cdi_specs(self):
        return list(self.cdi)

    def which(self, name):
        return self.installed.get(name)


def ok(stdout: str = "") -> Result:
    return Result(0, stdout, "")


def fail(stderr: str = "", code: int = 1) -> Result:
    return Result(code, "", stderr)


def http(status: int, body: bytes = b"", headers=None) -> HttpResponse:
    return HttpResponse(status, dict(headers or {}), body)


REPO = Path(__file__).resolve().parents[2]
