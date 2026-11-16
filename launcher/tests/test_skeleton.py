import ast
import http.server
import socket
import sys
import tempfile
import threading
import unittest
from pathlib import Path

import launcher
from launcher.host import RealHost, Result
from launcher.tests.fakes import FakeHost

ROOT = Path(__file__).resolve().parents[2]


class PythonVersionTest(unittest.TestCase):
    def test_old_python_is_refused_with_version(self):
        self.assertEqual(
            launcher.python_problem((3, 10, 12)),
            "llmctl needs Python 3.11 or newer; found 3.10.12",
        )

    def test_supported_python_passes(self):
        self.assertIsNone(launcher.python_problem((3, 11, 0)))
        self.assertIsNone(launcher.python_problem((3, 14, 1)))


class StdlibOnlyTest(unittest.TestCase):
    def test_stdlib_only(self):
        files = sorted((ROOT / "launcher").rglob("*.py")) + [ROOT / "llmctl"]
        self.assertGreater(len(files), 3)
        foreign = []
        for path in files:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    names = [node.module or ""]
                else:
                    continue
                for name in names:
                    top = name.split(".")[0]
                    if top != "launcher" and top not in sys.stdlib_module_names:
                        foreign.append(f"{path.relative_to(ROOT)}: {name}")
        self.assertEqual(foreign, [])


class FakeHostTest(unittest.TestCase):
    def test_unexpected_command_is_an_error(self):
        with self.assertRaises(AssertionError) as ctx:
            FakeHost().run(["docker", "ps"])
        self.assertIn("docker ps", str(ctx.exception))

    def test_longest_prefix_wins_and_calls_are_recorded(self):
        host = FakeHost()
        host.on(("docker",), Result(1, "", "generic"))
        host.on(("docker", "ps"), Result(0, "listed", ""))
        self.assertEqual(host.run(["docker", "ps", "-a"]).stdout, "listed")
        self.assertEqual(host.run(["docker", "info"]).stderr, "generic")
        self.assertEqual(host.calls, [["docker", "ps", "-a"], ["docker", "info"]])


class _Handler(http.server.BaseHTTPRequestHandler):
    body = b"0123456789"

    def log_message(self, *args):
        pass

    def do_HEAD(self):
        self._respond(head=True)

    def do_GET(self):
        self._respond(head=False)

    def _respond(self, head):
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "/file")
            self.send_header("X-Linked-Etag", '"abc"')
            self.end_headers()
            return
        data = self.body
        status = 200
        rng = self.headers.get("Range")
        if rng:
            start = int(rng.split("=")[1].split("-")[0])
            data = data[start:]
            status = 206
        self.send_response(status)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if not head:
            self.wfile.write(data)


class RealHostTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_run_captures_output_and_extra_env(self):
        result = RealHost().run(
            [sys.executable, "-c", "import os; print(os.environ['LLMCTL_T'])"],
            env_extra={"LLMCTL_T": "seen"},
        )
        self.assertEqual((result.returncode, result.stdout), (0, "seen\n"))

    def test_missing_command_is_exit_127(self):
        result = RealHost().run(["llmctl-no-such-command-xyz"])
        self.assertEqual(result.returncode, 127)
        self.assertIn("not found", result.stderr)

    def test_http_without_redirects_keeps_first_headers(self):
        url = f"http://127.0.0.1:{self.port}/redirect"
        first = RealHost().http("HEAD", url, follow_redirects=False)
        self.assertEqual(first.status, 302)
        self.assertEqual(first.headers.get("x-linked-etag"), '"abc"')
        followed = RealHost().http("GET", url)
        self.assertEqual((followed.status, followed.body), (200, b"0123456789"))

    def test_download_appends_from_offset(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "f.incomplete"
            dest.write_bytes(b"0123")
            seen = []
            RealHost().download(
                f"http://127.0.0.1:{self.port}/file", {"Range": "bytes=4-"}, dest, 4, seen.append
            )
            self.assertEqual(dest.read_bytes(), b"0123456789")
            self.assertEqual(b"".join(seen), b"456789")

    def test_port_open(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = listener.getsockname()[1]
            self.assertTrue(RealHost().port_open(port))
        self.assertFalse(RealHost().port_open(port))

    def test_cdi_specs_reads_spec_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "nvidia.yaml").write_text("kind: nvidia.com/gpu\n")
            Path(tmp, "notes.txt").write_text("ignored\n")
            specs = RealHost(cdi_dirs=(Path(tmp), Path(tmp) / "missing")).cdi_specs()
            self.assertEqual(specs, ["kind: nvidia.com/gpu\n"])


if __name__ == "__main__":
    unittest.main()
