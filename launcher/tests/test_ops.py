import unittest

from launcher import ops
from launcher.errors import PreflightError, StartError
from launcher.lock import hold_lock
from launcher.tests.fakes import http, ok
from launcher.tests.ops_fixture import OpsFixture, managed_line


class UpTest(unittest.TestCase):
    def setUp(self):
        self.f = OpsFixture()

    def tearDown(self):
        self.f.cleanup()

    def up(self, name="qwen3.6", runtime=None, replace=False, dry_run=False):
        return ops.up(self.f.host, self.f.catalog, name, runtime, replace=replace, dry_run=dry_run)

    def test_up_success_prints_opencode(self):
        self.assertEqual(self.up(), 0)
        runs = self.f.calls_starting("docker", "run")
        self.assertEqual(runs, [self.f.golden("llamacpp", "qwen3.6-35b-a3b")])
        out = self.f.host.out.getvalue()
        self.assertIn("http://127.0.0.1:8080/v1", out)
        self.assertIn("OpenCode model: llamacpp-docker/qwen3.6-35b-a3b", out)
        self.assertIn("/models", out)
        self.assertIn('opencode run --model llamacpp-docker/qwen3.6-35b-a3b "', out)
        self.assertNotIn("opencode -m", out)  # the interactive command has no -m flag in OpenCode v2.3.0
        self.assertIn("131072", out)
        self.assertIn("122.5 tok/s", out)
        self.assertIn("./llmctl down", out)

    def test_up_replace_stops_other(self):
        self.f.managed_lines = [managed_line("vllm", "running", "gpt-oss-20b", runtime="vllm")]
        self.assertEqual(self.up(replace=True), 0)
        calls = self.f.host.calls
        stop = calls.index(["docker", "stop", "-t", "60", "vllm"])
        run = calls.index(self.f.golden("llamacpp", "qwen3.6-35b-a3b"))
        self.assertLess(stop, run)

    def test_up_removes_own_stopped_container(self):
        self.f.managed_lines = [managed_line("llamacpp", "exited", "qwen3.8-27b")]
        self.assertEqual(self.up(), 0)
        self.assertEqual(self.f.calls_starting("docker", "rm"), [["docker", "rm", "llamacpp"]])

    def test_container_exits_during_start(self):
        self.f.health = [http(503)]
        self.f.state = "exited"
        with self.assertRaises(StartError) as ctx:
            self.up()
        self.assertEqual(ctx.exception.exit_code, 4)
        self.assertIn("exited", str(ctx.exception))
        self.assertIn(["docker", "logs", "--tail", "40", "llamacpp"], self.f.host.calls)
        self.assertEqual(self.f.calls_starting("docker", "rm"), [])

    def test_ready_timeout(self):
        self.f.health = [http(503)]
        with self.assertRaises(StartError) as ctx:
            self.up()
        self.assertIn("not ready after 300 s", str(ctx.exception))
        self.assertIn("./llmctl logs", str(ctx.exception))
        self.assertEqual(self.f.calls_starting("docker", "stop"), [])

    def test_interrupt_during_wait(self):
        self.f.health = [http(503)]

        def interrupt(host):
            raise KeyboardInterrupt

        self.f.host.on_sleep = interrupt
        self.assertEqual(self.up(), 130)
        self.assertIn("./llmctl status", self.f.host.err.getvalue())
        with hold_lock(self.f.host.env):
            pass

    def test_dry_run_starts_nothing(self):
        self.assertEqual(self.up(dry_run=True), 0)
        self.assertEqual(self.f.calls_starting("docker", "run"), [])
        self.assertIn("docker run -d --name llamacpp", self.f.host.out.getvalue())

    def test_estimate_warns(self):
        self.f.place_files("gemma-4-12b", runtime="vllm")
        self.f.host.on_http("GET", "http://127.0.0.1:8000/v1/models",
                            http(200, b'{"data":[{"id":"gemma-4-12b"}]}'))
        self.assertEqual(self.up("gemma-4-12b", "vllm"), 0)
        self.assertIn("never run on the reference laptop", self.f.host.out.getvalue())

    def test_chat_only_model_says_so(self):
        self.f.place_files("deepseek-r1-0528-qwen3-8b")
        self.assertEqual(self.up("deepseek-r1"), 0)
        out = self.f.host.out.getvalue()
        self.assertIn("chat only, no tool calling", out)
        self.assertIn("no OpenCode entry", out)

    def test_lock_held_refuses(self):
        with hold_lock(self.f.host.env):
            with self.assertRaises(PreflightError):
                self.up()


class DownStatusLogsTest(unittest.TestCase):
    def setUp(self):
        self.f = OpsFixture()

    def tearDown(self):
        self.f.cleanup()

    def test_down_only_managed(self):
        self.f.managed_lines = [managed_line("llamacpp", "running", "qwen3.6-35b-a3b")]
        self.assertEqual(ops.down(self.f.host, remove=False), 0)
        self.assertEqual(self.f.calls_starting("docker", "stop"), [["docker", "stop", "-t", "60", "llamacpp"]])
        self.assertEqual(self.f.calls_starting("docker", "rm"), [])
        self.assertEqual(ops.down(self.f.host, remove=True), 0)
        self.assertEqual(self.f.calls_starting("docker", "rm"), [["docker", "rm", "llamacpp"]])

    def test_down_with_nothing_running(self):
        self.assertEqual(ops.down(self.f.host, remove=False), 0)
        self.assertIn("no model server started by llmctl", self.f.host.out.getvalue())

    def test_status_reads_labels(self):
        self.f.managed_lines = [managed_line("llamacpp", "running", "qwen3.6-35b-a3b", sha="cd" * 32)]
        self.f.host.on(("docker", "port"), ok("8080/tcp -> 127.0.0.1:8080\n"))
        self.f.host.on(("docker", "stats"), ok("14.2GiB / 20GiB\n"))
        self.f.health = [http(200)]
        self.assertEqual(ops.status(self.f.host, self.f.catalog), 0)
        out = self.f.host.out.getvalue()
        for expected in ("qwen3.6-35b-a3b", "llamacpp", "shipped", "cdcdcdcdcdcd", "running", "127.0.0.1:8080",
                         "ready", "14.2GiB / 20GiB", "112 MiB"):
            self.assertIn(expected, out)

    def test_logs_follow_is_attached(self):
        self.f.managed_lines = [managed_line("llamacpp", "running", "qwen3.6-35b-a3b")]
        self.assertEqual(ops.logs(self.f.host, follow=True, tail=100), 0)
        self.assertEqual(self.f.host.attached, [["docker", "logs", "--tail", "100", "-f", "llamacpp"]])


if __name__ == "__main__":
    unittest.main()
