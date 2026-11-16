import hashlib
import re
import tempfile
import tomllib
import unittest
from pathlib import Path

from launcher.cli import main
from launcher.runtimes import llamacpp
from launcher.tests.fakes import REPO, FakeHost, fail, http, ok
from launcher.validate import read_toml


class CliTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.overrides = root / "overrides"
        self.run_dir = root / "run"
        self.run_dir.mkdir()
        self.env = {"HOME": "/home/user", "HF_HOME": str(root / "hf"), "LLMCTL_RECIPES_DIR": str(self.overrides),
                    "XDG_RUNTIME_DIR": str(self.run_dir)}

    def tearDown(self):
        self.tmp.cleanup()

    def host(self, tty=False):
        host = FakeHost(env=self.env, tty=tty)
        host.on(("docker", "image", "inspect"), fail("Error: No such image"))
        host.on(("docker", "ps"), ok(""))
        return host

    def list_rows(self, host):
        self.assertEqual(main(["list"], host), 0)
        lines = host.out.getvalue().splitlines()
        header = re.split(r"\s{2,}", lines[0])
        rows = {}
        for line in lines[1:]:
            cells = re.split(r"\s{2,}", line)
            if len(cells) == len(header):
                rows[(cells[0], cells[1])] = dict(zip(header, cells))
        return rows, host.out.getvalue()

    def running(self, host, recipe_id, runtime="llamacpp", sha=None, health=200):
        from launcher.catalog import load_catalog
        from launcher.tests.ops_fixture import managed_line
        catalog = load_catalog(REPO / "recipes", None)
        recipe = catalog.runtimes[runtime].recipes[recipe_id]
        line = managed_line(runtime, "running", recipe_id, runtime=runtime, sha=sha or recipe.source.sha256,
                            runtime_sha=catalog.runtimes[runtime].runtime.source.sha256)
        host.on(("docker", "ps"), ok(line + "\n"))
        port = catalog.runtimes[runtime].runtime.host_port
        path = catalog.runtimes[runtime].runtime.ready_path
        host.on_http("GET", f"http://127.0.0.1:{port}{path}", http(health, b'{"status":"ok"}'))

    def break_override(self):
        path = self.overrides / "llamacpp" / "qwen3.6-35b-a3b.toml"
        path.parent.mkdir(parents=True)
        path.write_text((REPO / "recipes/llamacpp/qwen3.6-35b-a3b.toml").read_text().replace("memory_gib", "memmory_gib"))
        return path

    def test_usage_error_exit_1(self):
        host = self.host()
        self.assertEqual(main(["frobnicate"], host), 1)
        self.assertIn("error:", host.err.getvalue())

    def test_recipe_error_exit_2(self):
        path = self.break_override()
        host = self.host()
        self.assertEqual(main(["up", "qwen3.6"], host), 2)
        self.assertIn(str(path), host.err.getvalue())

    def test_errors_have_no_traceback(self):
        host = self.host()
        host.on(("docker", "info"), fail("Cannot connect to the Docker daemon"))
        self.assertEqual(main(["up", "qwen3.6"], host), 3)
        err = host.err.getvalue()
        self.assertTrue(err.startswith("error: "), err)
        self.assertNotIn("Traceback", err)

    def test_list_shows_measured_and_estimate(self):
        host = self.host()
        self.assertEqual(main(["list"], host), 0)
        out = host.out.getvalue()
        for expected in ("qwen3.6-35b-a3b", "qwen3.6", "measured", "gemma-4-12b", "estimate", "15078", "92%", "131072"):
            self.assertIn(expected, out)
        self.assertNotIn("problem", out)

    def test_list_marks_the_running_model_and_footer(self):
        host = self.host()
        self.running(host, "gemma-4-26b-a4b")
        config = Path(self.tmp.name) / "xdg" / "opencode" / "opencode.json"
        config.parent.mkdir(parents=True)
        config.write_text('{"model": "llamacpp-docker/gemma-4-26b-a4b", "providers": {}}\n')
        host.env["XDG_CONFIG_HOME"] = str(config.parent.parent)
        rows, out = self.list_rows(host)
        self.assertEqual(rows[("gemma-4-26b-a4b", "llamacpp")]["STATUS"], "running")
        self.assertEqual(rows[("qwen3.6-35b-a3b", "llamacpp")]["STATUS"], "-")
        self.assertIn("running: gemma-4-26b-a4b on llamacpp, http://127.0.0.1:8080/v1", out)
        self.assertIn("OpenCode default: llamacpp-docker/gemma-4-26b-a4b", out)

    def test_list_status_starting_and_older_file(self):
        host = self.host()
        self.running(host, "qwen3.6-35b-a3b", health=503)
        rows, _ = self.list_rows(host)
        self.assertEqual(rows[("qwen3.6-35b-a3b", "llamacpp")]["STATUS"], "starting")
        host = self.host()
        self.running(host, "qwen3.6-35b-a3b", sha="ef" * 32)
        rows, _ = self.list_rows(host)
        self.assertEqual(rows[("qwen3.6-35b-a3b", "llamacpp")]["STATUS"], "running (older file)")

    def test_list_when_docker_does_not_answer(self):
        host = self.host()
        host.on(("docker", "ps"), fail("permission denied while trying to connect to the Docker daemon socket"))
        host.on(("docker", "info"), fail("permission denied while trying to connect to the Docker daemon socket"))
        rows, out = self.list_rows(host)
        self.assertEqual(rows[("qwen3.6-35b-a3b", "llamacpp")]["STATUS"], "?")
        self.assertIn("Docker did not answer", out)

    def test_list_speed_column(self):
        rows, _ = self.list_rows(self.host())
        self.assertEqual(rows[("qwen3.6-35b-a3b", "llamacpp")]["TOK/S"], "122.5")
        self.assertEqual(rows[("gpt-oss-20b", "vllm")]["TOK/S"], "83.5")
        self.assertEqual(rows[("gemma-4-12b", "vllm")]["TOK/S"], "-")

    def test_show_prints_command_and_sha(self):
        host = self.host()
        self.assertEqual(main(["show", "qwen3.6"], host), 0)
        out = host.out.getvalue()
        sha = hashlib.sha256((REPO / "recipes/llamacpp/qwen3.6-35b-a3b.toml").read_bytes()).hexdigest()
        self.assertIn(sha, out)
        self.assertIn("docker run -d --name llamacpp", out)

    def test_show_as_override_marks_estimate(self):
        host = self.host()
        self.assertEqual(main(["show", "qwen3.6", "--as-override"], host), 0)
        text = host.out.getvalue()
        data = tomllib.loads(text)
        self.assertEqual(data["evidence"]["kind"], "estimate")
        path = Path(self.tmp.name) / "qwen3.6-35b-a3b.toml"
        path.write_text(text)
        recipe = llamacpp.parse_recipe(*read_toml(path, "override"), "qwen3.6-35b-a3b")
        self.assertEqual(recipe.args, llamacpp.parse_recipe(*read_toml(REPO / "recipes/llamacpp/qwen3.6-35b-a3b.toml",
                                                                        "shipped"), "qwen3.6-35b-a3b").args)

    def test_pull_needs_yes_without_tty(self):
        host = self.host(tty=False)
        self.assertEqual(main(["pull", "qwen3.8"], host), 1)
        self.assertIn("--yes", host.err.getvalue())
        self.assertEqual(host.http_calls, [])
        self.assertEqual(host.attached, [])

    def test_pull_asks_on_tty(self):
        host = self.host(tty=True)
        host.answers = ["n"]
        self.assertEqual(main(["pull", "qwen3.8"], host), 1)
        self.assertEqual(len(host.prompts), 1)
        self.assertIn("13.1 GB", host.out.getvalue())
        self.assertEqual(host.http_calls, [])
        self.assertEqual(host.attached, [])

    def test_pull_with_yes_pulls_image_first(self):
        host = self.host(tty=False)
        host.on(("docker", "info"), ok("29.9.3\n"))
        order = []
        host.run_attached = lambda argv: order.append(argv) or 1
        self.assertEqual(main(["pull", "qwen3.8", "--yes"], host), 5)
        self.assertEqual(order, [["docker", "pull", "ghcr.io/ggml-org/llama.cpp:server-cuda-b12407"]])

    def test_check_reports_problems(self):
        host = self.host()
        self.assertEqual(main(["check"], host), 0)
        self.assertIn("ok: 19 recipes", host.out.getvalue())
        path = self.break_override()
        host = self.host()
        self.assertEqual(main(["check"], host), 2)
        self.assertIn(str(path), host.out.getvalue())

    def test_env_var_sets_opencode_default(self):
        from launcher.tests.ops_fixture import OpsFixture
        from launcher.tests.test_opencode_config import doc_config_text
        fixture = OpsFixture()
        self.addCleanup(fixture.cleanup)
        config = fixture.runtime_dir / "xdg" / "opencode" / "opencode.json"
        config.parent.mkdir(parents=True)
        config.write_text(doc_config_text().replace("llamacpp-docker/qwen3.6-35b-a3b", "vllm-docker/gpt-oss-20b", 1))
        fixture.host.env.update({"XDG_CONFIG_HOME": str(config.parent.parent), "LLMCTL_SET_OPENCODE_DEFAULT": "1"})
        fixture.host.on(("opencode", "reload"), ok())
        self.assertEqual(main(["up", "qwen3.6"], fixture.host), 0)
        self.assertIn('"model": "llamacpp-docker/qwen3.6-35b-a3b"', config.read_text())

    def test_test_without_server(self):
        host = self.host()
        host.on(("docker", "ps"), ok(""))
        self.assertEqual(main(["test"], host), 3)
        self.assertIn("./llmctl up", host.err.getvalue())


if __name__ == "__main__":
    unittest.main()
