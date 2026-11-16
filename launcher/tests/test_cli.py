import hashlib
import tempfile
import tomllib
import unittest
from pathlib import Path

from launcher.cli import main
from launcher.runtimes import llamacpp
from launcher.tests.fakes import REPO, FakeHost, fail, ok
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
        return host

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

    def test_test_without_server(self):
        host = self.host()
        host.on(("docker", "ps"), ok(""))
        self.assertEqual(main(["test"], host), 3)
        self.assertIn("./llmctl up", host.err.getvalue())


if __name__ == "__main__":
    unittest.main()
