import shlex
import unittest
from pathlib import Path

from launcher.catalog import load_catalog, resolve
from launcher.command import docker_run_argv, labels
from launcher.model import ContainerSpec, HostEnv
from launcher.runtimes import llamacpp
from launcher.tests.fakes import REPO


def qwen_argv(hf_home="/home/user/.cache/huggingface"):
    recipe, runtime = resolve(load_catalog(REPO / "recipes", None), "qwen3.6", None)
    spec = llamacpp.container_spec(recipe, runtime, HostEnv(Path(hf_home), 1000, 1000))
    return recipe, runtime, spec, docker_run_argv(recipe, runtime, spec)


class CommandRulesTest(unittest.TestCase):
    def test_rules_in_argv(self):
        _, _, _, argv = qwen_argv()
        self.assertEqual(argv[:5], ["docker", "run", "-d", "--name", "llamacpp"])
        joined = " ".join(argv)
        for expected in ("--device nvidia.com/gpu=all", "--memory 20g", "--memory-swap 20g",
                         "-p 127.0.0.1:8080:8080", "--user 1000:1000", "-e GGML_CUDA_DISABLE_GRAPHS=1"):
            self.assertIn(expected, joined)
        for forbidden in ("--restart", "--privileged", "--network", "--rm"):
            self.assertNotIn(forbidden, argv)
        image_at = argv.index("ghcr.io/ggml-org/llama.cpp:server-cuda-b12407")
        self.assertLess(argv.index("GGML_CUDA_DISABLE_GRAPHS=1"), image_at)
        self.assertEqual(argv[image_at + 1], "-m")

    def test_labels(self):
        recipe, runtime, _, argv = qwen_argv()
        expected_sha = __import__("hashlib").sha256((REPO / "recipes/llamacpp/qwen3.6-35b-a3b.toml").read_bytes()).hexdigest()
        self.assertEqual(labels(recipe, runtime)[:6], [
            ("io.llmctl.managed", "true"),
            ("io.llmctl.runtime", "llamacpp"),
            ("io.llmctl.recipe", "qwen3.6-35b-a3b"),
            ("io.llmctl.source", "shipped"),
            ("io.llmctl.sha256", expected_sha),
            ("io.llmctl.evidence", "measured"),
        ])
        self.assertIn(f"io.llmctl.sha256={expected_sha}", argv)

    def test_hf_home_with_space(self):
        _, _, _, argv = qwen_argv("/data/hf cache")
        self.assertIn("/data/hf cache:/hf", argv)
        self.assertIn("'/data/hf cache:/hf'", shlex.join(argv))

    def test_adapter_cannot_add_forbidden_flags(self):
        recipe, runtime, spec, _ = qwen_argv()
        for flag in ("--privileged", "--restart=always", "--network=host", "--rm", "-p", "--publish=0.0.0.0:80:80"):
            bad = ContainerSpec(spec.user, spec.env, spec.mounts, (flag,), spec.image, spec.args, spec.container_port)
            with self.assertRaises(ValueError):
                docker_run_argv(recipe, runtime, bad)


if __name__ == "__main__":
    unittest.main()
