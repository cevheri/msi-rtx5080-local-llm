import unittest

from launcher import ops
from launcher.errors import PreflightError
from launcher.tests.fakes import fail, ok
from launcher.tests.ops_fixture import GPU_APPS, GPU_QUERY, OpsFixture, managed_line


class PreflightRefusalTest(unittest.TestCase):
    def setUp(self):
        self.f = OpsFixture()

    def tearDown(self):
        self.f.cleanup()

    def refuse(self, *needles, name="qwen3.6", runtime=None):
        with self.assertRaises(PreflightError) as ctx:
            ops.up(self.f.host, self.f.catalog, name, runtime, replace=False, dry_run=False)
        self.assertEqual(ctx.exception.exit_code, 3)
        for needle in needles:
            self.assertIn(needle, str(ctx.exception))
        self.assertEqual(self.f.calls_starting("docker", "run"), [])
        return str(ctx.exception)

    def test_docker_down(self):
        self.f.host.on(("docker", "info"), fail("Cannot connect to the Docker daemon at unix:///var/run/docker.sock"))
        self.refuse("Docker is not answering", "systemctl start docker")

    def test_docker_permission_denied(self):
        self.f.host.on(("docker", "info"), fail("permission denied while trying to connect to the Docker daemon socket"))
        self.refuse("permission denied", "docker group")

    def test_docker_missing(self):
        self.f.host.on(("docker", "info"), fail("docker: command not found", code=127))
        self.refuse("docker was not found")

    def test_no_cdi_spec(self):
        self.f.host.cdi = ["kind: example.com/other\n"]
        self.refuse("nvidia.com/gpu", "NVIDIA Container Toolkit")

    def test_image_missing(self):
        self.f.host.on(("docker", "image", "inspect"), fail("Error: No such image"))
        self.refuse("ghcr.io/ggml-org/llama.cpp:server-cuda-b12407", "./llmctl pull qwen3.6-35b-a3b")

    def test_files_missing(self):
        message = self.refuse("./llmctl pull qwen3.8-27b", name="qwen3.8")
        self.assertIn("missing Qwen3.8-27B-UD-Q3_K_XL.gguf", message)

    def test_files_wrong_size(self):
        self.f.place_files("qwen3.6-35b-a3b", size=1000)
        self.refuse("add up to 1000 bytes", "./llmctl pull qwen3.6-35b-a3b")

    def test_other_managed_running(self):
        self.f.managed_lines = [managed_line("vllm", "running", "gpt-oss-20b", runtime="vllm")]
        self.refuse("gpt-oss-20b", "--replace", "./llmctl down")

    def test_foreign_container_name(self):
        self.f.foreign = ("llamacpp", "exited")
        self.refuse("'llamacpp'", "not started by llmctl", "docker rm -f llamacpp")
        self.assertEqual(self.f.calls_starting("docker", "rm"), [])
        self.assertEqual(self.f.calls_starting("docker", "stop"), [])

    def test_port_taken_by_other_process(self):
        self.f.host.open_ports.add(8080)
        self.refuse("127.0.0.1:8080", "already in use")

    def test_gpu_too_full(self):
        self.f.host.on(GPU_QUERY, ok("16303, 462, 1600, 14241\n"))
        self.f.host.on(GPU_APPS, ok("4242, /usr/bin/blender, 1488\n"))
        self.refuse("15078 MiB", "14241 MiB", "blender (pid 4242, 1488 MiB)")

    def test_nvidia_smi_missing(self):
        self.f.host.on(GPU_QUERY, fail("nvidia-smi: command not found", code=127))
        self.refuse("nvidia-smi was not found")

    def test_nvidia_smi_failing(self):
        self.f.host.on(GPU_QUERY, fail("NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver"))
        self.refuse("nvidia-smi failed", "couldn't communicate")

    def test_ram_too_low(self):
        self.f.host.meminfo_text = "MemTotal: 32000000 kB\nMemAvailable:   10485760 kB\n"
        self.refuse("20 GiB", "10.0 GiB")


if __name__ == "__main__":
    unittest.main()
