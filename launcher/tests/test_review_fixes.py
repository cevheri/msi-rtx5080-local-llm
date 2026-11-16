"""Tests for the findings of the final review, each written before its fix."""

import http.server
import io
import tempfile
import threading
import unittest
import urllib.error
from pathlib import Path

from launcher import ops
from launcher.catalog import load_catalog, resolve
from launcher.cli import main
from launcher.errors import DownloadError, PreflightError, RecipeError
from launcher.hfcache import download_file, snapshot_dir
from launcher.host import RealHost
from launcher.model import HostEnv
from launcher.runtimes import llamacpp, vllm
from launcher.tests.fakes import REPO, FakeHost, fail, ok
from launcher.tests.fakes import http as fake_http
from launcher.tests.ops_fixture import GPU_QUERY, OpsFixture
from launcher.validate import read_toml

REV = "52f9f9b8052fb9c8b69bba4a95eda0f319ac3d2a"
URL = f"https://huggingface.co/org/model/resolve/{REV}/m.gguf"
DENIED = "permission denied while trying to connect to the Docker daemon socket"


class DockerUnreachableTest(unittest.TestCase):
    """Finding 1: down, status, logs and test must not report 'nothing running' when Docker cannot answer."""

    def setUp(self):
        self.f = OpsFixture()
        self.f.host.on(("docker", "ps"), fail(DENIED))
        self.f.host.on(("docker", "info"), fail(DENIED))

    def tearDown(self):
        self.f.cleanup()

    def test_down_status_logs_refuse(self):
        for call in (lambda: ops.down(self.f.host, self.f.catalog, remove=False), lambda: ops.status(self.f.host, self.f.catalog),
                     lambda: ops.logs(self.f.host, follow=False, tail=10)):
            with self.assertRaises(PreflightError) as ctx:
                call()
            self.assertIn("docker group", str(ctx.exception))
        self.assertNotIn("no model server", self.f.host.out.getvalue())


class NetworkErrorTest(unittest.TestCase):
    """Finding 2: a network failure during pull is one error line with exit code 5."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_head_failure_is_download_error(self):
        host = FakeHost()
        host.on_http("HEAD", URL, urllib.error.URLError("Temporary failure in name resolution"))
        with self.assertRaises(DownloadError) as ctx:
            download_file(host, self.home, "org/model", REV, "m.gguf", None, io.StringIO())
        self.assertIn("name resolution", str(ctx.exception))

    def test_dropped_connection_keeps_partial_and_says_resume(self):
        host = FakeHost()
        sha = "a" * 64
        host.on_http("HEAD", URL, fake_http(302, headers={"x-linked-etag": f'"{sha}"', "x-linked-size": "10"}))
        host.downloads[URL] = ConnectionResetError("Connection reset by peer")
        with self.assertRaises(DownloadError) as ctx:
            download_file(host, self.home, "org/model", REV, "m.gguf", None, io.StringIO())
        self.assertIn("pull again", str(ctx.exception))
        self.assertIn("resume", str(ctx.exception))

    def test_cli_pull_network_error_exit_5(self):
        env = {"HOME": "/home/user", "HF_HOME": str(self.home / "hf"), "LLMCTL_RECIPES_DIR": str(self.home / "o"),
               "XDG_RUNTIME_DIR": str(self.home)}
        host = FakeHost(env=env)
        host.on(("docker", "image", "inspect"), ok("sha256:x"))
        recipe = load_catalog(REPO / "recipes", None).runtimes["llamacpp"].recipes["qwen3.8-27b"]
        url = f"https://huggingface.co/{recipe.model.repo}/resolve/{recipe.model.revision}/{recipe.model.files[0]}"
        host.on_http("HEAD", url, urllib.error.URLError("no network"))
        self.assertEqual(main(["pull", "qwen3.8", "--yes"], host), 5)
        self.assertNotIn("Traceback", host.err.getvalue())


class VllmCompletenessTest(unittest.TestCase):
    """Finding 3: a vLLM snapshot with config.json but missing weights is not complete."""

    def test_partial_snapshot_is_reported(self):
        recipe = load_catalog(REPO / "recipes", None).runtimes["vllm"].recipes["gpt-oss-20b"]
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            snapshot = snapshot_dir(home, recipe.model.repo, recipe.model.revision)
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text("{}")
            problems = vllm.file_problems(recipe, home)
            self.assertEqual(len(problems), 1)
            self.assertIn("13789264674", problems[0])
            (snapshot / "dangling.safetensors").symlink_to(home / "blobs" / "missing")
            self.assertIn("dangling.safetensors", vllm.file_problems(recipe, home)[0])
            (snapshot / "dangling.safetensors").unlink()
            with open(snapshot / "model.safetensors", "wb") as handle:
                handle.truncate(recipe.model.size_bytes - 2)
            self.assertEqual(vllm.file_problems(recipe, home), [])


class FreshCacheTest(unittest.TestCase):
    """Finding 4: pull creates the cache as the user before Docker can create it as root."""

    def test_cache_exists_before_docker_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "fresh" / "hf"
            env = {"HOME": "/home/user", "HF_HOME": str(home), "LLMCTL_RECIPES_DIR": str(Path(tmp) / "o"),
                   "XDG_RUNTIME_DIR": tmp}
            host = FakeHost(env=env)
            host.on(("docker", "image", "inspect"), ok("sha256:x"))
            seen = []
            host.on(("docker", "run"), lambda argv: seen.append((home / "hub").is_dir()) or fail("stop here"))
            self.assertEqual(main(["pull", "gpt-oss", "--runtime", "vllm", "--yes"], host), 5)
            self.assertEqual(seen, [True])


class _Recorder(http.server.BaseHTTPRequestHandler):
    seen = []
    target = ""

    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", type(self).target)
            self.end_headers()
            return
        type(self).seen.append(self.headers.get("Authorization"))
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")


class TokenRedirectTest(unittest.TestCase):
    """Finding 5: the token is not forwarded when a redirect leaves the host it was meant for."""

    def serve(self):
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Recorder)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server.server_address[1]

    def test_cross_host_redirect_drops_authorization(self):
        origin, cdn = self.serve(), self.serve()
        _Recorder.seen = []
        _Recorder.target = f"http://localhost:{cdn}/blob"
        with tempfile.TemporaryDirectory() as tmp:
            RealHost().download(f"http://127.0.0.1:{origin}/redirect", {"Authorization": "Bearer hf_secret"},
                                Path(tmp) / "f", 0, lambda chunk: None)
        self.assertEqual(_Recorder.seen, [None])

    def test_same_host_redirect_keeps_authorization(self):
        origin = self.serve()
        _Recorder.seen = []
        _Recorder.target = f"http://127.0.0.1:{origin}/blob"
        response = RealHost().http("GET", f"http://127.0.0.1:{origin}/redirect", headers={"Authorization": "Bearer t"})
        self.assertEqual((response.status, _Recorder.seen), (200, ["Bearer t"]))


class CudaGraphsRuleTest(unittest.TestCase):
    """Finding 6: GGML_CUDA_DISABLE_GRAPHS=1 holds whatever the runtime file says."""

    def runtime_data(self):
        return read_toml(REPO / "recipes/llamacpp/_runtime.toml", "override")

    def test_graphs_flag_survives_runtime_override(self):
        data, source = self.runtime_data()
        data.pop("env", None)
        runtime = llamacpp.parse_runtime(data, source)
        recipe = resolve(load_catalog(REPO / "recipes", None), "qwen3.6", None)[0]
        spec = llamacpp.container_spec(recipe, runtime, HostEnv(Path("/h"), 1, 1))
        self.assertIn(("GGML_CUDA_DISABLE_GRAPHS", "1"), spec.env)

    def test_runtime_env_cannot_set_graphs(self):
        data, source = self.runtime_data()
        data["env"] = {"GGML_CUDA_DISABLE_GRAPHS": "0"}
        with self.assertRaises(RecipeError) as ctx:
            llamacpp.parse_runtime(data, source)
        self.assertIn("GGML_CUDA_DISABLE_GRAPHS", str(ctx.exception))


class GpuAdviceTest(unittest.TestCase):
    """Finding 7: the GPU refusal gives advice that fits the runtime and the recipe."""

    def refuse(self, name, runtime=None):
        f = OpsFixture()
        self.addCleanup(f.cleanup)
        recipe = resolve(f.catalog, name, runtime)[0]
        f.place_files(recipe.id, runtime=recipe.runtime)
        f.host.on(GPU_QUERY, ok("16303, 462, 9000, 6841\n"))
        f.host.on_http("GET", "http://127.0.0.1:8000/v1/models", fake_http(503))
        with self.assertRaises(PreflightError) as ctx:
            ops.up(f.host, f.catalog, name, runtime, replace=False, dry_run=False)
        return str(ctx.exception)

    def test_moe_recipe_mentions_expert_cache(self):
        self.assertIn("--moe-cache-mib", self.refuse("qwen3.6"))

    def test_dense_recipe_mentions_context(self):
        message = self.refuse("qwen3.8")
        self.assertNotIn("--moe-cache-mib", message)
        self.assertIn("context", message)

    def test_vllm_recipe_mentions_utilization(self):
        message = self.refuse("gpt-oss", "vllm")
        self.assertNotIn("--moe-cache-mib", message)
        self.assertIn("--gpu-memory-utilization", message)


class RemoteCodeCacheTest(unittest.TestCase):
    """Finding 8: vLLM gets a writable modules cache, since the model cache is mounted read-only."""

    def test_modules_cache_is_writable(self):
        catalog = load_catalog(REPO / "recipes", None)
        runtime = catalog.runtimes["vllm"].runtime
        for recipe in catalog.runtimes["vllm"].recipes.values():
            spec = vllm.container_spec(recipe, runtime, HostEnv(Path("/h"), 1, 1))
            env = dict(spec.env)
            self.assertIn("HF_MODULES_CACHE", env, recipe.id)
            read_only = [mount.split(":")[1] for mount in spec.mounts if mount.endswith(":ro")]
            self.assertFalse(any(env["HF_MODULES_CACHE"].startswith(path) for path in read_only), recipe.id)


if __name__ == "__main__":
    unittest.main()
