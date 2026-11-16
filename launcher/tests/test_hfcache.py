import hashlib
import io
import os
import tempfile
import unittest
from pathlib import Path

from launcher.catalog import load_catalog, resolve
from launcher.errors import DownloadError, PreflightError, UsageError
from launcher.hfcache import HfAuth, download_file, hf_home, read_auth, snapshot_dir
from launcher.lock import hold_lock
from launcher.runtimes import llamacpp, vllm
from launcher.tests.fakes import REPO, FakeHost, ok, fail, http

REV = "52f9f9b8052fb9c8b69bba4a95eda0f319ac3d2a"
URL = f"https://huggingface.co/org/model/resolve/{REV}/m.gguf"
SECRET = "hf_secret123456789"


def complete_snapshot(snapshot: Path, size: int) -> None:
    """A vLLM snapshot whose files add up to the recipe's size, sparse so it takes no disk space."""
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / "config.json").write_text("{}")
    with open(snapshot / "model.safetensors", "wb") as handle:
        handle.truncate(size - 2)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def lfs_head(data: bytes, status=302):
    return http(status, headers={"x-linked-etag": f'"{sha(data)}"', "x-linked-size": str(len(data)),
                                 "location": "https://cdn.example/m.gguf"})


class PathsAndAuthTest(unittest.TestCase):
    def test_hf_home_default_and_override(self):
        self.assertEqual(hf_home({"HOME": "/home/u"}), Path("/home/u/.cache/huggingface"))
        self.assertEqual(hf_home({"HOME": "/home/u", "HF_HOME": "/data/hf cache"}), Path("/data/hf cache"))

    def test_snapshot_dir_layout(self):
        self.assertEqual(snapshot_dir(Path("/c"), "org/model", REV),
                         Path(f"/c/hub/models--org--model/snapshots/{REV}"))

    def test_read_auth(self):
        self.assertEqual(read_auth({"HF_TOKEN": SECRET}, None), HfAuth(SECRET, None))
        self.assertEqual(read_auth({}, None), HfAuth(None, None))
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / "hf.env"
            env_file.write_text(f"# token\nHF_TOKEN={SECRET}\n")
            self.assertEqual(read_auth({}, env_file), HfAuth(SECRET, env_file))
            env_file.write_text("OTHER=1\n")
            with self.assertRaises(UsageError) as ctx:
                read_auth({}, env_file)
            self.assertIn("no HF_TOKEN= line", str(ctx.exception))
            with self.assertRaises(UsageError) as ctx:
                read_auth({}, Path(tmp) / "missing.env")
            self.assertIn("does not exist", str(ctx.exception))


class DownloadTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        self.host = FakeHost()
        self.out = io.StringIO()
        self.blobs = self.home / "hub/models--org--model/blobs"
        self.link = self.home / f"hub/models--org--model/snapshots/{REV}/m.gguf"

    def tearDown(self):
        self.tmp.cleanup()

    def test_download_verifies_and_links(self):
        self.host.on_http("HEAD", URL, lfs_head(b"abc"))
        self.host.downloads[URL] = b"abc"
        path = download_file(self.host, self.home, "org/model", REV, "m.gguf", None, self.out)
        self.assertEqual(path, self.link)
        self.assertEqual(os.readlink(self.link), f"../../blobs/{sha(b'abc')}")
        self.assertEqual(self.link.read_bytes(), b"abc")
        self.assertFalse((self.blobs / f"{sha(b'abc')}.incomplete").exists())

    def test_download_mismatch_raises_and_cleans(self):
        self.host.on_http("HEAD", URL, lfs_head(b"abc"))
        self.host.downloads[URL] = b"abd"
        with self.assertRaises(DownloadError) as ctx:
            download_file(self.host, self.home, "org/model", REV, "m.gguf", None, self.out)
        self.assertIn("checksum", str(ctx.exception))
        self.assertEqual(list(self.blobs.iterdir()), [])
        self.assertFalse(self.link.exists())

    def test_download_resumes(self):
        self.host.on_http("HEAD", URL, lfs_head(b"abc"))
        self.host.downloads[URL] = b"abc"
        self.blobs.mkdir(parents=True)
        (self.blobs / f"{sha(b'abc')}.incomplete").write_bytes(b"a")
        download_file(self.host, self.home, "org/model", REV, "m.gguf", None, self.out)
        url, headers, offset = self.host.download_calls[0]
        self.assertEqual((headers.get("Range"), offset), ("bytes=1-", 1))
        self.assertEqual(self.link.read_bytes(), b"abc")

    def test_existing_blob_is_linked_without_download(self):
        self.host.on_http("HEAD", URL, lfs_head(b"abc"))
        self.blobs.mkdir(parents=True)
        (self.blobs / sha(b"abc")).write_bytes(b"abc")
        download_file(self.host, self.home, "org/model", REV, "m.gguf", None, self.out)
        self.assertEqual(self.host.download_calls, [])
        self.assertEqual(self.link.read_bytes(), b"abc")

    def test_token_goes_in_header_only(self):
        self.host.on_http("HEAD", URL, lfs_head(b"abc"))
        self.host.downloads[URL] = b"abc"
        download_file(self.host, self.home, "org/model", REV, "m.gguf", SECRET, self.out)
        self.assertEqual(self.host.http_calls[0][2]["Authorization"], f"Bearer {SECRET}")
        self.assertEqual(self.host.download_calls[0][1]["Authorization"], f"Bearer {SECRET}")
        self.assertNotIn(SECRET, self.out.getvalue())

    def test_gated_model_without_token(self):
        self.host.on_http("HEAD", URL, http(401))
        with self.assertRaises(DownloadError) as ctx:
            download_file(self.host, self.home, "org/model", REV, "m.gguf", None, self.out)
        self.assertIn("HF_TOKEN", str(ctx.exception))

    def test_not_lfs_is_refused(self):
        self.host.on_http("HEAD", URL, http(200, headers={"content-length": "3"}))
        with self.assertRaises(DownloadError) as ctx:
            download_file(self.host, self.home, "org/model", REV, "m.gguf", None, self.out)
        self.assertIn("no SHA-256", str(ctx.exception))


class PullTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        self.catalog = load_catalog(REPO / "recipes", None)
        self.out = io.StringIO()

    def tearDown(self):
        self.tmp.cleanup()

    def test_vllm_token_never_in_argv(self):
        recipe, runtime = resolve(self.catalog, "gpt-oss-20b", "vllm")
        host = FakeHost()
        snapshot = snapshot_dir(self.home, recipe.model.repo, recipe.model.revision)

        def done(argv):
            complete_snapshot(snapshot, recipe.model.size_bytes)
            return ok()

        host.on(("docker", "run"), done)
        vllm.pull(host, recipe, runtime, self.home, HfAuth(SECRET, None), self.out)
        argv = host.calls[0]
        self.assertNotIn(SECRET, " ".join(argv))
        self.assertEqual(host.env_extras[0], {"HF_TOKEN": SECRET})
        self.assertIn("HF_TOKEN", argv)
        self.assertEqual(argv[argv.index("--entrypoint") + 1:argv.index("--entrypoint") + 6],
                         ["hf", "vllm/vllm-openai:v0.33.0", "download", "openai/gpt-oss-20b", "--revision"])
        self.assertEqual(argv[-4:], ["--exclude", "original/*", "--exclude", "metal/*"])
        self.assertIn(f"{self.home}:/hf", argv)
        self.assertNotIn(SECRET, self.out.getvalue() + host.out.getvalue() + host.err.getvalue())

    def test_vllm_env_file_passed_as_flag(self):
        recipe, runtime = resolve(self.catalog, "gpt-oss-20b", "vllm")
        host = FakeHost()
        snapshot = snapshot_dir(self.home, recipe.model.repo, recipe.model.revision)
        complete_snapshot(snapshot, recipe.model.size_bytes)
        host.on(("docker", "run"), ok())
        vllm.pull(host, recipe, runtime, self.home, HfAuth(SECRET, Path("/x/hf.env")), self.out)
        argv = host.calls[0]
        self.assertEqual(argv[argv.index("--env-file") + 1], "/x/hf.env")
        self.assertNotIn("HF_TOKEN", argv)
        self.assertEqual(host.env_extras[0], {})

    def test_vllm_pull_failure(self):
        recipe, runtime = resolve(self.catalog, "gpt-oss-20b", "vllm")
        host = FakeHost()
        host.on(("docker", "run"), fail("Repository not found"))
        with self.assertRaises(DownloadError) as ctx:
            vllm.pull(host, recipe, runtime, self.home, HfAuth(None, None), self.out)
        self.assertIn("Repository not found", str(ctx.exception))

    def test_llamacpp_pull_downloads_missing_files_only(self):
        recipe, runtime = resolve(self.catalog, "qwen3.6", None)
        name = recipe.model.files[0]
        url = f"https://huggingface.co/{recipe.model.repo}/resolve/{recipe.model.revision}/{name}"
        host = FakeHost()
        data = b"x" * 10
        host.on_http("HEAD", url, lfs_head(data))
        host.downloads[url] = data
        with self.assertRaises(DownloadError) as ctx:
            llamacpp.pull(host, recipe, runtime, self.home, HfAuth(None, None), self.out)
        self.assertIn("20542732576", str(ctx.exception))  # a 10-byte stand-in cannot match the recorded size
        self.assertEqual(len(host.download_calls), 1)
        host.download_calls.clear()
        with self.assertRaises(DownloadError):
            llamacpp.pull(host, recipe, runtime, self.home, HfAuth(None, None), self.out)
        self.assertEqual(host.download_calls, [])


class LockTest(unittest.TestCase):
    def test_lock_refuses_second_holder(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {"XDG_RUNTIME_DIR": tmp}
            with hold_lock(env):
                with self.assertRaises(PreflightError) as ctx:
                    with hold_lock(env):
                        pass
                self.assertIn("another llmctl command is running", str(ctx.exception))
            with hold_lock(env):
                pass


if __name__ == "__main__":
    unittest.main()
