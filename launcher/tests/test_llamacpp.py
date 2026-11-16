import copy
import tempfile
import unittest
from pathlib import Path

from launcher.errors import RecipeError
from launcher.model import GpuInfo, HostEnv
from launcher.runtimes import llamacpp
from launcher.tests.fakes import REPO, FakeHost, http
from launcher.validate import read_toml

RECIPES = REPO / "recipes" / "llamacpp"
QWEN = RECIPES / "qwen3.6-35b-a3b.toml"
SNAPSHOT = "/hf/hub/models--ggml-org--Qwen3.6-35B-A3B-GGUF/snapshots/52f9f9b8052fb9c8b69bba4a95eda0f319ac3d2a"
ENV = HostEnv(Path("/home/user/.cache/huggingface"), 1000, 1000)


def load_runtime():
    data, source = read_toml(RECIPES / "_runtime.toml", "shipped")
    return llamacpp.parse_runtime(data, source)


def qwen_data():
    data, source = read_toml(QWEN, "shipped")
    return data, source


class LlamacppRecipeTest(unittest.TestCase):
    def test_runtime_file_parses(self):
        runtime = load_runtime()
        self.assertEqual(runtime.image, "ghcr.io/ggml-org/llama.cpp:server-cuda-b12407")
        self.assertEqual((runtime.container_name, runtime.host_port), ("llamacpp", 8080))
        self.assertEqual(runtime.served_name, "local")
        self.assertEqual(runtime.env, ())

    def test_runtime_file_must_name_its_runtime(self):
        data, source = read_toml(RECIPES / "_runtime.toml", "shipped")
        data["runtime"] = "vllm"
        with self.assertRaises(RecipeError) as ctx:
            llamacpp.parse_runtime(data, source)
        self.assertIn("'runtime' must be 'llamacpp'", str(ctx.exception))

    def test_qwen36_recipe_parses(self):
        data, source = qwen_data()
        recipe = llamacpp.parse_recipe(data, source, "qwen3.6-35b-a3b")
        self.assertEqual(recipe.id, "qwen3.6-35b-a3b")
        self.assertEqual((recipe.context, recipe.memory_gib, recipe.needs_gpu_mib), (131072, 20, 15078))
        self.assertEqual(recipe.model.revision, "52f9f9b8052fb9c8b69bba4a95eda0f319ac3d2a")
        self.assertEqual(recipe.aliases, ("qwen3.6",))
        self.assertTrue(recipe.opencode)
        self.assertFalse(recipe.chat_only)
        self.assertEqual(recipe.evidence.kind, "measured")

    def test_container_spec_qwen36(self):
        data, source = qwen_data()
        recipe = llamacpp.parse_recipe(data, source, "qwen3.6-35b-a3b")
        spec = llamacpp.container_spec(recipe, load_runtime(), ENV)
        self.assertEqual(spec.user, "1000:1000")
        self.assertEqual(spec.env, (("GGML_CUDA_DISABLE_GRAPHS", "1"), ("HF_HOME", "/hf")))
        self.assertEqual(spec.mounts, ("/home/user/.cache/huggingface:/hf",))
        self.assertEqual(spec.extra_flags, ())
        self.assertEqual((spec.image, spec.container_port), ("ghcr.io/ggml-org/llama.cpp:server-cuda-b12407", 8080))
        self.assertEqual(
            list(spec.args),
            ["-m", f"{SNAPSHOT}/Qwen3.6-35B-A3B-Q4_K_M.gguf", "--alias", "local", "-c", "131072",
             "-ngl", "all", "--n-cpu-moe", "25", "-fa", "on", "-np", "1", "--jinja", "-lm", "none",
             "-t", "8", "--moe-cache-mib", "3200", "--reasoning", "off", "--temp", "0.6",
             "--top-p", "0.95", "--top-k", "20", "--min-p", "0", "--presence-penalty", "0"],
        )

    def test_owned_flag_rejected(self):
        for extra in (["--port", "9000"], ["--port=9000"], ["-hf", "x/y"]):
            data, source = qwen_data()
            data = copy.deepcopy(data)
            data["server"]["args"] = data["server"]["args"] + extra
            with self.assertRaises(RecipeError) as ctx:
                llamacpp.parse_recipe(data, source, "qwen3.6-35b-a3b")
            self.assertIn(extra[0].split("=")[0], str(ctx.exception))

    def test_id_must_match_file_name(self):
        data, source = qwen_data()
        with self.assertRaises(RecipeError) as ctx:
            llamacpp.parse_recipe(data, source, "qwen")
        self.assertIn("must match the file name 'qwen'", str(ctx.exception))

    def test_split_gguf_uses_first_file(self):
        data, source = qwen_data()
        data = copy.deepcopy(data)
        data["model"]["files"] = ["m-00001-of-00002.gguf", "m-00002-of-00002.gguf"]
        recipe = llamacpp.parse_recipe(data, source, "qwen3.6-35b-a3b")
        spec = llamacpp.container_spec(recipe, load_runtime(), ENV)
        self.assertEqual(spec.args[:2], ("-m", f"{SNAPSHOT}/m-00001-of-00002.gguf"))

    def test_ram_need_is_optional(self):
        data, source = qwen_data()
        data = copy.deepcopy(data)
        del data["needs"]["ram_gib"]
        self.assertIsNone(llamacpp.parse_recipe(data, source, "qwen3.6-35b-a3b").needs_ram_gib)


class LlamacppFilesTest(unittest.TestCase):
    def setUp(self):
        data, source = qwen_data()
        data = copy.deepcopy(data)
        data["model"]["files"] = ["a.gguf", "b.gguf"]
        data["model"]["size_bytes"] = 7
        self.recipe = llamacpp.parse_recipe(data, source, "qwen3.6-35b-a3b")
        self.tmp = tempfile.TemporaryDirectory()
        self.hf_home = Path(self.tmp.name)
        self.snapshot = self.hf_home / "hub/models--ggml-org--Qwen3.6-35B-A3B-GGUF/snapshots/52f9f9b8052fb9c8b69bba4a95eda0f319ac3d2a"

    def tearDown(self):
        self.tmp.cleanup()

    def test_model_paths(self):
        self.assertEqual(
            llamacpp.model_paths(self.recipe, self.hf_home),
            [self.snapshot / "a.gguf", self.snapshot / "b.gguf"],
        )

    def test_missing_files_are_named(self):
        self.snapshot.mkdir(parents=True)
        (self.snapshot / "a.gguf").write_bytes(b"1234")
        self.assertEqual(llamacpp.file_problems(self.recipe, self.hf_home), ["missing b.gguf"])

    def test_wrong_total_size(self):
        self.snapshot.mkdir(parents=True)
        (self.snapshot / "a.gguf").write_bytes(b"1234")
        (self.snapshot / "b.gguf").write_bytes(b"12")
        self.assertEqual(
            llamacpp.file_problems(self.recipe, self.hf_home),
            ["the files add up to 6 bytes, the recipe expects 7"],
        )

    def test_complete_files_have_no_problems(self):
        self.snapshot.mkdir(parents=True)
        (self.snapshot / "a.gguf").write_bytes(b"1234")
        (self.snapshot / "b.gguf").write_bytes(b"123")
        self.assertEqual(llamacpp.file_problems(self.recipe, self.hf_home), [])


class LlamacppRuntimeBehaviourTest(unittest.TestCase):
    def setUp(self):
        data, source = qwen_data()
        self.recipe = llamacpp.parse_recipe(data, source, "qwen3.6-35b-a3b")
        self.runtime = load_runtime()

    def test_gpu_need_is_recipe_value(self):
        gpu = GpuInfo(total_mib=16303, reserved_mib=462, used_mib=112, free_mib=15731)
        self.assertEqual(llamacpp.gpu_need_mib(self.recipe, self.runtime, gpu), 15078)

    def test_served_name_is_runtime_alias(self):
        self.assertEqual(llamacpp.served_name(self.recipe, self.runtime), "local")

    def test_ready_on_health_200(self):
        host = FakeHost()
        host.on_http("GET", "http://127.0.0.1:8080/health", http(200, b'{"status":"ok"}'))
        self.assertTrue(llamacpp.ready(host, self.runtime, self.recipe))

    def test_not_ready_while_loading_or_refused(self):
        host = FakeHost()
        host.on_http("GET", "http://127.0.0.1:8080/health", http(503, b'{"error":"Loading model"}'))
        self.assertFalse(llamacpp.ready(host, self.runtime, self.recipe))
        self.assertFalse(llamacpp.ready(FakeHost(), self.runtime, self.recipe))


if __name__ == "__main__":
    unittest.main()
