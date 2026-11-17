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

    def test_runtime_facts(self):
        # A guard: llama.cpp accepts tools on every recipe and keeps both facts at their defaults (design D11).
        runtime = load_runtime()
        self.assertEqual((runtime.refuses_tools_without_parser, runtime.effort_enables_toggle_thinking), (False, False))

    def test_runtime_env_secret_refused(self):
        data, source = read_toml(RECIPES / "_runtime.toml", "shipped")
        data["env"] = {"HF_TOKEN": "x"}
        with self.assertRaises(RecipeError) as ctx:
            llamacpp.parse_runtime(data, source)
        self.assertIn("[env]: 'HF_TOKEN' would put a secret into the docker run command line", str(ctx.exception))
        load_runtime()

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


def recipe_from(stem, edit=None):
    data, source = read_toml(RECIPES / f"{stem}.toml", "shipped")
    data = copy.deepcopy(data)
    if edit:
        edit(data)
    return llamacpp.parse_recipe(data, source, stem)


def replace_args(data, old, new):
    args = data["server"]["args"]
    for index in range(len(args) - len(old) + 1):
        if args[index:index + len(old)] == old:
            data["server"]["args"] = args[:index] + new + args[index + len(old):]
            return
    raise AssertionError(f"{old} not in the args")


class LlamacppClientRulesTest(unittest.TestCase):
    def assertRefused(self, stem, edit, *fragments):
        with self.assertRaises(RecipeError) as ctx:
            recipe_from(stem, edit)
        message = str(ctx.exception)
        self.assertIn(str(RECIPES / f"{stem}.toml"), message)
        for fragment in fragments:
            self.assertIn(fragment, message)

    def test_label(self):
        self.assertEqual(llamacpp.LABEL, "llama.cpp")

    def test_served_name_pattern(self):
        self.assertTrue(llamacpp.SERVED_NAME.fullmatch("local"))
        self.assertIsNone(llamacpp.SERVED_NAME.fullmatch("-local"))

    def test_clients_table_required(self):
        self.assertRefused("qwen3.6-35b-a3b", lambda data: data.pop("clients"), "missing key 'clients'")

    def test_clients_parsed_into_recipe(self):
        recipe = recipe_from("qwen3.6-35b-a3b")
        self.assertEqual((recipe.clients.output, recipe.clients.thinking, recipe.clients.default),
                         (16384, "toggle", "off"))

    def test_clients_keys_are_checked(self):
        self.assertRefused("qwen3.6-35b-a3b", lambda data: data["clients"].update(outptu=1),
                           "[clients]", "did you mean 'output'")

    def test_toggle_default_must_match_reasoning_off(self):
        self.assertRefused("qwen3.6-35b-a3b", lambda data: data["clients"].update(default="on"),
                           "[clients]", "--reasoning")

    def test_toggle_off_through_template_kwargs(self):
        def edit(default):
            def apply(data):
                replace_args(data, ["--reasoning", "off"], ["--chat-template-kwargs", '{"enable_thinking":false}'])
                data["clients"]["default"] = default
            return apply
        self.assertEqual(recipe_from("qwen3.6-35b-a3b", edit("off")).clients.default, "off")
        self.assertRefused("qwen3.6-35b-a3b", edit("on"), "[clients]", "--chat-template-kwargs")

    def test_toggle_without_off_flag_is_on(self):
        def edit(default):
            def apply(data):
                replace_args(data, ["--reasoning", "off"], [])
                data["clients"]["default"] = default
            return apply
        self.assertRefused("qwen3.6-35b-a3b", edit("off"), "[clients]", "--reasoning")
        self.assertEqual(recipe_from("qwen3.6-35b-a3b", edit("on")).clients.default, "on")

    def test_template_kwargs_must_be_a_json_object(self):
        for bad in ("{not json", "[1]"):
            with self.subTest(value=bad):
                self.assertRefused(
                    "qwen3.6-35b-a3b",
                    lambda data, b=bad: replace_args(data, ["--reasoning", "off"], ["--chat-template-kwargs", b]),
                    "--chat-template-kwargs", "JSON object")

    def test_effort_needs_medium_server_effort(self):
        self.assertRefused("qwen3.8-27b",
                           lambda data: replace_args(data, ["--reasoning-effort", "medium"], ["--reasoning-effort", "high"]),
                           "[clients]", "--reasoning-effort")
        self.assertEqual(recipe_from("qwen3.8-27b", lambda data: replace_args(data, ["--reasoning-effort", "medium"], []))
                         .clients.thinking, "effort")

    def test_always_needs_high_effort_or_deepseek_format(self):
        self.assertRefused("k2-horizon-mova-36b",
                           lambda data: replace_args(data, ["--reasoning-effort", "high"], []),
                           "[clients]", "--reasoning-effort")
        for stem in ("k2-horizon-7b", "deepseek-r1-0528-qwen3-8b", "k2-horizon-mova-36b"):
            with self.subTest(recipe=stem):
                self.assertEqual(recipe_from(stem).clients.thinking, "always")

    def test_none_allows_no_reasoning_flags(self):
        for extra in (["--reasoning", "off"], ["--reasoning-effort", "low"], ["--reasoning-format", "none"],
                      ["--chat-template-kwargs", "{}"]):
            with self.subTest(extra=extra):
                self.assertRefused("qwen3-coder-next",
                                   lambda data, e=extra: data["server"]["args"].extend(e),
                                   "[clients]", extra[0])

    def test_runtime_served_name_pattern(self):
        data, source = read_toml(RECIPES / "_runtime.toml", "shipped")
        data["served_name"] = "lo/cal"
        with self.assertRaises(RecipeError) as ctx:
            llamacpp.parse_runtime(data, source)
        self.assertIn("served_name", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
