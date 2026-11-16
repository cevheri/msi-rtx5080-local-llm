import copy
import json
import tempfile
import unittest
from pathlib import Path

from launcher.errors import RecipeError
from launcher.model import GpuInfo, HostEnv
from launcher.runtimes import vllm
from launcher.tests.fakes import REPO, FakeHost, http
from launcher.validate import read_toml

RECIPES = REPO / "recipes" / "vllm"
GPT_OSS = RECIPES / "gpt-oss-20b.toml"
REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"
ENV = HostEnv(Path("/home/user/.cache/huggingface"), 1000, 1000)
MODELS_URL = "http://127.0.0.1:8000/v1/models"


def load_runtime():
    data, source = read_toml(RECIPES / "_runtime.toml", "shipped")
    return vllm.parse_runtime(data, source)


def gpt_oss(edit=None):
    data, source = read_toml(GPT_OSS, "shipped")
    data = copy.deepcopy(data)
    if edit:
        edit(data)
    return vllm.parse_recipe(data, source, "gpt-oss-20b")


class VllmRecipeTest(unittest.TestCase):
    def test_runtime_file_parses(self):
        runtime = load_runtime()
        self.assertEqual((runtime.image, runtime.container_name, runtime.host_port),
                         ("vllm/vllm-openai:v0.33.0", "vllm", 8000))
        self.assertIsNone(runtime.served_name)
        self.assertEqual(runtime.env, (("HF_HUB_OFFLINE", "1"), ("VLLM_NO_USAGE_STATS", "1"), ("MAX_JOBS", "2"),
                                       ("HF_MODULES_CACHE", "/tmp/hf_modules")))
        self.assertEqual(runtime.volumes, ("vllm-cache:/root/.cache/vllm", "flashinfer-cache:/root/.cache/flashinfer"))
        self.assertEqual(runtime.default_gpu_memory_utilization, 0.92)

    def test_runtime_rejects_host_path_volume(self):
        data, source = read_toml(RECIPES / "_runtime.toml", "shipped")
        data["volumes"] = ["/:/host"]
        with self.assertRaises(RecipeError) as ctx:
            vllm.parse_runtime(data, source)
        self.assertIn("named volume", str(ctx.exception))

    def test_gpt_oss_recipe_parses(self):
        recipe = gpt_oss()
        self.assertEqual((recipe.id, recipe.context, recipe.memory_gib), ("gpt-oss-20b", 65536, 32))
        self.assertEqual(recipe.model.revision, REVISION)
        self.assertEqual(recipe.model.exclude, ("original/*", "metal/*"))
        self.assertEqual(recipe.evidence.kind, "measured")
        self.assertTrue(recipe.opencode)

    def test_gpt_oss_container_spec(self):
        spec = vllm.container_spec(gpt_oss(), load_runtime(), ENV)
        self.assertIsNone(spec.user)
        self.assertEqual(spec.env, (("HF_HUB_OFFLINE", "1"), ("VLLM_NO_USAGE_STATS", "1"), ("MAX_JOBS", "2"),
                                    ("HF_MODULES_CACHE", "/tmp/hf_modules")))
        self.assertEqual(spec.mounts, (
            "/home/user/.cache/huggingface:/root/.cache/huggingface:ro",
            "vllm-cache:/root/.cache/vllm",
            "flashinfer-cache:/root/.cache/flashinfer",
        ))
        self.assertEqual(spec.extra_flags, ("--ipc=host",))
        self.assertEqual((spec.image, spec.container_port), ("vllm/vllm-openai:v0.33.0", 8000))
        self.assertEqual(list(spec.args), [
            f"/root/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/{REVISION}",
            "--served-model-name", "gpt-oss-20b", "--max-model-len", "65536",
            "--gpu-memory-utilization", "0.965", "--max-num-seqs", "1", "--max-num-batched-tokens", "256",
            "--enforce-eager", "--kv-cache-dtype", "fp8", "--attention-backend", "TRITON_ATTN",
            "--enable-auto-tool-choice", "--tool-call-parser", "openai", "--reasoning-parser", "openai_gptoss",
            "--reasoning-effort-map", '{"none":"low","minimal":"low","xhigh":"high","max":"high"}',
            "--override-generation-config", '{"temperature":1.0,"top_p":1.0}',
        ])

    def test_gpu_need_from_utilization(self):
        gpu = GpuInfo(total_mib=16303, reserved_mib=462, used_mib=112, free_mib=15731)
        self.assertEqual(vllm.gpu_need_mib(gpt_oss(), load_runtime(), gpu), 15287)

    def test_utilization_in_equals_form(self):
        def edit(data):
            args = data["server"]["args"]
            index = args.index("--gpu-memory-utilization")
            data["server"]["args"] = args[:index] + ["--gpu-memory-utilization=0.5"] + args[index + 2:]
        gpu = GpuInfo(total_mib=16303, reserved_mib=462, used_mib=112, free_mib=15731)
        self.assertEqual(vllm.gpu_need_mib(gpt_oss(edit), load_runtime(), gpu), 7921)

    def test_utilization_defaults_to_runtime(self):
        def edit(data):
            args = data["server"]["args"]
            index = args.index("--gpu-memory-utilization")
            data["server"]["args"] = args[:index] + args[index + 2:]
        gpu = GpuInfo(total_mib=16303, reserved_mib=462, used_mib=112, free_mib=15731)
        self.assertEqual(vllm.gpu_need_mib(gpt_oss(edit), load_runtime(), gpu), 14574)

    def test_bad_utilization_rejected(self):
        def edit(data):
            args = data["server"]["args"]
            index = args.index("--gpu-memory-utilization")
            args[index + 1] = "1.5"
        with self.assertRaises(RecipeError) as ctx:
            gpt_oss(edit)
        self.assertIn("--gpu-memory-utilization", str(ctx.exception))

    def test_utilization_without_value_rejected(self):
        def edit(data):
            args = data["server"]["args"]
            index = args.index("--gpu-memory-utilization")
            data["server"]["args"] = args[:index] + args[index + 2:] + ["--gpu-memory-utilization"]
        with self.assertRaises(RecipeError):
            gpt_oss(edit)

    def test_revision_flag_owned(self):
        for flag in ("--revision", "--served-model-name", "--api-key"):
            with self.assertRaises(RecipeError) as ctx:
                gpt_oss(lambda data, f=flag: data["server"]["args"].extend([f, "x"]))
            self.assertIn(flag, str(ctx.exception))

    def test_served_name_is_recipe_value(self):
        self.assertEqual(vllm.served_name(gpt_oss(), load_runtime()), "gpt-oss-20b")


class VllmFilesTest(unittest.TestCase):
    def test_snapshot_and_config_are_required(self):
        recipe = gpt_oss()
        with tempfile.TemporaryDirectory() as tmp:
            hf_home = Path(tmp)
            snapshot = hf_home / "hub/models--openai--gpt-oss-20b/snapshots" / REVISION
            self.assertEqual(vllm.file_problems(recipe, hf_home), [f"missing snapshot {REVISION} of openai/gpt-oss-20b"])
            snapshot.mkdir(parents=True)
            self.assertEqual(vllm.file_problems(recipe, hf_home), [f"missing config.json in snapshot {REVISION}"])
            (snapshot / "config.json").write_text("{}")
            with open(snapshot / "model.safetensors", "wb") as handle:
                handle.truncate(recipe.model.size_bytes - 2)
            self.assertEqual(vllm.file_problems(recipe, hf_home), [])


class VllmReadyTest(unittest.TestCase):
    def setUp(self):
        self.recipe = gpt_oss()
        self.runtime = load_runtime()

    def test_ready_needs_served_name(self):
        host = FakeHost()
        host.on_http("GET", MODELS_URL, http(200, json.dumps({"data": [{"id": "other"}]}).encode()))
        self.assertFalse(vllm.ready(host, self.runtime, self.recipe))
        host.on_http("GET", MODELS_URL, http(200, json.dumps({"data": [{"id": "gpt-oss-20b"}]}).encode()))
        self.assertTrue(vllm.ready(host, self.runtime, self.recipe))

    def test_not_ready_when_refused_or_garbled(self):
        self.assertFalse(vllm.ready(FakeHost(), self.runtime, self.recipe))
        host = FakeHost()
        host.on_http("GET", MODELS_URL, http(200, b"<html>"))
        self.assertFalse(vllm.ready(host, self.runtime, self.recipe))


if __name__ == "__main__":
    unittest.main()
